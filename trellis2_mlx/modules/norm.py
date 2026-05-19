"""
LayerNorm/GroupNorm variants that always normalize in fp32 (matches LayerNorm32 in
trellis2/modules/norm.py).
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..ops.sparse_tensor import VarLenTensor


def _layer_norm_fp32(x: mx.array, normalized_shape, weight: Optional[mx.array],
                     bias: Optional[mx.array], eps: float) -> mx.array:
    in_dtype = x.dtype
    xf = x.astype(mx.float32)
    axes = tuple(range(-len(normalized_shape), 0))
    mean = mx.mean(xf, axis=axes, keepdims=True)
    var = mx.mean((xf - mean) ** 2, axis=axes, keepdims=True)
    y = (xf - mean) * mx.rsqrt(var + eps)
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y.astype(in_dtype)


class LayerNorm32(nn.Module):
    """LayerNorm with computation always in float32 (matches reference)."""

    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = mx.ones(self.normalized_shape, dtype=mx.float32)
            self.bias = mx.zeros(self.normalized_shape, dtype=mx.float32)
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x):
        if isinstance(x, VarLenTensor):
            return x.replace(_layer_norm_fp32(x.feats, self.normalized_shape, self.weight, self.bias, self.eps))
        return _layer_norm_fp32(x, self.normalized_shape, self.weight, self.bias, self.eps)


class GroupNorm32(nn.Module):
    """GroupNorm fp32 over channel dim. Assumes channel-last input shape."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = mx.ones((num_channels,), dtype=mx.float32)
            self.bias = mx.zeros((num_channels,), dtype=mx.float32)
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """
        x: [..., C] (channel-last). Groups split along C.
        """
        in_dtype = x.dtype
        xf = x.astype(mx.float32)
        s = xf.shape
        C = s[-1]
        G = self.num_groups
        assert C % G == 0
        g = xf.reshape(*s[:-1], G, C // G)
        # normalize over (group's channels + spatial dims if any). The reference
        # GroupNorm normalizes over (C/G, *spatial); here channel-last means the
        # spatial dims sit before C. Match the math by reducing over the channel
        # group axis only; spatial reduction is handled by stacking.
        # For our use-cases x is either [N, C] (varlen) or [B, D, H, W, C] (dense
        # decoder). In both cases reducing over all dims except batch+group gives
        # the same result as torch.nn.GroupNorm with channel-first layout.
        reduce_axes = list(range(1, g.ndim - 1)) + [g.ndim - 1]
        # ndim - 2 is the group axis; everything else (except batch) gets reduced.
        reduce_axes = [a for a in reduce_axes if a != g.ndim - 2]
        mean = mx.mean(g, axis=tuple(reduce_axes), keepdims=True)
        var = mx.mean((g - mean) ** 2, axis=tuple(reduce_axes), keepdims=True)
        g = (g - mean) * mx.rsqrt(var + self.eps)
        g = g.reshape(*s)
        if self.affine and self.weight is not None:
            g = g * self.weight + self.bias
        return g.astype(in_dtype)


class ChannelLayerNorm32(LayerNorm32):
    """
    LayerNorm applied over the channel dim of an [N, C, *spatial] tensor.
    In our MLX path activations are stored channel-last, so we just delegate
    to LayerNorm32 on a [..., C]-shaped tensor.
    """
    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x)


def norm_layer(norm_type: str, channels: int) -> nn.Module:
    if norm_type == "group":
        return GroupNorm32(32, channels)
    if norm_type == "layer":
        return ChannelLayerNorm32(channels, elementwise_affine=True, eps=1e-6)
    raise ValueError(norm_type)


def modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    return x * (1 + scale[:, None]) + shift[:, None]


def manual_cast(x, dtype):
    if isinstance(x, VarLenTensor):
        return x.astype(dtype)
    return x.astype(dtype)

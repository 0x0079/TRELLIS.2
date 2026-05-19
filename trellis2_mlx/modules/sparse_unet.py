"""
Sparse U-Net building blocks for MLX.

Mirrors trellis2/models/sc_vaes/sparse_unet_vae.py:
- SparseResBlock3d
- SparseResBlockDownsample3d / SparseResBlockUpsample3d
- SparseResBlockS2C3d / SparseResBlockC2S3d
- SparseConvNeXtBlock3d
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..ops.sparse_tensor import SparseTensor
from ..ops.sparse_conv import SparseConv3d
from ..ops.sparse_pool import (
    SparseDownsample,
    SparseUpsample,
    SparseSpatial2Channel,
    SparseChannel2Spatial,
)
from .linear import SparseLinear
from .norm import LayerNorm32


class _Identity(nn.Module):
    def __call__(self, x):
        return x


def _silu_inplace(x: SparseTensor) -> SparseTensor:
    return x.replace(nn.silu(x.feats))


class SparseResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
        resample_mode: str = "nearest",
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        self.resample_mode = resample_mode
        assert not (downsample and upsample)

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)

        if resample_mode == "nearest":
            self.conv1 = SparseConv3d(channels, self.out_channels, 3)
        elif resample_mode == "spatial2channel" and not downsample:
            self.conv1 = SparseConv3d(channels, self.out_channels * 8, 3)
        elif resample_mode == "spatial2channel" and downsample:
            self.conv1 = SparseConv3d(channels, self.out_channels // 8, 3)

        self.conv2 = SparseConv3d(self.out_channels, self.out_channels, 3)

        if resample_mode == "nearest":
            self.skip_connection = (
                SparseLinear(channels, self.out_channels)
                if channels != self.out_channels
                else _Identity()
            )
            self._skip_kind = "linear" if channels != self.out_channels else "identity"
        elif resample_mode == "spatial2channel" and downsample:
            self._skip_kind = "s2c_skip"  # parameterless lambda — no module attribute
        elif resample_mode == "spatial2channel" and not downsample:
            self._skip_kind = "c2s_skip"

        if downsample:
            self.updown = SparseDownsample(2) if resample_mode == "nearest" else SparseSpatial2Channel(2)
        elif upsample:
            self.to_subdiv = SparseLinear(channels, 8)
            self.updown = SparseUpsample(2) if resample_mode == "nearest" else SparseChannel2Spatial(2)

    def _do_skip(self, x: SparseTensor) -> SparseTensor:
        if self._skip_kind == "identity":
            return x
        if self._skip_kind == "linear":
            return self.skip_connection(x)  # type: ignore[attr-defined]
        if self._skip_kind == "s2c_skip":
            # mean over groups of 8 (3D s2c packing): C * 8 / out -> out
            n = x.feats.shape[0]
            f = x.feats.reshape(n, self.out_channels, self.channels * 8 // self.out_channels).mean(axis=-1)
            return x.replace(f)
        if self._skip_kind == "c2s_skip":
            # repeat_interleave along channel dim
            reps = self.out_channels // (self.channels // 8)
            f = mx.repeat(x.feats, reps, axis=1)
            return x.replace(f)
        raise RuntimeError(self._skip_kind)

    def _updown(self, x: SparseTensor, subdiv: Optional[SparseTensor] = None) -> SparseTensor:
        if self.downsample:
            return self.updown(x)  # type: ignore[attr-defined]
        if self.upsample:
            return self.updown(x, subdiv.replace(subdiv.feats > 0))  # type: ignore[attr-defined]
        return x

    def __call__(self, x: SparseTensor):
        subdiv = None
        if self.upsample:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = _silu_inplace(h)
        if self.resample_mode == "spatial2channel":
            h = self.conv1(h)
        h = self._updown(h, subdiv)
        x = self._updown(x, subdiv)
        if self.resample_mode == "nearest":
            h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = _silu_inplace(h)
        h = self.conv2(h)
        h = h + self._do_skip(x)
        if self.upsample:
            return h, subdiv
        return h


class SparseResBlockDownsample3d(nn.Module):
    def __init__(self, channels: int, out_channels: Optional[int] = None, use_checkpoint: bool = False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = SparseConv3d(self.out_channels, self.out_channels, 3)
        self.skip_connection = (
            SparseLinear(channels, self.out_channels) if channels != self.out_channels else _Identity()
        )
        self.updown = SparseDownsample(2)

    def __call__(self, x: SparseTensor):
        h = x.replace(self.norm1(x.feats))
        h = _silu_inplace(h)
        h = self.updown(h)
        x = self.updown(x)
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = _silu_inplace(h)
        h = self.conv2(h)
        return h + self.skip_connection(x)


class SparseResBlockUpsample3d(nn.Module):
    def __init__(self, channels: int, out_channels: Optional[int] = None, use_checkpoint: bool = False, pred_subdiv: bool = True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.pred_subdiv = pred_subdiv
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = SparseConv3d(self.out_channels, self.out_channels, 3)
        self.skip_connection = (
            SparseLinear(channels, self.out_channels) if channels != self.out_channels else _Identity()
        )
        if pred_subdiv:
            self.to_subdiv = SparseLinear(channels, 8)
        self.updown = SparseUpsample(2)

    def __call__(self, x: SparseTensor, subdiv: Optional[SparseTensor] = None):
        if self.pred_subdiv:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = _silu_inplace(h)
        subdiv_bin = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
        h = self.updown(h, subdiv_bin)
        x = self.updown(x, subdiv_bin)
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = _silu_inplace(h)
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        if self.pred_subdiv:
            return h, subdiv
        return h


class SparseResBlockS2C3d(nn.Module):
    def __init__(self, channels: int, out_channels: Optional[int] = None, use_checkpoint: bool = False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = SparseConv3d(channels, self.out_channels // 8, 3)
        self.conv2 = SparseConv3d(self.out_channels, self.out_channels, 3)
        self.updown = SparseSpatial2Channel(2)

    def _skip(self, x: SparseTensor) -> SparseTensor:
        n = x.feats.shape[0]
        f = x.feats.reshape(n, self.out_channels, self.channels * 8 // self.out_channels).mean(axis=-1)
        return x.replace(f)

    def __call__(self, x: SparseTensor):
        h = x.replace(self.norm1(x.feats))
        h = _silu_inplace(h)
        h = self.conv1(h)
        h = self.updown(h)
        x = self.updown(x)
        h = h.replace(self.norm2(h.feats))
        h = _silu_inplace(h)
        h = self.conv2(h)
        return h + self._skip(x)


class SparseResBlockC2S3d(nn.Module):
    def __init__(self, channels: int, out_channels: Optional[int] = None, use_checkpoint: bool = False, pred_subdiv: bool = True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.pred_subdiv = pred_subdiv
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = SparseConv3d(channels, self.out_channels * 8, 3)
        self.conv2 = SparseConv3d(self.out_channels, self.out_channels, 3)
        if pred_subdiv:
            self.to_subdiv = SparseLinear(channels, 8)
        self.updown = SparseChannel2Spatial(2)

    def _skip(self, x: SparseTensor) -> SparseTensor:
        reps = self.out_channels // (self.channels // 8)
        f = mx.repeat(x.feats, reps, axis=1)
        return x.replace(f)

    def __call__(self, x: SparseTensor, subdiv: Optional[SparseTensor] = None):
        if self.pred_subdiv:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = _silu_inplace(h)
        h = self.conv1(h)
        subdiv_bin = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
        h = self.updown(h, subdiv_bin)
        x = self.updown(x, subdiv_bin)
        h = h.replace(self.norm2(h.feats))
        h = _silu_inplace(h)
        h = self.conv2(h)
        h = h + self._skip(x)
        if self.pred_subdiv:
            return h, subdiv
        return h


class SparseConvNeXtBlock3d(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0, use_checkpoint: bool = False):
        super().__init__()
        self.channels = channels
        self.norm = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.conv = SparseConv3d(channels, channels, 3)
        hidden = int(channels * mlp_ratio)
        # Reference: self.mlp = nn.Sequential(Linear, SiLU, Linear)
        # Use a list attribute so MLX exposes keys mlp.0/mlp.2 matching the checkpoint.
        from .transformer_blocks import _silu_module
        self.mlp = [
            nn.Linear(channels, hidden),
            _silu_module(),
            nn.Linear(hidden, channels),
        ]

    def __call__(self, x: SparseTensor) -> SparseTensor:
        h = self.conv(x)
        h = h.replace(self.norm(h.feats))
        f = self.mlp[0](h.feats)
        f = self.mlp[1]._fn(f)
        f = self.mlp[2](f)
        h = h.replace(f)
        return h + x

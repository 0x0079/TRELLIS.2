"""
Sparse Transformer blocks for MLX.

Mirrors trellis2/modules/sparse/transformer/{blocks,modulated}.py +
modules/sparse/attention/modules.py.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from ..ops.sparse_tensor import VarLenTensor, SparseTensor
from ..ops.sparse_attention import (
    sparse_scaled_dot_product_attention,
    sparse_windowed_scaled_dot_product_self_attention,
    SparseRotaryPositionEmbedder,
)
from .linear import SparseLinear
from .norm import LayerNorm32
from .transformer_blocks import _NoParamModule, _silu_module, _gelu_tanh_module


class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = mx.ones((heads, dim), dtype=mx.float32)

    def __call__(self, x):
        in_dtype = x.dtype
        if isinstance(x, VarLenTensor):
            xf = x.feats.astype(mx.float32)
            norm = mx.rsqrt(mx.sum(xf * xf, axis=-1, keepdims=True) + 1e-12)
            return x.replace((xf * norm * self.gamma * self.scale).astype(in_dtype))
        xf = x.astype(mx.float32)
        norm = mx.rsqrt(mx.sum(xf * xf, axis=-1, keepdims=True) + 1e-12)
        return (xf * norm * self.gamma * self.scale).astype(in_dtype)


def _linear_apply(layer: nn.Linear, x):
    if isinstance(x, VarLenTensor):
        return x.replace(layer(x.feats))
    return layer(x)


def _reshape_chs(x, shape):
    if isinstance(x, VarLenTensor):
        return x.reshape(*shape)
    return x.reshape(*x.shape[:2], *shape)


class SparseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: str = "self",
        attn_mode: str = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ("self", "cross")
        assert attn_mode in ("full", "windowed", "double_windowed")
        assert type == "self" or attn_mode == "full"
        if attn_mode == "double_windowed":
            assert window_size is not None and window_size % 2 == 0
            assert num_heads % 2 == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)

        if qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)

        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = SparseRotaryPositionEmbedder(self.head_dim, rope_freq=rope_freq)

    def _fused_pre(self, x, num_fused: int):
        if isinstance(x, VarLenTensor):
            x_feats = x.feats[None]
            x_feats = x_feats.reshape(*x_feats.shape[:2], num_fused, self.num_heads, -1)
            return x.replace(x_feats.squeeze(0) if x_feats.shape[0] == 1 else x_feats)
        return x.reshape(*x.shape[:2], num_fused, self.num_heads, -1)

    def __call__(self, x: SparseTensor, context: Optional[Union[VarLenTensor, mx.array]] = None) -> SparseTensor:
        if self._type == "self":
            qkv = _linear_apply(self.to_qkv, x)
            qkv = self._fused_pre(qkv, num_fused=3)

            if self.qk_rms_norm or self.use_rope:
                q = qkv.replace(qkv.feats[:, 0])
                k = qkv.replace(qkv.feats[:, 1])
                v = qkv.replace(qkv.feats[:, 2])
                if self.qk_rms_norm:
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                if self.use_rope:
                    q, k = self.rope(q, k)
                qkv = qkv.replace(mx.stack([q.feats, k.feats, v.feats], axis=1))

            if self.attn_mode == "full":
                h = sparse_scaled_dot_product_attention(qkv)
            elif self.attn_mode == "windowed":
                h = sparse_windowed_scaled_dot_product_self_attention(
                    qkv, self.window_size, shift_window=self.shift_window or (0, 0, 0)
                )
            elif self.attn_mode == "double_windowed":
                qkv0 = qkv.replace(qkv.feats[:, :, self.num_heads // 2:])
                qkv1 = qkv.replace(qkv.feats[:, :, : self.num_heads // 2])
                h0 = sparse_windowed_scaled_dot_product_self_attention(
                    qkv0, self.window_size, shift_window=(0, 0, 0)
                )
                h1 = sparse_windowed_scaled_dot_product_self_attention(
                    qkv1, self.window_size, shift_window=(self.window_size // 2,) * 3
                )
                h = qkv.replace(mx.concatenate([h0.feats, h1.feats], axis=1))
        else:
            q = _linear_apply(self.to_q, x)
            q = _reshape_chs(q, (self.num_heads, -1))
            kv = _linear_apply(self.to_kv, context)
            kv = self._fused_pre(kv, num_fused=2)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = kv.replace(kv.feats[:, 0]) if isinstance(kv, VarLenTensor) else kv[:, :, 0]
                v = kv.replace(kv.feats[:, 1]) if isinstance(kv, VarLenTensor) else kv[:, :, 1]
                k = self.k_rms_norm(k)
                h = sparse_scaled_dot_product_attention(q, k, v)
            else:
                h = sparse_scaled_dot_product_attention(q, kv)
        h = _reshape_chs(h, (-1,))
        h = _linear_apply(self.to_out, h)
        return h


class SparseFeedForwardNet(nn.Module):
    """Sparse FFN matching nn.Sequential(SparseLinear, GELU(tanh), SparseLinear)."""
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.mlp: List[nn.Module] = [
            SparseLinear(channels, hidden),
            _gelu_tanh_module(),
            SparseLinear(hidden, channels),
        ]

    def __call__(self, x: VarLenTensor) -> VarLenTensor:
        h = self.mlp[0](x)
        # _NoParamModule wraps a Python callable applied to feats
        h = h.replace(self.mlp[1]._fn(h.feats))
        return self.mlp[2](h)


class ModulatedSparseTransformerCrossBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: str = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_rope: bool = False,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels, num_heads, type="self", attn_mode=attn_mode,
            window_size=window_size, shift_window=shift_window,
            qkv_bias=qkv_bias, use_rope=use_rope, rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels, num_heads, ctx_channels=ctx_channels, type="cross", attn_mode="full",
            qkv_bias=qkv_bias, qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = SparseFeedForwardNet(channels, mlp_ratio)
        if not share_mod:
            self.adaLN_modulation: List[nn.Module] = [
                _silu_module(),
                nn.Linear(channels, 6 * channels, bias=True),
            ]
        else:
            self.modulation = mx.zeros((6 * channels,), dtype=mx.float32)

    def _ada(self, mod: mx.array):
        if self.share_mod:
            m = (self.modulation + mod).astype(mod.dtype)
        else:
            m = self.adaLN_modulation[1](self.adaLN_modulation[0](mod))
        return mx.split(m, 6, axis=1)

    def __call__(self, x: SparseTensor, mod: mx.array, context) -> SparseTensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._ada(mod)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.cross_attn(h, context)
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

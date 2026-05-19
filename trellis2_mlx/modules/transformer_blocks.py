"""
Dense Transformer blocks for MLX.

Mirrors trellis2/modules/transformer/blocks.py + modulated.py and
trellis2/modules/attention/modules.py (MultiHeadAttention).
"""
from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..ops.attention import scaled_dot_product_attention
from ..ops.rope import RotaryPositionEmbedder
from .norm import LayerNorm32


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = mx.ones((heads, dim), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        in_dtype = x.dtype
        xf = x.astype(mx.float32)
        # F.normalize over last dim: x / ||x||
        norm = mx.rsqrt(mx.sum(xf * xf, axis=-1, keepdims=True) + 1e-12)
        return (xf * norm * self.gamma * self.scale).astype(in_dtype)


class MultiHeadAttention(nn.Module):
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
        assert attn_mode == "full", "Only attn_mode='full' is supported on the dense path."
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self._type = type
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)

        if qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)

        self.to_out = nn.Linear(channels, channels)

    def __call__(
        self,
        x: mx.array,
        context: Optional[mx.array] = None,
        phases: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, C = x.shape
        if self._type == "self":
            qkv = self.to_qkv(x).reshape(B, L, 3, self.num_heads, -1)
            if self.qk_rms_norm or self.use_rope:
                q = qkv[:, :, 0]
                k = qkv[:, :, 1]
                v = qkv[:, :, 2]
                if self.qk_rms_norm:
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                if self.use_rope:
                    assert phases is not None
                    q = RotaryPositionEmbedder.apply_rotary_embedding(q, phases)
                    k = RotaryPositionEmbedder.apply_rotary_embedding(k, phases)
                h = scaled_dot_product_attention(q, k, v)
            else:
                h = scaled_dot_product_attention(qkv)
        else:
            Lkv = context.shape[1]
            q = self.to_q(x).reshape(B, L, self.num_heads, -1)
            kv = self.to_kv(context).reshape(B, Lkv, 2, self.num_heads, -1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = kv[:, :, 0]
                v = kv[:, :, 1]
                k = self.k_rms_norm(k)
                h = scaled_dot_product_attention(q, k, v)
            else:
                h = scaled_dot_product_attention(q, kv)
        h = h.reshape(B, L, -1)
        return self.to_out(h)


class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        # The reference uses nn.Sequential containing [Linear, GELU(tanh), Linear].
        # We replicate the parameter layout (`mlp.0`, `mlp.2`) explicitly.
        self.mlp_0 = nn.Linear(channels, hidden)
        self.mlp_2 = nn.Linear(hidden, channels)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.mlp_0(x)
        h = nn.gelu_approx(h)
        return self.mlp_2(h)


class ModulatedTransformerCrossBlock(nn.Module):
    """
    Cross-attention DiT block: norm1->self_attn->add, norm2->cross_attn->add,
    norm3->mlp->add. AdaLN modulation injected on norm1 and norm3 (gate/scale/shift).
    """

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: str = "full",
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels, num_heads, type="self", attn_mode=attn_mode,
            qkv_bias=qkv_bias, use_rope=use_rope, rope_freq=rope_freq, qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels, num_heads, ctx_channels=ctx_channels, type="cross", attn_mode="full",
            qkv_bias=qkv_bias, qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio)
        if not share_mod:
            self.adaLN_modulation_0 = nn.SiLU()
            self.adaLN_modulation_1 = nn.Linear(channels, 6 * channels, bias=True)
        else:
            self.modulation = mx.zeros((6 * channels,), dtype=mx.float32)

    def _ada(self, mod: mx.array) -> Tuple[mx.array, ...]:
        if self.share_mod:
            m = (self.modulation + mod).astype(mod.dtype)
        else:
            m = self.adaLN_modulation_1(self.adaLN_modulation_0(mod))
        return mx.split(m, 6, axis=1)

    def __call__(
        self,
        x: mx.array,
        mod: mx.array,
        context: mx.array,
        phases: Optional[mx.array] = None,
    ) -> mx.array:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._ada(mod)
        h = self.norm1(x)
        h = h * (1 + scale_msa[:, None]) + shift_msa[:, None]
        h = self.self_attn(h, phases=phases)
        h = h * gate_msa[:, None]
        x = x + h
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h
        h = self.norm3(x)
        h = h * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        h = self.mlp(h)
        h = h * gate_mlp[:, None]
        x = x + h
        return x

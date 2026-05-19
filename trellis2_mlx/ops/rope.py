"""
Rotary and absolute position embeddings on MLX.

These match trellis2/modules/attention/rope.py and modules/transformer/blocks.py
AbsolutePositionEmbedder semantics, so weights/results line up across backends.
"""
from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class AbsolutePositionEmbedder(nn.Module):
    """Sin/cos absolute position embedder over D dims (D channel-aligned).

    Note: `_freqs` is named with a leading underscore so that MLX does NOT track
    it as a learnable parameter (matches the reference where torch's `self.freqs`
    is a regular Python attribute, not a Parameter/buffer, so it is absent from
    the safetensors checkpoint).
    """

    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        freqs = mx.arange(self.freq_dim, dtype=mx.float32) / max(1, self.freq_dim)
        self._freqs = 1.0 / (10000.0 ** freqs)

    def _sin_cos(self, x: mx.array) -> mx.array:
        out = x[:, None] * self._freqs[None, :]
        return mx.concatenate([mx.sin(out), mx.cos(out)], axis=-1)

    def __call__(self, x: mx.array) -> mx.array:
        N, D = x.shape
        assert D == self.in_channels
        emb = self._sin_cos(x.reshape(-1).astype(mx.float32)).reshape(N, -1)
        if emb.shape[1] < self.channels:
            pad = mx.zeros((N, self.channels - emb.shape[1]), dtype=emb.dtype)
            emb = mx.concatenate([emb, pad], axis=-1)
        return emb


class RotaryPositionEmbedder(nn.Module):
    """
    RoPE that consumes integer (or float) spatial coords and produces a complex
    (cos, sin) phase tensor with shape [..., head_dim//2, 2].
    """

    def __init__(self, head_dim: int, dim: int = 3, rope_freq: Tuple[float, float] = (1.0, 10000.0)):
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        self.dim = dim
        self.rope_freq = rope_freq
        self.freq_dim = head_dim // 2 // dim
        freqs = mx.arange(self.freq_dim, dtype=mx.float32) / max(1, self.freq_dim)
        # Private (underscore) attribute: see AbsolutePositionEmbedder._freqs note.
        self._freqs = rope_freq[0] / (rope_freq[1] ** freqs)

    def _get_phases_complex(self, indices: mx.array) -> mx.array:
        """Returns [..., freq_dim, 2] with (cos, sin)."""
        phi = indices.reshape(-1).astype(mx.float32)[:, None] * self._freqs[None, :]
        return mx.stack([mx.cos(phi), mx.sin(phi)], axis=-1)

    def __call__(self, indices: mx.array) -> mx.array:
        """indices: [..., N, dim]. Returns phases [..., N, head_dim//2, 2]."""
        assert indices.shape[-1] == self.dim
        flat = indices.reshape(-1, self.dim)
        phases = self._get_phases_complex(flat)  # [N*dim, freq_dim, 2]
        phases = phases.reshape(flat.shape[0], self.dim * self.freq_dim, 2)
        # Pad to head_dim/2 if necessary.
        cur = phases.shape[1]
        target = self.head_dim // 2
        if cur < target:
            pad = mx.zeros((phases.shape[0], target - cur, 2), dtype=phases.dtype)
            pad = pad.at[:, :, 0].add(1.0) if hasattr(pad, "at") else pad
            # construct (1,0) pad explicitly:
            pad_cos = mx.ones((phases.shape[0], target - cur, 1), dtype=phases.dtype)
            pad_sin = mx.zeros((phases.shape[0], target - cur, 1), dtype=phases.dtype)
            pad = mx.concatenate([pad_cos, pad_sin], axis=-1)
            phases = mx.concatenate([phases, pad], axis=1)
        new_shape = tuple(indices.shape[:-1]) + (target, 2)
        return phases.reshape(new_shape)

    @staticmethod
    def apply_rotary_embedding(x: mx.array, phases: mx.array) -> mx.array:
        """
        Apply RoPE.

        x: [..., L, H, head_dim], dtype f16/bf16/f32.
        phases: [..., L, head_dim//2, 2] (cos, sin), broadcastable along H.

        Returns x with same dtype as input.
        """
        in_dtype = x.dtype
        xf = x.astype(mx.float32)
        s = xf.shape
        xf = xf.reshape(*s[:-1], s[-1] // 2, 2)
        # phases shape: [..., L, head_dim//2, 2]; insert head axis broadcast.
        # x shape: [..., L, H, head_dim//2, 2]; expand phases over H.
        ph = phases
        # ph is one-fewer ndim than xf (no H axis); add one before the head_dim//2 axis.
        ph = mx.expand_dims(ph, axis=-3)
        cos = ph[..., 0]
        sin = ph[..., 1]
        x_re = xf[..., 0]
        x_im = xf[..., 1]
        out_re = x_re * cos - x_im * sin
        out_im = x_re * sin + x_im * cos
        out = mx.stack([out_re, out_im], axis=-1)
        out = out.reshape(*s)
        return out.astype(in_dtype)

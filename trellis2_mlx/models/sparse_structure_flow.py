"""
SparseStructureFlowModel: dense Transformer over a [B, C, R, R, R] grid.

Mirrors trellis2/models/sparse_structure_flow.py.

Input format: x is provided as channel-first [B, C, R, R, R] (compat with the
reference); we permute internally to [B, R*R*R, C] for the transformer and back.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from ..modules.norm import LayerNorm32
from ..modules.transformer_blocks import ModulatedTransformerCrossBlock
from ..ops.rope import AbsolutePositionEmbedder, RotaryPositionEmbedder


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep encoding + 2-layer MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        # nn.Sequential(Linear, SiLU, Linear): flatten to mlp_0 / mlp_2 in MLX.
        self.mlp_0 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.mlp_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    @staticmethod
    def timestep_embedding(t: mx.array, dim: int, max_period: int = 10000) -> mx.array:
        half = dim // 2
        freqs = mx.exp(
            -math.log(max_period) * mx.arange(half, dtype=mx.float32) / max(1, half)
        )
        args = t[:, None].astype(mx.float32) * freqs[None, :]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            emb = mx.concatenate([emb, mx.zeros_like(emb[:, :1])], axis=-1)
        return emb

    def __call__(self, t: mx.array) -> mx.array:
        emb = self.timestep_embedding(t, self.frequency_embedding_size)
        emb = self.mlp_0(emb)
        emb = nn.silu(emb)
        return self.mlp_2(emb)


class SparseStructureFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        pe_mode: str = "ape",
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        dtype: str = "float32",
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.pe_mode = pe_mode
        self.share_mod = share_mod
        from ..ops import str_to_dtype as _dt
        self.compute_dtype = _dt(dtype)

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation_0 = nn.SiLU()
            self.adaLN_modulation_1 = nn.Linear(model_channels, 6 * model_channels, bias=True)

        if pe_mode == "ape":
            ape = AbsolutePositionEmbedder(model_channels, 3)
            grid = np.stack(np.meshgrid(*[np.arange(resolution)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
            self.pos_emb = ape(mx.array(grid, dtype=mx.float32))  # [R^3, C]
            self.rope_phases = None
        elif pe_mode == "rope":
            rope = RotaryPositionEmbedder(model_channels // self.num_heads, 3, rope_freq)
            grid = np.stack(np.meshgrid(*[np.arange(resolution)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
            self.rope_phases = rope(mx.array(grid, dtype=mx.float32))
            self.pos_emb = None
        else:
            raise ValueError(pe_mode)

        self.input_layer = nn.Linear(in_channels, model_channels)
        self.blocks = [
            ModulatedTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=mlp_ratio,
                attn_mode="full",
                use_rope=(pe_mode == "rope"),
                rope_freq=rope_freq,
                share_mod=share_mod,
                qk_rms_norm=qk_rms_norm,
                qk_rms_norm_cross=qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ]
        self.out_layer = nn.Linear(model_channels, out_channels)

    def __call__(self, x: mx.array, t: mx.array, cond: mx.array) -> mx.array:
        # x: [B, C, R, R, R] -> [B, R^3, C]
        B = x.shape[0]
        R = self.resolution
        assert tuple(x.shape) == (B, self.in_channels, R, R, R), \
            f"input shape {x.shape} != {(B, self.in_channels, R, R, R)}"
        h = x.transpose(0, 2, 3, 4, 1).reshape(B, R * R * R, self.in_channels)
        h = self.input_layer(h)
        if self.pe_mode == "ape":
            h = h + self.pos_emb[None]
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation_1(self.adaLN_modulation_0(t_emb))
        t_emb = t_emb.astype(self.compute_dtype)
        h = h.astype(self.compute_dtype)
        cond = cond.astype(self.compute_dtype)
        for blk in self.blocks:
            h = blk(h, t_emb, cond, self.rope_phases)
        h = h.astype(x.dtype)
        # final fp32 layer-norm (no params)
        from ..modules.norm import _layer_norm_fp32
        h = _layer_norm_fp32(h, (h.shape[-1],), None, None, 1e-5)
        h = self.out_layer(h)
        h = h.reshape(B, R, R, R, self.out_channels).transpose(0, 4, 1, 2, 3)
        return h

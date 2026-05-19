"""
SLatFlowModel: sparse Transformer in shape/texture latent stages.

Mirrors trellis2/models/structured_latent_flow.py.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from ..modules.linear import SparseLinear
from ..modules.sparse_blocks import ModulatedSparseTransformerCrossBlock
from ..modules.norm import _layer_norm_fp32
from ..ops.sparse_tensor import SparseTensor, VarLenTensor, sparse_cat
from ..ops.rope import AbsolutePositionEmbedder
from .sparse_structure_flow import TimestepEmbedder


class SLatFlowModel(nn.Module):
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
        self.cond_channels = cond_channels
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
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)
        else:
            self.pos_embedder = None

        self.input_layer = SparseLinear(in_channels, model_channels)
        self.blocks = [
            ModulatedSparseTransformerCrossBlock(
                model_channels, cond_channels, num_heads=self.num_heads,
                mlp_ratio=mlp_ratio, attn_mode="full",
                use_rope=(pe_mode == "rope"), rope_freq=rope_freq,
                share_mod=share_mod,
                qk_rms_norm=qk_rms_norm, qk_rms_norm_cross=qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ]
        self.out_layer = SparseLinear(model_channels, out_channels)

    def __call__(
        self,
        x: SparseTensor,
        t: mx.array,
        cond: Union[mx.array, List[mx.array]],
        concat_cond: Optional[SparseTensor] = None,
        **kwargs,
    ) -> SparseTensor:
        if concat_cond is not None:
            x = sparse_cat([x, concat_cond], dim=-1)
        if isinstance(cond, list):
            cond = VarLenTensor.from_tensor_list(cond)

        h = self.input_layer(x)
        h = h.astype(self.compute_dtype)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation_1(self.adaLN_modulation_0(t_emb))
        t_emb = t_emb.astype(self.compute_dtype)
        if isinstance(cond, VarLenTensor):
            cond = cond.astype(self.compute_dtype)
        else:
            cond = cond.astype(self.compute_dtype)

        if self.pe_mode == "ape":
            pe = self.pos_embedder(h.coords[:, 1:].astype(mx.float32))
            h = h + pe.astype(self.compute_dtype)

        for blk in self.blocks:
            h = blk(h, t_emb, cond)

        h = h.astype(x.dtype)
        h = h.replace(_layer_norm_fp32(h.feats, (h.feats.shape[-1],), None, None, 1e-5))
        h = self.out_layer(h)
        return h

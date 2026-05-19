"""
SparseUnetVaeEncoder / SparseUnetVaeDecoder for MLX.

Mirrors trellis2/models/sc_vaes/sparse_unet_vae.py.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ...modules.linear import SparseLinear
from ...modules.sparse_unet import (
    SparseResBlock3d,
    SparseResBlockDownsample3d,
    SparseResBlockUpsample3d,
    SparseResBlockS2C3d,
    SparseResBlockC2S3d,
    SparseConvNeXtBlock3d,
)
from ...modules.norm import _layer_norm_fp32
from ...ops.sparse_tensor import SparseTensor


_BLOCK_TYPES = {
    "SparseResBlock3d": SparseResBlock3d,
    "SparseResBlockDownsample3d": SparseResBlockDownsample3d,
    "SparseResBlockUpsample3d": SparseResBlockUpsample3d,
    "SparseResBlockS2C3d": SparseResBlockS2C3d,
    "SparseResBlockC2S3d": SparseResBlockC2S3d,
    "SparseConvNeXtBlock3d": SparseConvNeXtBlock3d,
}


def _make_block(name: str, in_ch: int, out_ch: Optional[int] = None, **kwargs):
    cls = _BLOCK_TYPES[name]
    if out_ch is not None:
        return cls(in_ch, out_ch, **kwargs)
    return cls(in_ch, **kwargs)


class SparseUnetVaeEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        down_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks_list = num_blocks

        self.input_layer = SparseLinear(in_channels, model_channels[0])
        self.to_latent = SparseLinear(model_channels[-1], 2 * latent_channels)

        # Build blocks as a list-of-lists. We store them in `blocks` (list of lists)
        # and ensure mlx.nn discovers them via a flat attribute view.
        self.blocks: List[List[nn.Module]] = []
        for i in range(len(num_blocks)):
            level: List[nn.Module] = []
            for _ in range(num_blocks[i]):
                level.append(_make_block(block_type[i], model_channels[i], **block_args[i]))
            if i < len(num_blocks) - 1:
                level.append(
                    _make_block(down_block_type[i], model_channels[i], model_channels[i + 1], **block_args[i])
                )
            self.blocks.append(level)

    def __call__(self, x: SparseTensor, sample_posterior: bool = False, return_raw: bool = False):
        h = self.input_layer(x)
        for level in self.blocks:
            for blk in level:
                out = blk(h)
                h = out if not isinstance(out, tuple) else out[0]
        h = h.astype(x.dtype)
        h = h.replace(_layer_norm_fp32(h.feats, (h.feats.shape[-1],), None, None, 1e-5))
        h = self.to_latent(h)
        mean, logvar = mx.split(h.feats, 2, axis=-1)
        if sample_posterior:
            std = mx.exp(0.5 * logvar)
            z = mean + std * mx.random.normal(mean.shape)
        else:
            z = mean
        z = h.replace(z)
        if return_raw:
            return z, mean, logvar
        return z


class SparseUnetVaeDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
        pred_subdiv: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_blocks_list = num_blocks
        self.pred_subdiv = pred_subdiv
        self.low_vram = False

        self.output_layer = SparseLinear(model_channels[-1], out_channels)
        self.from_latent = SparseLinear(latent_channels, model_channels[0])

        self.blocks: List[List[nn.Module]] = []
        for i in range(len(num_blocks)):
            level: List[nn.Module] = []
            for _ in range(num_blocks[i]):
                level.append(_make_block(block_type[i], model_channels[i], **block_args[i]))
            if i < len(num_blocks) - 1:
                level.append(
                    _make_block(
                        up_block_type[i],
                        model_channels[i],
                        model_channels[i + 1],
                        pred_subdiv=pred_subdiv,
                        **block_args[i],
                    )
                )
            self.blocks.append(level)

    def __call__(
        self,
        x: SparseTensor,
        guide_subs: Optional[List[SparseTensor]] = None,
        return_subs: bool = False,
    ):
        if guide_subs is not None:
            assert not self.pred_subdiv, \
                "guide_subs can only be passed to decoders built with pred_subdiv=False"

        h = self.from_latent(x)
        subs = []
        for i, level in enumerate(self.blocks):
            for j, blk in enumerate(level):
                last_in_level = (i < len(self.blocks) - 1) and (j == len(level) - 1)
                if last_in_level:
                    if self.pred_subdiv:
                        h, sub = blk(h)
                        subs.append(sub)
                    else:
                        guide = guide_subs[i] if guide_subs is not None else None
                        h = blk(h, subdiv=guide)
                else:
                    out = blk(h)
                    h = out if not isinstance(out, tuple) else out[0]

        h = h.astype(x.dtype)
        h = h.replace(_layer_norm_fp32(h.feats, (h.feats.shape[-1],), None, None, 1e-5))
        h = self.output_layer(h)
        if return_subs:
            return h, subs
        return h

    def upsample(self, x: SparseTensor, upsample_times: int) -> mx.array:
        assert self.pred_subdiv, "upsample requires pred_subdiv=True"
        h = self.from_latent(x)
        for i, level in enumerate(self.blocks):
            if i == upsample_times:
                return h.coords
            for j, blk in enumerate(level):
                last_in_level = (i < len(self.blocks) - 1) and (j == len(level) - 1)
                if last_in_level and self.pred_subdiv:
                    h, _ = blk(h)
                else:
                    out = blk(h)
                    h = out if not isinstance(out, tuple) else out[0]
        return h.coords

    def set_resolution(self, resolution: int) -> None:
        self.resolution = resolution

"""
SparseStructureEncoder / SparseStructureDecoder: dense 3D ConvNet over [B, C, D, H, W].

MLX uses channel-last for Conv3d: [B, D, H, W, C]. We permute on entry/exit so the
caller can keep using channel-first tensors.

Mirrors trellis2/models/sparse_structure_vae.py.
"""
from __future__ import annotations

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..modules.norm import GroupNorm32, ChannelLayerNorm32


def _norm(norm_type: str, channels: int) -> nn.Module:
    if norm_type == "group":
        return GroupNorm32(32, channels)
    if norm_type == "layer":
        return ChannelLayerNorm32(channels, elementwise_affine=True, eps=1e-6)
    raise ValueError(norm_type)


def _pixel_shuffle_3d_channel_last(x: mx.array, factor: int) -> mx.array:
    """
    x: [B, D, H, W, C * factor^3] -> [B, D*factor, H*factor, W*factor, C].

    Mirrors the channel-first pixel_shuffle_3d in modules/spatial.py but in NDHWC layout.
    """
    B, D, H, W, C = x.shape
    f = factor
    Cnew = C // (f ** 3)
    x = x.reshape(B, D, H, W, f, f, f, Cnew)
    # We want output[b, d*f+kd, h*f+kh, w*f+kw, c] = x[b,d,h,w,kd,kh,kw,c]
    x = x.transpose(0, 1, 4, 2, 5, 3, 6, 7)  # B, D, f, H, f, W, f, Cnew
    return x.reshape(B, D * f, H * f, W * f, Cnew)


class ResBlock3d(nn.Module):
    def __init__(self, channels: int, out_channels: Optional[int] = None, norm_type: str = "layer"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.norm1 = _norm(norm_type, channels)
        self.norm2 = _norm(norm_type, self.out_channels)
        self.conv1 = nn.Conv3d(channels, self.out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.skip_connection = (
            nn.Conv3d(channels, self.out_channels, kernel_size=1)
            if channels != self.out_channels
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, D, H, W, C] (channel-last)
        h = self.norm1(x)
        h = nn.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nn.silu(h)
        h = self.conv2(h)
        skip = x if self.skip_connection is None else self.skip_connection(x)
        return h + skip


class DownsampleBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode: str = "conv"):
        super().__init__()
        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)
            self.mode = "conv"
        else:
            assert in_channels == out_channels
            self.mode = "avgpool"

    def __call__(self, x: mx.array) -> mx.array:
        if self.mode == "conv":
            return self.conv(x)
        # avg_pool with kernel=2 stride=2 over D,H,W (channel-last)
        B, D, H, W, C = x.shape
        return x.reshape(B, D // 2, 2, H // 2, 2, W // 2, 2, C).mean(axis=(2, 4, 6))


class UpsampleBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode: str = "conv"):
        super().__init__()
        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels * 8, kernel_size=3, padding=1)
            self.mode = "conv"
        else:
            assert in_channels == out_channels
            self.mode = "nearest"

    def __call__(self, x: mx.array) -> mx.array:
        if self.mode == "conv":
            x = self.conv(x)
            return _pixel_shuffle_3d_channel_last(x, 2)
        B, D, H, W, C = x.shape
        x = mx.broadcast_to(x[:, :, None, :, None, :, None, :], (B, D, 2, H, 2, W, 2, C))
        return x.reshape(B, D * 2, H * 2, W * 2, C)


class _BaseSparseStructureVAE(nn.Module):
    """Shared boilerplate for the encoder + decoder."""

    @staticmethod
    def _to_channel_last(x: mx.array) -> mx.array:
        return x.transpose(0, 2, 3, 4, 1)

    @staticmethod
    def _to_channel_first(x: mx.array) -> mx.array:
        return x.transpose(0, 4, 1, 2, 3)


class SparseStructureEncoder(_BaseSparseStructureVAE):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: str = "layer",
        use_fp16: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.channels = channels

        self.input_layer = nn.Conv3d(in_channels, channels[0], kernel_size=3, padding=1)
        blocks = []
        for i, ch in enumerate(channels):
            for _ in range(num_res_blocks):
                blocks.append(ResBlock3d(ch, ch, norm_type=norm_type))
            if i < len(channels) - 1:
                blocks.append(DownsampleBlock3d(ch, channels[i + 1]))
        self.blocks = blocks

        self.middle_block = [ResBlock3d(channels[-1], channels[-1], norm_type=norm_type)
                             for _ in range(num_res_blocks_middle)]

        # Reference: out_layer = nn.Sequential(norm, SiLU, Conv3d) -> keys out_layer.0/2
        from ..modules.transformer_blocks import _silu_module
        self.out_layer = [
            _norm(norm_type, channels[-1]),
            _silu_module(),
            nn.Conv3d(channels[-1], latent_channels * 2, kernel_size=3, padding=1),
        ]

    def __call__(self, x: mx.array, sample_posterior: bool = False) -> mx.array:
        h = self._to_channel_last(x)
        h = self.input_layer(h)
        for blk in self.blocks:
            h = blk(h)
        for blk in self.middle_block:
            h = blk(h)
        h = self.out_layer[0](h)
        h = self.out_layer[1]._fn(h)
        h = self.out_layer[2](h)
        h = self._to_channel_first(h)
        mean, logvar = mx.split(h, 2, axis=1)
        if sample_posterior:
            std = mx.exp(0.5 * logvar)
            return mean + std * mx.random.normal(mean.shape)
        return mean


class SparseStructureDecoder(_BaseSparseStructureVAE):
    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: str = "layer",
        use_fp16: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.channels = channels

        self.input_layer = nn.Conv3d(latent_channels, channels[0], kernel_size=3, padding=1)
        self.middle_block = [ResBlock3d(channels[0], channels[0], norm_type=norm_type)
                             for _ in range(num_res_blocks_middle)]
        blocks = []
        for i, ch in enumerate(channels):
            for _ in range(num_res_blocks):
                blocks.append(ResBlock3d(ch, ch, norm_type=norm_type))
            if i < len(channels) - 1:
                blocks.append(UpsampleBlock3d(ch, channels[i + 1]))
        self.blocks = blocks

        from ..modules.transformer_blocks import _silu_module
        self.out_layer = [
            _norm(norm_type, channels[-1]),
            _silu_module(),
            nn.Conv3d(channels[-1], out_channels, kernel_size=3, padding=1),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        h = self._to_channel_last(x)
        h = self.input_layer(h)
        for blk in self.middle_block:
            h = blk(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.out_layer[0](h)
        h = self.out_layer[1]._fn(h)
        h = self.out_layer[2](h)
        return self._to_channel_first(h)

"""
Submanifold sparse 3D convolution on MLX (slow but correct).

Mirrors flex_gemm.ops.spconv.sparse_submanifold_conv3d (stride=1, no padding):
the output sites are the same as the input active sites, and the kernel
gathers feats from active neighbors only.

Algorithm:
  1. For each active voxel i, find which kernel offsets land on another active
     voxel; build neighbor_map [N, K^3] with -1 where missing (cached on the
     SparseTensor spatial cache, scoped by current scale).
  2. gather feats[neigh.clip(0)] -> [N, K^3, Ci]
  3. mask invalid entries to 0
  4. einsum with weight [Co, K^3, Ci]
  5. add bias

Optimization knobs:
- The neighbor map is cached so subsequent convs at the same scale reuse it.
- For kernel=1x1x1 we degenerate to a plain matmul.

Weight convention matches the reference: [Co, Kd, Kh, Kw, Ci] (channel-last in
both K and C dims). We flatten to [Co, K^3, Ci] internally.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .sparse_tensor import SparseTensor


__all__ = ["SparseConv3d", "build_submanifold_neighbor_map"]


def _coord_key(coords_np: np.ndarray, spatial_shape: Tuple[int, int, int]) -> np.ndarray:
    """Pack (b, x, y, z) -> int64 hash key."""
    b, x, y, z = coords_np[:, 0], coords_np[:, 1], coords_np[:, 2], coords_np[:, 3]
    D, H, W = spatial_shape
    return (((b.astype(np.int64) * (D + 1) + x) * (H + 1) + y) * (W + 1) + z)


def build_submanifold_neighbor_map(
    coords: mx.array,
    spatial_shape: Tuple[int, int, int],
    kernel_size: Tuple[int, int, int] = (3, 3, 3),
    dilation: Tuple[int, int, int] = (1, 1, 1),
) -> mx.array:
    """
    Returns [N, Kd*Kh*Kw] int32, with -1 for missing neighbors.
    Built on CPU via numpy/dict; cheap relative to the conv itself.
    """
    coords_np = np.asarray(coords, dtype=np.int64)
    N = coords_np.shape[0]
    Kd, Kh, Kw = kernel_size
    dD, dH, dW = dilation
    K = Kd * Kh * Kw
    if N == 0:
        return mx.zeros((0, K), dtype=mx.int32)

    keys = _coord_key(coords_np, spatial_shape)
    table = {int(k): i for i, k in enumerate(keys)}

    nbr = np.full((N, K), -1, dtype=np.int32)
    half = (Kd // 2, Kh // 2, Kw // 2)

    # Precompute offsets in raster (kd, kh, kw) order: matches reference weight layout.
    offsets = []
    for kd in range(Kd):
        for kh in range(Kh):
            for kw in range(Kw):
                offsets.append((
                    (kd - half[0]) * dD,
                    (kh - half[1]) * dH,
                    (kw - half[2]) * dW,
                ))

    bs = coords_np[:, 0]
    xs = coords_np[:, 1]
    ys = coords_np[:, 2]
    zs = coords_np[:, 3]
    D, H, W = spatial_shape
    for k_idx, (od, oh, ow) in enumerate(offsets):
        nx = xs + od
        ny = ys + oh
        nz = zs + ow
        valid = (nx >= 0) & (nx < D) & (ny >= 0) & (ny < H) & (nz >= 0) & (nz < W)
        keys_k = _coord_key(np.stack([bs, nx, ny, nz], axis=1), spatial_shape)
        # Lookup
        for i in np.where(valid)[0]:
            j = table.get(int(keys_k[i]), -1)
            nbr[i, k_idx] = j

    return mx.array(nbr, dtype=mx.int32)


class SparseConv3d(nn.Module):
    """
    Submanifold sparse Conv3D. Only supports stride=1 (matches flex_gemm).

    Parameters loaded from .safetensors are expected with shape [Co, Kd, Kh, Kw, Ci]
    (same as the reference flex_gemm path which already permutes torch weights).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: int = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        padding=None,
        bias: bool = True,
        indice_key: Optional[str] = None,
    ):
        super().__init__()
        assert stride == 1, "Only stride=1 is supported (submanifold)."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
        self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation,) * 3

        # MLX modules use plain attribute assignment for params; arrays are auto-tracked.
        self.weight = mx.zeros(
            (out_channels, *self.kernel_size, in_channels), dtype=mx.float32
        )
        if bias:
            self.bias = mx.zeros((out_channels,), dtype=mx.float32)
        # else: don't create the attribute — MLX would treat None as a missing param.

    def __call__(self, x: SparseTensor) -> SparseTensor:
        Kd, Kh, Kw = self.kernel_size
        K = Kd * Kh * Kw
        bias = getattr(self, "bias", None)

        # Fast path for 1x1x1: behaves like a Linear.
        if Kd == Kh == Kw == 1:
            w = self.weight.reshape(self.out_channels, self.in_channels)
            out = x.feats @ w.T
            if bias is not None:
                out = out + bias
            return x.replace(out)

        cache_key = f"submconv_nbr_{Kd}x{Kh}x{Kw}_d{self.dilation}"
        nbr = x.get_spatial_cache(cache_key)
        if nbr is None:
            nbr = build_submanifold_neighbor_map(
                x.coords, x.spatial_shape, self.kernel_size, self.dilation
            )
            x.register_spatial_cache(cache_key, nbr)

        # gather feats[nbr.clip(0)] -> [N, K, Ci]; mask -1 -> 0
        N = x.feats.shape[0]
        nbr_clipped = mx.maximum(nbr, 0)  # [N, K]
        gathered = x.feats[nbr_clipped]  # [N, K, Ci]
        mask = (nbr >= 0).astype(gathered.dtype)[:, :, None]
        gathered = gathered * mask

        # einsum('nki,oki->no') == reshape + matmul
        # w_flat: [Co, K, Ci] -> reshape to [Co, K*Ci]; g_flat: [N, K*Ci]
        w_flat = self.weight.reshape(self.out_channels, K * self.in_channels)
        g_flat = gathered.reshape(N, K * self.in_channels)
        out = g_flat @ w_flat.T
        if bias is not None:
            out = out + bias
        return x.replace(out)


class SparseInverseConv3d(nn.Module):
    """Placeholder for parity with the reference module set; not used at inference."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "SparseInverseConv3d is not used in the MLX inference path; "
            "the reference flex_gemm backend also raises NotImplementedError."
        )

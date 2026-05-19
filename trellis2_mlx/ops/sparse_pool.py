"""
Sparse downsample/upsample + spatial2channel/channel2spatial on MLX.

Mirrors trellis2/modules/sparse/spatial/*.py. Implemented with numpy assists
for the unique/scatter steps (negligible runtime compared to convs).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .sparse_tensor import SparseTensor


__all__ = [
    "SparseDownsample",
    "SparseUpsample",
    "SparseSpatial2Channel",
    "SparseChannel2Spatial",
    "SparseSubdivide",
    "sparse_nearest_interpolate",
    "sparse_trilinear_interpolate",
]


def _downsample_index(coords_np: np.ndarray, factor: int):
    """Return (new_coords [M, 4], idx [N] mapping each input row to its parent slot)."""
    new = coords_np.copy()
    new[:, 1:] //= factor
    keys = (((new[:, 0].astype(np.int64) * (1 << 40))
             + new[:, 1].astype(np.int64) * (1 << 20))
            + new[:, 2].astype(np.int64) * (1 << 10)) + new[:, 3].astype(np.int64)
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    # Reconstruct unique coordinates from any representative row.
    order = np.argsort(inverse, kind="stable")
    first = order[np.concatenate(([0], np.where(np.diff(inverse[order]) != 0)[0] + 1))]
    new_coords = new[first][np.argsort(inverse[first])]
    return new_coords.astype(np.int32), inverse.astype(np.int32)


def _segment_reduce(feats: mx.array, idx: mx.array, M: int, mode: str = "mean") -> mx.array:
    """Group-reduce rows of feats by integer group id idx into M groups."""
    feats_np = np.asarray(feats)
    idx_np = np.asarray(idx, dtype=np.int64)
    C = feats_np.shape[1]
    out = np.zeros((M, C), dtype=feats_np.dtype)
    if mode == "mean":
        cnt = np.zeros((M,), dtype=np.int64)
        np.add.at(out, idx_np, feats_np)
        np.add.at(cnt, idx_np, 1)
        cnt = np.maximum(cnt, 1)
        out = out / cnt[:, None]
    elif mode == "max":
        out[:] = -np.inf
        np.maximum.at(out, idx_np, feats_np)
    elif mode == "sum":
        np.add.at(out, idx_np, feats_np)
    else:
        raise ValueError(mode)
    return mx.array(out)


class SparseDownsample(nn.Module):
    def __init__(self, factor: int, mode: str = "mean"):
        super().__init__()
        self.factor = factor
        assert mode in ("mean", "max")
        self.mode = mode

    def __call__(self, x: SparseTensor) -> SparseTensor:
        cache_key = f"downsample_{self.factor}"
        cache = x.get_spatial_cache(cache_key)
        if cache is None:
            coords_np = np.asarray(x.coords, dtype=np.int32)
            new_coords_np, idx_np = _downsample_index(coords_np, self.factor)
            spatial = tuple((s + self.factor - 1) // self.factor for s in x.spatial_shape)
            new_coords = mx.array(new_coords_np)
            idx = mx.array(idx_np)
        else:
            new_coords, idx = cache
            spatial = None

        new_feats = _segment_reduce(x.feats, idx, new_coords.shape[0], self.mode)
        out = SparseTensor(new_feats, new_coords)
        out._scale = tuple(s * self.factor for s in x._scale)
        out._spatial_cache = x._spatial_cache  # share cache across scales (keyed internally)
        if cache is None:
            x.register_spatial_cache(cache_key, (new_coords, idx))
            out.register_spatial_cache(f"upsample_{self.factor}", (x.coords, idx))
            out.register_spatial_cache("shape", spatial)
        return out


class SparseUpsample(nn.Module):
    """Nearest-neighbor upsample driven by a binary subdivision mask."""

    def __init__(self, factor: int):
        super().__init__()
        self.factor = factor

    def __call__(self, x: SparseTensor, subdivision: Optional[SparseTensor] = None) -> SparseTensor:
        cache_key = f"upsample_{self.factor}"
        cache = x.get_spatial_cache(cache_key)
        if cache is None:
            if subdivision is None:
                raise ValueError(
                    "SparseUpsample without cache requires a subdivision SparseTensor "
                    "(bool, [N, factor^3])."
                )
            sub_np = np.asarray(subdivision.feats).astype(bool)  # [N, F^3]
            n_leaf = sub_np.sum(axis=-1)  # [N]
            subidx = np.where(sub_np)
            child_local = subidx[1]  # which child slot inside each parent
            parent_idx = np.repeat(np.arange(sub_np.shape[0]), n_leaf)  # NB ordering matches np.where
            # Reorder: np.where iterates over rows then columns, which matches the
            # reference torch.nonzero(); the parent index per child therefore is just
            # subidx[0]; but the reference uses repeat_interleave by sum-per-row, which
            # is identical to subidx[0]. Keep subidx[0] for clarity.
            parent_idx = subidx[0]
            coords_np = np.asarray(x.coords, dtype=np.int32).copy()
            new_coords_np = coords_np[parent_idx]
            new_coords_np[:, 1:] *= self.factor
            f = self.factor
            for i in range(3):
                offset = (child_local // (f ** i)) % f
                new_coords_np[:, 1 + i] += offset.astype(np.int32)
            idx_np = parent_idx.astype(np.int32)
            new_coords = mx.array(new_coords_np)
            idx = mx.array(idx_np)
        else:
            new_coords, idx = cache

        new_feats = x.feats[idx]
        out = SparseTensor(new_feats, new_coords)
        out._scale = tuple(s / self.factor for s in x._scale)
        if cache is not None:
            out._spatial_cache = x._spatial_cache
        return out


class SparseSubdivide(SparseUpsample):
    """Alias used by some block configs; behaves like SparseUpsample with subdivision."""
    pass


class SparseSpatial2Channel(nn.Module):
    """Pack each (factor^3) cube into the channel dimension."""

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = factor

    def __call__(self, x: SparseTensor) -> SparseTensor:
        cache_key = f"spatial2channel_{self.factor}"
        cache = x.get_spatial_cache(cache_key)
        f = self.factor
        f3 = f ** 3
        if cache is None:
            coords_np = np.asarray(x.coords, dtype=np.int32)
            parent = coords_np.copy()
            parent[:, 1:] //= f
            sub_xyz = coords_np[:, 1:] % f
            subidx = sub_xyz[:, 0] + sub_xyz[:, 1] * f + sub_xyz[:, 2] * (f * f)
            new_coords_np, idx_np = _downsample_index(coords_np, f)
            new_coords = mx.array(new_coords_np)
            idx = mx.array(idx_np)
            subidx_m = mx.array(subidx.astype(np.int32))
            spatial = tuple((s + f - 1) // f for s in x.spatial_shape)
        else:
            new_coords, idx, subidx_m = cache
            spatial = None

        N_in = x.feats.shape[0]
        C = x.feats.shape[1]
        N_out = new_coords.shape[0]
        # Place input feats at (idx * f3 + subidx) slots in a [N_out * f3, C] buffer.
        flat = mx.zeros((N_out * f3, C), dtype=x.feats.dtype)
        flat_idx = idx.astype(mx.int32) * f3 + subidx_m
        flat = _scatter_assign(flat, flat_idx, x.feats)
        new_feats = flat.reshape(N_out, C * f3)
        out = SparseTensor(new_feats, new_coords)
        out._scale = tuple(s * f for s in x._scale)
        out._spatial_cache = x._spatial_cache
        if cache is None:
            x.register_spatial_cache(cache_key, (new_coords, idx, subidx_m))
            out.register_spatial_cache(f"channel2spatial_{f}", (x.coords, idx, subidx_m))
            if spatial is not None:
                out.register_spatial_cache("shape", spatial)
        return out


class SparseChannel2Spatial(nn.Module):
    """Inverse of SparseSpatial2Channel."""

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = factor

    def __call__(self, x: SparseTensor, subdivision: Optional[SparseTensor] = None) -> SparseTensor:
        f = self.factor
        f3 = f ** 3
        cache_key = f"channel2spatial_{f}"
        cache = x.get_spatial_cache(cache_key)
        if cache is None:
            if subdivision is None:
                raise ValueError("SparseChannel2Spatial without cache requires a subdivision tensor.")
            sub_np = np.asarray(subdivision.feats).astype(bool)
            where = np.where(sub_np)
            parent_idx = where[0]
            child_local = where[1]
            coords_np = np.asarray(x.coords, dtype=np.int32).copy()
            new_coords_np = coords_np[parent_idx]
            new_coords_np[:, 1:] *= f
            for i in range(3):
                offset = (child_local // (f ** i)) % f
                new_coords_np[:, 1 + i] += offset.astype(np.int32)
            new_coords = mx.array(new_coords_np)
            idx = mx.array(parent_idx.astype(np.int32))
            subidx = mx.array(child_local.astype(np.int32))
        else:
            new_coords, idx, subidx = cache

        # Split feats from [N, C*f^3] back into [N*f^3, C] and gather.
        x_flat = x.feats.reshape(x.feats.shape[0] * f3, -1)
        gather_idx = idx.astype(mx.int32) * f3 + subidx
        new_feats = x_flat[gather_idx]
        out = SparseTensor(new_feats, new_coords)
        out._scale = tuple(s / f for s in x._scale)
        if cache is not None:
            out._spatial_cache = x._spatial_cache
        return out


def _scatter_assign(target: mx.array, indices: mx.array, source: mx.array) -> mx.array:
    """
    target[indices] = source. MLX exposes this as target.at[indices].add / fancy indexing.
    """
    out = target
    out[indices] = source
    return out


# Light-weight passthroughs for completeness — not required by the inference path
def sparse_nearest_interpolate(x: SparseTensor, *args, **kwargs):
    raise NotImplementedError("sparse_nearest_interpolate is not used at inference time.")


def sparse_trilinear_interpolate(x: SparseTensor, *args, **kwargs):
    raise NotImplementedError("sparse_trilinear_interpolate is not used at inference time.")

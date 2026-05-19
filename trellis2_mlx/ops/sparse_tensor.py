"""
SparseTensor / VarLenTensor for MLX.

Mirrors the API of trellis2/modules/sparse/basic.py but backed by mlx.core.array.
Coordinates are stored as int32 (b, x, y, z) and features as float arrays.

The spatial cache (used by sparse conv neighbor maps, downsample/upsample
permutations, RoPE phases, etc.) is keyed by the current scale just like the
reference, so caches survive across .replace() calls.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import mlx.core as mx


# ---------------------------------------------------------------------------
# VarLenTensor: a [Ntotal, *feat_shape] array with per-batch row slices
# ---------------------------------------------------------------------------


class VarLenTensor:
    def __init__(self, feats: mx.array, layout: Optional[List[slice]] = None):
        self.feats = feats
        if layout is None:
            layout = [slice(0, feats.shape[0])]
        self.layout = layout
        self._cache: Dict[str, Any] = {}

    @staticmethod
    def layout_from_seqlen(seqlen) -> List[slice]:
        layout = []
        start = 0
        for l in seqlen:
            layout.append(slice(start, start + int(l)))
            start += int(l)
        return layout

    @staticmethod
    def from_tensor_list(tensor_list: List[mx.array]) -> "VarLenTensor":
        feats = mx.concatenate(tensor_list, axis=0)
        layout = []
        start = 0
        for t in tensor_list:
            layout.append(slice(start, start + t.shape[0]))
            start += t.shape[0]
        return VarLenTensor(feats, layout)

    def to_tensor_list(self) -> List[mx.array]:
        return [self.feats[s] for s in self.layout]

    def __len__(self) -> int:
        return len(self.layout)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (len(self.layout),) + tuple(self.feats.shape[1:])

    def dim(self) -> int:
        return len(self.shape)

    @property
    def ndim(self) -> int:
        return self.dim()

    @property
    def dtype(self):
        return self.feats.dtype

    @property
    def seqlen(self) -> mx.array:
        if "seqlen" not in self._cache:
            self._cache["seqlen"] = mx.array(
                [l.stop - l.start for l in self.layout], dtype=mx.int32
            )
        return self._cache["seqlen"]

    @property
    def cum_seqlen(self) -> mx.array:
        if "cum_seqlen" not in self._cache:
            sl = self.seqlen
            self._cache["cum_seqlen"] = mx.concatenate(
                [mx.array([0], dtype=mx.int32), mx.cumsum(sl)]
            )
        return self._cache["cum_seqlen"]

    @property
    def batch_broadcast_map(self) -> mx.array:
        if "batch_broadcast_map" not in self._cache:
            parts = [mx.full((l.stop - l.start,), i, dtype=mx.int32) for i, l in enumerate(self.layout)]
            self._cache["batch_broadcast_map"] = mx.concatenate(parts) if parts else mx.array([], dtype=mx.int32)
        return self._cache["batch_broadcast_map"]

    def astype(self, dtype) -> "VarLenTensor":
        return self.replace(self.feats.astype(dtype))

    # NB: MLX has a single device (unified memory). `.to(device)` is a no-op.
    def to(self, *args, **kwargs) -> "VarLenTensor":
        dtype = kwargs.get("dtype", None)
        for a in args:
            if isinstance(a, mx.Dtype):
                dtype = a
        if dtype is not None:
            return self.astype(dtype)
        return self

    def type(self, dtype) -> "VarLenTensor":
        return self.astype(dtype)

    def cpu(self):
        return self

    def cuda(self):
        return self

    def half(self):
        return self.astype(mx.float16)

    def float(self):
        return self.astype(mx.float32)

    def reshape(self, *shape) -> "VarLenTensor":
        return self.replace(self.feats.reshape(self.feats.shape[0], *shape))

    def replace(self, feats: mx.array) -> "VarLenTensor":
        out = VarLenTensor(feats=feats, layout=self.layout)
        out._cache = self._cache
        return out

    def to_dense(self, max_length: Optional[int] = None) -> Tuple[mx.array, mx.array]:
        N = len(self)
        L = int(max_length) if max_length is not None else int(mx.max(self.seqlen).item())
        spatial = self.feats.shape[1:]
        out = mx.zeros((N, L) + spatial, dtype=self.feats.dtype)
        # Scatter rows of feats into [N, L, ...] using layout offsets.
        # Use a python loop here for clarity; this is hit only on cross-attention contexts.
        for i, s in enumerate(self.layout):
            ln = s.stop - s.start
            if ln == 0:
                continue
            out[i, :ln] = self.feats[s]
        mask_idx = mx.arange(L, dtype=mx.int32)
        mask = mask_idx[None] < self.seqlen[:, None]
        return out, mask

    # ---- elementwise ops ----

    def __neg__(self) -> "VarLenTensor":
        return self.replace(-self.feats)

    def _broadcast_other(self, other):
        if isinstance(other, VarLenTensor):
            return other.feats
        if isinstance(other, mx.array) and other.ndim == self.feats.ndim:
            # Per-batch broadcast: expand along seqlen by the broadcast map.
            try:
                if other.shape[0] == self.shape[0]:
                    bmap = self.batch_broadcast_map
                    return other[bmap]
            except Exception:
                pass
        return other

    def __add__(self, other):
        return self.replace(self.feats + self._broadcast_other(other))

    def __radd__(self, other):
        return self.replace(self._broadcast_other(other) + self.feats)

    def __sub__(self, other):
        return self.replace(self.feats - self._broadcast_other(other))

    def __rsub__(self, other):
        return self.replace(self._broadcast_other(other) - self.feats)

    def __mul__(self, other):
        return self.replace(self.feats * self._broadcast_other(other))

    def __rmul__(self, other):
        return self.replace(self._broadcast_other(other) * self.feats)

    def __truediv__(self, other):
        return self.replace(self.feats / self._broadcast_other(other))

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = list(range(*idx.indices(self.shape[0])))
        parts, new_layout, start = [], [], 0
        for old_idx in idx:
            seg = self.feats[self.layout[old_idx]]
            parts.append(seg)
            new_layout.append(slice(start, start + seg.shape[0]))
            start += seg.shape[0]
        feats = mx.concatenate(parts, axis=0) if parts else mx.zeros((0,) + self.feats.shape[1:])
        return VarLenTensor(feats, new_layout)

    def __repr__(self) -> str:
        return f"VarLenTensor(shape={tuple(self.shape)}, dtype={self.dtype})"


def varlen_cat(inputs: List[VarLenTensor], dim: int = 0) -> VarLenTensor:
    if dim == 0:
        feats = mx.concatenate([x.feats for x in inputs], axis=0)
        new_layout, start = [], 0
        for x in inputs:
            for l in x.layout:
                new_layout.append(slice(start, start + l.stop - l.start))
                start += l.stop - l.start
        return VarLenTensor(feats=feats, layout=new_layout)
    feats = mx.concatenate([x.feats for x in inputs], axis=dim)
    return inputs[0].replace(feats)


def varlen_unbind(input: VarLenTensor, dim: int) -> List[VarLenTensor]:
    if dim == 0:
        return [input[i] for i in range(len(input))]
    pieces = mx.split(input.feats, input.feats.shape[dim], axis=dim)
    return [input.replace(mx.squeeze(p, axis=dim)) for p in pieces]


# ---------------------------------------------------------------------------
# SparseTensor: VarLenTensor + 3D coords (b, x, y, z)
# ---------------------------------------------------------------------------


class SparseTensor(VarLenTensor):
    """
    Sparse 3D tensor.

    Args:
        feats: [Ntotal, C] features
        coords: [Ntotal, 4] int32 coordinates (b, x, y, z). Rows of the same batch
                must be contiguous.
        shape: optional torch.Size-equivalent (batch, *feat_shape)
    """

    def __init__(
        self,
        feats: Optional[mx.array] = None,
        coords: Optional[mx.array] = None,
        shape: Optional[Tuple[int, ...]] = None,
        scale: Tuple[Fraction, Fraction, Fraction] = (Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)),
        spatial_cache: Optional[Dict[str, Any]] = None,
    ):
        assert feats is not None and coords is not None, "SparseTensor needs both feats and coords"
        if coords.dtype != mx.int32:
            coords = coords.astype(mx.int32)
        self._data = {"feats": feats, "coords": coords}
        self._shape = shape
        self._scale = scale
        self._spatial_cache: Dict[str, Dict[str, Any]] = spatial_cache if spatial_cache is not None else {}
        self._cache: Dict[str, Any] = {}

    # ---- accessors that mimic the reference ----
    @property
    def feats(self) -> mx.array:
        return self._data["feats"]

    @feats.setter
    def feats(self, v: mx.array):
        self._data["feats"] = v

    @property
    def coords(self) -> mx.array:
        return self._data["coords"]

    @coords.setter
    def coords(self, v: mx.array):
        self._data["coords"] = v

    @property
    def dtype(self):
        return self.feats.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        if self._shape is None:
            batch = int(mx.max(self.coords[:, 0]).item()) + 1 if self.coords.shape[0] > 0 else 0
            self._shape = (batch,) + tuple(self.feats.shape[1:])
        return self._shape

    @property
    def layout(self) -> List[slice]:
        layout = self.get_spatial_cache("layout")
        if layout is None:
            layout = self._calc_layout()
            self.register_spatial_cache("layout", layout)
        return layout

    def _calc_layout(self) -> List[slice]:
        B = self.shape[0]
        b = self.coords[:, 0]
        # bincount equivalent in MLX: use scatter-add into a zero array of length B.
        counts_np = np.bincount(np.asarray(b, dtype=np.int64), minlength=B)
        layout, off = [], 0
        for c in counts_np:
            layout.append(slice(int(off), int(off + c)))
            off += int(c)
        return layout

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        s = self.get_spatial_cache("shape")
        if s is None:
            mx_max = mx.max(self.coords[:, 1:], axis=0)
            s = tuple(int(v) + 1 for v in mx_max.tolist())
            self.register_spatial_cache("shape", s)
        return s

    # ---- API needed by other modules ----
    @staticmethod
    def from_tensor_list(feats_list: List[mx.array], coords_list: List[mx.array]) -> "SparseTensor":
        feats = mx.concatenate(feats_list, axis=0)
        coords = []
        for i, c in enumerate(coords_list):
            b = mx.full((c.shape[0], 1), i, dtype=mx.int32)
            spatial = c if c.shape[1] == 3 else c[:, 1:]
            coords.append(mx.concatenate([b, spatial.astype(mx.int32)], axis=1))
        coords = mx.concatenate(coords, axis=0)
        return SparseTensor(feats, coords)

    def to_tensor_list(self) -> Tuple[List[mx.array], List[mx.array]]:
        return [self.feats[s] for s in self.layout], [self.coords[s] for s in self.layout]

    def astype(self, dtype) -> "SparseTensor":
        return self.replace(self.feats.astype(dtype))

    def replace(self, feats: mx.array, coords: Optional[mx.array] = None) -> "SparseTensor":
        new_coords = self.coords if coords is None else coords
        new_shape = (
            (self._shape[0],) + tuple(feats.shape[1:]) if self._shape is not None else None
        )
        out = SparseTensor(
            feats=feats,
            coords=new_coords,
            shape=new_shape,
            scale=self._scale,
            spatial_cache=self._spatial_cache,
        )
        return out

    def to_dense(self) -> mx.array:
        B = self.shape[0]
        C = self.feats.shape[1] if self.feats.ndim > 1 else 1
        D, H, W = self.spatial_shape
        out = mx.zeros((B, D, H, W, C) if self.feats.ndim > 1 else (B, D, H, W), dtype=self.feats.dtype)
        b = self.coords[:, 0]
        x = self.coords[:, 1]
        y = self.coords[:, 2]
        z = self.coords[:, 3]
        if self.feats.ndim > 1:
            out[b, x, y, z] = self.feats
        else:
            out[b, x, y, z] = self.feats
        return out

    # ---- spatial cache (scoped by current scale) ----
    def clear_spatial_cache(self) -> None:
        self._spatial_cache = {}

    def register_spatial_cache(self, key: str, value: Any) -> None:
        scale_key = str(self._scale)
        if scale_key not in self._spatial_cache:
            self._spatial_cache[scale_key] = {}
        self._spatial_cache[scale_key][key] = value

    def get_spatial_cache(self, key: Optional[str] = None):
        scale_key = str(self._scale)
        cur = self._spatial_cache.get(scale_key, {})
        if key is None:
            return cur
        return cur.get(key, None)

    # ---- elementwise: merge spatial caches when both sides are sparse ----
    def _merge_cache(self, other: "SparseTensor") -> Dict[str, Dict[str, Any]]:
        new_cache: Dict[str, Dict[str, Any]] = {}
        for k in set(list(self._spatial_cache.keys()) + list(other._spatial_cache.keys())):
            if k in self._spatial_cache:
                new_cache[k] = dict(self._spatial_cache[k])
            if k in other._spatial_cache:
                new_cache.setdefault(k, {}).update(other._spatial_cache[k])
        return new_cache

    def _elemwise(self, other, op):
        if isinstance(other, SparseTensor):
            out = self.replace(op(self.feats, other.feats))
            out._spatial_cache = self._merge_cache(other)
            return out
        if isinstance(other, VarLenTensor):
            return self.replace(op(self.feats, other.feats))
        if isinstance(other, mx.array) and other.ndim == self.feats.ndim:
            try:
                if other.shape[0] == self.shape[0]:
                    bmap = self.batch_broadcast_map
                    return self.replace(op(self.feats, other[bmap]))
            except Exception:
                pass
        return self.replace(op(self.feats, other))

    def __add__(self, other):
        return self._elemwise(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._elemwise(other, lambda a, b: b + a)

    def __sub__(self, other):
        return self._elemwise(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._elemwise(other, lambda a, b: b - a)

    def __mul__(self, other):
        return self._elemwise(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self._elemwise(other, lambda a, b: b * a)

    def __truediv__(self, other):
        return self._elemwise(other, lambda a, b: a / b)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = list(range(*idx.indices(self.shape[0])))
        feats_parts, coords_parts, new_layout, start = [], [], [], 0
        for new_i, old_i in enumerate(idx):
            sl = self.layout[old_i]
            c = self.coords[sl]
            c = mx.concatenate([mx.full((c.shape[0], 1), new_i, dtype=mx.int32), c[:, 1:]], axis=1)
            f = self.feats[sl]
            coords_parts.append(c)
            feats_parts.append(f)
            new_layout.append(slice(start, start + f.shape[0]))
            start += f.shape[0]
        if not feats_parts:
            return SparseTensor(
                mx.zeros((0,) + self.feats.shape[1:], dtype=self.feats.dtype),
                mx.zeros((0, 4), dtype=mx.int32),
                shape=(len(idx),) + tuple(self.feats.shape[1:]),
            )
        new_feats = mx.concatenate(feats_parts, axis=0)
        new_coords = mx.concatenate(coords_parts, axis=0)
        out = SparseTensor(new_feats, new_coords, shape=(len(idx),) + tuple(self.feats.shape[1:]))
        out.register_spatial_cache("layout", new_layout)
        return out

    def __repr__(self) -> str:
        return (
            f"SparseTensor(shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"spatial={self.spatial_shape if self.coords.shape[0] > 0 else None})"
        )


def sparse_cat(inputs: List[SparseTensor], dim: int = 0) -> SparseTensor:
    if dim == 0:
        coords, feats, start = [], [], 0
        for x in inputs:
            c = mx.array(x.coords)
            c = mx.concatenate([c[:, :1] + start, c[:, 1:]], axis=1)
            coords.append(c)
            feats.append(x.feats)
            start += x.shape[0]
        return SparseTensor(mx.concatenate(feats, axis=0), mx.concatenate(coords, axis=0))
    feats = mx.concatenate([x.feats for x in inputs], axis=dim)
    return inputs[0].replace(feats)


def sparse_unbind(input: SparseTensor, dim: int) -> List[SparseTensor]:
    if dim == 0:
        return [input[i] for i in range(input.shape[0])]
    pieces = mx.split(input.feats, input.feats.shape[dim], axis=dim)
    return [input.replace(mx.squeeze(p, axis=dim)) for p in pieces]

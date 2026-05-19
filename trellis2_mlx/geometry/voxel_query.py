"""Trilinear lookup of per-voxel attributes at arbitrary 3D points (CPU)."""
from __future__ import annotations

import numpy as np
import mlx.core as mx


def query_vertex_attrs(mesh, attr_slice=slice(0, 3)) -> np.ndarray:
    """
    Sample mesh.attrs at mesh.vertices using trilinear interpolation, returning
    attr[:, attr_slice].

    `mesh` must be a MeshWithVoxel-like object with: vertices, coords, attrs,
    voxel_size, origin (origin defaults to [-0.5, -0.5, -0.5] if absent).
    """
    if isinstance(mesh.vertices, mx.array):
        v = np.asarray(mesh.vertices)
    else:
        v = np.asarray(mesh.vertices)
    coords = np.asarray(mesh.coords) if not isinstance(mesh.coords, np.ndarray) else mesh.coords
    if isinstance(mesh.attrs, mx.array):
        attrs = np.asarray(mesh.attrs)
    else:
        attrs = np.asarray(mesh.attrs)

    voxel_size = float(mesh.voxel_size)
    origin = np.array(getattr(mesh, "origin_np", [-0.5, -0.5, -0.5]), dtype=np.float32)

    grid = (v.astype(np.float32) - origin[None, :]) / voxel_size
    D = H = W = int(round(1.0 / voxel_size))

    # Build coord -> idx
    keys = (
        coords[:, 0].astype(np.int64) * (H + 1) * (W + 1)
        + coords[:, 1].astype(np.int64) * (W + 1)
        + coords[:, 2].astype(np.int64)
    )
    table = {int(k): i for i, k in enumerate(keys)}

    C = attrs.shape[1] if attrs.ndim == 2 else 1
    sl = attr_slice
    out_c = (attrs[:, sl]).shape[1]
    out = np.zeros((v.shape[0], out_c), dtype=np.float32)
    a_view = attrs[:, sl]
    for i in range(v.shape[0]):
        x, y, z = grid[i]
        x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
        dx, dy, dz = x - x0, y - y0, z - z0
        acc = np.zeros((out_c,), dtype=np.float32)
        tot = 0.0
        for ix in (0, 1):
            for iy in (0, 1):
                for iz in (0, 1):
                    wx = (1 - dx) if ix == 0 else dx
                    wy = (1 - dy) if iy == 0 else dy
                    wz = (1 - dz) if iz == 0 else dz
                    w = wx * wy * wz
                    if w <= 0:
                        continue
                    cx, cy, cz = x0 + ix, y0 + iy, z0 + iz
                    if not (0 <= cx < D and 0 <= cy < H and 0 <= cz < W):
                        continue
                    key = int(cx) * (H + 1) * (W + 1) + int(cy) * (W + 1) + int(cz)
                    j = table.get(key, -1)
                    if j < 0:
                        continue
                    acc += w * a_view[j]
                    tot += w
        if tot > 0:
            out[i] = acc
    return out

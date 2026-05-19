"""
Pure-Python Flexible Dual Grid -> triangle mesh.

Equivalent to o_voxel.convert.flexible_dual_grid_to_mesh (inference-only path,
i.e. `train=False`). Replaces the CUDA hashmap lookup with a Python dict and the
torch ops with numpy.

Inputs (per-voxel, all aligned by voxel index):
    coords:           [N, 3] int          voxel grid coords (x, y, z)
    dual_vertices:    [N, 3] float        sub-voxel offsets in [-margin, 1+margin]
    intersected_flag: [N, 3] bool         whether the voxel's x/y/z edge is intersected
    split_weight:     [N, 1] float | None per-vertex quad-split heuristic (softplus output)
    aabb:             [[x0, y0, z0], [x1, y1, z1]] world-space bounds
    grid_size:        int or [gx, gy, gz]
    voxel_size:       float or [vx, vy, vz]  (one of grid_size/voxel_size)

Returns (vertices_np [N,3] float32, faces_np [M,3] int32).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import mlx.core as mx


# Voxel offsets for the 4 neighbors that share an edge in each axis direction.
_EDGE_NEIGHBORS = np.array([
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],   # x-axis edge
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],   # y-axis edge
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],   # z-axis edge
], dtype=np.int32)


def _as_np(a) -> np.ndarray:
    if isinstance(a, mx.array):
        return np.asarray(a)
    return np.asarray(a)


def flexible_dual_grid_to_mesh(
    coords,
    dual_vertices,
    intersected_flag,
    split_weight,
    aabb,
    voxel_size=None,
    grid_size=None,
) -> Tuple[np.ndarray, np.ndarray]:
    coords = _as_np(coords).astype(np.int32)
    dual_vertices = _as_np(dual_vertices).astype(np.float32)
    intersected_flag = _as_np(intersected_flag).astype(bool)
    if split_weight is not None:
        split_weight = _as_np(split_weight).astype(np.float32)

    aabb = np.asarray(aabb, dtype=np.float32)
    assert aabb.shape == (2, 3), f"aabb must be (2,3), got {aabb.shape}"

    if voxel_size is not None:
        if np.isscalar(voxel_size):
            voxel_size = np.array([voxel_size] * 3, dtype=np.float32)
        else:
            voxel_size = np.asarray(voxel_size, dtype=np.float32)
        grid_size = np.round((aabb[1] - aabb[0]) / voxel_size).astype(np.int32)
    else:
        assert grid_size is not None
        if np.isscalar(grid_size):
            grid_size = np.array([grid_size] * 3, dtype=np.int32)
        else:
            grid_size = np.asarray(grid_size, dtype=np.int32)
        voxel_size = (aabb[1] - aabb[0]) / grid_size

    # Build coord -> index map (hashmap replacement).
    N = coords.shape[0]
    key_mul = np.array([
        (grid_size[1] + 1) * (grid_size[2] + 1),
        (grid_size[2] + 1),
        1,
    ], dtype=np.int64)

    def _pack(arr):
        a = arr.astype(np.int64)
        return a[..., 0] * key_mul[0] + a[..., 1] * key_mul[1] + a[..., 2] * key_mul[2]

    keys = _pack(coords)
    table = {int(k): i for i, k in enumerate(keys)}

    # Mesh vertices in world coords.
    mesh_vertices = (coords.astype(np.float32) + dual_vertices) * voxel_size[None, :] + aabb[0][None, :]

    # For each axis a in {x,y,z}, for each voxel that has intersected[a]=True,
    # gather 4 neighbor voxel coords; quad indices = lookup of those.
    # Build list of quads.
    quad_indices_list = []
    for ax in range(3):
        mask = intersected_flag[:, ax]
        if not mask.any():
            continue
        active = coords[mask]  # [Ma, 3]
        # neighbors: [Ma, 4, 3]
        neigh = active[:, None, :] + _EDGE_NEIGHBORS[ax][None, :, :]
        # bounds check
        in_bounds = ((neigh >= 0) & (neigh < grid_size[None, None, :])).all(axis=-1)  # [Ma, 4]
        valid_quad = in_bounds.all(axis=-1)  # [Ma]
        if not valid_quad.any():
            continue
        neigh_v = neigh[valid_quad]  # [Mv, 4, 3]
        keys_v = _pack(neigh_v.reshape(-1, 3)).reshape(-1, 4)  # [Mv, 4]
        # Lookup in table; -1 if missing
        flat_keys = keys_v.reshape(-1).tolist()
        idx = np.fromiter((table.get(k, -1) for k in flat_keys), dtype=np.int64, count=len(flat_keys))
        idx = idx.reshape(-1, 4)
        present = (idx >= 0).all(axis=-1)
        if not present.any():
            continue
        quad_indices_list.append(idx[present].astype(np.int32))

    if not quad_indices_list:
        return mesh_vertices, np.zeros((0, 3), dtype=np.int32)

    quad_indices = np.concatenate(quad_indices_list, axis=0)  # [L, 4]

    # Split each quad into two triangles; choose the splitting diagonal.
    # split_1: (0,1,2), (0,2,3) — diagonal AC
    # split_2: (0,1,3), (3,1,2) — diagonal BD
    if split_weight is None:
        v = mesh_vertices
        # diagonal AC
        a = v[quad_indices[:, 0]]
        b = v[quad_indices[:, 1]]
        c = v[quad_indices[:, 2]]
        d = v[quad_indices[:, 3]]
        n_abc = np.cross(b - a, c - a)
        n_acd = np.cross(c - a, d - a)
        align_ac = np.abs((n_abc * n_acd).sum(axis=-1, keepdims=True))
        # diagonal BD
        n_abd = np.cross(b - a, d - a)
        n_bcd = np.cross(c - b, d - b)
        align_bd = np.abs((n_abd * n_bcd).sum(axis=-1, keepdims=True))
        use_split_1 = (align_ac > align_bd).squeeze(-1)
    else:
        w = split_weight[quad_indices, 0]  # [L, 4]
        s02 = w[:, 0] * w[:, 2]
        s13 = w[:, 1] * w[:, 3]
        use_split_1 = s02 > s13

    split_1 = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    split_2 = np.array([0, 1, 3, 3, 1, 2], dtype=np.int32)
    tris_1 = quad_indices[:, split_1].reshape(-1, 3)
    tris_2 = quad_indices[:, split_2].reshape(-1, 3)
    pick = np.broadcast_to(use_split_1[:, None, None], (use_split_1.shape[0], 2, 3)).reshape(-1, 3)
    faces = np.where(pick, tris_1, tris_2)

    return mesh_vertices, faces

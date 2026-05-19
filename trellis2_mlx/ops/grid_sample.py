"""
Sparse 3D grid sample. Used by MeshWithVoxel.query_attrs for per-vertex color lookup.

Inputs (matching the FlexGEMM API):
    feats:      [N, C]      voxel features
    coords:     [N, 4] int  (b, x, y, z) active voxel indices
    voxel_shape: (B, C, D, H, W) — only B / D / H / W are used
    grid:       [B, M, 3] world-space sample positions in voxel-index space
                (the caller already maps world-xyz -> voxel coords).

Output: [B, M, C] sampled features.

Trilinear interpolation: read the 8 corners (each = active voxel or 0).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import mlx.core as mx


def _build_index_table(coords_np: np.ndarray, spatial_shape: Tuple[int, int, int]):
    D, H, W = spatial_shape
    keys = (
        coords_np[:, 0].astype(np.int64) * (D + 1) * (H + 1) * (W + 1)
        + coords_np[:, 1].astype(np.int64) * (H + 1) * (W + 1)
        + coords_np[:, 2].astype(np.int64) * (W + 1)
        + coords_np[:, 3].astype(np.int64)
    )
    return {int(k): i for i, k in enumerate(keys)}, (D, H, W)


def grid_sample_3d_sparse(
    feats: mx.array,
    coords: mx.array,
    voxel_shape: Tuple[int, int, int, int, int],
    grid: mx.array,
    mode: str = "trilinear",
) -> mx.array:
    assert mode == "trilinear", "Only trilinear mode is implemented."
    B, _, D, H, W = voxel_shape
    coords_np = np.asarray(coords, dtype=np.int32)
    table, spatial_shape = _build_index_table(coords_np, (D, H, W))

    grid_np = np.asarray(grid)  # [B, M, 3]
    M = grid_np.shape[1]
    feats_np = np.asarray(feats)
    C = feats_np.shape[1]
    out = np.zeros((B, M, C), dtype=feats_np.dtype)

    for b in range(B):
        for m in range(M):
            x, y, z = grid_np[b, m]
            x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
            dx, dy, dz = x - x0, y - y0, z - z0
            acc = np.zeros((C,), dtype=feats_np.dtype)
            tot_w = 0.0
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
                        key = (
                            int(b) * (D + 1) * (H + 1) * (W + 1)
                            + cx * (H + 1) * (W + 1)
                            + cy * (W + 1)
                            + cz
                        )
                        j = table.get(key, -1)
                        if j < 0:
                            continue
                        acc += w * feats_np[j]
                        tot_w += w
            out[b, m] = acc
    return mx.array(out)

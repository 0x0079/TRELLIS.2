"""
FlexiDualGridVaeDecoder for MLX.

Wraps SparseUnetVaeDecoder; after decoding it emits one Mesh per batch entry by
calling the pure-Python flexible_dual_grid_to_mesh on the decoded vertex offsets.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from ...ops.sparse_tensor import SparseTensor
from ...geometry.dual_grid_to_mesh import flexible_dual_grid_to_mesh
from .sparse_unet_vae import SparseUnetVaeDecoder


class _MeshOutput:
    """
    Lightweight return type. Matches the fields of trellis2.representations.Mesh
    needed by export_mesh and downstream MeshWithVoxel construction.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = vertices  # numpy [V, 3]
        self.faces = faces  # numpy [F, 3]

    def fill_holes(self, max_hole_perimeter: float = 3e-2):
        # CuMesh substitute. We intentionally no-op: the MLX path skips topology
        # cleanup. See docs/MLX_MIGRATION.md sec. 4.7.
        return self


class FlexiDualGridVaeDecoder(SparseUnetVaeDecoder):
    def __init__(
        self,
        resolution: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List,
        voxel_margin: float = 0.5,
        use_fp16: bool = False,
        **kwargs,
    ):
        super().__init__(
            out_channels=7,
            model_channels=model_channels,
            latent_channels=latent_channels,
            num_blocks=num_blocks,
            block_type=block_type,
            up_block_type=up_block_type,
            block_args=block_args,
            use_fp16=use_fp16,
            pred_subdiv=True,
        )
        self.resolution = resolution
        self.voxel_margin = voxel_margin

    def set_resolution(self, resolution: int) -> None:
        self.resolution = resolution

    def __call__(self, x: SparseTensor, return_subs: bool = False, **kwargs):
        out = super().__call__(x, return_subs=True)
        if isinstance(out, tuple):
            h, subs = out
        else:
            h, subs = out, []

        # Apply head: 7 channels = (vertex offset [0:3], intersected mask logits [3:6], quad split [6:7])
        margin = self.voxel_margin
        feats = h.feats
        # vertices
        v_offset_logits = feats[..., 0:3]
        v_offset = (1 + 2 * margin) * mx.sigmoid(v_offset_logits.astype(mx.float32)) - margin
        # intersected (bool from sign)
        intersected = feats[..., 3:6] > 0
        # quad split lerp (softplus)
        q = nn.softplus(feats[..., 6:7].astype(mx.float32))

        # Convert per-batch.
        meshes: List[_MeshOutput] = []
        layout = h.layout
        coords_all = np.asarray(h.coords, dtype=np.int32)
        v_offset_np = np.asarray(v_offset)
        intersected_np = np.asarray(intersected)
        q_np = np.asarray(q)
        for sl in layout:
            c = coords_all[sl, 1:]
            v_off = v_offset_np[sl]
            it = intersected_np[sl]
            qq = q_np[sl]
            verts, faces = flexible_dual_grid_to_mesh(
                c, v_off, it, qq,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                grid_size=self.resolution,
            )
            meshes.append(_MeshOutput(verts, faces))

        if return_subs:
            return meshes, subs
        return meshes

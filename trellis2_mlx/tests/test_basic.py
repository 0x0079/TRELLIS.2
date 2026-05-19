"""
Self-contained sanity tests that run without any pretrained weights.

These cover the data structures and the slow-path sparse ops, comparing them
against a pure-numpy reference where applicable. Skip with pytest -k if MLX is
not installed on this machine.

Run:
    pytest trellis2_mlx/tests -v
"""
from __future__ import annotations

import numpy as np
import pytest


mx = pytest.importorskip("mlx.core")


def test_sparse_tensor_roundtrip():
    from trellis2_mlx.ops.sparse_tensor import SparseTensor
    feats = mx.array(np.random.RandomState(0).randn(7, 4).astype(np.float32))
    coords = mx.array(np.array([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 0],
        [1, 0, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 2],
    ], dtype=np.int32))
    t = SparseTensor(feats=feats, coords=coords)
    assert t.shape[0] == 2
    assert len(t.layout) == 2
    fs, cs = t.to_tensor_list()
    assert fs[0].shape[0] == 3 and fs[1].shape[0] == 4


def test_submanifold_neighbor_map_grid():
    """On a fully-connected 2x2x2 cube every voxel should see 27 neighbors,
    most of them at -1 (outside) except for the 8 active ones — but every
    pair of adjacent voxels should be reachable."""
    from trellis2_mlx.ops.sparse_conv import build_submanifold_neighbor_map
    coords_list = []
    for x in range(2):
        for y in range(2):
            for z in range(2):
                coords_list.append([0, x, y, z])
    coords = mx.array(np.array(coords_list, dtype=np.int32))
    nbr = build_submanifold_neighbor_map(coords, (2, 2, 2), (3, 3, 3), (1, 1, 1))
    nbr_np = np.asarray(nbr)
    assert nbr_np.shape == (8, 27)
    # Center voxel (1,1,1) shouldn't exist here; for voxel at origin (0,0,0)
    # the central kernel offset (0,0,0) corresponds to itself.
    self_idx = (3 // 2) * 9 + (3 // 2) * 3 + (3 // 2)  # = 13
    assert nbr_np[0, self_idx] == 0


def test_dual_grid_to_mesh_minimal():
    """Two voxels sharing an x-edge produce a single quad (= 2 triangles)."""
    from trellis2_mlx.geometry.dual_grid_to_mesh import flexible_dual_grid_to_mesh

    # 4 voxels in a yz plane so that voxel (0,0,0) has its x-axis edge shared
    # by the 4 voxels (0,0,0), (0,0,1), (0,1,1), (0,1,0).
    coords = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
    ], dtype=np.int32)
    dual = np.full((4, 3), 0.5, dtype=np.float32)
    intersected = np.zeros((4, 3), dtype=bool)
    intersected[0, 0] = True  # x-axis edge of voxel 0 intersected -> emit quad
    q = np.ones((4, 1), dtype=np.float32)
    v, f = flexible_dual_grid_to_mesh(
        coords, dual, intersected, q,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        grid_size=2,
    )
    assert v.shape == (4, 3)
    # one quad = two triangles
    assert f.shape == (2, 3)


def test_layer_norm32_matches_numpy():
    from trellis2_mlx.modules.norm import LayerNorm32
    ln = LayerNorm32(8, elementwise_affine=False, eps=1e-6)
    rng = np.random.RandomState(42)
    x_np = rng.randn(3, 8).astype(np.float32)
    x = mx.array(x_np)
    out = np.asarray(ln(x))
    mean = x_np.mean(axis=-1, keepdims=True)
    var = ((x_np - mean) ** 2).mean(axis=-1, keepdims=True)
    expected = (x_np - mean) / np.sqrt(var + 1e-6)
    assert np.allclose(out, expected, atol=1e-5)


def test_rope_phase_shape():
    from trellis2_mlx.ops.rope import RotaryPositionEmbedder
    rope = RotaryPositionEmbedder(head_dim=64, dim=3)
    coords = mx.array(np.random.RandomState(0).randint(0, 32, size=(10, 3)).astype(np.float32))
    phases = rope(coords)
    assert tuple(phases.shape) == (10, 32, 2)

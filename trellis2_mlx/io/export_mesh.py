"""
Minimal mesh export: GLB with per-vertex color, or OBJ + MTL.

These don't depend on nvdiffrast / CuMesh / O-Voxel — they take the
MeshWithVoxel produced by the MLX pipeline (vertices, faces, voxel attrs,
voxel layout) and write a viewable file.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import mlx.core as mx


def _to_np(a) -> np.ndarray:
    if isinstance(a, mx.array):
        return np.asarray(a)
    return np.asarray(a)


def write_obj_with_vertex_color(
    path: str,
    vertices: "np.ndarray | mx.array",
    faces: "np.ndarray | mx.array",
    vertex_colors: Optional["np.ndarray | mx.array"] = None,
) -> None:
    v = _to_np(vertices).astype(np.float32)
    f = _to_np(faces).astype(np.int64)
    if vertex_colors is not None:
        c = _to_np(vertex_colors).astype(np.float32)
        if c.ndim == 2 and c.shape[1] >= 3:
            c = c[:, :3]
        else:
            c = None
    else:
        c = None
    with open(path, "w") as fp:
        if c is not None:
            for i in range(v.shape[0]):
                fp.write(f"v {v[i,0]:.6f} {v[i,1]:.6f} {v[i,2]:.6f} {c[i,0]:.4f} {c[i,1]:.4f} {c[i,2]:.4f}\n")
        else:
            for i in range(v.shape[0]):
                fp.write(f"v {v[i,0]:.6f} {v[i,1]:.6f} {v[i,2]:.6f}\n")
        for i in range(f.shape[0]):
            fp.write(f"f {f[i,0]+1} {f[i,1]+1} {f[i,2]+1}\n")


def write_glb_with_vertex_color(
    path: str,
    vertices: "np.ndarray | mx.array",
    faces: "np.ndarray | mx.array",
    vertex_colors: Optional["np.ndarray | mx.array"] = None,
) -> None:
    """
    Write a glTF 2.0 GLB containing a single mesh with per-vertex color (COLOR_0).
    Falls back to no color when `vertex_colors` is None.
    """
    try:
        import trimesh
    except ImportError as e:
        raise RuntimeError(
            "write_glb_with_vertex_color requires trimesh; install with `pip install trimesh`."
        ) from e

    v = _to_np(vertices).astype(np.float32)
    f = _to_np(faces).astype(np.int64)
    if vertex_colors is not None:
        c = _to_np(vertex_colors).astype(np.float32)
        c = np.clip(c[:, :3], 0.0, 1.0)
        c = (c * 255).astype(np.uint8)
        c = np.concatenate([c, np.full((c.shape[0], 1), 255, dtype=np.uint8)], axis=1)
        mesh = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=c, process=False)
    else:
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    mesh.export(path)


def export_mesh_with_voxel(mesh, path: str) -> None:
    """
    Convenience: write a MeshWithVoxel (output of the MLX pipeline) to GLB with
    per-vertex base_color queried via trilinear interpolation from its voxel grid.
    """
    from ..geometry.voxel_query import query_vertex_attrs
    base_color = query_vertex_attrs(mesh, attr_slice=mesh.layout.get("base_color", slice(0, 3)))
    if path.lower().endswith(".obj"):
        write_obj_with_vertex_color(path, mesh.vertices, mesh.faces, base_color)
    else:
        write_glb_with_vertex_color(path, mesh.vertices, mesh.faces, base_color)

"""
TRELLIS.2 Apple MLX (experimental) image -> 3D demo.

Usage:
    python example_mlx.py path/to/image.png [out.glb]

Notes:
- Currently runs the `pipeline_type='512'` path on a single image. Cascade and
  1024 paths will follow as the sparse ops are optimized.
- The image must be PNG with an alpha channel (background already removed) —
  the MLX path does not yet wrap rembg.
- Output is a GLB containing the raw mesh + per-vertex base_color queried via
  trilinear interpolation from the predicted voxel attributes. UV unwrap /
  texture baking / mesh simplification are NOT performed (see
  docs/MLX_MIGRATION.md).
"""
from __future__ import annotations

import sys
from PIL import Image

from trellis2_mlx.pipelines import Trellis2ImageTo3DPipelineMLX
from trellis2_mlx.io.export_mesh import export_mesh_with_voxel


def main():
    if len(sys.argv) < 2:
        print("usage: python example_mlx.py <input.png> [out.glb]")
        sys.exit(2)
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "sample_mlx.glb"

    image = Image.open(in_path)
    if image.mode != "RGBA":
        raise SystemExit(
            "Input image must be RGBA with the background already removed. "
            "Run `rembg` or equivalent on the CUDA path first; MLX rembg is TBD."
        )

    pipe = Trellis2ImageTo3DPipelineMLX.from_pretrained("microsoft/TRELLIS.2-4B")
    meshes = pipe.run(image, pipeline_type="512")
    export_mesh_with_voxel(meshes[0], out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

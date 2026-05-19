# TRELLIS.2 · Apple MLX backend (inference)

Experimental Apple Silicon inference path for [TRELLIS.2](https://huggingface.co/microsoft/TRELLIS.2-4B).

This package (`trellis2_mlx/`) is **separate** from the CUDA reference under
`trellis2/`. It loads the same HuggingFace checkpoints
(`microsoft/TRELLIS.2-4B`) but re-implements the model graph on top of
[Apple MLX](https://github.com/ml-explore/mlx) so it can run on Mac
M-series GPUs.

> ⚠️ Experimental. The MLX path is faithful to the reference architecture but
> currently supports a **subset** of the pipeline. See
> [§ How does this compare to the original?](#how-does-this-compare-to-the-original)
> and [`docs/MLX_MIGRATION.md`](../docs/MLX_MIGRATION.md) for the full design.

---

## Installation

### 1. Create an environment

```bash
# (recommended) fresh conda env
bash setup_mlx.sh --new-env
conda activate trellis2_mlx

# install MLX + dependencies
bash setup_mlx.sh
```

Or manually:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-mlx.txt
```

The full requirements file pulls:

| Group | Packages | Why |
|---|---|---|
| Runtime | `mlx`, `numpy`, `tqdm` | MLX tensors + sampler progress bar |
| Weights | `safetensors`, `huggingface_hub` | Load `microsoft/TRELLIS.2-4B` |
| Mesh I/O | `Pillow`, `trimesh` | Read PNG, write GLB with per-vertex color |
| Image cond | `torch`, `torchvision`, `transformers` | DINOv3 currently runs on PyTorch + MPS (see [Design doc §4.6](../docs/MLX_MIGRATION.md#46-dinov3-图像特征提取m2-后期)) |
| Dev | `pytest` | Unit tests |

If you want a leaner install without the PyTorch image-conditioning piece
(e.g. you want to feed conditioning features in yourself), use:

```bash
bash setup_mlx.sh --no-torch
```

### 2. Smoke-test the install

```bash
pytest trellis2_mlx/tests -v
```

This runs five self-contained tests (no checkpoint download needed) that
exercise the sparse-tensor data structure, the submanifold convolution
neighbor map, dual-grid → mesh extraction, LayerNorm32, and RoPE.

---

## Running inference

### Minimal example

```bash
# input image must be RGBA with the subject already cut out
python example_mlx.py path/to/image.png out.glb
```

`example_mlx.py` is a 30-line driver. The same flow in Python:

```python
from PIL import Image
from trellis2_mlx.pipelines import Trellis2ImageTo3DPipelineMLX
from trellis2_mlx.io.export_mesh import export_mesh_with_voxel

pipe = Trellis2ImageTo3DPipelineMLX.from_pretrained("microsoft/TRELLIS.2-4B")
image = Image.open("assets/example_image/T.png")  # RGBA
meshes = pipe.run(image, pipeline_type="512", seed=42)
export_mesh_with_voxel(meshes[0], "out.glb")
```

### What `pipe.run()` does

1. **DINOv3 conditioning** — runs the image through HuggingFace
   `DINOv3ViTModel` on PyTorch's MPS backend, returns the patch tokens.
2. **Stage 1 — Sparse Structure** — a 32³ dense Transformer (DiT) runs
   50 Euler steps; the dense decoder converts the latent into an occupancy
   volume and `argwhere(>0)` produces sparse voxel coordinates.
3. **Stage 2 — Shape SLat** — the sparse DiT runs 50 Euler steps over those
   voxels; the Flexible Dual Grid VAE decoder turns the result into a
   triangle mesh (vertex offsets + intersected mask + quad-split lerp).
4. **Stage 3 — Texture SLat** — second sparse DiT conditioned on shape
   produces per-voxel PBR (base_color / metallic / roughness / alpha).
5. **Export** — vertex colors are sampled from the PBR voxel grid via
   trilinear interpolation and written to GLB.

Output object (`MeshWithVoxel`):
- `.vertices` — numpy `[V, 3]` float32
- `.faces` — numpy `[F, 3]` int32
- `.coords` — `[N, 3]` int32 active-voxel indices
- `.attrs` — `[N, 6]` float (base_color [0:3], metallic [3:4], roughness [4:5], alpha [5:6])
- `.voxel_size`, `.layout`, `.origin_np`

### Choosing the input image

The official model expects a 512×512 RGB image with the subject on a
clean background. The reference pipeline (`trellis2/`) includes a `rembg`
step using BiRefNet; the MLX path does **not** wrap that yet, so:

- pass an RGBA PNG with alpha as the cutout mask, **or**
- run `rembg` separately on macOS:
  ```bash
  pip install rembg
  rembg i input.jpg input.png
  python example_mlx.py input.png out.glb
  ```

### Configuring the sampler

The sampler parameters come from `pipeline.json` in the HF checkpoint
(guidance strength, guidance interval, number of Euler steps, etc.). You
can override per-call:

```python
meshes = pipe.run(
    image,
    seed=42,
    sparse_structure_sampler_params={"steps": 25, "guidance_strength": 3.0},
    shape_slat_sampler_params={"steps": 25},
    tex_slat_sampler_params={"steps": 25},
)
```

Reducing `steps` is the simplest way to trade quality for runtime while
you're tuning.

---

## How does this compare to the original?

The MLX path **reuses the same checkpoints** so the model graph is
arithmetically identical. The differences below are about *which parts of
the pipeline we run* and *which postprocessing we skip*.

### Identical to upstream

- Model architectures (every `nn.Module` is a 1:1 translation of
  `trellis2/models/...`).
- Pretrained weights from `microsoft/TRELLIS.2-4B`. We load the same
  `*.safetensors` and only permute Conv3d weights to channel-last and
  flatten `nn.Sequential` attribute names — no parameter remapping.
- Sampler math: `FlowEulerGuidanceIntervalSampler` with the same
  guidance interval / rescale defaults from `pipeline.json`.
- DINOv3 conditioning (we call the official HF `DINOv3ViTModel` directly).
- Output PBR attribute layout: `base_color` / `metallic` / `roughness` / `alpha`.

### Differs from upstream

| Aspect | Original (`trellis2/`) | This MLX path | Why |
|---|---|---|---|
| Resolution | `512` / `1024` / `1024_cascade` / `1536_cascade` | **`512` only** | Cascade requires `shape_slat_decoder.upsample()` validation; planned for a follow-up. |
| Background removal | Built-in (BiRefNet) | **Not wrapped** — input must be RGBA cutout | rembg ships ONNX models; we leave it to the user for now. |
| Sparse Conv3D | `flex_gemm` Triton CUDA kernel | Pure-Python gather + einsum (≈30–100× slower) | No MLX equivalent. See [Design doc §4.3.2](../docs/MLX_MIGRATION.md). |
| Sparse attention | `flash_attn` varlen | Padded varlen SDPA via MLX `fast.sdpa` | Functionally equivalent but pads to `max_seqlen`. |
| Mesh extraction | `o_voxel` CUDA + hashmap | Pure-Python dual contouring (`geometry/dual_grid_to_mesh.py`) | Numerically identical algorithm. Slower at 1M+ voxels. |
| Mesh simplify / fill-holes | `CuMesh` CUDA | **Skipped** | Output mesh keeps every dual vertex (~1M tris at 512³). |
| Output format | `o_voxel.postprocess.to_glb` (UV unwrap + PBR bake via nvdiffrast/nvdiffrec) | **Per-vertex color GLB** | nvdiffrast is NVIDIA-only. We bake `base_color` to vertex colors. |
| Video preview | `make_pbr_vis_frames` (nvdiffrec PBR) | **Not provided** | Render the GLB in Blender / threejs / Open3D for now. |
| Texturing pipeline | `Trellis2TexturingPipeline` (mesh → PBR) | **Not provided** | Depends more heavily on nvdiffrast + CuMesh; planned for a later milestone. |
| Low-VRAM mode | `low_vram=True` rotates models on/off GPU | **N/A** | MLX uses unified memory. The whole model graph stays resident. |

### How to verify the math matches

This is the right question to ask. Run both backends on the **same seed
and same image**, compare intermediate outputs:

```python
# CUDA reference (on a Linux box)
import torch, numpy as np
from trellis2.pipelines import Trellis2ImageTo3DPipeline
pipe_t = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B").cuda()
torch.manual_seed(42)
ref = pipe_t.run(image, num_samples=1, seed=42, pipeline_type="512", return_latent=True)
np.savez("ref_latents.npz",
         shape_slat_feats=ref[1][0].feats.cpu().numpy(),
         shape_slat_coords=ref[1][0].coords.cpu().numpy(),
         tex_slat_feats=ref[1][1].feats.cpu().numpy(),
         vertices=ref[0][0].vertices.cpu().numpy(),
         faces=ref[0][0].faces.cpu().numpy())
```

```python
# MLX (on Mac)
import numpy as np
from trellis2_mlx.pipelines import Trellis2ImageTo3DPipelineMLX
pipe_m = Trellis2ImageTo3DPipelineMLX.from_pretrained("microsoft/TRELLIS.2-4B")
out = pipe_m.run(image, seed=42, pipeline_type="512")
ref = np.load("ref_latents.npz")
# Acceptable tolerances per Design doc §7:
#  - fp32 forward      : max|diff| < 1e-4
#  - bf16/fp16 forward : max|diff| < 5e-2
#  - 50-step Euler     : max|diff| < 5e-2 (accumulated error)
#  - vertex Chamfer    : < voxel_size
```

If any of these tolerances are exceeded, the most likely causes (in
order of frequency) are: (1) Conv3d weight permutation mismatch for a
specific block — check `io/weight_convert.py:_is_conv3d_weight`; (2) a
sub-module's parameter-name mapping needing an extra entry in
`_maybe_remap_sequential`; (3) attention-mask handling for short
sequences inside `ops/sparse_attention.py`.

---

## Performance expectations

Estimated on M-series (rough, not measured yet — please report actual numbers):

| Stage | CUDA (H100) | MLX (M-Pro) | Comment |
|---|---|---|---|
| DINOv3 image cond | < 50 ms | ~100 ms (MPS) | PyTorch MPS path |
| Stage 1 Flow (32³, 50 steps) | ~0.5 s | ~5 s | Dense Transformer — fast on MLX |
| Stage 2/3 Sparse Flow | ~1–2 s | ~30–120 s | Bottleneck: sparse attention padding |
| Sparse VAE decode (512³) | ~0.5 s | ~30–90 s | Bottleneck: sparse Conv3D slow path |
| Dual grid → mesh | < 100 ms (CUDA) | ~5–30 s (CPU) | Pure Python, ~1M voxels |
| **End-to-end @ 512** | **~3 s** | **~2–5 min** | First call also pays HF download + JIT |

Subsequent calls reuse the JIT-compiled MLX graph and the neighbor-map
caches on `SparseTensor`, so they're noticeably faster than the cold start.

---

## Troubleshooting

### `RuntimeError: DinoV3FeatureExtractorMLX requires torch + transformers`
Install the optional deps: `pip install torch torchvision transformers`,
or skip image cond and feed your own features (advanced).

### `mlx.core has no attribute X` / API errors
MLX is still evolving fast. Pin to the version that works for you and
report the version with the failure. We target MLX ≥ 0.20.

### Weight load complains about missing keys
Run with verbose logging:

```python
import logging
logging.getLogger("trellis2_mlx").setLevel(logging.DEBUG)
```

Then check which key from the safetensors didn't match an MLX module
attribute. Most cases need a one-line addition to
`trellis2_mlx/io/weight_convert.py:_maybe_remap_sequential`.

### `pipeline_type='1024_cascade'` errors out
That's expected — the MLX path currently supports `'512'` only. Cascade
adds the `shape_slat_decoder.upsample()` path and the 1024 flow model,
which are scaffolded but not yet validated end-to-end.

### Diagnosing weight-load failures

The pipeline prints a side-by-side diff before each load if the MLX module
parameter tree doesn't match the checkpoint exactly:

```
[trellis2_mlx] checkpoint <-> model key mismatch for 'SparseStructureDecoder':
=== 4 key(s) in checkpoint but NOT in MLX model ===
  + out_layer.0.bias
  + out_layer.0.weight
  + out_layer.2.bias
  + out_layer.2.weight
=== 4 key(s) in MLX model but NOT in checkpoint ===
  - out_norm.bias
  - out_norm.weight
  - out_conv.bias
  - out_conv.weight
```

To inspect a checkpoint yourself (useful when reporting a mismatch):

```bash
python -m trellis2_mlx.io.weight_convert --inspect microsoft/TRELLIS.2-4B/ckpts/<model_name>
# or for a cross-repo reference:
python -m trellis2_mlx.io.weight_convert --inspect microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16
# config only:
python -m trellis2_mlx.io.weight_convert --inspect <prefix> --config-only
# first 30 keys only:
python -m trellis2_mlx.io.weight_convert --inspect <prefix> --head 30
```

Output is `(key_name, shape, dtype)` per tensor plus a `(dense conv3d, will permute)`
marker on weights the loader will reshape.

### Output mesh is broken / has holes
The MLX path **does not** call `Mesh.fill_holes()` (which needs CuMesh).
Open-surface artifacts that the reference pipeline fixes post-hoc will
remain in MLX output. Run the result through `trimesh.repair` or your
DCC of choice if needed.

---

## Where things live

```
trellis2_mlx/
├── ops/             # SparseTensor, sparse conv/attn slow paths, RoPE
├── modules/         # Transformer / U-Net building blocks
├── models/          # SparseStructureFlow, SLatFlow, SparseUnetVAE, FdgVAE
├── pipelines/       # FlowEuler sampler + Trellis2ImageTo3DPipelineMLX
├── extractors/      # DINOv3 (PyTorch + MPS adapter)
├── geometry/        # Pure-Python dual-grid → mesh, trilinear voxel query
├── io/              # weight_convert, export_mesh
└── tests/           # pytest smoke tests
```

For the architectural rationale (why slow path, what's blocked, the
M1–M6 milestone plan), read [`docs/MLX_MIGRATION.md`](../docs/MLX_MIGRATION.md).

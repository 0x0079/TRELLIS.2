# MLX Migration — Status & Issue Tracker

> Living document. Update this on every meaningful change.
> Companion to the architecture/design doc: [`docs/MLX_MIGRATION.md`](../docs/MLX_MIGRATION.md).
> Branch: `claude/apple-mlx-migration-BwgCZ`

**Last updated:** 2026-05-21
**Scope:** inference-only (image → 3D mesh + per-vertex PBR), `pipeline_type='512'` first.
**Backend strategy:** pure-Python/MLX slow paths for CUDA-only ops (FlexGEMM, O-Voxel, nvdiffrast, CuMesh).

---

## 1. Overall status

| Phase | What | Code | On-device validated |
|---|---|:---:|:---:|
| **M1** | Weight loading (safetensors → mx.array), config parsing, diagnostics | ✅ | 🟡 partial (loading SS flow + SS decoder works after fixes) |
| **M2** | Dense Transformer, Flow Euler sampler, SS dense decoder, DINOv3 cond | ✅ | ⬜ not yet |
| **M3** | SparseTensor, submanifold Conv3D, sparse/windowed attention, pool ops | ✅ | ⬜ not yet |
| **M4** | Sparse U-Net VAE decoder, Flexible Dual Grid → mesh | ✅ | ⬜ not yet |
| **M5** | Minimal mesh export (GLB / OBJ with per-vertex color) | ✅ | ⬜ not yet |
| **M6** | End-to-end pipeline + `example_mlx.py` | ✅ | 🟡 weight load being debugged on Mac |

Legend: ✅ done · 🟡 in progress · ⬜ not started · ❌ blocked

**Current focus:** getting the full checkpoint set to load cleanly on the user's
Mac (M6 cold start). Each model loads through a pre-flight key-diff so mismatches
surface precisely.

---

## 2. Component status (detail)

### Models (`trellis2_mlx/models/`)

| Module | File | Status | Notes |
|---|---|:---:|---|
| `SparseStructureFlowModel` | `sparse_structure_flow.py` | ✅ code | dense DiT; loads `ss_flow_img_dit_*` |
| `TimestepEmbedder` | `sparse_structure_flow.py` | ✅ | list-attr MLP (`mlp.0/2`) |
| `SparseStructureEncoder/Decoder` | `sparse_structure_vae.py` | ✅ code | dense 3D ConvNet; channel-last; loads cross-repo `ss_dec_conv3d_*` |
| `SLatFlowModel` | `structured_latent_flow.py` | ✅ code | sparse DiT; shape + texture stages |
| `SparseUnetVaeEncoder/Decoder` | `sc_vaes/sparse_unet_vae.py` | ✅ code | nested list blocks |
| `FlexiDualGridVaeDecoder` | `sc_vaes/fdg_vae.py` | ✅ code | head split + dual-grid extraction |

### Ops (`trellis2_mlx/ops/`)

| Op | File | Status | Notes |
|---|---|:---:|---|
| `SparseTensor` / `VarLenTensor` | `sparse_tensor.py` | ✅ | scale-scoped spatial cache |
| Submanifold `SparseConv3d` | `sparse_conv.py` | ✅ | gather+einsum slow path; neighbor map cached |
| Pool (Down/Up/S2C/C2S) | `sparse_pool.py` | ✅ | numpy unique/scatter assists |
| Sparse / windowed attention | `sparse_attention.py` | ✅ | padded varlen SDPA |
| Dense SDPA | `attention.py` | ✅ | uses `mx.fast.sdpa` when available |
| RoPE / APE | `rope.py` | ✅ | `_freqs` private (not a param) |
| Sparse `grid_sample_3d` | `grid_sample.py` | ✅ | CPU trilinear |

### Pipeline / IO / geometry

| Piece | File | Status | Notes |
|---|---|:---:|---|
| Flow Euler + CFG + interval | `pipelines/samplers.py` | ✅ code | |
| Image→3D pipeline (512) | `pipelines/image_to_3d.py` | ✅ code | pre-flight key diff, cross-repo resolve |
| DINOv3 cond (torch+MPS) | `extractors/dinov3.py` | ✅ code | path A (PyTorch MPS), not yet run |
| Dual grid → mesh | `geometry/dual_grid_to_mesh.py` | ✅ | pure Python, has unit test |
| Voxel trilinear query | `geometry/voxel_query.py` | ✅ | per-vertex color |
| Weight convert + CLI | `io/weight_convert.py` | ✅ | shape-based Conv3d detection |
| Minimal GLB/OBJ export | `io/export_mesh.py` | ✅ | trimesh vertex colors |

---

## 3. Resolved issues (changelog)

| # | Symptom | Root cause | Fix | Commit |
|---|---|---|---|---|
| R1 | `Received N parameters not in model: out_layer.0/2...` | Reference `nn.Sequential` produces keys `x.0/x.2`; MLX side split them into named attrs (`out_norm`/`out_conv`, `mlp_0/2`) | Store all Sequential equivalents as **Python list attributes**; MLX emits `attr.0/.1/.2` matching the checkpoint | `a10a41a` |
| R2 | dense Conv3d weights mis-detected (e.g. 1×1 `skip_connection`) | name-based heuristic missed convs not named `conv*` | **shape-based** detection: kernel dims equal & ≤7 → permute | `a10a41a` |
| R3 | 404 noise loading `sparse_structure_decoder` | cross-repo ref (`microsoft/TRELLIS-image-large/ckpts/...`) probed as a sub-path first | `_resolve_and_load` retries on `RemoteEntryNotFound` quietly | `a10a41a` |
| R4 | `Missing 1 parameters: rope_phases` | `self.x = None` placeholder is tracked as a param-tree leaf by MLX | remove all `self.x=None`; use `getattr` at use sites | `85c98a6` |
| R5 | (preempted) `pos_embedder.freqs` would be missing | `freqs` was a public `mx.array` attr; reference keeps it as a plain (unsaved) tensor | rename to `self._freqs` (private → untracked) | `85c98a6` |
| R6 | `Module does not have parameter named X` on `out_layer` | LayerNorm32 set `weight/bias=None` when `affine=False` | omit the attributes entirely when non-affine | `a10a41a` |

---

## 4. Open issues / TODO

### Blocking M6 (end-to-end on 512)

- [ ] **Finish full checkpoint load.** SS flow + SS decoder confirmed loading;
      still need to confirm `shape_slat_flow_model_512`, `shape_slat_decoder`,
      `tex_slat_flow_model_512`, `tex_slat_decoder` load with zero key mismatch.
      *Action:* run, paste any key-diff.
- [ ] **First forward pass.** Verify the SS flow forward + Euler sampling runs
      without MLX API errors (fancy indexing, `mx.fast.sdpa` availability,
      `mx.split`, scatter assignment in `sparse_pool`).
- [ ] **Sparse path execution.** First real run of submanifold conv + sparse
      attention on non-trivial voxel counts — watch for OOM and slow steps.
- [ ] **Dual-grid output sanity.** Confirm `FlexiDualGridVaeDecoder` emits a
      non-degenerate mesh (vertex count, face count > 0; no inverted normals).

### Correctness validation (Design doc §7)

- [ ] Capture CUDA reference tensors (`ref_latents.npz`) on a Linux+GPU box.
- [ ] M1 test: fp32 weight round-trip (max|diff| == 0).
- [ ] M2 test: SS flow single forward vs CUDA (`< 1e-2` bf16).
- [ ] M2 test: 50-step Euler output vs CUDA (`< 5e-2`).
- [ ] M3 test: single `SparseConv3d` layer vs CUDA (`< 1e-4` fp32).
- [ ] M4 test: dual-grid vertices vs CUDA (Chamfer `< voxel_size`).
- [ ] e2e: 512 pipeline vertex Chamfer vs CUDA.

### Known API risks to watch on first run

- [ ] `mx.array[indices] = source` fancy-index assignment used in
      `sparse_pool._scatter_assign`, `sparse_tensor.to_dense`,
      `sparse_attention._pad_groups/_unpad_groups`. Confirm the installed MLX
      version supports in-place scatter; if not, swap for `mx.scatter` / `.at[]`.
- [ ] `mx.fast.scaled_dot_product_attention` presence/signature
      (`attention.py` has a manual fallback already).
- [ ] `mx.repeat` semantics for the channel-repeat skip connections in
      `sparse_unet.SparseResBlockC2S3d._skip`.
- [ ] `nn.gelu_approx` vs reference `GELU(approximate='tanh')` numerical match.
- [ ] MLX list-attribute parameter naming actually produces `attr.0`,
      `attr.1`, … (relied on by R1 fix). If MLX uses a different scheme the
      key-diff will show it immediately.

### Deferred (post-M6)

- [ ] `pipeline_type='1024'` and `'1024_cascade'` / `'1536_cascade'`
      (needs `shape_slat_decoder.upsample()` validation + 1024 flow models).
- [ ] Background removal (rembg / BiRefNet) wrapper for the MLX path.
- [ ] `Trellis2TexturingPipeline` (mesh → PBR) — heavier nvdiffrast dependence.
- [ ] Mesh post-processing parity: simplify / fill-holes (currently skipped;
      CuMesh substitute or trimesh fallback).
- [ ] Real nvdiffrast-free preview rendering (trimesh / pyrender / Open3D).
- [ ] Performance: measure per-stage timings; optimize sparse conv neighbor-map
      build (Python dict → `mx.searchsorted` on Z-order keys), consider numba
      for dual-grid extraction.
- [ ] DINOv3 path B: native MLX ViT to drop the PyTorch dependency.
- [ ] bf16 end-to-end (currently upcasting weights to fp32 on load).

---

## 5. Verification log

> Record each on-device run here: date, command, result, next action.

| Date | Env | Command | Result | Next |
|---|---|---|---|---|
| 2026-05-21 | M-series (sd env) | `example_mlx.py` | `out_layer.0/2` key mismatch | → R1 fix |
| 2026-05-21 | M-series | `example_mlx.py` | `Missing rope_phases` | → R4 fix |
| _next_ | | `example_mlx.py` | _pending_ | paste key-diff or first runtime error |

---

## 6. How to report a failure (so it's quick to fix)

1. **Key mismatch** → paste the `[trellis2_mlx] checkpoint <-> model key mismatch`
   block (it shows both directions of the diff).
2. **Checkpoint structure** → `python -m trellis2_mlx.io.weight_convert --inspect <prefix> --head 40`
   and paste the output.
3. **Runtime/MLX API error** → paste the traceback + `python -c "import mlx; print(mlx.__version__)"`.
4. **Numerical drift** → paste the tolerance that failed and the max|diff|.

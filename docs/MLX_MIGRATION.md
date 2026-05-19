# TRELLIS.2 → Apple MLX 推理迁移设计文档

> 目标读者：实现者本人 / 后续协作者
> 状态：**Design** —— 已对齐范围，待按本文档分阶段实现
> 范围：**仅推理**（image → 3D mesh + PBR）；训练保持 CUDA。
> 取舍：对无 MLX 等价物的 CUDA 组件采用 **纯 Python / NumPy / MLX 慢路径** 实现，优先正确性，性能可接受牺牲。

---

## 0. TL;DR

- TRELLIS.2 是 4B 参数、三阶段 Flow Matching 的 image→3D 生成模型。当前实现强依赖 CUDA（flash-attn、FlexGEMM、nvdiffrast、CuMesh、自写的 O-Voxel CUDA kernels）。
- **迁移策略**：保留 PyTorch 张量代数作为「中间表示层」可用做 ground-truth 对照，**新增独立的 MLX 推理后端** `trellis2_mlx/`，加上一个 **后端切换抽象层**。先**不动现有 CUDA 代码**，避免污染训练路径。
- **核心工作量**集中在三个慢路径实现上：(1) 子流形稀疏卷积 (2) 子流形稀疏注意力 (3) 双重网格 → 三角网格的提取与导出。
- **里程碑**：
  - M1: MLX 张量与权重加载（safetensors → mlx.array）
  - M2: 稠密 Transformer + Flow Sampler（稀疏结构 flow）
  - M3: SparseTensor MLX 实现 + 子流形稀疏卷积慢路径
  - M4: Sparse U-Net VAE 解码器（形状）
  - M5: 体素属性导出（绕过 O-Voxel/CuMesh，直接写最小可用 GLB）
  - M6: 端到端 `example.py` 在 Mac 上跑通

---

## 1. 项目架构速览（已勘察）

### 1.1 端到端推理流水线（`trellis2/pipelines/trellis2_image_to_3d.py`）

```
PIL.Image
  │
  ├─► preprocess_image (rembg, 可选)
  │       └─ trellis2/pipelines/rembg/BiRefNet.py
  │
  ├─► get_cond (DINOv3 ViT 提取 image feature)
  │       └─ trellis2/modules/image_feature_extractor.py  (DinoV3FeatureExtractor)
  │
  ├─► sample_sparse_structure  ── Stage 1 ──────────────────────
  │     ├─ flow_model: SparseStructureFlowModel (稠密 Transformer 在 32³/64³ 体素上)
  │     │     └─ trellis2/models/sparse_structure_flow.py
  │     ├─ sampler: FlowEulerGuidanceIntervalSampler  (50 steps Euler)
  │     │     └─ trellis2/pipelines/samplers/flow_euler.py
  │     └─ decoder: SparseStructureDecoder (3D 稠密 ConvNet)
  │           └─ trellis2/models/sparse_structure_vae.py
  │     ▼ 输出: coords [N,4] 稀疏占据坐标
  │
  ├─► sample_shape_slat / sample_shape_slat_cascade  ── Stage 2 ───
  │     ├─ flow_model: SLatFlowModel (Transformer on SparseTensor)
  │     │     └─ trellis2/models/structured_latent_flow.py
  │     │     └─ 内部使用 sparse attention (windowed / serialized)
  │     │     └─ trellis2/modules/sparse/attention/*.py
  │     └─ 级联模式还会 upsample coords via shape_slat_decoder.upsample()
  │     ▼ 输出: SparseTensor (shape latent)
  │
  ├─► sample_tex_slat   ── Stage 3 ───────────────────────────────
  │     ├─ flow_model: SLatFlowModel (Transformer on SparseTensor)
  │     ├─ concat_cond = shape_slat (拼接到 noise)
  │     ▼ 输出: SparseTensor (texture latent)
  │
  ├─► decode_shape_slat
  │     └─ FlexiDualGridVaeDecoder (Sparse U-Net VAE)
  │           └─ trellis2/models/sc_vaes/fdg_vae.py + sparse_unet_vae.py
  │     └─ flexible_dual_grid_to_mesh (来自 o_voxel.convert)
  │     ▼ 输出: Mesh(vertices, faces), List[SparseTensor] subs (多分辨率引导)
  │
  ├─► decode_tex_slat
  │     └─ SparseUnetVaeDecoder (Sparse U-Net)
  │     ▼ 输出: SparseTensor 体素属性 (base_color/metallic/roughness/alpha)
  │
  ▼
MeshWithVoxel (顶点 + 面 + 体素属性 + voxel_size)
  │
  ├─► render_utils.render_video / make_pbr_vis_frames  (nvdiffrast + nvdiffrec)
  │     └─ trellis2/renderers/pbr_mesh_renderer.py
  │
  └─► o_voxel.postprocess.to_glb  (网格简化 + UV 展开 + 烘焙)
        └─ o-voxel/o_voxel/postprocess.py  (依赖 CuMesh + nvdiffrast)
```

### 1.2 模块清单（按 PyTorch/CUDA 依赖分级）

| 层级 | 模块 | 文件路径 | 依赖 | 迁移难度 |
|---|---|---|---|---|
| **纯计算 / 易迁移** | 时间步嵌入、APE/RoPE | `trellis2/modules/attention/rope.py`, `models/sparse_structure_flow.py:TimestepEmbedder` | torch only | ★ |
| | LayerNorm / GroupNorm / SiLU / GELU | `trellis2/modules/norm.py`, `modules/sparse/norm.py`, `modules/sparse/nonlinearity.py` | torch only | ★ |
| | TransformerBlock / ModulatedCrossBlock | `trellis2/modules/transformer/*.py` | torch + attn backend | ★★ |
| | Flow Euler Sampler + CFG + Guidance Interval | `trellis2/pipelines/samplers/*.py` | torch only | ★ |
| | DINOv3 ViT 图像特征 | `trellis2/modules/image_feature_extractor.py` | `transformers.DINOv3ViTModel` | ★★ |
| | rembg / BiRefNet | `trellis2/pipelines/rembg/BiRefNet.py` | torch + onnx | ★★ |
| | SparseStructureDecoder (稠密 3D ConvNet 在 64³) | `trellis2/models/sparse_structure_vae.py` | torch.nn.Conv3d | ★★ |
| **慢路径可重写** | SparseTensor 数据结构 | `trellis2/modules/sparse/basic.py` | torch | ★★ |
| | `scaled_dot_product_attention`（稠密） | `trellis2/modules/attention/full_attn.py` | flash-attn / xformers / sdpa | ★★ |
| | `sparse_scaled_dot_product_attention` (varlen) | `trellis2/modules/sparse/attention/full_attn.py` | flash-attn varlen | ★★★ |
| | windowed / serialized sparse attention | `trellis2/modules/sparse/attention/windowed_attn.py` | flash-attn + 坐标重排 | ★★★ |
| | **子流形稀疏 Conv3D**（k=3 only） | `trellis2/modules/sparse/conv/conv_flex_gemm.py` | **flex_gemm.ops.spconv** | ★★★★ |
| | SparseDownsample / SparseUpsample / S2C / C2S | `trellis2/modules/sparse/spatial/*.py` | torch + scatter ops | ★★★ |
| | `grid_sample_3d` (FlexGEMM 稀疏体素采样) | `representations/mesh/base.py`, 多个 renderer | **flex_gemm.ops.grid_sample** | ★★★ |
| **CUDA 强依赖** | `flexible_dual_grid_to_mesh` (双重网格 → 三角网格) | `o-voxel/o_voxel/convert/` (CUDA) | **o_voxel C++/CUDA** | ★★★★ |
| | CuMesh: fill_holes / simplify / remove_faces | `representations/mesh/base.py` | **cumesh** | ★★★★ |
| | nvdiffrast 光栅化（可视化 + 烘焙） | `trellis2/renderers/*.py`, `o-voxel/postprocess.py` | **nvdiffrast** | ★★★★ |
| | nvdiffrec PBR split-sum 渲染 | `trellis2/renderers/pbr_mesh_renderer.py` | **nvdiffrec** | ★★★★ |
| | O-Voxel hash/rasterize/serialize CUDA kernels | `o-voxel/src/*.cu` | **自写 CUDA** | ★★★★★ |

★~★★★★★ = 0.5 天 ~ 数月

---

## 2. 目标 / 非目标

### 2.1 目标（M6 完成时）

1. 在 Apple M 系列芯片（Metal）上能跑 `example.py` 的简化版：
   - 输入：单张 PNG / RGBA 图片
   - 输出：顶点 / 三角形 / 体素属性（即 `MeshWithVoxel`）
   - 落盘：能保存为 `.obj` + 顶点色（或最小 `.glb`，不要求 UV 展开 / 纹理烘焙 / 简化）
2. 数值正确性：与 CUDA 参考实现（fp32 路径）在合理误差范围内一致
   - flow models 中间张量 max-abs-err < 1e-2（fp16 / bf16 容差更高）
   - 最终顶点坐标 Chamfer Distance 与 CUDA 输出 < 网格分辨率的 1%
3. 可在 16GB / 32GB Mac 上完成 `512` pipeline_type 的推理（**先不追求 1024 cascade / 1536 cascade**）
4. 加载官方 `microsoft/TRELLIS.2-4B` HuggingFace 权重（无需重训）

### 2.2 非目标

- 训练（保持 CUDA）
- nvdiffrast 替代渲染器（可视化阶段，由 trimesh / matplotlib 替代）
- 高质量 GLB 导出 / UV 展开 / 网格简化 / 纹理烘焙（短期跳过 `to_glb`）
- 真实硬件性能 parity（慢路径接受 10–100× 性能损失）
- texturing pipeline（`Trellis2TexturingPipeline`，依赖更深的 nvdiffrast / o_voxel，留到后续）

---

## 3. 整体架构：双后端 + 抽象层

### 3.1 原则

- **不破坏现有 CUDA 路径**。MLX 路径放在独立的 `trellis2_mlx/` 包；`trellis2/` 维持现状。
- 借鉴现有的 backend 切换模式（见 `trellis2/modules/attention/config.py`、`trellis2/modules/sparse/config.py`），新增 `mlx` 作为后端选项。
- 模型权重通过 **一次性转换脚本** 从 PyTorch 的 `.safetensors` 转出为 MLX 的 `.safetensors`（处理 Conv 权重布局、param 命名、dtype）。
- 抽象层最小化：不要预先建一个"通用 NN 框架"。MLX 子包直接重写关键 nn.Module 子类，权重逐层加载即可。

### 3.2 目录布局（新增）

```
trellis2_mlx/                   # 新增的 MLX 推理包
├── __init__.py
├── ops/                         # 基础算子层
│   ├── attention.py             # 稠密 SDPA on MLX
│   ├── sparse_tensor.py         # SparseTensor (mlx-backed)
│   ├── sparse_conv.py           # 子流形稀疏 Conv3D 慢路径
│   ├── sparse_attention.py      # varlen 稀疏 SDPA (block-diag mask)
│   ├── sparse_pool.py           # Downsample/Upsample/Spatial2Channel
│   ├── grid_sample.py           # 稀疏体素三线性采样 (替代 flex_gemm)
│   └── rope.py                  # rotary position embedding
├── modules/                     # nn.Module 等价物
│   ├── norm.py                  # LayerNorm / GroupNorm (mlx.nn)
│   ├── linear.py                # SparseLinear
│   ├── transformer_blocks.py    # ModulatedTransformerCrossBlock
│   └── sparse_unet.py           # SparseResBlock3d 等
├── models/                      # 高层模型
│   ├── sparse_structure_flow.py
│   ├── sparse_structure_vae.py  # 稠密 3D ConvNet decoder
│   ├── structured_latent_flow.py
│   └── sc_vaes/
│       ├── sparse_unet_vae.py
│       └── fdg_vae.py
├── pipelines/
│   ├── samplers.py              # Flow Euler + CFG + Guidance Interval
│   └── image_to_3d.py           # 主入口
├── extractors/
│   └── dinov3.py                # DINOv3 (转换 weights 或调 transformers + MPS)
├── geometry/
│   └── dual_grid_to_mesh.py     # 纯 Python 双重网格 → 三角网格
├── io/
│   ├── weight_convert.py        # torch.safetensors → mlx weights
│   └── export_mesh.py           # 最小 OBJ/GLB 写出 (绕开 CuMesh)
└── tests/                       # 与 CUDA 参考对齐的数值测试

example_mlx.py                   # 顶层 demo
```

### 3.3 张量约定

| 维度 | PyTorch (现行) | MLX (新) |
|---|---|---|
| Linear weight | `[out, in]` | `[out, in]` ✅ 同 |
| Conv3d weight | `[Co, Ci, kD, kH, kW]` | `[Co, kD, kH, kW, Ci]` (NHWC-like) |
| FlexGEMM SubMConv 权重 | `[Co, kD, kH, kW, Ci]` (代码里已 permute) | 同 ✅ |
| Activation tensor | `[N, C, D, H, W]` | `[N, D, H, W, C]` |
| Attention QKV | `[N, L, 3, H, C]` | 同 ✅ |

**结论**：权重转换主要工作量是 `Conv3d`（包括 `SparseStructureDecoder` 内部）。`SparseConv3d` 的权重已经是 channel-last，零成本。

---

## 4. 组件迁移详表（按依赖顺序）

### 4.1 基础设施（M1）

| 组件 | 原文件 | 新文件 | 说明 |
|---|---|---|---|
| 权重加载 | `trellis2/models/__init__.py:from_pretrained` | `trellis2_mlx/io/weight_convert.py` | 用 `safetensors.numpy.load_file` → `mx.array`；Conv3d 权重做 `[Co,Ci,D,H,W] → [Co,D,H,W,Ci]` 的 permute |
| Config 解析 | HF `*.json` | 复用，原样消费 | |
| Dtype 表 | `trellis2/modules/utils.py:str_to_dtype` | `trellis2_mlx/ops/__init__.py` | `'float16' → mx.float16` 等 |

**关键陷阱**：`safetensors.torch.load_file` 不能直接用（它返回 `torch.Tensor`）。改用 `safetensors.numpy.load_file` 或 `mlx.utils.tree_unflatten`。

### 4.2 稠密 Transformer + Flow Sampler（M2）

迁移目标：让 `SparseStructureFlowModel` + `FlowEulerGuidanceIntervalSampler` 在 MLX 跑通。这个模块**完全是稠密 Transformer**，对应 `examples/example.py` 里的 Stage 1 第一步。

| 子组件 | 原实现 | MLX 实现要点 |
|---|---|---|
| `TimestepEmbedder` | sinusoidal + 2-layer MLP | 直接 `mlx.nn.Linear` |
| `AbsolutePositionEmbedder` | `trellis2/modules/transformer/...` | 简单 sinusoidal，纯算子 |
| `RotaryPositionEmbedder` | `trellis2/modules/attention/rope.py` | 用 `mx.cos`/`mx.sin` 即可，无外部依赖 |
| `ModulatedTransformerCrossBlock` | `trellis2/modules/transformer/modulated.py` | Self-attn + Cross-attn + adaLN modulation + MLP |
| `scaled_dot_product_attention` | flash-attn / xformers / sdpa | **MLX 已有 `mlx.fast.scaled_dot_product_attention`** ✅ |
| `LayerNorm` (no affine + affine) | `LayerNorm32` 强制 fp32 | `mlx.nn.LayerNorm`，必要时手写 fp32 路径 |
| `FlowEulerSampler` | numpy + tqdm + torch.randn | 一行一行翻译，`torch.randn` → `mx.random.normal` |
| CFG / Guidance Interval | mixin 形式 | 同 |

**M2 验收**：固定随机种子的情况下，加载预训练 SS flow + 32³ 稠密 decoder，输入零向量 image cond，对比 CUDA 与 MLX 路径的 sparse structure logits（`max|diff| / max|gt|`）< 1e-2。

### 4.3 SparseTensor 与稀疏算子（M3）—— **核心难点**

#### 4.3.1 SparseTensor 数据结构

PyTorch 版的 `SparseTensor` (`trellis2/modules/sparse/basic.py`) 本质是：
```python
feats   : Tensor [Ntotal, C]                   # 拼接所有 batch 的 active 体素特征
coords  : Tensor [Ntotal, 4] int   (b, x, y, z)
layout  : List[slice]                          # 每个 batch 的切片
spatial_shape : tuple                          # 全局 3D 网格尺寸
spatial_cache : Dict[str, Any]                 # 缓存的 neighbor map 等
```

MLX 版直接复刻成 `mx.array` + Python dict 缓存：

```python
class MLXSparseTensor:
    feats: mx.array        # [Ntotal, C]
    coords: mx.array       # [Ntotal, 4] int32
    layout: List[slice]
    spatial_shape: Tuple[int, int, int]
    _cache: Dict[str, Any]
```

迁移直接逐 method 翻译 (`.replace`, `.to`, `+ - * /` 重载等)，无算法改动。

#### 4.3.2 子流形稀疏 Conv3D 慢路径 —— 最关键的一段

**原 FlexGEMM 调用**（`conv_flex_gemm.py:46`）：
```python
out, neighbor_cache_ = sparse_submanifold_conv3d(
    x.feats, x.coords, [*x.shape, *x.spatial_shape],
    self.weight,  # [Co, kD, kH, kW, Ci]
    self.bias,
    neighbor_cache, self.dilation
)
```

**算法**（子流形 = 输出坐标 == 输入坐标，只在已激活体素上算）：
1. 给每个 active voxel `i`，对 3×3×3 个 kernel offset，查找 active voxel 索引 `nbr[i, k]`（若该邻居不存在，标记 -1）
2. `out[i, co] = bias[co] + Σ_{k} Σ_{ci} weight[co, k, ci] * feats[nbr[i, k], ci]` （nbr=-1 时贡献 0）

**MLX 慢路径实现**（`trellis2_mlx/ops/sparse_conv.py`）：

```python
def build_neighbor_map(coords, kernel_size=3, dilation=1, spatial_shape):
    # coords: mlx [N,4], 返回 nbr: [N, K^3] int32, -1 表示不存在
    # 实现：Python dict {(b,x,y,z) -> idx} O(N·K^3)，一次性建好后缓存
    ...

def submanifold_conv3d(feats, coords, weight, bias, nbr_map):
    # feats: [N, Ci], weight: [Co, K^3, Ci] (flatten kernel), bias: [Co]
    # 1. gather: gathered = feats[nbr_map.clip(0)]            # [N, K^3, Ci]
    # 2. mask:  gathered = gathered * (nbr_map >= 0)[..., None]
    # 3. einsum: out = gathered @ weight.transpose             # einsum('nki,oki->no')
    # 4. + bias
    return out
```

数学等价于 FlexGEMM 但慢 **~30–100×**：每个 conv 调用涉及 N×27 次 gather + N×Co×27×Ci 次 FMA。**优化点**（按需打开）：
- (a) `nbr_map` 在第一次 conv 后缓存到 `SparseTensor._cache`（已有机制）。
- (b) 用 `mx.compile` 包住核心 einsum；MLX 的 lazy graph 会合并 gather + matmul。
- (c) 内核为 1×1×1 时退化为 `feats @ W.T + b`，绕过 nbr_map。
- (d) 用 hashmap 加速 neighbor lookup —— Python dict 在 1M voxel 量级上约 0.5–2s/conv；可后续替换为基于 `mx.searchsorted` 的 Z-order key 二分查找。

**性能预估**（单步 forward，512³ 体素 ~ 250k active voxels，C=128，48 层 SubMConv3d）：
- FlexGEMM CUDA: 估 50–150ms
- MLX 慢路径 (M-Pro): 估 30–90s ⟶ **可接受用于 demo**

#### 4.3.3 SparseDownsample / SparseUpsample / S2C / C2S / Subdivide

| 算子 | 实现思路 |
|---|---|
| `SparseDownsample(2)` | `coords' = coords // 2`，按 (b,x',y',z') 聚合 mean，复用现有 `mx.scatter_add` 等价（手写 segment_mean） |
| `SparseUpsample(2)` | 每个父体素根据 `subdiv` 8-bit mask 生成至多 8 个子体素，重复 feats |
| `SparseSpatial2Channel(2)` | 把 2×2×2 邻域拼到 channel 维度，需先建 2-邻域 map（和 conv 同机制） |
| `SparseChannel2Spatial(2)` | S2C 的逆 |
| `SparseSubdivide` | 类似 Upsample 但保持原 voxel 不变 |

**关键依赖**：`mx.scatter_add` / `mx.segment_sum`。MLX 1.x 提供 `mx.scatter`（with reduction）和 indexing；如果某 API 缺失，用 `argsort + cumsum + diff` 手工实现 segment ops。

#### 4.3.4 子流形稀疏 Attention（windowed / serialized）

`trellis2/modules/sparse/attention/windowed_attn.py` 把 SparseTensor 按窗口（3D bbox）分组，每个窗口内做 dense attention。MLX 路径：
1. 按窗口 ID 排序 feats（Z-order curve / window id）
2. 用 padding mask 拼成 `[Nwin, Lmax, H, C]` 稠密张量
3. 调 MLX SDPA + 加性 mask
4. 反排列回 SparseTensor

对应 `_serialize` / `_windowize` 函数中的 Z-order 计算：纯 Python 即可，O(N log N)。

#### 4.3.5 grid_sample_3d 替代

只有 `representations/mesh/base.py:query_attrs` 和 renderer 用到。**短期方案**：只支持 `mode='trilinear'`、`align_corners=False`、`out-of-bounds=zero`，在 active voxel 集合上做三线性插值：
1. 把 8 个角点坐标量化，dict-lookup active voxel
2. 加权和

`example.py` 的核心路径只用一次 query_attrs (`query_vertex_attrs`)，体量可控。

### 4.4 形状解码器：SparseStructureDecoder（稠密 3D ConvNet）

`trellis2/models/sparse_structure_vae.py` 是稠密 3D ConvNet（输入 8³~64³，无稀疏），**MLX 直接支持**：`mlx.nn.Conv3d` + GroupNorm + SiLU。这是 M2 就能顺带打通的。

### 4.5 Sparse U-Net VAE 解码器（M4）

`trellis2/models/sc_vaes/sparse_unet_vae.py` (521 行) + `fdg_vae.py` (110 行) 是 M3 算子的最大消费者。迁移=把 `SparseResBlock3d`/`SparseConvNeXtBlock3d`/`SparseResBlockS2C3d`/`SparseResBlockC2S3d` 等的 forward 一一翻译。

**最后一步**：`FlexiDualGridVaeDecoder.forward` 调用 `o_voxel.convert.flexible_dual_grid_to_mesh` —— 这是 CUDA。**纯 Python 替代方案**（`trellis2_mlx/geometry/dual_grid_to_mesh.py`）：

输入：active voxel coords `[N,3]` + 每个 voxel 的 corner offset `[N,3]`(vertex feats) + intersected mask `[N,3]` + quad lerp `[N,1]`。

算法（标准 Dual Contouring / Flexible Dual Grid 变体）：
1. 对每个 active voxel，根据 6 条边的 `intersected` 状态判断要生成的 quad
2. 4 个相邻 voxel 的 corner offset 组成 quad 顶点
3. 输出 `vertices [V,3]`、`faces [F,3]` (quad → 2 三角形)

CPU 单线程参考实现 ~50 行 Python，处理 1M voxel 量级 < 30s。后续可用 numba 加速。

### 4.6 DINOv3 图像特征提取（M2 后期）

两条路：
- **路 A（更快）**：保留 `transformers.DINOv3ViTModel`，让它跑在 PyTorch 的 MPS 后端。优点：零工作量，DINOv3 是 ViT，MPS 完全支持；缺点：与 MLX 张量需要在 numpy 中转。
- **路 B（更纯）**：把 DINOv3 也搬到 MLX。HuggingFace 有 `mlx-community/dinov3` 类似仓库（验证一下），或者手写 ViT。

**建议**：M2 先走路 A（pytorch+mps），M6 之后再决定是否换路 B。

### 4.7 后处理 / 导出（M5）

例子 `example.py` 的最后两步严重依赖 CUDA：
- `mesh.simplify(16777216)` ← CuMesh ⟶ **跳过**（导出时不简化）
- `make_pbr_vis_frames(render_video(...))` ← nvdiffrast/rec ⟶ **跳过**（不生成 mp4）
- `o_voxel.postprocess.to_glb(...)` ← O-Voxel + CuMesh + nvdiffrast ⟶ **替换为最小导出**

**最小导出**（`trellis2_mlx/io/export_mesh.py`）：
1. 给每个 vertex 查询体素属性（`base_color`），写 OBJ + per-vertex color；或写最小 GLB：mesh + per-vertex COLOR_0
2. 不做 UV 展开，不烘焙纹理

例子用法：
```python
mesh = pipeline.run(image)[0]
export_mesh.write_glb(mesh, "sample.glb")  # 顶点色版本
```

---

## 5. 后端切换抽象

新增 env：

```
TRELLIS_BACKEND = "torch" | "mlx"   # 默认 "torch"
```

入口：

```python
# trellis2_mlx/pipelines/image_to_3d.py
def from_pretrained(path):
    # 1. 下载 HF 仓库
    # 2. 用 weight_convert 转换所有 *.safetensors 到 mx.array dict
    # 3. 装配 MLX 版的 model graph
    # 4. 返回 MLXPipeline(对外接口与 Trellis2ImageTo3DPipeline.run 一致)
```

用户 demo：

```python
# example_mlx.py
from trellis2_mlx.pipelines import Trellis2ImageTo3DPipelineMLX
from PIL import Image

pipe = Trellis2ImageTo3DPipelineMLX.from_pretrained("microsoft/TRELLIS.2-4B")
mesh = pipe.run(Image.open("assets/example_image/T.png"), pipeline_type='512')[0]
pipe.export_glb(mesh, "sample.glb")
```

**重要约束**：不修改 `trellis2/` 下任何 .py 文件的运行时行为（只允许加 type hint / docstring / 不影响导入的小修补）；任何 MLX 相关代码都放 `trellis2_mlx/`。

---

## 6. 关键风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| MLX 缺关键算子（如 `scatter_add` reduction='mean'） | 影响 SparseDownsample/Upsample | 用 `argsort + segment_sum + count` 手工实现；做一个小型 ops shim |
| 子流形 conv 慢路径 OOM（27× gather 临时张量爆） | 解码器跑不动 | 沿 N 维度分块（chunk=8192）；最坏情况单 conv 用 Python 双层 for 循环（仍可用，但更慢） |
| 双重网格 → 网格的 CPU 实现有 bug，输出网格破碎 | M5 验收失败 | 用一个低分辨率（32³）case 与 CUDA 输出做 vertex set 对照；预留 numpy/trimesh 后备路径 |
| DINOv3 在 transformers 库依赖 flash-attn / sdpa | image cond 失败 | 强制 `attn_implementation="eager"`，或转写到 MLX |
| 权重 dtype 不匹配（模型 bf16，MLX 不支持的某些 dtype 组合） | 数值偏差 | 中间张量统一升 fp32，与 `LayerNorm32` 的现有约定一致 |
| `mlx.nn.Conv3d` 的 padding/stride 边界与 PyTorch 不一致 | SparseStructureDecoder 输出偏移 | 单元测试对照 PyTorch CPU 参考；必要时手写 conv3d 的 NHWC 实现 |
| HF safetensors 中 dtype 是 bf16，转 MLX 时丢精度 | 累积误差 | 转换脚本支持 `dtype='upcast_fp32'` 选项 |
| `flexible_dual_grid_to_mesh` 的输出格式有 CUDA 仅有的细节（顶点排序约定、quad 方向） | 网格法线反转 | 实现时严格按 `o-voxel/src/convert/` 的 C++ 头文件复现；为此 M4 末加一个 numerical regression test |

---

## 7. 数值对齐 / 测试策略

每个 milestone 必须有一个 **CUDA vs MLX 对照测试**（在 CUDA 机上预生成 reference tensor，存为 `.npz`，Mac 端加载比对）：

```
tests/
├── data/                       # CUDA 预生成的参考张量 (.npz)
├── test_m1_weight_load.py      # 验证权重转换无 bit 丢失（fp32 路径）
├── test_m2_flow_model.py       # SparseStructureFlowModel 单次 forward
├── test_m2_euler_sampler.py    # 50 steps Euler，固定 seed
├── test_m3_sparse_conv.py      # 单个 SubMConv3d，N=1000 active voxels
├── test_m3_sparse_attn.py      # windowed sparse attention
├── test_m4_shape_decoder.py    # FlexiDualGridVaeDecoder 输出顶点
├── test_m4_dual_grid.py        # Python flexible_dual_grid_to_mesh 与 CUDA 对照
├── test_e2e_512.py             # 端到端 512 pipeline
```

容差表：

| 路径 | max relative error | max abs error |
|---|---|---|
| fp32 权重加载 | 0 | 0 |
| fp32 forward (Transformer) | 1e-4 | 1e-4 |
| fp16/bf16 forward | 5e-2 | 1e-2 |
| 50-step Euler 采样输出 | 5e-2 | 5e-2 |
| sparse conv 单层输出 | 1e-4 (fp32) | 1e-4 |
| 端到端顶点 Chamfer | — | < voxel_size |

---

## 8. 实施排期建议

> 单人 / 全职估计；并行可压缩 30%。

| 里程碑 | 范围 | 估时 | 关键交付 |
|---|---|---|---|
| **M1** | 权重加载 + dtype 表 + `mx.array` 基础 + 测试脚手架 | 3 天 | `weight_convert.py` + M1 test 通过 |
| **M2** | 稠密 Transformer、attention、APE/RoPE、Flow Euler、SS Flow + SS Decoder（稠密）；DINOv3 走 MPS | 1.5 周 | `pipeline.sample_sparse_structure` 输出 coords 与 CUDA 一致 |
| **M3** | `MLXSparseTensor` + 子流形 SubMConv3D + Downsample/Upsample + S2C/C2S + sparse SDPA + windowed sparse attn | 2.5 周 | 单算子测试全绿；性能日志记录 |
| **M4** | SLat Flow Models (shape + tex) + Sparse U-Net VAE Decoder + 双重网格 → 三角网格 (CPU) | 2 周 | 给定 512 coords，输出 Mesh 顶点 Chamfer 对照 |
| **M5** | 体素属性查询 + 最小 GLB / OBJ 导出（顶点色） | 4 天 | `sample.glb` 在 Blender / threejs 中可正常显示 |
| **M6** | `example_mlx.py` 端到端 + README 段 + CI 加 Mac runner（可选） | 1 周 | 在 M-Pro 上从 PNG 输入 → GLB 输出 < 10 分钟 |

**合计**：约 **8 周**（仅推理、`pipeline_type='512'`）。

后续可选扩展：
- `1024` / `1024_cascade`（更大显存压力，需要分块更精细）
- texturing pipeline（依赖 nvdiffrast，工作量再 +6 周）
- numba / metal kernel 加速 sparse conv（性能 5–10×）
- MLX bf16 端到端

---

## 9. 不做的事（明确）

- 不重写 FlexGEMM 为 Metal kernel —— 慢路径足够
- 不替换 nvdiffrast 渲染 —— 短期跳过可视化，长期用 trimesh / pyrender / Open3D 即可
- 不重写 CuMesh —— 跳过网格简化 / 填洞，必要时用 trimesh CPU 等价
- 不动 `trellis2/trainers/` 与 `trellis2/datasets/`
- 不动 `o-voxel/src/*.cu`（仅在 M4 用 Python 重写 `flexible_dual_grid_to_mesh` 这一处算法）

---

## 10. 下一步

1. 用户审阅本设计文档；确认范围 / 排期 / 取舍
2. 启动 M1：
   - 新建 `trellis2_mlx/` 包骨架
   - 实现 `weight_convert.py`
   - 下载 HF checkpoint，跑通 fp32 权重转换 + 单 Linear forward 对照
3. 在 README.md 增加 "Apple Silicon (MLX, inference)" 段，标注 experimental

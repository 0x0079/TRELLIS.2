"""
Load and convert PyTorch safetensors checkpoints to MLX-friendly form.

The reference checkpoints in `microsoft/TRELLIS.2-4B` are saved by safetensors as
contiguous float arrays. The conversions we need are:

1. Dtype:    bfloat16 / float16 -> float32 (we upcast for stability; MLX bf16
             support is fine but mixing with our LayerNorm32 path is simpler in fp32).
2. Conv3d:   PyTorch nn.Conv3d weight is [Co, Ci, kD, kH, kW]; mlx.nn.Conv3d uses
             [Co, kD, kH, kW, Ci]. Permute.
3. SparseConv3d (flex_gemm): weight already [Co, kD, kH, kW, Ci] -> no-op.
4. Param-name remapping for the few cases where our MLX module structure flattens
   a nn.Sequential (e.g. `mlp` -> `mlp_0` / `mlp_2`, `adaLN_modulation` -> `0`/`1`).

Use `convert_state_dict()` on a flat dict of np arrays loaded from safetensors.

Usage:

    state = load_safetensors_to_numpy("model.safetensors")
    mlx_weights = convert_state_dict(state)  # dict[str, mx.array]
    module.load_weights(list(mlx_weights.items()))
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Tuple

import numpy as np
import mlx.core as mx


# ---- Loading raw safetensors as numpy --------------------------------------


def load_safetensors_to_numpy(path: str) -> Dict[str, np.ndarray]:
    """
    Load a safetensors file as a flat dict of numpy arrays (upcasting bf16/fp16
    to float32).  Uses safetensors.numpy when available, otherwise falls back
    to safetensors.torch + numpy().
    """
    try:
        from safetensors.numpy import load_file as _np_load
        return _np_load(path)
    except Exception:
        from safetensors.torch import load_file as _t_load
        loaded = _t_load(path)
        out = {}
        for k, v in loaded.items():
            arr = v.detach().cpu()
            if str(arr.dtype) in ("torch.bfloat16",):
                arr = arr.to(dtype=__import__("torch").float32)
            out[k] = arr.numpy()
        return out


def safe_to_mx(arr: np.ndarray, dtype=mx.float32) -> mx.array:
    if arr.dtype == np.float16 or arr.dtype.name == "bfloat16":
        arr = arr.astype(np.float32)
    elif arr.dtype.kind == "f" and arr.dtype.itemsize > 4:
        arr = arr.astype(np.float32)
    return mx.array(arr).astype(dtype)


# ---- Conversion rules ------------------------------------------------------


def _is_conv3d_weight(name: str, shape) -> bool:
    """Detect a dense Conv3d weight: nn.Conv3d.weight is 5D [Co, Ci, kD, kH, kW]."""
    return name.endswith(".weight") and len(shape) == 5 and \
        ("conv" in name.split(".")[-2].lower() or name.split(".")[-2] in ("input_layer", "output_layer"))


def _maybe_remap_sequential(name: str) -> str:
    """
    The reference uses nn.Sequential in several spots, naming children `0`, `1`, `2`.
    We flatten them in MLX to explicit attribute names. Remap accordingly.
    """
    # FeedForwardNet: self.mlp = nn.Sequential(Linear, GELU, Linear)
    #   reference param key:  '*.mlp.0.weight', '*.mlp.2.weight'
    #   MLX module attrs:     mlp.mlp_0.weight, mlp.mlp_2.weight
    # The 'mlp' attr is itself an MLX Module, so we need to insert `mlp_0`/`mlp_2`.
    name = name.replace(".mlp.0.", ".mlp.mlp_0.").replace(".mlp.2.", ".mlp.mlp_2.")

    # TimestepEmbedder.mlp = nn.Sequential(Linear, SiLU, Linear)
    #   '*.t_embedder.mlp.0.weight' -> 't_embedder.mlp.mlp_0.weight'  (same as FFN)
    # already handled by the rule above.

    # adaLN_modulation = nn.Sequential(SiLU, Linear)
    #   '*.adaLN_modulation.1.weight' -> '*.adaLN_modulation_1.weight'
    #   (the SiLU has no params, so .0 doesn't appear)
    name = name.replace(".adaLN_modulation.1.", ".adaLN_modulation_1.")

    # SparseConvNeXtBlock3d uses a Linear-SiLU-Linear MLP under `mlp`.
    # Handled by the same `mlp.0` -> `mlp.mlp_0` rule above.

    return name


def convert_state_dict(
    state: Dict[str, np.ndarray],
    *,
    target_dtype=mx.float32,
    is_dense_conv3d=None,
) -> Dict[str, mx.array]:
    """
    Convert a torch-style state dict (numpy arrays) into a dict of mx.array values
    ready to be loaded into an MLX module.

    Args:
        state: flat dict of np arrays
        target_dtype: final dtype for floating-point parameters
        is_dense_conv3d: optional callable (key, shape) -> bool to identify Conv3d
            weights that need a [Co, Ci, kD, kH, kW] -> [Co, kD, kH, kW, Ci] permute.
            Defaults to a heuristic based on the parameter name.
    """
    if is_dense_conv3d is None:
        is_dense_conv3d = _is_conv3d_weight

    out: Dict[str, mx.array] = {}
    for k, v in state.items():
        new_k = _maybe_remap_sequential(k)
        if is_dense_conv3d(k, v.shape):
            v = np.transpose(v, (0, 2, 3, 4, 1))  # [Co, kD, kH, kW, Ci]
        # Some normalization weights might be 1-D fp32 already; leave as is.
        if v.dtype.kind == "f":
            arr = safe_to_mx(v, dtype=target_dtype)
        else:
            arr = mx.array(v)
        out[new_k] = arr
    return out


def load_config_and_state(prefix: str) -> Tuple[dict, Dict[str, np.ndarray]]:
    """
    Load (`{prefix}.json`, `{prefix}.safetensors`) from disk or HF Hub.

    `prefix` follows the same convention as trellis2.models.from_pretrained.
    """
    if os.path.exists(f"{prefix}.json") and os.path.exists(f"{prefix}.safetensors"):
        cfg_path = f"{prefix}.json"
        st_path = f"{prefix}.safetensors"
    else:
        from huggingface_hub import hf_hub_download

        parts = prefix.split("/")
        repo = f"{parts[0]}/{parts[1]}"
        name = "/".join(parts[2:])
        cfg_path = hf_hub_download(repo, f"{name}.json")
        st_path = hf_hub_download(repo, f"{name}.safetensors")
    with open(cfg_path) as f:
        cfg = json.load(f)
    state = load_safetensors_to_numpy(st_path)
    return cfg, state

"""
Load and convert PyTorch safetensors checkpoints to MLX-friendly form.

The reference checkpoints in `microsoft/TRELLIS.2-4B` are saved by safetensors as
contiguous float arrays. The conversions we need are:

1. Dtype:    bfloat16 / float16 -> float32 (we upcast for stability; MLX bf16
             support is fine but mixing with our LayerNorm32 path is simpler in fp32).
2. Conv3d:   PyTorch nn.Conv3d weight is [Co, Ci, kD, kH, kW]; mlx.nn.Conv3d uses
             [Co, kD, kH, kW, Ci]. Permute.
3. SparseConv3d (flex_gemm): weight already [Co, kD, kH, kW, Ci] -> no-op.

We identify dense vs sparse Conv3d weights by SHAPE, not by name, because both
appear under attributes called `conv1` / `conv2` / `skip_connection`:

  Dense Conv3d weight:   [Co, Ci, kD, kH, kW]   dims 2,3,4 are the kernel (small)
  Sparse Conv3d weight:  [Co, kD, kH, kW, Ci]   dims 1,2,3 are the kernel (small)

For typical kernels (kD == kH == kW <= 7), the two are unambiguous as long as
Ci > 7, which holds throughout the TRELLIS.2-4B architecture.

Usage:
    state = load_safetensors_to_numpy("model.safetensors")
    mlx_weights = convert_state_dict(state)  # dict[str, mx.array]
    module.load_weights(list(mlx_weights.items()))

Diagnostics:
    python -m trellis2_mlx.io.weight_convert --inspect <prefix>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np

# mx is imported lazily so this file is usable without mlx installed (for diagnostics).


# ---- Loading raw safetensors as numpy --------------------------------------


def load_safetensors_to_numpy(path: str) -> Dict[str, np.ndarray]:
    """Flat dict of np.ndarray, with bf16/fp16 upcast to fp32."""
    try:
        from safetensors.numpy import load_file as _np_load
        return _np_load(path)
    except Exception:
        from safetensors.torch import load_file as _t_load
        import torch
        loaded = _t_load(path)
        out = {}
        for k, v in loaded.items():
            arr = v.detach().cpu()
            if str(arr.dtype) == "torch.bfloat16":
                arr = arr.to(dtype=torch.float32)
            out[k] = arr.numpy()
        return out


def safe_to_mx(arr: np.ndarray, dtype=None):
    import mlx.core as mx
    if arr.dtype == np.float16 or arr.dtype.name == "bfloat16":
        arr = arr.astype(np.float32)
    elif arr.dtype.kind == "f" and arr.dtype.itemsize > 4:
        arr = arr.astype(np.float32)
    out = mx.array(arr)
    if dtype is not None:
        out = out.astype(dtype)
    return out


# ---- Conv3d weight detection ----------------------------------------------


def is_dense_conv3d_weight(name: str, shape) -> bool:
    """
    Identify a torch.nn.Conv3d weight that needs a [Co, Ci, kD, kH, kW] ->
    [Co, kD, kH, kW, Ci] permute. Distinguishes from SparseConv3d weight which
    is already [Co, kD, kH, kW, Ci] in the checkpoint.
    """
    if not name.endswith(".weight"):
        return False
    if len(shape) != 5:
        return False
    # Sparse Conv3d: kernel dims are 1,2,3 (small) and Ci is dim 4 (large).
    if shape[1] == shape[2] == shape[3] and shape[1] <= 7:
        return False
    # Dense Conv3d: kernel dims are 2,3,4 (small) and Ci is dim 1 (large).
    if shape[2] == shape[3] == shape[4] and shape[2] <= 7:
        return True
    return False


# ---- Conversion ------------------------------------------------------------


def convert_state_dict(state: Dict[str, np.ndarray], target_dtype=None):
    """Convert a torch-style flat state dict to a dict of mx.array values."""
    import mlx.core as mx
    if target_dtype is None:
        target_dtype = mx.float32

    out = {}
    for k, v in state.items():
        if is_dense_conv3d_weight(k, v.shape):
            v = np.transpose(v, (0, 2, 3, 4, 1)).copy()
        if v.dtype.kind == "f":
            arr = safe_to_mx(v, dtype=target_dtype)
        else:
            arr = mx.array(v)
        out[k] = arr
    return out


def load_config_and_state(prefix: str) -> Tuple[dict, Dict[str, np.ndarray]]:
    """Load (`{prefix}.json`, `{prefix}.safetensors`) from disk or HF Hub."""
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


# ---- Helpers for matching MLX module parameter trees ----------------------


def diff_keys(checkpoint_keys, model_keys, *, max_show: int = 20):
    """Format a nice diff of two key sets. Returns (missing_in_ckpt, missing_in_model)."""
    in_ckpt = set(checkpoint_keys)
    in_model = set(model_keys)
    missing_in_ckpt = sorted(in_model - in_ckpt)
    missing_in_model = sorted(in_ckpt - in_model)
    lines = []
    if missing_in_model:
        lines.append(f"=== {len(missing_in_model)} key(s) in checkpoint but NOT in MLX model ===")
        for k in missing_in_model[:max_show]:
            lines.append(f"  + {k}")
        if len(missing_in_model) > max_show:
            lines.append(f"  ... ({len(missing_in_model) - max_show} more)")
    if missing_in_ckpt:
        lines.append(f"=== {len(missing_in_ckpt)} key(s) in MLX model but NOT in checkpoint ===")
        for k in missing_in_ckpt[:max_show]:
            lines.append(f"  - {k}")
        if len(missing_in_ckpt) > max_show:
            lines.append(f"  ... ({len(missing_in_ckpt) - max_show} more)")
    return missing_in_ckpt, missing_in_model, "\n".join(lines)


# ---- CLI -------------------------------------------------------------------


def _cli():
    p = argparse.ArgumentParser(description="Inspect a TRELLIS.2 checkpoint")
    sub = p.add_subparsers(dest="cmd", required=True)
    insp = sub.add_parser("--inspect", help="dump (name, shape, dtype) for every tensor")
    insp.add_argument("prefix", help="local path stem or HF model id stem (no extension)")
    insp.add_argument("--head", type=int, default=0, help="print only N keys (0=all)")
    insp.add_argument("--config-only", action="store_true", help="only dump the JSON config")

    # Allow `--inspect <prefix>` (no explicit subcommand) for convenience.
    raw = sys.argv[1:]
    if raw and raw[0] == "--inspect":
        ns = argparse.Namespace(cmd="--inspect", prefix=raw[1] if len(raw) > 1 else None,
                                head=0, config_only=False)
        i = 2
        while i < len(raw):
            if raw[i] == "--head":
                ns.head = int(raw[i + 1]); i += 2
            elif raw[i] == "--config-only":
                ns.config_only = True; i += 1
            else:
                i += 1
        if not ns.prefix:
            p.print_help(); sys.exit(2)
        return ns
    return p.parse_args()


def main():
    args = _cli()
    cfg, state = load_config_and_state(args.prefix)
    print("=" * 80)
    print(f"prefix: {args.prefix}")
    print(f"model class: {cfg.get('name', '?')}")
    print(f"args (first 10 keys): {list(cfg.get('args', {}).keys())[:10]}")
    if args.config_only:
        print("\n" + json.dumps(cfg, indent=2))
        return
    print(f"\n{len(state)} tensors:")
    print("-" * 80)
    items = list(state.items())
    if args.head:
        items = items[: args.head]
    for k, v in items:
        marker = "  (dense conv3d, will permute)" if is_dense_conv3d_weight(k, v.shape) else ""
        print(f"{k:60s}  {str(v.shape):28s}  {v.dtype}{marker}")
    if args.head and len(state) > args.head:
        print(f"\n... ({len(state) - args.head} more — pass --head 0 to see all)")


if __name__ == "__main__":
    main()

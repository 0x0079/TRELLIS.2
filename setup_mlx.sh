#!/usr/bin/env bash
#
# Install the Apple MLX (inference-only) backend for TRELLIS.2.
#
# This script is a companion to the CUDA-focused setup.sh and is the path to use
# on Apple Silicon. It does NOT install flash-attn / FlexGEMM / nvdiffrast / CuMesh /
# o-voxel — the MLX backend reimplements (or skips) those pieces.
#
# Usage:
#   bash setup_mlx.sh [--new-env] [--no-torch]
#
# Options:
#   --new-env   Create a fresh conda env named `trellis2_mlx` with Python 3.11.
#   --no-torch  Skip the PyTorch + transformers install. Use this only if you plan
#               to compute DINOv3 image features yourself and feed them to the
#               pipeline as an mx.array (the MLX path delegates DINOv3 to PyTorch+MPS
#               for now; see docs/MLX_MIGRATION.md section 4.6).

set -e

NEW_ENV=false
NO_TORCH=false
for arg in "$@"; do
  case "$arg" in
    --new-env) NEW_ENV=true ;;
    --no-torch) NO_TORCH=true ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0 ;;
    *)
      echo "Unknown option: $arg"; exit 2 ;;
  esac
done

# Platform sanity check (not fatal — MLX runs on Linux too, just without Metal)
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "[setup_mlx] Note: MLX runs accelerated on macOS / Apple Silicon."
  echo "[setup_mlx] On Linux you'll get CPU execution only (slow)."
fi

if [ "$NEW_ENV" = true ]; then
  if ! command -v conda >/dev/null; then
    echo "[setup_mlx] conda not found; please install Miniconda first."
    exit 1
  fi
  conda create -y -n trellis2_mlx python=3.11
  echo "[setup_mlx] Now run: conda activate trellis2_mlx && bash setup_mlx.sh"
  exit 0
fi

WORKDIR=$(pwd)
REQ_FILE="$WORKDIR/requirements-mlx.txt"

if [ ! -f "$REQ_FILE" ]; then
  echo "[setup_mlx] Cannot find $REQ_FILE — run this script from the repo root."
  exit 1
fi

echo "[setup_mlx] Installing MLX runtime + tools from $REQ_FILE ..."
if [ "$NO_TORCH" = true ]; then
  # Strip the torch/torchvision/transformers lines for a leaner install.
  TMP_REQ=$(mktemp)
  grep -v -E '^(torch|torchvision|transformers)\b' "$REQ_FILE" > "$TMP_REQ"
  pip install -r "$TMP_REQ"
  rm -f "$TMP_REQ"
else
  pip install -r "$REQ_FILE"
fi

echo
echo "[setup_mlx] Done. Quick smoke test:"
echo "  pytest trellis2_mlx/tests -v"
echo
echo "[setup_mlx] Full e2e (needs ~10 GB of HuggingFace cache):"
echo "  python example_mlx.py assets/example_image/T.png sample_mlx.glb"

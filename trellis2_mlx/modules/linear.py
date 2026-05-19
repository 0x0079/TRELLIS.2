"""
Linear / SparseLinear for MLX.

SparseLinear is just an mlx.nn.Linear that consumes a VarLenTensor and
returns one with the same layout but new features.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..ops.sparse_tensor import VarLenTensor


class SparseLinear(nn.Linear):
    def __call__(self, x: VarLenTensor) -> VarLenTensor:
        return x.replace(super().__call__(x.feats))

"""Low-level operator implementations on top of MLX."""

from . import sparse_tensor
from . import sparse_conv
from . import sparse_pool
from . import sparse_attention
from . import attention
from . import rope
from . import grid_sample

from .sparse_tensor import SparseTensor, VarLenTensor, sparse_cat, sparse_unbind
from .attention import scaled_dot_product_attention
from .rope import RotaryPositionEmbedder, AbsolutePositionEmbedder

__all__ = [
    "sparse_tensor",
    "sparse_conv",
    "sparse_pool",
    "sparse_attention",
    "attention",
    "rope",
    "grid_sample",
    "SparseTensor",
    "VarLenTensor",
    "sparse_cat",
    "sparse_unbind",
    "scaled_dot_product_attention",
    "RotaryPositionEmbedder",
    "AbsolutePositionEmbedder",
]


def str_to_dtype(s):
    import mlx.core as mx
    table = {
        "f16": mx.float16, "fp16": mx.float16, "float16": mx.float16,
        "bf16": mx.bfloat16, "bfloat16": mx.bfloat16,
        "f32": mx.float32, "fp32": mx.float32, "float32": mx.float32,
    }
    return table[s]

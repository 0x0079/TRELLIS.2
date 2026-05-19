"""
Dense scaled-dot-product attention for MLX.

Mirrors trellis2/modules/attention/full_attn.py signatures (qkv | q,kv | q,k,v).
Tensors are in PyTorch-style layout [N, L, H, C] and the MLX SDPA expects [N, H, L, C],
so we permute around the call.
"""
from __future__ import annotations

import math
from typing import overload

import mlx.core as mx


@overload
def scaled_dot_product_attention(qkv: mx.array) -> mx.array: ...
@overload
def scaled_dot_product_attention(q: mx.array, kv: mx.array) -> mx.array: ...
@overload
def scaled_dot_product_attention(q: mx.array, k: mx.array, v: mx.array) -> mx.array: ...


def scaled_dot_product_attention(*args, **kwargs):
    keys = {1: ["qkv"], 2: ["q", "kv"], 3: ["q", "k", "v"]}
    n = len(args) + len(kwargs)
    assert n in keys, f"Invalid arg count: {n}"
    for k in keys[n][len(args):]:
        assert k in kwargs, f"Missing arg {k}"

    if n == 1:
        qkv = args[0] if args else kwargs["qkv"]
        # [N, L, 3, H, C] -> three [N, L, H, C]
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]
    elif n == 2:
        q = args[0] if len(args) > 0 else kwargs["q"]
        kv = args[1] if len(args) > 1 else kwargs["kv"]
        # q: [N, Lq, H, C], kv: [N, Lkv, 2, H, C]
        k = kv[:, :, 0]
        v = kv[:, :, 1]
    else:
        q = args[0] if len(args) > 0 else kwargs["q"]
        k = args[1] if len(args) > 1 else kwargs["k"]
        v = args[2] if len(args) > 2 else kwargs["v"]

    # [N, L, H, C] -> [N, H, L, C]
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    scale = 1.0 / math.sqrt(q.shape[-1])
    # Use the fused fast SDPA if available, fall back to manual.
    if hasattr(mx, "fast") and hasattr(mx.fast, "scaled_dot_product_attention"):
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    else:
        attn = (q @ k.swapaxes(-2, -1)) * scale
        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(q.dtype)
        out = attn @ v
    return out.transpose(0, 2, 1, 3)  # back to [N, L, H, C]

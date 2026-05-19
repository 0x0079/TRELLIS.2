"""
Sparse / varlen attention on MLX.

We pad each per-batch (or per-window) varlen sequence into a dense block, run the
standard SDPA with an additive mask, and unpad. This is the equivalent of
flash-attn's varlen API — slower but portable.

Public entrypoints:
- sparse_scaled_dot_product_attention(qkv|q,kv|q,k,v) for full attention
- sparse_windowed_scaled_dot_product_self_attention for windowed self-attention
- SparseRotaryPositionEmbedder applies RoPE on a SparseTensor's feats
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .sparse_tensor import SparseTensor, VarLenTensor
from .rope import RotaryPositionEmbedder


__all__ = [
    "sparse_scaled_dot_product_attention",
    "sparse_windowed_scaled_dot_product_self_attention",
    "SparseRotaryPositionEmbedder",
]


def _pad_groups(feats: mx.array, seqlens: List[int]) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Pack [T, ...] -> dense [G, Lmax, ...] with bool mask [G, Lmax] and the
    original row index for un-padding.
    """
    G = len(seqlens)
    Lmax = max(seqlens) if seqlens else 0
    T = feats.shape[0]
    pad_shape = (G, Lmax) + tuple(feats.shape[1:])
    padded = mx.zeros(pad_shape, dtype=feats.dtype)
    mask = mx.zeros((G, Lmax), dtype=mx.bool_)
    start = 0
    for g, l in enumerate(seqlens):
        if l == 0:
            continue
        padded[g, :l] = feats[start : start + l]
        mask[g, :l] = True
        start += l
    return padded, mask, mx.array([Lmax] * G, dtype=mx.int32)


def _unpad_groups(padded: mx.array, seqlens: List[int], T: int) -> mx.array:
    out = mx.zeros((T,) + tuple(padded.shape[2:]), dtype=padded.dtype)
    start = 0
    for g, l in enumerate(seqlens):
        if l == 0:
            continue
        out[start : start + l] = padded[g, :l]
        start += l
    return out


def _dense_sdpa_padded(
    q: mx.array, k: mx.array, v: mx.array, q_mask: mx.array, kv_mask: mx.array
) -> mx.array:
    """
    q: [G, Lq, H, C], k,v: [G, Lk, H, C], masks bool [G, Lq], [G, Lk].
    Returns [G, Lq, H, C].
    """
    qh = q.transpose(0, 2, 1, 3)  # [G, H, Lq, C]
    kh = k.transpose(0, 2, 1, 3)
    vh = v.transpose(0, 2, 1, 3)
    scale = 1.0 / math.sqrt(q.shape[-1])
    attn = (qh @ kh.swapaxes(-2, -1)) * scale  # [G, H, Lq, Lk]
    neg_inf = mx.array(-1e30, dtype=attn.dtype)
    # Mask out invalid KV columns (per group).
    kv_bcast = kv_mask[:, None, None, :]  # [G, 1, 1, Lk]
    attn = mx.where(kv_bcast, attn, neg_inf)
    attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(q.dtype)
    out = attn @ vh  # [G, H, Lq, C]
    out = out.transpose(0, 2, 1, 3)  # [G, Lq, H, C]
    return out


def sparse_scaled_dot_product_attention(*args, **kwargs):
    """
    Varlen attention. Mirrors trellis2/modules/sparse/attention/full_attn.py.
    Each branch groups Q (and K/V) by batch and applies per-batch dense attention.
    """
    keys = {1: ["qkv"], 2: ["q", "kv"], 3: ["q", "k", "v"]}
    n = len(args) + len(kwargs)
    assert n in keys

    if n == 1:
        qkv = args[0] if args else kwargs["qkv"]
        assert isinstance(qkv, VarLenTensor) and qkv.shape[1] == 3
        q_layout = qkv.layout
        q_seqlens = [s.stop - s.start for s in q_layout]
        kv_seqlens = q_seqlens
        # Pad each batch entry
        feats = qkv.feats  # [T, 3, H, C]
        padded, mask, _ = _pad_groups(feats, q_seqlens)
        q = padded[:, :, 0]
        k = padded[:, :, 1]
        v = padded[:, :, 2]
        out = _dense_sdpa_padded(q, k, v, mask, mask)  # [G, L, H, C]
        out_flat = _unpad_groups(out, q_seqlens, feats.shape[0])
        return qkv.replace(out_flat)

    if n == 2:
        q = args[0] if len(args) > 0 else kwargs["q"]
        kv = args[1] if len(args) > 1 else kwargs["kv"]
        # Allow q or kv to be dense [N, L, H, C] / [N, L, 2, H, C].
        q_vl = q if isinstance(q, VarLenTensor) else None
        kv_vl = kv if isinstance(kv, VarLenTensor) else None

        if q_vl is not None:
            q_seqlens = [s.stop - s.start for s in q_vl.layout]
            q_feats = q_vl.feats  # [Tq, H, C]
        else:
            N, L = q.shape[0], q.shape[1]
            q_seqlens = [L] * N
            q_feats = q.reshape(N * L, *q.shape[2:])

        if kv_vl is not None:
            kv_seqlens = [s.stop - s.start for s in kv_vl.layout]
            kv_feats = kv_vl.feats  # [Tk, 2, H, C]
            kv_padded, kv_mask, _ = _pad_groups(kv_feats, kv_seqlens)
            k_pad = kv_padded[:, :, 0]
            v_pad = kv_padded[:, :, 1]
        else:
            N, L = kv.shape[0], kv.shape[1]
            kv_seqlens = [L] * N
            k_pad = kv[:, :, 0]
            v_pad = kv[:, :, 1]
            kv_mask = mx.ones((N, L), dtype=mx.bool_)

        q_padded, q_mask, _ = _pad_groups(q_feats, q_seqlens)
        out = _dense_sdpa_padded(q_padded, k_pad, v_pad, q_mask, kv_mask)
        out_flat = _unpad_groups(out, q_seqlens, q_feats.shape[0])
        if q_vl is not None:
            return q_vl.replace(out_flat)
        N = q.shape[0]
        L = q.shape[1]
        return out_flat.reshape(N, L, *out_flat.shape[1:])

    # n == 3: separate q, k, v
    q = args[0] if len(args) > 0 else kwargs["q"]
    k = args[1] if len(args) > 1 else kwargs["k"]
    v = args[2] if len(args) > 2 else kwargs["v"]
    q_vl = q if isinstance(q, VarLenTensor) else None
    k_vl = k if isinstance(k, VarLenTensor) else None

    if q_vl is not None:
        q_seqlens = [s.stop - s.start for s in q_vl.layout]
        q_feats = q_vl.feats
    else:
        N, L = q.shape[0], q.shape[1]
        q_seqlens = [L] * N
        q_feats = q.reshape(N * L, *q.shape[2:])
    if k_vl is not None:
        kv_seqlens = [s.stop - s.start for s in k_vl.layout]
        k_feats = k_vl.feats
        v_feats = v.feats
        k_pad, kv_mask, _ = _pad_groups(k_feats, kv_seqlens)
        v_pad, _, _ = _pad_groups(v_feats, kv_seqlens)
    else:
        N, L = k.shape[0], k.shape[1]
        kv_seqlens = [L] * N
        k_pad = k
        v_pad = v
        kv_mask = mx.ones((N, L), dtype=mx.bool_)
    q_padded, q_mask, _ = _pad_groups(q_feats, q_seqlens)
    out = _dense_sdpa_padded(q_padded, k_pad, v_pad, q_mask, kv_mask)
    out_flat = _unpad_groups(out, q_seqlens, q_feats.shape[0])
    if q_vl is not None:
        return q_vl.replace(out_flat)
    N = q.shape[0]
    L = q.shape[1]
    return out_flat.reshape(N, L, *out_flat.shape[1:])


# ---------------------------------------------------------------------------
# Windowed self-attention
# ---------------------------------------------------------------------------


def _window_partition(
    coords_np: np.ndarray,
    spatial_shape: Tuple[int, int, int],
    window_size: Tuple[int, int, int],
    shift: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    shifted = coords_np.copy()
    shifted[:, 1] += shift[0]
    shifted[:, 2] += shift[1]
    shifted[:, 3] += shift[2]
    max_coords = [spatial_shape[i] + shift[i] for i in range(3)]
    num_windows = [int(math.ceil((mc + 1) / ws)) for mc, ws in zip(max_coords, window_size)]
    shifted[:, 1] //= window_size[0]
    shifted[:, 2] //= window_size[1]
    shifted[:, 3] //= window_size[2]

    # Composite window id; include batch dim.
    win_id = (
        shifted[:, 0].astype(np.int64) * (num_windows[0] * num_windows[1] * num_windows[2])
        + shifted[:, 1] * (num_windows[1] * num_windows[2])
        + shifted[:, 2] * num_windows[2]
        + shifted[:, 3]
    )
    fwd = np.argsort(win_id, kind="stable")
    bwd = np.empty_like(fwd)
    bwd[fwd] = np.arange(fwd.shape[0])
    sorted_ids = win_id[fwd]
    if sorted_ids.shape[0] == 0:
        return fwd.astype(np.int32), bwd.astype(np.int32), []
    boundaries = np.concatenate(([True], sorted_ids[1:] != sorted_ids[:-1]))
    starts = np.where(boundaries)[0]
    seq_lens = np.diff(np.concatenate([starts, [sorted_ids.shape[0]]])).astype(np.int64)
    return fwd.astype(np.int32), bwd.astype(np.int32), seq_lens.tolist()


def sparse_windowed_scaled_dot_product_self_attention(
    qkv: SparseTensor,
    window_size: Union[int, Tuple[int, int, int]],
    shift_window: Tuple[int, int, int] = (0, 0, 0),
) -> SparseTensor:
    if isinstance(window_size, int):
        window_size = (window_size, window_size, window_size)

    cache_key = f"windowed_attn_{window_size}_{shift_window}"
    cache = qkv.get_spatial_cache(cache_key)
    if cache is None:
        fwd_np, bwd_np, seq_lens = _window_partition(
            np.asarray(qkv.coords, dtype=np.int32),
            qkv.spatial_shape,
            window_size,
            shift_window,
        )
        fwd = mx.array(fwd_np)
        bwd = mx.array(bwd_np)
        qkv.register_spatial_cache(cache_key, (fwd, bwd, seq_lens))
    else:
        fwd, bwd, seq_lens = cache

    qkv_feats = qkv.feats[fwd]  # [T, 3, H, C]
    padded, mask, _ = _pad_groups(qkv_feats, seq_lens)
    q = padded[:, :, 0]
    k = padded[:, :, 1]
    v = padded[:, :, 2]
    out = _dense_sdpa_padded(q, k, v, mask, mask)  # [G, L, H, C]
    out_flat = _unpad_groups(out, seq_lens, qkv_feats.shape[0])
    out_back = out_flat[bwd]
    return qkv.replace(out_back)


class SparseRotaryPositionEmbedder(nn.Module):
    """RoPE on a SparseTensor's feats using its (x,y,z) coords."""

    def __init__(
        self,
        head_dim: int,
        dim: int = 3,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
    ):
        super().__init__()
        self.head_dim = head_dim
        self.dim = dim
        self.rope_freq = rope_freq
        self._rope = RotaryPositionEmbedder(head_dim, dim, rope_freq)

    def __call__(self, q: SparseTensor, k: Optional[SparseTensor] = None):
        cache_name = f"rope_phases_{self.dim}d_freq{self.rope_freq[0]}-{self.rope_freq[1]}_hd{self.head_dim}"
        phases = q.get_spatial_cache(cache_name)
        if phases is None:
            coords = q.coords[:, 1:]  # [N, 3]
            phases = self._rope(coords)  # [N, head_dim//2, 2]
            q.register_spatial_cache(cache_name, phases)

        q_emb = q.replace(self._rope.apply_rotary_embedding(q.feats, phases))
        if k is None:
            return q_emb
        k_emb = k.replace(self._rope.apply_rotary_embedding(k.feats, phases))
        return q_emb, k_emb

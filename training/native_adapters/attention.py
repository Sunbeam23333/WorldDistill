"""Autograd-preserving SDPA implementation of the native BLHD attention API.

Unlike inference-only fused kernels this operator retains all input gradients.
Padding, bottom-right causal alignment and local windows follow FlashAttention
semantics; no low-precision cast or padding-mask omission is implicit.
"""

import torch
import torch.nn.functional as F


def flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None,
                    q_scale=None, causal=False, window_size=(-1, -1), deterministic=False,
                    dtype=None, version=None, fa_version=None):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or k.shape[:3] != v.shape[:3]:
        raise ValueError("Native attention expects aligned B,L,H,D tensors")
    b, qlen, heads, dim = q.shape
    klen = k.shape[1]
    if k.shape[0] != b or k.shape[-1] != dim or k.shape[2] != heads:
        raise ValueError("Native SDPA requires matching batch/head/query-key dimensions")
    q_lengths = torch.full((b,), qlen, device=q.device) if q_lens is None else torch.as_tensor(q_lens, device=q.device)
    k_lengths = torch.full((b,), klen, device=q.device) if k_lens is None else torch.as_tensor(k_lens, device=q.device)
    if q_lengths.shape != (b,) or k_lengths.shape != (b,):
        raise ValueError("Attention lengths must have one entry per batch sample")
    if (q_lengths < 0).any() or (q_lengths > qlen).any() or (k_lengths < 0).any() or (k_lengths > klen).any():
        raise ValueError("Attention lengths exceed their padded tensors")
    left, right = window_size
    full = bool((q_lengths == qlen).all() and (k_lengths == klen).all())
    use_builtin_causal = causal and qlen == klen and full and left < 0 and right < 0
    valid = None
    if not (full and left < 0 and right < 0 and (not causal or use_builtin_causal)):
        qi = torch.arange(qlen, device=q.device)[None, :, None]
        ki = torch.arange(klen, device=q.device)[None, None, :]
        valid = (qi < q_lengths[:, None, None]) & (ki < k_lengths[:, None, None])
        aligned_q = qi + (k_lengths - q_lengths)[:, None, None]
        if causal:
            valid = valid & (ki <= aligned_q)
        if left >= 0:
            valid = valid & (ki >= aligned_q - left)
        if right >= 0:
            valid = valid & (ki <= aligned_q + right)
        valid = valid[:, None]
    query = q.transpose(1, 2).to(v.dtype)
    if q_scale is not None:
        query = query * q_scale
    result = F.scaled_dot_product_attention(query, k.transpose(1, 2).to(v.dtype), v.transpose(1, 2),
        attn_mask=valid, dropout_p=dropout_p, scale=softmax_scale, is_causal=use_builtin_causal)
    return result.transpose(1, 2).contiguous()


attention = flash_attention


def flash_attn_func(q, k, v, dropout_p=0., softmax_scale=None, causal=False,
                    window_size=(-1, -1), deterministic=False, **kwargs):
    if kwargs:
        raise ValueError(f"Unsupported native attention options: {sorted(kwargs)}")
    return flash_attention(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale,
                           causal=causal, window_size=window_size, deterministic=deterministic)


def flex_attention(query, key, value, *, block_mask, **kwargs):
    """Use native CUDA FlexAttention or the identical mask with CPU SDPA."""
    if query.device.type == "cuda" and torch.cuda.get_device_capability(query.device)[0] >= 8:
        from torch.nn.attention.flex_attention import flex_attention as native_flex_attention
        return native_flex_attention(query, key, value, block_mask=block_mask, **kwargs)
    if kwargs:
        raise ValueError(f"Unsupported reference FlexAttention options: {sorted(kwargs)}")
    b, h, qlen, _ = query.shape
    mask = block_mask.mask_mod(
        torch.arange(b, device=query.device)[:, None, None, None],
        torch.arange(h, device=query.device)[None, :, None, None],
        torch.arange(qlen, device=query.device)[None, None, :, None],
        torch.arange(key.shape[2], device=query.device)[None, None, None, :])
    return F.scaled_dot_product_attention(query, key, value, attn_mask=mask)

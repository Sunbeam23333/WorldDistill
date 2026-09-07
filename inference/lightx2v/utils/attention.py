"""Shared dense-attention contract, independent of optional CUDA extensions.

Inputs use [tokens, heads, dim] or [batch, tokens, heads, dim]. Causal masks
are bottom-right aligned, including cached decoding with unequal Q/K lengths,
matching FlashAttention. Public ``attention`` preserves the input rank.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F

from cuda_compat import DENSE_ATTENTION_BACKENDS, import_attention_callable, resolve_attention_config


def validate_qkv(q, k, v):
    if not all(isinstance(x, torch.Tensor) for x in (q, k, v)):
        raise ValueError("q/k/v must be tensors")
    if len({q.ndim, k.ndim, v.ndim}) != 1 or q.ndim not in (3, 4):
        raise ValueError("q/k/v must have the same rank (3-D or 4-D)")
    if len({q.device, k.device, v.device}) != 1 or len({q.dtype, k.dtype, v.dtype}) != 1:
        raise ValueError("q/k/v must have matching devices and dtypes")
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError("q/k head dimensions and k/v head counts must match")
    if k.shape[-2] == 0 or q.shape[-2] % k.shape[-2]:
        raise ValueError("query heads must be divisible by key/value heads")
    if k.shape[-3] != v.shape[-3]:
        raise ValueError("k/v must contain the same number of tokens")
    if q.ndim == 4 and len({q.shape[0], k.shape[0], v.shape[0]}) != 1:
        raise ValueError("Batched q/k/v must have the same batch size")


def _offset_values(offsets, total, name):
    if not isinstance(offsets, torch.Tensor) or offsets.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional tensor")
    if offsets.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"{name} must use an integer dtype")
    values = offsets.detach().cpu().tolist()
    if len(values) < 2 or values[0] != 0 or values[-1] != total:
        raise ValueError(f"{name} must start at 0 and end at {total}; got {values}")
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def pack_qkv(q, k, v, cu_q=None, cu_k=None, max_q=None, max_k=None):
    """Validate segment boundaries and supply missing dense-batch metadata."""
    validate_qkv(q, k, v)
    if (cu_q is None) != (cu_k is None):
        raise ValueError("cu_seqlens_q and cu_seqlens_kv must be provided together")
    batch = q.shape[0] if q.ndim == 4 else 1
    expected_q = [i * q.shape[-3] for i in range(batch + 1)]
    expected_k = [i * k.shape[-3] for i in range(batch + 1)]
    original_rank = q.ndim
    q, k, v = (x.reshape(-1, x.shape[-2], x.shape[-1]) for x in (q, k, v))
    if cu_q is None:
        q_offsets, k_offsets = expected_q, expected_k
        if q.shape[0] == 0 or k.shape[0] == 0:
            raise ValueError("Attention segments must not be empty")
    else:
        q_offsets = _offset_values(cu_q, q.shape[0], "cu_seqlens_q")
        k_offsets = _offset_values(cu_k, k.shape[0], "cu_seqlens_kv")
        if len(q_offsets) != len(k_offsets):
            raise ValueError("Cumulative lengths must describe the same number of segments")
        if original_rank == 4 and (q_offsets != expected_q or k_offsets != expected_k):
            raise ValueError("4-D cumulative lengths must preserve every batch boundary")
    actual_q = max(b - a for a, b in zip(q_offsets, q_offsets[1:]))
    actual_k = max(b - a for a, b in zip(k_offsets, k_offsets[1:]))
    if max_q is not None and int(max_q) < actual_q:
        raise ValueError("max_seqlen_q is smaller than the packed maximum")
    if max_k is not None and int(max_k) < actual_k:
        raise ValueError("max_seqlen_kv is smaller than the packed maximum")
    cu_q = torch.tensor(q_offsets, dtype=torch.int32, device=q.device)
    cu_k = torch.tensor(k_offsets, dtype=torch.int32, device=q.device)
    return q, k, v, cu_q, cu_k, actual_q, actual_k


def native_attention(q, k, v, *, causal=False, dropout_p=0.0, softmax_scale=None, attn_mask=None):
    """Torch SDPA with explicit GQA and FlashAttention-compatible causality."""
    validate_qkv(q, k, v)
    original_rank = q.ndim
    if original_rank == 3:
        q, k, v = (x.unsqueeze(0) for x in (q, k, v))
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    if q.shape[1] != k.shape[1]:
        repeats = q.shape[1] // k.shape[1]
        k, v = (x.repeat_interleave(repeats, dim=1) for x in (k, v))
    mask = attn_mask
    if mask is not None and mask.dtype != torch.bool:
        mask = mask.to(q.dtype)
    use_native_causal = causal and q.shape[-2] == k.shape[-2] and mask is None
    if causal and not use_native_causal:
        nq, nk = q.shape[-2], k.shape[-2]
        allowed = torch.arange(nk, device=q.device)[None, :] <= (
            torch.arange(nq, device=q.device)[:, None] + nk - nq
        )
        if mask is None:
            mask = allowed
        elif mask.dtype == torch.bool:
            mask = mask & allowed
        else:
            mask = mask.masked_fill(~allowed, float("-inf"))
    options = {} if softmax_scale is None else {"scale": float(softmax_scale)}
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
        dropout_p=float(dropout_p), is_causal=use_native_causal, **options).transpose(1, 2)
    return out.squeeze(0) if original_rank == 3 else out


@lru_cache(maxsize=64)
def _resolved_backend(device_index, requested, strict):
    with torch.cuda.device(device_index):
        config = {"attn_type": requested, "strict_cuda_backend": strict}
        resolve_attention_config(config, torch_module=torch)
    return config["attn_type"]


def attention(q, k, v, *, config=None, backend=None, cu_seqlens_q=None,
              cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None,
              causal=False, dropout_p=0.0, softmax_scale=None, attn_mask=None,
              deterministic=False):
    """Resolve dense model-internal calls instead of hardcoding FA2/FA3.

    Sparse algorithms are not approximated here: this function is only for
    genuinely dense sublayers (text/action/audio adapters). A sparse outer
    layer's config therefore uses its dense parallel backend or ``auto``.
    """
    validate_qkv(q, k, v)
    config = config or {}
    requested = backend or config.get("attn_type", "auto")
    if requested not in DENSE_ATTENTION_BACKENDS:
        requested = config.get("parallel", {}).get("seq_p_attn_type", "auto")
    if requested not in DENSE_ATTENTION_BACKENDS:
        requested = "auto"
    strict = bool(config.get("strict_cuda_backend", False))
    selected = "torch_sdpa"
    if q.is_cuda and requested != "torch_sdpa":
        selected = _resolved_backend(q.device.index, requested, strict)
    unsupported = attn_mask is not None or q.dtype not in (torch.float16, torch.bfloat16)
    unsupported |= selected in ("flash_attn3", "flash_attn4", "sage_attn2", "sage_attn3") and dropout_p != 0
    unsupported |= selected.startswith("sage_") and (softmax_scale is not None or (causal and q.shape[-3] != k.shape[-3]))
    unsupported |= selected.startswith("sage_") and causal and cu_seqlens_q is not None
    # Sage is inference-only; automatic FA4 admission is forward-only. Neither
    # is silently treated as a verified differentiable training backend.
    unsupported |= selected in ("flash_attn4", "sage_attn2", "sage_attn3") and torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v))
    unsupported |= selected != "torch_sdpa" and (q.shape[-1] != v.shape[-1] or q.shape[-1] > 256 or q.shape[-1] % 8 != 0)
    if selected != "torch_sdpa" and unsupported:
        if strict:
            raise ValueError(f"{selected} cannot preserve this attention shape/dtype/mask/dropout/scale/gradient contract")
        selected = "torch_sdpa"
    if selected != "torch_sdpa":
        # Registry import is delayed so CPU references have no optional-kernel
        # dependency, and model paths use the same adapters as outer layers.
        from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER
        result = ATTN_WEIGHT_REGISTER[selected]().apply(q, k, v,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_k,
            causal=causal, drop_rate=dropout_p, softmax_scale=softmax_scale,
            deterministic=deterministic)
        return result.reshape(*q.shape[:-1], v.shape[-1])
    if cu_seqlens_q is None and cu_seqlens_k is None:
        return native_attention(q, k, v, causal=causal, dropout_p=dropout_p,
            softmax_scale=softmax_scale, attn_mask=attn_mask)
    original_shape = (*q.shape[:-1], v.shape[-1])
    q, k, v, cq, ck, _, _ = pack_qkv(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    qs, ks = cq.cpu().tolist(), ck.cpu().tolist()
    if attn_mask is not None and len(qs) > 2:
        raise ValueError("A packed multi-segment attn_mask cannot be mapped safely")
    outputs = [native_attention(q[qa:qb], k[ka:kb], v[ka:kb], causal=causal,
        dropout_p=dropout_p, softmax_scale=softmax_scale, attn_mask=attn_mask)
        for qa, qb, ka, kb in zip(qs, qs[1:], ks, ks[1:])]
    return torch.cat(outputs).reshape(original_shape)


def attention_with_lse(q, k, v, *, backend="auto", strict=False, causal=False,
                       softmax_scale=None, query_chunk=128, key_chunk=512):
    """Exact output + log-sum-exp for ring merging, without private FA APIs.

    Native SDPA does not expose LSE. Its portable fallback therefore uses
    bounded-memory online softmax, not a full quadratic attention matrix.
    This is a correctness fallback, not a claim of optimized ring throughput.
    """
    validate_qkv(q, k, v)
    if q.ndim != 4:
        raise ValueError("Ring sub-attention requires batched 4-D q/k/v")
    selected = "torch_sdpa"
    if q.is_cuda and backend != "torch_sdpa":
        selected = _resolved_backend(q.device.index, backend, strict)
    if selected in ("flash_attn2", "flash_attn3", "flash_attn4") and q.dtype in (torch.float16, torch.bfloat16):
        fn = import_attention_callable(selected, dense=True)
        if not callable(fn):
            if strict:
                raise RuntimeError(f"{selected} has no callable dense LSE interface")
        else:
            option = {"return_lse": True} if selected == "flash_attn4" else {"return_attn_probs": True}
            result = fn(q.contiguous(), k.contiguous(), v.contiguous(),
                softmax_scale=softmax_scale, causal=causal, **option)
            if not isinstance(result, tuple) or len(result) < 2:
                raise RuntimeError(f"{selected} did not return attention output and LSE")
            out, lse = result[:2]
            if out.shape != (*q.shape[:-1], v.shape[-1]) or lse.shape != (q.shape[0], q.shape[2], q.shape[1]):
                raise RuntimeError(f"{selected} returned an invalid output/LSE layout")
            return out, lse.float()
    if query_chunk < 1 or key_chunk < 1:
        raise ValueError("Attention chunk sizes must be positive")
    qh, kh, vh = (x.transpose(1, 2).float() for x in (q, k, v))
    if qh.shape[1] != kh.shape[1]:
        repeats = qh.shape[1] // kh.shape[1]
        kh, vh = (x.repeat_interleave(repeats, dim=1) for x in (kh, vh))
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    outputs, lses = [], []
    nq, nk = q.shape[1], k.shape[1]
    for start in range(0, nq, query_chunk):
        qr = qh[:, :, start:start + query_chunk]
        shape = qr.shape[:-1]
        row_max = torch.full(shape, float("-inf"), device=q.device)
        denom = torch.zeros(shape, device=q.device)
        numerator = torch.zeros((*shape, v.shape[-1]), device=q.device)
        for ks in range(0, nk, key_chunk):
            scores = torch.matmul(qr, kh[:, :, ks:ks + key_chunk].transpose(-1, -2)) * scale
            if causal:
                allowed = torch.arange(ks, min(nk, ks + key_chunk), device=q.device)[None, :] <= (
                    torch.arange(start, start + qr.shape[-2], device=q.device)[:, None] + nk - nq)
                scores = scores.masked_fill(~allowed, float("-inf"))
            new_max = torch.maximum(row_max, scores.amax(dim=-1))
            safe_max = torch.where(torch.isfinite(new_max), new_max, 0.0)
            alpha = torch.exp(row_max - safe_max)
            probabilities = torch.exp(scores - safe_max.unsqueeze(-1))
            numerator = numerator * alpha.unsqueeze(-1) + probabilities @ vh[:, :, ks:ks + key_chunk]
            denom = denom * alpha + probabilities.sum(dim=-1)
            row_max = new_max
        outputs.append(numerator / denom.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1))
        lses.append(torch.where(denom > 0, row_max + denom.log(), float("-inf")))
    if not outputs:
        return q.new_empty((*q.shape[:-1], v.shape[-1])), q.new_empty((q.shape[0], q.shape[2], 0), dtype=torch.float32)
    return torch.cat(outputs, dim=2).transpose(1, 2).to(q.dtype), torch.cat(lses, dim=2)

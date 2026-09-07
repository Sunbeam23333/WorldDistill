import torch
import torch.nn.functional as F


def torch_sdpa_no_pad(qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False):
    """Native masked SDPA fallback matching the no-pad helpers' output contract."""

    if qkv.ndim != 5 or qkv.shape[2] != 3:
        raise ValueError("qkv must have shape [batch, tokens, 3, heads, head_dim]")
    batch_size, query_length, _, _, _ = qkv.shape
    query, key, value = qkv.unbind(dim=2)
    key_length = key.shape[1]

    attention_mask = None
    query_mask = None
    use_native_causal = causal
    if key_padding_mask is not None:
        if key_padding_mask.dtype != torch.bool:
            key_padding_mask = key_padding_mask.bool()
        if key_padding_mask.shape != (batch_size, key_length):
            raise ValueError(
                "key_padding_mask must have shape "
                f"[{batch_size}, {key_length}], got {tuple(key_padding_mask.shape)}"
            )
        attention_mask = key_padding_mask[:, None, None, :]
        query_mask = key_padding_mask[:, :, None, None]
        if causal:
            if query_length != key_length:
                raise ValueError("Masked causal SDPA requires equal query and key lengths")
            causal_mask = torch.ones(
                (query_length, key_length),
                dtype=torch.bool,
                device=qkv.device,
            ).tril()
            attention_mask = attention_mask & causal_mask[None, None, :, :]
            use_native_causal = False

    sdpa_kwargs = {}
    if softmax_scale is not None:
        sdpa_kwargs["scale"] = softmax_scale
    output = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=attention_mask,
        dropout_p=dropout_p,
        is_causal=use_native_causal,
        **sdpa_kwargs,
    ).transpose(1, 2)
    if query_mask is not None:
        output = output.masked_fill(~query_mask, 0)
    return output


def _backend_no_pad(qkv, key_padding_mask, backend, causal=False, dropout_p=0.0,
                    softmax_scale=None, deterministic=False):
    # Each unpadded row is a separate sequence. No optional FlashAttention
    # padding helpers and no concatenation across sample boundaries.
    from lightx2v.utils.attention import attention

    if qkv.ndim != 5 or qkv.shape[2] != 3:
        raise ValueError("qkv must have shape [batch, tokens, 3, heads, head_dim]")
    batch, tokens = qkv.shape[:2]
    if key_padding_mask is None:
        key_padding_mask = torch.ones(batch, tokens, dtype=torch.bool, device=qkv.device)
    if key_padding_mask.shape != (batch, tokens):
        raise ValueError("key_padding_mask must match qkv batch and tokens")
    mask = key_padding_mask.to(device=qkv.device, dtype=torch.bool)
    output = torch.zeros_like(qkv[:, :, 0])
    for index in range(batch):
        positions = mask[index].nonzero().flatten()
        if positions.numel() == 0:
            continue
        q, k, v = qkv[index].index_select(0, positions).unbind(dim=1)
        result = attention(q, k, v, backend=backend, causal=causal,
            dropout_p=dropout_p, softmax_scale=softmax_scale, deterministic=deterministic)
        output[index].index_copy_(0, positions, result)
    return output


def flash_attn_no_pad(qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False):
    return _backend_no_pad(qkv, key_padding_mask, "flash_attn2", causal, dropout_p, softmax_scale, deterministic)


def flash_attn_no_pad_v3(qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False):
    return _backend_no_pad(qkv, key_padding_mask, "flash_attn3", causal, dropout_p, softmax_scale, deterministic)


def flash_attn_no_pad_v4(qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False):
    return _backend_no_pad(qkv, key_padding_mask, "flash_attn4", causal, dropout_p, softmax_scale, deterministic)


def sage_attn_no_pad_v2(qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False):
    return _backend_no_pad(qkv, key_padding_mask, "sage_attn2", causal, dropout_p, softmax_scale, deterministic)


def sage_attn_no_pad_v3(qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False):
    return _backend_no_pad(qkv, key_padding_mask, "sage_attn3", causal, dropout_p, softmax_scale, deterministic)

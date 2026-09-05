import torch
import torch.nn.functional as F

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate


@ATTN_WEIGHT_REGISTER("torch_sdpa")
class TorchSDPAWeight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(
        self,
        q,
        k,
        v,
        drop_rate=0,
        attn_mask=None,
        causal=False,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **kwargs,
    ):
        if (cu_seqlens_q is None) != (cu_seqlens_kv is None):
            raise ValueError("cu_seqlens_q and cu_seqlens_kv must be provided together")

        if cu_seqlens_q is not None:
            return self._apply_varlen(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                drop_rate=drop_rate,
                attn_mask=attn_mask,
                causal=causal,
            )

        ranks = {q.ndim, k.ndim, v.ndim}
        if len(ranks) != 1 or q.ndim not in (3, 4):
            raise ValueError(
                "Torch SDPA expects q/k/v to have the same rank and shape "
                "[tokens, heads, head_dim] or [batch, tokens, heads, head_dim]"
            )
        if q.ndim == 3:
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        elif q.shape[0] != k.shape[0] or k.shape[0] != v.shape[0]:
            raise ValueError("Batched q/k/v must have the same batch size")
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal)
        x = x.transpose(1, 2)
        b, s, a, d = x.shape
        # Match the FlashAttention adapters: every call returns a packed
        # [total_query_tokens, hidden_dim] tensor, including dense 4-D input.
        return x.reshape(b * s, a * d)

    @staticmethod
    def _validated_offsets(offsets, *, total_tokens, name):
        if not isinstance(offsets, torch.Tensor) or offsets.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional tensor")
        values = [int(value) for value in offsets.detach().cpu().tolist()]
        if len(values) < 2 or values[0] != 0 or values[-1] != total_tokens:
            raise ValueError(
                f"{name} must start at 0 and end at {total_tokens}; got {values}"
            )
        if any(end <= start for start, end in zip(values, values[1:])):
            raise ValueError(f"{name} must be strictly increasing; got {values}")
        return values

    def _apply_varlen(
        self,
        q,
        k,
        v,
        *,
        cu_seqlens_q,
        cu_seqlens_kv,
        drop_rate,
        attn_mask,
        causal,
    ):
        """Evaluate packed variable-length attention without cross-segment leakage."""

        ranks = {q.ndim, k.ndim, v.ndim}
        if len(ranks) != 1 or q.ndim not in (3, 4):
            raise ValueError(
                "Torch SDPA variable-length fallback expects q/k/v to have the "
                "same rank and shape [total_tokens, heads, head_dim] or "
                "[batch, tokens, heads, head_dim]"
            )

        q_batch_shape = None
        kv_batch_shape = None
        if q.ndim == 4:
            if q.shape[0] != k.shape[0] or k.shape[0] != v.shape[0]:
                raise ValueError("Batched q/k/v must have the same batch size")
            q_batch_shape = q.shape[:2]
            kv_batch_shape = k.shape[:2]
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])

        if k.shape[0] != v.shape[0]:
            raise ValueError("Packed k and v must contain the same number of tokens")

        q_offsets = self._validated_offsets(
            cu_seqlens_q,
            total_tokens=q.shape[0],
            name="cu_seqlens_q",
        )
        kv_offsets = self._validated_offsets(
            cu_seqlens_kv,
            total_tokens=k.shape[0],
            name="cu_seqlens_kv",
        )
        if len(q_offsets) != len(kv_offsets):
            raise ValueError(
                "cu_seqlens_q and cu_seqlens_kv must describe the same number of segments"
            )
        if q_batch_shape is not None:
            q_batch, q_tokens = q_batch_shape
            kv_batch, kv_tokens = kv_batch_shape
            expected_q = [index * q_tokens for index in range(q_batch + 1)]
            expected_kv = [index * kv_tokens for index in range(kv_batch + 1)]
            if q_offsets != expected_q or kv_offsets != expected_kv:
                raise ValueError(
                    "For 4-D q/k/v, cumulative lengths must preserve every batch "
                    f"boundary; expected q={expected_q}, kv={expected_kv}, got "
                    f"q={q_offsets}, kv={kv_offsets}"
                )
        if attn_mask is not None and len(q_offsets) > 2:
            raise ValueError(
                "A packed multi-segment attn_mask cannot be mapped safely by the Torch SDPA "
                "fallback; pass per-segment inputs or omit the mask."
            )

        outputs = []
        for q_start, q_end, kv_start, kv_end in zip(
            q_offsets,
            q_offsets[1:],
            kv_offsets,
            kv_offsets[1:],
        ):
            q_segment = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
            k_segment = k[kv_start:kv_end].transpose(0, 1).unsqueeze(0)
            v_segment = v[kv_start:kv_end].transpose(0, 1).unsqueeze(0)
            mask = attn_mask
            if mask is not None and mask.dtype != torch.bool:
                mask = mask.to(q_segment.dtype)
            segment = F.scaled_dot_product_attention(
                q_segment,
                k_segment,
                v_segment,
                attn_mask=mask,
                dropout_p=drop_rate,
                is_causal=causal,
            )
            outputs.append(segment.squeeze(0).transpose(0, 1).reshape(q_end - q_start, -1))
        return torch.cat(outputs, dim=0)

import torch
from loguru import logger

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate

capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
if capability == (8, 9):
    try:
        from sageattention import sageattn_qk_int8_pv_fp16_triton as sageattn
    except (ImportError, OSError, RuntimeError):
        logger.info("sageattn not found, please install sageattention first")
        sageattn = None
else:
    try:
        from sageattention import sageattn
    except (ImportError, OSError, RuntimeError):
        logger.info("sageattn not found, please install sageattention first")
        sageattn = None

try:
    from sageattn3 import sageattn3_blackwell
except (ImportError, OSError, RuntimeError):
    logger.info("sageattn3 not found, please install sageattention first")
    sageattn3_blackwell = None


class _SageAttnWeightBase(AttnWeightTemplate):
    """Common dense/packed contract for SageAttention inference kernels.

    SageAttention 2/3 expose dense batched kernels, not a cumulative-length
    API.  Packed inputs therefore have to be split at every declared boundary;
    treating the packed token axis as one sequence silently mixes samples.
    """

    backend_name = "sage_attn"

    @staticmethod
    def _kernel():
        raise NotImplementedError

    @staticmethod
    def _call_kernel(kernel, q, k, v, *, causal):
        raise NotImplementedError

    def __init__(self):
        if not callable(self._kernel()):
            raise RuntimeError(
                f"{self.backend_name} was selected, but its SageAttention callable "
                "is unavailable. Use the CUDA compatibility resolver or select "
                "torch_sdpa."
            )
        self.config = {}

    @staticmethod
    def _validate_qkv(q, k, v):
        if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
            raise ValueError("SageAttention expects q/k/v to be torch tensors")
        ranks = {q.ndim, k.ndim, v.ndim}
        if len(ranks) != 1 or q.ndim not in (3, 4):
            raise ValueError(
                "SageAttention expects q/k/v to have the same rank and shape "
                "[tokens, heads, head_dim] or [batch, tokens, heads, head_dim]"
            )
        if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
            raise ValueError("SageAttention q/k/v must have the same head dimension")
        if k.shape[-2] != v.shape[-2]:
            raise ValueError("SageAttention k/v must have the same number of heads")
        if q.shape[-2] % k.shape[-2] != 0:
            raise ValueError(
                "SageAttention query heads must be divisible by key/value heads"
            )
        sequence_axis = 0 if q.ndim == 3 else 1
        if k.shape[sequence_axis] != v.shape[sequence_axis]:
            raise ValueError("SageAttention k/v must contain the same number of tokens")
        if q.ndim == 4 and (
            q.shape[0] != k.shape[0] or k.shape[0] != v.shape[0]
        ):
            raise ValueError("Batched SageAttention q/k/v must have the same batch size")

    @staticmethod
    def _validated_offsets(offsets, *, total_tokens, name):
        if not isinstance(offsets, torch.Tensor) or offsets.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional tensor")
        if offsets.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError(f"{name} must use an integer dtype")
        values = [int(value) for value in offsets.detach().cpu().tolist()]
        if len(values) < 2 or values[0] != 0 or values[-1] != total_tokens:
            raise ValueError(
                f"{name} must start at 0 and end at {total_tokens}; got {values}"
            )
        if any(end <= start for start, end in zip(values, values[1:])):
            raise ValueError(f"{name} must be strictly increasing; got {values}")
        return values

    @staticmethod
    def _flatten_output(output, q):
        if not isinstance(output, torch.Tensor) or output.shape != q.shape:
            actual = getattr(output, "shape", type(output).__name__)
            raise RuntimeError(
                "SageAttention returned an unexpected output shape: "
                f"expected {tuple(q.shape)}, got {actual}"
            )
        batch, tokens, heads, head_dim = output.shape
        return output.reshape(batch * tokens, heads * head_dim)

    def _run_dense(self, q, k, v, *, causal):
        if q.ndim == 3:
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        if causal and q.shape[1] != k.shape[1]:
            raise ValueError(
                "Causal SageAttention requires equal query and key/value lengths"
            )
        output = self._call_kernel(
            self._kernel(),
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            causal=causal,
        )
        return self._flatten_output(output, q)

    def _run_varlen(
        self,
        q,
        k,
        v,
        *,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        causal,
    ):
        q_batch_shape = None
        kv_batch_shape = None
        if q.ndim == 4:
            q_batch_shape = q.shape[:2]
            kv_batch_shape = k.shape[:2]
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])

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

        q_lengths = [end - start for start, end in zip(q_offsets, q_offsets[1:])]
        kv_lengths = [
            end - start for start, end in zip(kv_offsets, kv_offsets[1:])
        ]
        if causal and q_lengths != kv_lengths:
            raise ValueError(
                "Causal SageAttention requires equal query and key/value lengths "
                "for every packed segment"
            )
        if max_seqlen_q is not None and int(max_seqlen_q) < max(q_lengths):
            raise ValueError(
                f"max_seqlen_q={max_seqlen_q} is smaller than the packed maximum "
                f"{max(q_lengths)}"
            )
        if max_seqlen_kv is not None and int(max_seqlen_kv) < max(kv_lengths):
            raise ValueError(
                f"max_seqlen_kv={max_seqlen_kv} is smaller than the packed maximum "
                f"{max(kv_lengths)}"
            )

        outputs = []
        for q_start, q_end, kv_start, kv_end in zip(
            q_offsets,
            q_offsets[1:],
            kv_offsets,
            kv_offsets[1:],
        ):
            # Each slice becomes its own batch item.  This is deliberately a
            # per-segment kernel call because SageAttention 2/3 do not expose a
            # cumulative-length interface equivalent to FlashAttention varlen.
            q_segment = q[q_start:q_end].unsqueeze(0)
            k_segment = k[kv_start:kv_end].unsqueeze(0)
            v_segment = v[kv_start:kv_end].unsqueeze(0)
            output = self._call_kernel(
                self._kernel(),
                q_segment.contiguous(),
                k_segment.contiguous(),
                v_segment.contiguous(),
                causal=causal,
            )
            outputs.append(self._flatten_output(output, q_segment))
        return torch.cat(outputs, dim=0)

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
        if drop_rate not in (0, 0.0):
            raise ValueError(
                "SageAttention inference kernels do not support attention dropout"
            )
        if attn_mask is not None:
            raise ValueError(
                "This SageAttention adapter cannot safely map a custom attention mask; "
                "select torch_sdpa instead"
            )
        if "is_causal" in kwargs:
            requested_is_causal = bool(kwargs["is_causal"])
            if bool(causal) and requested_is_causal != bool(causal):
                raise ValueError("causal and is_causal specify conflicting values")
            causal = requested_is_causal

        self._validate_qkv(q, k, v)
        if (cu_seqlens_q is None) != (cu_seqlens_kv is None):
            raise ValueError("cu_seqlens_q and cu_seqlens_kv must be provided together")
        if cu_seqlens_q is None:
            return self._run_dense(q, k, v, causal=bool(causal))
        return self._run_varlen(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            causal=bool(causal),
        )


@ATTN_WEIGHT_REGISTER("sage_attn2")
class SageAttn2Weight(_SageAttnWeightBase):
    backend_name = "sage_attn2"

    @staticmethod
    def _kernel():
        return sageattn

    @staticmethod
    def _call_kernel(kernel, q, k, v, *, causal):
        return kernel(
            q,
            k,
            v,
            tensor_layout="NHD",
            is_causal=causal,
        )


@ATTN_WEIGHT_REGISTER("sage_attn3")
class SageAttn3Weight(_SageAttnWeightBase):
    backend_name = "sage_attn3"

    @staticmethod
    def _kernel():
        return sageattn3_blackwell

    @staticmethod
    def _call_kernel(kernel, q, k, v, *, causal):
        return kernel(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=causal,
        ).transpose(1, 2)

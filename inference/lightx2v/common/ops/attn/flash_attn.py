from loguru import logger

try:
    import flash_attn  # noqa: F401
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except (ImportError, OSError, RuntimeError):
    logger.info("flash_attn_varlen_func not found, please install flash_attn2 first")
    flash_attn_varlen_func = None

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3
except (ImportError, OSError, RuntimeError):
    logger.info("flash_attn_varlen_func_v3 not found, please install flash_attn3 first")
    flash_attn_varlen_func_v3 = None

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate


@ATTN_WEIGHT_REGISTER("flash_attn2")
class FlashAttn2Weight(AttnWeightTemplate):
    def __init__(self):
        if not callable(flash_attn_varlen_func):
            raise RuntimeError(
                "flash_attn2 was selected, but flash_attn_varlen_func is unavailable. "
                "Use the CUDA compatibility resolver or select torch_sdpa."
            )
        self.config = {}

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **kwargs,
    ):
        ranks = {q.ndim, k.ndim, v.ndim}
        if len(ranks) != 1 or q.ndim not in (3, 4):
            raise ValueError("FlashAttention expects q/k/v with matching 3-D or 4-D ranks")
        if q.ndim == 4:
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])
        total_q = q.shape[0]
        x = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        ).reshape(total_q, -1)
        return x


@ATTN_WEIGHT_REGISTER("flash_attn3")
class FlashAttn3Weight(AttnWeightTemplate):
    def __init__(self):
        if not callable(flash_attn_varlen_func_v3):
            raise RuntimeError(
                "flash_attn3 was selected, but flash_attn_varlen_func_v3 is unavailable. "
                "Use the CUDA compatibility resolver or select torch_sdpa."
            )
        self.config = {}

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **kwargs,
    ):
        ranks = {q.ndim, k.ndim, v.ndim}
        if len(ranks) != 1 or q.ndim not in (3, 4):
            raise ValueError("FlashAttention expects q/k/v with matching 3-D or 4-D ranks")
        if q.ndim == 4:
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])
        total_q = q.shape[0]
        x = flash_attn_varlen_func_v3(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        ).reshape(total_q, -1)
        return x

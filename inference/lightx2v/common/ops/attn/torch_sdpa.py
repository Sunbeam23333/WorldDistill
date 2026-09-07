from lightx2v.utils.attention import attention
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate


@ATTN_WEIGHT_REGISTER("torch_sdpa")
class TorchSDPAWeight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(self, q, k, v, drop_rate=0, attn_mask=None, causal=False,
              cu_seqlens_q=None, cu_seqlens_kv=None, max_seqlen_q=None,
              max_seqlen_kv=None, softmax_scale=None, **kwargs):
        if "is_causal" in kwargs:
            if causal and not bool(kwargs["is_causal"]):
                raise ValueError("causal and is_causal specify conflicting values")
            causal = bool(kwargs["is_causal"])
        if "dropout_p" in kwargs:
            if drop_rate and float(kwargs["dropout_p"]) != float(drop_rate):
                raise ValueError("drop_rate and dropout_p specify conflicting values")
            drop_rate = float(kwargs["dropout_p"])
        if "scale" in kwargs:
            if softmax_scale is not None and kwargs["scale"] != softmax_scale:
                raise ValueError("scale and softmax_scale specify conflicting values")
            softmax_scale = kwargs["scale"]
        for name, neutral in (("window_size", (-1, -1)), ("softcap", 0.0), ("alibi_slopes", None)):
            if name in kwargs and kwargs[name] != neutral:
                raise ValueError(f"Torch SDPA adapter cannot silently discard {name}")
        result = attention(q, k, v, backend="torch_sdpa", causal=causal,
            dropout_p=drop_rate, softmax_scale=softmax_scale, attn_mask=attn_mask,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_kv)
        return result.reshape(-1, q.shape[-2] * v.shape[-1])

from cuda_compat import import_attention_callable
from lightx2v.utils.attention import pack_qkv
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate


flash_attn_varlen_func = import_attention_callable("flash_attn2")
flash_attn_varlen_func_v3 = import_attention_callable("flash_attn3")
flash_attn_varlen_func_v4 = import_attention_callable("flash_attn4")


class _FlashAttentionWeight(AttnWeightTemplate):
    backend_name = "flash_attn"

    def __init__(self):
        if not callable(self._kernel()):
            raise RuntimeError(f"{self.backend_name} was selected, but its callable is unavailable. "
                               "Use the CUDA compatibility resolver or select torch_sdpa.")
        self.config = {}

    def apply(self, q, k, v, cu_seqlens_q=None, cu_seqlens_kv=None,
              max_seqlen_q=None, max_seqlen_kv=None, causal=False, drop_rate=0.0,
              softmax_scale=None, deterministic=False, attn_mask=None, **kwargs):
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
        if attn_mask is not None:
            raise ValueError("FlashAttention cannot map a custom attn_mask; select torch_sdpa")
        if self.backend_name != "flash_attn2" and drop_rate:
            raise ValueError(f"{self.backend_name} adapter does not support dropout; select torch_sdpa")
        q, k, v, cq, ck, mq, mk = pack_qkv(q, k, v, cu_seqlens_q, cu_seqlens_kv,
                                         max_seqlen_q, max_seqlen_kv)
        options = dict(softmax_scale=softmax_scale, causal=bool(causal), deterministic=deterministic)
        if self.backend_name == "flash_attn2":
            options["dropout_p"] = float(drop_rate)
        for name in ("window_size", "softcap", "alibi_slopes"):
            if name in kwargs:
                if name == "alibi_slopes" and self.backend_name != "flash_attn2":
                    raise ValueError(f"{self.backend_name} adapter does not support alibi_slopes")
                options[name] = kwargs[name]
        if self.backend_name == "flash_attn4":
            # FA4 places qv before the varlen metadata; positional FA2/3 calls
            # would bind cumulative lengths to the wrong parameter.
            output = self._kernel()(q, k, v, cu_seqlens_q=cq, cu_seqlens_k=ck,
                                    max_seqlen_q=mq, max_seqlen_k=mk, **options)
        else:
            output = self._kernel()(q, k, v, cq, ck, mq, mk, **options)
        if output.shape != (*q.shape[:-1], v.shape[-1]):
            raise RuntimeError(f"{self.backend_name} returned an unexpected output shape")
        return output.reshape(q.shape[0], -1)


@ATTN_WEIGHT_REGISTER("flash_attn2")
class FlashAttn2Weight(_FlashAttentionWeight):
    backend_name = "flash_attn2"

    @staticmethod
    def _kernel():
        return flash_attn_varlen_func


@ATTN_WEIGHT_REGISTER("flash_attn3")
class FlashAttn3Weight(_FlashAttentionWeight):
    backend_name = "flash_attn3"

    @staticmethod
    def _kernel():
        return flash_attn_varlen_func_v3


@ATTN_WEIGHT_REGISTER("flash_attn4")
class FlashAttn4Weight(_FlashAttentionWeight):
    backend_name = "flash_attn4"

    @staticmethod
    def _kernel():
        return flash_attn_varlen_func_v4

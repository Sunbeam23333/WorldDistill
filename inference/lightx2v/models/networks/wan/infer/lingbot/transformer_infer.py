"""
LingBot Camera Control - Transformer Inference Module.

Extends WanTransformerInfer to inject camera control signal per-block:
  - cam_injector_layer1/2: MLP that processes cam_emb
  - cam_scale_layer / cam_shift_layer: modulate hidden states with camera signal

The injection follows the lingbot-world official implementation:
  After self-attention residual, before cross-attention:
    cam_feat = silu(cam_injector_layer1(cam_emb))
    cam_feat = cam_injector_layer2(cam_feat)
    cam_scale = cam_scale_layer(cam_feat)
    cam_shift = cam_shift_layer(cam_feat)
    x = x * (1 + cam_scale) + cam_shift
"""

import torch

from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
from lightx2v.utils.envs import *


class LingBotTransformerInfer(WanTransformerInfer):
    """Transformer inference with per-block camera control injection."""

    def __init__(self, config):
        super().__init__(config)

    def infer_block(self, block, x, pre_infer_out):
        """Override to inject camera control between self-attn and cross-attn."""
        if hasattr(block.compute_phases[0], "before_proj") and block.compute_phases[0].before_proj.weight is not None:
            x = block.compute_phases[0].before_proj.apply(x) + pre_infer_out.x

        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.pre_process(
            block.compute_phases[0].modulation,
            pre_infer_out.embed0,
        )
        y_out = self.infer_self_attn(
            block.compute_phases[0],
            x,
            shift_msa,
            scale_msa,
        )

        # ===== Camera control injection (after self-attn residual, before cross-attn) =====
        cam_emb = pre_infer_out.adapter_args.get("cam_emb", None)
        cam_tokens = pre_infer_out.adapter_args.get("cam_tokens", None)

        if cam_emb is not None and hasattr(block.compute_phases[0], "cam_injector_layer1"):
            # Apply self-attn residual first
            if self.sensitive_layer_dtype != self.infer_dtype:
                x = x.to(self.sensitive_layer_dtype) + y_out.to(self.sensitive_layer_dtype) * gate_msa.squeeze()
            else:
                x = x + y_out * gate_msa.squeeze()

            # Camera injection
            cam_phase = block.compute_phases[0]  # cam weights stored in self-attn phase
            cam_feat = cam_phase.cam_injector_layer1.apply(cam_emb)
            cam_feat = torch.nn.functional.silu(cam_feat)
            cam_feat = cam_phase.cam_injector_layer2.apply(cam_feat)

            cam_scale = cam_phase.cam_scale_layer.apply(cam_feat)
            cam_shift = cam_phase.cam_shift_layer.apply(cam_feat)

            # cam_tokens are per-frame tokens, need to broadcast to spatial dim
            # cam_emb: (num_frames, dim) -> need to expand to match x: (seq_len, dim)
            # grid_sizes: (t, h, w), seq_len = t * h * w
            t, h, w = pre_infer_out.grid_sizes.tuple
            # cam_scale/cam_shift: (num_frames, dim) -> (t, 1, 1, dim) -> (t*h*w, dim)
            if cam_scale.dim() == 2 and cam_scale.shape[0] == t:
                cam_scale = cam_scale.unsqueeze(1).unsqueeze(1).expand(t, h, w, -1).reshape(t * h * w, -1)
                cam_shift = cam_shift.unsqueeze(1).unsqueeze(1).expand(t, h, w, -1).reshape(t * h * w, -1)

            x = x * (1 + cam_scale) + cam_shift

            # Now do cross-attn (without re-adding y_out)
            norm3_out = block.compute_phases[1].norm3.apply(x)
            # Continue with cross-attn but skip the initial residual in infer_cross_attn
            x, attn_out = self._infer_cross_attn_no_residual(block.compute_phases[1], x, pre_infer_out.context, norm3_out)
        else:
            # Standard path: no camera control
            x, attn_out = self.infer_cross_attn(block.compute_phases[1], x, pre_infer_out.context, y_out, gate_msa)

        y = self.infer_ffn(block.compute_phases[2], x, attn_out, c_shift_msa, c_scale_msa)
        x = self.post_process(x, y, c_gate_msa, pre_infer_out)

        if hasattr(block.compute_phases[2], "after_proj"):
            pre_infer_out.adapter_args["hints"].append(block.compute_phases[2].after_proj.apply(x))

        if self.has_post_adapter:
            x = self.infer_post_adapter(block.compute_phases[3], x, pre_infer_out)

        return x

    def _infer_cross_attn_no_residual(self, phase, x, context, norm3_out):
        """Cross attention without the initial self-attn residual (already applied in cam injection)."""
        context_img = None  # lingbot doesn't use CLIP

        if self.sensitive_layer_dtype != self.infer_dtype:
            context = context.to(self.infer_dtype)

        n, d = self.num_heads, self.head_dim
        q = phase.cross_attn_norm_q.apply(phase.cross_attn_q.apply(norm3_out)).view(-1, n, d)
        k = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context)).view(-1, n, d)
        v = phase.cross_attn_v.apply(context).view(-1, n, d)

        if self.cross_attn_cu_seqlens_q is None:
            if self.cross_attn_1_type == "flash_attn2" or self.cross_attn_1_type == "flash_attn3":
                self.cross_attn_cu_seqlens_q = torch.tensor([0, q.shape[0]]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True)
            else:
                self.cross_attn_cu_seqlens_q = torch.tensor([0, q.shape[0]]).cumsum(0, dtype=torch.int32)
        if self.cross_attn_cu_seqlens_kv is None:
            if self.cross_attn_1_type == "flash_attn2" or self.cross_attn_1_type == "flash_attn3":
                self.cross_attn_cu_seqlens_kv = torch.tensor([0, k.shape[0]]).cumsum(0, dtype=torch.int32).to(k.device, non_blocking=True)
            else:
                self.cross_attn_cu_seqlens_kv = torch.tensor([0, k.shape[0]]).cumsum(0, dtype=torch.int32)

        attn_out = phase.cross_attn_1.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self.cross_attn_cu_seqlens_q,
            cu_seqlens_kv=self.cross_attn_cu_seqlens_kv,
            max_seqlen_q=q.size(0),
            max_seqlen_kv=k.size(0),
        )

        attn_out = phase.cross_attn_o.apply(attn_out)

        if self.clean_cuda_cache:
            del q, k, v, norm3_out, context
            torch.cuda.empty_cache()
        return x, attn_out

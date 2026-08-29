"""
LingBot Camera Control - Model.

Extends WanModel with LingBot-specific pre_weight_class and transformer_weight_class
that include camera control modules.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.infer.lingbot.pre_infer import LingBotPreInfer
from lightx2v.models.networks.wan.infer.lingbot.transformer_infer import LingBotTransformerInfer
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.weights.lingbot_pre_weights import LingBotPreWeights
from lightx2v.models.networks.wan.weights.lingbot_transformer_weights import LingBotTransformerWeights
from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer


class LingBotModel(WanModel):
    pre_weight_class = LingBotPreWeights
    transformer_weight_class = LingBotTransformerWeights

    def __init__(self, model_path, config, device, model_type="wan2.1", lora_path=None, lora_strength=1.0):
        super().__init__(model_path, config, device, model_type, lora_path, lora_strength)

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        """Override to also chunk cam_emb along sequence dimension for seq parallel."""
        pre_infer_out = super()._seq_parallel_pre_process(pre_infer_out)

        # Chunk cam_emb the same way as x
        cam_emb = pre_infer_out.adapter_args.get("cam_emb", None)
        if cam_emb is not None:
            world_size = dist.get_world_size(self.seq_p_group)
            cur_rank = dist.get_rank(self.seq_p_group)
            f, _, _ = pre_infer_out.grid_sizes.tuple
            multiple = world_size * f
            padding_size = (multiple - (cam_emb.shape[0] % multiple)) % multiple
            if padding_size > 0:
                cam_emb = F.pad(cam_emb, (0, 0, 0, padding_size))
            pre_infer_out.adapter_args["cam_emb"] = torch.chunk(cam_emb, world_size, dim=0)[cur_rank]

        return pre_infer_out

    def _init_infer_class(self):
        self.pre_infer_class = LingBotPreInfer
        self.post_infer_class = WanPostInfer

        if self.config["feature_caching"] == "NoCaching":
            self.transformer_infer_class = LingBotTransformerInfer if not self.cpu_offload else WanOffloadTransformerInfer
        else:
            # For feature caching modes, fall back to standard (cam injection may not be cached-compatible yet)
            # TODO: Implement lingbot-specific caching if needed
            from lightx2v.models.networks.wan.infer.feature_caching.transformer_infer import (
                WanTransformerInferTeaCaching,
                WanTransformerInferTaylorCaching,
                WanTransformerInferAdaCaching,
                WanTransformerInferCustomCaching,
                WanTransformerInferFirstBlock,
                WanTransformerInferDualBlock,
                WanTransformerInferDynamicBlock,
                WanTransformerInferMagCaching,
            )
            caching_map = {
                "Tea": WanTransformerInferTeaCaching,
                "TaylorSeer": WanTransformerInferTaylorCaching,
                "Ada": WanTransformerInferAdaCaching,
                "Custom": WanTransformerInferCustomCaching,
                "FirstBlock": WanTransformerInferFirstBlock,
                "DualBlock": WanTransformerInferDualBlock,
                "DynamicBlock": WanTransformerInferDynamicBlock,
                "Mag": WanTransformerInferMagCaching,
            }
            fc = self.config["feature_caching"]
            if fc in caching_map:
                self.transformer_infer_class = caching_map[fc]
            else:
                raise NotImplementedError(f"Unsupported feature_caching type: {fc}")

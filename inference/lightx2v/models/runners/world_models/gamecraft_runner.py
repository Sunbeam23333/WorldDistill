"""Hunyuan-GameCraft / GameCraft-2 World Model Runner.

GameCraft is Tencent's game world generation model built on the HunyuanVideo
architecture. It supports action-conditioned video generation with dual action
representation (camera + character movement).

Paper: https://arxiv.org/abs/2501.09261 (GameCraft)
Paper: https://arxiv.org/abs/2506.06게 (GameCraft-2)
"""

from lightx2v.models.runners.world_models.base_world_model_runner import BaseWorldModelRunner
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@RUNNER_REGISTER("gamecraft")
class GameCraftRunner(BaseWorldModelRunner):
    MODEL_NAME = "Hunyuan-GameCraft"
    PAPER_URL = "https://arxiv.org/abs/2501.09261"
    STATUS = "stub"
    ARCHITECTURE = "hunyuan_video"

    def __init__(self, config):
        super().__init__(config)
        self.action_conditioning = True
        self.camera_conditioning = True
        # GameCraft uses Dual Action Representation:
        # - Camera actions (9 types: forward/backward/left/right/up/down/yaw_left/yaw_right/still)
        # - Character actions (9 types: same movement categories)
        # Combined: 9x9 = 81 action combinations
        self.num_action_classes = config.get("num_action_classes", 81)
        self.action_embed_dim = config.get("action_embed_dim", 512)

    def load_transformer(self):
        # TODO: Load HunyuanVideo-based transformer with action conditioning layers
        # The model extends HunyuanVideo 1.5 with:
        # - Action embedding MLP (class_id -> action_embed_dim)
        # - Cross-attention between action embeddings and video features
        raise NotImplementedError(
            "GameCraft transformer loading pending. "
            "Based on HunyuanVideo 1.5 with Dual Action Representation."
        )

    def prepare_action_input(self, action_data):
        # TODO: Convert action labels to embedding vectors
        # action_data: dict with 'camera_action' and 'character_action' keys
        # Each is an integer in [0, 8] representing the 9 action types
        raise NotImplementedError(
            "GameCraft action input preparation pending. "
            "Requires Dual Action Representation encoding."
        )

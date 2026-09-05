"""GameFactory World Model Runner.

GameFactory generates game videos with multi-modal control signals including
text, images, actions, and camera poses. Built on a DiT architecture with
action-aware conditioning modules.

Paper: https://arxiv.org/abs/2501.08325
"""

from lightx2v.models.runners.world_models.base_world_model_runner import BaseWorldModelRunner
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@RUNNER_REGISTER("gamefactory")
class GameFactoryRunner(BaseWorldModelRunner):
    MODEL_NAME = "GameFactory"
    PAPER_URL = "https://arxiv.org/abs/2501.08325"
    STATUS = "stub"
    ARCHITECTURE = "dit"

    def __init__(self, config):
        super().__init__(config)
        self.action_conditioning = True
        self.camera_conditioning = True
        # GameFactory uses multi-modal control:
        # - Text description
        # - Reference images
        # - Discrete action tokens
        # - Camera trajectory
        self.action_vocab_size = config.get("action_vocab_size", 256)

    def load_transformer(self):
        # TODO: Load DiT backbone with multi-modal conditioning adapters
        raise NotImplementedError(
            "GameFactory transformer loading pending. "
            "Requires DiT with multi-modal control adapters."
        )

    def prepare_action_input(self, action_data):
        # TODO: Tokenize and embed action sequences
        raise NotImplementedError(
            "GameFactory action input preparation pending."
        )

    def prepare_camera_input(self, camera_poses):
        # TODO: Encode camera trajectory as conditioning signal
        raise NotImplementedError(
            "GameFactory camera input preparation pending."
        )

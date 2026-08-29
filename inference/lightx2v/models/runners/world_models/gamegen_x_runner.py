"""GameGen-X World Model Runner.

GameGen-X is a diffusion-transformer-based game video generation model
that supports interactive control through structured action signals.
It uses a two-phase training: pre-training on open-domain video and
fine-tuning on game-specific data with InstructNet for action alignment.

Paper: https://arxiv.org/abs/2411.00769
"""

from lightx2v.models.runners.world_models.base_world_model_runner import BaseWorldModelRunner
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@RUNNER_REGISTER("gamegen_x")
class GameGenXRunner(BaseWorldModelRunner):
    MODEL_NAME = "GameGen-X"
    PAPER_URL = "https://arxiv.org/abs/2411.00769"
    STATUS = "stub"
    ARCHITECTURE = "dit"

    def __init__(self, config):
        super().__init__(config)
        self.action_conditioning = True
        self.camera_conditioning = True
        # GameGen-X uses InstructNet for action-to-video alignment
        self.use_instruct_net = config.get("use_instruct_net", True)
        # Structured action: (character_action, camera_action, event_description)
        self.action_types = ["character", "camera", "event"]

    def load_transformer(self):
        # TODO: Load DiT backbone + InstructNet adapter
        # GameGen-X uses a two-stage approach:
        # 1. Foundation model pre-trained on OGameData
        # 2. InstructNet fine-tuned for action-conditioned generation
        raise NotImplementedError(
            "GameGen-X transformer loading pending. "
            "Requires DiT + InstructNet for action alignment."
        )

    def prepare_action_input(self, action_data):
        # TODO: Parse structured actions into conditioning tensors
        # action_data should be a dict with:
        #   'character_action': str or int
        #   'camera_action': str or int
        #   'event': str (natural language event description)
        raise NotImplementedError(
            "GameGen-X action preparation pending. "
            "Requires structured action parsing."
        )

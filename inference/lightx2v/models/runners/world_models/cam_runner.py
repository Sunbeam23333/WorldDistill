"""CAM (Context as Memory) World Model Runner.

CAM uses context frames as implicit memory for autoregressive world generation.
It concatenates context frames with noise frames in the latent space,
allowing the model to maintain temporal consistency without explicit memory banks.

Paper: https://arxiv.org/abs/2501.07814
"""

from lightx2v.models.runners.world_models.base_world_model_runner import BaseWorldModelRunner
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@RUNNER_REGISTER("cam")
class CAMRunner(BaseWorldModelRunner):
    MODEL_NAME = "CAM (Context as Memory)"
    PAPER_URL = "https://arxiv.org/abs/2501.07814"
    STATUS = "stub"
    ARCHITECTURE = "dit"

    def __init__(self, config):
        super().__init__(config)
        # CAM concatenates clean context frames with noisy target frames
        self.context_frames = config.get("context_frames", 8)
        self.generation_frames = config.get("generation_frames", 16)
        # Context Forcing: use teacher-generated frames as context during training
        self.use_context_forcing = config.get("use_context_forcing", True)

    def load_transformer(self):
        # TODO: Load DiT that accepts concatenated context + noisy frames
        # The model processes [context_frames | noisy_frames] as a single sequence
        # with positional encoding distinguishing context vs. generation positions
        raise NotImplementedError(
            "CAM transformer loading pending. "
            "Requires DiT with context-frame concatenation support."
        )

    def prepare_memory_context(self, previous_frames, memory_config=None):
        # TODO: Select context frames from generation history
        # CAM uses the most recent N frames as context
        # Optionally: use Context Forcing with teacher-generated frames
        raise NotImplementedError(
            "CAM context preparation pending. "
            "Requires frame selection and latent concatenation."
        )

"""SkyReels-V2 World Model Runner.

SkyReels-V2 is built on the Wan2.1 architecture with Diffusion Forcing
for infinite-length video generation. It supports both text-to-video and
image-to-video tasks with action/camera conditioning.

Paper: https://arxiv.org/abs/2504.13074
Repo: https://github.com/SkyworkAI/SkyReels-V2
"""

from lightx2v.models.runners.world_models.base_world_model_runner import BaseWorldModelRunner
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@RUNNER_REGISTER("skyreels_v2")
class SkyReelsV2Runner(BaseWorldModelRunner):
    MODEL_NAME = "SkyReels-V2"
    PAPER_URL = "https://arxiv.org/abs/2504.13074"
    STATUS = "stub"
    ARCHITECTURE = "wan"

    def __init__(self, config):
        super().__init__(config)
        # SkyReels-V2 uses Diffusion Forcing with per-frame noise levels
        self.diffusion_forcing = config.get("diffusion_forcing", True)
        self.window_size = config.get("window_size", 16)
        self.overlap_frames = config.get("overlap_frames", 4)

    def init_scheduler(self):
        # TODO: Initialize StreamDistillScheduler or DiffusionForcingScheduler
        raise NotImplementedError(
            "SkyReels-V2 scheduler initialization pending. "
            "Requires Diffusion Forcing scheduler with per-frame noise levels."
        )

    def load_transformer(self):
        # TODO: Load Wan2.1-based transformer with Diffusion Forcing modifications
        # The model uses the same Wan2.1 DiT backbone but with modified
        # timestep embedding to support per-frame noise levels.
        raise NotImplementedError(
            "SkyReels-V2 transformer loading pending. "
            "Based on Wan2.1 14B DiT with per-frame timestep support."
        )

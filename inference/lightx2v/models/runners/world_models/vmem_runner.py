"""V-Mem / SPMem World Model Runner.

V-Mem uses visual memory conditioning for long-horizon video generation,
maintaining temporal consistency through a memory bank mechanism.
SPMem (Spatial Memory) extends this with spatial-aware memory selection.

Paper (V-Mem): https://arxiv.org/abs/2412.11918
Paper (SPMem): https://arxiv.org/abs/2501.00478
"""

from lightx2v.models.runners.world_models.base_world_model_runner import BaseWorldModelRunner
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@RUNNER_REGISTER("vmem")
class VMemRunner(BaseWorldModelRunner):
    MODEL_NAME = "V-Mem"
    PAPER_URL = "https://arxiv.org/abs/2412.11918"
    STATUS = "stub"
    ARCHITECTURE = "dit"

    def __init__(self, config):
        super().__init__(config)
        # V-Mem maintains a memory bank of key frames
        self.memory_frames = config.get("memory_frames", 16)
        self.memory_selection = config.get("memory_selection", "temporal")
        # Memory bank stores latent representations of past frames
        self.memory_bank = None

    def load_transformer(self):
        # TODO: Load DiT with memory-conditioned cross-attention
        # V-Mem adds memory cross-attention layers to the base DiT
        raise NotImplementedError(
            "V-Mem transformer loading pending. "
            "Requires DiT with memory cross-attention modules."
        )

    def prepare_memory_context(self, previous_frames, memory_config=None):
        # TODO: Select and encode memory frames
        # Temporal selection: use evenly-spaced past frames
        # Content-based selection: use CLIP similarity
        raise NotImplementedError(
            "V-Mem memory context preparation pending. "
            "Requires temporal/content-based memory selection."
        )


@RUNNER_REGISTER("spmem")
class SPMemRunner(VMemRunner):
    MODEL_NAME = "SPMem"
    PAPER_URL = "https://arxiv.org/abs/2501.00478"
    STATUS = "stub"

    def __init__(self, config):
        super().__init__(config)
        # SPMem extends V-Mem with spatial-aware memory
        self.memory_selection = config.get("memory_selection", "spatial")
        self.spatial_overlap_threshold = config.get("spatial_overlap_threshold", 0.3)

    def prepare_memory_context(self, previous_frames, memory_config=None):
        # TODO: Spatial-aware memory selection using FOV overlap
        # Select frames with maximum spatial overlap with current viewpoint
        raise NotImplementedError(
            "SPMem spatial memory selection pending. "
            "Requires FOV overlap computation and spatial indexing."
        )

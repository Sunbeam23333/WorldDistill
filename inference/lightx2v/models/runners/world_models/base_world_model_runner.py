"""Base World Model Runner.

Provides a common base class for all world model runners with shared
metadata fields and a standardized interface for action/camera conditioning.
"""

from loguru import logger

from lightx2v.models.runners.default_runner import DefaultRunner


class BaseWorldModelRunner(DefaultRunner):
    """Base class for world model runners.

    Extends DefaultRunner with world-model-specific capabilities:
    - Action conditioning (keyboard/gamepad inputs)
    - Camera pose conditioning (6-DoF or spherical)
    - Memory/context management for autoregressive generation
    - Interactive generation support

    Subclasses should override the stub methods to provide model-specific logic.
    """

    MODEL_NAME: str = "base_world_model"
    PAPER_URL: str = ""
    STATUS: str = "stub"  # "supported" | "stub" | "experimental"
    ARCHITECTURE: str = "unknown"  # "wan" | "hunyuan_video" | "dit" | "unet"

    def __init__(self, config):
        super().__init__(config)
        self.action_conditioning = config.get("action_conditioning", False)
        self.camera_conditioning = config.get("camera_conditioning", False)
        self.memory_frames = config.get("memory_frames", 0)
        self.model_architecture = config.get("model_architecture", self.ARCHITECTURE)
        self.model_family = config.get("model_family", "world_model")
        self.checkpoint_format = config.get("checkpoint_format", "unknown")
        self.supported_tasks = config.get("supported_tasks", [])
        self._check_status()
        logger.info(
            f"[WorldDistill] runner={self.MODEL_NAME} | architecture={self.model_architecture} | "
            f"family={self.model_family} | checkpoint_format={self.checkpoint_format} | "
            f"tasks={self.supported_tasks}"
        )

    def _check_status(self):
        if self.STATUS == "stub":
            logger.warning(
                f"[WorldDistill] {self.MODEL_NAME} runner is a STUB. "
                f"Full implementation pending. Paper: {self.PAPER_URL}"
            )

    def prepare_action_input(self, action_data):
        """Prepare action conditioning input (keyboard/gamepad).

        Args:
            action_data: Raw action data (format depends on model).

        Returns:
            Processed action embeddings ready for transformer input.
        """
        raise NotImplementedError(
            f"{self.MODEL_NAME}: action conditioning not yet implemented."
        )

    def prepare_camera_input(self, camera_poses):
        """Prepare camera pose conditioning input.

        Args:
            camera_poses: Camera poses as (N, 6) tensor or dict with
                rotation/translation fields.

        Returns:
            Processed camera embeddings (e.g., Plücker coordinates, RoPE offsets).
        """
        raise NotImplementedError(
            f"{self.MODEL_NAME}: camera conditioning not yet implemented."
        )

    def prepare_memory_context(self, previous_frames, memory_config=None):
        """Prepare memory context from previously generated frames.

        Args:
            previous_frames: Latent representations of previous frames.
            memory_config: Optional dict with memory selection strategy
                (e.g., FOV overlap, temporal distance).

        Returns:
            Memory context tensor to condition the current generation.
        """
        raise NotImplementedError(
            f"{self.MODEL_NAME}: memory context not yet implemented."
        )

    def run_interactive(self, input_info, action_stream):
        """Run interactive (real-time) generation loop.

        Args:
            input_info: Standard input info dict.
            action_stream: Iterator yielding action inputs at each step.

        Yields:
            Generated video frames (latent or decoded).
        """
        raise NotImplementedError(
            f"{self.MODEL_NAME}: interactive generation not yet implemented."
        )

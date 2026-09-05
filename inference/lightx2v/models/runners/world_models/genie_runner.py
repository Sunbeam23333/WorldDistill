"""Genie / Genie 2 World Model Runner.

Genie is Google DeepMind's generative interactive environment model.
It learns to generate interactive worlds from unlabeled video data,
using a spatiotemporal video tokenizer and autoregressive transformer.

Paper: https://arxiv.org/abs/2402.15391 (Genie)
Blog: https://deepmind.google/discover/blog/genie-2/ (Genie 2)
"""

from lightx2v.models.runners.world_models.base_world_model_runner import BaseWorldModelRunner
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@RUNNER_REGISTER("genie")
class GenieRunner(BaseWorldModelRunner):
    MODEL_NAME = "Genie"
    PAPER_URL = "https://arxiv.org/abs/2402.15391"
    STATUS = "stub"
    ARCHITECTURE = "dit"  # Spatiotemporal Transformer

    def __init__(self, config):
        super().__init__(config)
        self.action_conditioning = True
        # Genie uses a latent action model (LAM) to infer actions
        # from unlabeled video, then conditions generation on those actions
        self.use_latent_action_model = config.get("use_latent_action_model", True)
        self.action_latent_dim = config.get("action_latent_dim", 8)
        # Genie uses VQ-VAE tokenizer
        self.vq_codebook_size = config.get("vq_codebook_size", 1024)

    def load_transformer(self):
        # TODO: Load spatiotemporal transformer with VQ-VAE tokenizer
        # Genie uses:
        # - ST-ViViT tokenizer for video -> discrete tokens
        # - Latent Action Model for action inference
        # - Dynamics model (autoregressive transformer) for next-frame prediction
        raise NotImplementedError(
            "Genie transformer loading pending. "
            "Requires ST-ViViT tokenizer + autoregressive dynamics model."
        )

    def prepare_action_input(self, action_data):
        # TODO: Either use provided actions or infer via Latent Action Model
        raise NotImplementedError(
            "Genie action preparation pending. "
            "Requires Latent Action Model for unsupervised action inference."
        )

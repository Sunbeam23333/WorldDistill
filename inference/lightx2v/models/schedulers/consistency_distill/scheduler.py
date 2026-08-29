"""Consistency Distillation Scheduler.

Implements Trajectory Consistency Distillation (TCD) and Latent Consistency
Distillation (LCD) schedulers for few-step generation.

Key concepts:
- Consistency function: maps any point on the ODE trajectory to the origin
- Semi-linear consistency function for stable training
- Supports 1-8 step generation after distillation
- EMA teacher for stable target computation

References:
- Consistency Models: https://arxiv.org/abs/2303.01469
- Latent Consistency Models: https://arxiv.org/abs/2310.04378
- Trajectory Consistency Distillation: https://arxiv.org/abs/2402.19159
"""

from typing import Optional, Tuple, Union

import torch

from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v.utils.envs import *
from lightx2v_platform.base.global_var import AI_DEVICE


class ConsistencyDistillScheduler(BaseScheduler):
    """Scheduler for Consistency Distillation (TCD/LCD).

    During inference, the consistency model directly maps noisy samples
    to clean predictions in very few steps (1-8). This scheduler handles
    the multi-step consistency sampling process.

    Config keys:
        num_inference_steps (int): Number of sampling steps (1-8).
        num_train_timesteps (int): Total training timesteps (default 1000).
        sigma_min (float): Minimum sigma (default 0.002).
        sigma_max (float): Maximum sigma (default 80.0 for EDM, 1.0 for flow).
        sigma_data (float): Data standard deviation (default 0.5).
        rho (float): Karras schedule parameter (default 7.0).
        use_flow_matching (bool): Use flow matching sigmas (default True).
        sample_shift (float): Shift factor for flow matching schedule.
        prediction_type (str): "epsilon" | "v_prediction" | "flow" (default "flow").
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_train_timesteps = config.get("num_train_timesteps", 1000)
        self.sigma_min = config.get("sigma_min", 0.002)
        self.sigma_max = config.get("sigma_max", 1.0)
        self.sigma_data = config.get("sigma_data", 0.5)
        self.rho = config.get("rho", 7.0)
        self.use_flow_matching = config.get("use_flow_matching", True)
        self.sample_shift = config.get("sample_shift", 1.0)
        self.prediction_type = config.get("prediction_type", "flow")

        # Will be set in prepare()
        self.sigmas: Optional[torch.Tensor] = None
        self.timesteps: Optional[torch.Tensor] = None

    def _compute_karras_sigmas(self, num_steps: int) -> torch.Tensor:
        """Compute Karras et al. noise schedule.

        sigma_i = (sigma_max^(1/rho) + i/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
        """
        ramp = torch.linspace(0, 1, num_steps)
        min_inv_rho = self.sigma_min ** (1.0 / self.rho)
        max_inv_rho = self.sigma_max ** (1.0 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return sigmas

    def _compute_flow_matching_sigmas(self, num_steps: int) -> torch.Tensor:
        """Compute flow matching sigma schedule with optional shift."""
        sigmas = torch.linspace(self.sigma_max, self.sigma_min, num_steps + 1)[:-1]
        if self.sample_shift != 1.0:
            sigmas = self.sample_shift * sigmas / (1 + (self.sample_shift - 1) * sigmas)
        return sigmas

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
    ):
        """Set up the sigma/timestep schedule for inference.

        Args:
            num_inference_steps: Number of denoising steps (1-8 typical).
            device: Target device.
        """
        self.infer_steps = num_inference_steps

        if self.use_flow_matching:
            # For flow matching: evenly spaced sigmas
            all_sigmas = self._compute_flow_matching_sigmas(self.num_train_timesteps)
            # Select evenly spaced indices for the given number of steps
            indices = torch.linspace(0, len(all_sigmas) - 1, num_inference_steps).long()
            self.sigmas = all_sigmas[indices]
            self.timesteps = self.sigmas * self.num_train_timesteps
        else:
            # EDM-style Karras schedule
            self.sigmas = self._compute_karras_sigmas(num_inference_steps)
            self.timesteps = self.sigmas * self.num_train_timesteps

        if device is not None:
            self.timesteps = self.timesteps.to(device)
            self.sigmas = self.sigmas.to("cpu")

    def prepare(
        self,
        seed: int,
        latent_shape: Tuple[int, ...],
        image_encoder_output=None,
    ):
        """Initialize latents and set timestep schedule.

        Args:
            seed: Random seed for reproducibility.
            latent_shape: Shape of latent tensor.
            image_encoder_output: Optional image conditioning.
        """
        self.prepare_latents(seed, latent_shape, dtype=torch.float32)
        self.set_timesteps(self.infer_steps, device=AI_DEVICE)

    def consistency_function(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """Apply the semi-linear consistency function.

        For flow matching prediction type:
            x_0 = x_t - sigma * v_pred

        For epsilon prediction:
            x_0 = (x_t - sigma * eps_pred) / (1 - sigma)

        The consistency function ensures that f(x_t, t) = x_0 for all t.

        Args:
            model_output: Model prediction (v, epsilon, or x_0 depending on type).
            sample: Current noisy sample x_t.
            sigma: Current noise level.

        Returns:
            Denoised prediction x_0.
        """
        if self.prediction_type == "flow":
            # Flow matching: v = (x_1 - x_0), so x_0 = x_t - sigma * v
            denoised = sample - sigma * model_output
        elif self.prediction_type == "epsilon":
            denoised = (sample - sigma * model_output) / max(1 - sigma, 1e-8)
        elif self.prediction_type == "v_prediction":
            # v-prediction: v = alpha * eps - sigma * x_0
            # Approximate for flow matching regime
            denoised = sample - sigma * model_output
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        return denoised

    def step_pre(self, step_index):
        """Pre-step: set timestep for current step."""
        super().step_pre(step_index)
        self.timestep_input = torch.stack([self.timesteps[self.step_index]])

    def step_post(self):
        """Post-step: apply consistency function and optionally re-noise.

        For multi-step consistency sampling:
        1. Apply consistency function to get x_0 estimate
        2. If not the last step, add noise at the next sigma level
           x_{t-1} = x_0 + sigma_{t-1} * z, where z ~ N(0, I)
        """
        flow_pred = self.noise_pred.to(torch.float32)
        sigma = self.sigmas[self.step_index].item()

        # Apply consistency function to get clean prediction
        denoised = self.consistency_function(
            model_output=flow_pred,
            sample=self.latents.to(torch.float32),
            sigma=sigma,
        )

        if self.step_index < self.infer_steps - 1:
            # Re-noise for next step (stochastic sampling)
            sigma_next = self.sigmas[self.step_index + 1].item()
            noise = torch.randn_like(denoised)
            # For flow matching: x_t = (1 - sigma) * x_0 + sigma * noise
            self.latents = ((1 - sigma_next) * denoised + sigma_next * noise).to(
                self.latents.dtype
            )
        else:
            # Last step: return clean prediction
            self.latents = denoised.to(self.latents.dtype)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """Add noise to clean samples at the given sigma level.

        For flow matching: x_t = (1 - sigma) * x_0 + sigma * noise

        Args:
            original_samples: Clean samples x_0.
            noise: Gaussian noise.
            sigma: Noise level.

        Returns:
            Noisy samples x_t.
        """
        return ((1 - sigma) * original_samples + sigma * noise).type_as(noise)

    def clear(self):
        """Reset scheduler state."""
        self.sigmas = None
        self.timesteps = None
        self.latents = None

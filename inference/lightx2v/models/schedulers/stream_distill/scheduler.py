"""Stream Distillation Scheduler.

Implements Diffusion Forcing-based streaming generation where each frame
maintains an independent noise level. Supports infinite-length video
generation via sliding window with overlap.

Key concepts:
- Per-frame independent noise timesteps (non-decreasing constraint)
- Sliding window generation with configurable overlap
- Monotonic noise schedule ensuring temporal causal consistency
- Compatible with both Wan and HunyuanVideo architectures

References:
- Diffusion Forcing: https://arxiv.org/abs/2407.01392
- SkyReels-V2: https://arxiv.org/abs/2504.13074
"""

import math
from typing import List, Optional, Tuple, Union

import torch

from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v.utils.envs import *
from lightx2v_platform.base.global_var import AI_DEVICE


class StreamDistillScheduler(BaseScheduler):
    """Scheduler for stream (Diffusion Forcing) distillation.

    Unlike standard schedulers that apply the same timestep to all frames,
    this scheduler maintains per-frame noise levels with a non-decreasing
    constraint, enabling autoregressive-style generation within a diffusion
    framework.

    Config keys:
        window_size (int): Number of frames in each generation window.
        overlap_frames (int): Number of overlapping frames between windows.
        noise_schedule (str): Schedule type - "linear", "cosine", or "monotonic_linear".
        denoising_steps (int): Number of denoising steps per frame position.
        num_train_timesteps (int): Total training timesteps (default 1000).
        sample_shift (float): Timestep shift factor for flow matching.
        causal_window (bool): If True, enforce causal masking within window.
    """

    def __init__(self, config):
        super().__init__(config)
        self.window_size = config.get("window_size", 16)
        self.overlap_frames = config.get("overlap_frames", 4)
        self.noise_schedule = config.get("noise_schedule", "monotonic_linear")
        self.denoising_steps = config.get("denoising_steps", 4)
        self.num_train_timesteps = config.get("num_train_timesteps", 1000)
        self.sample_shift = config.get("sample_shift", 1.0)
        self.causal_window = config.get("causal_window", True)

        self.sigma_max = 1.0
        self.sigma_min = 0.0

        # Per-frame noise state
        self.frame_sigmas: Optional[torch.Tensor] = None
        self.frame_timesteps: Optional[torch.Tensor] = None
        self.window_index = 0
        self.total_frames_generated = 0

        # Denoising schedule for each frame position
        self._build_denoising_schedule()

    def _build_denoising_schedule(self):
        """Build the base sigma schedule used for per-frame denoising."""
        sigmas = torch.linspace(self.sigma_max, self.sigma_min, self.num_train_timesteps + 1)[:-1]
        if self.sample_shift != 1.0:
            sigmas = self.sample_shift * sigmas / (1 + (self.sample_shift - 1) * sigmas)
        self.base_sigmas = sigmas
        self.base_timesteps = sigmas * self.num_train_timesteps

    def _get_frame_sigma_schedule(self, frame_idx_in_window: int) -> torch.Tensor:
        """Get the denoising sigma schedule for a specific frame position.

        For Diffusion Forcing, earlier frames in the window have lower noise
        (closer to clean), and later frames have higher noise. This creates
        a monotonically non-decreasing noise profile across the window.

        Args:
            frame_idx_in_window: Position of the frame within the current window.

        Returns:
            Tensor of sigma values for denoising steps of this frame.
        """
        if self.noise_schedule == "monotonic_linear":
            # Linear interpolation: frame 0 gets lowest noise, last frame gets highest
            ratio = frame_idx_in_window / max(self.window_size - 1, 1)
            # Starting noise level for this frame position
            sigma_start = self.sigma_min + ratio * (self.sigma_max - self.sigma_min)
            # Each frame is denoised from sigma_start to sigma_min
            frame_sigmas = torch.linspace(sigma_start, self.sigma_min, self.denoising_steps + 1)
            return frame_sigmas[:-1]  # Exclude final 0

        elif self.noise_schedule == "cosine":
            ratio = frame_idx_in_window / max(self.window_size - 1, 1)
            sigma_start = self.sigma_min + (1 - math.cos(ratio * math.pi / 2)) * (self.sigma_max - self.sigma_min)
            frame_sigmas = torch.linspace(sigma_start, self.sigma_min, self.denoising_steps + 1)
            return frame_sigmas[:-1]

        elif self.noise_schedule == "linear":
            # All frames use the same schedule (standard diffusion)
            frame_sigmas = torch.linspace(self.sigma_max, self.sigma_min, self.denoising_steps + 1)
            return frame_sigmas[:-1]

        else:
            raise ValueError(f"Unknown noise_schedule: {self.noise_schedule}")

    def prepare(
        self,
        seed: int,
        latent_shape: Tuple[int, ...],
        image_encoder_output=None,
    ):
        """Initialize latents and per-frame noise schedules for the first window.

        Args:
            seed: Random seed.
            latent_shape: Shape of latent tensor (B, C, T, H, W).
            image_encoder_output: Optional conditioning from image encoder.
        """
        self.prepare_latents(seed, latent_shape, dtype=torch.float32)

        # Initialize per-frame sigma schedules
        num_frames = latent_shape[2] if len(latent_shape) == 5 else self.window_size
        self.frame_sigmas = torch.zeros(num_frames, self.denoising_steps, device="cpu")
        self.frame_timesteps = torch.zeros(num_frames, self.denoising_steps, device="cpu")

        for f in range(num_frames):
            sigmas = self._get_frame_sigma_schedule(f)
            self.frame_sigmas[f] = sigmas
            self.frame_timesteps[f] = sigmas * self.num_train_timesteps

        # Track per-frame denoising progress
        self.frame_step_indices = torch.zeros(num_frames, dtype=torch.long)
        self.window_index = 0
        self.total_frames_generated = 0

    def get_frame_timesteps(self, frame_idx: int) -> torch.Tensor:
        """Get current timestep for a specific frame.

        Args:
            frame_idx: Frame index within current window.

        Returns:
            Timestep tensor for the frame at its current denoising step.
        """
        step_idx = self.frame_step_indices[frame_idx].item()
        if step_idx >= self.denoising_steps:
            return torch.tensor([0.0], device=AI_DEVICE)
        return self.frame_timesteps[frame_idx, step_idx].unsqueeze(0).to(AI_DEVICE)

    def get_window_timesteps(self) -> torch.Tensor:
        """Get timesteps for all frames in the current window.

        Returns:
            Tensor of shape (num_frames,) with current timestep per frame.
        """
        timesteps = []
        num_frames = self.frame_timesteps.shape[0]
        for f in range(num_frames):
            timesteps.append(self.get_frame_timesteps(f))
        return torch.cat(timesteps, dim=0)

    def step_pre(self, step_index):
        """Pre-step: set up timesteps for all frames at current denoising step.

        In stream distillation, step_index refers to the global denoising
        iteration, not a per-frame step. Each frame may be at a different
        stage of denoising.
        """
        super().step_pre(step_index)
        # Compute per-frame timestep inputs
        self.timestep_input = self.get_window_timesteps()

    def step_post(self):
        """Post-step: update latents using Euler step with per-frame noise levels.

        For each frame, applies: x_{t-1} = x_t - sigma_t * v_pred + sigma_{t-1} * v_pred
        where sigma values come from the per-frame schedule.
        """
        flow_pred = self.noise_pred.to(torch.float32)
        latents = self.latents.to(torch.float32)

        num_frames = self.frame_sigmas.shape[0]

        for f in range(num_frames):
            step_idx = self.frame_step_indices[f].item()
            if step_idx >= self.denoising_steps:
                continue

            sigma = self.frame_sigmas[f, step_idx].item()

            # Extract per-frame flow prediction
            if flow_pred.dim() == 5:  # (B, C, T, H, W)
                frame_flow = flow_pred[:, :, f : f + 1]
                frame_latent = latents[:, :, f : f + 1]
            else:
                frame_flow = flow_pred
                frame_latent = latents

            # Euler step: x_clean = x_t - sigma * v
            denoised = frame_latent - sigma * frame_flow

            # Re-noise to next sigma level if not the last step
            if step_idx < self.denoising_steps - 1:
                sigma_next = self.frame_sigmas[f, step_idx + 1].item()
                new_latent = denoised + sigma_next * frame_flow
            else:
                new_latent = denoised

            if flow_pred.dim() == 5:
                latents[:, :, f : f + 1] = new_latent
            else:
                latents = new_latent

            # Advance this frame's denoising step
            self.frame_step_indices[f] += 1

        self.latents = latents.to(self.latents.dtype)

    def advance_window(self) -> Tuple[int, int]:
        """Advance the sliding window for infinite-length generation.

        Returns:
            Tuple of (new_start_frame, new_end_frame) indices in the
            global frame sequence.
        """
        # The first `overlap_frames` of the new window reuse the last
        # `overlap_frames` of the current window (already denoised)
        new_frames = self.window_size - self.overlap_frames
        self.total_frames_generated += new_frames
        self.window_index += 1

        # Reset denoising progress for new frames
        for f in range(self.overlap_frames, self.window_size):
            self.frame_step_indices[f] = 0

        # Shift latents: move overlap region to the beginning
        if self.latents.dim() == 5 and self.latents.shape[2] >= self.window_size:
            overlap_latents = self.latents[:, :, -self.overlap_frames :].clone()
            # Re-initialize noise for new frames
            generator = torch.Generator(device=self.latents.device)
            new_noise = torch.randn(
                self.latents[:, :, : new_frames].shape,
                generator=generator,
                device=self.latents.device,
                dtype=self.latents.dtype,
            )
            self.latents = torch.cat([overlap_latents, new_noise], dim=2)

        # Recompute per-frame schedules
        for f in range(self.window_size):
            sigmas = self._get_frame_sigma_schedule(f)
            self.frame_sigmas[f] = sigmas
            self.frame_timesteps[f] = sigmas * self.num_train_timesteps

        start = self.total_frames_generated
        end = start + self.window_size
        return start, end

    @property
    def is_window_complete(self) -> bool:
        """Check if all frames in current window are fully denoised."""
        return (self.frame_step_indices >= self.denoising_steps).all().item()

    def clear(self):
        """Reset scheduler state."""
        self.frame_sigmas = None
        self.frame_timesteps = None
        self.frame_step_indices = None
        self.window_index = 0
        self.total_frames_generated = 0
        self.latents = None

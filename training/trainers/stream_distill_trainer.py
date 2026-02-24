"""Stream Distillation Trainer.

Implements Diffusion Forcing-based streaming distillation where each frame
maintains an independent noise level. The student learns to denoise frames
with non-decreasing noise levels, enabling infinite-length generation.

Key training dynamics:
- Per-frame independent timestep sampling with monotonic non-decreasing constraint
- Sliding window training with overlap for temporal consistency
- Optional causal attention masking (future frames don't attend to past)
- Teacher provides guidance at matched per-frame noise levels
- Supports autoregressive unrolling for end-to-end training

References:
- Diffusion Forcing: https://arxiv.org/abs/2407.01392
- SkyReels-V2: https://arxiv.org/abs/2504.13074
"""

import math
from typing import Any, Dict

import torch
import torch.nn.functional as F
from loguru import logger

from training.trainers.base_distill_trainer import BaseDistillTrainer


class StreamDistillTrainer(BaseDistillTrainer):
    """Trainer for stream (Diffusion Forcing) distillation.

    The student learns to denoise video frames where each frame has
    an independently sampled noise level, subject to a monotonically
    non-decreasing constraint across the temporal dimension.

    Supports:
    - Sliding window training for long videos
    - Multiple noise schedule options
    - Causal temporal attention masking
    - Optional autoregressive multi-step unrolling for end-to-end training
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.window_size = self.args.window_size
        self.overlap_frames = self.args.overlap_frames
        self.noise_schedule = self.args.noise_schedule
        self.denoising_steps = self.args.denoising_steps_per_frame
        self.causal_attention = getattr(self.args, "causal_attention", True)

    def _sample_per_frame_timesteps(
        self, batch_size: int, num_frames: int
    ) -> torch.Tensor:
        """Sample per-frame timesteps with non-decreasing constraint.

        Generates monotonically non-decreasing noise levels across frames,
        ensuring temporal causality in the diffusion process.

        Returns:
            Tensor of shape (B, T) with per-frame timesteps in [0, num_train_timesteps].
        """
        # Sample sorted uniform values
        u = torch.rand(batch_size, num_frames, device=self.device)
        # Sort to enforce non-decreasing constraint
        u_sorted, _ = torch.sort(u, dim=1)

        if self.noise_schedule == "monotonic_linear":
            timesteps = u_sorted * self.num_train_timesteps
        elif self.noise_schedule == "cosine":
            timesteps = (1 - torch.cos(u_sorted * math.pi / 2)) * self.num_train_timesteps
        elif self.noise_schedule == "sigmoid":
            timesteps = torch.sigmoid((u_sorted - 0.5) * 6) * self.num_train_timesteps
        elif self.noise_schedule == "beta":
            # Beta distribution for more concentration at extremes
            # alpha=2, beta=5 gives more weight to lower noise levels
            from torch.distributions import Beta
            beta_dist = Beta(2.0, 5.0)
            u = beta_dist.sample((batch_size, num_frames)).to(self.device)
            u_sorted, _ = torch.sort(u, dim=1)
            timesteps = u_sorted * self.num_train_timesteps
        else:
            timesteps = u_sorted * self.num_train_timesteps

        return timesteps

    def _build_causal_mask(self, num_frames: int) -> torch.Tensor:
        """Build causal attention mask for temporal dimension.

        Each frame can only attend to itself and previous frames.
        This enforces autoregressive generation order.

        Returns:
            Mask of shape (T, T), True = attend, False = mask out.
        """
        mask = torch.tril(torch.ones(num_frames, num_frames, device=self.device, dtype=torch.bool))
        return mask

    def _compute_window_loss(
        self,
        latents_window: torch.Tensor,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """Compute loss for a single sliding window.

        Args:
            latents_window: Latent frames for this window (B, C, T_win, H, W).
            batch: Full batch dict (for text/image conditioning).

        Returns:
            Scalar loss for this window.
        """
        bs = latents_window.shape[0]
        num_frames = latents_window.shape[2]

        # Sample per-frame timesteps (non-decreasing)
        per_frame_timesteps = self._sample_per_frame_timesteps(bs, num_frames)
        per_frame_sigmas = per_frame_timesteps / self.num_train_timesteps

        # Create per-frame noisy latents
        noise = torch.randn_like(latents_window)
        sigmas_expanded = per_frame_sigmas.view(bs, 1, num_frames, 1, 1)
        noisy_latents = (1 - sigmas_expanded) * latents_window + sigmas_expanded * noise

        # Build input kwargs
        teacher_input = self.prepare_teacher_input(batch, noisy_latents, per_frame_timesteps)
        student_input = self.prepare_student_input(batch, noisy_latents, per_frame_timesteps)

        # Add causal mask if enabled
        if self.causal_attention:
            causal_mask = self._build_causal_mask(num_frames)
            teacher_input["temporal_mask"] = causal_mask
            student_input["temporal_mask"] = causal_mask

        # Teacher forward with per-frame timesteps
        with torch.no_grad():
            teacher_output = self.teacher_model(**teacher_input)
            if isinstance(teacher_output, (tuple, list)):
                teacher_output = teacher_output[0]

        # Student forward
        student_output = self.student_model(**student_input)
        if isinstance(student_output, (tuple, list)):
            student_output = student_output[0]

        loss = self.compute_distill_loss(
            teacher_output, student_output, batch, per_frame_timesteps
        )
        return loss

    def _forward_and_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Override forward to use sliding window with per-frame noise levels."""
        latents = batch["latents"]  # (B, C, T, H, W)
        num_frames = latents.shape[2]

        # If video fits in a single window, process directly
        if num_frames <= self.window_size:
            return self._compute_window_loss(latents, batch)

        # Sliding window processing
        total_loss = torch.tensor(0.0, device=self.device)
        num_windows = 0
        stride = max(1, self.window_size - self.overlap_frames)

        for start in range(0, num_frames - self.overlap_frames, stride):
            end = min(start + self.window_size, num_frames)
            latents_window = latents[:, :, start:end]

            if latents_window.shape[2] < 2:
                continue

            window_loss = self._compute_window_loss(latents_window, batch)
            total_loss = total_loss + window_loss
            num_windows += 1

        return total_loss / max(num_windows, 1)

    def compute_distill_loss(
        self,
        teacher_output: torch.Tensor,
        student_output: torch.Tensor,
        batch: Dict[str, Any],
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Per-frame weighted MSE loss.

        Applies per-frame weighting based on noise level: frames with higher
        noise contribute more to the loss (harder to denoise).

        SNR-based weighting (optional): weight_i = sigma_i / (1 - sigma_i + eps)
        """
        if teacher_output.dim() == 5:
            # (B, C, T, H, W) -> compute per-frame loss
            per_frame_loss = F.mse_loss(
                student_output.float(), teacher_output.float(), reduction="none"
            )
            # Mean over C, H, W dimensions -> (B, T)
            per_frame_loss = per_frame_loss.mean(dim=(1, 3, 4))

            # Weight by noise level (higher noise = higher weight)
            if timesteps.dim() == 2:
                sigmas = timesteps / self.num_train_timesteps  # (B, T)
                # SNR-aware weighting: higher noise gets more weight
                weights = sigmas / (1 - sigmas + 1e-6)
                weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
                loss = (per_frame_loss * weights).sum(dim=1).mean()
            else:
                loss = per_frame_loss.mean()
        else:
            loss = F.mse_loss(student_output.float(), teacher_output.float())

        return loss

    def prepare_teacher_input(
        self,
        batch: Dict[str, Any],
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Dict[str, Any]:
        """Prepare teacher inputs with per-frame timestep conditioning."""
        input_kwargs = {
            "hidden_states": noisy_latents,
            "timestep": timesteps,  # (B, T) per-frame timesteps
        }

        if "encoder_hidden_states" in batch:
            input_kwargs["encoder_hidden_states"] = batch["encoder_hidden_states"]
        if "image_cond" in batch:
            input_kwargs["image_cond"] = batch["image_cond"]

        return input_kwargs

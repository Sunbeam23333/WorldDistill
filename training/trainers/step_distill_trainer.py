"""Step Distillation Trainer.

Implements fixed N-step distillation where the student learns to match
the teacher's output at specific timesteps. Supports MoE dual-model
(high/low noise) architecture used in Wan2.2 and WorldPlay.

The student is trained to directly predict the teacher's flow output
at each of the N denoising steps (e.g., 4 steps: [1000, 750, 500, 250]).

Dual-model mode routes samples at the *per-sample* level (not batch level)
to either the high-noise or low-noise student model based on the boundary.

References:
- LightX2V WanStepDistillScheduler
- LightX2V Wan22StepDistillScheduler (dual model with boundary)
"""

import copy
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from training.trainers.base_distill_trainer import BaseDistillTrainer


class StepDistillTrainer(BaseDistillTrainer):
    """Trainer for N-step distillation.

    The student learns to match teacher predictions at fixed timesteps.
    For dual-model mode, two student networks handle high-noise and
    low-noise regions respectively, with per-sample routing.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.denoising_step_list = self.args.denoising_step_list
        self.use_dual_model = self.args.use_dual_model
        self.boundary_step_index = self.args.boundary_step_index
        self.num_distill_steps = len(self.denoising_step_list)

        # Dual-model support: separate high_noise / low_noise student
        self.student_high = None
        self.student_low = None
        self._dual_optimizer = None
        if self.use_dual_model:
            self.student_high = self.student_model
            # Load or clone the low-noise student
            student_low_path = getattr(self.args, "student_low_model", None)
            if student_low_path and isinstance(student_low_path, str):
                import os
                if os.path.exists(student_low_path):
                    self.student_low = copy.deepcopy(self._unwrap_model(self.student_model))
                    state = torch.load(student_low_path, map_location=self.device, weights_only=True)
                    self.student_low.load_state_dict(state, strict=False)
                    self.student_low.to(self.device)
                    logger.info(f"Loaded low-noise student from {student_low_path}")
                else:
                    self.student_low = copy.deepcopy(self._unwrap_model(self.student_model))
                    self.student_low.to(self.device)
                    logger.info("Cloned student as low-noise model (path not found)")
            else:
                self.student_low = copy.deepcopy(self._unwrap_model(self.student_model))
                self.student_low.to(self.device)
                logger.info("Cloned student as low-noise model")

            # Ensure student_low has gradients enabled
            for p in self.student_low.parameters():
                p.requires_grad = True

            # Create a combined optimizer for both models
            self._build_dual_optimizer()

        # Pre-compute sigma values for distillation steps
        self._precompute_sigmas()

    def _build_dual_optimizer(self):
        """Build optimizer that covers both high and low noise student params."""
        from training.utils.optimizers import build_optimizer
        # Combine parameters from both models
        high_params = list(self._unwrap_model(self.student_high).parameters())
        low_params = list(self.student_low.parameters())

        # Replace the original optimizer with one that covers both
        param_groups = [
            {"params": [p for p in high_params if p.requires_grad], "lr": self.args.learning_rate, "name": "high"},
            {"params": [p for p in low_params if p.requires_grad], "lr": self.args.learning_rate, "name": "low"},
        ]
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
        )
        logger.info(
            f"Dual-model optimizer: high={sum(p.numel() for p in high_params if p.requires_grad)/1e6:.1f}M, "
            f"low={sum(p.numel() for p in low_params if p.requires_grad)/1e6:.1f}M"
        )

    def train(self):
        """Override train to also DDP-wrap student_low if needed."""
        # DDP wrap student_low
        if self.use_dual_model and self.is_distributed:
            if not isinstance(self.student_low, nn.parallel.DistributedDataParallel):
                self.student_low = nn.parallel.DistributedDataParallel(
                    self.student_low, device_ids=[self.rank], find_unused_parameters=False,
                )
        super().train()

    def _precompute_sigmas(self):
        """Pre-compute sigma schedule for the fixed distillation timesteps."""
        sigma_start = 1.0
        sigmas = torch.linspace(sigma_start, 0.0, self.num_train_timesteps + 1)[:-1]
        sample_shift = self.args.sample_shift
        if sample_shift != 1.0:
            sigmas = sample_shift * sigmas / (1 + (sample_shift - 1) * sigmas)

        # Clamp step values to valid range
        step_indices = []
        for s in self.denoising_step_list:
            idx = max(0, min(self.num_train_timesteps - 1, self.num_train_timesteps - s))
            step_indices.append(idx)

        self.distill_sigmas = sigmas[step_indices]
        self.distill_timesteps = self.distill_sigmas * self.num_train_timesteps

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample from the fixed distillation timesteps instead of uniform."""
        indices = torch.randint(0, self.num_distill_steps, (batch_size,))
        timesteps = self.distill_timesteps[indices]
        return timesteps.to(self.device)

    def _forward_and_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Override forward to support per-sample dual-model routing."""
        latents = batch["latents"]
        bs = latents.shape[0]

        # Sample timesteps from fixed schedule
        timesteps = self._sample_timesteps(bs)
        sigmas = timesteps / self.num_train_timesteps

        # Create noisy latents
        noise = torch.randn_like(latents)
        sigmas_expanded = sigmas.view(bs, *([1] * (latents.dim() - 1)))
        noisy_latents = (1 - sigmas_expanded) * latents + sigmas_expanded * noise

        # Teacher forward (no grad)
        teacher_input = self.prepare_teacher_input(batch, noisy_latents, timesteps)
        with torch.no_grad():
            teacher_output = self.teacher_model(**teacher_input)
            if isinstance(teacher_output, (tuple, list)):
                teacher_output = teacher_output[0]

        if not self.use_dual_model:
            # Single model path
            student_input = self.prepare_student_input(batch, noisy_latents, timesteps)
            student_output = self.student_model(**student_input)
            if isinstance(student_output, (tuple, list)):
                student_output = student_output[0]
            loss = self.compute_distill_loss(teacher_output, student_output, batch, timesteps)
            return loss

        # --- Dual-model: per-sample routing ---
        boundary_sigma = self.distill_sigmas[self.boundary_step_index].item()
        boundary_t = boundary_sigma * self.num_train_timesteps

        # Mask: True for high-noise samples, False for low-noise
        high_mask = timesteps >= boundary_t  # (B,)
        low_mask = ~high_mask

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        count = 0

        # High-noise samples through student_high
        if high_mask.any():
            h_idx = high_mask.nonzero(as_tuple=True)[0]
            h_input = self.prepare_student_input(
                self._subset_batch(batch, h_idx),
                noisy_latents[h_idx],
                timesteps[h_idx],
            )
            h_output = self.student_high(**h_input)
            if isinstance(h_output, (tuple, list)):
                h_output = h_output[0]
            h_loss = F.mse_loss(h_output.float(), teacher_output[h_idx].float())
            total_loss = total_loss + h_loss * h_idx.shape[0]
            count += h_idx.shape[0]

        # Low-noise samples through student_low
        if low_mask.any():
            l_idx = low_mask.nonzero(as_tuple=True)[0]
            l_input = self.prepare_student_input(
                self._subset_batch(batch, l_idx),
                noisy_latents[l_idx],
                timesteps[l_idx],
            )
            l_output = self.student_low(**l_input)
            if isinstance(l_output, (tuple, list)):
                l_output = l_output[0]
            l_loss = F.mse_loss(l_output.float(), teacher_output[l_idx].float())
            total_loss = total_loss + l_loss * l_idx.shape[0]
            count += l_idx.shape[0]

        return total_loss / max(count, 1)

    def _subset_batch(self, batch: Dict[str, Any], indices: torch.Tensor) -> Dict[str, Any]:
        """Extract a subset of batch by sample indices."""
        sub = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == indices.shape[0]:
                # Already subset or broadcast
                sub[k] = v
            elif isinstance(v, torch.Tensor) and v.dim() > 0:
                sub[k] = v[indices]
            else:
                sub[k] = v
        return sub

    def compute_distill_loss(
        self,
        teacher_output: torch.Tensor,
        student_output: torch.Tensor,
        batch: Dict[str, Any],
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss between teacher and student flow predictions."""
        loss = F.mse_loss(student_output.float(), teacher_output.float(), reduction="none")

        # Apply mask if available (e.g., i2v condition frames)
        if "loss_mask" in batch:
            mask = batch["loss_mask"]
            loss = loss * mask
            loss = loss.sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()

        return loss

    def prepare_teacher_input(
        self,
        batch: Dict[str, Any],
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Dict[str, Any]:
        """Prepare teacher model inputs."""
        input_kwargs = {
            "hidden_states": noisy_latents,
            "timestep": timesteps,
        }
        if "encoder_hidden_states" in batch:
            input_kwargs["encoder_hidden_states"] = batch["encoder_hidden_states"]
        if "image_cond" in batch:
            input_kwargs["image_cond"] = batch["image_cond"]
        return input_kwargs

    def save_checkpoint(self, step: int, output_dir: str):
        """Save checkpoint including dual model if used."""
        super().save_checkpoint(step, output_dir)
        if self.use_dual_model and self.student_low is not None:
            import os
            ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
            low_state = (
                self.student_low.module.state_dict()
                if hasattr(self.student_low, "module") else self.student_low.state_dict()
            )
            torch.save(low_state, os.path.join(ckpt_dir, "student_low.pt"))

    def load_checkpoint(self, path: str):
        """Load checkpoint including dual model."""
        super().load_checkpoint(path)
        if self.use_dual_model and self.student_low is not None:
            import os
            ckpt_dir = path if os.path.isdir(path) else os.path.dirname(path)
            low_path = os.path.join(ckpt_dir, "student_low.pt")
            if os.path.exists(low_path):
                state = torch.load(low_path, map_location=self.device, weights_only=True)
                raw_low = self.student_low.module if hasattr(self.student_low, "module") else self.student_low
                raw_low.load_state_dict(state)
                logger.info("Loaded student_low from checkpoint.")

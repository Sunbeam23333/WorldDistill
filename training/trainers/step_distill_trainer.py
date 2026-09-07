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
from contextlib import ExitStack
import os
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from training.trainers.base_distill_trainer import BaseDistillTrainer
from training.utils.checkpoint_io import load_model_state_file


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
        from training.model_adapter import NoiseRoutedDenoiser
        if self.use_dual_model and isinstance(self.student_model, NoiseRoutedDenoiser):
            if getattr(self.args, "student_low_model", None):
                raise ValueError("student_low_model cannot override a checkpoint-native NoiseRoutedDenoiser")
            # The checkpoint already contains both trainable experts.
            self.use_dual_model = False
            self.args.use_dual_model = False
            logger.info("Using checkpoint-native noise routing; not duplicating its two student experts.")
        self.boundary_step_index = self.args.boundary_step_index
        self.num_distill_steps = len(self.denoising_step_list)

        # Dual-model support: separate high_noise / low_noise student
        self.student_high = None
        self.student_low = None
        self._dual_optimizer = None
        if self.use_dual_model:
            if self.parallel_mode != "ddp":
                raise ValueError(
                    "Dual high/low-noise students currently support serial/DDP training only; "
                    "FSDP and DeepSpeed would shard only the high-noise model."
                )
            if self.args.gradient_accumulation_steps != 1:
                raise ValueError(
                    "Dual high/low-noise DDP currently requires gradient_accumulation_steps=1; "
                    "a branch used only inside no_sync microbatches would otherwise update with "
                    "unsynchronized gradients."
                )
            self.student_high = self.student_model
            # Load or clone the low-noise student
            student_low_path = getattr(self.args, "student_low_model", None)
            if student_low_path and isinstance(student_low_path, str):
                self.student_low = copy.deepcopy(self._unwrap_model(self.student_model))
                state = load_model_state_file(student_low_path, map_location=self.device)
                self.student_low.load_state_dict(state, strict=True)
                self.student_low.to(self.device)
                logger.info(f"Loaded low-noise student from {student_low_path}")
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
        from training.utils.schedulers import build_lr_scheduler

        high_model = self._unwrap_model(self.student_model)
        combined_students = nn.ModuleList([high_model, self.student_low])
        self.optimizer = build_optimizer(
            combined_students,
            optimizer_type=self.args.optimizer,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            adam_beta1=self.args.adam_beta1,
            adam_beta2=self.args.adam_beta2,
            adam_epsilon=self.args.adam_epsilon,
            muon_momentum=getattr(self.args, "muon_momentum", 0.95),
            muon_nesterov=getattr(self.args, "muon_nesterov", True),
            muon_ns_steps=getattr(self.args, "muon_ns_steps", 5),
        )
        self.lr_scheduler = build_lr_scheduler(
            self.optimizer,
            scheduler_type=self.args.lr_scheduler,
            warmup_steps=self.args.warmup_steps,
            total_steps=self.args.max_train_steps,
            min_lr_ratio=self.args.lr_min_ratio,
        )
        high_params = list(high_model.parameters())
        low_params = list(self.student_low.parameters())
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
                    self.student_low,
                    device_ids=[int(os.environ.get("LOCAL_RANK", 0))] if self.device.type == "cuda" else None,
                    find_unused_parameters=False,
                )
        super().train()

    def _get_no_sync_context(self):
        """Disable gradient synchronization for both DDP students on accumulation steps."""

        stack = ExitStack()
        for model in (self.student_model, self.student_low):
            if model is not None and hasattr(model, "no_sync"):
                stack.enter_context(model.no_sync())
        return stack

    def _trainable_parameters(self):
        """Include both high- and low-noise students in norm/clipping."""

        seen = set()
        for model in (self.student_model, self.student_low):
            if model is None:
                continue
            for parameter in self._unwrap_model(model).parameters():
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    yield parameter

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

        self.distill_sigmas = sigmas[step_indices].to(self.device)
        self.distill_timesteps = self.distill_sigmas * self.num_train_timesteps

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample from the fixed distillation timesteps instead of uniform."""
        indices = torch.randint(
            0,
            self.num_distill_steps,
            (batch_size,),
            device=self.device,
        )
        if self.use_dual_model and dist.is_available() and dist.is_initialized():
            # Conditional DDP branches must match across ranks. Otherwise one
            # rank can enter the high-noise reducer while another skips it.
            dist.broadcast(indices, src=0)
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

        teacher_input = self.prepare_teacher_input(batch, noisy_latents, timesteps)

        if not self.use_dual_model:
            student_input = self.prepare_student_input(batch, noisy_latents, timesteps)
            teacher_output, student_output = self._run_teacher_student_pair(
                teacher_input=teacher_input,
                student_input=student_input,
                batch=batch,
                cache_extra={"timesteps": timesteps, "mode": "step_distill"},
            )
            return self.compute_distill_loss(teacher_output, student_output, batch, timesteps)

        pending_teacher = self.launch_teacher(
            teacher_input,
            batch,
            cache_extra={"timesteps": timesteps, "mode": "step_distill"},
        )

        boundary_sigma = self.distill_sigmas[self.boundary_step_index].item()
        boundary_t = boundary_sigma * self.num_train_timesteps
        high_mask = timesteps >= boundary_t
        low_mask = ~high_mask

        teacher_output = self.wait_teacher(pending_teacher)
        if teacher_output is None:
            teacher_output = self.run_teacher(
                teacher_input,
                batch,
                cache_extra={"timesteps": timesteps, "mode": "step_distill"},
            )

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        count = 0

        if high_mask.any():
            h_idx = high_mask.nonzero(as_tuple=True)[0]
            h_batch = self._subset_batch(batch, h_idx)
            h_input = self.prepare_student_input(
                h_batch,
                noisy_latents[h_idx],
                timesteps[h_idx],
            )
            h_output = self.run_student(h_input, h_batch, model=self.student_model, tag="student_high")
            h_mask = h_batch.get("loss_mask") if isinstance(h_batch, dict) else None
            h_loss = self.compute_supervision_loss(h_output, teacher_output[h_idx].detach(), mask=h_mask)
            total_loss = total_loss + h_loss * h_idx.shape[0]
            count += h_idx.shape[0]

        if low_mask.any():
            l_idx = low_mask.nonzero(as_tuple=True)[0]
            l_batch = self._subset_batch(batch, l_idx)
            l_input = self.prepare_student_input(
                l_batch,
                noisy_latents[l_idx],
                timesteps[l_idx],
            )
            l_output = self.run_student(l_input, l_batch, model=self.student_low, tag="student_low")
            l_mask = l_batch.get("loss_mask") if isinstance(l_batch, dict) else None
            l_loss = self.compute_supervision_loss(l_output, teacher_output[l_idx].detach(), mask=l_mask)
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
        return self.compute_supervision_loss(
            prediction=student_output,
            target=teacher_output.detach(),
            mask=batch.get("loss_mask"),
        )

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
        return self._augment_model_input(input_kwargs, batch)

    def save_checkpoint(self, step: int, output_dir: str):
        """Save checkpoint including dual model if used."""
        super().save_checkpoint(step, output_dir)
        if self.use_dual_model and self.student_low is not None:
            import os

            ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
            low_path = os.path.join(ckpt_dir, "student_low.pt")

            def _low_state():
                raw_low = (
                    self.student_low.module
                    if hasattr(self.student_low, "module")
                    else self.student_low
                )
                return raw_low.state_dict()

            self._atomic_save_rank0(
                low_path,
                _low_state,
                f"Write low-noise student checkpoint {low_path}",
            )

    def load_checkpoint(self, path: str):
        """Load checkpoint including dual model."""
        super().load_checkpoint(path)
        if self.use_dual_model and self.student_low is not None:
            import os
            ckpt_dir = path if os.path.isdir(path) else os.path.dirname(path)
            low_path = os.path.join(ckpt_dir, "student_low.pt")

            state = self._load_checkpoint_file_all_ranks(
                low_path,
                description=f"Load low-noise student checkpoint {low_path}",
                weights_only=True,
            )

            def _restore_low_student():
                raw_low = self.student_low.module if hasattr(self.student_low, "module") else self.student_low
                raw_low.load_state_dict(state, strict=True)

            self._run_all_ranks_or_raise(
                _restore_low_student,
                f"Restore low-noise student checkpoint {low_path}",
            )
            logger.info("Loaded student_low from checkpoint.")

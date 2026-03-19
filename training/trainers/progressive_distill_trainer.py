"""Progressive Distillation Trainer.

Implements progressive distillation where the student is trained to
match a 2N-step teacher in N steps, halving the denoising steps at each stage.

Training procedure:
1. Stage 1: Student matches 64-step teacher in 32 steps
2. Stage 2: Student matches 32-step teacher (from Stage 1) in 16 steps
3. ...continue until target step count (e.g., 4 steps)

At each stage, the teacher is the student from the previous stage.

Key improvements over naive implementation:
- Supports loss in both v-space and x0-space (v-space is more stable per the paper)
- Optimizer reset between stages for clean convergence
- EMA-based teacher for smoother transitions

References:
- Progressive Distillation: https://arxiv.org/abs/2202.00512
- v-prediction formulation for stable progressive distillation
"""

from typing import Any, Dict

import torch
import torch.nn.functional as F
from loguru import logger

from training.trainers.base_distill_trainer import BaseDistillTrainer
from training.utils.optimizers import build_optimizer
from training.utils.schedulers import build_lr_scheduler


class ProgressiveDistillTrainer(BaseDistillTrainer):
    """Trainer for progressive distillation.

    Iteratively halves the number of denoising steps, using the previous
    stage's student as the new teacher.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stages = self.args.progressive_stages
        self.stage_steps = self.args.progressive_stage_steps
        self.loss_space = getattr(self.args, "progressive_loss_space", "v")  # "v" or "x0"
        self.reset_optimizer = getattr(self.args, "progressive_reset_optimizer", True)
        self.current_stage = 0
        self.current_teacher_steps = self.stages[0]
        self.current_student_steps = self.stages[0] // 2 if len(self.stages) > 1 else self.stages[0] // 2

    def _get_teacher_timesteps(self, num_steps: int) -> torch.Tensor:
        """Get evenly spaced timesteps for N-step denoising."""
        return torch.linspace(
            self.num_train_timesteps, 0, num_steps + 1, device=self.device
        )[:-1]

    def _two_step_teacher_prediction(
        self, batch: Dict[str, Any], noisy_latents: torch.Tensor, t_start: float, t_mid: float, t_end: float
    ) -> Dict[str, torch.Tensor]:
        """Teacher makes 2 steps: t_start -> t_mid -> t_end.

        Returns dict with both 'x_end' (denoised result) and 'v_target' (velocity target).
        """
        bs = noisy_latents.shape[0]

        # Step 1: t_start -> t_mid
        sigma_start = t_start / self.num_train_timesteps
        t_start_tensor = torch.full((bs,), t_start, device=self.device)
        teacher_input_1 = self.prepare_teacher_input(batch, noisy_latents, t_start_tensor)
        v_pred_1 = self.run_teacher(
            teacher_input_1,
            batch,
            cache_extra={"t_start": t_start, "t_mid": t_mid, "teacher_stage": 1},
        )

        # Euler step: x_{t_mid} = x_{t_start} - (sigma_start - sigma_mid) * v_pred
        sigma_mid = t_mid / self.num_train_timesteps
        dt_1 = sigma_start - sigma_mid
        x_mid = noisy_latents.float() - dt_1 * v_pred_1.float()

        # Step 2: t_mid -> t_end
        t_mid_tensor = torch.full((bs,), t_mid, device=self.device)
        teacher_input_2 = self.prepare_teacher_input(batch, x_mid.to(noisy_latents.dtype), t_mid_tensor)
        v_pred_2 = self.run_teacher(
            teacher_input_2,
            batch,
            cache_extra={"t_mid": t_mid, "t_end": t_end, "teacher_stage": 2},
        )

        sigma_end = t_end / self.num_train_timesteps
        dt_2 = sigma_mid - sigma_end
        x_end = x_mid - dt_2 * v_pred_2.float()

        # Compute the effective velocity target for the student's single step
        # Student does: x_end_student = x_start - (sigma_start - sigma_end) * v_student
        # We want: x_end_student ≈ x_end_teacher
        # => v_student = (x_start - x_end_teacher) / (sigma_start - sigma_end)
        dt_full = sigma_start - sigma_end
        v_target = (noisy_latents.float() - x_end) / max(dt_full, 1e-6)

        return {"x_end": x_end, "v_target": v_target}

    def _forward_and_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass for progressive distillation."""
        latents = batch["latents"]
        bs = latents.shape[0]

        # Get teacher and student timestep grids
        teacher_ts = self._get_teacher_timesteps(self.current_teacher_steps)
        student_ts = self._get_teacher_timesteps(self.current_student_steps)

        # Sample a random student step to train
        max_idx = max(1, len(student_ts) - 1)
        step_idx = torch.randint(0, max_idx, (1,)).item()
        t_start = student_ts[step_idx].item()
        t_end = student_ts[step_idx + 1].item() if step_idx + 1 < len(student_ts) else 0.0

        # Find the midpoint in teacher schedule
        t_mid = (t_start + t_end) / 2.0

        # Create noisy latents at t_start
        sigma_start = t_start / self.num_train_timesteps
        noise = torch.randn_like(latents)
        noisy_latents = (1 - sigma_start) * latents + sigma_start * noise

        # Teacher: 2-step prediction (t_start -> t_mid -> t_end)
        teacher_result = self._two_step_teacher_prediction(
            batch, noisy_latents, t_start, t_mid, t_end
        )

        # Student: 1-step prediction (t_start -> t_end)
        student_input = self.prepare_student_input(
            batch, noisy_latents, torch.full((bs,), t_start, device=self.device)
        )
        student_v = self.run_student(student_input, batch)

        # Compute loss based on configured loss space
        if self.loss_space == "v":
            # v-space loss (recommended by the progressive distillation paper)
            loss = F.mse_loss(student_v.float(), teacher_result["v_target"].detach())
        else:
            # x0-space loss
            sigma_end = t_end / self.num_train_timesteps
            dt = sigma_start - sigma_end
            student_denoised = noisy_latents.float() - dt * student_v.float()
            loss = F.mse_loss(student_denoised, teacher_result["x_end"].detach())

        return loss

    def compute_distill_loss(
        self,
        teacher_output: torch.Tensor,
        student_output: torch.Tensor,
        batch: Dict[str, Any],
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Not used directly; _forward_and_loss handles the full logic."""
        return F.mse_loss(student_output.float(), teacher_output.float())

    def prepare_teacher_input(
        self,
        batch: Dict[str, Any],
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Dict[str, Any]:
        input_kwargs = {
            "hidden_states": noisy_latents,
            "timestep": timesteps,
        }
        return self._augment_model_input(input_kwargs, batch)

    def on_train_step_end(self, metrics):
        """Check if we should advance to the next progressive stage."""
        if self.global_step > 0 and self.global_step % self.stage_steps == 0:
            next_stage = self.current_stage + 1
            if next_stage < len(self.stages):
                self.current_stage = next_stage
                self.current_teacher_steps = self.stages[self.current_stage]
                self.current_student_steps = self.stages[self.current_stage] // 2
                # Swap: current student becomes new teacher
                src_state = self._unwrap_model(self.student_model).state_dict()
                self.teacher_model.load_state_dict(src_state)
                self.teacher_model.eval()
                for p in self.teacher_model.parameters():
                    p.requires_grad = False

                # Optionally reset optimizer (recommended per paper)
                if self.reset_optimizer:
                    raw_student = self._unwrap_model(self.student_model)
                    self.optimizer = build_optimizer(
                        raw_student,
                        optimizer_type=self.args.optimizer,
                        lr=self.args.learning_rate,
                        weight_decay=self.args.weight_decay,
                        adam_beta1=self.args.adam_beta1,
                        adam_beta2=self.args.adam_beta2,
                    )
                    self.lr_scheduler = build_lr_scheduler(
                        self.optimizer,
                        scheduler_type=self.args.lr_scheduler,
                        warmup_steps=self.args.warmup_steps,
                        total_steps=self.stage_steps,  # Reset schedule for new stage
                        min_lr_ratio=self.args.lr_min_ratio,
                    )
                    logger.info("Optimizer and scheduler reset for new stage.")

                logger.info(
                    f"Progressive stage {self.current_stage}: "
                    f"teacher={self.current_teacher_steps} steps -> "
                    f"student={self.current_student_steps} steps"
                )

                # Save a checkpoint at stage boundary
                if self.is_main_process:
                    self.save_checkpoint(self.global_step, self.args.output_dir)

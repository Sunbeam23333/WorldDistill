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
        self._require_velocity_prediction("Progressive distillation")
        if self.parallel_mode != "ddp":
            raise ValueError(
                "Progressive stage transitions currently support serial/DDP training only; "
                "FSDP/DeepSpeed require a full-state teacher handoff and engine rebuild."
            )
        if bool(getattr(self.args, "use_lora", False)):
            raise ValueError(
                "Progressive distillation with LoRA is not supported because stage-boundary "
                "teacher handoff requires identical full-model state keys."
            )
        self.stages = self.args.progressive_stages
        if len(self.stages) < 2 or any(step <= 0 for step in self.stages):
            raise ValueError("progressive_stages must contain at least two positive step counts")
        if any(left <= right for left, right in zip(self.stages, self.stages[1:])):
            raise ValueError("progressive_stages must be strictly descending")
        self.stage_steps = self.args.progressive_stage_steps
        self.loss_space = getattr(self.args, "progressive_loss_space", "v")  # "v" or "x0"
        if self.loss_space not in {"v", "x0"}:
            raise ValueError("progressive_loss_space must be 'v' or 'x0'")
        self.reset_optimizer = getattr(self.args, "progressive_reset_optimizer", True)
        self.current_stage = 0
        self.current_teacher_steps, self.current_student_steps = self._stage_pair(0)

    def _stage_pair(self, stage_index: int) -> tuple[int, int]:
        if not 0 <= stage_index < len(self.stages) - 1:
            raise IndexError(f"Progressive stage index has no student target: {stage_index}")
        return self.stages[stage_index], self.stages[stage_index + 1]

    def _reset_stage_optimizer(self) -> None:
        """Rebuild the optimizer and scheduler with the stage-local horizon."""

        raw_student = self._unwrap_model(self.student_model)
        self.optimizer = build_optimizer(
            raw_student,
            optimizer_type=self.args.optimizer,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            adam_beta1=self.args.adam_beta1,
            adam_beta2=self.args.adam_beta2,
            adam_epsilon=self.args.adam_epsilon,
            muon_momentum=self.args.muon_momentum,
            muon_nesterov=self.args.muon_nesterov,
            muon_ns_steps=self.args.muon_ns_steps,
        )
        self.lr_scheduler = build_lr_scheduler(
            self.optimizer,
            scheduler_type=self.args.lr_scheduler,
            warmup_steps=self.args.warmup_steps,
            total_steps=self.stage_steps,
            min_lr_ratio=self.args.lr_min_ratio,
        )

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
            loss = self.compute_supervision_loss(
                student_v,
                teacher_result["v_target"].detach(),
            )
        else:
            # x0-space loss
            sigma_end = t_end / self.num_train_timesteps
            dt = sigma_start - sigma_end
            student_denoised = noisy_latents.float() - dt * student_v.float()
            loss = self.compute_supervision_loss(
                student_denoised,
                teacher_result["x_end"].detach(),
            )

        return loss

    def compute_distill_loss(
        self,
        teacher_output: torch.Tensor,
        student_output: torch.Tensor,
        batch: Dict[str, Any],
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Not used directly; _forward_and_loss handles the full logic."""
        return self.compute_supervision_loss(student_output, teacher_output)

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
            if next_stage < len(self.stages) - 1:
                self.current_stage = next_stage
                self.current_teacher_steps, self.current_student_steps = self._stage_pair(
                    self.current_stage
                )
                # Swap: current student becomes new teacher
                src_state = self._unwrap_model(self.student_model).state_dict()
                self.teacher_model.load_state_dict(src_state)
                self.teacher_model.eval()
                for p in self.teacher_model.parameters():
                    p.requires_grad = False

                # Optionally reset optimizer (recommended per paper)
                if self.reset_optimizer:
                    self._reset_stage_optimizer()
                    logger.info("Optimizer and scheduler reset for new stage.")

                logger.info(
                    f"Progressive stage {self.current_stage}: "
                    f"teacher={self.current_teacher_steps} steps -> "
                    f"student={self.current_student_steps} steps"
                )

                # Save a checkpoint at stage boundary
                self.save_checkpoint(self.global_step, self.args.output_dir)

    def save_checkpoint(self, step: int, output_dir: str):
        """Save the stage-local teacher and progressive schedule state."""
        import os

        super().save_checkpoint(step, output_dir)
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
        progressive_path = os.path.join(ckpt_dir, "progressive_state.pt")
        self._atomic_save_rank0(
            progressive_path,
            lambda: {
                "teacher_model": self.teacher_model.state_dict(),
                "current_stage": self.current_stage,
                "current_teacher_steps": self.current_teacher_steps,
                "current_student_steps": self.current_student_steps,
            },
            f"Write progressive checkpoint {progressive_path}",
        )

    def load_checkpoint(self, path: str):
        """Restore the exact progressive stage and its teacher."""
        import os

        ckpt_dir = path if os.path.isdir(path) else os.path.dirname(path)
        progressive_path = os.path.join(ckpt_dir, "progressive_state.pt")
        state = self._load_checkpoint_file_all_ranks(
            progressive_path,
            description=f"Load progressive checkpoint {progressive_path}",
            weights_only=False,
        )

        def _validate_progressive_state() -> tuple[int, int, int]:
            if not isinstance(state, dict):
                raise TypeError(
                    f"Expected progressive checkpoint dict, got {type(state).__name__}"
                )
            required = {
                "teacher_model",
                "current_stage",
                "current_teacher_steps",
                "current_student_steps",
            }
            missing = sorted(required.difference(state))
            if missing:
                raise KeyError(f"Progressive checkpoint is missing required keys: {missing}")

            current_stage = int(state["current_stage"])
            if current_stage < 0 or current_stage >= len(self.stages) - 1:
                raise ValueError(
                    f"Progressive checkpoint current_stage={current_stage} is outside "
                    f"the configured stage range [0, {len(self.stages) - 2}]"
                )
            expected_teacher_steps, expected_student_steps = self._stage_pair(current_stage)
            if int(state["current_teacher_steps"]) != expected_teacher_steps:
                raise ValueError("Progressive checkpoint teacher-step schedule does not match config")
            if int(state["current_student_steps"]) != expected_student_steps:
                raise ValueError("Progressive checkpoint student-step schedule does not match config")
            return current_stage, expected_teacher_steps, expected_student_steps

        stage_state = self._run_all_ranks_or_raise(
            _validate_progressive_state,
            f"Validate progressive checkpoint {progressive_path}",
        )
        self.current_stage, self.current_teacher_steps, self.current_student_steps = stage_state

        # LambdaLR does not serialize its lambda closure. Recreate the
        # stage-local optimizer/scheduler before the base checkpoint restores
        # their tensor/scalar state, so resumed LR behavior matches an
        # uninterrupted run.
        if self.reset_optimizer and self.current_stage > 0:
            self._reset_stage_optimizer()

        super().load_checkpoint(path)

        def _restore_progressive_teacher():
            self.teacher_model.load_state_dict(state["teacher_model"], strict=True)

        self._run_all_ranks_or_raise(
            _restore_progressive_teacher,
            f"Restore progressive checkpoint {progressive_path}",
        )
        logger.info(f"Progressive state restored at stage {self.current_stage}.")

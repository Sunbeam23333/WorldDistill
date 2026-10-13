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
import os
from typing import Any, Dict

import torch
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
        from training.model_adapter import NoiseRoutedDenoiser
        from training.utils.noise_objective import resolve_noise_objectives
        args, high = kwargs["args"], kwargs["student_model"]
        native = high
        while getattr(native, "_orig_mod", None) is not None:
            native = native._orig_mod
        if callable(getattr(native, "get_base_model", None)):
            native = native.get_base_model()
        self.use_dual_model = bool(args.use_dual_model)
        self.student_high = self.student_low = None
        if self.use_dual_model and isinstance(native, NoiseRoutedDenoiser):
            if getattr(args, "student_low_model", None):
                raise ValueError("student_low_model cannot override a checkpoint-native NoiseRoutedDenoiser")
            self.use_dual_model = False
            args.use_dual_model = False
            logger.info("Using checkpoint-native noise routing; not duplicating its two student experts.")
        if self.use_dual_model:
            if getattr(args, "dual_student_layout", "") not in {"", "unified_v1"}:
                raise ValueError("Unknown dual_student_layout")
            low = copy.deepcopy(high)  # preserve LoRA trainability and architecture
            if args.student_low_model:
                low.load_state_dict(load_model_state_file(args.student_low_model, map_location="cpu"), strict=True)
            _, objective = resolve_noise_objectives(args, kwargs["teacher_model"], high)
            # Both experts enter one optimizer, EMA, and sharding lifecycle.
            # Set the exact shifted/native boundary after schedule construction.
            routed = NoiseRoutedDenoiser(high, low, 0.5, objective.num_train_timesteps)
            kwargs["student_model"] = routed
            kwargs["optimizer"], kwargs["lr_scheduler"] = self._build_dual_optimizer(routed, args)
            if kwargs.get("runtime") is not None:
                kwargs["runtime"].student_model = routed
            self.student_high, self.student_low = high, low
            args.dual_student_layout = "unified_v1"
        super().__init__(**kwargs)
        self.denoising_step_list = self.args.denoising_step_list
        self.boundary_step_index = self.args.boundary_step_index
        self.num_distill_steps = len(self.denoising_step_list)
        self._precompute_sigmas()
        if self.use_dual_model:
            if not 0 <= self.boundary_step_index < self.num_distill_steps:
                raise ValueError("boundary_step_index must index the distillation schedule")
            self.student_model.boundary_ratio = float(self.distill_timesteps[self.boundary_step_index]) / self.num_train_timesteps
            if not 0 < self.student_model.boundary_ratio <= 1:
                raise ValueError("Dual-student boundary must be a positive native timestep")
        self._legacy_resume_directory = None
        self._dual_resume_metadata_validated = False

    @staticmethod
    def _build_dual_optimizer(combined_students, args):
        """Build optimizer that covers both high and low noise student params."""
        from training.utils.optimizers import build_optimizer
        from training.utils.schedulers import build_lr_scheduler

        optimizer = build_optimizer(
            combined_students,
            optimizer_type=args.optimizer,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_epsilon=args.adam_epsilon,
            muon_momentum=args.muon_momentum,
            muon_nesterov=args.muon_nesterov,
            muon_ns_steps=args.muon_ns_steps,
        )
        scheduler = build_lr_scheduler(
            optimizer,
            scheduler_type=args.lr_scheduler,
            warmup_steps=args.warmup_steps,
            total_steps=args.max_train_steps,
            min_lr_ratio=args.lr_min_ratio,
        )
        return optimizer, scheduler

    def _precompute_sigmas(self):
        """Pre-compute sigma schedule for the fixed distillation timesteps."""
        # The legacy default denotes four evenly spaced steps. Preserve that
        # intent on checkpoints with a different native time scale.
        if self.denoising_step_list == [1000, 750, 500, 250] and self.num_train_timesteps != 1000:
            self.denoising_step_list = [max(1, round(self.num_train_timesteps * fraction))
                                        for fraction in (1, 0.75, 0.5, 0.25)]
        if not self.denoising_step_list or any(s < 0 or s > self.num_train_timesteps for s in self.denoising_step_list):
            raise ValueError("denoising_step_list must contain native schedule timesteps in [0, num_train_timesteps]")
        if self.noise_objective.process == "vp":
            if self.args.sample_shift != 1:
                raise ValueError("VP training uses its checkpoint alpha schedule; sample_shift is flow-only")
            self.distill_timesteps = torch.tensor(self.denoising_step_list, device=self.device).clamp(max=self.num_train_timesteps - 1)
            self.distill_sigmas = self.noise_objective.coefficients(self.distill_timesteps, 1)[1]
            return
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
        timesteps = self.distill_timesteps[indices]
        return timesteps.to(self.device)

    def _forward_and_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Override forward to support per-sample dual-model routing."""
        latents = batch["latents"]
        bs = latents.shape[0]

        # Sample timesteps from fixed schedule
        timesteps = self._sample_timesteps(bs)
        # Create noisy latents
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_objective.add_noise(latents, noise, timesteps)
        teacher_input = self.prepare_teacher_input(batch, noisy_latents, timesteps)
        student_input = self.prepare_student_input(batch, noisy_latents, timesteps)
        teacher_output, student_output = self._run_teacher_student_pair(
            teacher_input=teacher_input, student_input=student_input, batch=batch,
            cache_extra={"timesteps": timesteps, "mode": "step_distill"},
        )
        teacher_output = self._teacher_target(teacher_output, noisy_latents, timesteps)
        return self.compute_distill_loss(teacher_output, student_output, batch, timesteps)

    def _teacher_target(self, output, noisy, timesteps):
        return self.noise_objective.convert_prediction(
            output, noisy, timesteps,
            self.teacher_noise_objective.prediction_type, self.noise_objective.prediction_type,
        )

    def _forward_and_loss_eval(self, batch):
        # Match the exact schedule and teacher prediction convention in training.
        return self._forward_and_loss(batch)

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

    def load_checkpoint(self, path: str):
        """Unified checkpoints use Base's complete model/optimizer lifecycle."""
        self._legacy_resume_directory = path if os.path.isdir(path) else os.path.dirname(path)
        self._dual_resume_metadata_validated = False
        try:
            if self.use_dual_model:
                metadata_path = os.path.join(self._legacy_resume_directory, "dual_student_state.pt")
                if os.path.isfile(metadata_path) or self._deepspeed_engine is not None or self._is_fsdp_wrapped(self.student_model):
                    metadata = self._load_checkpoint_file_all_ranks(metadata_path,
                        description="Load unified dual-student layout", weights_only=True)
                    expected = self._dual_checkpoint_metadata()
                    if metadata != expected:
                        raise ValueError(f"Dual-student checkpoint layout or routing schedule mismatch: {metadata}")
                    self._dual_resume_metadata_validated = True
            super().load_checkpoint(path)
        finally:
            self._legacy_resume_directory = None
            self._dual_resume_metadata_validated = False

    def _dual_checkpoint_metadata(self):
        return {"dual_student_layout": "unified_v1", "num_train_timesteps": self.num_train_timesteps,
                "distill_timesteps": self.distill_timesteps.detach().cpu().tolist(),
                "boundary_step_index": self.boundary_step_index}

    def save_checkpoint(self, step, output_dir):
        super().save_checkpoint(step, output_dir)
        if self.use_dual_model:
            path = os.path.join(output_dir, f"checkpoint-{step}", "dual_student_state.pt")
            self._atomic_save_rank0(path, self._dual_checkpoint_metadata, "Write unified dual-student layout")

    def _validate_standard_checkpoint_state(self, state):
        super()._validate_standard_checkpoint_state(state)
        if self.use_dual_model:
            recorded = state.get("args", {})
            layout = recorded.get("dual_student_layout", "") if isinstance(recorded, dict) else ""
            if layout not in {"", "unified_v1"}:
                raise ValueError(f"Unknown dual_student_layout in checkpoint: {layout}")
            if layout == "unified_v1" and not self._dual_resume_metadata_validated:
                raise ValueError("Unified dual checkpoint requires its dual_student_state.pt routing metadata")

    def _restore_standard_checkpoint_state(self, state):
        """Upgrade legacy DDP high-state + low-sidecar without touching disk."""
        if self.use_dual_model:
            recorded = state.get("args", {})
            layout = recorded.get("dual_student_layout", "") if isinstance(recorded, dict) else ""
            if layout not in {"", "unified_v1"}:
                raise ValueError(f"Unknown dual_student_layout in checkpoint: {layout}")
            if not layout:
                if not recorded.get("use_dual_model") or self._legacy_resume_directory is None:
                    raise ValueError("Legacy dual resume requires dual metadata and its low-noise sidecar")
                low_path = os.path.join(self._legacy_resume_directory, "student_low.pt")
                low = self._load_checkpoint_file_all_ranks(low_path,
                    description=f"Load legacy low-noise student {low_path}", weights_only=True)
                state = dict(state)
                state["student_model"] = {**{"high." + key: value for key, value in state["student_model"].items()},
                                          **{"low." + key: value for key, value in low.items()}}
                if self.ema is not None and "ema" in state:
                    ema = dict(state["ema"])
                    ema["shadow"] = {**{"high." + key: value for key, value in ema["shadow"].items()},
                        **{"low." + key: low[key].detach().clone()
                           for key, p in self.student_low.named_parameters() if p.requires_grad}}
                    state["ema"] = ema
                    logger.warning("Migrating legacy dual EMA: high history preserved; low EMA initializes from saved low weights because legacy checkpoints contain no low EMA history.")
                logger.info("Migrated legacy dual checkpoint to unified_v1; optimizer order remains high then low.")
        super()._restore_standard_checkpoint_state(state)

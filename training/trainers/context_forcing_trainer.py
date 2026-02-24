"""Context Forcing Distillation Trainer.

Implements memory-aware Context Forcing distillation where the student
learns to generate video conditioned on context (memory) frames,
preventing autoregressive error accumulation.

Key concepts:
- Context frames: teacher-denoised or GT frames used as conditioning
- Teacher forces context alignment: student sees teacher-generated context
  instead of its own (potentially drifted) generations
- Memory selection: choose which past frames to use as context
- Curriculum: gradually increase generation length during training

Two modes:
1. use_teacher_context=True (default): Teacher generates context frames first,
   then student denoises target frames conditioned on teacher's context.
   This bridges the train-test gap (Context Forcing paper's core idea).
2. use_teacher_context=False: Uses GT frames as context (Teacher Forcing).
   Simpler but has train-test mismatch.

References:
- Context Forcing: concept used in HY-WorldPlay for world model distillation
- Reconstituted Context Memory: FOV-based memory selection from HY-WorldPlay
"""

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from loguru import logger

from training.trainers.base_distill_trainer import BaseDistillTrainer


class ContextForcingTrainer(BaseDistillTrainer):
    """Trainer for Context Forcing distillation.

    The student generates video in chunks. For each chunk, context frames
    (from teacher or ground truth) are concatenated with noisy target frames.
    The student must denoise the target frames conditioned on the context.

    This prevents autoregressive drift by ensuring the student always
    conditions on high-quality context.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory_frames = self.args.memory_frames
        self.temporal_context_size = self.args.temporal_context_size
        self.curriculum_training = self.args.curriculum_training
        self.curriculum_stages = self.args.curriculum_stages
        self.generator_update_interval = self.args.generator_update_interval
        self.use_teacher_context = getattr(self.args, "use_teacher_context", True)

        # Track curriculum stage
        self.current_curriculum_stage = 0
        self.current_num_frames = self.curriculum_stages[0] if self.curriculum_training else self.args.num_frames
        self._curriculum_advanced = set()

    def _select_memory_frames(
        self,
        all_frames: torch.Tensor,
        current_chunk_idx: int,
        chunk_size: int,
    ) -> Optional[torch.Tensor]:
        """Select memory (context) frames for the current generation chunk.

        Returns:
            Memory context tensor (B, C, T_mem, H, W) or None for first chunk.
        """
        start_frame = current_chunk_idx * chunk_size
        mem_start = max(0, start_frame - self.memory_frames)
        mem_end = start_frame

        if mem_end <= mem_start:
            return None

        return all_frames[:, :, mem_start:mem_end]

    def _generate_teacher_context(
        self,
        batch: Dict[str, Any],
        latents: torch.Tensor,
        chunk_start: int,
        chunk_end: int,
    ) -> torch.Tensor:
        """Generate context frames using the teacher model.

        Simulates the teacher denoising the context region, so the student
        conditions on realistic (but imperfect) context during training.
        This bridges the train-test gap.

        Args:
            batch: Full batch dict.
            latents: Ground truth latents (B, C, T, H, W).
            chunk_start: Start frame index of context.
            chunk_end: End frame index of context (exclusive).

        Returns:
            Teacher-denoised context frames (B, C, T_ctx, H, W).
        """
        if chunk_start >= chunk_end:
            return None

        context_latents = latents[:, :, chunk_start:chunk_end]
        bs = context_latents.shape[0]

        # Add moderate noise to context (not fully noisy)
        noise_level = 0.3  # 30% noise — enough to simulate imperfect generation
        noise = torch.randn_like(context_latents)
        noisy_context = (1 - noise_level) * context_latents + noise_level * noise

        # Teacher denoises the context
        t_ctx = torch.full((bs,), noise_level * self.num_train_timesteps, device=self.device)
        teacher_input = self.prepare_teacher_input(batch, noisy_context, t_ctx)
        with torch.no_grad():
            teacher_v = self.teacher_model(**teacher_input)
            if isinstance(teacher_v, (tuple, list)):
                teacher_v = teacher_v[0]

        # Recover x_0 from velocity: x_0 = x_t - sigma * v
        teacher_context = noisy_context.float() - noise_level * teacher_v.float()
        return teacher_context.to(context_latents.dtype)

    def _build_context_input(
        self,
        memory_frames: Optional[torch.Tensor],
        noisy_target: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate memory context with noisy target frames."""
        if memory_frames is None:
            return noisy_target
        return torch.cat([memory_frames, noisy_target], dim=2)

    def _build_context_mask(
        self,
        num_memory: int,
        num_target: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Build mask: 0 for context frames, 1 for target frames (loss only on targets)."""
        mask = torch.zeros(batch_size, 1, num_memory + num_target, 1, 1, device=self.device)
        mask[:, :, num_memory:] = 1.0
        return mask

    def _build_context_timesteps(
        self,
        num_memory: int,
        target_timesteps: torch.Tensor,
        batch_size: int,
        num_target_frames: int,
    ) -> torch.Tensor:
        """Build per-frame timesteps: 0 for context, sampled for target.

        Args:
            num_memory: Number of memory/context frames.
            target_timesteps: Timesteps for target frames (B,) or (B, T_target).
            batch_size: Batch size.
            num_target_frames: Actual number of target frames (not self.temporal_context_size).

        Returns:
            Per-frame timesteps (B, T_total).
        """
        context_t = torch.zeros(batch_size, num_memory, device=self.device)

        if target_timesteps.dim() == 1:
            target_t = target_timesteps.unsqueeze(1).expand(batch_size, num_target_frames)
        else:
            target_t = target_timesteps

        return torch.cat([context_t, target_t], dim=1)

    def _forward_and_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Context Forcing forward pass.

        For each training sample:
        1. Split video into chunks
        2. For each chunk (except first):
           a. Select memory frames from teacher output or ground truth
           b. Noise the target chunk
           c. Concatenate context + noisy target
           d. Student denoises target conditioned on context
           e. Compute loss only on target frames
        """
        latents = batch["latents"]  # (B, C, T, H, W)
        bs = latents.shape[0]
        total_frames = latents.shape[2]

        # Determine chunk size based on curriculum
        effective_frames = min(self.current_num_frames, total_frames) if self.curriculum_training else total_frames
        chunk_size = min(self.temporal_context_size, effective_frames)
        num_chunks = max(1, effective_frames // chunk_size)

        total_loss = torch.tensor(0.0, device=self.device)
        num_loss_chunks = 0

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, total_frames)
            num_target = end - start
            target_frames = latents[:, :, start:end]

            # Select memory context
            if self.use_teacher_context and chunk_idx > 0:
                # Teacher generates context (bridges train-test gap)
                mem_start = max(0, start - self.memory_frames)
                memory_context = self._generate_teacher_context(batch, latents, mem_start, start)
            else:
                # GT context (simpler, teacher forcing)
                memory_context = self._select_memory_frames(latents, chunk_idx, chunk_size)

            num_memory = memory_context.shape[2] if memory_context is not None else 0

            # Sample timesteps for target frames
            timesteps = self._sample_timesteps(bs)
            sigmas = timesteps / self.num_train_timesteps
            sigmas_expanded = sigmas.view(bs, 1, 1, 1, 1)

            # Noise target frames
            noise = torch.randn_like(target_frames)
            noisy_target = (1 - sigmas_expanded) * target_frames + sigmas_expanded * noise

            # Build context-conditioned input
            model_input = self._build_context_input(memory_context, noisy_target)
            loss_mask = self._build_context_mask(num_memory, num_target, bs)
            frame_timesteps = self._build_context_timesteps(num_memory, timesteps, bs, num_target)

            # Teacher forward
            teacher_input = self.prepare_teacher_input(batch, model_input, frame_timesteps)
            with torch.no_grad():
                teacher_output = self.teacher_model(**teacher_input)
                if isinstance(teacher_output, (tuple, list)):
                    teacher_output = teacher_output[0]

            # Student forward (with its own input preparation)
            student_input = self.prepare_student_input(batch, model_input, frame_timesteps)
            student_output = self.student_model(**student_input)
            if isinstance(student_output, (tuple, list)):
                student_output = student_output[0]

            # Masked loss (only on target frames)
            diff = (student_output.float() - teacher_output.float().detach()) ** 2
            chunk_loss = (diff * loss_mask).sum() / loss_mask.sum().clamp(min=1)
            total_loss = total_loss + chunk_loss
            num_loss_chunks += 1

        return total_loss / max(num_loss_chunks, 1)

    def compute_distill_loss(
        self,
        teacher_output: torch.Tensor,
        student_output: torch.Tensor,
        batch: Dict[str, Any],
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Not used directly; _forward_and_loss handles full logic."""
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
        if "encoder_hidden_states" in batch:
            input_kwargs["encoder_hidden_states"] = batch["encoder_hidden_states"]
        if "image_cond" in batch:
            input_kwargs["image_cond"] = batch["image_cond"]
        if "camera_poses" in batch:
            input_kwargs["camera_poses"] = batch["camera_poses"]
        if "actions" in batch:
            input_kwargs["actions"] = batch["actions"]
        return input_kwargs

    def on_train_step_end(self, metrics):
        """Advance curriculum if needed."""
        if not self.curriculum_training:
            return

        for stage_idx, stage_frames in enumerate(self.curriculum_stages):
            stage_boundary = (stage_idx + 1) * (self.args.max_train_steps // len(self.curriculum_stages))
            if (
                self.global_step >= stage_boundary
                and stage_idx > self.current_curriculum_stage
                and stage_idx not in self._curriculum_advanced
            ):
                self._curriculum_advanced.add(stage_idx)
                self.current_curriculum_stage = stage_idx
                self.current_num_frames = stage_frames
                logger.info(
                    f"Curriculum advanced to stage {stage_idx}: "
                    f"num_frames={stage_frames}"
                )

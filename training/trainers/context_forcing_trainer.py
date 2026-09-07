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
        return self.select_runtime_memory_frames(
            all_frames=all_frames,
            current_chunk_idx=current_chunk_idx,
            chunk_size=chunk_size,
            memory_frames=self.memory_frames,
        )

    def _select_memory_frame_indices(
        self,
        total_frames: int,
        current_chunk_idx: int,
        chunk_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        return self.select_runtime_memory_indices(
            total_frames=total_frames,
            current_chunk_idx=current_chunk_idx,
            chunk_size=chunk_size,
            memory_frames=self.memory_frames,
            device=device,
        )

    def _generate_teacher_context(
        self,
        batch: Dict[str, Any],
        latents: torch.Tensor,
        frame_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Generate context frames using the teacher model.

        Simulates the teacher denoising the context region, so the student
        conditions on realistic (but imperfect) context during training.
        This bridges the train-test gap.

        Args:
            batch: Full batch dict.
            latents: Ground truth latents (B, C, T, H, W).
            frame_indices: Chronological context indices selected by the active
                memory policy. They may be sparse.

        Returns:
            Teacher-denoised context frames (B, C, T_ctx, H, W).
        """
        if frame_indices is None or frame_indices.numel() == 0:
            return None

        frame_indices = frame_indices.to(device=latents.device, dtype=torch.long)
        context_latents = latents.index_select(2, frame_indices)
        context_batch = self._slice_temporal_conditions(
            batch,
            frame_indices=frame_indices,
            total_frames=latents.shape[2],
        )
        bs = context_latents.shape[0]
        selected_indices = [int(index) for index in frame_indices.detach().cpu().tolist()]
        chunk_start = selected_indices[0]
        chunk_end = selected_indices[-1] + 1

        # Add moderate noise to context (not fully noisy)
        noise_level = 0.3  # 30% noise — enough to simulate imperfect generation
        noise = torch.randn_like(context_latents)
        noisy_context = (1 - noise_level) * context_latents + noise_level * noise

        def _produce_teacher_context() -> torch.Tensor:
            t_ctx = torch.full((bs,), noise_level * self.num_train_timesteps, device=self.device)
            teacher_input = self.prepare_teacher_input(context_batch, noisy_context, t_ctx)
            teacher_v = self.run_teacher(
                teacher_input,
                context_batch,
                cache_namespace="teacher_context_forward",
                cache_extra={"chunk_start": chunk_start, "chunk_end": chunk_end, "timesteps": t_ctx},
                allow_cache=False,
            )
            teacher_context = noisy_context.float() - noise_level * teacher_v.float()
            return teacher_context.to(context_latents.dtype)

        if self.runtime is not None and hasattr(self.runtime, "get_or_create_teacher_context"):
            return self.runtime.get_or_create_teacher_context(
                batch=context_batch,
                global_step=self.global_step,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                frame_indices=selected_indices,
                producer=_produce_teacher_context,
            )
        return _produce_teacher_context()

    @staticmethod
    def _chunk_ranges(num_frames: int, chunk_size: int) -> List[tuple[int, int]]:
        """Return loss-bearing chunks without dropping a non-divisible tail."""

        if num_frames <= 0 or chunk_size <= 0:
            return []
        return [
            (start, min(start + chunk_size, num_frames))
            for start in range(0, num_frames, chunk_size)
        ]

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
        chunk_ranges = self._chunk_ranges(effective_frames, chunk_size)

        total_loss = torch.tensor(0.0, device=self.device)
        num_loss_chunks = 0

        for chunk_idx, (start, end) in enumerate(chunk_ranges):
            if (
                self.use_teacher_context
                and chunk_idx + 1 < len(chunk_ranges)
                and self.runtime is not None
                and hasattr(self.runtime, "prefetch_teacher_context")
            ):
                next_indices = self._select_memory_frame_indices(
                    effective_frames, chunk_idx + 1, chunk_size, latents.device
                )
                if next_indices is not None:
                    next_batch = self._slice_temporal_conditions(batch, next_indices, total_frames)
                    self.runtime.prefetch_teacher_context(
                        next_batch, self.global_step, next_indices.detach().cpu().tolist()
                    )
            num_target = end - start
            target_frames = latents[:, :, start:end]

            # Select memory context
            memory_indices = self._select_memory_frame_indices(
                total_frames=effective_frames,
                current_chunk_idx=chunk_idx,
                chunk_size=chunk_size,
                device=latents.device,
            )
            if self.use_teacher_context and memory_indices is not None:
                # Teacher generates context (bridges train-test gap)
                memory_context = self._generate_teacher_context(batch, latents, memory_indices)
            elif memory_indices is not None:
                # GT context uses the exact same dense/sparse selection policy.
                memory_context = latents.index_select(2, memory_indices)
            else:
                memory_context = None

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

            target_indices = torch.arange(start, end, device=latents.device, dtype=torch.long)
            packed_indices = (
                torch.cat([memory_indices, target_indices])
                if memory_indices is not None
                else target_indices
            )
            chunk_batch = self._slice_temporal_conditions(
                batch,
                frame_indices=packed_indices,
                total_frames=total_frames,
            )

            teacher_input = self.prepare_teacher_input(chunk_batch, model_input, frame_timesteps)
            student_input = self.prepare_student_input(chunk_batch, model_input, frame_timesteps)
            teacher_output, student_output = self._run_teacher_student_pair(
                teacher_input=teacher_input,
                student_input=student_input,
                batch=chunk_batch,
                cache_extra={"mode": "context_chunk", "chunk_idx": chunk_idx, "timesteps": frame_timesteps},
            )

            chunk_loss = self.compute_supervision_loss(
                prediction=student_output,
                target=teacher_output.detach(),
                mask=loss_mask,
            )
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
        """Advance curriculum if needed."""
        if not self.curriculum_training:
            return

        num_stages = len(self.curriculum_stages)
        for stage_idx, stage_frames in enumerate(self.curriculum_stages[1:], start=1):
            stage_boundary = max(
                1,
                math.ceil(stage_idx * self.args.max_train_steps / num_stages),
            )
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

    def save_checkpoint(self, step: int, output_dir: str):
        """Save curriculum progress alongside the base trainer state."""
        import os

        super().save_checkpoint(step, output_dir)
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
        context_path = os.path.join(ckpt_dir, "context_state.pt")
        self._atomic_save_rank0(
            context_path,
            lambda: {
                "current_curriculum_stage": self.current_curriculum_stage,
                "current_num_frames": self.current_num_frames,
                "curriculum_advanced": sorted(self._curriculum_advanced),
            },
            f"Write context-forcing checkpoint {context_path}",
        )

    def load_checkpoint(self, path: str):
        """Restore curriculum progress strictly on every rank."""
        import os

        super().load_checkpoint(path)
        ckpt_dir = path if os.path.isdir(path) else os.path.dirname(path)
        context_path = os.path.join(ckpt_dir, "context_state.pt")
        state = self._load_checkpoint_file_all_ranks(
            context_path,
            description=f"Load context-forcing checkpoint {context_path}",
            weights_only=False,
        )

        def _restore_context_state():
            if not isinstance(state, dict):
                raise TypeError(
                    f"Expected context-forcing checkpoint dict, got {type(state).__name__}"
                )
            required = {
                "current_curriculum_stage",
                "current_num_frames",
                "curriculum_advanced",
            }
            missing = sorted(required.difference(state))
            if missing:
                raise KeyError(f"Context-forcing checkpoint is missing required keys: {missing}")

            stage = int(state["current_curriculum_stage"])
            if self.curriculum_training:
                if stage < 0 or stage >= len(self.curriculum_stages):
                    raise ValueError(
                        f"Context checkpoint stage={stage} is outside the configured "
                        f"range [0, {len(self.curriculum_stages) - 1}]"
                    )
                expected_frames = int(self.curriculum_stages[stage])
            else:
                if stage != 0:
                    raise ValueError("A disabled curriculum must resume at stage 0")
                expected_frames = int(self.args.num_frames)
            if int(state["current_num_frames"]) != expected_frames:
                raise ValueError("Context checkpoint frame curriculum does not match config")

            advanced = {int(value) for value in state["curriculum_advanced"]}
            if any(value < 0 or value >= len(self.curriculum_stages) for value in advanced):
                raise ValueError("Context checkpoint contains an invalid advanced stage")
            self.current_curriculum_stage = stage
            self.current_num_frames = expected_frames
            self._curriculum_advanced = advanced

        self._run_all_ranks_or_raise(
            _restore_context_state,
            f"Restore context-forcing checkpoint {context_path}",
        )
        logger.info(
            f"Context-forcing curriculum restored at stage {self.current_curriculum_stage}."
        )

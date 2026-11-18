"""Classifier-free guidance distillation.

The teacher is evaluated twice at the same noisy latent.  The conditional and
unconditional predictions are combined with the standard CFG rule and the
student learns that guided target with one conditional forward pass.  Raw
training batches receive ``uncond_*`` embeddings from ``DiffusersRawBatchEncoder``
and cached batches must provide the same explicit fields.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from training.trainers.base_distill_trainer import BaseDistillTrainer


class CFGDistillTrainer(BaseDistillTrainer):
    """Distill a guided teacher into a single conditional student forward."""

    _TEXT_CONDITION_KEYS = (
        "encoder_hidden_states",
        "encoder_attention_mask",
        "pooled_projections",
        "encoder_hidden_states_2",
        "encoder_attention_mask_2",
        "audio_encoder_hidden_states",
        "audio_encoder_attention_mask",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cfg_scale = float(getattr(self.args, "cfg_scale", 4.0))
        self.unconditional_prefix = str(getattr(self.args, "cfg_unconditional_prefix", "uncond_"))
        if self.cfg_scale < 1.0:
            raise ValueError("cfg_scale must be at least 1.0")

    def _unconditional_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(batch)
        replaced = 0
        has_conditional_text = any(key in batch for key in self._TEXT_CONDITION_KEYS)
        for key in self._TEXT_CONDITION_KEYS:
            unconditional_key = f"{self.unconditional_prefix}{key}"
            if unconditional_key in batch:
                result[key] = batch[unconditional_key]
                replaced += 1
        if replaced == 0:
            if not has_conditional_text:
                # An explicitly unconditional toy/native model has no text
                # branch to duplicate. Keeping the inputs unchanged makes the
                # trainer useful for model-contract tests and unconditional
                # video generators without fabricating conditioning tensors.
                return result
            raise ValueError(
                "CFG distillation requires explicit unconditional text embeddings. "
                "Provide uncond_encoder_hidden_states (and matching masks/pooled fields) "
                "or use raw-video mode so the batch encoder can create them."
            )
        return result

    def _guided_teacher_target(
        self,
        conditional: torch.Tensor,
        unconditional: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        conditional = self.noise_objective.convert_prediction(
            conditional,
            noisy_latents,
            timesteps,
            self.teacher_noise_objective.prediction_type,
            self.noise_objective.prediction_type,
        )
        unconditional = self.noise_objective.convert_prediction(
            unconditional,
            noisy_latents,
            timesteps,
            self.teacher_noise_objective.prediction_type,
            self.noise_objective.prediction_type,
        )
        return unconditional + self.cfg_scale * (conditional - unconditional)

    def _forward_and_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        latents = batch["latents"]
        batch_size = latents.shape[0]
        timesteps = self._sample_timesteps(batch_size)
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_objective.add_noise(latents, noise, timesteps)

        conditional_input = self.prepare_teacher_input(batch, noisy_latents, timesteps)
        unconditional_batch = self._unconditional_batch(batch)
        unconditional_input = self.prepare_teacher_input(unconditional_batch, noisy_latents, timesteps)
        student_input = self.prepare_student_input(batch, noisy_latents, timesteps)

        conditional = self.run_teacher(
            conditional_input,
            batch,
            cache_namespace="cfg_teacher_conditional",
            cache_extra={"timesteps": timesteps, "cfg_branch": "conditional"},
        )
        unconditional = self.run_teacher(
            unconditional_input,
            unconditional_batch,
            cache_namespace="cfg_teacher_unconditional",
            cache_extra={"timesteps": timesteps, "cfg_branch": "unconditional"},
        )
        student = self.run_student(student_input, batch)
        target = self._guided_teacher_target(
            conditional, unconditional, noisy_latents, timesteps
        ).detach()
        return self.compute_supervision_loss(student, target, mask=batch.get("loss_mask"))

    def compute_distill_loss(self, teacher_output, student_output, batch, timesteps):
        # The full guided target is formed in _forward_and_loss.
        return self.compute_supervision_loss(student_output, teacher_output.detach(), batch.get("loss_mask"))

    def prepare_teacher_input(self, batch, noisy_latents, timesteps):
        return self._augment_model_input(
            {"hidden_states": noisy_latents, "timestep": timesteps}, batch
        )

"""Phased and truncated consistency training variants.

These trainers share the audited TCD/LCD flow-space target implementation while
exposing the sampling contracts that distinguish PCM and TCM.  They are useful
for video latents because phase or truncation boundaries can be chosen in the
native noise schedule instead of being hidden in a generic adjacent-pair
sampler.
"""

from __future__ import annotations

import copy

import torch

from training.trainers.consistency_distill_trainer import ConsistencyDistillTrainer
from training.utils.model_output import extract_prediction_tensor


class PCMTrainer(ConsistencyDistillTrainer):
    """Phased Consistency Model training with phase-aware trajectory pairs."""

    def _sample_consistency_pair(self, batch_size: int):
        phases = int(self.args.pcm_num_phases)
        phase = torch.randint(phases, (batch_size,), device=self.device)
        low = phase.float() / phases
        high = (phase.float() + 1.0) / phases
        width = high - low
        if getattr(self.args, "pcm_phase_sampling", "uniform") == "boundary":
            # Concentrate samples near the noisy end of each phase while
            # retaining coverage of the complete sub-trajectory.
            offset = torch.rand(batch_size, device=self.device).square()
            t_n_sigma = high - offset * width
        else:
            t_n_sigma = low + torch.rand(batch_size, device=self.device) * width
        t_n1_sigma = low
        return t_n_sigma * self.num_train_timesteps, t_n1_sigma * self.num_train_timesteps
class TCMTrainer(ConsistencyDistillTrainer):
    """Truncated Consistency Model training with an explicit two-stage switch."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tcm_stage = int(getattr(self.args, "tcm_stage", 2))
        self.tcm_boundary_weight = float(getattr(self.args, "tcm_boundary_weight", 1.0))
        if self.tcm_stage == 2 and not getattr(self.args, "resume_from", ""):
            raise ValueError("TCM stage 2 requires resume_from pointing to a completed stage-1 checkpoint")
        self.tcm_boundary_model = copy.deepcopy(self.ema_model)
        self.tcm_boundary_model.eval()
        for parameter in self.tcm_boundary_model.parameters():
            parameter.requires_grad = False
        self._sync_tcm_boundary()

    @torch.no_grad()
    def _sync_tcm_boundary(self):
        """Freeze the stage-1 boundary model used by stage 2."""
        source = self._unwrap_model(self.ema_model)
        self.tcm_boundary_model.load_state_dict(source.state_dict(), strict=True)

    def _initialize_parallel_targets(self):
        super()._initialize_parallel_targets()
        # The boundary model is a frozen inference copy. Keep it on the active
        # execution device so stage-2 can evaluate the declared boundary even
        # when the consistency target itself is sharded or offloaded.
        self.tcm_boundary_model.to(self.device)
        self._sync_tcm_boundary()

    def load_checkpoint(self, path: str):
        super().load_checkpoint(path)
        if self.tcm_stage == 2:
            self._sync_tcm_boundary()

    def _forward_and_loss(self, batch):
        loss = super()._forward_and_loss(batch)
        if self.tcm_stage != 2 or self.tcm_boundary_weight == 0:
            return loss

        latents = batch["latents"]
        batch_size = latents.shape[0]
        boundary_t = torch.full(
            (batch_size,),
            float(self.args.tcm_truncation_start * self.num_train_timesteps),
            device=self.device,
        )
        sigma = boundary_t / self.num_train_timesteps
        noise = torch.randn_like(latents)
        sigma_expanded = sigma.view(batch_size, *([1] * (latents.ndim - 1)))
        noisy = self.noise_objective.add_noise(latents, noise, boundary_t)
        student_input = self.prepare_student_input(batch, noisy, boundary_t)
        student_output = self.run_student(student_input, batch, tag="tcm boundary student")
        student_x0 = self._consistency_function(student_output, noisy, sigma_expanded)
        boundary_input = self.prepare_teacher_input(batch, noisy, boundary_t)
        with torch.no_grad():
            boundary_output = extract_prediction_tensor(
                self.tcm_boundary_model(**boundary_input), tag="tcm stage-1 boundary"
            )
        boundary_x0 = self._consistency_function(boundary_output, noisy, sigma_expanded)
        boundary_loss = self._compute_consistency_loss(student_x0, boundary_x0.detach())
        return loss + self.tcm_boundary_weight * boundary_loss

    def _sample_consistency_pair(self, batch_size: int):
        stage = int(getattr(self.args, "tcm_stage", 2))
        if stage == 1:
            start, end = 0.0, 1.0
        else:
            start = float(self.args.tcm_truncation_start)
            end = float(self.args.tcm_truncation_end)
        t_n_sigma = start + torch.rand(batch_size, device=self.device) * (end - start)
        delta_sigma = self._get_delta() / self.num_train_timesteps
        t_n1_sigma = torch.maximum(
            torch.full_like(t_n_sigma, start), t_n_sigma - delta_sigma
        )
        return t_n_sigma * self.num_train_timesteps, t_n1_sigma * self.num_train_timesteps

"""Teacher-velocity MeanFlow distillation with a true two-time JVP target."""
from __future__ import annotations

from contextlib import contextmanager
import torch

from training.trainers.base_distill_trainer import BaseDistillTrainer
from training.utils.meanflow import (
    deterministic_meanflow_forward, eager_meanflow_model, meanflow_jvp, meanflow_target, require_meanflow_model,
)


class MeanFlowDistillTrainer(BaseDistillTrainer):
    """Learn average velocity over [r,t] from a frozen instantaneous teacher.

The diagonal r=t reduces to velocity distillation. Off the diagonal the exact
directional derivative uses tangent (teacher_velocity, 1, 0), and only the
student primal participates in parameter backpropagation. This is a teacher-
guided MeanFlow objective, not the data-velocity training recipe in the paper.
"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._require_velocity_prediction("MeanFlow distillation")
        require_meanflow_model(self._unwrap_model(self.student_model))

    def prepare_teacher_input(self, batch, noisy_latents, timesteps):
        return self._augment_model_input({"hidden_states": noisy_latents, "timestep": timesteps}, batch)

    def compute_distill_loss(self, teacher_output, student_output, batch, timesteps):
        return self.compute_supervision_loss(student_output, teacher_output.detach(), mask=batch.get("loss_mask"))

    def _forward_and_loss_eval(self, batch):
        return self._forward_and_loss(batch)

    def _sample_time_pair(self, batch_size):
        times = torch.randn(batch_size, 2, device=self.device)
        times = torch.sigmoid(times * self.args.logit_normal_std + self.args.logit_normal_mean)
        t, r = times.max(dim=1).values, times.min(dim=1).values
        diagonal = torch.rand(batch_size, device=self.device) < self.args.meanflow_diagonal_ratio
        return t, torch.where(diagonal, t, r)

    @contextmanager
    def _full_jvp_student(self):
        """Read complete weights outside reducer hooks for the detached JVP."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if isinstance(self.student_model, FSDP):
            with FSDP.summon_full_params(self.student_model, recurse=True, writeback=False):
                # Nested FSDP wrappers cannot run their normal forward while
                # in SUMMON_FULL_PARAMS. The detached target uses the already
                # materialized underlying modules, then restores every wrapper
                # before the trainable forward/reducer is entered.
                raw = eager_meanflow_model(self._unwrap_model(self.student_model))
                replacements = []
                def unwrap_children(parent):
                    for name, child in list(parent.named_children()):
                        if isinstance(child, FSDP):
                            replacements.append((parent, name, child))
                            setattr(parent, name, child.module)
                            child = child.module
                        unwrap_children(child)
                unwrap_children(raw)
                try:
                    yield raw
                finally:
                    for parent, name, child in reversed(replacements):
                        setattr(parent, name, child)
        elif self._deepspeed_engine is not None and self.args.deepspeed_stage == 3:
            import deepspeed
            from training.utils.zero3_context import suspend_zero3_parameter_hooks
            raw = eager_meanflow_model(self._unwrap_model(self.student_model))
            with deepspeed.zero.GatheredParameters(list(raw.parameters()), modifier_rank=None):
                with suspend_zero3_parameter_hooks(self._deepspeed_engine):
                    yield raw
        else:
            yield eager_meanflow_model(self._unwrap_model(self.student_model))

    def _forward_and_loss(self, batch):
        clean = batch["latents"]
        t, r = self._sample_time_pair(clean.shape[0])
        sigma = t.reshape((t.shape[0],) + (1,) * (clean.ndim - 1))
        noisy = ((1 - sigma) * clean + sigma * torch.randn_like(clean)).to(clean.dtype)
        timestep = t * self.num_train_timesteps
        teacher_input = self.prepare_teacher_input(batch, noisy, timestep)
        velocity = self.run_teacher(teacher_input, batch, cache_extra={"mode": "meanflow", "t": timestep})
        student_input = self.prepare_student_input(batch, noisy, timestep)
        conditions = {key: value for key, value in student_input.items() if key not in {"hidden_states", "timestep"}}
        raw_student = self._unwrap_model(self.student_model)
        with deterministic_meanflow_forward(raw_student):
            with self._full_jvp_student() as full_student:
                derivative = meanflow_jvp(full_student, noisy, t, r, velocity, conditions, self.num_train_timesteps)
            target = meanflow_target(velocity, derivative, t, r)
            student_input["timestep_r"] = r * self.num_train_timesteps
            prediction = self.run_student(student_input, batch)
        loss_per_sample = (prediction.float() - target).square()
        mask = batch.get("loss_mask")
        if mask is not None:
            # Reuse the shared mask contract; adaptive weighting remains per
            # sample rather than allowing padded frames to dominate the norm.
            losses = []
            for index in range(clean.shape[0]):
                item_mask = mask[index:index + 1] if mask.ndim and mask.shape[0] == clean.shape[0] else mask
                losses.append(self.compute_supervision_loss(prediction[index:index + 1],
                              target[index:index + 1], mask=item_mask))
            loss_per_sample = torch.stack(losses)
        else:
            loss_per_sample = loss_per_sample.flatten(1).mean(1)
        weights = (loss_per_sample.detach() + self.args.meanflow_norm_eps).pow(self.args.meanflow_norm_power)
        return (loss_per_sample / weights).mean()

    @torch.no_grad()
    def generate_samples(self, batch, num_steps=1):
        """MeanFlow sampling must condition on both endpoints at every step."""
        if num_steps < 1:
            raise ValueError("MeanFlow sample steps must be positive")
        latents = torch.randn_like(batch["latents"])
        schedule = torch.linspace(1, 0, num_steps + 1, device=latents.device)
        with deterministic_meanflow_forward(self._unwrap_model(self.student_model)):
            for index in range(num_steps):
                t = schedule[index].expand(latents.shape[0]) * self.num_train_timesteps
                r = schedule[index + 1].expand_as(t) * self.num_train_timesteps
                model_input = self.prepare_student_input(batch, latents, t)
                model_input["timestep_r"] = r
                prediction = self.run_student(model_input, batch)
                latents = latents - (schedule[index] - schedule[index + 1]) * prediction
        return latents

"""Additional video distillation objectives.

The implementations in this module keep the model-independent parts of the
recent video training recipes explicit.  Model-specific action/control packing
continues to use the shared adapter hooks from :mod:`base_distill_trainer`.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from training.trainers.base_distill_trainer import BaseDistillTrainer
from training.trainers.consistency_distill_trainer import ConsistencyDistillTrainer
from training.trainers.dmd_distill_trainer import DMDDistillTrainer
from training.trainers.stream_distill_trainer import StreamDistillTrainer


class SelfForcingTrainer(StreamDistillTrainer):
    """Self-forcing rollout training with generated history in later windows."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.self_forcing_rollout_steps = int(getattr(self.args, "self_forcing_rollout_steps", 1))
        self.self_forcing_context_ratio = float(getattr(self.args, "self_forcing_context_ratio", 1.0))
        if self.self_forcing_rollout_steps < 1:
            raise ValueError("self_forcing_rollout_steps must be positive")
        if not 0.0 <= self.self_forcing_context_ratio <= 1.0:
            raise ValueError("self_forcing_context_ratio must lie in [0, 1]")

    def _forward_and_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        # StreamDistillTrainer already performs autoregressive generated-context
        # unrolling.  Repeating the rollout creates the self-forcing curriculum
        # while preserving gradients through the generated overlap.
        total = torch.zeros((), device=self.device)
        for _ in range(self.self_forcing_rollout_steps):
            total = total + super()._forward_and_loss(batch)
        return total / self.self_forcing_rollout_steps

    def _compute_window_loss(self, latents_window, batch, frame_indices, total_frames,
                             generated_context=None, return_generated=False):
        if generated_context is not None and self.self_forcing_context_ratio < 1.0:
            context_frames = generated_context.shape[2]
            ground_truth_context = latents_window[:, :, :context_frames].to(generated_context)
            ratio = self.self_forcing_context_ratio
            generated_context = ratio * generated_context + (1.0 - ratio) * ground_truth_context
        return super()._compute_window_loss(
            latents_window, batch, frame_indices, total_frames,
            generated_context=generated_context, return_generated=return_generated,
        )


class CausalForcingTrainer(SelfForcingTrainer):
    """Self-forcing with an explicit causal temporal attention contract."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.causal_attention = True
        if hasattr(self.args, "causal_attention"):
            self.args.causal_attention = True
        context_frames = int(getattr(self.args, "causal_forcing_context_frames", 0))
        if context_frames:
            if context_frames >= self.window_size:
                raise ValueError("causal_forcing_context_frames must be smaller than window_size")
            self.overlap_frames = context_frames


class RCMTrainer(ConsistencyDistillTrainer):
    """Continuous-time consistency regularization for video flow models."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rcm_min_sigma = float(getattr(self.args, "rcm_min_sigma", 0.001))
        self.rcm_max_sigma = float(getattr(self.args, "rcm_max_sigma", 0.999))
        self.rcm_endpoint_weight = float(getattr(self.args, "rcm_endpoint_weight", 0.1))
        if not 0.0 <= self.rcm_min_sigma < self.rcm_max_sigma <= 1.0:
            raise ValueError("rcm interval must satisfy 0 <= min < max <= 1")
        if self.rcm_endpoint_weight < 0:
            raise ValueError("rcm_endpoint_weight must be non-negative")

    def _sample_consistency_pair(self, batch_size: int):
        low = torch.rand(batch_size, device=self.device) * (self.rcm_max_sigma - self.rcm_min_sigma)
        low = low + self.rcm_min_sigma
        high = torch.rand(batch_size, device=self.device) * (self.rcm_max_sigma - low) + low
        return high * self.num_train_timesteps, low * self.num_train_timesteps

    def _compute_consistency_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = super()._compute_consistency_loss(prediction, target)
        if self.rcm_endpoint_weight == 0:
            return loss
        # The endpoint penalty anchors the continuous mapping at the clean
        # boundary and prevents pair sampling from drifting away from t=0.
        endpoint = F.mse_loss(prediction.float().mean(dim=tuple(range(1, prediction.ndim))),
                              target.float().mean(dim=tuple(range(1, target.ndim))))
        return loss + self.rcm_endpoint_weight * endpoint


class SCMTrainer(RCMTrainer):
    """Score-consistency variant sharing the continuous pair contract."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scm_score_weight = float(getattr(self.args, "scm_score_weight", 1.0))
        if self.scm_score_weight < 0:
            raise ValueError("scm_score_weight must be non-negative")

    def _compute_consistency_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.scm_score_weight * super()._compute_consistency_loss(prediction, target)


class DiagonalDistillTrainer(StreamDistillTrainer):
    """Diagonal distillation with a monotone frame/noise schedule."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diagonal_slope = float(getattr(self.args, "diagonal_slope", 1.0))
        self.diagonal_jitter = float(getattr(self.args, "diagonal_jitter", 0.05))
        if self.diagonal_slope <= 0 or self.diagonal_jitter < 0:
            raise ValueError("diagonal_slope must be positive and diagonal_jitter non-negative")

    def _sample_per_frame_timesteps(self, batch_size: int, num_frames: int) -> torch.Tensor:
        anchor = torch.rand(batch_size, 1, device=self.device)
        offsets = torch.arange(num_frames, device=self.device, dtype=anchor.dtype).view(1, -1)
        offsets = offsets / max(1, num_frames - 1) * self.diagonal_slope
        jitter = torch.randn(batch_size, num_frames, device=self.device) * self.diagonal_jitter / max(1, num_frames)
        values = (anchor + offsets + jitter).clamp(0.0, 1.0)
        values, _ = torch.sort(values, dim=1)
        return values * self.num_train_timesteps


class ReflowTrainer(BaseDistillTrainer):
    """Rectified-flow reflow objective using paired clean/noise endpoints."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._require_velocity_prediction("Reflow")
        self.reflow_endpoint_mix = float(getattr(self.args, "reflow_endpoint_mix", 1.0))
        if not 0.0 <= self.reflow_endpoint_mix <= 1.0:
            raise ValueError("reflow_endpoint_mix must lie in [0, 1]")

    def _forward_and_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        latents = batch["latents"]
        bs = latents.shape[0]
        timesteps = self._sample_timesteps(bs)
        noise = torch.randn_like(latents)
        noisy = self.noise_objective.add_noise(latents, noise, timesteps)
        student_input = self.prepare_student_input(batch, noisy, timesteps)
        student = self.run_student(student_input, batch)
        straight_target = noise - latents
        target = straight_target
        if self.reflow_endpoint_mix < 1.0:
            teacher_input = self.prepare_teacher_input(batch, noisy, timesteps)
            teacher = self.run_teacher(teacher_input, batch, cache_namespace="reflow_teacher")
            target = self.reflow_endpoint_mix * straight_target + (1.0 - self.reflow_endpoint_mix) * teacher.detach()
        return self.compute_supervision_loss(student, target, mask=batch.get("loss_mask"))

    def compute_distill_loss(self, teacher_output, student_output, batch, timesteps):
        return self.compute_supervision_loss(student_output, teacher_output, batch.get("loss_mask"))

    def prepare_teacher_input(self, batch, noisy_latents, timesteps):
        return self._augment_model_input({"hidden_states": noisy_latents, "timestep": timesteps}, batch)


class _VelocityPairTrainer(BaseDistillTrainer):
    """Shared neighboring-time flow contract for research extensions."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._require_velocity_prediction(type(self).__name__)

    def prepare_teacher_input(self, batch, noisy_latents, timesteps):
        return self._augment_model_input({"hidden_states": noisy_latents, "timestep": timesteps}, batch)

    def compute_distill_loss(self, teacher_output, student_output, batch, timesteps):
        return self.compute_supervision_loss(student_output, teacher_output, batch.get("loss_mask"))

    def _sample_pair(self, batch):
        latents = batch["latents"]
        bs = latents.shape[0]
        high = self._sample_timesteps(bs)
        gap = float(getattr(self.args, "tmd_transition_gap", 0.05)) * self.num_train_timesteps
        low = (high - gap).clamp(min=1.0)
        noise = torch.randn_like(latents)
        return high, low, self.noise_objective.add_noise(latents, noise, high), self.noise_objective.add_noise(latents, noise, low)


class TMDTrainer(_VelocityPairTrainer):
    """Transition matching with a trapezoidal teacher transition target."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tmd_transition_gap = float(getattr(self.args, "tmd_transition_gap", 0.05))
        self.tmd_transition_weight = float(getattr(self.args, "tmd_transition_weight", 1.0))
        if not 0.0 < self.tmd_transition_gap <= 1.0 or self.tmd_transition_weight <= 0:
            raise ValueError("TMD gap must lie in (0, 1] and weight must be positive")

    def _forward_and_loss(self, batch):
        high, low, high_latents, low_latents = self._sample_pair(batch)
        with torch.no_grad():
            high_teacher = self.run_teacher(self.prepare_teacher_input(batch, high_latents, high), batch,
                                             cache_namespace="tmd_teacher_high", cache_extra={"timesteps": high})
            low_teacher = self.run_teacher(self.prepare_teacher_input(batch, low_latents, low), batch,
                                            cache_namespace="tmd_teacher_low", cache_extra={"timesteps": low})
        student = self.run_student(self.prepare_student_input(batch, high_latents, high), batch, tag="tmd student")
        target = 0.5 * (high_teacher + low_teacher)
        return self.tmd_transition_weight * self.compute_supervision_loss(student, target.detach(), batch.get("loss_mask"))


class SiDDiTTrainer(_VelocityPairTrainer):
    """Score-identity distillation using pointwise and finite-difference losses."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sid_identity_weight = float(getattr(self.args, "sid_identity_weight", 1.0))
        self.sid_pointwise_weight = float(getattr(self.args, "sid_pointwise_weight", 1.0))
        if min(self.sid_identity_weight, self.sid_pointwise_weight) < 0 or (self.sid_identity_weight + self.sid_pointwise_weight) == 0:
            raise ValueError("SiD-DiT weights must be non-negative and not both zero")

    def _forward_and_loss(self, batch):
        high, low, high_latents, low_latents = self._sample_pair(batch)
        with torch.no_grad():
            teacher_high = self.run_teacher(self.prepare_teacher_input(batch, high_latents, high), batch,
                                             cache_namespace="sid_teacher_high", cache_extra={"timesteps": high})
            teacher_low = self.run_teacher(self.prepare_teacher_input(batch, low_latents, low), batch,
                                            cache_namespace="sid_teacher_low", cache_extra={"timesteps": low})
        student_high = self.run_student(self.prepare_student_input(batch, high_latents, high), batch, tag="sid student high")
        student_low = self.run_student(self.prepare_student_input(batch, low_latents, low), batch, tag="sid student low")
        pointwise = self.compute_supervision_loss(student_high, teacher_high.detach(), batch.get("loss_mask"))
        delta = (high - low).view(high.shape[0], *([1] * (student_high.ndim - 1))).clamp_min(1.0)
        identity = self.compute_supervision_loss((student_high - student_low) / delta,
                                                 (teacher_high - teacher_low).detach() / delta,
                                                 batch.get("loss_mask"))
        return self.sid_pointwise_weight * pointwise + self.sid_identity_weight * identity


class VSDRLTrainer(_VelocityPairTrainer):
    """Variational score distillation with an optional reward/policy surrogate."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vsd_kl_weight = float(getattr(self.args, "vsd_kl_weight", 1.0))
        self.vsd_rl_weight = float(getattr(self.args, "vsd_rl_weight", 0.1))
        self.vsd_reward_baseline_momentum = float(getattr(self.args, "vsd_reward_baseline_momentum", 0.9))
        if min(self.vsd_kl_weight, self.vsd_rl_weight) < 0 or not 0 <= self.vsd_reward_baseline_momentum < 1:
            raise ValueError("Invalid VSD/RL weights or reward baseline momentum")
        self._vsd_reward_baseline = 0.0

    def _collect_resume_state(self):
        state = super()._collect_resume_state()
        state["vsd_reward_baseline"] = float(self._vsd_reward_baseline)
        return state

    def _restore_resume_state(self, resume_state):
        baseline = 0.0 if resume_state is None else resume_state.get("vsd_reward_baseline", 0.0)
        super()._restore_resume_state(resume_state)
        self._vsd_reward_baseline = float(baseline)

    def _forward_and_loss(self, batch):
        latents = batch["latents"]
        bs = latents.shape[0]
        timesteps = self._sample_timesteps(bs)
        noise = torch.randn_like(latents)
        noisy = self.noise_objective.add_noise(latents, noise, timesteps)
        with torch.no_grad():
            teacher = self.run_teacher(self.prepare_teacher_input(batch, noisy, timesteps), batch,
                                       cache_namespace="vsd_teacher", cache_extra={"timesteps": timesteps})
        student = self.run_student(self.prepare_student_input(batch, noisy, timesteps), batch, tag="vsd student")
        kl = self.compute_supervision_loss(student, teacher.detach(), batch.get("loss_mask"))
        reward = batch.get("reward", batch.get("preference_reward"))
        if reward is None:
            reward = -((student.detach() - teacher.detach()) ** 2).flatten(1).mean(dim=1)
        else:
            reward = torch.as_tensor(reward, device=student.device, dtype=student.dtype).reshape(-1)
            if reward.numel() == 1:
                reward = reward.expand(bs)
            if reward.numel() != bs:
                raise ValueError("VSD/RL reward must contain one value per batch element")
        # Use a batch-local baseline so exact checkpoint continuation does not
        # depend on an unobserved reward history. The momentum field remains a
        # compatibility knob for callers that provide an external baseline.
        mean_reward = float(reward.detach().mean())
        self._vsd_reward_baseline = mean_reward
        advantage = reward.detach() - mean_reward
        policy_score = (student.float() * noise.float()).flatten(1).mean(dim=1)
        rl = -(advantage * policy_score).mean()
        return self.vsd_kl_weight * kl + self.vsd_rl_weight * rl


class DataFreeDMDTrainer(DMDDistillTrainer):
    """DMD variant that synthesizes the latent source when a dataset has none."""

    def __init__(self, **kwargs):
        args = kwargs.get("args")
        super().__init__(**kwargs)
        shape = getattr(args, "data_free_latent_shape", None) if args is not None else None
        if shape is not None and len(shape) not in (4, 5):
            raise ValueError("data_free_latent_shape must have four or five dimensions")
        self.data_free_latent_shape = tuple(int(v) for v in shape) if shape else None
        for name, default in (("data_free_latent_channels", 16), ("data_free_num_frames", 1),
                              ("data_free_height", 32), ("data_free_width", 32)):
            value = int(getattr(self.args, name, default))
            if value < 1:
                raise ValueError(f"{name} must be positive")
            setattr(self, name, value)

    def _synthetic_latents(self, batch):
        supplied = batch.get("data_free_noise", batch.get("noise"))
        if isinstance(supplied, torch.Tensor):
            return supplied.to(self.device)
        batch_size = next((int(v.shape[0]) for v in batch.values() if isinstance(v, torch.Tensor) and v.ndim),
                          int(getattr(self.args, "batch_size", 1)))
        shape = self.data_free_latent_shape
        if shape is None:
            shape = (batch_size, self.data_free_latent_channels, self.data_free_num_frames,
                     self.data_free_height, self.data_free_width)
        else:
            shape = (batch_size, *shape[1:])
        return torch.randn(shape, device=self.device)

    def _forward_and_loss(self, batch):
        if "latents" in batch:
            return super()._forward_and_loss(batch)
        synthetic = dict(batch)
        synthetic["latents"] = self._synthetic_latents(batch)
        return super()._forward_and_loss(synthetic)

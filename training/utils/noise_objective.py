"""Noise and prediction contracts for rectified flow and VP diffusion.

VP coefficients are taken from the checkpoint scheduler, never approximated by
the flow interpolation. The v_prediction convention follows Diffusers DDIM:
v = alpha * epsilon - sigma * x0, with alpha**2 + sigma**2 = 1.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class NoiseObjective:
    process: str = "flow"
    num_train_timesteps: int = 1000
    prediction_type: str = "velocity"
    alphas_cumprod: torch.Tensor | None = None

    def __post_init__(self):
        if self.num_train_timesteps < 1:
            raise ValueError("num_train_timesteps must be positive")
        if self.process == "vp" and self.alphas_cumprod is None:
            raise ValueError("VP training requires checkpoint scheduler training_alphas_cumprod")
        allowed = {"flow": {"velocity", "epsilon", "x0"},
                   "vp": {"epsilon", "v_prediction", "sample", "x0"}}
        if self.process not in allowed or self.prediction_type not in allowed[self.process]:
            raise ValueError(f"Invalid {self.process!r} prediction type {self.prediction_type!r}")
        if self.process == "vp":
            if self.alphas_cumprod is None:
                raise ValueError("VP training requires checkpoint scheduler training_alphas_cumprod")
            alpha = torch.as_tensor(self.alphas_cumprod).detach().double().cpu().clone()
            if alpha.shape != (self.num_train_timesteps,) or not torch.isfinite(alpha).all():
                raise ValueError("VP alphas_cumprod must be a finite vector with num_train_timesteps entries")
            if not ((alpha >= 0) & (alpha <= 1)).all() or (alpha[1:] > alpha[:-1]).any():
                raise ValueError("VP alphas_cumprod must decrease monotonically in [0, 1]")
            object.__setattr__(self, "alphas_cumprod", alpha)

    def coefficients(self, timesteps: torch.Tensor, ndim: int):
        if not torch.isfinite(timesteps).all():
            raise ValueError("Noise timesteps must be finite")
        if self.process == "flow":
            if ((timesteps < 0) | (timesteps > self.num_train_timesteps)).any():
                raise ValueError("Flow timesteps must lie in [0, num_train_timesteps]")
            sigma = timesteps.float() / self.num_train_timesteps
            alpha = 1 - sigma
        else:
            if not torch.equal(timesteps, timesteps.round()):
                raise ValueError("VP scheduler timesteps must be integer indices")
            if ((timesteps < 0) | (timesteps >= self.num_train_timesteps)).any():
                raise ValueError("VP timesteps must lie in [0, num_train_timesteps - 1]")
            a = self.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)[timesteps.long()]
            alpha, sigma = a.sqrt(), (1 - a).sqrt()
        if timesteps.ndim != 1:
            raise ValueError("NoiseObjective expects one scalar timestep per sample")
        shape = (timesteps.shape[0],) + (1,) * (ndim - 1)
        return alpha.reshape(shape), sigma.reshape(shape)

    def add_noise(self, clean, noise, timesteps):
        if clean.shape != noise.shape:
            raise ValueError("Clean latents and noise must have the same shape")
        alpha, sigma = self.coefficients(timesteps, clean.ndim)
        return (alpha * clean.float() + sigma * noise.float()).to(clean.dtype)

    def to_x0_epsilon(self, prediction, noisy, timesteps, prediction_type=None):
        kind = prediction_type or self.prediction_type
        if prediction.shape != noisy.shape:
            raise ValueError("Denoiser prediction and noisy latents must have the same shape")
        alpha, sigma = self.coefficients(timesteps, noisy.ndim)
        x, p = noisy.float(), prediction.float()
        if kind == "velocity" and self.process == "flow":
            return x - sigma * p, x + alpha * p
        if kind == "v_prediction" and self.process == "vp":
            return alpha * x - sigma * p, sigma * x + alpha * p
        if kind == "epsilon":
            if (alpha == 0).any():
                raise ValueError("epsilon-to-x0 conversion is singular at alpha=0")
            return (x - sigma * p) / alpha, p
        if kind in {"sample", "x0"}:
            if (sigma == 0).any():
                raise ValueError("x0-to-epsilon conversion is singular at sigma=0")
            return p, (x - alpha * p) / sigma
        raise ValueError(f"Prediction type {kind!r} is not defined for {self.process}")

    def convert_prediction(self, prediction, noisy, timesteps, source, target):
        allowed = {"velocity", "epsilon", "x0"} if self.process == "flow" else {"epsilon", "v_prediction", "sample", "x0"}
        if source not in allowed or target not in allowed:
            raise ValueError(f"Prediction conversion {source!r} -> {target!r} is not defined for {self.process}")
        if source == target or {source, target} <= {"x0", "sample"}:
            return prediction
        x0, epsilon = self.to_x0_epsilon(prediction, noisy, timesteps, source)
        if target == "epsilon":
            return epsilon
        if target in {"sample", "x0"}:
            return x0
        if target == "velocity" and self.process == "flow":
            return epsilon - x0
        if target == "v_prediction" and self.process == "vp":
            alpha, sigma = self.coefficients(timesteps, noisy.ndim)
            return alpha * epsilon - sigma * x0
        raise ValueError(f"Prediction type {target!r} is not defined for {self.process}")

    def ddim_step(self, prediction, noisy, timesteps, next_timesteps):
        """Deterministic VP DDIM / flow Euler endpoint at a supplied next time."""
        x0, epsilon = self.to_x0_epsilon(prediction, noisy, timesteps)
        return self.add_noise(x0, epsilon, next_timesteps).to(noisy.dtype)


def resolve_noise_objectives(args, teacher, student):
    """Return compatible teacher/student objectives without guessing schedules."""
    requested = int(getattr(args, "num_train_timesteps", 0))
    models = (teacher, student)
    declared = [getattr(model, "num_train_timesteps", None) for model in models]
    steps = requested or next((int(value) for value in declared if value is not None), 1000)
    result = []
    for role, model, native_steps in zip(("teacher", "student"), models, declared):
        if native_steps is not None and int(native_steps) != steps:
            raise ValueError(f"{role} num_train_timesteps={native_steps} disagrees with schedule {steps}")
        process = getattr(model, "training_noise_process", "flow")
        kind = getattr(model, "training_prediction_type", getattr(args, "prediction_type", "velocity"))
        result.append(NoiseObjective(process, steps, kind, getattr(model, "training_alphas_cumprod", None)))
    left, right = result
    if left.process != right.process:
        raise ValueError("Teacher and student noise processes must match")
    if left.process == "vp" and not torch.equal(left.alphas_cumprod, right.alphas_cumprod):
        raise ValueError("Teacher and student VP alpha schedules must match")
    return tuple(result)

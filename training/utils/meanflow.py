"""Dual-time conditioning and the exact MeanFlow directional derivative.

Identity: u(z,t,r) = v(z,t) - (t-r) * [D_z u v + partial_t u].
The target is detached, so no second-order parameter derivatives are trained.
Reference: https://github.com/Gsunshine/meanflow/blob/main/meanflow.py
"""
from __future__ import annotations

from contextlib import contextmanager
import inspect
import torch
from torch import nn
from training.utils.model_output import extract_prediction_tensor


class MeanFlowTimeAdapter(nn.Module):
    """Add explicit endpoint conditioning to a pretrained single-time denoiser.

This adapter is not a native MeanFlow checkpoint conversion: its new two-time
MLP is trained with the denoiser. A zero-initialized output projection preserves
the original denoiser at initialization. Channel-wise scale/bias residuals act
on latent inputs; the base model still receives its own native timestep.
"""
    supports_meanflow = True

    def __init__(self, base_model, latent_channels=None, hidden_dim=128):
        super().__init__()
        self.base_model = base_model
        config = getattr(base_model, "config", {})
        get = config.get if isinstance(config, dict) else lambda key, default=None: getattr(config, key, default)
        channels = latent_channels or get("out_channels") or get("in_channels")
        if not channels or int(channels) < 1:
            raise ValueError("MeanFlowTimeAdapter requires latent_channels or model config out_channels")
        self.latent_channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.num_train_timesteps = int(getattr(base_model, "num_train_timesteps", 1000))
        self.training_noise_process = getattr(base_model, "training_noise_process", "flow")
        self.training_prediction_type = getattr(base_model, "training_prediction_type", "velocity")
        for name in ("requires_unused_parameter_detection", "requires_worlddistill_collective_leaf",
                     "supports_worlddistill_sequence_parallel"):
            if hasattr(base_model, name):
                setattr(self, name, getattr(base_model, name))
        if self.training_noise_process != "flow":
            raise ValueError("MeanFlowTimeAdapter requires a flow-matching denoiser")
        self.time_conditioner = nn.Sequential(nn.Linear(2, self.hidden_dim), nn.SiLU(),
                                             nn.Linear(self.hidden_dim, 2 * self.latent_channels))
        nn.init.zeros_(self.time_conditioner[-1].weight)
        nn.init.zeros_(self.time_conditioner[-1].bias)
        parameter = next(base_model.parameters(), None)
        if parameter is not None:
            self.time_conditioner.to(device=parameter.device, dtype=parameter.dtype)

    @property
    def config(self):
        return getattr(self.base_model, "config", {})

    def enable_gradient_checkpointing(self):
        return self.base_model.enable_gradient_checkpointing()

    def forward(self, hidden_states, timestep, timestep_r, **conditions):
        if hidden_states.shape[1] != self.latent_channels:
            raise ValueError("MeanFlow latent channel count differs from adapter configuration")
        if timestep.ndim != 1 or timestep_r.shape != timestep.shape:
            raise ValueError("MeanFlow expects one t and r scalar per sample")
        if ((timestep_r < 0) | (timestep_r > timestep) | (timestep > self.num_train_timesteps)).any():
            raise ValueError("MeanFlow requires 0 <= r <= t <= num_train_timesteps")
        conditioning = torch.stack((timestep, timestep - timestep_r), dim=-1) / self.num_train_timesteps
        dtype = next(self.time_conditioner.parameters()).dtype
        scale, bias = self.time_conditioner(conditioning.to(dtype)).to(hidden_states.dtype).chunk(2, -1)
        shape = (hidden_states.shape[0], self.latent_channels) + (1,) * (hidden_states.ndim - 2)
        x = hidden_states * (1 + scale.reshape(shape)) + bias.reshape(shape)
        return extract_prediction_tensor(self.base_model(hidden_states=x, timestep=timestep, **conditions), tag="MeanFlow")


def require_meanflow_model(model):
    candidate = model
    seen = set()
    while id(candidate) not in seen:
        seen.add(id(candidate))
        if getattr(candidate, "_orig_mod", None) is not None:
            candidate = candidate._orig_mod
        elif callable(getattr(candidate, "get_base_model", None)):
            nested = candidate.get_base_model()
            if nested is candidate:
                break
            candidate = nested
        else:
            break
    parameters = inspect.signature(candidate.forward).parameters
    if not getattr(candidate, "supports_meanflow", False) or "timestep_r" not in parameters:
        raise ValueError("MeanFlow needs an explicit dual-time student: use MeanFlowTimeAdapter or a "
                         "supports_meanflow model with forward(..., timestep_r, ...)")


def eager_meanflow_model(model):
    """The JVP uses eager modules; the normal parameter-gradient pass may compile."""
    while getattr(model, "_orig_mod", None) is not None:
        model = model._orig_mod
    return model


@contextmanager
def deterministic_meanflow_forward(model):
    """Use the same deterministic function for the JVP and trainable primal.

Eval mode disables dropout and buffer updates, not autograd. Checkpoint wrappers
are disabled for the derivative pass because reverse-over-reverse needs the
ordinary differentiable graph. Every module flag is restored on exit.
"""
    states = [(module, module.training, getattr(module, "gradient_checkpointing", None))
              for module in model.modules()]
    try:
        model.eval()
        for module, _, checkpointing in states:
            if isinstance(checkpointing, bool):
                module.gradient_checkpointing = False
        yield
    finally:
        for module, training, checkpointing in states:
            module.training = training
            if isinstance(checkpointing, bool):
                module.gradient_checkpointing = checkpointing


def meanflow_jvp(model, noisy, t, r, velocity, conditions, time_scale):
    """Exact input JVP, leaving optimizer parameter gradients intact.

Reverse-over-reverse supports models without a forward-AD rule. Math SDPA is
selected for its differentiable backward; no finite-difference substitute or
silent zero derivative is used when an operator lacks the necessary derivative.
"""
    def evaluate(x, end, start):
        kwargs = dict(conditions, hidden_states=x, timestep=end * time_scale, timestep_r=start * time_scale)
        # autograd.functional.jvp requests derivatives only for (x,end,start);
        # unlike backward(), it never accumulates gradients into model params.
        # Keep real Parameter objects to preserve sharding/offload metadata.
        result = model(**kwargs)
        return extract_prediction_tensor(result, tag="MeanFlow JVP")

    with torch.enable_grad(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        _, derivative = torch.autograd.functional.jvp(
            evaluate, (noisy.detach(), t.detach(), r.detach()),
            (velocity.detach().to(noisy), torch.ones_like(t), torch.zeros_like(r)),
            create_graph=False, strict=False,
        )
    return derivative.detach()


def meanflow_target(velocity, derivative, t, r):
    interval = (t - r).reshape((t.shape[0],) + (1,) * (velocity.ndim - 1))
    return (velocity.float() - interval * derivative.float()).detach()

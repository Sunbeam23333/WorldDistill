"""Explicit bridge to installed, differentiable world-model implementations.

A descriptor preserves the author's actual architecture. Importing its factory
is executable Python and requires opt-in; checkpoint weights use safetensors,
strict loading, and an in-directory path. This is a plugin interface, not an
implementation of any model merely listed in the model catalog.
"""
from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

import torch
from torch import nn

from training.utils.model_output import extract_prediction_tensor


class NativeTrainingAdapter(nn.Module):
    def __init__(self, model: nn.Module, descriptor: dict):
        super().__init__()
        self.model = model
        self.descriptor = dict(descriptor)
        self.input_map = dict(descriptor.get("input_map", {}))
        self.required_conditions = tuple(descriptor.get("required_conditions", ()))
        self.layout = descriptor.get("latent_layout", "BCTHW")
        if self.layout not in {"BCTHW", "BTCHW"}:
            raise ValueError("Native latent_layout must be BCTHW or BTCHW")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in self.input_map.items()):
            raise ValueError("Native input_map must map canonical input names to native names")
        if len(set(self.input_map.values())) != len(self.input_map):
            raise ValueError("Native input_map cannot alias two inputs to one argument")
        if not all(isinstance(k, str) for k in self.required_conditions):
            raise ValueError("Native required_conditions must contain input names")
        self._parameters_by_name = dict(inspect.signature(model.forward).parameters)
        self.num_train_timesteps = int(descriptor.get("num_train_timesteps", 1000))
        if self.num_train_timesteps <= 0:
            raise ValueError("Native num_train_timesteps must be positive")
        self.training_noise_process = descriptor.get("noise_process", "flow")
        if self.training_noise_process not in {"flow", "vp"}:
            raise ValueError("Native noise_process must be flow or vp")
        if self.training_noise_process == "flow":
            self.training_prediction_type = descriptor.get("prediction_type", "velocity")
            if self.training_prediction_type not in {"velocity", "epsilon", "x0"}:
                raise ValueError("Native flow prediction_type must be velocity, epsilon or x0")
        if self.training_noise_process == "vp":
            alpha = torch.as_tensor(descriptor.get("alphas_cumprod", []), dtype=torch.float32, device="cpu")
            if (alpha.ndim != 1 or len(alpha) != self.num_train_timesteps or not torch.isfinite(alpha).all()
                    or not ((alpha > 0) & (alpha <= 1)).all() or not (alpha[1:] <= alpha[:-1]).all()):
                raise ValueError("Native VP model requires its explicit valid cumulative alpha schedule")
            self.register_buffer("training_alphas_cumprod", alpha, persistent=False)
            self.training_prediction_type = descriptor.get("prediction_type")
            if self.training_prediction_type not in {"epsilon", "v_prediction", "sample"}:
                raise ValueError("Native VP model requires its explicit prediction_type")
        self.supports_meanflow = bool(descriptor.get("supports_meanflow", False))
        if self.supports_meanflow and self.input_map.get("timestep_r", "timestep_r") not in self._parameters_by_name:
            raise ValueError("Native MeanFlow model must explicitly implement a second timestep argument")
        if not any(True for _ in model.parameters()):
            raise ValueError("Native factory must return a parameterized torch.nn.Module")

    @property
    def config(self):
        return getattr(self.model, "config", self.descriptor.get("model_config", {}))

    def enable_gradient_checkpointing(self):
        enable = getattr(self.model, "enable_gradient_checkpointing", None)
        if not callable(enable):
            raise ValueError("Native model does not implement gradient checkpointing")
        enable()

    def forward(self, hidden_states, timestep, timestep_r=None, **conditions):
        if hidden_states.ndim != 5:
            raise ValueError("Native adapter requires B,C,T,H,W training latents")
        if timestep_r is not None and not self.supports_meanflow:
            raise ValueError("Native model does not declare dual-time MeanFlow support")
        inputs = {"hidden_states": hidden_states, "timestep": timestep,
                  **{k: v for k, v in conditions.items() if v is not None}}
        if timestep_r is not None:
            inputs["timestep_r"] = timestep_r
        missing = set(self.required_conditions) - inputs.keys()
        if missing:
            raise ValueError(f"Native model requires conditioning fields {sorted(missing)}")
        for name in ("actions", "camera_poses"):
            value = inputs.get(name)
            if value is not None and (not torch.is_tensor(value) or value.ndim < 2
                    or value.shape[:2] != (hidden_states.shape[0], hidden_states.shape[2])):
                raise ValueError(f"Native {name} must have layout B,T,... matching video frames")
        if self.layout == "BTCHW":
            inputs["hidden_states"] = hidden_states.permute(0, 2, 1, 3, 4).contiguous()
        kwargs = {}
        for name, value in inputs.items():
            native_name = self.input_map.get(name, name)
            if native_name in kwargs:
                raise ValueError(f"Native input aliases collide at {native_name}")
            # Requiring explicit parameters catches misspelled/ignored action
            # keys even when the underlying module happens to accept **kwargs.
            if native_name not in self._parameters_by_name:
                raise ValueError(f"Native model has no explicit forward parameter {native_name!r} for {name!r}")
            kwargs[native_name] = value
        required = {name for name, p in self._parameters_by_name.items()
                    if p.default is inspect.Parameter.empty and p.kind in
                    (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
        if required - kwargs.keys():
            raise ValueError(f"Native forward requires arguments {sorted(required - kwargs.keys())}")
        output = self.model(**kwargs)
        output_key = self.descriptor.get("prediction_key")
        if output_key is not None:
            output = output[output_key] if isinstance(output, dict) else getattr(output, output_key)
        output = extract_prediction_tensor(output, tag="native model")
        if self.layout == "BTCHW":
            output = output.permute(0, 2, 1, 3, 4).contiguous()
        if output.shape != hidden_states.shape:
            raise ValueError("Native prediction shape does not match the video latent shape")
        return output

    def save_pretrained(self, directory: str | Path) -> None:
        from safetensors.torch import save_file
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        config = {**self.descriptor, "weights": "model.safetensors"}
        state = {k: v.detach().cpu().contiguous().clone() for k, v in self.model.state_dict().items()}
        save_file(state, directory / "model.safetensors")
        (directory / "worlddistill_model.json").write_text(json.dumps(config, indent=2) + "\n")


def load_native_training_model(path, *, dtype=torch.float32, weights=True, allow_native_code=False):
    """Call an explicitly trusted installed factory, then load exact weights."""
    root = Path(path).resolve()
    config = json.loads((root / "worlddistill_model.json").read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Native model descriptor requires schema_version=1")
    if "architecture" in config:
        if "factory" in config:
            raise ValueError("Choose a built-in native architecture or an installed factory, not both")
        if config["architecture"] == "hy_worldplay":
            from training.native_adapters.worldplay import load_hy_worldplay
            return load_hy_worldplay(root, config, dtype=dtype, weights=weights, allow_native_code=allow_native_code)
        from training.native_adapters.world_models import load_builtin_world_model
        return load_builtin_world_model(root, config, dtype=dtype, weights=weights)
    if not allow_native_code:
        raise ValueError("Native model factory executes installed Python code; set allow_native_model_code explicitly")
    factory = config.get("factory", "")
    if not isinstance(factory, str) or factory.count(":") != 1:
        raise ValueError("Native factory must be an installed.module:factory name")
    module_name, symbol = factory.split(":")
    if not module_name or not symbol.isidentifier() or not all(p.isidentifier() for p in module_name.split(".")):
        raise ValueError("Native factory must be an installed.module:factory name")
    constructor = getattr(importlib.import_module(module_name), symbol)
    kwargs = config.get("constructor_kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("Native constructor_kwargs must be a JSON object")
    model = constructor(**kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError("Native factory did not return torch.nn.Module")
    if weights:
        from safetensors.torch import load_file
        relative = config.get("weights", "model.safetensors")
        weight_path = (root / relative).resolve()
        if not weight_path.is_relative_to(root) or weight_path.suffix != ".safetensors":
            raise ValueError("Native weights must be an in-directory safetensors file")
        model.load_state_dict(load_file(str(weight_path)), strict=True)
    return NativeTrainingAdapter(model.to(dtype=dtype), config)

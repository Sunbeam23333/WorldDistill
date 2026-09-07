"""Differentiable model adapters, separate from inference-only tensor runners.

The training contract is B,C,T,H,W flow latents plus explicit conditioning.
Architecture classes come from the checkpoint's Diffusers config; unsupported
conditioning is rejected instead of silently training an unconditional model.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from training.utils.model_output import extract_prediction_tensor


def config_value(config: Any, key: str, default: Any = None) -> Any:
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


class DiffusersTrainingAdapter(nn.Module):
    """Preserve a denoiser's real architecture and translate its input layout."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.family = type(model).__name__
        if "CogVideoX" in self.family:
            # CogVideoX DDIM/DPM use sqrt(alpha_bar) * x0 +
            # sqrt(1-alpha_bar) * noise, not the runtime's flow interpolation.
            # Layout translation alone is not a training scheduler adapter.
            self.training_noise_process = "vp"
        self.num_train_timesteps = 1000
        self._forward_parameters = dict(inspect.signature(model.forward).parameters)

    @property
    def config(self):
        return self.model.config

    def enable_gradient_checkpointing(self):
        fn = getattr(self.model, "enable_gradient_checkpointing", None)
        if fn is None:
            raise ValueError(f"{self.family} does not support gradient checkpointing")
        fn()

    def forward(self, hidden_states, timestep, image_cond=None, **conditions):
        original_shape = hidden_states.shape
        if hidden_states.ndim != 5:
            raise ValueError("Diffusers video training requires B,C,T,H,W flow latents")
        if timestep.ndim == 2 and timestep.shape != (hidden_states.shape[0], hidden_states.shape[2]):
            raise ValueError("Per-frame timestep must have shape B,T matching the flow latents")
        kwargs = {key: value for key, value in conditions.items() if value is not None}
        if "image_embeds" in kwargs and "encoder_hidden_states_image" in self._forward_parameters:
            kwargs["encoder_hidden_states_image"] = kwargs.pop("image_embeds")
        # Image conditioning here is already encoded, with any model-specific
        # mask channels. It must not be mistaken for CLIP image embeddings.
        if image_cond is not None:
            if image_cond.ndim != hidden_states.ndim or image_cond.shape[0] != hidden_states.shape[0] or image_cond.shape[2:] != hidden_states.shape[2:]:
                raise ValueError("image_cond must be a frame-aligned encoded latent/mask tensor")
            hidden_states = torch.cat((hidden_states, image_cond.to(hidden_states)), dim=1)
        in_channels = config_value(self.config, "in_channels")
        if hidden_states.ndim == 5 and in_channels is not None and "LTX" not in self.family:
            if hidden_states.shape[1] != in_channels:
                raise ValueError(
                    f"{self.family} expects {in_channels} input channels, got {hidden_states.shape[1]}; "
                    "provide the model-specific image_cond latent/mask (do not silently omit I2V conditioning)."
                )
        if "CogVideoX" in self.family:
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous()
        elif self.family == "LTXVideoTransformer3DModel":
            b, c, t, h, w = hidden_states.shape
            p = int(config_value(self.config, "patch_size", 1))
            pt = int(config_value(self.config, "patch_size_t", 1))
            if t % pt or h % p or w % p:
                raise ValueError("LTX latent dimensions must be divisible by patch sizes")
            hidden_states = hidden_states.reshape(b, c, t // pt, pt, h // p, p, w // p, p)
            hidden_states = hidden_states.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4).flatten(1, 3)
            kwargs.update(num_frames=t, height=h, width=w)
        if timestep.ndim == 2 and self.family == "WanTransformer3DModel":
            # Diffusers Wan2.2's per-token timestep contract; temporal patches
            # cannot represent different noise levels within the same patch.
            pt, ph, pw = config_value(self.config, "patch_size", (1, 2, 2))
            if pt != 1:
                raise ValueError("Per-frame Wan timesteps require temporal patch_size=1")
            timestep = timestep.repeat_interleave((original_shape[-2] // ph) * (original_shape[-1] // pw), dim=1)
        elif timestep.ndim == 2 and self.family not in {"LTXVideoTransformer3DModel"}:
            if not torch.all(timestep == timestep[:, :1]):
                raise ValueError(f"{self.family} has no per-frame timestep adapter; use scalar-noise training")
            timestep = timestep[:, 0]
        elif timestep.ndim == 2 and self.family == "LTXVideoTransformer3DModel":
            grouped = timestep.reshape(b, t // pt, pt)
            if pt > 1 and not torch.all(grouped == grouped[:, :, :1]):
                raise ValueError("LTX temporal patches cannot mix distinct per-frame noise levels")
            timestep = grouped[:, :, 0].repeat_interleave((h // p) * (w // p), dim=1)
        # Some Diffusers signatures mark inputs optional although the selected
        # checkpoint's conditioning modules unconditionally consume them.
        required_conditions = []
        if self.family == "HunyuanVideo15Transformer3DModel":
            required_conditions = ["encoder_hidden_states_2", "encoder_attention_mask_2", "image_embeds"]
        elif self.family == "HunyuanVideoTransformer3DModel" and config_value(self.config, "guidance_embeds", False):
            required_conditions = ["guidance"]
        absent = [name for name in required_conditions if name not in kwargs]
        if absent:
            raise ValueError(f"{self.family} requires explicit model-specific conditioning {absent}")
        kwargs.update(hidden_states=hidden_states, timestep=timestep, return_dict=False)
        has_variadic = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in self._forward_parameters.values())
        unknown = set(kwargs) - set(self._forward_parameters)
        if unknown and not has_variadic:
            raise ValueError(f"{self.family} does not implement conditioning fields {sorted(unknown)}; a native adapter is required")
        missing = [name for name, p in self._forward_parameters.items()
                   if p.default is inspect.Parameter.empty and p.kind not in
                   {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD} and name not in kwargs]
        if missing:
            raise ValueError(f"{self.family} requires conditioning fields {missing}")
        output = extract_prediction_tensor(self.model(**kwargs), tag=self.family)
        if "CogVideoX" in self.family:
            output = output.permute(0, 2, 1, 3, 4).contiguous()
        elif self.family == "LTXVideoTransformer3DModel":
            output = output.reshape(b, t // pt, h // p, w // p, -1, pt, p, p)
            output = output.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(b, -1, t, h, w)
        if output.shape != original_shape:
            raise ValueError(f"Denoiser output {tuple(output.shape)} does not match flow latent {tuple(original_shape)}")
        return output


class NoiseRoutedDenoiser(nn.Module):
    """Wan2.2 high/low-noise teachers AND students, with per-example routing."""

    requires_unused_parameter_detection = True
    # Every rank must gather the same expert group before per-example routing.
    # FSDP/ZeRO-3 must not place independent collective hooks inside this leaf.
    requires_worlddistill_collective_leaf = True

    def __init__(self, high: nn.Module, low: nn.Module, boundary_ratio: float, num_train_timesteps: int = 1000):
        super().__init__()
        if not 0 < boundary_ratio < 1:
            raise ValueError("Dual-expert boundary_ratio must be between zero and one")
        if num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive")
        self.high, self.low = high, low
        self.boundary_ratio = float(boundary_ratio)
        self.num_train_timesteps = int(num_train_timesteps)

    @property
    def config(self):
        return self.high.config

    def enable_gradient_checkpointing(self):
        for model in (self.high, self.low):
            model.enable_gradient_checkpointing()

    def forward(self, hidden_states, timestep, **kwargs):
        batch_size = hidden_states.shape[0]
        if timestep.ndim == 0:
            timestep = timestep.expand(batch_size)
        # Context frames can have timestep zero; route by the target/noisy frame.
        routing_t = timestep.reshape(batch_size, -1).amax(dim=1)
        high_mask = routing_t >= self.boundary_ratio * self.num_train_timesteps
        outputs, indices = [], []
        for selected, model in ((high_mask, self.high), (~high_mask, self.low)):
            idx = selected.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            def subset(value):
                if torch.is_tensor(value) and value.ndim and value.shape[0] == batch_size:
                    return value.index_select(0, idx)
                if isinstance(value, dict):
                    return {k: subset(v) for k, v in value.items()}
                return value
            result = model(hidden_states=hidden_states.index_select(0, idx),
                           timestep=timestep.index_select(0, idx), **{k: subset(v) for k, v in kwargs.items()})
            outputs.append(extract_prediction_tensor(result, tag="noise-routed expert"))
            indices.append(idx)
        order = torch.cat(indices).argsort()
        return torch.cat(outputs, dim=0).index_select(0, order)


def wrap_denoiser(model: nn.Module) -> nn.Module:
    from diffusers import ModelMixin
    return DiffusersTrainingAdapter(model) if isinstance(model, ModelMixin) else model


def load_diffusers_denoiser(path: str | Path, *, dtype=torch.float32, weights: bool = True) -> nn.Module:
    """Load exact configured classes without materializing VAE/text encoders."""
    import diffusers
    from diffusers import ModelMixin

    path = Path(path)
    with (path / "config.json").open() as f:
        config = json.load(f)
    class_name = config.get("_class_name", "")
    cls = getattr(diffusers, class_name, None)
    if not isinstance(cls, type) or not issubclass(cls, ModelMixin):
        raise ValueError(f"Unsupported Diffusers model class {class_name!r} in {path}")
    if weights:
        model = cls.from_pretrained(str(path), torch_dtype=dtype)
    else:
        model = cls.from_config(config).to(dtype=dtype)
    return wrap_denoiser(model)


def load_diffusers_training_model(path: str | Path, *, dtype=torch.float32, weights: bool = True) -> nn.Module:
    path = Path(path)
    if (path / "worlddistill_export.json").is_file():
        from training.student_export import read_student_manifest
        config = read_student_manifest(path)
        parts = {name: load_diffusers_denoiser(path / name, dtype=dtype, weights=weights)
                 for name in config["components"]}
        if "transformer_2" in parts:
            return NoiseRoutedDenoiser(parts["transformer"], parts["transformer_2"], config["boundary_ratio"],
                                      config.get("num_train_timesteps", 1000))
        model = next(iter(parts.values()))
        model.num_train_timesteps = int(config.get("num_train_timesteps", 1000))
        return model
    if (path / "model_index.json").is_file():
        with (path / "model_index.json").open() as f:
            config = json.load(f)
        component = "transformer" if config.get("transformer", [None])[0] else "unet"
        scheduler_config = path / "scheduler" / "scheduler_config.json"
        scheduler = json.loads(scheduler_config.read_text()) if scheduler_config.is_file() else {}
        native_steps = int(scheduler.get("num_train_timesteps", 1000))
        if native_steps <= 0:
            raise ValueError("Pipeline scheduler num_train_timesteps must be positive")
        high = load_diffusers_denoiser(path / component, dtype=dtype, weights=weights)
        high.num_train_timesteps = native_steps
        if config.get("transformer_2", [None])[0]:
            low = load_diffusers_denoiser(path / "transformer_2", dtype=dtype, weights=weights)
            low.num_train_timesteps = native_steps
            boundary = config.get("boundary_ratio")
            if boundary is None:
                raise ValueError("Dual-expert pipeline must declare boundary_ratio in model_index.json")
            return NoiseRoutedDenoiser(high, low, boundary, native_steps)
        return high
    return load_diffusers_denoiser(path, dtype=dtype, weights=weights)


__all__ = ["DiffusersTrainingAdapter", "NoiseRoutedDenoiser", "load_diffusers_training_model", "wrap_denoiser"]

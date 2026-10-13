"""Differentiable model adapters, separate from inference-only tensor runners.

The training contract is B,C,T,H,W flow latents plus explicit conditioning.
Architecture classes come from the checkpoint's Diffusers config; unsupported
conditioning is rejected instead of silently training an unconditional model.
"""
from __future__ import annotations

import inspect
import json
import math
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
        self.supports_meanflow = bool(config_value(self.config, "use_meanflow", False)
                                      and "timestep_r" in self._forward_parameters)
        self.supports_worlddistill_sequence_parallel = self.family == "WanTransformer3DModel"
        self.worlddistill_managed_context_parallel = self.supports_worlddistill_sequence_parallel
        self._context_parallel_group = None
        self._context_parallel_size = 1

    def enable_worlddistill_sequence_parallel(self, size, device):
        from training.utils.context_parallel import enable_wan_context_parallel
        self._context_parallel_group = enable_wan_context_parallel(self.model, size, device)
        self._context_parallel_size = size

    @property
    def config(self):
        return getattr(self.model, "config", {})

    def enable_gradient_checkpointing(self):
        fn = getattr(self.model, "enable_gradient_checkpointing", None)
        if fn is None:
            raise ValueError(f"{self.family} does not support gradient checkpointing")
        fn()

    def forward(self, hidden_states, timestep, image_cond=None, timestep_r=None, **conditions):
        if self._context_parallel_group is not None:
            from training.utils.context_parallel import replicated_input
            hidden_states = replicated_input(hidden_states, self._context_parallel_group)
            timestep = replicated_input(timestep, self._context_parallel_group)
            image_cond = replicated_input(image_cond, self._context_parallel_group)
            conditions = replicated_input(conditions, self._context_parallel_group)
            patch = config_value(self.config, "patch_size", (1, 2, 2))
            if any(dimension % factor for dimension, factor in zip(hidden_states.shape[2:], patch)):
                raise ValueError("Wan latent dimensions must be divisible by the 3D patch size")
            tokens = math.prod(dimension // factor for dimension, factor in zip(hidden_states.shape[2:], patch))
            if tokens % self._context_parallel_size:
                raise ValueError("Wan patch-token count must be divisible by sp_size")
        original_shape = hidden_states.shape
        if hidden_states.ndim != 5:
            raise ValueError("Diffusers video training requires B,C,T,H,W flow latents")
        if timestep.ndim == 2 and timestep.shape != (hidden_states.shape[0], hidden_states.shape[2]):
            raise ValueError("Per-frame timestep must have shape B,T matching the flow latents")
        kwargs = {key: value for key, value in conditions.items() if value is not None}
        if timestep_r is not None:
            kwargs["timestep_r"] = timestep_r
        if "timestep_r" in kwargs and not self.supports_meanflow:
            raise ValueError(f"{self.family} is not configured for dual-time MeanFlow conditioning")
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
        if not 0 < boundary_ratio <= 1:
            raise ValueError("Dual-expert boundary_ratio must be in (0,1]")
        if num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive")
        self.high, self.low = high, low
        self.supports_worlddistill_sequence_parallel = all(
            getattr(expert, "worlddistill_managed_context_parallel", False) for expert in (high, low))
        self.worlddistill_managed_context_parallel = self.supports_worlddistill_sequence_parallel
        self._context_parallel_group = None
        self.boundary_ratio = float(boundary_ratio)
        self.num_train_timesteps = int(num_train_timesteps)
        self.training_noise_process = getattr(high, "training_noise_process", "flow")
        self.training_prediction_type = getattr(high, "training_prediction_type", "velocity")
        if (getattr(low, "training_noise_process", "flow") != self.training_noise_process
                or getattr(low, "training_prediction_type", "velocity") != self.training_prediction_type):
            raise ValueError("Routed experts require the same noise process and prediction type")
        if self.training_noise_process == "vp":
            alphas = getattr(high, "training_alphas_cumprod", None)
            low_alphas = getattr(low, "training_alphas_cumprod", None)
            if (alphas is None or low_alphas is None or not torch.equal(alphas, low_alphas)
                    or len(alphas) != self.num_train_timesteps):
                raise ValueError("Routed VP experts require identical explicit cumulative alpha schedules")
            self.register_buffer("training_alphas_cumprod", alphas.detach().clone(), persistent=False)
            if hasattr(high, "training_scheduler_config"):
                self.training_scheduler_config = dict(high.training_scheduler_config)

    @property
    def config(self):
        return self.high.config

    def enable_gradient_checkpointing(self):
        for model in (self.high, self.low):
            model.enable_gradient_checkpointing()

    def enable_worlddistill_sequence_parallel(self, size, device):
        if not self.supports_worlddistill_sequence_parallel:
            raise ValueError("Both routed experts must implement the same managed context-parallel contract")
        for expert in (self.high, self.low):
            expert.enable_worlddistill_sequence_parallel(size, device)
        self._context_parallel_group = self.high._context_parallel_group

    def forward(self, hidden_states, timestep, **kwargs):
        batch_size = hidden_states.shape[0]
        if timestep.ndim == 0:
            timestep = timestep.expand(batch_size)
        # Context frames can have timestep zero; route by the target/noisy frame.
        routing_t = timestep.reshape(batch_size, -1).amax(dim=1)
        high_mask = routing_t >= self.boundary_ratio * self.num_train_timesteps
        if self._context_parallel_group is not None:
            import torch.distributed as dist
            minimum, maximum = high_mask.int(), high_mask.int()
            dist.all_reduce(minimum, op=dist.ReduceOp.MIN, group=self._context_parallel_group)
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=self._context_parallel_group)
            if not torch.equal(minimum, maximum):
                raise ValueError("Noise-routed experts require identical routing decisions within every CP group")
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


class LTX2TrainingAdapter(DiffusersTrainingAdapter):
    """LTX-2 audiovisual denoiser with both differentiable outputs preserved.

    Video uses B,C,T,H,W; audio uses the upstream normalized packed B,L,D
    layout. Audio and video conditioning are outputs of LTX text connectors,
    not raw Gemma hidden states. Both output modalities remain in the graph.
    """

    supports_joint_av = True

    def forward(self, hidden_states, timestep, audio_hidden_states=None,
                audio_timestep=None, **conditions):
        if hidden_states.ndim != 5 or not torch.is_tensor(audio_hidden_states) or audio_hidden_states.ndim != 3:
            raise ValueError("LTX2 requires video B,C,T,H,W and normalized packed audio B,L,D latents")
        b, c, t, h, w = hidden_states.shape
        if audio_hidden_states.shape[0] != b:
            raise ValueError("LTX2 audio and video batch sizes must agree")
        p, pt = int(config_value(self.config, "patch_size", 1)), int(config_value(self.config, "patch_size_t", 1))
        if min(p, pt) <= 0 or t % pt or h % p or w % p:
            raise ValueError("LTX2 video dimensions must be divisible by patch sizes")
        video = hidden_states.reshape(b, c, t // pt, pt, h // p, p, w // p, p)
        video = video.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4).flatten(1, 3)
        if video.shape[-1] != config_value(self.config, "in_channels"):
            raise ValueError("LTX2 packed video channels disagree with the transformer config")
        if audio_hidden_states.shape[-1] != config_value(self.config, "audio_in_channels"):
            raise ValueError("LTX2 packed audio channels disagree with the transformer config")
        if timestep.ndim != 1 or timestep.shape[0] != b:
            raise ValueError("Joint LTX2 training requires one video timestep per sample")
        audio_timestep = timestep if audio_timestep is None else audio_timestep
        if audio_timestep.shape != timestep.shape:
            raise ValueError("Joint LTX2 audio timestep must have shape B")
        kwargs = {k: v for k, v in conditions.items() if v is not None}
        required = {"encoder_hidden_states", "audio_encoder_hidden_states"}
        if required - kwargs.keys():
            raise ValueError("LTX2 requires separate video and audio connector text embeddings")
        kwargs.update(hidden_states=video, audio_hidden_states=audio_hidden_states, timestep=timestep,
                      audio_timestep=audio_timestep, num_frames=t, height=h, width=w,
                      audio_num_frames=audio_hidden_states.shape[1], return_dict=False)
        # LTX2.3 prompt modulation consumes sigma in the same upstream timestep
        # convention; older versions without these arguments retain their API.
        if "sigma" in self._forward_parameters:
            kwargs.setdefault("sigma", timestep)
        if "audio_sigma" in self._forward_parameters:
            kwargs.setdefault("audio_sigma", audio_timestep)
        unknown = kwargs.keys() - self._forward_parameters.keys()
        if unknown:
            raise ValueError(f"LTX2 does not implement conditioning fields {sorted(unknown)}")
        result = self.model(**kwargs)
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise TypeError("LTX2 must return both video and audio predictions")
        video_out, audio_out = result
        if video_out.shape != video.shape or audio_out.shape != audio_hidden_states.shape:
            raise ValueError("LTX2 output layout differs from its input modality layouts")
        video_out = video_out.reshape(b, t // pt, h // p, w // p, c, pt, p, p)
        video_out = video_out.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(b, c, t, h, w)
        return {"sample": video_out, "audio_sample": audio_out}


def wrap_denoiser(model: nn.Module) -> nn.Module:
    from diffusers import ModelMixin
    if type(model).__name__ == "LTX2VideoTransformer3DModel":
        return LTX2TrainingAdapter(model)
    if isinstance(model, ModelMixin):
        from training.native_adapters.diffusers_controls import CONTROL_ADAPTERS
        return CONTROL_ADAPTERS.get(type(model).__name__, DiffusersTrainingAdapter)(model)
    return model


def attach_training_scheduler(model: nn.Module, scheduler_config: dict) -> None:
    """Attach the checkpoint's actual VP alpha schedule, never infer from layout."""
    if getattr(model, "training_noise_process", "flow") != "vp":
        return
    import diffusers
    class_name = scheduler_config.get("_class_name")
    cls = getattr(diffusers, class_name or "", None)
    if cls is None or not class_name.startswith("CogVideoX"):
        raise ValueError("CogVideoX training requires its explicit CogVideoX scheduler configuration")
    # Export architecture reconstruction may run under torch.device('meta').
    # Schedules are small numerical metadata and must retain real values.
    with torch.device("cpu"):
        scheduler = cls.from_config(scheduler_config)
    alphas = scheduler.alphas_cumprod.detach().float()
    prediction = scheduler.config.prediction_type
    if prediction not in {"epsilon", "v_prediction", "sample"} or alphas.ndim != 1 or not torch.isfinite(alphas).all():
        raise ValueError("Invalid VP prediction type or cumulative alpha schedule")
    # Metadata belongs to the scheduler/config rather than the denoiser weight
    # inventory. Keep it nonpersistent so model-only safetensors stay portable.
    model.register_buffer("training_alphas_cumprod", alphas, persistent=False)
    model.training_prediction_type = prediction
    model.training_scheduler_config = dict(scheduler_config)
    model.num_train_timesteps = len(alphas)


def load_diffusers_denoiser(path: str | Path, *, dtype=torch.float32, weights: bool = True,
                           allow_native_code: bool = False) -> nn.Module:
    """Load exact configured classes without materializing VAE/text encoders."""
    path = Path(path)
    if (path / "worlddistill_model.json").is_file():
        from training.native_model_adapter import load_native_training_model
        return load_native_training_model(path, dtype=dtype, weights=weights, allow_native_code=allow_native_code)
    if (path / "worlddistill_adapter.json").is_file():
        from training.utils.meanflow import MeanFlowTimeAdapter
        config = json.loads((path / "worlddistill_adapter.json").read_text())
        if config.get("schema_version") != 1:
            raise ValueError("Unknown training adapter configuration")
        if config.get("type") == "noise_routed":
            parts = [load_diffusers_denoiser(path / branch, dtype=dtype, weights=weights,
                                             allow_native_code=allow_native_code) for branch in ("high", "low")]
            return NoiseRoutedDenoiser(*parts, config["boundary_ratio"], config["num_train_timesteps"])
        if config.get("type") != "meanflow_time":
            raise ValueError("Unknown training adapter configuration")
        base = load_diffusers_denoiser(path / "base_model", dtype=dtype, weights=weights,
                                      allow_native_code=allow_native_code)
        if "num_train_timesteps" in config:
            base.num_train_timesteps = int(config["num_train_timesteps"])
            if base.num_train_timesteps <= 0:
                raise ValueError("MeanFlow adapter num_train_timesteps must be positive")
        model = MeanFlowTimeAdapter(base, latent_channels=config["latent_channels"], hidden_dim=config["hidden_dim"])
        if weights:
            from safetensors.torch import load_file
            model.time_conditioner.load_state_dict(load_file(str(path / "meanflow.safetensors")), strict=True)
        return model.to(dtype=dtype)
    import diffusers
    from diffusers import ModelMixin

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
    adapter = wrap_denoiser(model)
    scheduler_path = path / "scheduler_config.json"
    if scheduler_path.is_file():
        attach_training_scheduler(adapter, json.loads(scheduler_path.read_text()))
    return adapter


def save_training_component(model: nn.Module, directory: str | Path) -> None:
    """Serialize architecture and trained parameters without flattening wrappers."""
    from training.native_model_adapter import NativeTrainingAdapter
    from training.utils.meanflow import MeanFlowTimeAdapter
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(model, NoiseRoutedDenoiser):
        save_training_component(model.high, directory / "high")
        save_training_component(model.low, directory / "low")
        (directory / "worlddistill_adapter.json").write_text(json.dumps({
            "schema_version": 1, "type": "noise_routed", "boundary_ratio": model.boundary_ratio,
            "num_train_timesteps": model.num_train_timesteps}, indent=2) + "\n")
    elif isinstance(model, MeanFlowTimeAdapter):
        from safetensors.torch import save_file
        save_training_component(model.base_model, directory / "base_model")
        (directory / "worlddistill_adapter.json").write_text(json.dumps({
            "schema_version": 1, "type": "meanflow_time", "latent_channels": model.latent_channels,
            "hidden_dim": model.hidden_dim, "num_train_timesteps": model.num_train_timesteps}, indent=2) + "\n")
        save_file({k: v.detach().cpu().contiguous().clone() for k, v in model.time_conditioner.state_dict().items()},
                  str(directory / "meanflow.safetensors"))
    elif isinstance(model, NativeTrainingAdapter):
        model.save_pretrained(directory)
    elif isinstance(model, DiffusersTrainingAdapter):
        model.model.save_pretrained(directory, safe_serialization=True)
        scheduler = getattr(model, "training_scheduler_config", None)
        if scheduler:
            (directory / "scheduler_config.json").write_text(json.dumps(scheduler, indent=2) + "\n")
    else:
        raise TypeError("Export requires a Diffusers, declared native, or MeanFlow training adapter")


def load_diffusers_training_model(path: str | Path, *, dtype=torch.float32, weights: bool = True,
                                 allow_native_code: bool = False) -> nn.Module:
    path = Path(path)
    if (path / "worlddistill_model.json").is_file():
        from training.native_model_adapter import load_native_training_model
        return load_native_training_model(path, dtype=dtype, weights=weights, allow_native_code=allow_native_code)
    if (path / "worlddistill_export.json").is_file():
        from training.student_export import read_student_manifest
        config = read_student_manifest(path)
        parts = {name: load_diffusers_denoiser(path / name, dtype=dtype, weights=weights, allow_native_code=allow_native_code)
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
        high = load_diffusers_denoiser(path / component, dtype=dtype, weights=weights,
                                      allow_native_code=allow_native_code)
        high.num_train_timesteps = native_steps
        attach_training_scheduler(high, scheduler)
        if config.get("transformer_2", [None])[0]:
            low = load_diffusers_denoiser(path / "transformer_2", dtype=dtype, weights=weights,
                                         allow_native_code=allow_native_code)
            low.num_train_timesteps = native_steps
            attach_training_scheduler(low, scheduler)
            boundary = config.get("boundary_ratio")
            if boundary is None:
                raise ValueError("Dual-expert pipeline must declare boundary_ratio in model_index.json")
            return NoiseRoutedDenoiser(high, low, boundary, native_steps)
        return high
    model = load_diffusers_denoiser(path, dtype=dtype, weights=weights, allow_native_code=allow_native_code)
    for scheduler_path in (path / "scheduler_config.json", path.parent / "scheduler" / "scheduler_config.json"):
        if scheduler_path.is_file():
            attach_training_scheduler(model, json.loads(scheduler_path.read_text()))
            break
    return model


__all__ = ["DiffusersTrainingAdapter", "NoiseRoutedDenoiser", "load_diffusers_training_model", "wrap_denoiser"]

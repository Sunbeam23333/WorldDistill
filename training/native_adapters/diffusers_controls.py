"""Model-specific differentiable control contracts for public Diffusers models.

These adapters consume encoded conditioning, not a guessed raw-video pose/face
extractor. Parameters and saved config remain the original Diffusers model.
"""
from __future__ import annotations

import torch

from training.model_adapter import DiffusersTrainingAdapter, config_value
from training.utils.model_output import extract_prediction_tensor


def _tensor(value, shape, name):
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(shape) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape {tuple(shape)}")
    return value


class WanVACETrainingAdapter(DiffusersTrainingAdapter):
    """VACE context contains the checkpoint's encoded source/mask/reference channels."""
    def forward(self, hidden_states, timestep, vace_context=None, vace_scale=None, **conditions):
        if hidden_states.ndim != 5:
            raise ValueError("VACE requires B,C,T,H,W latents")
        b, _, t, h, w = hidden_states.shape
        channels = config_value(self.config, "vace_in_channels")
        _tensor(vace_context, (b, channels, t, h, w), "vace_context")
        layers = len(config_value(self.config, "vace_layers"))
        if vace_scale is None:
            scale = hidden_states.new_ones(layers)
        elif vace_scale.ndim == 2:
            # Dataset stores one vector per sample. The native module unbinds
            # the first axis and multiplies each [B,sequence,dim] control hint.
            scale = _tensor(vace_scale, (b, layers), "vace_scale").T[:, :, None, None]
        else:
            scale = _tensor(vace_scale, (layers,), "vace_scale")
        if "control_hidden_states" in conditions or "control_hidden_states_scale" in conditions:
            raise ValueError("Use canonical vace_context/vace_scale without duplicate native fields")
        return super().forward(hidden_states, timestep,
            control_hidden_states=vace_context.to(hidden_states),
            control_hidden_states_scale=scale.to(hidden_states), **conditions)


class WanAnimateTrainingAdapter(DiffusersTrainingAdapter):
    """Animate's reference-prefixed latent, pose-latent and face-pixel protocol."""
    def forward(self, hidden_states, timestep, pose_latents=None, face_pixels=None, **conditions):
        if hidden_states.ndim != 5 or hidden_states.shape[2] < 2:
            raise ValueError("Animate clean/noisy latents must include a reference prefix plus video frames")
        b, c, t, h, w = hidden_states.shape
        if config_value(self.config, "patch_size")[0] != 1:
            raise ValueError("Animate reference-prefix conditioning requires temporal patch_size=1")
        channels = config_value(self.config, "latent_channels", c)
        if channels != c:
            raise ValueError("Animate noisy latent channels do not match latent_channels")
        _tensor(pose_latents, (b, c, t - 1, h, w), "pose_latents")
        size = config_value(self.config, "motion_encoder_size", 512)
        # The face encoder has two causal stride-2 layers: ceil(pixel_T/4).
        # Follow the official VAE protocol (pixel_T = 4*(latent_T-1)+1).
        _tensor(face_pixels, (b, 3, (t - 2) * 4 + 1, size, size), "face_pixels")
        if face_pixels.min() < -1 or face_pixels.max() > 1:
            raise ValueError("Animate face_pixels must be normalized to [-1,1]")
        if "pose_hidden_states" in conditions or "face_pixel_values" in conditions:
            raise ValueError("Use canonical pose_latents/face_pixels without duplicate native fields")
        return super().forward(hidden_states, timestep, pose_hidden_states=pose_latents.to(hidden_states),
            face_pixel_values=face_pixels.to(hidden_states), **conditions)


class SkyReelsV2TrainingAdapter(DiffusersTrainingAdapter):
    """Native per-frame diffusion-forcing time embedding and FPS category mapping."""
    def forward(self, hidden_states, timestep, image_cond=None, encoder_hidden_states=None,
                image_embeds=None, fps_condition=None, **conditions):
        unknown = [key for key, value in conditions.items() if value is not None]
        if unknown:
            raise ValueError(f"SkyReelsV2 does not accept conditioning {sorted(unknown)}")
        if hidden_states.ndim != 5:
            raise ValueError("SkyReelsV2 requires B,C,T,H,W latents")
        b, c, t, h, w = hidden_states.shape
        patch = config_value(self.config, "patch_size")
        if any(value % factor for value, factor in zip((t,h,w),patch)):
            raise ValueError("SkyReelsV2 latent dimensions must be divisible by patch_size")
        if timestep.shape not in {(b,), (b,t)}:
            raise ValueError("SkyReelsV2 timestep must be B or B,T")
        per_frame = timestep.ndim == 2
        if per_frame and patch[0] != 1:
            raise ValueError("SkyReelsV2 per-frame diffusion forcing requires temporal patch_size=1")
        block = int(config_value(self.config, "num_frame_per_block", 1))
        if block > 1 and (t // patch[0]) % block:
            raise ValueError("SkyReelsV2 causal temporal blocks must divide latent frames")
        if image_cond is not None:
            _tensor(image_cond, (b, config_value(self.config,"in_channels")-c,t,h,w), "image_cond")
            model_input = torch.cat((hidden_states,image_cond.to(hidden_states)), 1)
        else:
            model_input = hidden_states
        if model_input.shape[1] != config_value(self.config, "in_channels"):
            raise ValueError("SkyReelsV2 requires checkpoint-specific encoded image_cond channels")
        if not torch.is_tensor(encoder_hidden_states) or encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != b:
            raise ValueError("SkyReelsV2 requires B,L,D encoder_hidden_states")
        inject_fps = bool(config_value(self.config, "inject_sample_info", False))
        if inject_fps:
            _tensor(fps_condition, (b,), "fps_condition")
            if (fps_condition <= 0).any():
                raise ValueError("fps_condition contains physical positive FPS, not embedding indices")
            fps_category = torch.where(fps_condition == 16, 0, 1)
        elif fps_condition is not None:
            raise ValueError("This SkyReelsV2 checkpoint has no FPS embedding")
        else:
            fps_category = None
        # Diffusers' FPS + per-frame branch flattens FPS differently from time
        # for B>1. Run each sample independently, preserving autograd and exact
        # author time/FPS semantics instead of broadcasting across examples.
        pieces = []
        for i in range(b):
            result = self.model(hidden_states=model_input[i:i+1], timestep=timestep[i:i+1],
                encoder_hidden_states=encoder_hidden_states[i:i+1],
                encoder_hidden_states_image=None if image_embeds is None else image_embeds[i:i+1],
                enable_diffusion_forcing=per_frame,
                fps=None if fps_category is None else fps_category[i:i+1], return_dict=False)
            pieces.append(extract_prediction_tensor(result, tag=self.family))
        result = torch.cat(pieces)
        if result.shape != hidden_states.shape:
            raise ValueError("SkyReelsV2 prediction shape does not match video latents")
        return result


CONTROL_ADAPTERS = {
    "WanVACETransformer3DModel": WanVACETrainingAdapter,
    "WanAnimateTransformer3DModel": WanAnimateTrainingAdapter,
    "SkyReelsV2Transformer3DModel": SkyReelsV2TrainingAdapter,
}

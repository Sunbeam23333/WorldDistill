"""Concrete Matrix-Game 2 and LingBot-CAM training/loading contracts.

The modules are the authors' actual parameterized architectures, not LightX2V
inference runners or arbitrary user factories. Descriptors preserve the exact
architecture configuration and source revision alongside trained weights.
"""
from __future__ import annotations

import inspect
import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

from training.native_model_adapter import NativeTrainingAdapter


SOURCE_REVISIONS = {
    "matrix_game2": "756776d516233027631008761d4d40ddaea3e23a",
    "lingbot_cam": "a43bec7f8091c83e9b30b16b912f6fc906236fa6",
}


def _video(x, patch):
    if not torch.is_tensor(x) or x.ndim != 5 or min(x.shape) <= 0:
        raise ValueError("Native world model requires nonempty B,C,T,H,W latents")
    if any(d % p for d, p in zip(x.shape[2:], patch)):
        raise ValueError("Video dimensions must be divisible by the native patch_size")


def _times(timestep, x, *, per_frame):
    if timestep.shape == (x.shape[0],):
        return timestep[:, None].expand(-1, x.shape[2]) if per_frame else timestep
    if timestep.shape != (x.shape[0], x.shape[2]):
        raise ValueError("Timesteps must be B or B,T matching latent video frames")
    if per_frame:
        return timestep
    if not torch.equal(timestep, timestep[:, :1].expand_as(timestep)):
        raise ValueError("This native checkpoint requires one timestep per sample")
    return timestep[:, 0]


def _condition(value, shape, name):
    if not torch.is_tensor(value) or value.shape != shape or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape {tuple(shape)}")
    return value


def _autocast_native(x):
    return torch.autocast(device_type=x.device.type,dtype=x.dtype) if x.dtype in {torch.float16,torch.bfloat16} else nullcontext()


class _ConcreteWorldAdapter(NativeTrainingAdapter):
    def __init__(self, model, descriptor):
        nn.Module.__init__(self)
        self.model = model
        self.descriptor = dict(descriptor)
        self.num_train_timesteps = int(descriptor.get("num_train_timesteps", 1000))
        if self.num_train_timesteps < 1 or descriptor.get("noise_process", "flow") != "flow":
            raise ValueError("Native world checkpoints require a positive time scale and flow noise process")
        if descriptor.get("prediction_type", "velocity") != "velocity":
            raise ValueError("These author architectures predict flow velocity")
        self.training_noise_process = "flow"
        self.training_prediction_type = "velocity"
        self.supports_meanflow = False
        self.supports_worlddistill_sequence_parallel = False

    @property
    def config(self):
        return {**dict(self.model.config), "in_channels": self.model.out_dim, "out_channels": self.model.out_dim}

    def enable_gradient_checkpointing(self):
        self.model.gradient_checkpointing = True

    def save_pretrained(self, directory):
        from safetensors.torch import save_file
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        config = {**self.descriptor, "weights": "model.safetensors", "weights_prefix": ""}
        config.pop("weights_index", None)
        state = {key: value.detach().cpu().contiguous().clone() for key, value in self.model.state_dict().items()}
        save_file(state, directory / "model.safetensors")
        (directory / "worlddistill_model.json").write_text(json.dumps(config, indent=2) + "\n")


class MatrixGame2TrainingAdapter(_ConcreteWorldAdapter):
    """Native causal/foundation Matrix2 with pixel-rate keyboard/mouse actions."""
    def __init__(self, model, descriptor):
        super().__init__(model, descriptor)
        self.is_causal = descriptor.get("variant", "causal") == "causal"
        self._mask_signature = None
        if self.model.patch_size[0] != 1:
            raise ValueError("Matrix2 pixel-rate action windows require temporal patch_size=1")
        if self.is_causal:
            self.model.num_frame_per_block = int(descriptor.get("num_frame_per_block", 1))
            if self.model.num_frame_per_block < 1:
                raise ValueError("Matrix num_frame_per_block must be positive")

    def forward(self, hidden_states, timestep, image_cond=None, image_embeds=None,
                keyboard_actions=None, mouse_actions=None, **conditions):
        if any(value is not None for value in conditions.values()):
            raise ValueError(f"Matrix2 does not accept conditions {sorted(conditions)}")
        _video(hidden_states, self.model.patch_size)
        b, c, t, h, w = hidden_states.shape
        image_cond = _condition(image_cond, (b, self.model.in_dim - c, t, h, w), "image_cond")
        if (not torch.is_tensor(image_embeds) or image_embeds.ndim != 3 or image_embeds.shape[0] != b
                or image_embeds.shape[-1] != 1280 or not torch.isfinite(image_embeds).all()):
            raise ValueError("Matrix2 requires B,N,1280 CLIP image_embeds, not text embeddings")
        action_config = self.descriptor["model_config"]["action_config"]
        ratio = int(action_config.get("vae_time_compression_ratio", 4))
        frames = (t - 1) * ratio + 1
        keyboard_actions = _condition(keyboard_actions,
            (b, frames, int(action_config.get("keyboard_dim_in", 6))), "keyboard_actions")
        mouse_actions = _condition(mouse_actions,
            (b, frames, int(action_config.get("mouse_dim_in", 2))), "mouse_actions")
        time = _times(timestep, hidden_states, per_frame=self.is_causal)
        if self.is_causal:
            signature = (t, h, w, hidden_states.device)
            if self._mask_signature != signature:
                # The author implementation caches shape-dependent causal masks.
                # Different data buckets must rebuild them instead of reusing
                # the first sample's temporal/spatial mask.
                self.model.block_mask = self.model.block_mask_mouse = self.model.block_mask_keyboard = None
                self._mask_signature = signature
        with _autocast_native(hidden_states):
            output = self.model(x=hidden_states, t=time, visual_context=image_embeds.to(hidden_states),
                cond_concat=image_cond.to(hidden_states),mouse_cond=mouse_actions.to(hidden_states),
                keyboard_cond=keyboard_actions.to(hidden_states))
        if output.shape != hidden_states.shape:
            raise ValueError("Matrix2 prediction does not match video latent shape")
        return output.to(hidden_states.dtype)


def lingbot_camera_features(camera_poses, camera_intrinsics, latent_height, latent_width):
    """Author camera protocol: relative framewise C2W, normalized translation,
    pixel-centered rays, origin+direction (NOT a cross-product Pluecker moment),
    and 8x8 pixel-unshuffle into the native 384-channel conditioning tensor.
    Poses must already be aligned to latent frames; no temporal interpolation is
    guessed. Intrinsics are in pixels at the VAE-decoded resolution.
    """
    if not torch.is_tensor(camera_poses) or camera_poses.ndim != 4 or camera_poses.shape[-2:] != (4, 4) or min(camera_poses.shape[:2]) < 1:
        raise ValueError("LingBot camera_poses must be B,T,4,4 camera-to-world matrices")
    if not torch.is_tensor(camera_intrinsics) or camera_intrinsics.device != camera_poses.device:
        raise ValueError("Camera intrinsics must be tensors on the same device as poses")
    b, t = camera_poses.shape[:2]
    pose = camera_poses.float()
    if not torch.isfinite(pose).all():
        raise ValueError("Camera poses must be finite")
    if camera_intrinsics.shape == (b, t, 3, 3):
        intrinsics = torch.stack((camera_intrinsics[..., 0, 0], camera_intrinsics[..., 1, 1],
                                 camera_intrinsics[..., 0, 2], camera_intrinsics[..., 1, 2]), -1).float()
    else:
        intrinsics = _condition(camera_intrinsics, (b, t, 4), "camera_intrinsics").float()
    if not torch.isfinite(intrinsics).all() or (intrinsics[..., :2] <= 0).any():
        raise ValueError("Camera focal lengths must be finite and positive")
    rotation, translation = pose[..., :3, :3], pose[..., :3, 3]
    relative_r = rotation[:, :-1].transpose(-1, -2) @ rotation[:, 1:]
    relative_t = (rotation[:, :-1].transpose(-1, -2) @ (translation[:, 1:] - translation[:, :-1])[..., None])[..., 0]
    relative_r = torch.cat((torch.eye(3, device=pose.device).expand(b, 1, 3, 3), relative_r), 1)
    relative_t = torch.cat((torch.zeros(b, 1, 3, device=pose.device), relative_t), 1)
    scale = relative_t.norm(dim=-1).amax(dim=1, keepdim=True).clamp_min(1e-12)
    relative_t = relative_t / scale[..., None]
    h, w = latent_height * 8, latent_width * 8
    yy, xx = torch.meshgrid(torch.arange(h, device=pose.device) + .5,
                           torch.arange(w, device=pose.device) + .5, indexing="ij")
    fx, fy, cx, cy = (intrinsics[..., index, None, None] for index in range(4))
    dx, dy = (xx - cx) / fx, (yy - cy) / fy
    directions = torch.stack((dx, dy, torch.ones_like(dx)), -1)
    directions = torch.nn.functional.normalize(directions, dim=-1)
    directions = torch.einsum("bthwc,btdc->bthwd", directions, relative_r)
    rays = torch.cat((relative_t[:, :, None, None].expand(-1, -1, h, w, -1), directions), -1)
    return rays.reshape(b, t, latent_height, 8, latent_width, 8, 6).permute(
        0, 6, 3, 5, 1, 2, 4).reshape(b, 384, t, latent_height, latent_width)


class LingBotCameraTrainingAdapter(_ConcreteWorldAdapter):
    """Native camera-conditioned LingBot denoiser, including every camera block."""
    def forward(self, hidden_states, timestep, encoder_hidden_states=None, image_cond=None,
                camera_poses=None, camera_intrinsics=None, camera_features=None, **conditions):
        if any(value is not None for value in conditions.values()):
            raise ValueError(f"LingBot-CAM does not accept conditions {sorted(conditions)}")
        _video(hidden_states, self.model.patch_size)
        if self.model.patch_size[0] != 1:
            raise ValueError("LingBot camera frames require temporal patch_size=1")
        b, c, t, h, w = hidden_states.shape
        if not torch.is_tensor(encoder_hidden_states) or encoder_hidden_states.ndim != 3:
            raise ValueError("LingBot requires B,L,D T5 encoder_hidden_states")
        if (encoder_hidden_states.shape[0] != b or encoder_hidden_states.shape[1] > self.model.text_len
                or encoder_hidden_states.shape[2] != self.model.text_dim):
            raise ValueError("LingBot text features do not match the native checkpoint")
        if self.model.in_dim != c:
            image_cond = _condition(image_cond, (b, self.model.in_dim - c, t, h, w), "image_cond")
        elif image_cond is not None:
            raise ValueError("This LingBot checkpoint does not accept extra image channels")
        if camera_features is not None:
            if camera_poses is not None or camera_intrinsics is not None:
                raise ValueError("Supply either camera_features or camera_poses + camera_intrinsics")
        else:
            if camera_poses is None or camera_intrinsics is None:
                raise ValueError("LingBot requires camera_features or C2W poses and pixel intrinsics")
            _condition(camera_poses, (b, t, 4, 4), "camera_poses")
            camera_features = lingbot_camera_features(camera_poses, camera_intrinsics, h, w)
        camera_features = _condition(camera_features, (b, 384, t, h, w), "camera_features")
        time = _times(timestep, hidden_states, per_frame=True)
        spatial = (h // self.model.patch_size[1]) * (w // self.model.patch_size[2])
        time = time.repeat_interleave(spatial, dim=1)
        # The author's camera branch concatenates samples along its sequence
        # axis. Execute each member independently to preserve batch semantics
        # (never let sample A's camera control sample B), retaining autograd.
        outputs = []
        for index in range(b):
            with _autocast_native(hidden_states):
                output = self.model(x=[hidden_states[index]], t=time[index:index + 1],
                    context=[encoder_hidden_states[index].to(hidden_states)], seq_len=t * spatial,
                    y=None if image_cond is None else [image_cond[index].to(hidden_states)],
                    dit_cond_dict={"c2ws_plucker_emb": (camera_features[index:index + 1].to(hidden_states),)})
            outputs.append(output[0])
        result = torch.stack(outputs)
        if result.shape != hidden_states.shape:
            raise ValueError("LingBot prediction does not match video latent shape")
        return result.to(hidden_states.dtype)


def _read_native_weights(root, descriptor):
    from safetensors.torch import load_file
    def resolve(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Native weight paths must stay inside the model directory")
        return path
    relative = descriptor.get("weights", "model.safetensors")
    path = resolve(relative)
    if path.name.endswith(".safetensors.index.json"):
        index = json.loads(path.read_text())
        mapping = index.get("weight_map", {})
        if not mapping:
            raise ValueError("Native safetensors index has no weight map")
        state = {}
        for shard in sorted(set(mapping.values())):
            shard_path = resolve(shard)
            if shard_path.suffix != ".safetensors":
                raise ValueError("Native shards must use safetensors")
            current = load_file(str(shard_path))
            if state.keys() & current.keys():
                raise ValueError("Native weight shards contain duplicate keys")
            if any(mapping.get(key) != shard for key in current):
                raise ValueError("Native weight shard disagrees with its index")
            state.update(current)
        if state.keys() != mapping.keys():
            raise ValueError("Native weight inventory disagrees with its index")
    elif path.suffix == ".safetensors":
        state = load_file(str(path))
    else:
        raise ValueError("Native pretrained weights must be safetensors or a safetensors index")
    prefix = descriptor.get("weights_prefix", "")
    if prefix:
        if prefix != "model." or not state or not all(key.startswith(prefix) for key in state):
            raise ValueError("Only the explicit Matrix generator 'model.' prefix is supported")
        state = {key[len(prefix):]: value for key, value in state.items()}
    return state


def load_builtin_world_model(path, descriptor, *, dtype=torch.float32, weights=True):
    root = Path(path).resolve()
    if descriptor.get("noise_process", "flow") != "flow" or descriptor.get("prediction_type", "velocity") != "velocity":
        raise ValueError("Built-in native architectures require flow velocity metadata")
    architecture = descriptor["architecture"]
    if architecture not in SOURCE_REVISIONS:
        raise ValueError(f"Unknown built-in native architecture {architecture!r}")
    if descriptor.get("source_revision", SOURCE_REVISIONS[architecture]) != SOURCE_REVISIONS[architecture]:
        raise ValueError("Native source revision does not match the bundled architecture")
    config = dict(descriptor.get("model_config", {}))
    if architecture == "matrix_game2":
        from .matrix_game2.model import WanModel
        from .matrix_game2.causal_model import CausalWanModel
        variant = descriptor.get("variant", "causal")
        if variant not in {"causal", "foundation"}:
            raise ValueError("Matrix2 variant must be causal or foundation")
        cls = CausalWanModel if variant == "causal" else WanModel
        adapter = MatrixGame2TrainingAdapter
        if not config.get("action_config"):
            raise ValueError("Matrix2 requires its actual action_config; plain Wan is not Matrix2")
    else:
        from .lingbot.model import WanModel
        cls, adapter = WanModel, LingBotCameraTrainingAdapter
        if config.get("control_type", "cam") != "cam":
            raise ValueError("lingbot_cam requires the camera checkpoint, not the action variant")
        config["control_type"] = "cam"
    arguments = inspect.signature(cls.__init__).parameters
    unknown = set(config) - set(arguments)
    if unknown:
        raise ValueError(f"Unknown native architecture configuration fields: {sorted(unknown)}")
    model = cls(**config)
    if weights:
        model.load_state_dict(_read_native_weights(root, descriptor), strict=True)
    metadata = {**descriptor, "source_revision": SOURCE_REVISIONS[architecture], "model_config": config,
                "noise_process": "flow", "prediction_type": "velocity"}
    return adapter(model.to(dtype=dtype), metadata)

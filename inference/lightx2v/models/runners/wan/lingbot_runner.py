"""
LingBot Camera Control - Runner.

Registers as 'lingbot_cam_moe' model_cls. Handles:
  - MoE dual model (high_noise + low_noise) like Wan2.2 MoE
  - Camera control data loading (Plücker coordinates / c2ws)
  - Uses Wan2.1 VAE (stride=[4,8,8])
  - No CLIP encoder (use_image_encoder=False)
"""

import gc
import json
import math
import os

import numpy as np
import torch
from loguru import logger

from lightx2v.models.networks.wan.lingbot_model import LingBotModel
from lightx2v.models.runners.wan.wan_runner import WanRunner, Wan22MoeRunner, MultiModelStruct, build_wan_model_with_lora
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import *
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


def generate_plucker_embedding(c2ws, image_h, image_w, num_frames, fx=None, fy=None):
    """
    Generate Plücker ray embedding from camera-to-world matrices.

    Args:
        c2ws: (N, 4, 4) camera-to-world matrices
        image_h: image height
        image_w: image width
        num_frames: number of video frames
        fx, fy: focal lengths (default: image_w/2)

    Returns:
        plucker: (6, num_frames, image_h, image_w) Plücker embedding
    """
    if fx is None:
        fx = image_w / 2.0
    if fy is None:
        fy = image_h / 2.0

    cx = image_w / 2.0
    cy = image_h / 2.0

    # Generate pixel grid
    u = torch.arange(image_w, dtype=torch.float32)
    v = torch.arange(image_h, dtype=torch.float32)
    u, v = torch.meshgrid(u, v, indexing="xy")  # (H, W)

    # Pixel to camera coordinates
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = torch.ones_like(x)
    dirs_cam = torch.stack([x, y, z], dim=-1)  # (H, W, 3)
    dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True)

    plucker_list = []
    for i in range(num_frames):
        c2w = c2ws[i]  # (4, 4)
        R = c2w[:3, :3]  # (3, 3)
        T = c2w[:3, 3]   # (3,)

        # Transform directions to world space
        dirs_world = torch.einsum("ij,hwj->hwi", R, dirs_cam)  # (H, W, 3)
        dirs_world = dirs_world / dirs_world.norm(dim=-1, keepdim=True)

        # Camera origin in world space
        origin = T.view(1, 1, 3).expand(image_h, image_w, 3)  # (H, W, 3)

        # Plücker coordinates: (direction, moment) where moment = origin x direction
        moment = torch.cross(origin, dirs_world, dim=-1)  # (H, W, 3)

        # Concat direction and moment
        plucker_frame = torch.cat([dirs_world, moment], dim=-1)  # (H, W, 6)
        plucker_list.append(plucker_frame)

    plucker = torch.stack(plucker_list, dim=0)  # (num_frames, H, W, 6)
    plucker = plucker.permute(3, 0, 1, 2)  # (6, num_frames, H, W)

    return plucker


def load_camera_poses(action_path, num_frames):
    """
    Load camera poses from a JSON file or numpy file.

    Supported formats:
      - JSON: list of 4x4 matrices
      - NPY/NPZ: numpy array of shape (N, 4, 4)

    Returns:
        c2ws: (num_frames, 4, 4) tensor
    """
    if action_path.endswith(".json"):
        with open(action_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            c2ws = torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, dict) and "c2ws" in data:
            c2ws = torch.tensor(data["c2ws"], dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported JSON format in {action_path}")
    elif action_path.endswith(".npy"):
        c2ws = torch.from_numpy(np.load(action_path)).float()
    elif action_path.endswith(".npz"):
        data = np.load(action_path)
        c2ws = torch.from_numpy(data["c2ws"]).float()
    else:
        raise ValueError(f"Unsupported camera pose file format: {action_path}")

    # Interpolate or truncate to match num_frames
    if c2ws.shape[0] != num_frames:
        # Simple linear interpolation along frame dimension
        indices = torch.linspace(0, c2ws.shape[0] - 1, num_frames)
        idx_low = indices.long().clamp(0, c2ws.shape[0] - 2)
        idx_high = (idx_low + 1).clamp(0, c2ws.shape[0] - 1)
        alpha = (indices - idx_low.float()).unsqueeze(-1).unsqueeze(-1)
        c2ws = c2ws[idx_low] * (1 - alpha) + c2ws[idx_high] * alpha

    return c2ws


def generate_default_orbit_camera(num_frames, radius=3.0, elevation=0.0):
    """
    Generate a default orbit camera trajectory (circular orbit).

    Returns:
        c2ws: (num_frames, 4, 4) tensor
    """
    c2ws = []
    for i in range(num_frames):
        angle = 2 * math.pi * i / num_frames
        eye = torch.tensor([
            radius * math.cos(angle),
            elevation,
            radius * math.sin(angle),
        ], dtype=torch.float32)

        # Look at origin
        forward = -eye / eye.norm()
        up = torch.tensor([0.0, 1.0, 0.0])
        right = torch.cross(forward, up)
        right = right / right.norm()
        up = torch.cross(right, forward)

        c2w = torch.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = eye
        c2ws.append(c2w)

    return torch.stack(c2ws)


@RUNNER_REGISTER("lingbot_cam_moe")
class LingBotCamMoeRunner(Wan22MoeRunner):
    """
    LingBot Camera-Controlled I2V Runner with MoE (high/low noise models).

    Key differences from Wan22MoeRunner:
      - Uses LingBotModel instead of WanModel (includes cam control weights)
      - Uses Wan2.1 VAE (stride=[4,8,8])
      - No CLIP encoder
      - Handles camera pose input (action_path) and generates Plücker embeddings
    """

    def __init__(self, config):
        # Force lingbot-specific settings
        config["use_image_encoder"] = False
        config.setdefault("vae_stride", [4, 8, 8])
        config.setdefault("patch_size", [1, 2, 2])
        config.setdefault("num_channels_latents", 16)
        config.setdefault("fps", 16)
        super().__init__(config)

    def load_transformer(self):
        """Load dual LingBot models (high_noise + low_noise) with camera control."""
        if not self.config.get("lazy_load", False) and not self.config.get("unload_modules", False):
            lora_configs = self.config.get("lora_configs")
            high_model_kwargs = {
                "model_path": self.high_noise_model_path,
                "config": self.config,
                "device": self.init_device,
                "model_type": "wan2.2_moe_high_noise",
            }
            low_model_kwargs = {
                "model_path": self.low_noise_model_path,
                "config": self.config,
                "device": self.init_device,
                "model_type": "wan2.2_moe_low_noise",
            }
            if not lora_configs:
                high_noise_model = LingBotModel(**high_model_kwargs)
                low_noise_model = LingBotModel(**low_model_kwargs)
            else:
                high_noise_model = build_wan_model_with_lora(LingBotModel, self.config, high_model_kwargs, lora_configs, model_type="high_noise_model")
                low_noise_model = build_wan_model_with_lora(LingBotModel, self.config, low_model_kwargs, lora_configs, model_type="low_noise_model")

            return MultiModelStruct([high_noise_model, low_noise_model], self.config, self.config["boundary"])
        else:
            model_struct = MultiModelStruct([None, None], self.config, self.config["boundary"])
            model_struct.low_noise_model_path = self.low_noise_model_path
            model_struct.high_noise_model_path = self.high_noise_model_path
            model_struct.init_device = self.init_device
            return model_struct

    def load_image_encoder(self):
        """No CLIP encoder for lingbot."""
        return None

    def get_encoder_output_i2v(self, clip_encoder_out, vae_encoder_out, text_encoder_output, img=None):
        """Extend encoder output with camera control data."""
        image_encoder_output = {
            "clip_encoder_out": None,
            "vae_encoder_out": vae_encoder_out,
            "c2ws_hidden_states": self._c2ws_hidden_states,
            "cam_latent": self._cam_latent,
        }
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": image_encoder_output,
        }

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_i2v(self):
        """Override to handle camera control data."""
        img, img_ori = self.read_image_input(self.input_info.image_path)

        # Run VAE encoder (Wan2.1 style - uses img tensor, not img_ori)
        vae_encode_out, latent_shape = self.run_vae_encoder(img_ori if self.vae_encoder_need_img_original else img)
        self.input_info.latent_shape = latent_shape

        # Process camera poses
        self._prepare_camera_control(latent_shape)

        # Run text encoder
        text_encoder_output = self.run_text_encoder(self.input_info)

        torch_device_module.empty_cache()
        gc.collect()

        return self.get_encoder_output_i2v(None, vae_encode_out, text_encoder_output, img)

    def _prepare_camera_control(self, latent_shape):
        """
        Prepare camera control embeddings from action_path or default.

        Sets self._c2ws_hidden_states and self._cam_latent.
        """
        num_frames = self.config["target_video_length"]
        latent_h = latent_shape[2]
        latent_w = latent_shape[3]
        image_h = latent_h * self.config["vae_stride"][1]
        image_w = latent_w * self.config["vae_stride"][2]

        # Get action_path from input_info
        action_path = getattr(self.input_info, "action_path", None)
        if action_path is None:
            action_path = self.config.get("action_path", None)

        if action_path is not None and os.path.exists(action_path):
            logger.info(f"Loading camera poses from: {action_path}")
            c2ws = load_camera_poses(action_path, num_frames)
        else:
            logger.info("No camera poses provided, using identity cameras (no camera motion)")
            c2ws = torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(num_frames, -1, -1).clone()

        # Generate Plücker embedding
        plucker = generate_plucker_embedding(c2ws, image_h, image_w, num_frames)
        # plucker: (6, num_frames, H, W)

        # Encode Plücker through VAE-like processing to get cam_latent
        # The cam_latent should have same spatial dims as the noisy latent
        # For simplicity, we downsample the Plücker embedding to match latent spatial dims
        # In the official code, Plücker goes through a dedicated encoder
        # Here we just resize to match latent dimensions
        latent_f = (num_frames - 1) // self.config["vae_stride"][0] + 1
        cam_latent = torch.nn.functional.interpolate(
            plucker.unsqueeze(0),  # (1, 6, F, H, W)
            size=(latent_f, latent_h, latent_w),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0).to(GET_DTYPE()).to(AI_DEVICE)
        # cam_latent: (6, latent_f, latent_h, latent_w)

        # c2ws hidden states: flatten the c2ws and project
        # c2ws: (num_frames, 4, 4) -> (num_frames, 16)
        c2ws_flat = c2ws.reshape(num_frames, 16).to(GET_DTYPE()).to(AI_DEVICE)
        # Downsample to latent temporal dim
        if num_frames != latent_f:
            c2ws_flat = torch.nn.functional.interpolate(
                c2ws_flat.unsqueeze(0).transpose(1, 2),  # (1, 16, num_frames)
                size=latent_f,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).squeeze(0)  # (latent_f, 16)

        self._c2ws_hidden_states = c2ws_flat
        self._cam_latent = cam_latent

        logger.info(f"Camera control prepared: c2ws_hidden_states={self._c2ws_hidden_states.shape}, cam_latent={self._cam_latent.shape}")

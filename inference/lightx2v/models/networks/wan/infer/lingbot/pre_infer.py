"""
LingBot Camera Control - Pre Inference Module.

Extends WanPreInfer to handle Plücker camera control embeddings.

Original lingbot-world logic (wan/modules/model.py):
  1. c2ws_plucker_emb -> rearrange to patches -> patch_embedding_wancamctrl (Linear: 1536->dim)
  2. c2ws_hidden_states = layer2(silu(layer1(c2ws_plucker_emb)))  # MLP: dim->dim->dim
  3. cam_emb = c2ws_plucker_emb + c2ws_hidden_states  # residual connection
  4. cam_emb is passed to each block for scale/shift injection
"""

import torch
import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.models.networks.wan.infer.module_io import GridOutput, WanPreInferModuleOutput
from lightx2v.models.networks.wan.infer.utils import sinusoidal_embedding_1d
from lightx2v.utils.envs import *
from lightx2v_platform.base.global_var import AI_DEVICE


def plucker_positional_encoding(x, num_freqs=32):
    """
    NeRF-style Fourier positional encoding for Plucker coordinates.

    Expands each channel using sinusoidal frequencies:
      x_channel -> [sin(2^0 * pi * x), cos(2^0 * pi * x), ..., sin(2^(L-1) * pi * x), cos(2^(L-1) * pi * x)]

    Args:
        x: (C, F, H, W) Plucker coordinates with C=6
        num_freqs: number of frequency bands (default 32, giving 64 dims per channel)

    Returns:
        encoded: (C * num_freqs * 2, F, H, W) = (384, F, H, W) for C=6, num_freqs=32
    """
    # freq_bands: (num_freqs,) = [2^0, 2^1, ..., 2^(L-1)]
    freq_bands = (2.0 ** torch.arange(num_freqs, dtype=x.dtype, device=x.device)) * torch.pi
    # x: (C, F, H, W) -> (C, 1, F, H, W)
    # freq_bands: (1, num_freqs, 1, 1)
    x_expanded = x.unsqueeze(1)  # (C, 1, F, H, W)
    freq_bands = freq_bands.view(1, num_freqs, 1, 1, 1)  # broadcastable
    # (C, num_freqs, F, H, W)
    x_freq = x_expanded * freq_bands
    # sin and cos: each (C, num_freqs, F, H, W)
    encoded = torch.cat([torch.sin(x_freq), torch.cos(x_freq)], dim=1)  # (C, 2*num_freqs, F, H, W)
    # Reshape to (C * 2 * num_freqs, F, H, W) = (384, F, H, W)
    C = x.shape[0]
    encoded = encoded.reshape(C * 2 * num_freqs, *x.shape[1:])
    return encoded


class LingBotPreInfer(WanPreInfer):
    """Pre-inference for LingBot camera-controlled I2V model."""

    def __init__(self, config):
        super().__init__(config)
        self.patch_size = config.get("patch_size", [1, 2, 2])
        self.plucker_num_freqs = 32  # 6 channels * 32 freqs * 2 (sin+cos) = 384 channels

    @torch.no_grad()
    def infer(self, weights, inputs, kv_start=0, kv_end=0):
        x = self.scheduler.latents
        t = self.scheduler.timestep_input

        if self.scheduler.infer_condition:
            context = inputs["text_encoder_output"]["context"]
        else:
            context = inputs["text_encoder_output"]["context_null"]

        # I2V: concat VAE encoder output (mask + encoded first frame)
        if self.task in ["i2v"]:
            if self.config.get("changing_resolution", False):
                image_encoder = inputs["image_encoder_output"]["vae_encoder_out"][self.scheduler.changing_resolution_index]
            else:
                image_encoder = inputs["image_encoder_output"]["vae_encoder_out"]

            if image_encoder is not None:
                y = image_encoder
                x = torch.cat([x, y], dim=0)

        # Standard patch embedding
        x = weights.patch_embedding.apply(x.unsqueeze(0))

        grid_sizes_t, grid_sizes_h, grid_sizes_w = x.shape[2:]
        x = x.flatten(2).transpose(1, 2).contiguous()

        # ===== Camera control injection =====
        # Following original lingbot-world logic:
        # 1. cam_latent (Plucker 6ch) -> Fourier PE -> 384ch -> rearrange to patches -> Linear projection to dim
        # 2. MLP: layer1 -> silu -> layer2 -> c2ws_hidden_states
        # 3. cam_emb = projected_plucker + c2ws_hidden_states (residual)
        cam_latent = inputs.get("image_encoder_output", {}).get("cam_latent", None)
        if cam_latent is not None:
            # cam_latent: (6, F, H, W) - raw Plucker coordinates
            # Step 0: Fourier positional encoding: (6, F, H, W) -> (384, F, H, W)
            cam_encoded = plucker_positional_encoding(cam_latent, num_freqs=self.plucker_num_freqs)

            # Step 1: Rearrange into patches
            # cam_encoded: (C_enc, F, H, W) where C_enc = 6 * 2 * num_freqs = 384
            C = cam_encoded.shape[0]  # 384
            p1, p2, p3 = self.patch_size
            F_lat = cam_encoded.shape[1] // p1
            H_lat = cam_encoded.shape[2] // p2
            W_lat = cam_encoded.shape[3] // p3

            # Reshape: (C, F*p1, H*p2, W*p3) -> (F, H, W, C*p1*p2*p3) -> (F*H*W, C*p1*p2*p3)
            cam_patches = cam_encoded.reshape(C, F_lat, p1, H_lat, p2, W_lat, p3)
            cam_patches = cam_patches.permute(1, 3, 5, 0, 2, 4, 6)  # (F, H, W, C, p1, p2, p3)
            cam_patches = cam_patches.reshape(F_lat * H_lat * W_lat, C * p1 * p2 * p3)  # (seq_len, 1536)

            # Linear projection: 1536 -> dim (5120)
            c2ws_plucker_emb = weights.patch_embedding_wancamctrl.apply(cam_patches)

            # MLP: c2ws_hidden_states = layer2(silu(layer1(c2ws_plucker_emb)))
            c2ws_hidden_states = weights.c2ws_hidden_states_layer1.apply(c2ws_plucker_emb)
            c2ws_hidden_states = torch.nn.functional.silu(c2ws_hidden_states)
            c2ws_hidden_states = weights.c2ws_hidden_states_layer2.apply(c2ws_hidden_states)

            # Residual connection: cam_emb = projected_plucker + c2ws_hidden_states
            cam_emb = c2ws_plucker_emb + c2ws_hidden_states
        else:
            cam_emb = None

        # Time embedding
        embed = sinusoidal_embedding_1d(self.freq_dim, t.flatten())
        if self.sensitive_layer_dtype != self.infer_dtype:
            embed = weights.time_embedding_0.apply(embed.to(self.sensitive_layer_dtype))
        else:
            embed = weights.time_embedding_0.apply(embed)
        embed = torch.nn.functional.silu(embed)
        embed = weights.time_embedding_2.apply(embed)
        embed0 = torch.nn.functional.silu(embed)

        embed0 = weights.time_projection_1.apply(embed0).unflatten(1, (6, self.dim))

        # Text embeddings
        if self.sensitive_layer_dtype != self.infer_dtype:
            out = weights.text_embedding_0.apply(context.squeeze(0).to(self.sensitive_layer_dtype))
        else:
            out = weights.text_embedding_0.apply(context.squeeze(0))
        out = torch.nn.functional.gelu(out, approximate="tanh")
        context = weights.text_embedding_2.apply(out)

        if self.clean_cuda_cache:
            del out
            torch.cuda.empty_cache()

        # No CLIP embedding for lingbot (use_image_encoder=False)

        grid_sizes = GridOutput(
            tensor=torch.tensor([[grid_sizes_t, grid_sizes_h, grid_sizes_w]], dtype=torch.int32, device=x.device),
            tuple=(grid_sizes_t, grid_sizes_h, grid_sizes_w),
        )

        if self.cos_sin is None or self.grid_sizes != grid_sizes.tuple:
            freqs = self.freqs.clone()
            self.grid_sizes = grid_sizes.tuple
            self.cos_sin = self.prepare_cos_sin(grid_sizes.tuple, freqs)

        return WanPreInferModuleOutput(
            embed=embed,
            grid_sizes=grid_sizes,
            x=x.squeeze(0),
            embed0=embed0.squeeze(0),
            context=context,
            cos_sin=self.cos_sin,
            adapter_args={
                "motion_vec": None,
                "cam_emb": cam_emb,
                "cam_tokens": None,
            },
        )

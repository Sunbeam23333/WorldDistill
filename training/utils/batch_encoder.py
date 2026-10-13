"""Utilities for converting raw-video batches into latent-space training batches."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import torch
from loguru import logger


class DiffusersRawBatchEncoder:
    """Encode raw video batches with auxiliary modules loaded from a diffusers pipeline.

    This adapter keeps the training loop in latent space while allowing the dataset to
    provide raw `pixel_values` + prompt text. Missing model-specific conditions
    are errors; raw examples never silently become unconditional examples.
    """

    def __init__(self, model_path: str, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype

        try:
            from diffusers import DiffusionPipeline
        except ImportError as exc:
            raise ImportError(
                "Raw video mode requires diffusers to be installed. "
                "Please `pip install diffusers transformers accelerate`."
            ) from exc

        logger.info(f"Loading diffusers auxiliary encoders from {model_path} for raw-video training")
        with (Path(model_path) / "model_index.json").open() as f:
            pipeline_config = json.load(f)
        unused = {name: None for name in ("transformer", "transformer_2", "unet") if name in pipeline_config}
        pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype, **unused)

        self.pipe = pipe
        self.vae = getattr(pipe, "vae", None)
        self.text_encoder = getattr(pipe, "text_encoder", None)
        self.tokenizer = getattr(pipe, "tokenizer", None)
        self.image_encoder = getattr(pipe, "image_encoder", None)
        self.feature_extractor = getattr(pipe, "feature_extractor", None)
        self.encode_prompt = getattr(pipe, "encode_prompt", None)
        self.vae_scaling_factor = getattr(getattr(self.vae, "config", None), "scaling_factor", 1.0)
        self.temporal_compression_ratio = int(getattr(pipe, "vae_scale_factor_temporal",
            getattr(pipe, "vae_temporal_compression_ratio", getattr(self.vae, "temporal_compression_ratio",
                getattr(getattr(self.vae, "config", None), "temporal_compression_ratio", 1)))))
        if self.temporal_compression_ratio < 1:
            raise ValueError("VAE temporal compression ratio must be positive")

        if self.vae is None:
            raise RuntimeError(
                "Raw video mode requires a VAE in the teacher pipeline, but no `vae` component was found."
            )
        # Auxiliary-only loading must still use the checkpoint's vision width,
        # rather than the pipeline constructor's default for transformer=None.
        transformer_config = Path(model_path) / "transformer" / "config.json"
        if "HunyuanVideo15" in type(pipe).__name__ and transformer_config.is_file():
            denoiser_config = json.loads(transformer_config.read_text())
            pipe.vision_states_dim = denoiser_config.get("image_embed_dim", pipe.vision_states_dim)

        self._freeze_module(self.vae)
        self._freeze_module(self.image_encoder)

        self.vae.to(device)
        for attr in ("text_encoder", "text_encoder_2", "text_encoder_3", "connectors", "audio_vae"):
            encoder = getattr(pipe, attr, None)
            self._freeze_module(encoder)
            if encoder is not None:
                encoder.to(device)
        if self.image_encoder is not None:
            self.image_encoder.to(device)

        for attr in ("transformer", "transformer_2", "unet", "prior", "decoder"):
            if hasattr(self.pipe, attr):
                try:
                    setattr(self.pipe, attr, None)
                except Exception:
                    pass

    @staticmethod
    def _freeze_module(module: Any) -> None:
        if module is None or not hasattr(module, "parameters"):
            return
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def encode_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if "latents" not in batch:
            if "pixel_values" not in batch:
                raise KeyError("Raw batch encoding requires `pixel_values` when `latents` are absent.")
            batch["latents"] = self._encode_pixels(batch["pixel_values"])
            raw_t, latent_t = batch["pixel_values"].shape[2], batch["latents"].shape[2]
            ratio = getattr(self, "temporal_compression_ratio", 1)
            if ratio < 1:
                raise ValueError("VAE temporal compression ratio must be positive")
            if latent_t != raw_t and latent_t != (raw_t - 1) // ratio + 1:
                raise ValueError("VAE temporal layout has no declared causal frame/control mapping")
            # A VAE may preserve time (e.g. an explicitly frame-wise encoder)
            # even if its enclosing pipeline advertises a compression factor.
            anchors = torch.arange(latent_t, device=self.device) * (1 if latent_t == raw_t else ratio)
            for key in ("actions", "camera_poses", "camera_intrinsics"):
                if key not in batch:
                    continue
                value = batch[key]
                if value.ndim < 2 or value.shape[1] != raw_t:
                    raise ValueError(f"Raw {key} must have layout B,T,... aligned to sampled video frames")
                batch[key] = value.index_select(1, anchors.to(value.device))
            for key in ("keyboard_actions", "mouse_actions"):
                if key in batch:
                    value = batch[key]
                    if value.ndim != 3 or value.shape[1] != raw_t:
                        raise ValueError(f"Raw {key} must retain pixel-rate B,T,D controls")
                    # Matrix's action encoder consumes complete causal windows
                    # at pixel rate. Do NOT downsample them like generic actions.
                    if raw_t != (latent_t - 1) * ratio + 1:
                        raise ValueError("Matrix pixel-rate controls require exact causal VAE frame alignment")

        if "encoder_hidden_states" not in batch and "text" in batch:
            prompt_embeds = self._encode_text(batch["text"])
            if isinstance(prompt_embeds, dict):
                batch.update(prompt_embeds)
            elif prompt_embeds is not None:
                batch["encoder_hidden_states"] = prompt_embeds
            else:
                raise RuntimeError("Prompt encoding produced no conditioning; refusing unconditional fallback")

        if "HunyuanVideo15" in type(getattr(self, "pipe", None)).__name__:
            self._encode_hunyuan15_conditions(batch)
        if "LTX2" in type(getattr(self, "pipe", None)).__name__ and "audio_latents" not in batch:
            # Audio inputs are log-mel features from the checkpoint's prescribed
            # preprocessing; no guessed sample-rate/STFT transform is applied.
            if "audio_pixel_values" not in batch:
                raise ValueError("LTX2 raw AV encoding requires audio_pixel_values (checkpoint log-mel features) "
                                 "or normalized packed audio_latents")
            with torch.no_grad():
                audio = self._extract_latents(self.pipe.audio_vae.encode(
                    batch["audio_pixel_values"].to(self.device, dtype=self.dtype)))
                if audio.ndim != 4:
                    raise ValueError("LTX2 audio VAE must encode B,C,L,M latents")
                packed = audio.transpose(1, 2).flatten(2, 3)
                mean = self.pipe.audio_vae.latents_mean.to(packed)
                std = self.pipe.audio_vae.latents_std.to(packed)
                if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
                    raise ValueError("Invalid audio VAE normalization statistics")
                batch["audio_latents"] = (packed - mean) / std
        return batch

    @torch.no_grad()
    def _encode_hunyuan15_conditions(self, batch: Dict[str, Any]) -> None:
        """Match the upstream latent+mask and SigLIP branches for T2V/I2V."""
        latents = batch["latents"]
        if latents.ndim != 5:
            raise ValueError("HunyuanVideo1.5 conditioning requires B,C,T,H,W latents")
        b, c, t, h, w = latents.shape
        is_i2v = "ImageToVideo" in type(self.pipe).__name__
        if not is_i2v:
            batch.setdefault("image_cond", latents.new_zeros(b, c + 1, t, h, w))
            batch.setdefault("image_embeds", latents.new_zeros(
                b, self.pipe.vision_num_semantic_tokens, self.pipe.vision_states_dim))
            return
        if "image_cond" in batch and "image_embeds" in batch:
            return
        pixels = batch.get("conditioning_pixel_values", batch.get("pixel_values"))
        if not torch.is_tensor(pixels) or pixels.ndim not in (4, 5) or pixels.shape[0] != b or pixels.shape[1] != 3:
            raise ValueError("HunyuanVideo1.5 I2V requires aligned RGB conditioning_pixel_values or raw pixel_values")
        first = pixels[:, :, 0] if pixels.ndim == 5 else pixels
        if not torch.isfinite(first).all() or first.min() < -1.001 or first.max() > 1.001:
            raise ValueError("I2V conditioning pixels must be finite and normalized to [-1,1]")
        if "image_cond" not in batch:
            # Encode only the reference image, never target future video frames.
            encoded = self.vae.encode(first.unsqueeze(2).to(self.device, dtype=self.dtype))
            distribution = getattr(encoded, "latent_dist", None)
            image_latent = distribution.mode() if distribution is not None else self._extract_latents(encoded)
            image_latent = self._normalize_latents(image_latent)
            if image_latent.shape != (b, c, 1, h, w):
                raise ValueError("I2V reference image VAE latent does not match the target spatial/channel layout")
            condition, mask = latents.new_zeros(latents.shape), latents.new_zeros(b, 1, t, h, w)
            condition[:, :, :1] = image_latent.to(latents)
            mask[:, :, :1] = 1
            batch["image_cond"] = torch.cat((condition, mask), dim=1)
        if "image_embeds" not in batch:
            from PIL import Image
            # Upstream encode_image repeats one image batch_size times; call it
            # once per sample so different batch members keep their own image.
            images = ((first.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte()
            batch["image_embeds"] = torch.cat([
                self.pipe.encode_image(Image.fromarray(value.permute(1, 2, 0).numpy()),
                    batch_size=1, device=self.device, dtype=self.dtype)
                for value in images
            ], dim=0)

    def _encode_pixels(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixels = pixel_values.to(self.device, dtype=self.dtype)
        if pixels.dim() not in (4, 5):
            raise ValueError(f"Expected `pixel_values` to be 4D/5D, got shape={tuple(pixels.shape)}")

        with torch.no_grad():
            try:
                encoded = self.vae.encode(pixels)
                latents = self._extract_latents(encoded)
                latents = self._reshape_latents_if_needed(latents, pixels.shape)
            except Exception as exc:
                if pixels.dim() != 5 or getattr(self, "temporal_compression_ratio", 1) != 1:
                    raise RuntimeError(f"VAE encode failed for raw batch: {exc}") from exc

                logger.warning(
                    "Direct 5D VAE encoding failed; falling back to per-frame encoding for raw-video training. "
                    f"Reason: {exc}"
                )
                batch_size, channels, num_frames, height, width = pixels.shape
                flat_pixels = pixels.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
                encoded = self.vae.encode(flat_pixels)
                latents = self._extract_latents(encoded)
                if latents.dim() != 4:
                    raise RuntimeError(
                        "Per-frame VAE fallback expected 4D latents, "
                        f"but received shape={tuple(latents.shape)}"
                    )
                latents = latents.reshape(batch_size, num_frames, *latents.shape[1:]).permute(0, 2, 1, 3, 4).contiguous()

        return self._normalize_latents(latents)

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        config = self.vae.config
        # LTX stores these as registered VAE buffers, not config attributes.
        mean = getattr(self.vae, "latents_mean", getattr(config, "latents_mean", None))
        std = getattr(self.vae, "latents_std", getattr(config, "latents_std", None))
        if mean is not None or std is not None:
            if mean is None or std is None:
                raise ValueError("VAE must declare both latents_mean and latents_std")
            shape = (1, -1, *([1] * (latents.ndim - 2)))
            mean = torch.as_tensor(mean, device=latents.device, dtype=torch.float32).reshape(shape)
            std = torch.as_tensor(std, device=latents.device, dtype=torch.float32).reshape(shape)
            if (std <= 0).any() or not torch.isfinite(std).all() or not torch.isfinite(mean).all() or mean.shape[1] != latents.shape[1] or std.shape[1] != latents.shape[1]:
                raise ValueError("Invalid per-channel VAE normalization statistics")
            scale = self.vae_scaling_factor if "LTX" in type(self.vae).__name__ else 1.0
            if not isinstance(scale, (float, int)) or scale <= 0:
                raise ValueError("Invalid VAE scaling factor")
            return ((latents.float() - mean) * scale / std).to(latents.dtype)
        shift = getattr(config, "shift_factor", 0.0) or 0.0
        return (latents - shift) * self.vae_scaling_factor

    def _encode_text(self, texts: Sequence[str] | str) -> torch.Tensor | None:
        prompt_list = self._normalize_prompts(texts)
        if not prompt_list:
            return None

        if self.encode_prompt is not None:
            try:
                prompt_embeds = self._encode_text_via_pipeline(prompt_list)
                if prompt_embeds is not None:
                    return prompt_embeds
            except Exception as exc:
                raise RuntimeError(f"Model-specific prompt encoding failed: {exc}") from exc

        if self.tokenizer is None or self.text_encoder is None:
            logger.warning(
                "Raw-video batch has prompt text, but tokenizer/text_encoder are unavailable; "
                "training will continue without `encoder_hidden_states`."
            )
            return None

        max_length = getattr(self.tokenizer, "model_max_length", 77)
        if not isinstance(max_length, int) or max_length <= 0 or max_length > 8192:
            max_length = 512

        tokenized = self.tokenizer(
            prompt_list,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

        with torch.no_grad():
            outputs = self.text_encoder(**tokenized)
        return {"encoder_hidden_states": self._extract_hidden_states(outputs),
                "encoder_attention_mask": tokenized["attention_mask"]}

    def _encode_text_via_pipeline(self, prompt_list: Sequence[str]) -> torch.Tensor | None:
        if self.encode_prompt is None:
            return None

        signature = inspect.signature(self.encode_prompt)
        kwargs: Dict[str, Any] = {}
        candidates = {
            "prompt": list(prompt_list),
            "prompt_2": list(prompt_list),
            "device": self.device,
            "do_classifier_free_guidance": False,
            "negative_prompt": None,
            "negative_prompt_2": None,
            "num_images_per_prompt": 1,
            "num_videos_per_prompt": 1,
            "batch_size": len(prompt_list),
            "dtype": self.dtype,
        }
        for key, value in candidates.items():
            if key in signature.parameters:
                kwargs[key] = value

        encoded = self.encode_prompt(**kwargs)
        if isinstance(encoded, torch.Tensor):
            return encoded.to(self.device)
        if isinstance(encoded, (tuple, list)):
            family = type(self.pipe).__name__
            if "LTX2" in family:
                if len(encoded) < 2 or not all(torch.is_tensor(v) for v in encoded[:2]):
                    raise TypeError("LTX2 prompt encoder must return embeddings and attention mask")
                with torch.no_grad():
                    video, audio, mask = self.pipe.connectors(encoded[0], encoded[1],
                        padding_side=getattr(self.tokenizer, "padding_side", "left"))
                return {"encoder_hidden_states": video.to(self.device),
                        "audio_encoder_hidden_states": audio.to(self.device),
                        "encoder_attention_mask": mask.to(self.device),
                        "audio_encoder_attention_mask": mask.to(self.device)}
            if "HunyuanVideo15" in family:
                names = ("encoder_hidden_states", "encoder_attention_mask", "encoder_hidden_states_2", "encoder_attention_mask_2")
            elif "HunyuanVideo" in family:
                names = ("encoder_hidden_states", "pooled_projections", "encoder_attention_mask")
            elif "LTX" in family:
                names = ("encoder_hidden_states", "encoder_attention_mask")
            else:
                names = ("encoder_hidden_states",)
            if len(encoded) < len(names) or any(not isinstance(value, torch.Tensor) for value in encoded[:len(names)]):
                raise TypeError(f"{family}.encode_prompt returned an invalid conditioning tuple")
            return {name: value.to(self.device) for name, value in zip(names, encoded)}
        return None

    @staticmethod
    def _normalize_prompts(texts: Sequence[str] | str) -> list[str]:
        if isinstance(texts, str):
            return [texts]
        if isinstance(texts, Iterable):
            return ["" if text is None else str(text) for text in texts]
        return [str(texts)]

    @staticmethod
    def _extract_hidden_states(outputs: Any) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return outputs
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        if isinstance(outputs, (tuple, list)) and outputs:
            first = outputs[0]
            if isinstance(first, torch.Tensor):
                return first
        raise RuntimeError(f"Unable to extract prompt embeddings from type={type(outputs)}")

    @staticmethod
    def _extract_latents(encoded: Any) -> torch.Tensor:
        if isinstance(encoded, torch.Tensor):
            return encoded
        if hasattr(encoded, "latent_dist") and encoded.latent_dist is not None:
            return encoded.latent_dist.sample()
        if hasattr(encoded, "latents") and encoded.latents is not None:
            return encoded.latents
        if isinstance(encoded, (tuple, list)) and encoded:
            first = encoded[0]
            if isinstance(first, torch.Tensor):
                return first
            if hasattr(first, "sample"):
                return first.sample()
        raise RuntimeError(f"Unable to extract latents from VAE output type={type(encoded)}")

    @staticmethod
    def _reshape_latents_if_needed(latents: torch.Tensor, pixel_shape: torch.Size) -> torch.Tensor:
        if len(pixel_shape) != 5 or latents.dim() != 4:
            return latents

        batch_size, _, num_frames, _, _ = pixel_shape
        if latents.shape[0] != batch_size * num_frames:
            return latents

        return latents.reshape(batch_size, num_frames, *latents.shape[1:]).permute(0, 2, 1, 3, 4).contiguous()

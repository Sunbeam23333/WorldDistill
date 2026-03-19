"""Utilities for converting raw-video batches into latent-space training batches."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Iterable, Sequence

import torch
from loguru import logger


class DiffusersRawBatchEncoder:
    """Encode raw video batches with auxiliary modules loaded from a diffusers pipeline.

    This adapter keeps the training loop in latent space while allowing the dataset to
    provide raw `pixel_values` + prompt text. It is intentionally conservative:
    - latent encoding is required;
    - prompt encoding is best-effort and falls back gracefully when unavailable.
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
        pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)

        self.pipe = pipe
        self.vae = getattr(pipe, "vae", None)
        self.text_encoder = getattr(pipe, "text_encoder", None)
        self.tokenizer = getattr(pipe, "tokenizer", None)
        self.image_encoder = getattr(pipe, "image_encoder", None)
        self.feature_extractor = getattr(pipe, "feature_extractor", None)
        self.encode_prompt = getattr(pipe, "encode_prompt", None)
        self.vae_scaling_factor = getattr(getattr(self.vae, "config", None), "scaling_factor", 1.0)

        if self.vae is None:
            raise RuntimeError(
                "Raw video mode requires a VAE in the teacher pipeline, but no `vae` component was found."
            )

        self._freeze_module(self.vae)
        self._freeze_module(self.text_encoder)
        self._freeze_module(self.image_encoder)

        self.vae.to(device)
        if self.text_encoder is not None:
            self.text_encoder.to(device)
        if self.image_encoder is not None:
            self.image_encoder.to(device)

        for attr in ("transformer", "unet", "scheduler", "prior", "decoder"):
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

        if "encoder_hidden_states" not in batch and "text" in batch:
            prompt_embeds = self._encode_text(batch["text"])
            if prompt_embeds is not None:
                batch["encoder_hidden_states"] = prompt_embeds

        return batch

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
                if pixels.dim() != 5:
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

        return latents * self.vae_scaling_factor

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
                logger.warning(f"Pipeline `encode_prompt` failed in raw-video mode, falling back: {exc}")

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
        return self._extract_hidden_states(outputs)

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
        }
        for key, value in candidates.items():
            if key in signature.parameters:
                kwargs[key] = value

        encoded = self.encode_prompt(**kwargs)
        if isinstance(encoded, torch.Tensor):
            return encoded.to(self.device)
        if isinstance(encoded, (tuple, list)):
            for item in encoded:
                if isinstance(item, torch.Tensor):
                    return item.to(self.device)
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

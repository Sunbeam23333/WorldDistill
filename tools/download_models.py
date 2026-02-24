#!/usr/bin/env python3
"""
Download video generation / world model weights from HuggingFace.

Proxy & credentials are read from environment variables or .env file.
See .env.example for configuration details.

Usage:
    python tools/download_models.py --model wan2.2_moe --target-root ./models
    python tools/download_models.py --model all --target-root ./models
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm import tqdm

# ============================================================================
# Proxy Configuration - Read from environment variables
# ============================================================================
PROXY_USERNAME = os.environ.get("PROXY_USERNAME", "")
PROXY_PASSWORD = os.environ.get("PROXY_PASSWORD", "")
PROXY_HOST = os.environ.get("PROXY_HOST", "")
PROXY_PORT = os.environ.get("PROXY_PORT", "")

# ============================================================================
# Model Registry
# ============================================================================
MODEL_REGISTRY: dict[str, dict] = {
    # --- Video Generation Models ---
    "wan2.2_moe": {
        "repo": "Wan-AI/Wan2.2-T2V-A14B",
        "local_name": "Wan2.2-T2V-A14B",
        "desc": "Wan 2.2 MoE T2V (14B, original format)",
    },
    "wan2.2_moe_diffusers": {
        "repo": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "local_name": "Wan2.2-T2V-A14B-Diffusers",
        "desc": "Wan 2.2 MoE T2V (14B, Diffusers format)",
    },
    "wan2.2_moe_i2v": {
        "repo": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "local_name": "Wan2.2-I2V-A14B-Diffusers",
        "desc": "Wan 2.2 MoE I2V (14B, Diffusers format)",
    },
    "hunyuan_video_1.5": {
        "repo": "tencent/HunyuanVideo",
        "local_name": "HunyuanVideo",
        "desc": "HunyuanVideo 1.5",
    },
    "skyreels_v2": {
        "repo": "Skywork/SkyReels-V2-T2V-14B",
        "local_name": "SkyReels-V2-T2V-14B",
        "desc": "SkyReels-V2 T2V (Diffusion Forcing)",
    },
    # --- Distilled Models ---
    "wan2.2_distill_lora": {
        "repo": "lightx2v/Wan2.2-Distill-Loras",
        "local_name": "Wan2.2-Distill-Loras",
        "desc": "Wan 2.2 Step Distillation LoRA weights",
    },
    "wan2.2_distill_model": {
        "repo": "lightx2v/Wan2.2-Distill-Models",
        "local_name": "Wan2.2-Distill-Models",
        "desc": "Wan 2.2 Step Distillation full model weights",
    },
    # --- Text Encoders / Embeddings ---
    "jina_embeddings": {
        "repo": "jinaai/jina-embeddings-v3",
        "local_name": "jina-embeddings-v3",
        "desc": "Jina Embeddings v3",
    },
}

DEFAULT_TARGET = os.environ.get("MODEL_ROOT", "./models")


def setup_proxy() -> None:
    """Configure proxy from environment variables."""
    if os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy"):
        logging.info("Proxy already configured in environment")
        return

    if not PROXY_USERNAME or not PROXY_PASSWORD or not PROXY_HOST:
        logging.info("No proxy configured. Set PROXY_USERNAME/PROXY_PASSWORD/PROXY_HOST env vars if needed.")
        return

    proxy_url = f"http://{PROXY_USERNAME}:{PROXY_PASSWORD}@{PROXY_HOST}:{PROXY_PORT}"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ[key] = proxy_url
    logging.info("Proxy configured: %s:%s", PROXY_HOST, PROXY_PORT)


def parse_args() -> argparse.Namespace:
    model_choices = list(MODEL_REGISTRY.keys()) + ["all"]
    parser = argparse.ArgumentParser(description="Download models from HuggingFace")
    parser.add_argument("--target-root", default=DEFAULT_TARGET, help="Destination directory")
    parser.add_argument("--model", nargs="+", choices=model_choices, default=["all"], help="Models to download")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel download workers")
    parser.add_argument("--revision", default=None, help="Specific revision")
    parser.add_argument("--resume", action="store_true", help="Resume partial downloads")
    parser.add_argument("--list", action="store_true", help="List available models and exit")
    return parser.parse_args()


def list_models() -> None:
    """Print available models."""
    print("\nAvailable models:\n")
    for key, info in MODEL_REGISTRY.items():
        print(f"  {key:30s}  {info['repo']:45s}  {info['desc']}")
    print()


def download(target_root: Path, model_keys: list[str], max_workers: int,
             revision: str | None, resume: bool) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    entries = [(k, MODEL_REGISTRY[k]) for k in model_keys]

    with tqdm(total=len(entries), desc="Overall", unit="model") as pbar:
        for idx, (key, info) in enumerate(entries, 1):
            repo_id = info["repo"]
            local_dir = target_root / info["local_name"]

            if local_dir.exists() and any(local_dir.iterdir()):
                logging.info("[%d/%d] %s already exists, skipping", idx, len(entries), repo_id)
                pbar.update(1)
                continue

            logging.info("[%d/%d] Downloading %s -> %s", idx, len(entries), repo_id, local_dir)
            pbar.set_description(f"Downloading {info['local_name']}")

            try:
                snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                    resume_download=resume,
                    max_workers=max_workers,
                    tqdm_class=tqdm,
                )
                logging.info("Successfully downloaded %s", repo_id)
            except Exception as e:
                logging.error("Failed to download %s: %s", repo_id, e)
                raise
            finally:
                pbar.update(1)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.list:
        list_models()
        return

    setup_proxy()

    if "all" in args.model:
        model_keys = list(MODEL_REGISTRY.keys())
    else:
        model_keys = args.model

    logging.info("Models to download: %s", ", ".join(model_keys))
    try:
        download(Path(args.target_root), model_keys, args.max_workers, args.revision, args.resume)
        logging.info("All models downloaded successfully to %s", args.target_root)
    except KeyboardInterrupt:
        logging.warning("Download interrupted. Use --resume to continue later.")
        sys.exit(1)
    except Exception as e:
        logging.error("Download failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

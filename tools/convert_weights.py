#!/usr/bin/env python3
"""
Model weight format conversion tool.

Converts between different weight formats:
  - Diffusers <-> Original (e.g., Wan2.2 original vs Diffusers format)
  - Merge LoRA weights into base model
  - Extract distilled sub-models from MoE dual-model structure

Usage:
    python tools/convert_weights.py --mode diffusers_to_original --input ./models/Wan2.2-Diffusers --output ./models/Wan2.2-Original
    python tools/convert_weights.py --mode merge_lora --base ./models/Wan2.2 --lora ./models/Wan2.2-Distill-Loras --output ./models/Wan2.2-Merged
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def merge_lora_weights(base_path: Path, lora_path: Path, output_path: Path, alpha: float = 1.0) -> None:
    """Merge LoRA weights into base model."""
    logging.info("Loading base model from %s", base_path)
    base_files = sorted(base_path.glob("*.safetensors"))
    if not base_files:
        raise FileNotFoundError(f"No safetensors files found in {base_path}")

    logging.info("Loading LoRA weights from %s", lora_path)
    lora_files = sorted(lora_path.glob("*.safetensors"))
    if not lora_files:
        raise FileNotFoundError(f"No safetensors files found in {lora_path}")

    # Load LoRA state dict
    lora_state = {}
    for f in lora_files:
        lora_state.update(load_file(str(f)))

    # Parse LoRA pairs (lora_A, lora_B)
    lora_pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in lora_state.items():
        if "lora_A" in key:
            base_key = key.replace(".lora_A.weight", ".weight")
            lora_pairs.setdefault(base_key, {})["A"] = tensor
        elif "lora_B" in key:
            base_key = key.replace(".lora_B.weight", ".weight")
            lora_pairs.setdefault(base_key, {})["B"] = tensor

    logging.info("Found %d LoRA pairs to merge", len(lora_pairs))

    output_path.mkdir(parents=True, exist_ok=True)
    for base_file in base_files:
        state = load_file(str(base_file))
        merged = 0
        for key in list(state.keys()):
            if key in lora_pairs and "A" in lora_pairs[key] and "B" in lora_pairs[key]:
                A = lora_pairs[key]["A"].to(state[key].dtype)
                B = lora_pairs[key]["B"].to(state[key].dtype)
                delta = alpha * (B @ A)
                state[key] = state[key] + delta
                merged += 1

        out_file = output_path / base_file.name
        save_file(state, str(out_file))
        logging.info("Saved %s (merged %d LoRA layers)", out_file.name, merged)

    logging.info("LoRA merge complete -> %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Model weight format conversion")
    parser.add_argument("--mode", choices=["merge_lora", "diffusers_to_original", "original_to_diffusers"],
                        required=True, help="Conversion mode")
    parser.add_argument("--base", type=Path, help="Base model path (for merge_lora)")
    parser.add_argument("--lora", type=Path, help="LoRA weights path (for merge_lora)")
    parser.add_argument("--input", type=Path, help="Input model path (for format conversion)")
    parser.add_argument("--output", type=Path, required=True, help="Output path")
    parser.add_argument("--alpha", type=float, default=1.0, help="LoRA merge alpha")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

    if args.mode == "merge_lora":
        if not args.base or not args.lora:
            parser.error("--base and --lora required for merge_lora mode")
        merge_lora_weights(args.base, args.lora, args.output, args.alpha)
    else:
        # TODO: Implement diffusers <-> original format conversion
        logging.warning("Mode '%s' is not yet implemented. Contributions welcome!", args.mode)
        raise NotImplementedError(f"Conversion mode '{args.mode}' coming soon.")


if __name__ == "__main__":
    main()

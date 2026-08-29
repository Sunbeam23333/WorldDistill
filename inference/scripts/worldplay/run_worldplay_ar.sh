#!/bin/bash
# Run WorldPlay AR model inference with LightX2V
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIGHTX2V_PATH="${LIGHTX2V_PATH:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export PYTHONPATH="${PYTHONPATH:-}:${LIGHTX2V_PATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Model paths
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the HunyuanVideo-1.5 checkpoint}"
AR_ACTION_MODEL_PATH="${AR_ACTION_MODEL_PATH:?Set AR_ACTION_MODEL_PATH to the WorldPlay AR action checkpoint}"

# Input parameters
PROMPT='A paved pathway leads towards a stone arch bridge spanning a calm body of water. Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky. The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere.'
IMAGE_PATH="${IMAGE_PATH:?Set IMAGE_PATH to the initial frame}"
POSE='d-31'  # Camera trajectory: backward movement for 31 latents
SEED=1

# Output
OUTPUT_PATH="${OUTPUT_PATH:-${LIGHTX2V_PATH}/save_results/HY-WorldPlay/worldplay_ar_test.mp4}"

# Create output directory
mkdir -p "$(dirname "${OUTPUT_PATH}")"

# Run inference
python -m lightx2v.infer \
    --model_cls worldplay_ar \
    --task i2v \
    --model_path "${MODEL_PATH}" \
    --config_json "${LIGHTX2V_PATH}/configs/worldplay/worldplay_ar_i2v_480p.json" \
    --prompt "$PROMPT" \
    --image_path "${IMAGE_PATH}" \
    --pose "$POSE" \
    --action_ckpt "${AR_ACTION_MODEL_PATH}" \
    --seed "${SEED}" \
    --save_result_path "${OUTPUT_PATH}"

echo "Video saved to: ${OUTPUT_PATH}"

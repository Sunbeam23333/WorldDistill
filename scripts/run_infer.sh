#!/bin/bash
# ============================================================================
# WorldDistill - Unified Inference Script
#
# Usage:
#   bash scripts/run_infer.sh --model wan2.2_moe --task t2v --prompt "A cat" --gpus 8
#   bash scripts/run_infer.sh --model wan2.2_moe_distill --task t2v --steps 4 --gpus 8
#   bash scripts/run_infer.sh --model lingbot_cam_moe --task i2v --image input.jpg --gpus 8
# ============================================================================
set -e

# ===== Default Configuration =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
INFER_DIR="${PROJECT_ROOT}/inference"

MODEL_CLS="wan2.2_moe"
TASK="t2v"
MODEL_PATH="${MODEL_ROOT:-${PROJECT_ROOT}/models}"
SAVE_DIR="${RESULT_ROOT:-${PROJECT_ROOT}/results}"
CONFIG_JSON=""
PROMPT="A white cat wearing sunglasses on a surfboard at summer beach."
NEGATIVE_PROMPT=""
IMAGE_PATH=""
SEED=42
NUM_GPUS=1

# ===== Parse Arguments =====
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_CLS="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --config) CONFIG_JSON="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --negative_prompt) NEGATIVE_PROMPT="$2"; shift 2 ;;
        --image) IMAGE_PATH="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --gpus) NUM_GPUS="$2"; shift 2 ;;
        --save_dir) SAVE_DIR="$2"; shift 2 ;;
        --help)
            echo "Usage: bash run_infer.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model        Model class (default: wan2.2_moe)"
            echo "  --task         Task type: t2v, i2v (default: t2v)"
            echo "  --model_path   Path to model weights"
            echo "  --config       Path to config JSON"
            echo "  --prompt       Text prompt"
            echo "  --image        Input image path (for i2v)"
            echo "  --seed         Random seed (default: 42)"
            echo "  --gpus         Number of GPUs (default: 1)"
            echo "  --save_dir     Output directory"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ===== Auto-detect config if not specified =====
if [ -z "${CONFIG_JSON}" ]; then
    # Try to find a matching config in inference/configs/
    CONFIG_DIR="${INFER_DIR}/configs"
    case ${MODEL_CLS} in
        wan2.2_moe*) CONFIG_JSON="${CONFIG_DIR}/wan22/wan_moe_${TASK}.json" ;;
        lingbot*) CONFIG_JSON="${CONFIG_DIR}/lingbot/lingbot_cam_moe_i2v.json" ;;
        hunyuan*) CONFIG_JSON="${CONFIG_DIR}/hunyuan_video_15/hunyuan_video_15_${TASK}.json" ;;
        *) echo "WARNING: No auto-detected config for model_cls=${MODEL_CLS}. Specify --config." ;;
    esac
fi

# ===== Setup Environment =====
if [ -f "${INFER_DIR}/scripts/base/base.sh" ]; then
    source "${INFER_DIR}/scripts/base/base.sh"
fi

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
mkdir -p "${SAVE_DIR}"

echo "============================================"
echo "WorldDistill Inference"
echo "  Model:  ${MODEL_CLS}"
echo "  Task:   ${TASK}"
echo "  GPUs:   ${NUM_GPUS}"
echo "  Config: ${CONFIG_JSON}"
echo "============================================"

# ===== Build Command =====
CMD="torchrun --nproc_per_node=${NUM_GPUS} -m lightx2v.infer"
CMD+=" --model_cls ${MODEL_CLS}"
CMD+=" --task ${TASK}"
CMD+=" --model_path ${MODEL_PATH}"
CMD+=" --seed ${SEED}"
CMD+=" --prompt \"${PROMPT}\""
CMD+=" --save_result_path ${SAVE_DIR}/output.mp4"

if [ -n "${CONFIG_JSON}" ] && [ -f "${CONFIG_JSON}" ]; then
    CMD+=" --config_json ${CONFIG_JSON}"
fi

if [ -n "${NEGATIVE_PROMPT}" ]; then
    CMD+=" --negative_prompt \"${NEGATIVE_PROMPT}\""
fi

if [ -n "${IMAGE_PATH}" ]; then
    CMD+=" --image_path ${IMAGE_PATH}"
fi

# ===== Run =====
cd "${INFER_DIR}"
eval ${CMD}

echo ""
echo "Done! Result saved to: ${SAVE_DIR}/output.mp4"

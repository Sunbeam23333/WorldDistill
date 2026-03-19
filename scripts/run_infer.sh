#!/bin/bash
# ============================================================================
# WorldDistill - Unified Inference Script
#
# Usage:
#   bash scripts/run_infer.sh --model_cls wan2.2_moe --task t2v --prompt "A cat" --gpus 8
#   bash scripts/run_infer.sh --model_cls wan2.2_moe_distill --task t2v --gpus 8
#   bash scripts/run_infer.sh --model_cls lingbot_cam_moe --task i2v --image_path input.jpg --gpus 8
#   bash scripts/run_infer.sh --model_cls worldplay_distill --task game --image_path input.jpg --transformer_model_name 480p_i2v --action_ckpt ./action.safetensors
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
SAVE_PATH=""
CONFIG_JSON=""
CONFIG_WAS_EXPLICIT=0
PROMPT="A white cat wearing sunglasses on a surfboard at summer beach."
NEGATIVE_PROMPT=""
IMAGE_PATH=""
TRANSFORMER_MODEL_NAME=""
ACTION_CKPT=""
ACTION_PATH=""
POSE=""
SEED=42
NUM_GPUS=1

resolve_default_config() {
    local config_dir="$1"
    local raw_model_cls="$2"
    local task="$3"
    local model_key
    model_key="$(printf '%s' "${raw_model_cls}" | tr '[:upper:]' '[:lower:]')"

    case "${model_key}" in
        wan2.1|wan21|wan-2.1|wan2_1|wan2.1-t2v|wan2.1-i2v)
            case "${task}" in
                t2v) echo "${config_dir}/wan/wan_t2v.json" ;;
                i2v) echo "${config_dir}/wan/wan_i2v.json" ;;
                flf2v) echo "${config_dir}/wan/wan_flf2v.json" ;;
                vace) echo "${config_dir}/wan/wan_vace.json" ;;
                *) return 1 ;;
            esac
            ;;
        wan2.2|wan22|wan-2.2|wan2_2)
            case "${task}" in
                t2v) echo "${config_dir}/wan22/wan_ti2v_t2v.json" ;;
                i2v) echo "${config_dir}/wan22/wan_ti2v_i2v.json" ;;
                *) return 1 ;;
            esac
            ;;
        wan2.2_moe|wan22_moe|wan-2.2-moe|wan2_2_moe|wan2.2-a14b|wan22_a14b|wan2.2_moe_diffusers|wan2.2-diffusers|wan22_diffusers)
            case "${task}" in
                t2v) echo "${config_dir}/wan22/wan_moe_t2v.json" ;;
                i2v) echo "${config_dir}/wan22/wan_moe_i2v.json" ;;
                flf2v) echo "${config_dir}/wan22/wan_moe_flf2v.json" ;;
                animate) echo "${config_dir}/wan22/wan_animate.json" ;;
                i2av|t2av) echo "${config_dir}/wan22/wan_moe_i2v_audio.json" ;;
                *) return 1 ;;
            esac
            ;;
        wan2.2_moe_distill|wan2.2_distill|wan2.2-moe-distill|wan22_distill)
            case "${task}" in
                t2v) echo "${config_dir}/wan22/wan_moe_t2v_distill.json" ;;
                i2v) echo "${config_dir}/distill/wan_i2v_distill_4step_cfg.json" ;;
                flf2v) echo "${config_dir}/wan22/wan_distill_moe_flf2v.json" ;;
                *) return 1 ;;
            esac
            ;;
        hunyuan_video_1.5|hyvideo|hy-video|hunyuan-video|hunyuan-video-1.5|hunyuan_video_15|hunyuan_video_1_5)
            case "${task}" in
                t2v) echo "${config_dir}/hunyuan_video_15/hunyuan_video_t2v_480p.json" ;;
                i2v) echo "${config_dir}/hunyuan_video_15/hunyuan_video_i2v_480p.json" ;;
                *) return 1 ;;
            esac
            ;;
        hunyuan_video_1.5_distill|hyvideo_distill|hunyuan-video-distill)
            case "${task}" in
                t2v) echo "${config_dir}/hunyuan_video_15/hunyuan_video_t2v_480p_distill.json" ;;
                *) return 1 ;;
            esac
            ;;
        worldplay_distill|worldplay|hy-worldplay|hy_worldplay)
            case "${task}" in
                i2v|game) echo "${config_dir}/worldplay/worldplay_distill_i2v_480p.json" ;;
                *) return 1 ;;
            esac
            ;;
        worldplay_ar)
            case "${task}" in
                i2v|game) echo "${config_dir}/worldplay/worldplay_ar_i2v_480p.json" ;;
                *) return 1 ;;
            esac
            ;;
        worldplay_bi)
            case "${task}" in
                i2v|game) echo "${config_dir}/worldplay/worldplay_bi_i2v_480p.json" ;;
                *) return 1 ;;
            esac
            ;;
        lingbot_cam_moe|lingbot|lingbot-cam)
            case "${task}" in
                i2v) echo "${config_dir}/lingbot/lingbot_cam_moe_i2v.json" ;;
                *) return 1 ;;
            esac
            ;;
        matrix_game_2|matrix-game-2|matrix_game2)
            case "${task}" in
                game) echo "${config_dir}/matrix_game2/matrix_game2_universal.json" ;;
                *) return 1 ;;
            esac
            ;;
        qwen_image|qwen-image|qwen_image_edit|qwen-image-edit|qwen-image-edit-2509|qwen-image-edit-2511|qwen-image-2512)
            case "${task}" in
                t2i) echo "${config_dir}/qwen_image/qwen_image_t2i_2512.json" ;;
                i2i) echo "${config_dir}/qwen_image/qwen_image_i2i_2511.json" ;;
                *) return 1 ;;
            esac
            ;;
        longcat_image|longcat-image)
            case "${task}" in
                t2i) echo "${config_dir}/longcat_image/longcat_image_t2i.json" ;;
                i2i) echo "${config_dir}/longcat_image/longcat_image_i2i.json" ;;
                *) return 1 ;;
            esac
            ;;
        z_image|z-image|zimage)
            case "${task}" in
                t2i) echo "${config_dir}/z_image/z_image_turbo_t2i.json" ;;
                *) return 1 ;;
            esac
            ;;
        ltx2)
            echo "${config_dir}/ltx2/ltx2.json"
            ;;
        bagel)
            case "${task}" in
                t2i|i2i) echo "${config_dir}/bagel/bagel_t2i.json" ;;
                *) return 1 ;;
            esac
            ;;
        *)
            return 1
            ;;
    esac
}

# ===== Parse Arguments =====
while [[ $# -gt 0 ]]; do
    case $1 in
        --model|--model_cls) MODEL_CLS="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --config|--config_json) CONFIG_JSON="$2"; CONFIG_WAS_EXPLICIT=1; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --negative_prompt) NEGATIVE_PROMPT="$2"; shift 2 ;;
        --image|--image_path) IMAGE_PATH="$2"; shift 2 ;;
        --transformer_model_name) TRANSFORMER_MODEL_NAME="$2"; shift 2 ;;
        --action_ckpt) ACTION_CKPT="$2"; shift 2 ;;
        --action_path) ACTION_PATH="$2"; shift 2 ;;
        --pose) POSE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --gpus) NUM_GPUS="$2"; shift 2 ;;
        --save_dir) SAVE_DIR="$2"; shift 2 ;;
        --save_path|--save_result_path) SAVE_PATH="$2"; shift 2 ;;
        --help)
            echo "Usage: bash run_infer.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model_cls    Model class / alias (preferred, compatible with --model)"
            echo "  --model        Legacy alias of --model_cls"
            echo "  --task         Task type: t2v, i2v, t2i, i2i, game ... (default: t2v)"
            echo "  --model_path   Path to model weights"
            echo "  --config_json  Path to config JSON (preferred, compatible with --config)"
            echo "  --config       Legacy alias of --config_json"
            echo "  --prompt       Text prompt"
            echo "  --negative_prompt  Negative prompt"
            echo "  --image_path   Input image path (preferred, compatible with --image)"
            echo "  --image        Legacy alias of --image_path"
            echo "  --transformer_model_name  Transformer subdir name for HY/WorldPlay models"
            echo "  --action_ckpt  Action checkpoint path for WorldPlay models"
            echo "  --action_path  Camera/action path for LingBot or world models"
            echo "  --pose         Pose string or JSON path for WorldPlay models"
            echo "  --seed         Random seed (default: 42)"
            echo "  --gpus         Number of GPUs (default: 1)"
            echo "  --save_dir     Output directory"
            echo "  --save_path    Output file path (preferred, compatible with --save_result_path)"
            echo "  --save_result_path  Legacy alias of --save_path"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ===== Auto-detect config if not specified =====
if [ -z "${CONFIG_JSON}" ]; then
    CONFIG_DIR="${INFER_DIR}/configs"
    if CONFIG_JSON="$(resolve_default_config "${CONFIG_DIR}" "${MODEL_CLS}" "${TASK}")"; then
        echo ">>> Auto-selected config: ${CONFIG_JSON}"
    else
        CONFIG_JSON=""
        echo "WARNING: No shell auto-detected config for model_cls=${MODEL_CLS}, task=${TASK}. Python entry will still try model-catalog auto-discovery."
    fi
fi

if [ -n "${CONFIG_JSON}" ] && [ ! -f "${CONFIG_JSON}" ]; then
    if [ "${CONFIG_WAS_EXPLICIT}" -eq 1 ]; then
        echo "ERROR: Config JSON not found: ${CONFIG_JSON}"
    else
        echo "ERROR: Auto-detected config JSON not found: ${CONFIG_JSON}"
    fi
    exit 1
fi

# ===== Setup Environment =====
export lightx2v_path="${INFER_DIR}"
export model_path="${MODEL_PATH}"
if [ -f "${INFER_DIR}/scripts/base/base.sh" ]; then
    source "${INFER_DIR}/scripts/base/base.sh"
fi

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))

case "${TASK}" in
    t2i|i2i)
        DEFAULT_OUTPUT_NAME="output.png"
        ;;
    audio)
        DEFAULT_OUTPUT_NAME="output.wav"
        ;;
    *)
        DEFAULT_OUTPUT_NAME="output.mp4"
        ;;
esac

if [ -n "${SAVE_PATH}" ]; then
    FINAL_SAVE_PATH="${SAVE_PATH}"
else
    mkdir -p "${SAVE_DIR}"
    FINAL_SAVE_PATH="${SAVE_DIR}/${DEFAULT_OUTPUT_NAME}"
fi
mkdir -p "$(dirname "${FINAL_SAVE_PATH}")"


echo ">>> Checking inference dependency compatibility..."
(
    cd "${INFER_DIR}"
    python3 lightx2v/utils/env_compat.py --mode infer
)

echo "============================================"
echo "WorldDistill Inference"
echo "  Model CLS: ${MODEL_CLS}"
echo "  Task:      ${TASK}"
echo "  GPUs:      ${NUM_GPUS}"
echo "  Config:    ${CONFIG_JSON:-'(auto from model/runtime)'}"
echo "  Output:    ${FINAL_SAVE_PATH}"
echo "============================================"

# ===== Build Command =====
CMD=(
    torchrun
    "--nproc_per_node=${NUM_GPUS}"
    -m
    lightx2v.infer
    --model_cls
    "${MODEL_CLS}"
    --task
    "${TASK}"
    --model_path
    "${MODEL_PATH}"
    --seed
    "${SEED}"
    --prompt
    "${PROMPT}"
    --save_result_path
    "${FINAL_SAVE_PATH}"
)

if [ -n "${CONFIG_JSON}" ]; then
    CMD+=(--config_json "${CONFIG_JSON}")
fi

if [ -n "${NEGATIVE_PROMPT}" ]; then
    CMD+=(--negative_prompt "${NEGATIVE_PROMPT}")
fi

if [ -n "${IMAGE_PATH}" ]; then
    CMD+=(--image_path "${IMAGE_PATH}")
fi

if [ -n "${TRANSFORMER_MODEL_NAME}" ]; then
    CMD+=(--transformer_model_name "${TRANSFORMER_MODEL_NAME}")
fi

if [ -n "${ACTION_CKPT}" ]; then
    CMD+=(--action_ckpt "${ACTION_CKPT}")
fi

if [ -n "${ACTION_PATH}" ]; then
    CMD+=(--action_path "${ACTION_PATH}")
fi

if [ -n "${POSE}" ]; then
    CMD+=(--pose "${POSE}")
fi

# ===== Run =====
cd "${INFER_DIR}"
"${CMD[@]}"

echo ""
echo "Done! Result saved to: ${FINAL_SAVE_PATH}"

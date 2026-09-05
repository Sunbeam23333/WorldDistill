#!/bin/bash
# ============================================================================
# WorldDistill - Unified Inference Script
#
# Usage:
#   bash scripts/run_infer.sh --model_cls wan2.2_moe --task t2v --prompt "A cat" --gpus 1
#   bash scripts/run_infer.sh --model_cls wan2.2_moe_distill --task t2v --prompt "A cat" --gpus 1
#   bash scripts/run_infer.sh --model_cls lingbot_cam_moe --task i2v --image_path input.jpg --gpus 1
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
PROMPT=""
NEGATIVE_PROMPT=""
IMAGE_PATH=""
LAST_FRAME_PATH=""
AUDIO_PATH=""
IMAGE_STRENGTH="1.0"
SRC_REF_IMAGES=""
SRC_VIDEO=""
SRC_MASK=""
SRC_POSE_PATH=""
SRC_FACE_PATH=""
SRC_BG_PATH=""
SRC_MASK_PATH=""
TRANSFORMER_MODEL_NAME=""
ACTION_CKPT=""
ACTION_PATH=""
POSE=""
SEED=42
NUM_GPUS=1
DRY_RUN=0

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
                i2av|t2av) echo "${config_dir}/wan22/wan_moe_i2v_audio.json" ;;
                *) return 1 ;;
            esac
            ;;
        wan2.1_vace|wan2.1-vace|wan21_vace)
            case "${task}" in
                vace) echo "${config_dir}/wan/wan_vace.json" ;;
                *) return 1 ;;
            esac
            ;;
        wan2.2_moe_vace|wan2.2-moe-vace|wan22_moe_vace)
            case "${task}" in
                vace) echo "${config_dir}/wan22_vace/a800/bf16/wan22_moe_vace.json" ;;
                *) return 1 ;;
            esac
            ;;
        wan2.2_audio|wan2.2-audio|wan22_audio)
            case "${task}" in
                s2v) echo "${config_dir}/wan22/wan_moe_i2v_audio.json" ;;
                *) return 1 ;;
            esac
            ;;
        seko_talk|seko-talk)
            case "${task}" in
                s2v) echo "${config_dir}/seko_talk/shot/stream/s2v.json" ;;
                rs2v) echo "${config_dir}/seko_talk/shot/rs2v/main.json" ;;
                *) return 1 ;;
            esac
            ;;
        wan2.2_animate|wan2.2-animate|wan22_animate)
            case "${task}" in
                animate) echo "${config_dir}/wan22/wan_animate.json" ;;
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
        --last_frame_path) LAST_FRAME_PATH="$2"; shift 2 ;;
        --audio_path) AUDIO_PATH="$2"; shift 2 ;;
        --image_strength) IMAGE_STRENGTH="$2"; shift 2 ;;
        --src_ref_images) SRC_REF_IMAGES="$2"; shift 2 ;;
        --src_video) SRC_VIDEO="$2"; shift 2 ;;
        --src_mask) SRC_MASK="$2"; shift 2 ;;
        --src_pose_path) SRC_POSE_PATH="$2"; shift 2 ;;
        --src_face_path) SRC_FACE_PATH="$2"; shift 2 ;;
        --src_bg_path) SRC_BG_PATH="$2"; shift 2 ;;
        --src_mask_path) SRC_MASK_PATH="$2"; shift 2 ;;
        --transformer_model_name) TRANSFORMER_MODEL_NAME="$2"; shift 2 ;;
        --action_ckpt) ACTION_CKPT="$2"; shift 2 ;;
        --action_path) ACTION_PATH="$2"; shift 2 ;;
        --pose) POSE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --gpus) NUM_GPUS="$2"; shift 2 ;;
        --save_dir) SAVE_DIR="$2"; shift 2 ;;
        --save_path|--save_result_path) SAVE_PATH="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
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
            echo "  --last_frame_path  Last-frame image for flf2v"
            echo "  --audio_path   Input audio for s2v/rs2v"
            echo "  --image_strength  Image conditioning strength for i2av (default: 1.0)"
            echo "  --src_ref_images  Comma-separated VACE reference images"
            echo "  --src_video    VACE source video"
            echo "  --src_mask     VACE source mask"
            echo "  --src_pose_path  Animate pose video"
            echo "  --src_face_path  Animate face video"
            echo "  --src_bg_path    Optional Animate replacement background video"
            echo "  --src_mask_path  Optional Animate replacement mask video"
            echo "  --transformer_model_name  Transformer subdir name for HY/WorldPlay models"
            echo "  --action_ckpt  Action checkpoint path for WorldPlay models"
            echo "  --action_path  Camera/action path for LingBot or world models"
            echo "  --pose         Pose string or JSON path for WorldPlay models"
            echo "  --seed         Random seed (default: 42)"
            echo "  --gpus         Number of GPUs (default: 1)"
            echo "  --save_dir     Output directory"
            echo "  --save_path    Output file path (preferred, compatible with --save_result_path)"
            echo "  --save_result_path  Legacy alias of --save_path"
            echo "  --dry-run      Print the resolved torchrun command without executing it"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --gpus must be a positive integer, got: ${NUM_GPUS}"
    exit 1
fi

# ===== Auto-detect config if not specified =====
if [ -z "${CONFIG_JSON}" ]; then
    CONFIG_DIR="${INFER_DIR}/configs"
    if CONFIG_JSON="$(resolve_default_config "${CONFIG_DIR}" "${MODEL_CLS}" "${TASK}")"; then
        echo ">>> Auto-selected config: ${CONFIG_JSON}"
    else
        echo "ERROR: No public default config for model_cls=${MODEL_CLS}, task=${TASK}. Pass --config_json explicitly."
        exit 1
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

if [ "${TASK}" = "rs2v" ]; then
    case "$(printf '%s' "${MODEL_CLS}" | tr '[:upper:]' '[:lower:]')" in
        seko_talk|seko-talk) ;;
        *)
            echo "ERROR: Stateful RS2V is currently exposed only for --model_cls seko_talk."
            exit 1
            ;;
    esac
fi

case "${TASK}" in
    t2v|t2i|t2av)
        [ -n "${PROMPT}" ] || { echo "ERROR: --task ${TASK} requires --prompt."; exit 1; }
        ;;
    i2v|i2i|i2av|game)
        [ -n "${IMAGE_PATH}" ] || { echo "ERROR: --task ${TASK} requires --image_path."; exit 1; }
        ;;
    flf2v)
        [ -n "${IMAGE_PATH}" ] && [ -n "${LAST_FRAME_PATH}" ] || { echo "ERROR: --task flf2v requires --image_path and --last_frame_path."; exit 1; }
        ;;
    s2v|rs2v)
        [ -n "${IMAGE_PATH}" ] && [ -n "${AUDIO_PATH}" ] || { echo "ERROR: --task ${TASK} requires --image_path and --audio_path."; exit 1; }
        ;;
    vace)
        [ -n "${SRC_REF_IMAGES}" ] || { echo "ERROR: --task vace requires --src_ref_images."; exit 1; }
        ;;
    animate)
        [ -n "${SRC_POSE_PATH}" ] && [ -n "${SRC_FACE_PATH}" ] && [ -n "${SRC_REF_IMAGES}" ] || {
            echo "ERROR: --task animate requires --src_pose_path, --src_face_path, and --src_ref_images."
            exit 1
        }
        ;;
esac

EXPECTED_WORLD_SIZE="$(python3 "${PROJECT_ROOT}/tools/config_world_size.py" "${CONFIG_JSON}")"
if [ "${NUM_GPUS}" -ne "${EXPECTED_WORLD_SIZE}" ]; then
    echo "ERROR: --gpus=${NUM_GPUS} conflicts with config parallel world size ${EXPECTED_WORLD_SIZE}: ${CONFIG_JSON}"
    echo "Use --gpus ${EXPECTED_WORLD_SIZE} or pass a config whose parallel topology matches the launcher."
    exit 1
fi

# ===== Setup Environment =====
export lightx2v_path="${INFER_DIR}"
export model_path="${MODEL_PATH}"
if [ -f "${INFER_DIR}/scripts/base/base.sh" ]; then
    source "${INFER_DIR}/scripts/base/base.sh"
fi

export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS-1)))"

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
    FINAL_SAVE_PATH="${SAVE_DIR}/${DEFAULT_OUTPUT_NAME}"
fi

if [ "${DRY_RUN}" -eq 0 ]; then
    mkdir -p "$(dirname "${FINAL_SAVE_PATH}")"
fi

if [ "${DRY_RUN}" -eq 1 ]; then
    echo ">>> Dry run: dependency compatibility check skipped."
else
    echo ">>> Checking inference dependency compatibility..."
    (
        cd "${INFER_DIR}"
        python3 lightx2v/utils/env_compat.py --mode infer
    )
fi

echo "============================================"
echo "WorldDistill Inference"
echo "  Model CLS: ${MODEL_CLS}"
echo "  Task:      ${TASK}"
echo "  GPUs:      ${NUM_GPUS}"
echo "  Config:    ${CONFIG_JSON:-'(auto from model/runtime)'}"
echo "  Output:    ${FINAL_SAVE_PATH}"
echo "============================================"

# ===== Build Command =====
if [ "${TASK}" = "rs2v" ]; then
    CMD=(
        torchrun
        "--nproc_per_node=${NUM_GPUS}"
        -m
        lightx2v.shot_runner.rs2v_infer
        --config_json
        "${CONFIG_JSON}"
        --model_path
        "${MODEL_PATH}"
        --seed
        "${SEED}"
        --prompt
        "${PROMPT}"
        --save_result_path
        "${FINAL_SAVE_PATH}"
    )
else
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
        --config_json
        "${CONFIG_JSON}"
    )
fi

if [ -n "${NEGATIVE_PROMPT}" ]; then
    CMD+=(--negative_prompt "${NEGATIVE_PROMPT}")
fi

if [ -n "${IMAGE_PATH}" ]; then
    CMD+=(--image_path "${IMAGE_PATH}")
fi

if [ -n "${LAST_FRAME_PATH}" ]; then
    CMD+=(--last_frame_path "${LAST_FRAME_PATH}")
fi

if [ -n "${AUDIO_PATH}" ]; then
    CMD+=(--audio_path "${AUDIO_PATH}")
fi

if [ "${TASK}" != "rs2v" ]; then
    CMD+=(--image_strength "${IMAGE_STRENGTH}")
fi

if [ "${TASK}" != "rs2v" ] && [ -n "${SRC_REF_IMAGES}" ]; then
    CMD+=(--src_ref_images "${SRC_REF_IMAGES}")
fi

if [ "${TASK}" != "rs2v" ] && [ -n "${SRC_VIDEO}" ]; then
    CMD+=(--src_video "${SRC_VIDEO}")
fi

if [ "${TASK}" != "rs2v" ] && [ -n "${SRC_MASK}" ]; then
    CMD+=(--src_mask "${SRC_MASK}")
fi

if [ "${TASK}" != "rs2v" ] && [ -n "${SRC_POSE_PATH}" ]; then
    CMD+=(--src_pose_path "${SRC_POSE_PATH}")
fi

if [ "${TASK}" != "rs2v" ] && [ -n "${SRC_FACE_PATH}" ]; then
    CMD+=(--src_face_path "${SRC_FACE_PATH}")
fi

if [ "${TASK}" != "rs2v" ] && [ -n "${SRC_BG_PATH}" ]; then
    CMD+=(--src_bg_path "${SRC_BG_PATH}")
fi

if [ "${TASK}" != "rs2v" ] && [ -n "${SRC_MASK_PATH}" ]; then
    CMD+=(--src_mask_path "${SRC_MASK_PATH}")
fi

if [ "${TASK}" != "rs2v" ] && [ -n "${TRANSFORMER_MODEL_NAME}" ]; then
    CMD+=(--transformer_model_name "${TRANSFORMER_MODEL_NAME}")
fi

if [ "${TASK}" != "rs2v" ] && [ -n "${ACTION_CKPT}" ]; then
    CMD+=(--action_ckpt "${ACTION_CKPT}")
fi

if [ "${TASK}" != "rs2v" ] && [ -n "${ACTION_PATH}" ]; then
    CMD+=(--action_path "${ACTION_PATH}")
fi

if [ "${TASK}" != "rs2v" ] && [ -n "${POSE}" ]; then
    CMD+=(--pose "${POSE}")
fi

# ===== Run =====
cd "${INFER_DIR}"
if [ "${DRY_RUN}" -eq 1 ]; then
    printf 'Resolved command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

"${CMD[@]}"

echo ""
echo "Done! Result saved to: ${FINAL_SAVE_PATH}"

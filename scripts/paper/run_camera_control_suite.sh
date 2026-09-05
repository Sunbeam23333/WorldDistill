#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PHASE="${1:-all}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT:-${PROJECT_ROOT}/models}/lingbot-world-base-cam}"
INPUT_IMAGE="${INPUT_IMAGE:-${PROJECT_ROOT}/inference/assets/inputs/imgs/img_0.jpg}"
POSE_DIR="${POSE_DIR:-${RESULT_ROOT:-${PROJECT_ROOT}/results}/paper/camera_control/poses}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULT_ROOT:-${PROJECT_ROOT}/results}/paper/camera_control}"
NUM_GPUS="${NUM_GPUS:-8}"
PROMPT="${PROMPT:-A cinematic camera-controlled shot of a cat statue in a courtyard, preserving scene geometry while following the requested camera path.}"

generate_pose_bank() {
    mkdir -p "${POSE_DIR}"
    print_header "Generate camera trajectory bank"
    run_cmd python3 "${PROJECT_ROOT}/tools/generate_camera_poses.py" --num_frames 81 --output_dir "${POSE_DIR}"
}

run_infer_phase() {
    mkdir -p "${OUTPUT_ROOT}/samples"
    print_header "Camera-control I2V inference"
    for pose_name in orbit pan_left zoom_in; do
        run_cmd bash "${PROJECT_ROOT}/scripts/run_infer.sh" \
            --model_cls lingbot_cam_moe \
            --task i2v \
            --model_path "${MODEL_PATH}" \
            --config_json "${PROJECT_ROOT}/inference/configs/lingbot/lingbot_cam_moe_i2v_h20_8gpu.json" \
            --prompt "${PROMPT}" \
            --image_path "${INPUT_IMAGE}" \
            --action_path "${POSE_DIR}/${pose_name}.json" \
            --gpus "${NUM_GPUS}" \
            --save_path "${OUTPUT_ROOT}/samples/${pose_name}.mp4"
    done
}

case "${PHASE}" in
    poses) generate_pose_bank ;;
    infer) run_infer_phase ;;
    all)
        generate_pose_bank
        run_infer_phase
        ;;
    *)
        echo "Usage: $0 [poses|infer|all]" >&2
        exit 1
        ;;
esac

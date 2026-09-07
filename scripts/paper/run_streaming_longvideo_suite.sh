#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PHASE="${1:-all}"
TRAIN_JSON="${TRAIN_JSON:-${PROJECT_ROOT}/configs/data_templates/streaming_long_video.template.json}"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/data/streaming_videos}"
TEACHER_MODEL="${TEACHER_MODEL:-${MODEL_ROOT:-${PROJECT_ROOT}/models}/Wan2.1-SelfForcing}"
NUM_GPUS="${NUM_GPUS:-4}"
INFER_NUM_GPUS="${INFER_NUM_GPUS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-30000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULT_ROOT:-${PROJECT_ROOT}/results}/paper/streaming_longvideo}"
LONG_PROMPT="${LONG_PROMPT:-A continuous handheld journey through a rainy neon city, weaving between alleys, storefront reflections, pedestrians, and passing vehicles over a long uninterrupted take.}"

run_train_phase() {
    mkdir -p "${OUTPUT_ROOT}/train"
    warn_if_template_manifest "${TRAIN_JSON}"
    print_header "Streaming long-video training"
    run_cmd bash "${PROJECT_ROOT}/scripts/run_train.sh" \
        --method stream_distill \
        --model_cls wan2.1 \
        --teacher_model "${TEACHER_MODEL}" \
        --data_json "${TRAIN_JSON}" \
        --video_dir "${VIDEO_DIR}" \
        --output_dir "${OUTPUT_ROOT}/train" \
        --gpus "${NUM_GPUS}" \
        --batch_size 1 \
        --lr 5e-6 \
        --max_train_steps "${MAX_TRAIN_STEPS}" \
        --save_every 2000 \
        --seed 42 \
        --num_frames 129 \
        --resolution 480p \
        --gradient_checkpointing \
        --enable_tf32 \
        --report_to console,tensorboard
}

run_infer_phase() {
    if [[ "${INFER_NUM_GPUS}" != "1" ]]; then
        echo "ERROR: the exported-student Diffusers sampler currently supports INFER_NUM_GPUS=1 only." >&2
        exit 1
    fi
    mkdir -p "${OUTPUT_ROOT}/samples"
    run_cmd python3 "${PROJECT_ROOT}/tools/export_student.py" \
        --base_model "${TEACHER_MODEL}" --checkpoint "${OUTPUT_ROOT}/train" \
        --output_dir "${OUTPUT_ROOT}/student_export"
    print_header "Streaming long-video inference"
    run_cmd python3 "${PROJECT_ROOT}/tools/sample_student.py" \
        --bundle "${OUTPUT_ROOT}/student_export" \
        --num_frames 129 \
        --prompt "${LONG_PROMPT}" \
        --save_path "${OUTPUT_ROOT}/samples/streaming.mp4"
}

case "${PHASE}" in
    train) run_train_phase ;;
    infer) run_infer_phase ;;
    all)
        run_train_phase
        run_infer_phase
        ;;
    *)
        echo "Usage: $0 [train|infer|all]" >&2
        exit 1
        ;;
esac

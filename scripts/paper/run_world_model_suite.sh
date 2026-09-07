#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PHASE="${1:-all}"
TRAIN_JSON="${TRAIN_JSON:-${PROJECT_ROOT}/configs/data_templates/world_model_context_cached.template.json}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_ROOT}/data/cache/world_model}"
TEACHER_MODEL="${TEACHER_MODEL:-${MODEL_ROOT:-${PROJECT_ROOT}/models}/HunyuanVideo-WorldPlay}"
MODEL_ROOT_DIR="${MODEL_ROOT:-${PROJECT_ROOT}/models}"
ACTION_CKPT="${ACTION_CKPT:-${MODEL_ROOT_DIR}/worldplay/action_model.safetensors}"
ACTION_PATH="${ACTION_PATH:-${PROJECT_ROOT}/configs/data_templates/world_model_rollout_actions.template.json}"
INPUT_IMAGE="${INPUT_IMAGE:-${PROJECT_ROOT}/inference/assets/inputs/imgs/img_0.jpg}"
NUM_GPUS="${NUM_GPUS:-8}"
INFER_NUM_GPUS="${INFER_NUM_GPUS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-40000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULT_ROOT:-${PROJECT_ROOT}/results}/paper/world_model}"
PROMPT="${PROMPT:-A first-person driving scene through a colorful arcade city with stable long-horizon world consistency.}"

run_train_phase() {
    mkdir -p "${OUTPUT_ROOT}/train"
    warn_if_template_manifest "${TRAIN_JSON}"
    print_header "World-model context-forcing training"
    run_cmd bash "${PROJECT_ROOT}/scripts/run_train.sh" \
        --method context_forcing \
        --model_cls worldplay_distill \
        --teacher_model "${TEACHER_MODEL}" \
        --data_json "${TRAIN_JSON}" \
        --cache_dir "${CACHE_DIR}" \
        --output_dir "${OUTPUT_ROOT}/train" \
        --config "${PROJECT_ROOT}/configs/distill_presets/context_forcing.json,${PROJECT_ROOT}/configs/distill_presets/world_model_runtime.json" \
        --gpus "${NUM_GPUS}" \
        --batch_size 1 \
        --lr 1e-5 \
        --max_train_steps "${MAX_TRAIN_STEPS}" \
        --save_every 2000 \
        --seed 42 \
        --num_frames 125 \
        --resolution 480p \
        --gradient_checkpointing \
        --enable_tf32 \
        --report_to console,tensorboard,wandb
}

run_infer_phase() {
    if [[ "${INFER_NUM_GPUS}" != "1" ]]; then
        echo "ERROR: the exported-student Diffusers sampler currently supports INFER_NUM_GPUS=1 only." >&2
        exit 1
    fi
    mkdir -p "${OUTPUT_ROOT}/samples"
    warn_if_template_manifest "${ACTION_PATH}"
    # Original WorldPlay runners and Diffusers training weights are not
    # interchangeable. A compatible pipeline must declare its action inputs.
    if [[ -z "${STUDENT_PIPELINE_CONDITIONS:-}" && "${DRY_RUN}" != "1" ]]; then
        echo "ERROR: set STUDENT_PIPELINE_CONDITIONS to the trained pipeline's action/control kwargs JSON; controls are never silently dropped." >&2
        exit 1
    fi
    if [[ -z "${STUDENT_NUM_STEPS:-}" && "${DRY_RUN}" != "1" ]]; then
        echo "ERROR: set STUDENT_NUM_STEPS for the compatible world-model sampler; context-forcing does not define a universal inference timetable." >&2
        exit 1
    fi
    run_cmd python3 "${PROJECT_ROOT}/tools/export_student.py" \
        --base_model "${TEACHER_MODEL}" --checkpoint "${OUTPUT_ROOT}/train" \
        --output_dir "${OUTPUT_ROOT}/student_export" --num_steps "${STUDENT_NUM_STEPS:-4}"
    print_header "World-model distilled rollout"
    run_cmd python3 "${PROJECT_ROOT}/tools/sample_student.py" \
        --bundle "${OUTPUT_ROOT}/student_export" \
        --conditions "${STUDENT_PIPELINE_CONDITIONS:-/path/to/pipeline_conditions.json}" \
        --image_path "${INPUT_IMAGE}" \
        --prompt "${PROMPT}" \
        --save_path "${OUTPUT_ROOT}/samples/worldplay_distill.mp4"

    print_header "World-model AR baseline rollout"
    run_cmd bash "${PROJECT_ROOT}/scripts/run_infer.sh" \
        --model_cls worldplay_ar \
        --task game \
        --model_path "${MODEL_ROOT_DIR}" \
        --transformer_model_name 480p_i2v \
        --action_ckpt "${ACTION_CKPT}" \
        --action_path "${ACTION_PATH}" \
        --image_path "${INPUT_IMAGE}" \
        --prompt "${PROMPT}" \
        --gpus "${INFER_NUM_GPUS}" \
        --save_path "${OUTPUT_ROOT}/samples/worldplay_ar.mp4"
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

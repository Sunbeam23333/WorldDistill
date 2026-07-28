#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PHASE="${1:-all}"
TRAIN_JSON="${TRAIN_JSON:-${PROJECT_ROOT}/configs/data_templates/fewstep_cached_train.template.json}"
VAL_JSON="${VAL_JSON:-${PROJECT_ROOT}/configs/data_templates/fewstep_cached_val.template.json}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_ROOT}/data/cache/fewstep}"
TEACHER_MODEL="${TEACHER_MODEL:-${MODEL_ROOT:-${PROJECT_ROOT}/models}/Wan2.2-T2V-A14B}"
STUDENT_MODEL="${STUDENT_MODEL:-}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-50000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULT_ROOT:-${PROJECT_ROOT}/results}/paper/fewstep_t2v}"
REPORT_TO="${REPORT_TO:-console,tensorboard}"
PROMPT="${PROMPT:-A cinematic aerial shot of a mountain village at sunrise, drifting fog, realistic water reflections, and subtle camera motion.}"

run_train_phase() {
    mkdir -p "${OUTPUT_ROOT}/train"
    warn_if_template_manifest "${TRAIN_JSON}"
    warn_if_template_manifest "${VAL_JSON}"
    print_header "Few-step T2V training"
    run_cmd bash "${PROJECT_ROOT}/scripts/run_train.sh" \
        --method step_distill \
        --model_cls wan2.2_moe \
        --teacher_model "${TEACHER_MODEL}" \
        ${STUDENT_MODEL:+--student_model "${STUDENT_MODEL}"} \
        --data_json "${TRAIN_JSON}" \
        --val_data_json "${VAL_JSON}" \
        --cache_dir "${CACHE_DIR}" \
        --val_cache_dir "${CACHE_DIR}" \
        --output_dir "${OUTPUT_ROOT}/train" \
        --gpus "${NUM_GPUS}" \
        --steps 4 \
        --batch_size 1 \
        --lr 1e-5 \
        --max_train_steps "${MAX_TRAIN_STEPS}" \
        --eval_every 2000 \
        --eval_batches 4 \
        --save_every 2000 \
        --seed 42 \
        --num_frames 81 \
        --resolution 720p \
        --report_to "${REPORT_TO}" \
        --gradient_checkpointing \
        --enable_tf32 \
        --enable_torch_compile \
        --torch_compile_scope student \
        --torch_compile_mode reduce-overhead
}

run_teacher_infer_phase() {
    mkdir -p "${OUTPUT_ROOT}/teacher_samples"
    print_header "Few-step T2V teacher inference"
    run_cmd bash "${PROJECT_ROOT}/scripts/run_infer.sh" \
        --model_cls wan2.2_moe \
        --task t2v \
        --model_path "${TEACHER_MODEL}" \
        --prompt "${PROMPT}" \
        --gpus "${NUM_GPUS}" \
        --save_path "${OUTPUT_ROOT}/teacher_samples/teacher.mp4"
}

run_student_infer_phase() {
    mkdir -p "${OUTPUT_ROOT}/student_samples"
    print_header "Few-step T2V student inference"
    run_cmd bash "${PROJECT_ROOT}/scripts/run_infer.sh" \
        --model_cls wan2.2_moe_distill \
        --task t2v \
        --model_path "${MODEL_ROOT:-${PROJECT_ROOT}/models}" \
        --prompt "${PROMPT}" \
        --gpus "${NUM_GPUS}" \
        --save_path "${OUTPUT_ROOT}/student_samples/student.mp4"
}

case "${PHASE}" in
    train) run_train_phase ;;
    infer_teacher) run_teacher_infer_phase ;;
    infer_student) run_student_infer_phase ;;
    infer)
        run_teacher_infer_phase
        run_student_infer_phase
        ;;
    all)
        run_train_phase
        run_teacher_infer_phase
        run_student_infer_phase
        ;;
    *)
        echo "Usage: $0 [train|infer_teacher|infer_student|infer|all]" >&2
        exit 1
        ;;
esac

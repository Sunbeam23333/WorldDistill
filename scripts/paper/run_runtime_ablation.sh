#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TRAIN_JSON="${TRAIN_JSON:-${PROJECT_ROOT}/configs/data_templates/runtime_microbench_cached.template.json}"
VAL_JSON="${VAL_JSON:-${PROJECT_ROOT}/configs/data_templates/fewstep_cached_val.template.json}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_ROOT}/data/cache/runtime_microbench}"
TEACHER_MODEL="${TEACHER_MODEL:-${MODEL_ROOT:-${PROJECT_ROOT}/models}/Wan2.2-T2V-A14B}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULT_ROOT:-${PROJECT_ROOT}/results}/paper/runtime_ablation}"
CONFIG_ROOT="${OUTPUT_ROOT}/derived_configs"
BASE_PRESET="${PROJECT_ROOT}/configs/distill_presets/step_distill_4step.json"

mkdir -p "${OUTPUT_ROOT}" "${CONFIG_ROOT}"
warn_if_template_manifest "${TRAIN_JSON}"
warn_if_template_manifest "${VAL_JSON}"

print_header "Generate runtime ablation preset variants"
run_cmd python3 - "${BASE_PRESET}" "${CONFIG_ROOT}" <<'PY'
import json
import os
import sys

base_path, out_dir = sys.argv[1], sys.argv[2]
with open(base_path, 'r') as f:
    base = json.load(f)
variants = {
    'eager': {
        'enable_runtime': False,
        'runtime_name': 'noop',
        'runtime_teacher_cache_mode': 'disabled',
        'runtime_enable_dpp': False,
        'enable_fused_supervision_kernel': False,
        'fused_supervision_backend': 'none',
    },
    'cache_only': {
        'enable_runtime': True,
        'runtime_name': 'teacher_student',
        'runtime_teacher_cache_mode': 'teacher_output',
        'runtime_enable_dpp': False,
        'enable_fused_supervision_kernel': False,
        'fused_supervision_backend': 'none',
    },
    'cache_async': {
        'enable_runtime': True,
        'runtime_name': 'teacher_student',
        'runtime_teacher_cache_mode': 'teacher_output',
        'runtime_enable_dpp': True,
        'enable_fused_supervision_kernel': False,
        'fused_supervision_backend': 'none',
    },
    'cache_async_fused': {
        'enable_runtime': True,
        'runtime_name': 'teacher_student',
        'runtime_teacher_cache_mode': 'teacher_output',
        'runtime_enable_dpp': True,
        'enable_fused_supervision_kernel': True,
        'fused_supervision_backend': 'triton',
    },
    'cache_async_fused_compile': {
        'enable_runtime': True,
        'runtime_name': 'teacher_student',
        'runtime_teacher_cache_mode': 'teacher_output',
        'runtime_enable_dpp': True,
        'enable_fused_supervision_kernel': True,
        'fused_supervision_backend': 'triton',
    },
}
for name, overrides in variants.items():
    merged = dict(base)
    merged.update(overrides)
    with open(os.path.join(out_dir, f'{name}.json'), 'w') as f:
        json.dump(merged, f, indent=2)
PY

for variant in eager cache_only cache_async cache_async_fused cache_async_fused_compile; do
    extra_args=()
    if [[ "${variant}" == "cache_async_fused_compile" ]]; then
        extra_args=(
            --enable_tf32
            --enable_torch_compile
            --torch_compile_scope student
            --torch_compile_mode reduce-overhead
        )
    fi
    print_header "Runtime ablation variant: ${variant}"
    run_cmd bash "${PROJECT_ROOT}/scripts/run_train.sh" \
        --method step_distill \
        --model_cls wan2.2_moe \
        --teacher_model "${TEACHER_MODEL}" \
        --data_json "${TRAIN_JSON}" \
        --val_data_json "${VAL_JSON}" \
        --cache_dir "${CACHE_DIR}" \
        --val_cache_dir "${CACHE_DIR}" \
        --config "${CONFIG_ROOT}/${variant}.json" \
        --output_dir "${OUTPUT_ROOT}/${variant}" \
        --gpus "${NUM_GPUS}" \
        --steps 4 \
        --batch_size 1 \
        --lr 1e-5 \
        --max_train_steps "${MAX_TRAIN_STEPS}" \
        --eval_every 500 \
        --eval_batches 2 \
        --save_every 500 \
        --seed 42 \
        --num_frames 81 \
        --resolution 720p \
        --gradient_checkpointing \
        "${extra_args[@]}"
done

#!/bin/bash
# ============================================================================
# Wan2.2 MoE T2V Benchmark (Multi-GPU)
# Model: Wan2.2-T2V-A14B (MoE, original or Diffusers format)
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"
INFER_DIR="${PROJECT_ROOT}/inference"

# ===== Configurable Paths =====
MODEL_FORMAT="${1:-original}"  # "original" or "diffusers"
if [ "${MODEL_FORMAT}" = "diffusers" ]; then
    MODEL_PATH="${MODEL_ROOT:-${PROJECT_ROOT}/models}/Wan2.2-T2V-A14B-Diffusers"
    CONFIG_JSON="${INFER_DIR}/configs/wan22/wan_moe_t2v_h20_8gpu_diffusers.json"
else
    MODEL_PATH="${MODEL_ROOT:-${PROJECT_ROOT}/models}/Wan2.2-T2V-A14B"
    CONFIG_JSON="${INFER_DIR}/configs/wan22/wan_moe_t2v_h20_8gpu.json"
fi
SAVE_DIR="${RESULT_ROOT:-${PROJECT_ROOT}/results}/wan22_${MODEL_FORMAT}_t2v"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

if [ -f "${INFER_DIR}/scripts/base/base.sh" ]; then
    source "${INFER_DIR}/scripts/base/base.sh"
fi
mkdir -p "${SAVE_DIR}"

declare -a PROMPTS=(
    "A white cat wearing sunglasses sits on a surfboard at a summer beach, waves gently rolling."
    "A young woman in a red dress walks through a field of sunflowers, golden light illuminates the scene."
    "A futuristic cityscape at sunset with flying cars weaving between gleaming skyscrapers."
    "An astronaut floating in space, Earth visible in the background, stars twinkle in the darkness."
    "A traditional Chinese garden with koi pond, cherry blossoms falling gently onto the water surface."
)

echo "============================================"
echo "Wan2.2 MoE T2V Benchmark - ${#PROMPTS[@]} cases"
echo "Model: Wan2.2-T2V-A14B (${MODEL_FORMAT})"
echo "Config: 720p, 81 frames, 40 steps"
echo "Parallel: cfg_p=2 x seq_p=4 = 8 GPUs"
echo "============================================"

TOTAL_START=$(date +%s)

cd "${INFER_DIR}"
for i in $(seq 0 $((${#PROMPTS[@]}-1))); do
    CASE_START=$(date +%s)
    echo ">>> Case $((i+1))/${#PROMPTS[@]} - Seed: $((42+i))"

    torchrun --nproc_per_node=${NUM_GPUS} -m lightx2v.infer \
        --model_cls wan2.2_moe \
        --task t2v \
        --model_path "${MODEL_PATH}" \
        --config_json "${CONFIG_JSON}" \
        --seed $((42+i)) \
        --prompt "${PROMPTS[$i]}" \
        --save_result_path "${SAVE_DIR}/case_$((i+1)).mp4"

    echo ">>> Case $((i+1)) done in $(( $(date +%s) - CASE_START ))s"
done

TOTAL_TIME=$(( $(date +%s) - TOTAL_START ))
echo "Benchmark done! Total: ${TOTAL_TIME}s, Avg: $((TOTAL_TIME / ${#PROMPTS[@]}))s"

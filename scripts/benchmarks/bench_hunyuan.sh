#!/bin/bash
# ============================================================================
# HunyuanVideo 1.5 Benchmark
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"
INFER_DIR="${PROJECT_ROOT}/inference"

MODEL_PATH="${MODEL_ROOT:-${PROJECT_ROOT}/models}/HunyuanVideo"
SAVE_DIR="${RESULT_ROOT:-${PROJECT_ROOT}/results}/hunyuan_video_t2v"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

if [ -f "${INFER_DIR}/scripts/base/base.sh" ]; then
    source "${INFER_DIR}/scripts/base/base.sh"
fi
mkdir -p "${SAVE_DIR}"

declare -a PROMPTS=(
    "A white cat wearing sunglasses sits on a surfboard at a summer beach."
    "A young woman in a red dress walks through a field of sunflowers."
    "A futuristic cityscape at sunset with flying cars between skyscrapers."
    "An astronaut floating in space with Earth visible in the background."
    "A traditional Chinese garden with koi pond and falling cherry blossoms."
)

echo "============================================"
echo "HunyuanVideo 1.5 T2V Benchmark - ${#PROMPTS[@]} cases"
echo "============================================"

TOTAL_START=$(date +%s)

cd "${INFER_DIR}"
for i in $(seq 0 $((${#PROMPTS[@]}-1))); do
    echo ">>> Case $((i+1))/${#PROMPTS[@]} - Seed: $((42+i))"

    torchrun --nproc_per_node=${NUM_GPUS} -m lightx2v.infer \
        --model_cls hunyuan_video_1.5 \
        --task t2v \
        --model_path "${MODEL_PATH}" \
        --seed $((42+i)) \
        --prompt "${PROMPTS[$i]}" \
        --save_result_path "${SAVE_DIR}/case_$((i+1)).mp4"
done

TOTAL_TIME=$(( $(date +%s) - TOTAL_START ))
echo "Benchmark done! Total: ${TOTAL_TIME}s, Avg: $((TOTAL_TIME / ${#PROMPTS[@]}))s"

#!/bin/bash
# ============================================================================
# LingBot Camera Control I2V Benchmark (Multi-GPU)
# Model: lingbot-world-base-cam (MoE, Wan2.1 + Plucker Camera Control)
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"
INFER_DIR="${PROJECT_ROOT}/inference"

# ===== Configurable Paths =====
MODEL_PATH="${MODEL_ROOT:-${PROJECT_ROOT}/models}/lingbot-world-base-cam"
SAVE_DIR="${RESULT_ROOT:-${PROJECT_ROOT}/results}/lingbot_cam_i2v"
INPUT_IMAGES_DIR="${INFER_DIR}/assets/inputs/imgs"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

if [ -f "${INFER_DIR}/scripts/base/base.sh" ]; then
    source "${INFER_DIR}/scripts/base/base.sh"
fi
mkdir -p "${SAVE_DIR}"

declare -a PROMPTS=(
    "A white cat wearing sunglasses sits on a surfboard at a summer beach, waves gently rolling in the background."
    "A young woman in a red dress walks through a field of sunflowers, her hair flowing in the wind."
    "A futuristic cityscape at sunset with flying cars weaving between gleaming skyscrapers."
    "An astronaut floating in space, Earth visible in the background with the sun creating a brilliant halo."
    "A traditional Chinese garden with a koi pond, cherry blossoms falling gently onto the water surface."
)

declare -a IMAGES=(
    "${INPUT_IMAGES_DIR}/img_0.jpg"
    "${INPUT_IMAGES_DIR}/img_1.jpg"
    "${INPUT_IMAGES_DIR}/img_2.jpg"
    "${INPUT_IMAGES_DIR}/girl.png"
    "${INPUT_IMAGES_DIR}/woman.jpeg"
)

echo "============================================"
echo "LingBot Cam I2V Benchmark - ${#PROMPTS[@]} cases"
echo "Model: lingbot-world-base-cam (MoE)"
echo "Config: 720p, 81 frames, 40 steps"
echo "============================================"

TOTAL_START=$(date +%s)

cd "${INFER_DIR}"
for i in $(seq 0 $((${#PROMPTS[@]}-1))); do
    CASE_START=$(date +%s)
    echo ">>> Case $((i+1))/${#PROMPTS[@]} - Seed: $((42+i))"

    torchrun --nproc_per_node=${NUM_GPUS} -m lightx2v.infer \
        --model_cls lingbot_cam_moe \
        --task i2v \
        --model_path "${MODEL_PATH}" \
        --config_json "${INFER_DIR}/configs/lingbot/lingbot_cam_moe_i2v_h20_8gpu.json" \
        --seed $((42+i)) \
        --prompt "${PROMPTS[$i]}" \
        --image_path "${IMAGES[$i]}" \
        --save_result_path "${SAVE_DIR}/case_$((i+1)).mp4"

    echo ">>> Case $((i+1)) done in $(( $(date +%s) - CASE_START ))s"
done

TOTAL_TIME=$(( $(date +%s) - TOTAL_START ))
echo "Benchmark done! Total: ${TOTAL_TIME}s, Avg: $((TOTAL_TIME / ${#PROMPTS[@]}))s"

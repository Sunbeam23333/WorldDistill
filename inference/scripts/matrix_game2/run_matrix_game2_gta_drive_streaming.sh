#!/bin/bash

# Configure public paths through the environment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
model_path="${MODEL_PATH:?Set MODEL_PATH to the Matrix-Game-2.0 checkpoint directory}"
input_image="${INPUT_IMAGE:?Set INPUT_IMAGE to the initial game frame}"

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls wan2.1_sf_mtxg2 \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/matrix_game2/matrix_game2_gta_drive_streaming.json \
--prompt '' \
--image_path "${input_image}" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_matrix_game2_gta_drive_streaming.mp4 \
--seed 42

#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export lightx2v_path="${LIGHTX2V_PATH:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export model_path="${MODEL_PATH:?Set MODEL_PATH to the LongCat-Image-Edit checkpoint}"
input_image="${INPUT_IMAGE:?Set INPUT_IMAGE to the source image}"

export CUDA_VISIBLE_DEVICES=0,1

# Source base configuration if exists
if [ -f "${lightx2v_path}/scripts/base/base.sh" ]; then
    source ${lightx2v_path}/scripts/base/base.sh
fi

torchrun --nproc_per_node=2 -m lightx2v.infer \
--model_cls longcat_image \
--task i2i \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/longcat_image/longcat_image_i2i_cfg_parallel.json \
--prompt "将猫变成狗" \
--negative_prompt "" \
--image_path "${input_image}" \
--save_result_path ${lightx2v_path}/save_results/longcat_image_i2i_cfg_parallel.png \
--seed 43

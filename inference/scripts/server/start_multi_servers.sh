#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIGHTX2V_PATH="${LIGHTX2V_PATH:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
MODEL_CLS="${MODEL_CLS:-wan2.2_moe}"
TASK="${TASK:-t2v}"
NUM_GPUS="${NUM_GPUS:-1}"
PORT="${PORT:-8000}"

: "${MODEL_PATH:?Set MODEL_PATH to the checkpoint directory}"
: "${CONFIG_JSON:?Set CONFIG_JSON to a public config whose mesh matches NUM_GPUS}"

# CUDA_VISIBLE_DEVICES is intentionally caller-owned so the visible-device list
# can be matched to NUM_GPUS on each host.
source "${LIGHTX2V_PATH}/scripts/base/base.sh"

torchrun --nproc_per_node "${NUM_GPUS}" -m lightx2v.server \
    --model_cls "${MODEL_CLS}" \
    --task "${TASK}" \
    --model_path "${MODEL_PATH}" \
    --config_json "${CONFIG_JSON}" \
    --port "${PORT}"

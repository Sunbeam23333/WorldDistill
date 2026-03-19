#!/bin/bash
# ============================================================================
# WorldDistill - Unified Distillation Training Script
#
# Usage:
#   # Basic DDP training
#   bash scripts/run_train.sh --method step_distill --teacher_model ./models/Wan2.2 --model_cls wan2.2_moe --data_json ./data/train.json --gpus 8
#
#   # FSDP for large models
#   bash scripts/run_train.sh --method step_distill --teacher_model ./models/Wan2.2 --model_cls wan2.2_moe --data_json ./data/train.json --gpus 8 --parallel fsdp
#
#   # DeepSpeed ZeRO-2
#   bash scripts/run_train.sh --method step_distill --teacher_model ./models/Wan2.2 --model_cls wan2.2_moe --data_json ./data/train.json --gpus 8 --parallel deepspeed --ds_stage 2
#
#   # With Sequence Parallelism
#   bash scripts/run_train.sh --method stream_distill --teacher_model ./models/SkyReels-V2 --model_cls skyreels_v2 --data_json ./data/train.json --gpus 8 --sp_size 2
#
#   # With TensorBoard + W&B logging
#   bash scripts/run_train.sh --method context_forcing --teacher_model ./models/HY-WorldPlay --model_cls worldplay_distill --data_json ./data/train.json --report_to console,tensorboard,wandb
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# ===== Default Configuration =====
METHOD="step_distill"
MODEL_CLS="wan2.1"
TEACHER_MODEL=""
STUDENT_MODEL=""
DATA_JSON=""
VAL_DATA_JSON=""
EVAL_EVERY=""
EVAL_BATCHES=""
OUTPUT_DIR="${RESULT_ROOT:-${PROJECT_ROOT}/results/training}"
CONFIG=""
NUM_GPUS=8
NUM_STEPS=4
BATCH_SIZE=1
LR=1e-5
MAX_TRAIN_STEPS=10000
SAVE_EVERY=1000
PARALLEL_MODE="ddp"
SP_SIZE=1
DS_STAGE=2
FSDP_STRATEGY="full"
GRADIENT_CHECKPOINTING=""
CPU_OFFLOAD=""
REPORT_TO="console"
TENSORBOARD_LOG_DIR=""
WANDB_PROJECT="worlddistill"
WANDB_ENTITY=""
WANDB_RUN_NAME=""
WANDB_TAGS=""

resolve_default_config() {
    case "${METHOD}" in
        step_distill)
            local candidate="${PROJECT_ROOT}/configs/distill_presets/step_distill_${NUM_STEPS}step.json"
            if [ -f "${candidate}" ]; then
                echo "${candidate}"
                return 0
            fi
            ;;
        stream_distill|consistency_distill)
            local candidate="${PROJECT_ROOT}/configs/distill_presets/${METHOD}.json"
            if [ -f "${candidate}" ]; then
                echo "${candidate}"
                return 0
            fi
            ;;
        context_forcing)
            echo "${PROJECT_ROOT}/configs/distill_presets/context_forcing.json,${PROJECT_ROOT}/configs/distill_presets/world_model_runtime.json"
            return 0
            ;;
    esac
    return 1
}

validate_config_spec() {
    local config_spec="$1"
    IFS=',' read -r -a config_paths <<< "${config_spec}"
    for config_path in "${config_paths[@]}"; do
        if [ ! -f "${config_path}" ]; then
            echo "ERROR: Distill preset not found: ${config_path}"
            exit 1
        fi
    done
}

# ===== Parse Arguments =====
while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD="$2"; shift 2 ;;
        --model_cls) MODEL_CLS="$2"; shift 2 ;;
        --teacher_model) TEACHER_MODEL="$2"; shift 2 ;;
        --student_model) STUDENT_MODEL="$2"; shift 2 ;;
        --data_json) DATA_JSON="$2"; shift 2 ;;
        --val_data_json) VAL_DATA_JSON="$2"; shift 2 ;;
        --eval_every) EVAL_EVERY="$2"; shift 2 ;;
        --eval_batches) EVAL_BATCHES="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --gpus) NUM_GPUS="$2"; shift 2 ;;
        --steps) NUM_STEPS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --max_train_steps) MAX_TRAIN_STEPS="$2"; shift 2 ;;
        --save_every) SAVE_EVERY="$2"; shift 2 ;;
        --parallel) PARALLEL_MODE="$2"; shift 2 ;;
        --sp_size) SP_SIZE="$2"; shift 2 ;;
        --ds_stage) DS_STAGE="$2"; shift 2 ;;
        --fsdp_strategy) FSDP_STRATEGY="$2"; shift 2 ;;
        --report_to) REPORT_TO="$2"; shift 2 ;;
        --tensorboard_log_dir) TENSORBOARD_LOG_DIR="$2"; shift 2 ;;
        --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
        --wandb_entity) WANDB_ENTITY="$2"; shift 2 ;;
        --wandb_run_name) WANDB_RUN_NAME="$2"; shift 2 ;;
        --wandb_tags) WANDB_TAGS="$2"; shift 2 ;;
        --gradient_checkpointing) GRADIENT_CHECKPOINTING="--gradient_checkpointing"; shift ;;
        --cpu_offload) CPU_OFFLOAD="--cpu_offload"; shift ;;
        --help)
            echo "Usage: bash run_train.sh [OPTIONS]"
            echo ""
            echo "Distillation Methods:"
            echo "  step_distill          Fixed N-step distillation (default)"
            echo "  stream_distill        Diffusion Forcing stream distillation"
            echo "  progressive_distill   Progressive halving distillation"
            echo "  consistency_distill   Trajectory Consistency Distillation"
            echo "  context_forcing       Memory-aware context forcing distillation"
            echo "  adversarial_distill   Adversarial Diffusion Distillation (ADD/LADD)"
            echo "  dmd_distill           Distribution Matching Distillation (DMD/DMD2)"
            echo ""
            echo "Parallel Modes:"
            echo "  ddp                   DistributedDataParallel (default, <14B params)"
            echo "  fsdp                  FullyShardedDataParallel (large models)"
            echo "  deepspeed             DeepSpeed ZeRO (very large models, CPU offload)"
            echo ""
            echo "Options:"
            echo "  --method              Distillation method (default: step_distill)"
            echo "  --model_cls           Model class / alias (default: wan2.1)"
            echo "  --teacher_model       Path to teacher model weights"
            echo "  --student_model       Path to student model (optional, init from teacher)"
            echo "  --data_json           Training data json"
            echo "  --val_data_json       Validation data json (optional)"
            echo "  --eval_every          Evaluate every N steps (optional)"
            echo "  --eval_batches        Validation batches per eval (optional)"
            echo "  --output_dir          Output directory for checkpoints and logs"
            echo "  --config              Path to distill preset config"
            echo "  --gpus                Number of GPUs (default: 8)"
            echo "  --steps               Target inference steps (default: 4)"
            echo "  --batch_size          Batch size per GPU (default: 1)"
            echo "  --lr                  Learning rate (default: 1e-5)"
            echo "  --max_train_steps     Max training steps (default: 10000)"
            echo "  --save_every          Save checkpoint every N steps (default: 1000)"
            echo "  --parallel            Parallel mode: ddp | fsdp | deepspeed (default: ddp)"
            echo "  --sp_size             Sequence parallel group size (default: 1 = disabled)"
            echo "  --ds_stage            DeepSpeed ZeRO stage: 1 | 2 | 3 (default: 2)"
            echo "  --fsdp_strategy       FSDP strategy: full | hybrid (default: full)"
            echo "  --report_to           console,tensorboard,wandb,all,none (default: console)"
            echo "  --tensorboard_log_dir Custom TensorBoard log directory (optional)"
            echo "  --wandb_project       W&B project name (default: worlddistill)"
            echo "  --wandb_entity        W&B entity/team (optional)"
            echo "  --wandb_run_name      W&B run name (optional)"
            echo "  --wandb_tags          Comma-separated W&B tags (optional)"
            echo "  --gradient_checkpointing  Enable gradient checkpointing"
            echo "  --cpu_offload         Enable CPU offloading (FSDP/DeepSpeed)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "${TEACHER_MODEL}" ]; then
    echo "ERROR: --teacher_model is required"
    exit 1
fi

if [ -z "${DATA_JSON}" ]; then
    echo "ERROR: --data_json is required"
    exit 1
fi

# Auto-detect config
if [ -z "${CONFIG}" ]; then
    if ! CONFIG="$(resolve_default_config)"; then
        echo "ERROR: No default preset is available for method '${METHOD}'. Please pass --config explicitly."
        exit 1
    fi
fi
validate_config_spec "${CONFIG}"

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
mkdir -p "${OUTPUT_DIR}"

echo ">>> Checking dependency compatibility..."
python3 -m training.env_compat --mode train

echo "============================================"
echo "WorldDistill Training"
echo "  Method:      ${METHOD}"
echo "  Model CLS:   ${MODEL_CLS}"
echo "  Teacher:     ${TEACHER_MODEL}"
echo "  Student:     ${STUDENT_MODEL:-'(init from teacher)'}"
echo "  Steps:       ${NUM_STEPS}"
echo "  GPUs:        ${NUM_GPUS}"
echo "  LR:          ${LR}"
echo "  Parallel:    ${PARALLEL_MODE}"
echo "  Report To:   ${REPORT_TO}"
if [ -n "${TENSORBOARD_LOG_DIR}" ]; then
    echo "  TensorBoard: ${TENSORBOARD_LOG_DIR}"
fi
if [[ "${REPORT_TO}" == *"wandb"* || "${REPORT_TO}" == "all" ]]; then
    echo "  W&B Project: ${WANDB_PROJECT}"
fi
if [ "${PARALLEL_MODE}" = "deepspeed" ]; then
    echo "  DS Stage:    ${DS_STAGE}"
fi
if [ "${PARALLEL_MODE}" = "fsdp" ]; then
    echo "  FSDP:        ${FSDP_STRATEGY}"
fi
if [ "${SP_SIZE}" -gt 1 ]; then
    echo "  SP Size:     ${SP_SIZE}"
fi
echo "  Config:      ${CONFIG}"
echo "============================================"

torchrun --nproc_per_node=${NUM_GPUS} \
    -m training.train_distill \
    --method ${METHOD} \
    --teacher_model_path ${TEACHER_MODEL} \
    --model_cls ${MODEL_CLS} \
    ${STUDENT_MODEL:+--student_model_path ${STUDENT_MODEL}} \
    ${DATA_JSON:+--data_json ${DATA_JSON}} \
    ${VAL_DATA_JSON:+--val_data_json ${VAL_DATA_JSON}} \
    ${EVAL_EVERY:+--eval_every ${EVAL_EVERY}} \
    ${EVAL_BATCHES:+--eval_batches ${EVAL_BATCHES}} \
    --output_dir ${OUTPUT_DIR} \
    --num_inference_steps ${NUM_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --max_train_steps ${MAX_TRAIN_STEPS} \
    --save_every ${SAVE_EVERY} \
    --parallel_mode ${PARALLEL_MODE} \
    --sp_size ${SP_SIZE} \
    --deepspeed_stage ${DS_STAGE} \
    --fsdp_shard_strategy ${FSDP_STRATEGY} \
    --report_to ${REPORT_TO} \
    ${TENSORBOARD_LOG_DIR:+--tensorboard_log_dir ${TENSORBOARD_LOG_DIR}} \
    --wandb_project ${WANDB_PROJECT} \
    ${WANDB_ENTITY:+--wandb_entity ${WANDB_ENTITY}} \
    ${WANDB_RUN_NAME:+--wandb_run_name ${WANDB_RUN_NAME}} \
    ${WANDB_TAGS:+--wandb_tags ${WANDB_TAGS}} \
    ${GRADIENT_CHECKPOINTING} \
    ${CPU_OFFLOAD} \
    ${CONFIG:+--config ${CONFIG}}

echo ""
echo "Training complete! Checkpoints and logs saved to: ${OUTPUT_DIR}"

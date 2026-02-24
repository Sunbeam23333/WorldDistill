#!/bin/bash
# ============================================================================
# WorldDistill - Unified Distillation Training Script
#
# Usage:
#   # Basic DDP training
#   bash scripts/run_train.sh --method step_distill --teacher_model ./models/Wan2.2 --gpus 8
#
#   # FSDP for large models
#   bash scripts/run_train.sh --method step_distill --teacher_model ./models/Wan2.2 --gpus 8 --parallel fsdp
#
#   # DeepSpeed ZeRO-2
#   bash scripts/run_train.sh --method step_distill --teacher_model ./models/Wan2.2 --gpus 8 --parallel deepspeed --ds_stage 2
#
#   # With Sequence Parallelism
#   bash scripts/run_train.sh --method stream_distill --teacher_model ./models/SkyReels-V2 --gpus 8 --sp_size 2
#
#   # Adversarial distillation (ADD)
#   bash scripts/run_train.sh --method adversarial_distill --teacher_model ./models/Wan2.2 --gpus 8
#
#   # Distribution Matching Distillation (DMD)
#   bash scripts/run_train.sh --method dmd_distill --teacher_model ./models/Wan2.2 --gpus 8
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# ===== Default Configuration =====
METHOD="step_distill"
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

# ===== Parse Arguments =====
while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD="$2"; shift 2 ;;
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
            echo "  --teacher_model       Path to teacher model weights"
            echo "  --student_model       Path to student model (optional, init from teacher)"
            echo "  --data_json           Training data json"
            echo "  --val_data_json       Validation data json (optional)"
            echo "  --eval_every          Evaluate every N steps (optional)"
            echo "  --eval_batches        Validation batches per eval (optional)"
            echo "  --output_dir          Output directory for checkpoints"
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

# Auto-detect config
if [ -z "${CONFIG}" ]; then
    CONFIG="${PROJECT_ROOT}/configs/distill_presets/${METHOD}.json"
fi

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "WorldDistill Training"
echo "  Method:     ${METHOD}"
echo "  Teacher:    ${TEACHER_MODEL}"
echo "  Student:    ${STUDENT_MODEL:-'(init from teacher)'}"
echo "  Steps:      ${NUM_STEPS}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  LR:         ${LR}"
echo "  Parallel:   ${PARALLEL_MODE}"
if [ "${PARALLEL_MODE}" = "deepspeed" ]; then
    echo "  DS Stage:   ${DS_STAGE}"
fi
if [ "${PARALLEL_MODE}" = "fsdp" ]; then
    echo "  FSDP:       ${FSDP_STRATEGY}"
fi
if [ "${SP_SIZE}" -gt 1 ]; then
    echo "  SP Size:    ${SP_SIZE}"
fi
echo "  Config:     ${CONFIG}"
echo "============================================"

torchrun --nproc_per_node=${NUM_GPUS} \
    -m training.train_distill \
    --method ${METHOD} \
    --teacher_model_path ${TEACHER_MODEL} \
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
    ${GRADIENT_CHECKPOINTING} \
    ${CPU_OFFLOAD} \
    ${CONFIG:+--config ${CONFIG}}

echo ""
echo "Training complete! Checkpoints saved to: ${OUTPUT_DIR}"

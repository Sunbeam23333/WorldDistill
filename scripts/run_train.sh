#!/bin/bash
# ============================================================================
# WorldDistill - Unified Distillation Training Script
#
# Usage:
#   # Basic DDP training
#   bash scripts/run_train.sh --method step_distill --teacher_model ./models/Wan2.2 --model_cls wan2.2_moe --data_json ./data/train.json --gpus 8
#
#   # FSDP post-load parameter sharding (the model must fit during construction)
#   bash scripts/run_train.sh --method step_distill --teacher_model ./models/Wan2.2 --model_cls wan2.2_moe --data_json ./data/train.json --gpus 8 --parallel fsdp
#
#   # DeepSpeed ZeRO-2
#   bash scripts/run_train.sh --method step_distill --teacher_model ./models/Wan2.2 --model_cls wan2.2_moe --data_json ./data/train.json --gpus 8 --parallel deepspeed --ds_stage 2
#
#   # With TensorBoard + W&B logging
#   bash scripts/run_train.sh --method context_forcing --teacher_model ./models/HY-WorldPlay --model_cls worldplay_distill --data_json ./data/train.json --report_to console,tensorboard,wandb
# ============================================================================
set -euo pipefail

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
NUM_GPUS="${NPROC_PER_NODE:-8}"
NNODES_DECLARED=0
NODE_RANK_DECLARED=0
if [[ -n "${NNODES+x}" ]]; then NNODES_DECLARED=1; fi
if [[ -n "${NODE_RANK:-}" ]]; then NODE_RANK_DECLARED=1; fi
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-}"
LAUNCHER="manual"
RDZV_BACKEND="static"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-}"
RDZV_ID="${RDZV_ID:-}"
LOCAL_ADDR="${LOCAL_ADDR:-}"
RDZV_TIMEOUT=600
DIST_TIMEOUT=600
DIST_BACKEND="auto"
MAX_RESTARTS=0
DRY_RUN=0
MIXED_PRECISION="auto"
REQUIRED_TRANSFORMERS_VERSION="4.57.1"
RESUME_FROM=""
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
VIDEO_DIR=""
CACHE_DIR=""
VAL_CACHE_DIR=""
RESOLUTION="480p"
NUM_FRAMES=""
SEED=42
NUM_WORKERS=4
ENABLE_TF32=""
FLOAT32_MATMUL_PRECISION="high"
ENABLE_TORCH_COMPILE=""
TORCH_COMPILE_SCOPE="student"
TORCH_COMPILE_MODE="reduce-overhead"
TORCH_COMPILE_BACKEND="inductor"
TORCH_COMPILE_FULLGRAPH=""
TORCH_COMPILE_DYNAMIC=""
DUAL_MODEL_ARG=""

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
        --gpus|--nproc_per_node|--nproc-per-node) NUM_GPUS="$2"; shift 2 ;;
        --nnodes) NNODES="$2"; NNODES_DECLARED=1; shift 2 ;;
        --node_rank|--node-rank) NODE_RANK="$2"; NODE_RANK_DECLARED=1; shift 2 ;;
        --launcher) LAUNCHER="$2"; shift 2 ;;
        --rdzv_backend|--rdzv-backend) RDZV_BACKEND="$2"; shift 2 ;;
        --rdzv_endpoint|--rdzv-endpoint) RDZV_ENDPOINT="$2"; shift 2 ;;
        --rdzv_id|--rdzv-id) RDZV_ID="$2"; shift 2 ;;
        --local_addr|--local-addr) LOCAL_ADDR="$2"; shift 2 ;;
        --rdzv_timeout) RDZV_TIMEOUT="$2"; shift 2 ;;
        --dist_timeout) DIST_TIMEOUT="$2"; shift 2 ;;
        --dist_backend) DIST_BACKEND="$2"; shift 2 ;;
        --max_restarts|--max-restarts) MAX_RESTARTS="$2"; shift 2 ;;
        --dry_run|--dry-run) DRY_RUN=1; shift ;;
        --mixed_precision) MIXED_PRECISION="$2"; shift 2 ;;
        --required_transformers_version) REQUIRED_TRANSFORMERS_VERSION="$2"; shift 2 ;;
        --resume_from) RESUME_FROM="$2"; shift 2 ;;
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
        --video_dir) VIDEO_DIR="$2"; shift 2 ;;
        --cache_dir) CACHE_DIR="$2"; shift 2 ;;
        --val_cache_dir) VAL_CACHE_DIR="$2"; shift 2 ;;
        --resolution) RESOLUTION="$2"; shift 2 ;;
        --num_frames) NUM_FRAMES="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --num_workers) NUM_WORKERS="$2"; shift 2 ;;
        --gradient_checkpointing) GRADIENT_CHECKPOINTING="--gradient_checkpointing"; shift ;;
        --cpu_offload) CPU_OFFLOAD="--cpu_offload"; shift ;;
        --enable_tf32) ENABLE_TF32="--enable_tf32"; shift ;;
        --float32_matmul_precision) FLOAT32_MATMUL_PRECISION="$2"; shift 2 ;;
        --enable_torch_compile) ENABLE_TORCH_COMPILE="--enable_torch_compile"; shift ;;
        --torch_compile_scope) TORCH_COMPILE_SCOPE="$2"; shift 2 ;;
        --torch_compile_mode) TORCH_COMPILE_MODE="$2"; shift 2 ;;
        --torch_compile_backend) TORCH_COMPILE_BACKEND="$2"; shift 2 ;;
        --torch_compile_fullgraph) TORCH_COMPILE_FULLGRAPH="--torch_compile_fullgraph"; shift ;;
        --torch_compile_dynamic) TORCH_COMPILE_DYNAMIC="--torch_compile_dynamic"; shift ;;
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
            echo "  fsdp                  Post-load FSDP parameter sharding"
            echo "  deepspeed             Post-load DeepSpeed ZeRO optimizer/parameter sharding"
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
            echo "  --gpus/--nproc_per_node Workers per node, identical on every node (default: 8)"
            echo "  --nnodes              Fixed node count; elastic MIN:MAX is unsupported (default: 1)"
            echo "  --node_rank           Node index, required for manual multi-node launch"
            echo "  --launcher            manual | slurm (one torchrun agent per Slurm node)"
            echo "  --rdzv_backend        static | c10d (default: static)"
            echo "  --rdzv_endpoint       Rendezvous host:port; required for multi-node"
            echo "  --rdzv_id             Shared, unique job ID; required for multi-node"
            echo "  --local_addr          This node's reachable advertised host/IP (optional; c10d otherwise uses hostname)"
            echo "  --rdzv_timeout        Rendezvous timeout seconds (default: 600)"
            echo "  --dist_timeout        Process-group/collective timeout seconds (default: 600)"
            echo "  --dist_backend        auto | nccl | gloo; Gloo is CPU-only validation"
            echo "  --max_restarts        Fixed-size torchrun worker retries (default: 0)"
            echo "  --resume_from         Explicit checkpoint used on start/restart; no automatic discovery"
            echo "  --mixed_precision     auto | bf16 | fp16 | no (default: auto)"
            echo "  --required_transformers_version Version used by preflight AND training (default: 4.57.1)"
            echo "  --dry_run             Validate and print launch command without importing models or torch"
            echo "  --steps               Target inference steps (default: 4)"
            echo "  --batch_size          Batch size per GPU (default: 1)"
            echo "  --lr                  Learning rate (default: 1e-5)"
            echo "  --max_train_steps     Max training steps (default: 10000)"
            echo "  --save_every          Save checkpoint every N steps (default: 1000)"
            echo "  --parallel            Parallel mode: ddp | fsdp | deepspeed (default: ddp)"
            echo "  --sp_size             Adapter-gated sequence parallel size (default: 1)"
            echo "                        No generic TP/PP or stock-model SP implementation; uneven local workers are unsupported."
            echo "  --ds_stage            DeepSpeed ZeRO stage: 1 | 2 | 3 (default: 2)"
            echo "  --fsdp_strategy       FSDP strategy: full | hybrid (default: full)"
            echo "  --report_to           console,tensorboard,wandb,all,none (default: console)"
            echo "  --tensorboard_log_dir Custom TensorBoard log directory (optional)"
            echo "  --wandb_project       W&B project name (default: worlddistill)"
            echo "  --wandb_entity        W&B entity/team (optional)"
            echo "  --wandb_run_name      W&B run name (optional)"
            echo "  --wandb_tags          Comma-separated W&B tags (optional)"
            echo "  --video_dir           Root directory for raw-video manifests (optional)"
            echo "  --cache_dir           Root directory for cached latent manifests (optional)"
            echo "  --val_cache_dir       Cache directory for validation manifest (optional)"
            echo "  --resolution          Training resolution tag, e.g. 480p/720p (default: 480p)"
            echo "  --num_frames          Frames per clip (default: 160 for context_forcing, else 49)"
            echo "  --seed                Random seed (default: 42)"
            echo "  --num_workers         Dataloader workers per rank (default: 4)"
            echo "  --gradient_checkpointing  Enable gradient checkpointing"
            echo "  --cpu_offload         Enable CPU offloading (FSDP/DeepSpeed)"
            echo "  --enable_tf32         Enable TF32 matmul/cudnn on supported GPUs"
            echo "  --float32_matmul_precision highest|high|medium (default: high)"
            echo "  --enable_torch_compile Enable training-time torch.compile"
            echo "  --torch_compile_scope student|teacher|both (default: student)"
            echo "  --torch_compile_mode  default|reduce-overhead|max-autotune|max-autotune-no-cudagraphs"
            echo "  --torch_compile_backend torch.compile backend (default: inductor)"
            echo "  --torch_compile_fullgraph  Enable fullgraph compilation"
            echo "  --torch_compile_dynamic    Enable dynamic-shape compilation"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

fail_launch() { echo "ERROR: $*" >&2; exit 1; }
[[ -z "${RANK+x}" ]] || fail_launch "run_train.sh launches torchrun; do not invoke it inside an existing torchrun worker."
case "${LAUNCHER}" in
    manual) ;;
    slurm)
        [[ -n "${SLURM_NNODES:-}" && -n "${SLURM_PROCID:-}" && -n "${SLURM_LOCALID:-}" ]] || \
            fail_launch "--launcher slurm requires srun with exactly one agent task per node."
        [[ "${SLURM_LOCALID}" = "0" && "${SLURM_NTASKS:-}" = "${SLURM_NNODES}" ]] || \
            fail_launch "Use srun --ntasks-per-node=1, not one srun task per GPU."
        if [[ "${NNODES_DECLARED}" = "1" && "${NNODES}" != "${SLURM_NNODES}" ]]; then
            fail_launch "--nnodes/NNODES conflicts with SLURM_NNODES."
        fi
        if [[ "${NODE_RANK_DECLARED}" = "1" && "${NODE_RANK}" != "${SLURM_PROCID}" ]]; then
            fail_launch "--node_rank/NODE_RANK conflicts with SLURM_PROCID."
        fi
        [[ -z "${SLURM_NODEID:-}" || "${SLURM_NODEID}" = "${SLURM_PROCID}" ]] || \
            fail_launch "Slurm task/node ranks disagree with one-agent-per-node placement."
        NNODES="${SLURM_NNODES}"
        NODE_RANK="${SLURM_PROCID}"
        RDZV_ID="${RDZV_ID:-${SLURM_JOB_ID:-}}"
        ;;
    *) fail_launch "--launcher must be manual or slurm" ;;
esac
for value in "${NNODES}" "${NUM_GPUS}" "${RDZV_TIMEOUT}" "${DIST_TIMEOUT}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail_launch "Node/worker counts and timeouts must be positive integers; elastic MIN:MAX is unsupported."
done
[[ "${MAX_RESTARTS}" =~ ^(0|[1-9][0-9]*)$ ]] || fail_launch "--max_restarts must be a non-negative integer."
if [[ -z "${NODE_RANK}" ]]; then
    [[ "${NNODES}" = "1" ]] || fail_launch "--node_rank is required for manual multi-node launch."
    NODE_RANK=0
fi
[[ "${NODE_RANK}" =~ ^(0|[1-9][0-9]*)$ ]] || fail_launch "--node_rank must be a non-negative integer."
(( NODE_RANK < NNODES )) || fail_launch "--node_rank must be smaller than --nnodes."
case "${RDZV_BACKEND}" in static|c10d) ;; *) fail_launch "--rdzv_backend must be static or c10d." ;; esac
case "${DIST_BACKEND}" in auto|nccl|gloo) ;; *) fail_launch "--dist_backend must be auto, nccl or gloo." ;; esac
case "${PARALLEL_MODE}" in ddp|fsdp|deepspeed) ;; *) fail_launch "--parallel must be ddp, fsdp or deepspeed; generic TP/PP is not implemented." ;; esac
[[ "${SP_SIZE}" =~ ^[1-9][0-9]*$ ]] || fail_launch "--sp_size must be a positive integer; model-specific SP support is still required."
case "${MIXED_PRECISION}" in auto|bf16|fp16|no) ;; *) fail_launch "Invalid --mixed_precision." ;; esac
if (( NNODES > 1 )); then
    [[ -n "${RDZV_ENDPOINT}" && -n "${RDZV_ID}" ]] || fail_launch "Multi-node launch needs shared --rdzv_endpoint and --rdzv_id on every node."
fi
if [[ -n "${RDZV_ENDPOINT}" ]]; then
    [[ "${RDZV_ENDPOINT}" =~ ^[^[:space:]]+:[0-9]+$ ]] || fail_launch "--rdzv_endpoint must be host:port."
    rendezvous_port="${RDZV_ENDPOINT##*:}"
    (( 10#${rendezvous_port} > 0 && 10#${rendezvous_port} <= 65535 )) || fail_launch "Rendezvous port must be 1..65535."
fi
if [[ "${FSDP_STRATEGY}" = "hybrid" && "${PARALLEL_MODE}" = "fsdp" ]]; then
    (( NNODES >= 2 && NUM_GPUS >= 2 )) || fail_launch "Hybrid FSDP needs >=2 nodes and >=2 workers per node; use full otherwise."
fi

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

if [ -z "${NUM_FRAMES}" ]; then
    if [ "${METHOD}" = "context_forcing" ]; then
        NUM_FRAMES=160
    else
        NUM_FRAMES=49
    fi
fi

if [ "${METHOD}" = "step_distill" ] && [ "${PARALLEL_MODE}" != "ddp" ]; then
    # The stock 4-step preset enables two independently routed students. Only
    # the primary student is sharded today, so select the supported single-
    # student path for FSDP/DeepSpeed instead of failing after model loading.
    DUAL_MODEL_ARG="--no-use_dual_model"
    echo ">>> Disabling dual-student routing for ${PARALLEL_MODE}; serial/DDP is required for dual mode."
fi

# CUDA_VISIBLE_DEVICES (including UUID/MIG masks and an intentionally empty
# CPU mask) belongs to the scheduler/user. LOCAL_RANK indexes this visible set.
export WORLD_DISTILL_DIST_BACKEND="${DIST_BACKEND}"
export WORLD_DISTILL_DIST_TIMEOUT="${DIST_TIMEOUT}"
export WORLD_DISTILL_EXPECTED_WORLD_SIZE="$((NNODES * NUM_GPUS))"
export WORLD_DISTILL_EXPECTED_LOCAL_WORLD_SIZE="${NUM_GPUS}"
export WORLD_DISTILL_FIXED_WORLD_SIZE=1

echo "============================================"
echo "WorldDistill Training"
echo "  Method:      ${METHOD}"
echo "  Model CLS:   ${MODEL_CLS}"
echo "  Teacher:     ${TEACHER_MODEL}"
echo "  Student:     ${STUDENT_MODEL:-'(init from teacher)'}"
echo "  Steps:       ${NUM_STEPS}"
echo "  Topology:    ${NNODES} nodes x ${NUM_GPUS} workers; node rank ${NODE_RANK}"
echo "  GPU mask:    ${CUDA_VISIBLE_DEVICES-'<inherited unrestricted visibility>'}"
echo "  Backend:     ${DIST_BACKEND}"
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

launch_cmd=(torchrun
    "--nnodes=${NNODES}"
    "--nproc_per_node=${NUM_GPUS}"
    "--node_rank=${NODE_RANK}"
    "--max_restarts=${MAX_RESTARTS}"
)
if [[ -n "${LOCAL_ADDR}" ]]; then
    launch_cmd+=("--local_addr=${LOCAL_ADDR}")
fi
if [[ -z "${RDZV_ENDPOINT}" && "${NNODES}" = "1" ]]; then
    # Separate local jobs get separate ephemeral rendezvous endpoints.
    launch_cmd+=(--standalone)
else
    launch_cmd+=("--rdzv_backend=${RDZV_BACKEND}" "--rdzv_endpoint=${RDZV_ENDPOINT}" "--rdzv_id=${RDZV_ID:-worlddistill}")
fi
if [[ "${RDZV_BACKEND}" = "static" && -n "${RDZV_ENDPOINT}" ]]; then
    launch_cmd+=("--rdzv_conf=timeout=${RDZV_TIMEOUT}")
else
    launch_cmd+=("--rdzv_conf=join_timeout=${RDZV_TIMEOUT},read_timeout=${RDZV_TIMEOUT}")
fi
train_cmd=(
    "${launch_cmd[@]}"
    -m training.train_distill
    --distill_method "${METHOD}"
    --teacher_model_path "${TEACHER_MODEL}"
    --model_cls "${MODEL_CLS}"
    --data_json "${DATA_JSON}"
    --output_dir "${OUTPUT_DIR}"
    --num_inference_steps "${NUM_STEPS}"
    --batch_size "${BATCH_SIZE}"
    --learning_rate "${LR}"
    --max_train_steps "${MAX_TRAIN_STEPS}"
    --save_every "${SAVE_EVERY}"
    --seed "${SEED}"
    --num_workers "${NUM_WORKERS}"
    --resolution "${RESOLUTION}"
    --num_frames "${NUM_FRAMES}"
    --parallel_mode "${PARALLEL_MODE}"
    --sp_size "${SP_SIZE}"
    --deepspeed_stage "${DS_STAGE}"
    --fsdp_shard_strategy "${FSDP_STRATEGY}"
    --report_to "${REPORT_TO}"
    --wandb_project "${WANDB_PROJECT}"
    --float32_matmul_precision "${FLOAT32_MATMUL_PRECISION}"
    --torch_compile_scope "${TORCH_COMPILE_SCOPE}"
    --torch_compile_mode "${TORCH_COMPILE_MODE}"
    --torch_compile_backend "${TORCH_COMPILE_BACKEND}"
    --config "${CONFIG}"
    --mixed_precision "${MIXED_PRECISION}"
    --required_transformers_version "${REQUIRED_TRANSFORMERS_VERSION}"
)

if [ -n "${RESUME_FROM}" ]; then
    train_cmd+=(--resume_from "${RESUME_FROM}")
fi

if [ -n "${STUDENT_MODEL}" ]; then
    train_cmd+=(--student_model_path "${STUDENT_MODEL}")
fi
if [ -n "${VIDEO_DIR}" ]; then
    train_cmd+=(--video_dir "${VIDEO_DIR}")
fi
if [ -n "${CACHE_DIR}" ]; then
    train_cmd+=(--cache_dir "${CACHE_DIR}")
fi
if [ -n "${VAL_DATA_JSON}" ]; then
    train_cmd+=(--val_data_json "${VAL_DATA_JSON}")
fi
if [ -n "${VAL_CACHE_DIR}" ]; then
    train_cmd+=(--val_cache_dir "${VAL_CACHE_DIR}")
fi
if [ -n "${EVAL_EVERY}" ]; then
    train_cmd+=(--eval_every "${EVAL_EVERY}")
fi
if [ -n "${EVAL_BATCHES}" ]; then
    train_cmd+=(--eval_batches "${EVAL_BATCHES}")
fi
if [ -n "${DUAL_MODEL_ARG}" ]; then
    train_cmd+=("${DUAL_MODEL_ARG}")
fi
if [ -n "${TENSORBOARD_LOG_DIR}" ]; then
    train_cmd+=(--tensorboard_log_dir "${TENSORBOARD_LOG_DIR}")
fi
if [ -n "${WANDB_ENTITY}" ]; then
    train_cmd+=(--wandb_entity "${WANDB_ENTITY}")
fi
if [ -n "${WANDB_RUN_NAME}" ]; then
    train_cmd+=(--wandb_run_name "${WANDB_RUN_NAME}")
fi
if [ -n "${WANDB_TAGS}" ]; then
    train_cmd+=(--wandb_tags "${WANDB_TAGS}")
fi
if [ -n "${GRADIENT_CHECKPOINTING}" ]; then
    train_cmd+=("${GRADIENT_CHECKPOINTING}")
fi
if [ -n "${CPU_OFFLOAD}" ]; then
    train_cmd+=("${CPU_OFFLOAD}")
fi
if [ -n "${ENABLE_TF32}" ]; then
    train_cmd+=("${ENABLE_TF32}")
fi
if [ -n "${ENABLE_TORCH_COMPILE}" ]; then
    train_cmd+=("${ENABLE_TORCH_COMPILE}")
fi
if [ -n "${TORCH_COMPILE_FULLGRAPH}" ]; then
    train_cmd+=("${TORCH_COMPILE_FULLGRAPH}")
fi
if [ -n "${TORCH_COMPILE_DYNAMIC}" ]; then
    train_cmd+=("${TORCH_COMPILE_DYNAMIC}")
fi

if [[ "${DRY_RUN}" = "1" ]]; then
    printf 'WORLD_DISTILL_DIST_BACKEND=%q WORLD_DISTILL_DIST_TIMEOUT=%q WORLD_DISTILL_EXPECTED_WORLD_SIZE=%q WORLD_DISTILL_EXPECTED_LOCAL_WORLD_SIZE=%q WORLD_DISTILL_FIXED_WORLD_SIZE=1 ' \
        "${DIST_BACKEND}" "${DIST_TIMEOUT}" "${WORLD_DISTILL_EXPECTED_WORLD_SIZE}" "${NUM_GPUS}"
    printf '%q ' "${train_cmd[@]}"
    printf '\n'
    exit 0
fi
mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"
echo ">>> Checking dependency and local device compatibility..."
python3 -m training.env_compat --mode train --required_transformers_version "${REQUIRED_TRANSFORMERS_VERSION}" \
    --dist_backend "${DIST_BACKEND}" --nproc_per_node "${NUM_GPUS}"
"${train_cmd[@]}"

echo ""
echo "Training complete! Checkpoints and logs saved to: ${OUTPUT_DIR}"

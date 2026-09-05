#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL=""
INSTALL_KERNELS=0
INSTALL_DEEPSPEED=0
INSTALL_DEV=0

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_env.sh [OPTIONS]

Install into the currently active Python environment. Create and activate a
conda or venv environment before invoking this script.

Options:
  --python PATH             Python interpreter (default: $PYTHON_BIN or python3)
  --torch-index-url URL     Install/upgrade Torch from this official index
  --install-kernels         Install optional FlashAttention/sgl-kernel packages
  --install-deepspeed       Install optional DeepSpeed support
  --dev                     Install pytest and ruff
  -h, --help                Show this help

Examples:
  bash scripts/setup_env.sh
  bash scripts/setup_env.sh --torch-index-url https://download.pytorch.org/whl/cu128
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --torch-index-url) TORCH_INDEX_URL="$2"; shift 2 ;;
        --install-kernels) INSTALL_KERNELS=1; shift ;;
        --install-deepspeed) INSTALL_DEEPSPEED=1; shift ;;
        --dev) INSTALL_DEV=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ -n "${TORCH_INDEX_URL}" ]]; then
    "${PYTHON_BIN}" -m pip install --upgrade torch torchvision torchaudio \
        --index-url "${TORCH_INDEX_URL}"
fi

if ! "${PYTHON_BIN}" -c 'import torch' >/dev/null 2>&1; then
    echo "PyTorch is not installed. Select a build for the target GPU at:" >&2
    echo "  https://pytorch.org/get-started/locally/" >&2
    echo "Then rerun this script, or pass --torch-index-url." >&2
    exit 1
fi

ROOT_EXTRAS="train,inference"
if [[ "${INSTALL_DEV}" -eq 1 ]]; then
    ROOT_EXTRAS="${ROOT_EXTRAS},dev"
fi

"${PYTHON_BIN}" -m pip install -e "${PROJECT_ROOT}[${ROOT_EXTRAS}]"
# WorldDistill resolves its controlled training/inference dependency set first,
# then registers the vendored LightX2V package without pulling optional app,
# server, or architecture-specific extras into every environment.
"${PYTHON_BIN}" -m pip install --no-deps -e "${PROJECT_ROOT}/inference"

# The full LightX2V registry imports GPU-specific backends eagerly (including
# Triton) and is therefore a CUDA-host validation, not a portable CPU install
# check. Validate the training entry, installed package metadata, and syntax
# here; `run_infer.sh` performs the runtime dependency check on the GPU host.
"${PYTHON_BIN}" -c 'import training.train_distill; from importlib.metadata import version; print(version("lightx2v"))'
"${PYTHON_BIN}" -m compileall -q "${PROJECT_ROOT}/inference/lightx2v" "${PROJECT_ROOT}/inference/lightx2v_platform"

if [[ "${INSTALL_KERNELS}" -eq 1 ]]; then
    "${PYTHON_BIN}" -m pip install -r "${PROJECT_ROOT}/requirements-kernels.txt"
fi

if [[ "${INSTALL_DEEPSPEED}" -eq 1 ]]; then
    "${PYTHON_BIN}" -m pip install -r "${PROJECT_ROOT}/requirements-distributed.txt"
fi

"${PYTHON_BIN}" -m pip check
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/check_cuda_compat.py"
echo "Setup complete. Run: ${PYTHON_BIN} -m pytest -q training/tests"

#!/bin/bash
# ============================================================================
# WorldDistill - Environment Setup
# ============================================================================
set -e

echo "============================================"
echo "Step 1: Create conda environment"
echo "============================================"
conda create -n worlddistill python=3.10 -y
conda activate worlddistill

echo "============================================"
echo "Step 2: Install PyTorch (CUDA 12.1)"
echo "============================================"
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

echo "============================================"
echo "Step 3: Install Flash Attention"
echo "============================================"
pip install flash-attn --no-build-isolation

echo "============================================"
echo "Step 4: Install inference engine (based on LightX2V)"
echo "============================================"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

cd "${PROJECT_ROOT}/inference"
pip install -e .

echo "============================================"
echo "Step 5: Install additional dependencies"
echo "============================================"
cd "${PROJECT_ROOT}"
pip install -r requirements.txt

echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "Activate environment: conda activate worlddistill"
echo "Download models:      python tools/download_models.py --list"
echo "Run inference:        bash scripts/run_infer.sh --help"

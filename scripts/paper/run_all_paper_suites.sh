#!/bin/bash
set -euo pipefail

PAPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${PAPER_SCRIPT_DIR}/run_runtime_ablation.sh"
bash "${PAPER_SCRIPT_DIR}/run_fewstep_t2v_suite.sh"
bash "${PAPER_SCRIPT_DIR}/run_streaming_longvideo_suite.sh"
bash "${PAPER_SCRIPT_DIR}/run_world_model_suite.sh"
bash "${PAPER_SCRIPT_DIR}/run_camera_control_suite.sh"

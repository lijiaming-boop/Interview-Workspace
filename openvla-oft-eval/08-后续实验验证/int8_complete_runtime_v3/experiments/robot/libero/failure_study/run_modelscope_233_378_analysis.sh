#!/usr/bin/env bash
# Reproduce and diagnose LIBERO-Spatial failures 233 (task 4/state 32) and
# 378 (task 7/state 27) on a ModelScope GPU instance.
set -euo pipefail

MODE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OFT_ENV="${OFT_ENV:-oft}"
CKPT_DIR="${CKPT_DIR:-/mnt/workspace/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial}"
GPU="${GPU:-0}"
RUN_TAG="${RUN_TAG:-modelscope-233-378-$(date '+%Y%m%d-%H%M%S')}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/experiments/failure_study_runs/${RUN_TAG}}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Install/enable the OpenVLA-OFT runtime first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${OFT_ENV}"

if [ ! -d "${CKPT_DIR}" ]; then
  echo "Checkpoint not found: ${CKPT_DIR}" >&2
  echo "Set CKPT_DIR to the mounted OpenVLA-OFT LIBERO-Spatial checkpoint." >&2
  exit 1
fi
if ! nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU}" >/dev/null 2>&1; then
  echo "GPU ${GPU} is unavailable; set GPU to a visible GPU index." >&2
  exit 1
fi
python -c 'import torch, libero, transformers; assert torch.cuda.is_available(), "CUDA is unavailable"'

export CUDA_VISIBLE_DEVICES="${GPU}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export CKPT_DIR OUTPUT_ROOT RUN_TAG

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU}")"
echo "Checkpoint: ${CKPT_DIR}"
echo "Results: ${OUTPUT_ROOT}"

case "${MODE}" in
  boundary)
    bash "${SCRIPT_DIR}/run_simulation_study.sh" boundary8
    ;;
  recovery)
    bash "${SCRIPT_DIR}/run_simulation_study.sh" grasp_recovery_boundary
    ;;
  all)
    bash "${SCRIPT_DIR}/run_simulation_study.sh" boundary8
    bash "${SCRIPT_DIR}/run_simulation_study.sh" grasp_recovery_boundary
    ;;
  *)
    echo "Usage: $0 {boundary|recovery|all}" >&2
    exit 2
    ;;
esac

cat <<EOF

Completed. Read these generated files:
  ${OUTPUT_ROOT}/sim-boundary-chunk8/analysis/analysis.md
  ${OUTPUT_ROOT}/sim-boundary-chunk8/analysis/chunk_summary.csv
  ${OUTPUT_ROOT}/sim-grasp-recovery-233-378/actions/task_04/state_32/events.jsonl
  ${OUTPUT_ROOT}/sim-grasp-recovery-233-378/actions/task_07/state_27/events.jsonl
EOF

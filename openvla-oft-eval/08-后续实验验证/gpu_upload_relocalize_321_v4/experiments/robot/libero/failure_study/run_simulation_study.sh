#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/study_common.sh"
study_require_runtime

MODE="${1:-all}"
case "${MODE}" in
  boundary8)
    study_run_eval "sim-boundary-chunk8" "4:32,7:27" 8 1.0 True True 0.0 20
    ;;
  boundary4)
    study_run_eval "sim-boundary-chunk4" "4:32,7:27" 4 1.0 True True 0.0 20
    ;;
  relocalize_control)
    study_run_eval "sim-relocalize-control" "6:20" 4 1.0 True True 0.0 20
    ;;
  relocalize_motion)
    study_run_eval "sim-relocalize-motion-1cm" "6:20" 4 1.0 True True 0.01 20
    ;;
  grasp_control)
    study_run_eval "sim-grasp-control" "4:29,5:35" 8 1.0 False True 0.0 20 False
    ;;
  grasp_verify)
    study_run_eval "sim-grasp-verify" "4:29,5:35" 8 1.0 False True 0.0 20 True
    ;;
  grasp_recovery)
    study_run_eval "sim-grasp-recovery" "4:29,5:35" 8 1.0 False True 0.0 20 True
    ;;
  grasp_recovery_boundary)
    study_run_eval "sim-grasp-recovery-233-378" "4:32,7:27" 8 1.0 False True 0.0 20 True
    ;;
  freq20)
    study_run_eval "sim-frequency-20hz" "4:32,6:20,7:27" 4 1.0 False True 0.0 20
    ;;
  freq50)
    study_run_eval "sim-frequency-50hz" "4:32,6:20,7:27" 4 1.0 False True 0.0 50
    ;;
  all)
    study_run_eval "sim-boundary-chunk8" "4:32,7:27" 8 1.0 True True 0.0 20
    study_run_eval "sim-boundary-chunk4" "4:32,7:27" 4 1.0 True True 0.0 20
    study_run_eval "sim-relocalize-control" "6:20" 4 1.0 True True 0.0 20
    study_run_eval "sim-relocalize-motion-1cm" "6:20" 4 1.0 True True 0.01 20
    study_run_eval "sim-grasp-control" "4:29,5:35" 8 1.0 False True 0.0 20 False
    study_run_eval "sim-grasp-verify" "4:29,5:35" 8 1.0 False True 0.0 20 True
    study_run_eval "sim-frequency-20hz" "4:32,6:20,7:27" 4 1.0 False True 0.0 20
    study_run_eval "sim-frequency-50hz" "4:32,6:20,7:27" 4 1.0 False True 0.0 50
    ;;
  *)
    echo "Usage: $0 {boundary8|boundary4|relocalize_control|relocalize_motion|grasp_control|grasp_verify|grasp_recovery|grasp_recovery_boundary|freq20|freq50|all}" >&2
    exit 2
    ;;
esac

echo "Simulation study outputs: ${STUDY_OUTPUT_ROOT}"

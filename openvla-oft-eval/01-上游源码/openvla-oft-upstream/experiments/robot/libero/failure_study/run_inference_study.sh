#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/study_common.sh"
study_require_runtime

MODE="${1:-all}"
case "${MODE}" in
  chunk8_control)
    study_run_eval "inference-chunk8-control" "4:32,6:20,7:27" 8 1.0 False True 0.0 20
    ;;
  chunk4)
    study_run_eval "inference-chunk4-first4" "4:32,6:20,7:27" 4 1.0 False True 0.0 20
    ;;
  camera_control)
    study_run_eval "inference-camera-w1" "4:29,5:35" 8 1.0 True True 0.0 20
    ;;
  camera_w125)
    study_run_eval "inference-camera-w1p25" "4:29,5:35" 8 1.25 True True 0.0 20
    ;;
  all)
    study_run_eval "inference-chunk8-control" "4:32,6:20,7:27" 8 1.0 False True 0.0 20
    study_run_eval "inference-chunk4-first4" "4:32,6:20,7:27" 4 1.0 False True 0.0 20
    study_run_eval "inference-camera-w1" "4:29,5:35" 8 1.0 True True 0.0 20
    study_run_eval "inference-camera-w1p25" "4:29,5:35" 8 1.25 True True 0.0 20
    ;;
  *)
    echo "Usage: $0 {chunk8_control|chunk4|camera_control|camera_w125|all}" >&2
    exit 2
    ;;
esac

echo "Inference study outputs: ${STUDY_OUTPUT_ROOT}"

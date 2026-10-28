#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/study_common.sh"
study_require_runtime

TRIALS="${TRIALS:-50}"
if ! [[ "${TRIALS}" =~ ^[0-9]+$ ]] || [ "${TRIALS}" -lt 1 ] || [ "${TRIALS}" -gt 50 ]; then
  echo "TRIALS must be an integer in [1, 50], got ${TRIALS}" >&2
  exit 2
fi

TARGETS=""
for ((state_idx = 0; state_idx < TRIALS; state_idx++)); do
  if [ -n "${TARGETS}" ]; then
    TARGETS+=","
  fi
  TARGETS+="4:${state_idx}"
done

LABEL="training-top-drawer-sweep-${TRIALS}"
study_run_eval "${LABEL}" "${TARGETS}" 8 1.0 False True 0.0 20

python "${SCRIPT_DIR}/summarize_targeted_eval.py" \
  --log "${STUDY_OUTPUT_ROOT}/${LABEL}/console.log" \
  --output-dir "${STUDY_OUTPUT_ROOT}/${LABEL}/summary"

echo "Top-drawer sweep complete: ${STUDY_OUTPUT_ROOT}/${LABEL}"
echo "Fill expert corrections in summary/dagger_collection_manifest.csv before building the RLDS dataset."

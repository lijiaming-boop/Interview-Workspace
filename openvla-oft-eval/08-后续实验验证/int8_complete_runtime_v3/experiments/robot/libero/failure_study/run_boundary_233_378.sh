#!/usr/bin/env bash
# Replay the two empty-grasp failures and report whether a chunk-boundary
# grasp check would occur before the first post-close transfer action.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/mnt/workspace/libero-config}"
export RUN_TAG="${RUN_TAG:-boundary-233-378-$(date '+%Y%m%d-%H%M%S')}"

if [ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]; then
  echo "Missing LIBERO config: ${LIBERO_CONFIG_PATH}/config.yaml" >&2
  exit 1
fi

cd "${REPO_ROOT}"
bash "${SCRIPT_DIR}/run_simulation_study.sh" boundary8

REPORT="${REPO_ROOT}/experiments/failure_study_runs/${RUN_TAG}/sim-boundary-chunk8/analysis/analysis.md"
test -f "${REPORT}"
echo
echo "Boundary analysis: ${REPORT}"
cat "${REPORT}"

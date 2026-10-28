#!/usr/bin/env bash
set -euo pipefail

# Run raw and repaired conditions with identical seed/checkpoint/episode count.
# Isaac Lab's launcher is intentionally configurable because installations use
# either isaaclab.sh or isaaclab.bat.

ISAACLAB_LAUNCHER="${ISAACLAB_LAUNCHER:-./isaaclab.sh}"
CHECKPOINT="${1:?usage: run_domain_gap_eval.sh /path/to/checkpoint [episodes] [num_envs]}"
EPISODES="${2:-50}"
NUM_ENVS="${3:-8}"
OPENVLA_ROOT="${OPENVLA_OFT_ROOT:?set OPENVLA_OFT_ROOT to 02-实验代码/patch-check3}"
SEED="${SEED:-7}"
OUT_ROOT="${OUT_ROOT:-12-IsaacLab-Bridge-Franka/outputs/domain-gap}"

COMMON=(
  --policy openvla
  --checkpoint "$CHECKPOINT"
  --openvla_root "$OPENVLA_ROOT"
  --unnorm_key "${UNNORM_KEY:-bridge_orig}"
  --episodes "$EPISODES"
  --num_envs "$NUM_ENVS"
  --open_loop_steps 1
  --seed "$SEED"
  --headless
)

"$ISAACLAB_LAUNCHER" -p 12-IsaacLab-Bridge-Franka/run_closed_loop.py \
  "${COMMON[@]}" --domain_mode raw --output_dir "$OUT_ROOT/raw"

"$ISAACLAB_LAUNCHER" -p 12-IsaacLab-Bridge-Franka/run_closed_loop.py \
  "${COMMON[@]}" --domain_mode bridge_aligned --output_dir "$OUT_ROOT/bridge_aligned"

RAW_SUMMARY="$(find "$OUT_ROOT/raw" -type f -name summary.json | sort | tail -n 1)"
ALIGNED_SUMMARY="$(find "$OUT_ROOT/bridge_aligned" -type f -name summary.json | sort | tail -n 1)"

python 12-IsaacLab-Bridge-Franka/analyze_results.py \
  --before "$RAW_SUMMARY" \
  --after "$ALIGNED_SUMMARY" \
  --report "$OUT_ROOT/domain_gap_report.md"

#!/usr/bin/env bash
set -euo pipefail

cd /mnt/workspace/openvla-oft
source /mnt/workspace/venv_oft/bin/activate
export PYTHONPATH="/mnt/workspace/openvla-oft:/mnt/workspace/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=2

BASE=/mnt/workspace/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial
RUN=/mnt/workspace/runs/topdrawer_teacher_distill_v1
LOG_ROOT=/mnt/workspace/eval_logs/distill_acceptance_v1
EVAL_ROOT=/mnt/workspace/eval_ckpts/distill_acceptance_v1
mkdir -p "$LOG_ROOT" "$EVAL_ROOT"

eval_model() {
  local model=$1
  local offset=$2
  local trials=$3
  local note=$4
  local log=$5
  python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint "$model" \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --center_crop True \
    --num_open_loop_steps 8 \
    --task_suite_name libero_spatial \
    --num_trials_per_task "$trials" \
    --initial_state_offset "$offset" \
    --num_steps_wait 10 \
    --local_log_dir /mnt/workspace/eval_logs \
    --run_id_note "$note" \
    --seed 7 \
    2>&1 | tee "$log"
}

for step in 25 50 100; do
  python tools/verify_distill_checkpoint.py \
    --base "$BASE" \
    --candidate "$RUN/action_head--${step}_checkpoint.pt"
  bash tools/make_eval_checkpoint.sh \
    "$BASE" \
    "$RUN/action_head--${step}_checkpoint.pt" \
    "$EVAL_ROOT/step${step}"
done

# Gate 1: states where the old head-only model regressed must stay successful;
# at least one of the two base failure states must be repaired.
for state in 5 6 11 29 32 38; do
  eval_model "$BASE" "$state" 1 "distill-screen-base-state${state}" \
    "$LOG_ROOT/base-state${state}.log"
  for step in 25 50 100; do
    eval_model "$EVAL_ROOT/step${step}" "$state" 1 \
      "distill-screen-step${step}-state${state}" \
      "$LOG_ROOT/step${step}-state${state}.log"
  done
done

python - "$LOG_ROOT" "$RUN" <<'PY'
import re
import sys
from pathlib import Path

log_root, run = map(Path, sys.argv[1:])
states = [5, 6, 11, 29, 32, 38]
preserve = {5, 6, 11, 38}
repair = {29, 32}
candidates = []

for step in (25, 50, 100):
    result = {}
    for state in states:
        text = (log_root / f"step{step}-state{state}.log").read_text(errors="replace")
        found = re.findall(r"Success:\s+(True|False)", text)
        if len(found) != 1:
            raise SystemExit(f"step {step}, state {state}: expected one result, got {len(found)}")
        result[state] = found[0] == "True"
    preserved = all(result[state] for state in preserve)
    repaired = sum(result[state] for state in repair)
    print(f"step={step} result={result} preserved={preserved} repaired={repaired}")
    if preserved and repaired >= 1:
        candidates.append((sum(result.values()), -step, step))

if not candidates:
    raise SystemExit("FAIL gate 1: no checkpoint preserved 5/6/11/38 and repaired 29 or 32")

selected = max(candidates)[2]
(log_root / "selected_step.txt").write_text(str(selected), encoding="utf-8")
print(f"PASS gate 1: selected step {selected}")
PY

SELECTED=$(cat "$LOG_ROOT/selected_step.txt")

# Gate 2: exact 50-state paired evaluation under the same seed and evaluator.
eval_model "$BASE" 0 50 distill-full-base "$LOG_ROOT/full-base.log"
eval_model "$EVAL_ROOT/step${SELECTED}" 0 50 \
  "distill-full-step${SELECTED}" "$LOG_ROOT/full-candidate.log"

python tools/compare_paired_results.py \
  --base "$LOG_ROOT/full-base.log" \
  --candidate "$LOG_ROOT/full-candidate.log" \
  --offset 0 \
  --expected 50 \
  --min-rate 0.96 \
  --max-regressions 1

echo "PASS: candidate step ${SELECTED} passed offline, screen, and 50-state gates"

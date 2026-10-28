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
RUN=/mnt/workspace/runs/topdrawer_failure_correction_v1
ROOT=/mnt/workspace/eval_logs/topdrawer_failure_correction_v1
EVAL=/mnt/workspace/eval_ckpts/topdrawer_failure_correction_v1
mkdir -p "$ROOT" "$EVAL"

eval_one() {
  local model=$1 state=$2 note=$3 log=$4
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
    --num_trials_per_task 1 \
    --initial_state_offset "$state" \
    --num_steps_wait 10 \
    --local_log_dir /mnt/workspace/eval_logs \
    --run_id_note "$note" \
    --seed 7 \
    2>&1 | tee "$log"
}

for step in 25 50 75; do
  python tools/verify_distill_checkpoint.py \
    --base "$BASE" \
    --candidate "$RUN/action_head--${step}_checkpoint.pt"
  bash tools/make_eval_checkpoint.sh \
    "$BASE" "$RUN/action_head--${step}_checkpoint.pt" "$EVAL/step${step}"
done

for state in 5 6 11 29 32 38; do
  eval_one "$BASE" "$state" "correction-base-state${state}" "$ROOT/base-state${state}.log"
  for step in 25 50 75; do
    eval_one "$EVAL/step${step}" "$state" "correction-step${step}-state${state}" "$ROOT/step${step}-state${state}.log"
  done
done

echo "Screen results:"
for step in 25 50 75; do
  echo "step=${step}"
  for state in 5 6 11 29 32 38; do
    printf "state=%s " "$state"
    grep -m1 -E "Success: (True|False)" "$ROOT/step${step}-state${state}.log" || true
  done
done

echo "Choose a step that preserves 5,6,11,38 and repairs 29 or 32."
read -r -p "Selected step (25/50/75): " STEP
case "$STEP" in 25|50|75) ;; *) echo "invalid step"; exit 2 ;; esac

python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint "$BASE" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --center_crop True \
  --num_open_loop_steps 8 \
  --task_suite_name libero_spatial \
  --num_trials_per_task 50 \
  --initial_state_offset 0 \
  --num_steps_wait 10 \
  --local_log_dir /mnt/workspace/eval_logs \
  --run_id_note "correction-full-base" \
  --seed 7 \
  2>&1 | tee "$ROOT/full-base.log"
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint "$EVAL/step${STEP}" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --center_crop True \
  --num_open_loop_steps 8 \
  --task_suite_name libero_spatial \
  --num_trials_per_task 50 \
  --initial_state_offset 0 \
  --num_steps_wait 10 \
  --local_log_dir /mnt/workspace/eval_logs \
  --run_id_note "correction-full-step${STEP}" \
  --seed 7 \
  2>&1 | tee "$ROOT/full-candidate.log"

python tools/compare_paired_results.py \
  --base "$ROOT/full-base.log" \
  --candidate "$ROOT/full-candidate.log" \
  --expected 50 \
  --min-rate 0.96 \
  --max-regressions 1

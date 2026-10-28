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
ROOT=/mnt/workspace/eval_logs/distill_acceptance_v1
STEP=$(cat "$ROOT/selected_step.txt")
CANDIDATE=/mnt/workspace/eval_ckpts/distill_acceptance_v1/step${STEP}
ROBUST="$ROOT/robust_step${STEP}"
mkdir -p "$ROBUST"

eval_50() {
  local model=$1
  local seed=$2
  local note=$3
  local log=$4
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
    --num_trials_per_task 50 \
    --initial_state_offset 0 \
    --num_steps_wait 10 \
    --local_log_dir /mnt/workspace/eval_logs \
    --run_id_note "$note" \
    --seed "$seed" \
    2>&1 | tee "$log"
}

for seed in 1 2 3 4 5; do
  eval_50 "$BASE" "$seed" "distill-robust-base-seed${seed}" "$ROBUST/base-seed${seed}.log"
  eval_50 "$CANDIDATE" "$seed" "distill-robust-step${STEP}-seed${seed}" "$ROBUST/candidate-seed${seed}.log"
done

python tools/compare_paired_results.py \
  --base "$ROBUST"/base-seed*.log \
  --candidate "$ROBUST"/candidate-seed*.log \
  --expected 250 \
  --min-rate 0.972 \
  --max-regressions 2 \
  --require-p 0.05

echo "PASS: step ${STEP} passed the 250-episode robustness gate"

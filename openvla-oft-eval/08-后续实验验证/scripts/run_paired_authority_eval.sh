#!/usr/bin/env bash
set -euo pipefail

# 基座模型是候选权威；head-only 只作为同状态对照。
OFT_ROOT="/mnt/workspace/openvla-oft"
BASE_CKPT="/mnt/workspace/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial"
FINE_CKPT="/mnt/workspace/eval_ckpts/head_only_topdrawer_150200"
LOG_DIR="/mnt/workspace/eval_logs/authority_20260906"
EVAL_SCRIPT="$OFT_ROOT/experiments/robot/libero/run_libero_eval.py"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$OFT_ROOT"
source /mnt/workspace/activate_oft.sh

export PYTHONPATH="$OFT_ROOT:/mnt/workspace/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=2
export WANDB_MODE=offline

test -f "$EVAL_SCRIPT" || { echo "找不到评估脚本: $EVAL_SCRIPT" >&2; exit 2; }
test -d "$BASE_CKPT" || { echo "找不到基座 checkpoint: $BASE_CKPT" >&2; exit 2; }
test -d "$FINE_CKPT" || { echo "找不到 head-only checkpoint: $FINE_CKPT" >&2; exit 2; }

TARGET_ARGS=()
if python "$EVAL_SCRIPT" --help 2>&1 | grep -q -- "--target_episodes"; then
  TARGET_EPISODES="$(python - <<'PY'
print(",".join(f"4:{i}" for i in range(50)))
PY
  )"
  TARGET_ARGS=(--target_episodes "$TARGET_EPISODES")
else
  # 旧版评估器没有 target_episodes；只有代码已硬编码 task 4 时才允许继续。
  if ! grep -Eq "task_ids[[:space:]]*=[[:space:]]*\[[[:space:]]*4[[:space:]]*\]|target_task_id[[:space:]]*=[[:space:]]*4|task_id[[:space:]]*=[[:space:]]*4" "$EVAL_SCRIPT"; then
    echo "当前评估器不支持 --target_episodes，且未检测到 task_id=4 过滤，已停止。" >&2
    echo "请先将评估器改为只运行 task_id=4，避免误跑完整任务套件。" >&2
    exit 3
  fi
  echo "检测到旧版 task_id=4-only 评估器，将使用其内置的 50 个状态。"
fi

COMMON_ARGS=(
  --model_family openvla
  --task_suite_name libero_spatial
  --unnorm_key libero_spatial
  --use_l1_regression True
  --use_diffusion False
  --use_film False
  --num_images_in_input 2
  --use_proprio True
  --center_crop True
  --num_open_loop_steps 8
  --num_steps_wait 10
  --num_trials_per_task 50
  --initial_state_offset 0
  "${TARGET_ARGS[@]}"
  --local_log_dir "$LOG_DIR"
  --use_wandb False
  --seed 7
)

mkdir -p "$LOG_DIR"

echo "===== 1/2 评估基座模型（权威候选） ====="
python "$EVAL_SCRIPT" \
  "${COMMON_ARGS[@]}" \
  --pretrained_checkpoint "$BASE_CKPT" \
  --run_id_note authority-base \
  2>&1 | tee "$LOG_DIR/base.log"

echo "===== 2/2 评估 head-only 模型（同状态对照） ====="
python "$EVAL_SCRIPT" \
  "${COMMON_ARGS[@]}" \
  --pretrained_checkpoint "$FINE_CKPT" \
  --run_id_note authority-head-only \
  2>&1 | tee "$LOG_DIR/head_only.log"

echo
echo "===== 配对评估汇总 ====="
for file in "$LOG_DIR/base.log" "$LOG_DIR/head_only.log"; do
  echo "--- $file"
  grep -E "Total episodes:|Total successes:|Overall success rate:|Task:|initial_state_index=" "$file" | tail -n 8 || true
  grep -q "libero_spatial_no_noops.*libero_spatial" "$file" || {
    echo "统计量检查失败: $file 未确认使用基座统计量" >&2
    exit 4
  }
done

python "$SCRIPT_DIR/compare_paired_eval_logs.py" \
  --base-log "$LOG_DIR/base.log" \
  --fine-log "$LOG_DIR/head_only.log"

echo
echo "要求：两份日志都应出现 50 个 episode，并使用基座统计量映射到 libero_spatial。"

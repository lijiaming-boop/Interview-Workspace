#!/usr/bin/env bash
# =============================================================================
# 00_env_check.sh —— 阿里云 PWI 前置自检（跑任何后续实验之前必须通过）
# 用法：bash 00_env_check.sh；训练前用 CHECK_TRAINING=1 bash 00_env_check.sh
# =============================================================================
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SELF_DIR}/_common.sh"

echo "== 1. PWI 工作目录与 Python 环境 =="
export REQUIRE_TRAINING_STACK="${CHECK_TRAINING:-0}"
require_env
printf 'workspace: %s\nrepo: %s\npython: %s\n' "${PWI_WORKSPACE}" "${OFT_ROOT}" "$(command -v python)"

echo "== 2. 关键依赖 =="
python - <<'PY'
import importlib.util
required = ["draccus", "torch", "transformers", "libero", "robosuite", "mujoco"]
missing = [m for m in required if not importlib.util.find_spec(m)]
print("  missing:", missing or "none")
raise SystemExit(1 if missing else 0)
PY

echo "== 3. Checkpoint =="
require_checkpoint
echo "  OK ${CKPT_DIR}"
if [ "${CHECK_TRAINING:-0}" = "1" ]; then
  for component in action_head--150000_checkpoint.pt proprio_projector--150000_checkpoint.pt; do
    [ -f "${CKPT_DIR}/${component}" ] || { echo "  MISSING ${CKPT_DIR}/${component}" >&2; exit 1; }
    echo "  OK ${component}"
  done
fi

echo "== 4. GPU =="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo "== 4.1 PWI 渲染与离线设置 =="
printf '  MUJOCO_GL=%s\n  PYOPENGL_PLATFORM=%s\n  HF_ENDPOINT=%s\n  WANDB_MODE=%s\n' \
  "${MUJOCO_GL}" "${PYOPENGL_PLATFORM}" "${HF_ENDPOINT}" "${WANDB_MODE}"

echo "== 5. full-patch 工具（失败则先应用 0001-Add-full-LIBERO-failure-study-pipeline.patch） =="
for f in failure_study/analyze_action_chunks.py \
         failure_study/summarize_targeted_eval.py; do
  if [ -f "${OFT_ROOT}/experiments/robot/libero/${f}" ]; then
    echo "  OK ${f}"
  else
    echo "  MISSING ${f}" >&2; exit 1
  fi
done

if grep -Eq '^[[:space:]]*attn_implementation="flash_attention_2"' \
  "${OFT_ROOT}/experiments/robot/openvla_utils.py"; then
  echo "  FAIL: openvla_utils.py 仍强制 flash_attention_2；PWI 请改为 sdpa" >&2
  exit 1
else
  echo "  OK: 未强制不兼容的 flash_attention_2"
fi

echo "== 6. runner 是否已支持抓取恢复状态机（可选的后续实现） =="
if grep -q "grasp_recovery_max_retries" "${OFT_ROOT}/experiments/robot/libero/run_libero_eval.py"; then
  echo "  verify_grasp 恢复状态机：已实现"
else
  echo "  尚未实现（03_grasp_recovery.sh 会因此无法执行，见该脚本头部说明）"
fi

echo "前置检查全部通过。"

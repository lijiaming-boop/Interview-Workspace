#!/usr/bin/env bash
# =============================================================================
# 03_grasp_recovery.sh —— 仿真侧 · 抓取确认 + 恢复状态机（233/378/230）
#
# 为什么这样做：
#   推理侧已证明两件事：(1) 仅把 open-loop 8→4 无法挽救 233/378；
#   (2) 已有 grasp_replan_enabled 在 230 上触发 19 次重规划仍失败——
#   因为那只是“清空队列重新 roll 一遍”，策略会重复同样的抓取错误。
#   因此需要真正的闭环：在“首次闭合”动作被取出时立即阻断 transfer，
#   用仿真真值确认抓取（双指接触 / 目标-夹爪距离），失败则
#   强制张开 → 重新观察 → 重新规划，最多重试 3 次。
#
# 依赖：
#   ★ 这是唯一一个需要“先写新代码”的实验。full patch 只实现了
#     --target_motion_replan_threshold_m（目标位移触发），未实现下述接口。
#     需要在 run_libero_eval.py 的 GenerateConfig 增加：
#       verify_grasp: bool = False
#       grasp_recovery_max_retries: int = 3
#       grasp_verify_hold_steps: int = 2      # 闭合后保持的步数
#     并在 action_queue 取出首位动作满足“首次闭合”时，依据 collect_sim_snapshot
#     的 contacts（双指 vs 目标几何体）与 gripper-target 距离判定是否抓稳。
#
# 用法：bash 03_grasp_recovery.sh  （实现就绪后运行；未实现则给出差异化提示）
# 验收：233/378 在 verify_grasp+retry 下成功率与“空抓后仍执行搬运”比例。
# =============================================================================
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SELF_DIR}/_common.sh"
require_full_patch; require_checkpoint; require_env

if ! grep -q "grasp_recovery_max_retries" "${OFT_ROOT}/experiments/robot/libero/run_libero_eval.py"; then
  echo "!! run_libero_eval.py 尚未实现 'grasp_recovery_max_retries'（恢复状态机）。" >&2
  echo "   请按本脚本头部'为什么/依赖'一节在 full-patch 基础上补实现后再运行。" >&2
  exit 1
fi

MODE="${1:-all}"
TARGETS="${TARGETS:-4:32,7:27,4:29}"
grasp_eval() { # label targets retries
  local label="$1" targets="$2" retries="$3"
  local run_root="${RESULT_BASE}/${label}"
  mkdir -p "${run_root}"
  (
    cd "${OFT_ROOT}"
    python experiments/robot/libero/run_libero_eval.py \
      --pretrained_checkpoint "${CKPT_DIR}" \
      --task_suite_name libero_spatial --seed 7 \
      --num_open_loop_steps 8 --target_episodes "${targets}" \
      --dump_action_chunks True --action_dump_dir "${run_root}/actions" \
      --verify_grasp True --grasp_recovery_max_retries "${retries}" \
      --run_id_note "${label}"
  ) 2>&1 | tee "${run_root}/console.log"
  python "${OFT_ROOT}/experiments/robot/libero/failure_study/analyze_action_chunks.py" \
    --root "${run_root}/actions" --output-dir "${run_root}/analysis"
  echo "[OK] ${label}: ${run_root}"
}

case "${MODE}" in
  no_retry) grasp_eval "grasp-verify-noretry" "${TARGETS}" 0 ;;   # 只检测不恢复
  retry3)   grasp_eval "grasp-verify-retry3"   "${TARGETS}" 3 ;;   # 完整恢复状态机
  all)
    grasp_eval "grasp-verify-noretry" "${TARGETS}" 0
    grasp_eval "grasp-verify-retry3"   "${TARGETS}" 3
    ;;
  *) echo "Usage: $0 {no_retry|retry3|all}" >&2; exit 2 ;;
esac

echo "========== 验收 =========="
cat <<'EOF'
1) 成功率对比（应为 retry3 >>> no_retry，且 233/378 不再出现“空夹爪进盘子”）：
   grep -h "Success:" ${RESULT_BASE}/grasp-verify-*/console.log
2) 恢复事件：
   grep -h "grasp_verified\|empty_grasp_forced_replan\|recovery_retry" \
     ${RESULT_BASE}/grasp-verify-retry3/actions/*/*/events.jsonl | head -20
3) 关键安全指标：任何 episode 不得在 verify 失败时仍执行搬运轨迹
   （chunks.jsonl 中 transfer 段的 gripper 必须已通过验证）。
EOF

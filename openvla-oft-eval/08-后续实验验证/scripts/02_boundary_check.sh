#!/usr/bin/env bash
# =============================================================================
# 02_boundary_check.sh —— 仿真侧 · action-chunk 边界检查（233/378）
#
# 为什么这样做：
#   报告证据：233/378 在“空抓”后仍继续执行了完整的搬运/放置轨迹，与
#   一次输出 8 步、开环执行的 action-chunk 机制吻合。所以先确认 grasp→transfer
#   切换到底发生在 chunk 内部还是 chunk 边界——这决定“固定边界插入检测”
#   是否可行。已知旧日志提示 233 闭合在 c5/a1、378 在 c4/a6（chunk 内部），
#   需要用新实现重新确证（旧日志的 chunks.jsonl 已不在工作区）。
#
# 验证内容：
#   保存每个 chunk 的 8×7 原始/处理 action + gripper 命令 + 特权 sim 快照，
#   由 analyze_action_chunks.py 输出 first_close_* / first_transfer_* /
#   boundary_check_before_transfer。
# 验收标准：
#   - boundary_check_before_transfer 对 233/378 都是 False（闭合在 chunk 内部）
#     → “仅在边界插检测”方案证伪，转向 03 的“首次闭合时阻断”方案；
#   - 若出现 True → 边界方案可行，先做边界检测消融。
# 用法：bash 02_boundary_check.sh [boundary8|boundary4|all]
# =============================================================================
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SELF_DIR}/_common.sh"
require_full_patch; require_checkpoint; require_env

MODE="${1:-all}"
case "${MODE}" in
  boundary8) study_run_eval "sim-boundary-chunk8" "4:32,7:27" 8 1.0 True True 0.0 20 ;;
  boundary4) study_run_eval "sim-boundary-chunk4" "4:32,7:27" 4 1.0 True True 0.0 20 ;;
  all)
    study_run_eval "sim-boundary-chunk8" "4:32,7:27" 8 1.0 True True 0.0 20
    study_run_eval "sim-boundary-chunk4" "4:32,7:27" 4 1.0 True True 0.0 20
    ;;
  *) echo "Usage: $0 {boundary8|boundary4|all}" >&2; exit 2 ;;
esac

echo "========== 验收 =========="
cat <<EOF
逐个检查分析汇总：
  ls ${RESULT_BASE}/sim-boundary-*/analysis/
  grep -h "boundary_check_before_transfer" ${RESULT_BASE}/sim-boundary-*/analysis/*/*.csv 2>/dev/null || true
人工核对原始 chunk（8×7 + gripper）：
  ${RESULT_BASE}/sim-boundary-*/actions/task_04/state_32/chunks.jsonl
  ${RESULT_BASE}/sim-boundary-*/actions/task_07/state_27/chunks.jsonl
判定：若 first_close 的 chunk/action 与搬运发生位置重合于同一 chunk（非 action 0），
则边界检测不可行 → 执行 03_grasp_recovery.sh。
EOF
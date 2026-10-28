#!/usr/bin/env bash
# =============================================================================
# 04_relocalize.sh —— 仿真侧 · 推移后重新定位（321）
#
# 为什么这样做：
#   321 的证据：夹爪与碗接触后没有稳定抬升，反而把碗推向 cookie box
#   （横向接近误差/闭合时机错误使抓取退化为推挤），之后策略没有因目标
#   位姿改变而重新规划。这里利用仿真特权信号检测“碗被推移超过阈值”，
#   在夹爪处于张开状态时丢弃剩余 chunk 动作，用新观测重新规划（新观测
#   才能重新估计目标中心）。配合 open-loop=4 缩短“接触→重规划”延迟。
#
# 实施：仿真特权消融开关 --target_motion_replan_threshold_m
#   0（对照组，不开） vs 0.01（碗位移 >1cm 即重规划）。
# 默认每组跑 10 个配对重复；每个重复使用同一个 seed，控制组和处理组只改变阈值。
# 用法：bash 04_relocalize.sh [relocalize_control|relocalize_motion|all]
#       REPEATS=10 SEED_START=7 RUN_TAG=321-relocalize-20260910 bash 04_relocalize.sh all
# 验收：每组 ≥10 个完整 trial，配对配置一致；控制组无 forced replan；处理组所有事件
#       都满足“开爪 + 位移≥1cm + 丢弃剩余动作”，且至少触发 1 次。只有处理组成功率
#       高于对照组并出现成功翻转，才可下“重定位恢复 321”的结论。
# =============================================================================
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SELF_DIR}/_common.sh"
require_full_patch; require_checkpoint; require_env

MODE="${1:-all}"
VERIFY_SCRIPT="${SELF_DIR}/verify_relocalize_321.py"
REPEATS="${REPEATS:-10}"
SEED_START="${SEED_START:-7}"

[[ "${REPEATS}" =~ ^[0-9]+$ ]] && [ "${REPEATS}" -ge 10 ] || {
  echo "REPEATS 必须是大于等于 10 的整数，当前为: ${REPEATS}" >&2
  exit 2
}
[[ "${SEED_START}" =~ ^[0-9]+$ ]] || {
  echo "SEED_START 必须是非负整数，当前为: ${SEED_START}" >&2
  exit 2
}

run_group() {
  local group="$1" threshold="$2"
  local label="sim-relocalize-${group}"
  local group_root="${RESULT_BASE}/${label}"
  mkdir -p "${group_root}"
  for ((trial=1; trial<=REPEATS; trial++)); do
    local trial_id seed trial_root
    trial_id="$(printf '%02d' "${trial}")"
    seed=$((10#${SEED_START} + trial - 1))
    trial_root="${group_root}/trial-${trial_id}"
    if [ -e "${trial_root}" ] && [ "${ALLOW_EXISTING_TRIALS:-0}" != "1" ]; then
      echo "已有结果目录，拒绝覆盖: ${trial_root}" >&2
      echo "如需明确覆盖本 RUN_TAG，请设置 ALLOW_EXISTING_TRIALS=1。" >&2
      exit 3
    fi
    echo "===== ${label} trial ${trial}/${REPEATS} (seed=${seed}, threshold=${threshold}m) ====="
    study_run_eval "${label}-trial-${trial_id}" "6:20" 4 1.0 True True "${threshold}" 20 "${seed}" "${trial_root}"
  done
}

case "${MODE}" in
  relocalize_control) run_group control 0.0 ;;
  relocalize_motion)  run_group motion 0.01 ;;
  all)
    run_group control 0.0
    run_group motion 0.01
    python "${VERIFY_SCRIPT}" \
      --control-root "${RESULT_BASE}/sim-relocalize-control" \
      --treatment-root "${RESULT_BASE}/sim-relocalize-motion" \
      --threshold 0.01 \
      --min-trials "${REPEATS}"
    ;;
  *) echo "Usage: $0 {relocalize_control|relocalize_motion|all}" >&2; exit 2 ;;
esac

if [ "${MODE}" != "all" ]; then
  echo "单组运行完成。两组都完成后执行："
  echo "python ${VERIFY_SCRIPT} --control-root ${RESULT_BASE}/sim-relocalize-control --treatment-root ${RESULT_BASE}/sim-relocalize-motion --threshold 0.01 --min-trials ${REPEATS}"
fi

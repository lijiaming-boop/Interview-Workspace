#!/usr/bin/env bash
# 基座模型抽屉失败诊断：状态扫描 -> 动作记录 -> action chunk 消融。
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SELF_DIR}/_common.sh"
require_full_patch
require_checkpoint
require_env

TRIALS="${TRIALS:-50}"
RUN_TAG="${RUN_TAG:-base-drawer-diagnosis-$(date '+%Y%m%d-%H%M%S')}"
export RUN_TAG

if ! [[ "${TRIALS}" =~ ^[0-9]+$ ]] || [ "${TRIALS}" -lt 1 ] || [ "${TRIALS}" -gt 50 ]; then
  echo "TRIALS 必须是 1~50 的整数" >&2
  exit 2
fi

echo "===== 1/2 基座模型扫描 top drawer 的 ${TRIALS} 个状态 ====="
TRIALS="${TRIALS}" bash "${SELF_DIR}/06_top_drawer_sweep.sh"

SWEEP_DIR="${RESULT_BASE}/training-top-drawer-sweep-${TRIALS}"
RESULTS_CSV="${SWEEP_DIR}/summary/targeted_results.csv"
test -f "${RESULTS_CSV}" || {
  echo "找不到状态扫描结果: ${RESULTS_CSV}" >&2
  exit 4
}

FAILED_TARGETS="$(python - "${RESULTS_CSV}" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
failed = [row["initial_state_index"] for row in rows if row["success"].strip().lower() == "false"]
print(",".join(f"4:{index}" for index in failed))
print(f"[info] failed states: {failed}", file=sys.stderr)
PY
)"

if [ -z "${FAILED_TARGETS}" ]; then
  echo "基座扫描没有失败状态，不执行后续消融。"
  exit 0
fi

echo "===== 2/2 对基座失败状态做 action chunk 消融 ====="
echo "目标状态: ${FAILED_TARGETS}"
for OPEN_LOOP in 8 4 1; do
  study_run_eval \
    "base-drawer-failures-chunk${OPEN_LOOP}" \
    "${FAILED_TARGETS}" \
    "${OPEN_LOOP}" \
    1.0 \
    True \
    True \
    0.0 \
    20
done

echo
echo "========== 诊断产物 =========="
echo "状态成败: ${RESULTS_CSV}"
echo "动作与仿真快照: ${RESULT_BASE}/base-drawer-failures-chunk*/actions/"
echo "动作汇总: ${RESULT_BASE}/base-drawer-failures-chunk*/analysis/"
echo
echo "判定："
echo "  chunk=1 成功而 chunk=8 失败 -> 优先修复闭环/action chunk"
echo "  三种 chunk 都失败 -> 继续检查 raw/processed action、坐标和抓取姿态"
echo "  之后再按需执行：bash ${SELF_DIR}/03_grasp_recovery.sh retry3"

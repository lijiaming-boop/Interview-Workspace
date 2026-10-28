#!/usr/bin/env bash
# =============================================================================
# 06_top_drawer_sweep.sh —— 训练侧 · top drawer 全状态扫描（230/233）
#
# 为什么这样做：
#   两个失败都发生在 top drawer（task_id=4），且都是抓取阶段失败。先假设
#   失败并非随机：可能集中在特定碗位姿（抽屉边缘/遮挡/高度）。把 task 4
#   的 50 个初始状态全部扫一遍，得到“成功率随 state 分布”，找出失败状态集
#   ——这些状态就是后续 DAgger 要补专家示教的 target 集合。
#
# 实施：TRIALS 指定扫描多少个 state（默认 50），产物自动生成 4:0..N-1
#   targeted 列表；结束后用 summarize_targeted_eval.py 生成：
#     targeted_results.csv（state 级成败）
#     dagger_collection_manifest.csv（哪些 state 需要专家修正 → 07 的输入）
# 用法：TRIALS=50 bash 06_top_drawer_sweep.sh
# 验收：拿到 state 级成败清单；若失败集中在少数 state → 收集其 expert demo。
# =============================================================================
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SELF_DIR}/_common.sh"
require_full_patch; require_checkpoint; require_env

TRIALS="${TRIALS:-50}"
if ! [[ "${TRIALS}" =~ ^[0-9]+$ ]] || [ "${TRIALS}" -lt 1 ] || [ "${TRIALS}" -gt 50 ]; then
  echo "TRIALS 必须是 1~50 的整数" >&2; exit 2
fi

TARGETS="$(seq -s, 0 $((TRIALS - 1)) | sed 's/^/4:/; s/,/,4:/g')"
LABEL="training-top-drawer-sweep-${TRIALS}"
study_run_eval "${LABEL}" "${TARGETS}" 8 1.0 False True 0.0 20

python "${OFT_ROOT}/experiments/robot/libero/failure_study/summarize_targeted_eval.py" \
  --log "${RESULT_BASE}/${LABEL}/console.log" \
  --output-dir "${RESULT_BASE}/${LABEL}/summary"

echo "========== 验收 =========="
cat <<EOF
1) 成败统计：
   cat ${RESULT_BASE}/${LABEL}/summary/targeted_results.csv
2) 失败是否集中于特定 state（可画直方图）：
   python - <<'PY'
import csv, collections
rows = list(csv.DictReader(open("${RESULT_BASE}/${LABEL}/summary/targeted_results.csv", encoding="utf-8-sig")))
fails = [r["initial_state_index"] for r in rows if r["success"] == "False"]
print("states:", len(rows), "failures:", len(fails), "->", fails)
PY
3) DAgger 收集清单（07 的输入）：
   cat ${RESULT_BASE}/${LABEL}/summary/dagger_collection_manifest.csv
EOF
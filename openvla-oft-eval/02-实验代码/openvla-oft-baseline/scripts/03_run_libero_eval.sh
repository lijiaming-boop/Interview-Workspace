#!/usr/bin/env bash
# =============================================================================
# 03_run_libero_eval.sh —— 运行 OpenVLA-OFT 官方 LIBERO 评测
#
# 用法:
#   bash 03_run_libero_eval.sh <spatial|object|goal|10|all> [trials_per_task] [seed]
#     trials_per_task 默认 50（官方设置：10 任务 × 50 = 500 episodes/suite）
#     seed 默认 7（官方示例值；论文为 3 个 seed 平均）
#
# 环境变量:
#   SMOKE=1        冒烟模式：每任务 2 条（20 episodes，先在前台跑通全链路）
#   CKPT=combined  用四合一 checkpoint 跑当前 suite（磁盘紧张时的省盘方案）
#   GPU=1          指定 GPU 序号（默认 0）
#
# 重要：
#   - 首次运行 LIBERO 会交互式询问"数据集保存路径"（只问一次，写入
#     ~/.libero/config.yaml）。务必先在前台跑一次 SMOKE，再使用 nohup 后台跑。
#   - 官方脚本每次 episode 都会保存视频到  $OFT_ROOT/rollouts/<日期>/，
#     日志在 $OFT_ROOT/experiments/logs/EVAL-<suite>-openvla-<时间>.txt
#   - 本脚本会把每次运行记录追加到 results/runs.jsonl（供 04/05 脚本汇总）
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

SUITE="${1:?用法: 03_run_libero_eval.sh <spatial|object|goal|10|all> [trials_per_task] [seed]}"
TRIALS="${2:-50}"
SEED="${3:-7}"
[ "${SMOKE:-0}" = "1" ] && TRIALS=2

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export MUJOCO_GL=egl          # 无头渲染走 EGL（NVIDIA 驱动自带）
export PYOPENGL_PLATFORM=egl
export HF_ENDPOINT="${HF_MIRROR}"   # 防 transformers 偶发联网校验超时（本地 ckpt 通常不触发）

case "$SUITE" in
  all) SUITES="spatial object goal 10" ;;
  spatial|object|goal|10) SUITES="$SUITE" ;;
  *) echo "未知 suite: ${SUITE}（可选 spatial|object|goal|10|all）"; exit 1 ;;
esac

activate_env
mkdir -p "${RESULTS_DIR}"

run_one() {
  local s="$1"
  local suite_name="libero_${s}"
  local ckpt_dir
  if [ "${CKPT:-}" = "combined" ]; then
    ckpt_dir="${CKPT_ROOT}/$(suite2ckpt combined)"
  else
    ckpt_dir="${CKPT_ROOT}/$(suite2ckpt "$s")"
  fi
  if [ ! -d "${ckpt_dir}" ]; then
    echo "!! 缺 checkpoint 目录 ${ckpt_dir}（先运行 02_download_ckpts.sh）"
    FAILED="${FAILED} ${s}"
    return 0
  fi

  local t0=${SECONDS}
  echo "=================================================================="
  echo "==> [$(date '+%F %T')] 开始 ${suite_name}"
  echo "    checkpoint : ${ckpt_dir}"
  echo "    trials/task: ${TRIALS}   seed: ${SEED}   GPU: ${CUDA_VISIBLE_DEVICES}"
  echo "=================================================================="

  if ( cd "${OFT_ROOT}" && mkdir -p experiments/logs rollouts && \
       python experiments/robot/libero/run_libero_eval.py \
         --pretrained_checkpoint "${ckpt_dir}" \
         --task_suite_name "${suite_name}" \
         --num_trials_per_task "${TRIALS}" \
         --seed "${SEED}" ); then
    # 找到刚生成的日志（run_id = EVAL-<suite>-openvla-<DATE_TIME>.txt）
    local log
    log="$(cd "${OFT_ROOT}" && ls -t "experiments/logs/EVAL-${suite_name}-openvla-"*.txt 2>/dev/null | head -1 || true)"
    printf '{"ts":"%s","date":"%s","suite":"%s","ckpt":"%s","trials":%s,"seed":%s,"log":"%s/%s","smoke":%s}\n' \
      "$(date '+%F_%T')" "$(date '+%F')" "${s}" "$(basename "${ckpt_dir}")" \
      "${TRIALS}" "${SEED}" "${OFT_ROOT}" "${log}" \
      "$([ "${SMOKE:-0}" = "1" ] && echo true || echo false)" \
      >> "${RESULTS_DIR}/runs.jsonl"
    echo "==> ${suite_name} 完成，用时 $((SECONDS - t0))s"
    [ -n "${log}" ] && echo "    日志: ${OFT_ROOT}/${log}" || true
    echo "    全量耗时外推: 用时 × $(( 50 / TRIALS > 0 ? 50 / TRIALS : 1 )) 倍（每 suite 500 episodes）"
  else
    echo "!! ${suite_name} 评测失败（详见上方报错；常见原因见 docs/01 排查表）"
    FAILED="${FAILED} ${s}"
  fi
}

FAILED=""
for s in ${SUITES}; do
  run_one "$s" || true
done

echo ""
if [ -n "${FAILED}" ]; then
  echo "!! 以下 suite 未成功:${FAILED}"
  exit 1
fi
echo "==> 全部完成。汇总表格: python ${SCRIPT_DIR}/04_aggregate_results.py"
echo "==> 失败清单:      python ${SCRIPT_DIR}/05_collect_failure_cases.py"

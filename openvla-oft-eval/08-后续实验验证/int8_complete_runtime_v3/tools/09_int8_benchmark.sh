#!/usr/bin/env bash
# =============================================================================
# 09_int8_benchmark.sh —— OFT 并行解码：BF16 vs INT8 主干 + BF16 动作头
#
# 默认快速集：5 个已知困难状态，各精度运行一遍。
# 完整成功率评测：TARGETS="" bash 09_int8_benchmark.sh all（LIBERO-Spatial 500 episodes）。
#
# OFT 一次 forward 并行预测 NUM_ACTIONS_CHUNK=8 个 7D 动作，而不是逐动作自回归；
# num_open_loop_steps=8 会依次执行这 8 个动作，再读取新观测。汇总同时报告：
#   policy_query_fps = 每秒模型 forward 次数
#   action_fps       = 每秒可生成的动作数（8 / 平均 forward 延迟）
#
# INT8 只传给 VLA 主模型；外置 action_head / proprio_projector 按现有 OFT 代码保持 BF16。
# =============================================================================
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SELF_DIR}/_common.sh"
require_full_patch
require_checkpoint
require_env

MODE="${1:-all}"
TARGETS="${TARGETS-4:29,4:32,5:35,6:20,7:27}"
OPEN_LOOP="${OPEN_LOOP:-8}"
SEED="${SEED:-7}"
CONTROL_FREQ="${CONTROL_FREQ:-20}"
GPU_SAMPLE_SECONDS="${GPU_SAMPLE_SECONDS:-0.2}"
SUMMARY_SCRIPT="${SELF_DIR}/summarize_int8_benchmark.py"

[[ "${OPEN_LOOP}" =~ ^[0-9]+$ ]] && [ "${OPEN_LOOP}" -ge 1 ] && [ "${OPEN_LOOP}" -le 8 ] || {
  echo "OPEN_LOOP 必须是 1..8，当前为 ${OPEN_LOOP}" >&2
  exit 2
}
[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "SEED 必须是非负整数" >&2; exit 2; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "缺少 nvidia-smi" >&2; exit 2; }
grep -q 'load_in_8bit' "${OFT_ROOT}/experiments/robot/libero/run_libero_eval.py" || {
  echo "当前 run_libero_eval.py 不支持 --load_in_8bit" >&2
  exit 2
}
grep -Eq 'action_head[[:space:]]*=[[:space:]]*action_head\.to\(torch\.bfloat16\)' \
  "${OFT_ROOT}/experiments/robot/openvla_utils.py" || {
  echo "无法确认外置 action_head 保持 BF16；为避免误量化已停止" >&2
  exit 2
}
python - <<'PY'
import bitsandbytes as bnb
import torch
print(f"[OK] bitsandbytes={bnb.__version__}; torch={torch.__version__}; GPU={torch.cuda.get_device_name(0)}")
PY

TARGET_ARGS=()
if [ -n "${TARGETS}" ]; then
  TARGET_ARGS=(--target_episodes "${TARGETS}")
fi

MONITOR_PID=""
stop_monitor() {
  if [ -n "${MONITOR_PID}" ] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  MONITOR_PID=""
}
trap stop_monitor EXIT INT TERM

start_monitor() {
  local output_file="$1"
  printf 'timestamp_ms,memory_used_mib,gpu_util_percent\n' > "${output_file}"
  (
    while true; do
      local_now="$(date +%s%3N)"
      sample="$(nvidia-smi --id="${GPU_ID}" \
        --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
      printf '%s,%s\n' "${local_now}" "${sample}" >> "${output_file}"
      sleep "${GPU_SAMPLE_SECONDS}"
    done
  ) &
  MONITOR_PID="$!"
}

run_eval() {
  local label="$1" load_in_8bit="$2"
  local run_root="${RESULT_BASE}/${label}"
  if [ -e "${run_root}" ]; then
    echo "结果目录已存在，拒绝覆盖: ${run_root}" >&2
    exit 3
  fi
  mkdir -p "${run_root}"
  printf 'label=%s\nprecision=%s\nload_in_8bit=%s\ntask_suite=libero_spatial\nseed=%s\ntargets=%s\nopen_loop=%s\ncontrol_freq=%s\nchunk_size=8\ncheckpoint=%s\naction_head_dtype=bf16\n' \
    "${label}" "${label}" "${load_in_8bit}" "${SEED}" "${TARGETS:-ALL}" \
    "${OPEN_LOOP}" "${CONTROL_FREQ}" "${CKPT_DIR}" > "${run_root}/run_config.txt"

  echo "GPU 上若有其他进程，显存采样会被污染："
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
  start_monitor "${run_root}/gpu_memory.csv"

  set +e
  (
    cd "${OFT_ROOT}"
    PYTHONUNBUFFERED=1 python experiments/robot/libero/run_libero_eval.py \
      --pretrained_checkpoint "${CKPT_DIR}" \
      --task_suite_name libero_spatial \
      --seed "${SEED}" \
      --num_open_loop_steps "${OPEN_LOOP}" \
      "${TARGET_ARGS[@]}" \
      --load_in_8bit "${load_in_8bit}" \
      --load_in_4bit False \
      --dump_camera_inputs False \
      --dump_action_chunks True \
      --action_dump_dir "${run_root}/actions" \
      --target_motion_replan_threshold_m 0.0 \
      --control_freq "${CONTROL_FREQ}" \
      --use_wandb False \
      --run_id_note "${label}"
  ) 2>&1 | tee "${run_root}/console.log"
  eval_status="${PIPESTATUS[0]}"
  set -e
  stop_monitor
  if [ "${eval_status}" -ne 0 ]; then
    echo "${label} 运行失败，日志：${run_root}/console.log" >&2
    exit "${eval_status}"
  fi
  echo "[OK] ${label}: ${run_root}"
}

case "${MODE}" in
  bf16) run_eval bf16 False ;;
  int8) run_eval int8-mixed True ;;
  all)
    run_eval bf16 False
    run_eval int8-mixed True
    python "${SUMMARY_SCRIPT}" \
      --bf16-root "${RESULT_BASE}/bf16" \
      --int8-root "${RESULT_BASE}/int8-mixed" \
      --warmup-chunks 3 \
      --output-dir "${RESULT_BASE}/summary"
    ;;
  *) echo "Usage: $0 {bf16|int8|all}" >&2; exit 2 ;;
esac

if [ "${MODE}" != "all" ]; then
  echo "单组完成；两组结果必须位于同一 RUN_TAG 后再运行 summarize_int8_benchmark.py。"
fi

#!/usr/bin/env bash
# ============================================================================
# memwatch.sh —— 内存哨兵：在 nohang 动手之前先看到水位
# ----------------------------------------------------------------------------
# 为什么要它：
#   本实例 nohang 的触发线是「空闲内存 < 2% (≈573 MiB) → 杀整个进程组」，
#   它自己的日志拿不到（PSI 缺失、无 nohang.log）。跑基准时若内存悄悄爬升，
#   你只会看到进程突然消失。本脚本把水位持续写到日志，出事后可回溯。
#
# 用法：
#   bash memwatch.sh &                                  # 一直跑，Ctrl-C 或 kill 停
#   LOG=/mnt/workspace/_mem.log bash memwatch.sh &
#   SECS=600 bash memwatch.sh &                         # 只跑 10 分钟
#   bash memwatch.sh --once                             # 只打一行当前状态
#
# 日志默认: /mnt/workspace/_memwatch.log
# 另写摘要: /mnt/workspace/_memwatch.summary.log（只记峰值与越线事件）
# ============================================================================
set -uo pipefail

_SELF="${0##*/}"
_PASTED=0
case "${_SELF}" in bash|-bash|sh|-sh|dash|-dash|zsh|-zsh) _PASTED=1 ;; esac
if [ "${_PASTED}" = "1" ]; then
  echo "[注意] 本脚本被粘贴进交互终端。建议存成文件后执行: bash memwatch.sh &"
  echo
fi

INTERVAL="${INTERVAL:-5}"
SECS="${SECS:-0}"          # 0 = 无限
LOG="${LOG:-/mnt/workspace/_memwatch.log}"
SUM="${SUM:-/mnt/workspace/_memwatch.summary.log}"

# nohang 阈值换算成 MiB（基于 total 动态计算）
TOTAL_MB=$(awk '/MemTotal/{print int($2/1024)}' /proc/meminfo)
WARN_MB=$(( TOTAL_MB * 40 / 100 ))     # 40% 空闲 → nohang 发警告
SOFT_MB=$(( TOTAL_MB * 3  / 100 ))     # 3%  空闲 → 软动作
HARD_MB=$(( TOTAL_MB * 2  / 100 ))     # 2%  空闲 → 硬动作（杀进程组）

oneline() {
  awk -v ts="$(date '+%F %T')" \
      -v total="${TOTAL_MB}" \
      -v warn="${WARN_MB}" -v soft="${SOFT_MB}" -v hard="${HARD_MB}" '
    /MemAvailable/{a=$2/1024} /MemFree/{f=$2/1024}
    /SwapTotal/{st=$2/1024} /SwapFree/{sf=$2/1024}
    END{
      pct = a/total*100
      printf "%s  avail=%6.0f MiB (%5.1f%%)  free=%6.0f  swap_free=%6.0f  | warn<%d soft<%d hard<%d\n",
             ts, a, pct, f, sf, warn, soft, hard
    }' /proc/meminfo
}

top_procs() {
  echo "---- top RSS ----"
  ps -eo pid,ppid,rss,etime,cmd --sort=-rss 2>/dev/null | head -8
  echo "---- 完 ----"
}

if [ "${1:-}" = "--once" ]; then
  oneline
  exit 0
fi

{
  echo "================================================"
  echo " memwatch 启动 $(date '+%F %T')"
  echo " total=${TOTAL_MB} MiB  warn(40%)=${WARN_MB} soft(3%)=${SOFT_MB} hard(2%)=${HARD_MB} MiB"
  echo " interval=${INTERVAL}s  seconds=${SECS:-inf}"
  echo "================================================"
} | tee -a "$LOG"

{
  echo "=== memwatch session $(date '+%F %T') total=${TOTAL_MB}MiB ==="
  echo "阈值: warn(40%)=${WARN_MB}MiB soft(3%)=${SOFT_MB}MiB hard(2%)=${HARD_MB}MiB"
} >> "$SUM"

MIN_AVAIL_PCT=100
MIN_AVAIL_MB=999999
START=$(date +%s)
LAST_LEVEL=""

while :; do
  LINE="$(oneline)"
  echo "$LINE" >> "$LOG"

  AV=$(echo "$LINE" | sed -n 's/.*avail= *\([0-9]*\) .*/\1/p')
  PCT=$(echo "$LINE" | sed -n 's/.*( *\([0-9.]*\)%).*/\1/p')

  if [ -n "${AV:-}" ]; then
    [ "${AV}" -lt "${MIN_AVAIL_MB}" ] 2>/dev/null && MIN_AVAIL_MB="${AV}"
    LEVEL=""
    if   [ "${AV}" -lt "${HARD_MB}" ]; then LEVEL="HARD"
    elif [ "${AV}" -lt "${SOFT_MB}" ]; then LEVEL="SOFT"
    elif [ "${AV}" -lt "${WARN_MB}" ]; then LEVEL="WARN"
    fi

    # 只在进入某级别时输出一次，避免刷屏
    if [ -n "${LEVEL}" ] && [ "${LEVEL}" != "${LAST_LEVEL}" ]; then
      echo "" | tee -a "$LOG" "$SUM" >/dev/null
      {
        echo "### [${LEVEL}] $(date '+%F %T')  空闲 ${AV} MiB (${PCT}%)"
        if [ "${LEVEL}" = "HARD" ]; then
          echo "    !! 已低于 nohang 硬阈值 ${HARD_MB} MiB —— 随时可能杀掉整个进程组 !!"
        elif [ "${LEVEL}" = "SOFT" ]; then
          echo "    已低于 nohang 软阈值 ${SOFT_MB} MiB —— 接近悬崖"
        else
          echo "    已低于警告线 ${WARN_MB} MiB"
        fi
      } | tee -a "$LOG" "$SUM" >/dev/null
      top_procs | tee -a "$LOG" "$SUM" >/dev/null
      LAST_LEVEL="${LEVEL}"
    elif [ -z "${LEVEL}" ]; then
      LAST_LEVEL=""
    fi
  fi

  if [ "${SECS}" -gt 0 ] 2>/dev/null; then
    NOW=$(date +%s)
    [ $(( NOW - START )) -ge "${SECS}" ] && break
  fi
  sleep "${INTERVAL}"
done

{
  echo ""
  echo "=== memwatch 结束 $(date '+%F %T') ==="
  echo "全程最低空闲内存: ${MIN_AVAIL_MB} MiB"
  oneline
} | tee -a "$LOG" "$SUM"

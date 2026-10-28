#!/usr/bin/env bash
# ============================================================================
# run_detached.sh —— 让命令彻底脱离终端运行，输出落盘，终端死了也不受影响
# ----------------------------------------------------------------------------
# 原理：setsid 新建会话 + nohup 忽略 SIGHUP + stdin 接 /dev/null，
#       三者齐备后，终端关闭 / Web IDE 断连 / SSH 断开都不会杀掉进程。
#
# 用法（务必存成文件后执行，不要整段粘贴！）：
#   bash run_detached.sh <标签> <命令...>
#
# 例子：
#   bash run_detached.sh diag  bash diag_oom.sh
#   bash run_detached.sh ckpt  bash experiments/robot/libero/failure_study/verify_oft_step.sh checkpoint
#   bash run_detached.sh smoke bash tools/09_int8_benchmark.sh all
#
# 日志落点：/mnt/workspace/_run_<标签>_<时间戳>.log（NAS 盘，实例重启仍在）
# 另写一份固定名软链：/mnt/workspace/_run_<标签>.latest.log（方便 tail -f）
#
# ⚠️ 本脚本的全部逻辑都包在 main() 里，只有「作为文件运行」时才会调用。
#    若被整段粘贴进交互终端，只会定义函数、打印提示，然后干净返回，
#    **不会执行 exit，因此不会关闭你的终端**。
# ============================================================================
set -uo pipefail

# ---- 粘贴检测：作为脚本运行时 $0 是文件名；粘贴时是 bash/-bash/sh ----
SELF_NAME="${0##*/}"
_PASTED=0
case "${SELF_NAME}" in
  bash|-bash|sh|-sh|dash|-dash|zsh|-zsh) _PASTED=1 ;;
esac

main() {
  if [ $# -lt 2 ]; then
    echo "用法: bash run_detached.sh <标签> <命令...>"
    echo "例子: bash run_detached.sh diag bash diag_oom.sh"
    exit 2
  fi

  local TAG="$1"; shift
  local RUN_DIR="${RUN_DIR:-/mnt/workspace}"
  local TS; TS="$(date +%Y%m%d-%H%M%S)"
  local LOG="${RUN_DIR}/_run_${TAG}_${TS}.log"
  local LATEST="${RUN_DIR}/_run_${TAG}.latest.log"

  mkdir -p "${RUN_DIR}"

  # 用 setsid 开新会话，完全脱离当前终端的控制
  setsid nohup bash -c '
    echo "================================================================"
    echo " 标签    : '"${TAG}"'"
    echo " 启动    : $(date "+%F %T %Z")"
    echo " 工作目录: $(pwd)"
    echo " 命令    : $*"
    echo " PID     : $$"
    echo "================================================================"
    echo
    START=$(date +%s)
    "$@"
    RC=$?
    END=$(date +%s)
    echo
    echo "================================================================"
    echo " 退出码  : ${RC}"
    echo " 耗时    : $(( END - START )) 秒"
    echo " 结束    : $(date "+%F %T %Z")"
    echo "================================================================"
    exit ${RC}
  ' _ "$@" < /dev/null > "${LOG}" 2>&1 &

  local DETACHED_PID=$!

  # 等日志文件出现（最多 3 秒）
  local i
  for i in $(seq 1 30); do
    [ -f "${LOG}" ] && break
    sleep 0.1
  done

  ln -sf "${LOG}" "${LATEST}" 2>/dev/null || cp -f "${LOG}" "${LATEST}" 2>/dev/null || true

  cat <<EOF

已在后台脱离终端启动。

  日志      : ${LOG}
  固定软链  : ${LATEST}
  调度进程  : ${DETACHED_PID}

现在可以安全关闭终端。重连后查看结果：

  tail -f ${LATEST}
  cat     ${LATEST}
  grep -nE "FAIL|ERROR|Traceback|退出码" ${LATEST}

EOF
}

if [ "${_PASTED}" = "1" ]; then
  echo "================================================================"
  echo " [未执行] 检测到本脚本被【粘贴】进交互终端（\$0=${0}），"
  echo "          而不是作为文件运行。"
  echo
  echo " 粘贴执行有风险：脚本内的 exit 会直接关闭你的终端。"
  echo " 本脚本虽然已做了保护，但请改用正确方式："
  echo
  echo "   1) 先把 run_detached.sh 上传到 /mnt/workspace/"
  echo "   2) 再执行:  bash /mnt/workspace/run_detached.sh <标签> <命令...>"
  echo "================================================================"
else
  main "$@"
fi

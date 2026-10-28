#!/usr/bin/env bash
# ============================================================================
# diag_oom.sh —— 内存/磁盘/nohang 全量诊断，结果写文件（终端死了也不丢）
# ----------------------------------------------------------------------------
# 用法（推荐 detached 运行，见 run_detached.sh）：
#   bash diag_oom.sh                       # 输出到屏幕
#   bash diag_oom.sh > /mnt/workspace/_diag_oom.log 2>&1   # 落盘
#   bash run_detached.sh diag bash diag_oom.sh             # 脱离终端 + 自动落盘
#
# 设计约束（重要）：本脚本**不使用** `exec > >(tee ...)` 这类进程替换，
# 因为一旦被整段粘贴进交互终端，它会劫走整个终端的 stdout，
# 表现为「终端异常退出」。所有输出只走 stdout，由调用方决定重定向。
# ============================================================================
set -uo pipefail

LOG="${LOG:-/mnt/workspace/_diag_oom.log}"

hr() { printf '\n%s\n' "--------------------------------------------------------------"; }
sec() { hr; printf '### %s\n' "$*"; hr; }

printf '\n'
printf '================================================================\n'
printf ' 内存 / 磁盘 / nohang 诊断\n'
printf ' 时间: %s\n' "$(date '+%F %T %Z')"
printf ' 主机: %s\n' "$(hostname)"
printf ' 日志: %s\n' "${LOG}"
printf '================================================================\n'

# ---------------------------------------------------------------------------
sec "1. 磁盘空间（关键：加 swap 前必须确认有空间）"
df -h 2>/dev/null | sed 's/^/  /'
echo
echo "  [说明] fallocate 32G 会实占 32G 磁盘。若 / 或 /mnt/workspace 剩余不足，"
echo "         先扩盘或改用更小的 swap（8G/16G），切勿硬上 32G。"

sec "1b. 关键挂载点剩余空间（GiB）"
for M in / /mnt/workspace /tmp /dev/shm; do
  if [ -d "${M}" ]; then
    AVAIL=$(df -BG "${M}" 2>/dev/null | awk 'NR==2{gsub(/G/,"",$4); print $4}')
    printf '  %-16s 可用 %s GiB\n' "${M}" "${AVAIL:-?}"
  fi
done

# ---------------------------------------------------------------------------
sec "2. 内存与 swap"
free -h 2>/dev/null | sed 's/^/  /'
echo
SWAP_KB=$(awk '/SwapTotal/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
SWAP_GB=$(( ${SWAP_KB:-0} / 1048576 ))
echo "  SwapTotal = ${SWAP_GB} GiB"
if [ "${SWAP_KB:-0}" = "0" ]; then
  echo "  [FAIL] 零 swap —— 28G 宿主内存无任何缓冲，跑 7B 极易被杀"
else
  echo "  [OK]   swap 已配置"
fi

# ---------------------------------------------------------------------------
sec "3. nohang 守护进程（DSW 的用户态内存杀手）"
if pgrep -f nohang >/dev/null 2>&1; then
  echo "  --- 进程 ---"
  pgrep -af nohang 2>/dev/null | sed 's/^/    /'
  echo
  echo "  --- 配置 ---"
  for CFG in /etc/dsw/cache/pre-install/plugins/nohang/*/pai-nohang.conf \
             /etc/nohang/nohang.conf; do
    if [ -f "${CFG}" ]; then
      echo "    [${CFG}]"
      grep -vE '^\s*#|^\s*$' "${CFG}" 2>/dev/null | head -30 | sed 's/^/      /'
    fi
  done
  echo
  echo "  --- 日志 ---"
  FOUND=0
  for LG in /var/log/nohang/nohang.log /var/log/nohang.log \
            /var/log/nohang/*.log /mnt/workspace/.nohang*.log; do
    [ -f "${LG}" ] && { FOUND=1; echo "    [${LG}] 尾部:"; \
      tail -40 "${LG}" 2>/dev/null | sed 's/^/      /'; }
  done
  [ "${FOUND}" = "0" ] && echo "    (未找到 nohang 日志文件；试 journalctl -t nohang)"
  echo
  echo "  --- journald 过滤 ---"
  if command -v journalctl >/dev/null 2>&1; then
    journalctl --no-pager -n 200 2>/dev/null \
      | grep -iE "nohang|killed|SIGKILL|oom" | tail -20 | sed 's/^/    /' \
      || echo "    (无匹配)"
  else
    echo "    (无 journalctl)"
  fi
else
  echo "  [OK] 未检测到 nohang"
fi

# ---------------------------------------------------------------------------
sec "4. 内核 OOM 记录"
if command -v dmesg >/dev/null 2>&1; then
  D=$(dmesg 2>/dev/null | grep -iE "out of memory|killed process|oom-kill" | tail -10)
  if [ -n "${D}" ]; then
    echo "  [WARN] 发现 OOM 记录:"
    echo "${D}" | sed 's/^/    /'
  else
    echo "  dmesg 无 OOM 关键字（或不可读）。最近 15 行原始输出:"
    dmesg 2>/dev/null | tail -15 | sed 's/^/    /'
    if [ -z "$(dmesg 2>/dev/null)" ]; then
      echo "    (dmesg 完全不可读 —— 容器内 kernel.dmesg_restrict 限制，属正常；"
      echo "     真正的线索在 cgroup memory.events 与 nohang 日志，见第 3、5 节)"
    fi
  fi
else
  echo "  (无 dmesg 命令)"
fi

# ---------------------------------------------------------------------------
sec "5. cgroup 内存限制（容器真实上限，可能小于 28G）"
for F in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
  if [ -r "${F}" ]; then
    V=$(cat "${F}" 2>/dev/null)
    if [ "${V}" = "max" ]; then
      echo "  ${F} = max（无限制）"
    else
      echo "  ${F} = ${V}  ($(( ${V:-0} / 1073741824 )) GiB)"
    fi
  fi
done
for F in /sys/fs/cgroup/memory.events /sys/fs/cgroup/memory/memory.failcnt; do
  [ -r "${F}" ] && { echo "  --- ${F} ---"; cat "${F}" 2>/dev/null | sed 's/^/    /'; }
done
echo
echo "  [提示] memory.events 里 oom_kill 计数 > 0 即证明发生过 OOM 杀进程。"

# ---------------------------------------------------------------------------
sec "6. nohang 类似机制：PSI 压力指标"
if [ -r /proc/pressure/memory ]; then
  cat /proc/pressure/memory | sed 's/^/  /'
  echo "  [说明] some avg10 持续偏高（>10）说明内存压力大，nohang 会据此动手。"
else
  echo "  (无 /proc/pressure/memory)"
fi

# ---------------------------------------------------------------------------
sec "7. DSW 系统进程（勿杀）"
ps -eo pid,ppid,etime,rss,cmd --sort=-rss 2>/dev/null | head -18 | sed 's/^/  /'

# ---------------------------------------------------------------------------
sec "8. 业务残留进程（已排除 DSW 系统进程）"
ORPHAN=$(ps -eo pid,etime,rss,cmd 2>/dev/null \
  | grep -E "python|modelscope|openvla|libero" \
  | grep -vE "grep|diag_oom" \
  | grep -vE "/etc/dsw/|supervisord|nohang|paimcp|jupyter-lab")
if [ -n "${ORPHAN}" ]; then
  echo "  发现可疑残留:"
  echo "${ORPHAN}" | sed 's/^/    /'
  echo
  echo "  确认无用后 kill:"
  echo "${ORPHAN}" | awk '{print "    kill -9 " $1}'
else
  echo "  [OK] 无业务残留进程"
fi

# ---------------------------------------------------------------------------
sec "9. GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader | sed 's/^/  /'
  echo
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    | sed 's/^/  /'
fi

# ---------------------------------------------------------------------------
sec "10. tmux / screen 可用性（防止终端再死的关键工具）"
for T in tmux screen; do
  if command -v "${T}" >/dev/null 2>&1; then
    echo "  [OK] ${T} 可用: $(command -v ${T})"
  else
    echo "  [--] ${T} 未安装"
  fi
done
echo "  [建议] apt-get install -y tmux（若可联网）；或全程用 nohup+日志文件"

# ---------------------------------------------------------------------------
printf '\n'
printf '================================================================\n'
printf ' 诊断完成: %s\n' "$(date '+%F %T')"
printf ' 完整日志已写入: %s\n' "${LOG}"
printf ' 终端若再次退出，重连后执行看结果:\n'
printf '   cat %s\n' "${LOG}"
printf '   或只看末尾: tail -100 %s\n' "${LOG}"
printf '================================================================\n'

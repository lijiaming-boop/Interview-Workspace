#!/bin/bash
# 极简探针 v2：纯 ASCII，无进程替换/heredoc，只写日志文件。
# 用途：在终端反复退出的环境下，把关键状态落到磁盘。
# 运行: bash _probe.sh      结果: /mnt/workspace/_probe.log

L=/mnt/workspace/_probe.log
{
  echo "===== probe $(date '+%F %T') ====="
  echo "shell=\$0=$0   args=\$#=$#"

  echo "--- pwd ---"
  pwd

  echo "--- 当前目录下的脚本 ---"
  ls -la *.sh 2>/dev/null || echo "(本目录无 .sh)"

  echo "--- 全盘找这几个脚本（限深度，避免慢） ---"
  for n in run_detached.sh diag_oom.sh resume_verify.sh download_oft_ckpt.sh _probe.sh; do
    hit=$(find /mnt/workspace /root /tmp -maxdepth 4 -name "$n" 2>/dev/null | head -3)
    if [ -n "$hit" ]; then
      echo "$n FOUND:"
      echo "$hit" | sed 's/^/    /'
    else
      echo "$n not found"
    fi
  done

  echo "--- bash ---"
  bash --version | head -1

  echo "--- disk ---"
  df -h / /mnt/workspace 2>/dev/null

  echo "--- memory ---"
  free -h 2>/dev/null

  echo "--- swapfile? ---"
  ls -lh /swapfile 2>/dev/null || echo "no /swapfile"

  echo "--- cgroup 版本 ---"
  stat -fc %T /sys/fs/cgroup 2>/dev/null || echo "cannot stat /sys/fs/cgroup"

  echo "--- cgroup v2: memory.events ---"
  cat /sys/fs/cgroup/memory.events 2>/dev/null || echo "(无 v2 memory.events)"

  echo "--- cgroup v1: memory.failcnt / usage ---"
  cat /sys/fs/cgroup/memory/memory.failcnt 2>/dev/null || echo "(无 v1 memory.failcnt)"
  cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo "(无 v1 usage_in_bytes)"
  cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo "(无 v1 limit_in_bytes)"

  echo "--- cgroup 其他可能的 OOM 计数 ---"
  for p in /sys/fs/cgroup/memory.oom.group \
           /sys/fs/cgroup/memory/memory.oom_control \
           /sys/fs/cgroup/memory/memory.max_usage_in_bytes; do
    if [ -r "$p" ]; then
      echo "[$p]"
      cat "$p" 2>/dev/null | sed 's/^/    /'
    fi
  done

  echo "--- psi ---"
  cat /proc/pressure/memory 2>/dev/null || echo "(无 /proc/pressure/memory)"

  echo "--- nohang 进程 ---"
  pgrep -af nohang 2>/dev/null || echo "no nohang process"

  echo "--- nohang 配置 ---"
  for c in /etc/dsw/cache/pre-install/plugins/nohang/v0.1.0/pai-nohang.conf \
           /etc/dsw/cache/pre-install/plugin-packages/nohang/v0.1.0/pai-nohang.conf; do
    if [ -r "$c" ]; then
      echo "[$c]"
      grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$c" 2>/dev/null | head -40 | sed 's/^/    /'
    fi
  done

  echo "--- nohang 日志 ---"
  for g in /var/log/nohang/nohang.log /var/log/nohang.log /var/log/nohang/*.log; do
    if [ -f "$g" ]; then
      echo "[$g]"
      tail -30 "$g" 2>/dev/null | sed 's/^/    /'
    fi
  done
  echo "(若上面无输出，nohang 日志可能走 journald)"

  echo "--- journalctl 里的 nohang/killed ---"
  if command -v journalctl >/dev/null 2>&1; then
    journalctl --no-pager -n 300 2>/dev/null | grep -iE "nohang|killed|SIGKILL|oom" | tail -20
  else
    echo "(无 journalctl)"
  fi

  echo "--- gpu ---"
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null

  echo "--- /tmp 与 /dev/shm ---"
  df -h /tmp /dev/shm 2>/dev/null

  echo "===== done ====="
} > "$L" 2>&1

echo "written: $L"
echo "------------------------------------------------------------"
tail -80 "$L"

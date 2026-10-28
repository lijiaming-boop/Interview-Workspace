#!/usr/bin/env bash
# ============================================================================
# setup_swap.sh —— 给 DSW 实例补 swap（自动避开 overlayfs 陷阱）
# ----------------------------------------------------------------------------
# 背景：
#   本实例 28 GiB 宿主内存、Swap 0 B。nohang 的 hard_threshold_min_mem=2%
#   意味着空闲内存跌破 ~573 MiB 就会杀【整个进程组】。
#   加 swap 能把这个悬崖推远。
#
# ⚠️ 关键陷阱：`/` 是 overlayfs。**swapon 在 overlayfs 上通常会失败**
#   （报 "Invalid argument" / "swapfile has holes"）。
#   本脚本会先探测非 overlay 的可写目录（如 /tmp 挂在 /dev/vda3 真实块设备上），
#   再用 64MiB 小文件做可行性试挂，成功后才放大。
#
# 用法：
#   bash setup_swap.sh              # 自动选位置，默认 32G
#   SWAP_GB=16 bash setup_swap.sh   # 指定大小
#   bash setup_swap.sh --check      # 只探测可行性，不实际启用
# ============================================================================
set -uo pipefail

# ---- 粘贴保护 ----
_SELF="${0##*/}"
_PASTED=0
case "${_SELF}" in bash|-bash|sh|-sh|dash|-dash|zsh|-zsh) _PASTED=1 ;; esac
if [ "${_PASTED}" = "1" ]; then
  echo "[注意] 本脚本被粘贴进交互终端。请存成文件后执行: bash setup_swap.sh"
  echo
fi

SWAP_GB="${SWAP_GB:-32}"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m[OK]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*"; }

log "1. 当前内存 / swap / 磁盘"
free -h
echo
df -h / /tmp 2>/dev/null

if swapon --show 2>/dev/null | grep -q .; then
  ok "已有 swap 在用："
  swapon --show
fi

log "2. 探测各目录的文件系统类型（overlayfs 不能放 swapfile）"
CANDIDATES=""
for d in /tmp /var/tmp /root /opt /home /usr/local /mnt/data /.swap; do
  [ -d "$d" ] || continue
  [ -w "$d" ] || continue
  FSTYPE=$(stat -f -c %T "$d" 2>/dev/null || echo "?")
  AVAIL=$(df -BG "$d" 2>/dev/null | awk 'NR==2{gsub(/G/,"",$4); print $4}')
  MARK=""
  case "$FSTYPE" in
    overlayfs) MARK=" <-- 不能放 swap" ;;
    tmpfs)     MARK=" <-- 不能放 swap（内存盘）" ;;
    nfs*|fuse*|smb*|cifs*) MARK=" <-- 网络盘，不能放 swap" ;;
  esac
  printf '  %-16s fs=%-12s avail=%-6s%s\n' "$d" "$FSTYPE" "${AVAIL:-?}G" "$MARK"
  if [ -z "$MARK" ] && [ -n "${AVAIL:-}" ] && [ "${AVAIL}" -ge 8 ] 2>/dev/null; then
    CANDIDATES="${CANDIDATES} ${d}"
  fi
done

if [ -z "${CANDIDATES}" ]; then
  warn "没有找到可用于 swapfile 的非 overlay 目录"
  echo "  可用路径首选 /tmp（实测挂在 /dev/vda3 真实块设备上）"
  echo "  若 /tmp 也不可用，可考虑改用 zram（压缩内存盘），但那不解决 OOM。"
  CANDIDATES="/tmp"
fi

log "3. 逐个位置试挂（用 64 MiB 小文件验证可行性）"
PICK=""
for d in ${CANDIDATES}; do
  T="${d%/}/.swaptest"
  rm -f "$T" 2>/dev/null || true
  if ! dd if=/dev/zero of="$T" bs=1M count=64 status=none 2>/dev/null; then
    printf '  %-16s 试挂失败：无法创建文件\n' "$d"
    continue
  fi
  chmod 600 "$T" 2>/dev/null
  if ! mkswap "$T" >/dev/null 2>&1; then
    printf '  %-16s 试挂失败：mkswap 出错\n' "$d"
    rm -f "$T" 2>/dev/null; continue
  fi
  ERR=$(swapon "$T" 2>&1)
  if [ $? -eq 0 ]; then
    printf '  %-16s \033[1;32m试挂成功\033[0m\n' "$d"
    swapoff "$T" 2>/dev/null
    rm -f "$T" 2>/dev/null
    PICK="$d"
    break
  else
    printf '  %-16s 试挂失败: %s\n' "$d" "${ERR}"
    rm -f "$T" 2>/dev/null
  fi
done

if [ -z "${PICK}" ]; then
  fail "所有候选位置都无法启用 swap（很可能都是 overlayfs）"
  echo
  echo "  退而求其次的方案："
  echo "   a) 换个非 overlay 的持久目录挂载（需 DSW 控制台配合）"
  echo "   b) 使用 zram（压缩内存盘，不解决宿主内存不足，仅缓解）"
  echo "      modprobe zram && echo 8G > /sys/block/zram0/disksize && mkswap /dev/zram0 && swapon /dev/zram0"
  echo "   c) 严格控制并发：INT8 与 BF16 绝不同时跑，先冒烟看峰值"
  echo "     并用 memwatch.sh 盯水位（见工作区脚本）"
  if [ "${_PASTED}" = "1" ]; then return 0 2>/dev/null || true; else exit 1; fi
fi

if [ "${CHECK_ONLY}" = "1" ]; then
  ok "可行性探测完成，可用于 swap 的位置: ${PICK}（--check 模式，未实际启用）"
  if [ "${_PASTED}" = "1" ]; then return 0 2>/dev/null || true; else exit 0; fi
fi

log "4. 在 ${PICK} 创建 ${SWAP_GB} GiB swapfile"

# 再次确认磁盘够用
AVAIL=$(df -BG "${PICK}" 2>/dev/null | awk 'NR==2{gsub(/G/,"",$4); print $4}')
echo "  ${PICK} 可用 ${AVAIL:-?}G，需要 ${SWAP_GB}G"
if [ -n "${AVAIL:-}" ] && [ "${AVAIL}" -lt $((SWAP_GB + 2)) ] 2>/dev/null; then
  fail "磁盘不足，请减小 SWAP_GB"
  if [ "${_PASTED}" = "1" ]; then return 1 2>/dev/null || true; else exit 1; fi
fi

SF="${PICK%/}/swapfile"
if [ -f "${SF}" ]; then
  warn "已存在 ${SF}，先关掉旧的"
  swapoff "${SF}" 2>/dev/null || true
  rm -f "${SF}"
fi

# dd 比 fallocate 更可靠（overlayfs 上 fallocate 可能生成带洞文件导致 swapon 失败）
echo "  创建中（dd ${SWAP_GB}G，约 1-2 分钟）..."
dd if=/dev/zero of="${SF}" bs=1M count=$((SWAP_GB * 1024)) status=progress 2>&1 | tail -1
chmod 600 "${SF}"

mkswap "${SF}" >/dev/null 2>&1 || { fail "mkswap 失败"; if [ "${_PASTED}" = "1" ]; then return 1 2>/dev/null || true; else exit 1; fi; }

if ! swapon "${SF}" 2>&1; then
  fail "swapon 失败，删除文件"
  rm -f "${SF}"
  if [ "${_PASTED}" = "1" ]; then return 1 2>/dev/null || true; else exit 1; fi
fi

ok "swap 已启用"
free -h
echo
swapon --show | sed 's/^/  /'

log "5. 写入 /etc/fstab 以便重启后自动挂载"
if grep -q "${SF}" /etc/fstab 2>/dev/null; then
  ok "fstab 已有该条目"
else
  echo "${SF} none swap sw 0 0" >> /etc/fstab && ok "已追加: ${SF} none swap sw 0 0"
fi

log "6. 核对 nohang 阈值余量"
echo "  nohang: hard_threshold_min_mem = 2%  → 空闲内存跌破 ~573 MiB 会杀进程组"
echo "  nohang: soft_threshold_min_mem = 3%  → 空闲内存跌破 ~860 MiB 触发软动作"
echo "  当前状态:"
free -m | awk 'NR==1{print "    "$0} NR==2{printf "    %s 空闲 %s MiB (%.1f%%)\n", $1, $4, $4/$2*100} NR==3{printf "    %s 空闲 %s MiB\n", $1, $4}'

warn "注意：swapfile 位于 ${SF}，若该目录重启后被清空，需重跑本脚本"
echo
echo "建议：跑基准时用 memwatch.sh 盯内存水位"
echo "  bash memwatch.sh &"

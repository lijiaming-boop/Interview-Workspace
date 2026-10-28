#!/bin/bash
# ============================================================================
# fix_crlf.sh —— 清理脚本里的 Windows 换行符（CRLF）与 BOM
# ----------------------------------------------------------------------------
# 为什么需要：
#   脚本经 Windows 侧编辑/复制/上传后，行尾可能变成 CRLF。
#   Linux 上 bash 会把 `then\r`、`fi\r` 当成普通单词，
#   报 "syntax error near unexpected token"，**退出码恰为 2**。
#   这与"脚本本身有 bug"表现一致，极易误判。
#
# 用法：
#   bash fix_crlf.sh              # 检查+修复当前目录所有 .sh/.py
#   bash fix_crlf.sh --check      # 只报告，不改
#   bash fix_crlf.sh 目录         # 指定目录
# ============================================================================
set -uo pipefail

DIR="${2:-.}"
MODE="fix"
[ "${1:-}" = "--check" ] && { MODE="check"; DIR="${2:-.}"; }

echo "目录: $(cd "${DIR}" && pwd)"
echo

N_FIX=0
N_BAD=0

for f in "${DIR}"/*.sh "${DIR}"/*.py; do
  [ -f "$f" ] || continue

  # 统计 CRLF 行数（用 tr 去掉 CR 后比较；不依赖 file 命令）
  CR=$(tr -cd '\r' < "$f" | wc -c | tr -d ' ')
  SIZE=$(wc -c < "$f" | tr -d ' ')

  # BOM 检测（前 3 字节 EF BB BF）
  BOM="no"
  if [ "$(head -c 3 "$f" | od -An -tx1 | tr -d ' \n')" = "efbbbf" ]; then
    BOM="YES"
  fi

  if [ "${CR}" -gt 0 ] || [ "${BOM}" = "YES" ]; then
    N_BAD=$((N_BAD + 1))
    printf '  [BAD ] %-32s CR=%s BOM=%s size=%s\n' "$(basename "$f")" "${CR}" "${BOM}" "${SIZE}"
    if [ "${MODE}" = "fix" ]; then
      cp -f "$f" "$f.bak" 2>/dev/null || true
      # 去 CR
      tr -d '\r' < "$f.bak" > "$f.tmp" 2>/dev/null && mv -f "$f.tmp" "$f"
      # 去 BOM
      if [ "$(head -c 3 "$f" | od -An -tx1 | tr -d ' \n')" = "efbbbf" ]; then
        tail -c +4 "$f" > "$f.tmp" 2>/dev/null && mv -f "$f.tmp" "$f"
      fi
      chmod +x "$f" 2>/dev/null || true
      NEWCR=$(tr -cd '\r' < "$f" | wc -c | tr -d ' ')
      echo "          -> 已修复 (CR=${NEWCR})，备份 $f.bak"
      N_FIX=$((N_FIX + 1))
    fi
  else
    printf '  [ OK ] %-32s size=%s\n' "$(basename "$f")" "${SIZE}"
  fi
done

echo
if [ "${N_BAD}" = "0" ]; then
  echo "[OK] 无 CRLF / BOM 问题"
else
  if [ "${MODE}" = "fix" ]; then
    echo "[OK] 修复 ${N_FIX} 个文件。原文件已备份为 *.bak"
    echo "     建议复验: bash -n <脚本>"
  else
    echo "[WARN] ${N_BAD} 个文件有问题。执行 'bash fix_crlf.sh' 修复"
  fi
fi

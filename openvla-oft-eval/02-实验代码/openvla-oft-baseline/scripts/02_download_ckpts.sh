#!/usr/bin/env bash
# =============================================================================
# 02_download_ckpts.sh —— 从 hf-mirror.com 下载 OpenVLA-OFT 官方 LIBERO checkpoint
#
# 用法:
#   bash 02_download_ckpts.sh spatial            # 只下 LIBERO-Spatial（约 15GB）
#   bash 02_download_ckpts.sh all                # 四个 suite 全下（约 60GB）
#   bash 02_download_ckpts.sh combined           # 四合一 checkpoint（15GB，可跑全部 suite）
#
# 说明：
#   - 魔搭（ModelScope）没有官方 checkpoint 镜像，必须走 hf-mirror.com
#   - 支持断点续传：中断后重新执行同一命令即可继续
#   - 每个仓库除 safetensors 权重外还包含 action_head / proprio_projector 的
#     .pt 文件和 dataset_statistics.json（反归一化要用），脚本会校验
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

WHAT="${1:-spatial}"
case "$WHAT" in
  all) SUITES="spatial object goal 10" ;;
  spatial|object|goal|10|combined) SUITES="$WHAT" ;;
  *) echo "用法: bash 02_download_ckpts.sh <spatial|object|goal|10|combined|all>"; exit 1 ;;
esac

activate_env
export HF_ENDPOINT="${HF_MIRROR}"
echo "==> 使用 HuggingFace 镜像: ${HF_ENDPOINT}"

dl() {  # repo local_dir：新版用 hf，旧版退回 huggingface-cli
  local repo="$1" dest="$2"
  mkdir -p "${dest}"
  if command -v hf >/dev/null 2>&1; then
    hf download "${repo}" --local-dir "${dest}"
  else
    huggingface-cli download "${repo}" --local-dir "${dest}"
  fi
}

for s in ${SUITES}; do
  repo="moojink/$(suite2ckpt "$s")"
  dest="${CKPT_ROOT}/$(suite2ckpt "$s")"
  echo ""
  echo "==> 下载 ${repo} -> ${dest}"
  dl "${repo}" "${dest}"
done

echo ""
echo "==> 校验关键文件"
FAIL=0
for s in ${SUITES}; do
  d="${CKPT_ROOT}/$(suite2ckpt "$s")"
  ok=1
  [ -f "${d}/config.json" ]             || { echo "  [!!] ${d} 缺 config.json"; ok=0; }
  [ -f "${d}/dataset_statistics.json" ] || { echo "  [!!] ${d} 缺 dataset_statistics.json（反归一化必需）"; ok=0; }
  ls "${d}" 2>/dev/null | grep -q "safetensors"        || { echo "  [!!] ${d} 缺模型权重 safetensors"; ok=0; }
  ls "${d}" 2>/dev/null | grep -Eq "action_head|proprio_projector" || { echo "  [!!] ${d} 缺 action_head/proprio_projector .pt"; ok=0; }
  [ "$ok" = 1 ] && echo "  [OK ] $(suite2ckpt "$s") 文件齐全"
  [ "$ok" = 0 ] && FAIL=1
done

echo ""
echo "==> 各 checkpoint 占用："
du -sh "${CKPT_ROOT}"/* 2>/dev/null || true

if [ "${FAIL}" = 1 ]; then
  echo "!! 有文件缺失——重跑本命令续传（hf download 支持断点续传）"
  exit 1
fi
echo "==> 下载完成。下一步: SMOKE=1 bash ${SCRIPT_DIR}/03_run_libero_eval.sh spatial"

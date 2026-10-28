#!/usr/bin/env bash
# =============================================================================
# 00_preflight.sh —— 运行前自检：GPU / 磁盘 / conda / 国内源连通性
# 用法: bash 00_preflight.sh
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

echo "==================== 1. GPU ===================="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  echo "CUDA 驱动版本: $(nvidia-smi | grep -oE 'CUDA Version: [0-9.]+' | head -1)"
  echo "（OpenVLA-OFT bf16 推理约需 16GB 显存；A10 24GB 满足）"
else
  echo "!! nvidia-smi 不可用——请在 GPU 实例上运行本套件"
fi

echo "==================== 2. 磁盘 ===================="
df -h "$HOME" | tail -1
echo "磁盘预算参考: conda 环境 ~15GB + 每个 checkpoint ~15GB(×N) + 每个视频 ~1-3MB"
echo "四套 checkpoint 全下约 60GB；磁盘紧张可用四合一 checkpoint（15GB 跑全部 suite）"

echo "==================== 3. 内存 / CPU ===================="
free -g | head -2
echo "CPU 核数: $(nproc)"

echo "==================== 4. conda / git ===================="
if command -v conda >/dev/null 2>&1; then conda --version; else
  echo "!! 无 conda（01_setup_env.sh 会自动从清华镜像安装 Miniconda）"; fi
if command -v git >/dev/null 2>&1; then git --version; else echo "!! 无 git"; fi

echo "==================== 5. 国内源连通性 ===================="
check() {
  local name="$1" url="$2" code
  code=$(curl -s -o /dev/null -m 10 -w '%{http_code}' "$url" 2>/dev/null || echo FAIL)
  case "$code" in
    200|301|302) echo "  [OK ] $name -> $code" ;;
    *)           echo "  [!! ] $name -> $code （不通，请换备用源，见 common.sh 注释）" ;;
  esac
}
check "pip 清华源 " "${PIP_INDEX%/simple}/simple/"
check "hf-mirror   " "${HF_MIRROR}/"
check "GitHub 加速 " "${GH_PROXY}/https://raw.githubusercontent.com/moojink/openvla-oft/main/README.md"
check "清华 conda  " "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/"

echo "==================== 6. 实例自带环境（USE_BASE=1 模式会直接使用它） ===================="
python --version 2>/dev/null || true
python -c "import torch; print('base torch:', torch.__version__)" 2>/dev/null || true

echo ""
echo "自检完成。下一步: bash ${SCRIPT_DIR}/01_setup_env.sh"

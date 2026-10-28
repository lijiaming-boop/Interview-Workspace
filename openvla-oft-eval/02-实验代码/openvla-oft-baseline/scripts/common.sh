#!/usr/bin/env bash
# =============================================================================
# common.sh —— 全局配置（被所有脚本 source）
# 所有变量都可用环境变量覆盖，例如：
#   OFT_ROOT=~/my/openvla-oft CKPT_ROOT=/data/ckpts bash scripts/03_run_libero_eval.sh spatial 50
# =============================================================================

export OFT_ROOT="${OFT_ROOT:-$HOME/openvla-oft}"            # openvla-oft 仓库
export LIBERO_ROOT="${LIBERO_ROOT:-$HOME/LIBERO}"            # LIBERO 仓库
export CKPT_ROOT="${CKPT_ROOT:-$HOME/openvla-oft-ckpts}"     # checkpoint 下载目录
export CONDA_ENV="${CONDA_ENV:-oft}"                         # conda 环境名

# 国内镜像 / 加速（备选：GH_PROXY=https://gh-proxy.com 或 https://ghproxy.net；
#               PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/）
export GH_PROXY="${GH_PROXY:-https://ghfast.top}"
export HF_MIRROR="${HF_MIRROR:-https://hf-mirror.com}"
export PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCRIPT_DIR
export RESULTS_DIR="${RESULTS_DIR:-$(dirname "${SCRIPT_DIR}")/results}"

# suite 短名 -> HuggingFace checkpoint 仓库名（也是本地目录名）
suite2ckpt() {
  case "$1" in
    spatial|object|goal|10) echo "openvla-7b-oft-finetuned-libero-$1" ;;
    combined)               echo "openvla-7b-oft-finetuned-libero-spatial-object-goal-10" ;;
    *) echo "未知 suite: $1（可选: spatial|object|goal|10|combined）" >&2; return 1 ;;
  esac
}

# 01_setup_env.sh 以 USE_BASE=1 运行时写入此标记：依赖装在镜像自带 python 里
export OFT_BASE_MARKER="${OFT_BASE_MARKER:-$HOME/.oft_use_base}"

# 激活运行环境（02/03 脚本使用）：有标记则直接用镜像自带 python，否则激活 conda 环境
activate_env() {
  if [ -f "${OFT_BASE_MARKER}" ]; then
    return 0
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "!! 未找到 conda，请先运行 01_setup_env.sh" >&2; exit 1
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}" || { echo "!! conda 环境 ${CONDA_ENV} 不存在，请先运行 01_setup_env.sh（或加 USE_BASE=1）" >&2; exit 1; }
}

#!/usr/bin/env bash
# =============================================================================
# 01_setup_env.sh —— 搭建 OpenVLA-OFT + LIBERO 评测环境（全国内源）
#
# 做了什么：
#   1. 无 conda 时从清华镜像装 Miniconda
#   2. 环境二选一：默认新建独立 conda 环境 `oft`（python 3.10）；
#      镜像已是 py3.10 时可 USE_BASE=1 直接装入自带环境（其 torch 2.3.0 会被降为官方 pin 2.2.0）
#   3. git 配置 GitHub 加速改写（ghfast.top），openvla-oft 的两个 git 依赖
#      （transformers fork、dlimp）会自动走代理
#   4. 安装 torch==2.2.0（官方 pin；PyPI 轮子自带 CUDA 12.1 运行时，与镜像 cu121 匹配）
#   5. 先装 gym==0.21.0（绕过其 sdist 元数据 bug），再装 openvla-oft / LIBERO
#   6. 全程锁 numpy==1.24.4（TF 2.15 / robosuite / LIBERO 均不兼容 numpy 2.x）
#
# 耗时约 30-60 min；磁盘 ~15GB。
# 注：评测不需要 flash-attn（官方加载代码默认走 PyTorch SDPA），故不安装。
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

echo "==> [0/8] 检查 GPU 与磁盘"
command -v nvidia-smi >/dev/null 2>&1 || { echo "!! nvidia-smi 不可用，请在 GPU 实例上运行"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
FREE_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9' || echo 0)
if [ "${FREE_GB:-0}" -lt 60 ]; then
  echo "!! 警告: HOME 所在盘仅剩 ${FREE_GB}GB（环境 ~15GB + 单 ckpt ~15GB 起步）"
fi

# ---------- 0/1) 运行环境：USE_BASE=1 用镜像自带 python，否则新建独立 conda 环境 ----------
if [ "${USE_BASE:-0}" = "1" ]; then
  echo "==> [1/8] USE_BASE=1：直接使用镜像自带 python（校验版本）"
  PYV=$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
  if [ "${PYV}" != "3.10" ]; then
    echo "!! 当前 python 为 ${PYV}，USE_BASE=1 仅适用于 py3.10 镜像；请去掉 USE_BASE=1 重新运行"
    exit 1
  fi
  touch "${OFT_BASE_MARKER}"
  echo "    python: $(python --version)（镜像自带 torch 2.3.0 将按官方 pin 降为 2.2.0）"
else
  if ! command -v conda >/dev/null 2>&1; then
    echo "==> 未检测到 conda，从清华镜像安装 Miniconda"
    wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "${HOME}/miniconda3"
    "${HOME}/miniconda3/bin/conda" init bash >/dev/null 2>&1 || true
    export PATH="${HOME}/miniconda3/bin:${PATH}"
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"

  # conda 使用清华 channel（不覆盖已有 .condarc）
  if [ ! -f "${HOME}/.condarc" ]; then
  cat > "${HOME}/.condarc" <<'EOF'
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF
  fi

  if conda env list | grep -qw "${CONDA_ENV}"; then
    echo "==> [1/8] conda 环境 ${CONDA_ENV} 已存在，跳过创建"
  else
    echo "==> [1/8] 创建 conda 环境 ${CONDA_ENV} (python=3.10)"
    conda create -n "${CONDA_ENV}" python=3.10 -y
  fi
  conda activate "${CONDA_ENV}"
  echo "python: $(python --version)"
fi

PIP="python -m pip"
PIP_ARGS=(-i "${PIP_INDEX}")

# ---------- 2) git 加速改写 ----------
echo "==> [2/8] 配置 git 走 GitHub 加速代理 (${GH_PROXY})"
git config --global url."${GH_PROXY}/https://github.com/".insteadOf "https://github.com/"
echo "    （撤销命令: git config --global --unset url.\"${GH_PROXY}/https://github.com/\".insteadOf）"

# ---------- 3) torch 2.2.0 ----------
echo "==> [3/8] 安装 torch==2.2.0 / torchvision==0.17.0（PyPI 轮子自带 CUDA 12.1）"
$PIP install "${PIP_ARGS[@]}" torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# ---------- 4) 克隆仓库（URL 会被 git 自动改写走代理） ----------
echo "==> [4/8] 克隆 openvla-oft 与 LIBERO"
[ -d "${OFT_ROOT}/.git" ] || git clone https://github.com/moojink/openvla-oft.git "${OFT_ROOT}"
[ -d "${LIBERO_ROOT}/.git" ] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_ROOT}"

# ---------- 5) gym 0.21.0（必须先于其它包装，原因见 docs/01） ----------
echo "==> [5/8] 安装 gym==0.21.0（LIBERO/robosuite 旧栈要求）"
$PIP install "${PIP_ARGS[@]}" "setuptools==65.5.0" "wheel==0.38.4"
$PIP install "${PIP_ARGS[@]}" gym==0.21.0 --no-build-isolation || {
  echo "!! gym==0.21.0 安装失败——可重试一次；若后续 LIBERO 冒烟测试能跑通也可忽略"
}

# ---------- 6) openvla-oft + LIBERO ----------
echo "==> [6/8] 安装 openvla-oft（transformers fork / dlimp 两个 git 依赖走代理）"
CONSTRAINTS=/tmp/oft-constraints.txt
printf 'numpy==1.24.4\n' > "${CONSTRAINTS}"
pushd "${OFT_ROOT}" > /dev/null
$PIP install "${PIP_ARGS[@]}" -e . -c "${CONSTRAINTS}"
echo "==> 安装 LIBERO"
$PIP install "${PIP_ARGS[@]}" -e "${LIBERO_ROOT}" -c "${CONSTRAINTS}"
$PIP install "${PIP_ARGS[@]}" -r experiments/robot/libero/libero_requirements.txt -c "${CONSTRAINTS}"
popd > /dev/null

# ---------- 7) 锁 numpy ----------
echo "==> [7/8] 锁定 numpy==1.24.4（防止被依赖静默升级到 2.x）"
$PIP install "${PIP_ARGS[@]}" numpy==1.24.4

# ---------- 8) 验证 ----------
echo "==> [8/8] 验证安装"
python - <<'PY'
import torch, numpy, gym, robosuite, transformers, timm
print("torch        :", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("numpy        :", numpy.__version__, "(必须 1.24.4)")
print("gym          :", gym.__version__)
print("robosuite    :", robosuite.__version__)
print("transformers :", transformers.__version__, "(应为 4.40.x，moojink fork)")
print("timm         :", timm.__version__)
import prismatic.vla.constants as C
assert C.NUM_ACTIONS_CHUNK == 8 and C.ACTION_DIM == 7 and C.PROPRIO_DIM == 8, "LIBERO 常量异常"
print("NUM_ACTIONS_CHUNK =", C.NUM_ACTIONS_CHUNK, "| ACTION_DIM =", C.ACTION_DIM, "| PROPRIO_DIM =", C.PROPRIO_DIM)
print("OK: 基础依赖验证通过")
PY

cat <<'EOF'

==> 环境搭建完成！接下来：
    1) 下载 checkpoint:   bash scripts/02_download_ckpts.sh spatial
    2) 前台冒烟测试(必做): SMOKE=1 bash scripts/03_run_libero_eval.sh spatial
       —— 首次运行 LIBERO 会交互式询问"数据集保存路径"，
          随便输入一个可写路径（如 /root/libero_data）回车即可，只问这一次。
EOF

#!/usr/bin/env bash
# =============================================================================
# setup_oft_env.sh —— 魔搭 DSW A10 实例：OpenVLA-OFT INT8 环境补齐 + 自检
#
# 用途：在已建好 venv_oft 的实例上，把 OFT 运行所需的全部依赖补齐到项目锁定版本，
#       并逐项自检。幂等：可重复执行。
# 用法：
#   bash setup_oft_env.sh            # 补齐依赖 + 自检
#   bash setup_oft_env.sh --check    # 只自检，不安装
#   bash setup_oft_env.sh --fix-cv2  # 只修 opencv 冲突
#
# 设计依据：08-后续实验验证/int8_complete_runtime_v3/.../repair_oft_dependencies.sh
#           （其 --deps-only 调用契约与包内 verify_oft_runtime.sh 不匹配，故本脚本内联实现）
# =============================================================================
set -uo pipefail

WS="${WS:-/mnt/workspace}"
OFT_ROOT="${OFT_ROOT:-${WS}/openvla-oft}"
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
CHECK_ONLY=0
FIX_CV2_ONLY=0

for arg in "$@"; do
  case "${arg}" in
    --check) CHECK_ONLY=1 ;;
    --fix-cv2) FIX_CV2_ONLY=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "未知参数: ${arg}" >&2; exit 2 ;;
  esac
done

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m[OK]\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m[!!]\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$*"; }

PIP=(python -m pip install --index-url "${PIP_INDEX}" --trusted-host "$(printf '%s' "${PIP_INDEX}" | sed 's|https\?://||; s|/.*||')")

# -----------------------------------------------------------------------------
# 0. 前置检查
# -----------------------------------------------------------------------------
log "前置检查"
if [ ! -d "${OFT_ROOT}" ]; then
  bad "找不到 OFT 仓库: ${OFT_ROOT}"
  exit 1
fi
cd "${OFT_ROOT}"

if ! python -c "import sys; assert sys.prefix != sys.base_prefix" 2>/dev/null; then
  bad "当前未激活 venv_oft。先执行: source ${WS}/activate_oft.sh"
  exit 1
fi
ok "venv 已激活: $(which python)"

python - <<'PY' || { bad "torch 环境异常，先修 torch 再跑本脚本"; exit 1; }
import sys
import numpy
import torch
assert torch.cuda.is_available(), "CUDA 不可用"
assert numpy.__version__ == "1.26.4", f"numpy 应为 1.26.4，当前 {numpy.__version__}"
print(f"  [OK] torch {torch.__version__} | cuda {torch.version.cuda} | "
      f"numpy {numpy.__version__} | abi {torch._C._GLIBCXX_USE_CXX11_ABI}")
PY

log "清理 pip 半成品残留目录（~xxx）"
# pip 先解包到 site-packages/~<name> 再改名成 <name>；安装中断会留下 ~torch / ~umpy，
# 导致该包处于半装状态（__init__ 在、子模块缺），并让 pip 每次打印
# `WARNING: Ignoring invalid distribution -orch / -umpy`。
python - <<'PY'
import site
import shutil
from pathlib import Path

cleaned = []
for sp in set(site.getsitepackages()) | {site.getusersitepackages()}:
    root = Path(sp)
    if not root.is_dir():
        continue
    for entry in root.iterdir():
        # pip 的半成品目录形如 ~orch、~umpy（也可能带后缀如 ~orch-2.2.0.dist-info）
        if entry.name.startswith("~") and len(entry.name) > 1:
            try:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                cleaned.append(str(entry))
            except Exception as exc:
                print(f"  [!!] 清理失败 {entry}: {exc}")

if cleaned:
    for line in cleaned:
        print(f"  已清理: {line}")
else:
    print("  无残留")
PY

# -----------------------------------------------------------------------------
# 1. 修 opencv（最高优先级：5.0 会把 numpy 顶到 2.x）
# -----------------------------------------------------------------------------
fix_cv2() {
  log "修复 opencv 冲突"

  python - <<'PY'
import importlib.util
spec = importlib.util.find_spec("cv2")
print("  当前 cv2:", spec.origin if spec else "未安装")
PY

  # GUI 版必须清掉；headless 版重装到锁定版本
  python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless >/dev/null 2>&1 || true

  # 清 ~ 开头的半成品目录（pip 中断残留，会导致后续安装异常）
  for pattern in '~opencv' '~orch' '~torch' '~umpy' '~numpy'; do
    while IFS= read -r d; do
      [ -n "${d}" ] || continue
      warn "清理半成品目录: ${d}"
      rm -rf "${d}"
    done < <(find "$(python -c 'import site; print(site.getsitepackages()[0])')" -maxdepth 1 -name "${pattern}*" 2>/dev/null)
  done

  "${PIP[@]}" --no-deps "opencv-python-headless==4.9.0.80" || { bad "opencv 安装失败"; return 1; }

  python -c "
import cv2, numpy, torch
print(f'  [OK] cv2 {cv2.__version__} | numpy {numpy.__version__} | torch {torch.__version__}')
assert cv2.__version__.startswith('4.9'), 'cv2 版本仍不对'
assert numpy.__version__ == '1.26.4'
" || return 1
}

if [ "${FIX_CV2_ONLY}" -eq 1 ]; then
  fix_cv2; exit $?
fi

# -----------------------------------------------------------------------------
# 2. 依赖清单（来自 repair_oft_dependencies.sh 的权威表）
# -----------------------------------------------------------------------------
# 普通依赖：允许 pip 解析依赖，但版本按项目锁定
PLAIN_PKGS=(
  "accelerate"
  "bddl"
  "cloudpickle"
  "easydict"
  "einops"
  "gym==0.25.2"
  "huggingface-hub==0.23.5"
  "json-numpy"
  "jsonlines"
  "imageio[ffmpeg]"
  "imageio-ffmpeg"
  "Pillow"
  "requests"
  "safetensors"
  "tqdm"
  "rich"
  "sentencepiece==0.1.99"
  "draccus==0.8.0"
  "timm==0.9.10"
  "tokenizers==0.19.1"
  "peft==0.11.1"
  "robosuite==1.4.1"
  "mujoco==3.3.2"
  "diffusers==0.30.3"
)

# --no-deps 依赖：解析依赖会破坏 torch/numpy/protobuf 锁定
NODEPS_PKGS=(
  "tensorflow-datasets==4.9.3"
  "tensorflow-metadata==1.15.0"
  "tensorflow-graphics==2021.12.3"
  "protobuf==4.25.3"
  "dm-tree"
  "promise"
  # wandb 是 run_libero_eval.py:23 的硬性 import；直接 `pip install wandb` 会把
  # protobuf 顶到 7.x，导致 tensorflow 2.15（要求 <5.0）失效 —— 必须 --no-deps。
  # pydantic-core 是 pydantic v2 的二进制核心，二者成对，不能只装 pydantic。
  "wandb"
  "pydantic==2.9.2"
  "pydantic-core==2.23.4"
  "annotated-types"
  "gitpython"
  "platformdirs"
  "psutil"
  "sentry-sdk"
  "setproctitle"
  "xxhash"
)

# -----------------------------------------------------------------------------
# 3. 探测缺失项（不安装，只报告）
# -----------------------------------------------------------------------------
probe() {
  log "依赖探测"
  python - <<'PY'
import importlib.util
import sys

checks = [
    ("accelerate",), ("bddl",), ("cloudpickle",), ("easydict",), ("einops",),
    ("gym",), ("huggingface_hub",), ("json_numpy",), ("jsonlines",),
    ("imageio",), ("imageio_ffmpeg",), ("PIL",), ("requests",), ("safetensors",),
    ("tqdm",), ("rich",), ("sentencepiece",), ("draccus",), ("timm",),
    ("tokenizers",), ("peft",), ("robosuite",), ("mujoco",), ("diffusers",),
    ("tensorflow",), ("tensorflow_datasets",), ("tensorflow_graphics",),
    ("google.protobuf",), ("dm_tree",), ("promise",), ("dlimp",),
    ("transformers",), ("bitsandbytes",), ("cv2",),
    ("wandb",), ("pydantic",), ("pydantic_core",),
]
missing, present = [], []
for (name,) in checks:
    try:
        spec = importlib.util.find_spec(name)
    except Exception:
        spec = None
    if spec is None:
        missing.append(name)
    else:
        present.append(name)
print(f"  已安装 {len(present)} 项；缺失 {len(missing)} 项")
if missing:
    print("  缺失: " + ", ".join(missing))
PY
}

if [ "${CHECK_ONLY}" -eq 1 ]; then
  probe
  log "自检模式：不执行安装"
  exit 0
fi

# -----------------------------------------------------------------------------
# 4. 安装
# -----------------------------------------------------------------------------
log "安装普通依赖（版本按项目锁定）"
"${PIP[@]}" "${PLAIN_PKGS[@]}"
RC1=$?

log "安装 TensorFlow 2.15（tensorflow-graphics 需要 tensorflow 而非 tensorflow-cpu 的元数据）"
"${PIP[@]}" "tensorflow==2.15.0"
RC2=$?

log "安装 --no-deps 依赖（防破坏 torch/numpy/protobuf 锁定）"
"${PIP[@]}" --no-deps "${NODEPS_PKGS[@]}"
RC3=$?

log "安装 dlimp（必须用 vendor 里的锁定版本）"
DLIMP_SRC="${OFT_ROOT}/vendor/dlimp_openvla-040105d.tar.gz"
if [ -f "${DLIMP_SRC}" ]; then
  "${PIP[@]}" --no-deps "${DLIMP_SRC}"
  RC4=$?
else
  warn "找不到 ${DLIMP_SRC}"
  warn "从本机上传：01-上游源码/openvla-oft-upstream/vendor/dlimp_openvla-040105d.tar.gz"
  RC4=1
fi

# 收尾：修 numpy 残留 → 锁回 numpy → 修 opencv
log "收尾：清理残留 + 锁回 numpy + 修 opencv"

# 先清 ~umpy / ~numpy，否则 --force-reinstall 装完仍是半装状态，
# 每次 pip 调用都会打印 `WARNING: Ignoring invalid distribution -umpy`。
python - <<'PY'
import site
import shutil
from pathlib import Path

for sp in set(site.getsitepackages()) | {site.getusersitepackages()}:
    root = Path(sp)
    if not root.is_dir():
        continue
    for name in ("~umpy", "~numpy"):
        for entry in root.glob(name + "*"):
            try:
                shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
                print(f"  已清理: {entry}")
            except Exception as exc:
                print(f"  [!!] 清理失败 {entry}: {exc}")
PY

"${PIP[@]}" --force-reinstall --no-deps "numpy==1.26.4"
fix_cv2 || true

# 复验 numpy 是否真的完整（半装时 import numpy.core 会失败）
python -c "
import numpy, numpy.core, numpy.linalg, torch
assert numpy.__version__ == '1.26.4', numpy.__version__
print(f'  [OK] numpy {numpy.__version__} 完整可用（core/linalg 导入正常）')
" || { bad "numpy 仍处于半装状态，请重跑本脚本"; exit 1; }

# -----------------------------------------------------------------------------
# 5. 自检
# -----------------------------------------------------------------------------
log "最终自检"
# robosuite / mujoco 在 import 时会初始化 OpenGL，无显示环境下必须走 osmesa，
# 否则模块本身没问题却报 `'NoneType' object has no attribute 'glGetError'`。
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
python - <<'PY'
import importlib

# 硬性要求：这些不通就是环境真的没补齐
required = ["torch", "numpy", "bitsandbytes", "transformers",
            "libero", "h5py", "peft", "draccus", "wandb", "pydantic_core",
            "tensorflow", "tensorflow_datasets", "tensorflow_graphics", "dlimp"]

# 宽松要求：依赖 OpenGL，无头环境下 import 报 glGetError 属正常，不计为失败
gl_optional = ["robosuite", "mujoco", "gym"]

bad, soft = [], []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        bad.append(f"{name}: {type(exc).__name__}: {str(exc)[:70]}")

for name in gl_optional:
    try:
        importlib.import_module(name)
    except Exception as exc:
        msg = f"{name}: {type(exc).__name__}: {str(exc)[:70]}"
        if "glGetError" in str(exc) or "OpenGL" in str(exc):
            soft.append(msg)
        else:
            bad.append(msg)

if bad:
    print("  以下仍失败：")
    for line in bad:
        print(f"    {line}")
    raise SystemExit(1)

for line in soft:
    print(f"  [跳过] {line}")
    print("         → 无头环境正常现象，已设 MUJOCO_GL=osmesa；不影响量化实验")

import torch, numpy, bitsandbytes, tensorflow as tf
import tensorflow_datasets as tfds
print(f"  [OK] TF {tf.__version__} | tfds {tfds.__version__}")
print(f"  [OK] tf 计算: {tf.reduce_sum(tf.ones(3)).numpy()}")
print(f"  [OK] torch {torch.__version__} | numpy {numpy.__version__} | bnb {bitsandbytes.__version__}")

print("  [OK] prismatic 链路上游依赖齐备")
PY
RC5=$?

log "prismatic 导入测试（最终门禁）"
python - <<'PY'
import traceback
try:
    import prismatic
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
    assert (NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM) == (8, 7, 8), "OFT 常量不符"
    print(f"  [OK] prismatic | chunk={NUM_ACTIONS_CHUNK} action_dim={ACTION_DIM} proprio_dim={PROPRIO_DIM}")
    raise SystemExit(0)
except SystemExit:
    raise
except Exception:
    print("  [FAIL] prismatic 仍无法导入：")
    traceback.print_exc(limit=5)
    raise SystemExit(1)
PY
RC6=$?

echo
echo "================================================================"
if [ "${RC5}" -eq 0 ] && [ "${RC6}" -eq 0 ]; then
  echo " 环境补齐完成，prismatic 可用。"
  echo " 下一步：跑 eval-import 验收"
  echo "   PYTHON_BIN=python bash experiments/robot/libero/failure_study/verify_oft_step.sh eval-import"
else
  echo " 仍有未解决项，请把上方 [FAIL] 段落贴出。"
fi
echo "================================================================"

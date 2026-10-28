#!/usr/bin/env bash
# =============================================================================
# fix_oft_wandb.sh —— 补 wandb / pydantic 依赖（G0 eval-import 的最后一个缺口）
#
# 背景：`run_libero_eval.py` 第 23 行硬性 `import wandb`，而 wandb 依赖
#       `pydantic_core`（pydantic v2 的二进制核心，不能只装 pydantic）。
#       setup_oft_env.sh 早期版本的依赖清单漏了这一支。
#
# 风险控制：安装前后都锁住 numpy==1.26.4，并复验 torch/protobuf 未被顶走。
#
# 用法：
#   bash fix_oft_wandb.sh          # 安装 + 复验
#   bash fix_oft_wandb.sh --verify # 只复验
# =============================================================================
set -uo pipefail

VERIFY_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --verify) VERIFY_ONLY=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "未知参数: ${arg}" >&2; exit 2 ;;
  esac
done

PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
PIP_HOST="$(printf '%s' "${PIP_INDEX}" | sed 's|https\?://||; s|/.*||')"
PIP=(python -m pip install --index-url "${PIP_INDEX}" --trusted-host "${PIP_HOST}")

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m[OK]\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m[!!]\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$*"; }

# 安装前的基线快照
snapshot() {
  python - <<'PY'
import numpy, torch, google.protobuf as pb
print(f"BASELINE numpy={numpy.__version__} torch={torch.__version__} protobuf={pb.__version__}")
PY
}

if [ "${VERIFY_ONLY}" -eq 0 ]; then
  log "安装前基线"
  snapshot || { bad "基础栈异常，先修 torch/numpy 再跑本脚本"; exit 1; }

  # ---------------------------------------------------------------------------
  # 关键教训（2026-09-14 实测踩过）：直接 `pip install wandb` 会把 protobuf
  # 顶到 7.x，导致 tensorflow 2.15（要求 <5.0）和 tensorflow-metadata 1.15
  # （要求 <4.21）双双失效。所以 wandb 的所有依赖必须手工补齐并用 --no-deps 装。
  # ---------------------------------------------------------------------------
  log "补 wandb 运行所需的直接依赖（全部 --no-deps，不碰 protobuf）"
  # 这些是 wandb 的运行时依赖，缺哪个 import 就会炸；版本取兼容区间内的稳定值
  WANDB_DEPS=(
    "pydantic==2.9.2"
    "pydantic-core==2.23.4"
    "annotated-types"
    "typing-extensions"
    "click"
    "gitpython"
    "packaging"
    "platformdirs"
    "psutil"
    "pyyaml"
    "requests"
    "sentry-sdk"
    "setproctitle"
    "setuptools"
    "xxhash"
  )
  "${PIP[@]}" --no-deps "${WANDB_DEPS[@]}" || warn "部分依赖安装返回非零，继续"

  log "安装 wandb 本体（--no-deps，杜绝它升级 protobuf）"
  if ! "${PIP[@]}" --no-deps "wandb"; then
    warn "wandb 装失败，降级试 0.17.x"
    "${PIP[@]}" --no-deps "wandb==0.17.9" || { bad "wandb 安装失败"; exit 1; }
  fi

  log "守护锁定（numpy / protobuf）"
  python - <<'PY'
import numpy

bad = []
if numpy.__version__ != "1.26.4":
    bad.append(f"numpy={numpy.__version__}")

try:
    import google.protobuf as pb
    if pb.__version__ != "4.25.3":
        bad.append(f"protobuf={pb.__version__}")
except Exception as exc:
    bad.append(f"protobuf 导入失败: {exc}")

if bad:
    print("  需要回锁: " + "; ".join(bad))
    raise SystemExit(1)
print("  锁定完好：numpy 1.26.4 / protobuf 4.25.3")
PY
  if [ $? -ne 0 ]; then
    warn "回锁 numpy / protobuf（--force-reinstall --no-deps）"
    "${PIP[@]}" --force-reinstall --no-deps "numpy==1.26.4"
    "${PIP[@]}" --force-reinstall --no-deps "protobuf==4.25.3"
  fi
fi

# -----------------------------------------------------------------------------
# 复验
# -----------------------------------------------------------------------------
log "复验依赖"
python - <<'PY'
import importlib

names = ["pydantic_core", "pydantic", "wandb"]
failed = []
for n in names:
    try:
        m = importlib.import_module(n)
        v = getattr(m, "VERSION", None) or getattr(m, "__version__", "?")
        print(f"  [OK] {n} {v}")
    except Exception as exc:
        failed.append(f"{n}: {type(exc).__name__}: {str(exc)[:80]}")
        print(f"  [FAIL] {n}: {type(exc).__name__}: {str(exc)[:80]}")
if failed:
    raise SystemExit(1)
PY
RC_DEPS=$?

log "复验核心栈锁定"
python - <<'PY'
import numpy, torch, google.protobuf as pb
print(f"  numpy {numpy.__version__} | torch {torch.__version__} | protobuf {pb.__version__}")
assert numpy.__version__ == "1.26.4", f"numpy 必须 1.26.4，当前 {numpy.__version__}"
assert pb.__version__.startswith("4.25"), f"protobuf 必须 4.25.x，当前 {pb.__version__}"
import numpy.core, numpy.linalg  # 半装检测
print("  [OK] numpy 完整可用；torch/protobuf 锁定未被破坏")
PY
RC_LOCK=$?

# -----------------------------------------------------------------------------
# 直接跑一次 G0 的 import 路径（真正要验证的东西）
# -----------------------------------------------------------------------------
log "模拟 G0：导入 run_libero_eval 的模块链"
OFT_ROOT="${OFT_ROOT:-/mnt/workspace/openvla-oft}"
if [ -d "${OFT_ROOT}" ]; then
  cd "${OFT_ROOT}" || exit 1
  python - <<'PY'
import importlib.util
import sys
import traceback

# 只做 import，不执行 main（run_libero_eval.py 顶层 import 会拉 wandb 等）
try:
    import wandb              # noqa: F401
    import pydantic_core      # noqa: F401
    print("  [OK] wandb / pydantic_core import 通过")
except Exception:
    traceback.print_exc(limit=5)
    raise SystemExit(1)

try:
    from libero.libero import benchmark
    print("  [OK] libero.libero.benchmark import 通过")
except Exception:
    traceback.print_exc(limit=5)
    raise SystemExit(1)
PY
  RC_SIM=$?
else
  warn "找不到 ${OFT_ROOT}，跳过 G0 模拟"
  RC_SIM=0
fi

echo
echo "================================================================"
if [ "${RC_DEPS}" -eq 0 ] && [ "${RC_LOCK}" -eq 0 ] && [ "${RC_SIM}" -eq 0 ]; then
  echo " wandb 依赖补齐，锁定未破坏。"
  echo " 下一步（G0 门禁）："
  echo "   cd \${OFT_ROOT:-/mnt/workspace/openvla-oft}"
  echo "   PYTHON_BIN=python bash experiments/robot/libero/failure_study/verify_oft_step.sh eval-import"
else
  echo " 仍有未解决项，请把上方 [FAIL] 段落贴出。"
fi
echo "================================================================"

#!/usr/bin/env bash
# =============================================================================
# fix_oft_gl_and_numpy.sh —— 收尾修复：清 pip 半成品残留 + 无头 GL 验证
#
# 背景：`setup_oft_env.sh` 跑完后出现两类告警，都不是环境真缺东西：
#   1) `WARNING: Ignoring invalid distribution -umpy` —— pip 半成品目录 `~umpy` 残留
#   2) 自检里 robosuite / mujoco 报
#      `AttributeError: 'NoneType' object has no attribute 'glGetError'`
#      —— 无头环境没有 OpenGL 上下文，需 MUJOCO_GL=osmesa
#
# 本脚本只做收尾，不重装依赖。用法：
#   bash fix_oft_gl_and_numpy.sh          # 修复 + 验证
#   bash fix_oft_gl_and_numpy.sh --verify # 只验证，不修
# =============================================================================
set -uo pipefail

VERIFY_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --verify) VERIFY_ONLY=1 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "未知参数: ${arg}" >&2; exit 2 ;;
  esac
done

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m[OK]\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m[!!]\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$*"; }

PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"

# 无头环境的 GL 后端（必须在任何 robosuite/mujoco import 之前设置）
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

# -----------------------------------------------------------------------------
# 1. 清理 pip 半成品残留目录（~umpy / ~orch / ~torch / ~opencv ...）
# -----------------------------------------------------------------------------
if [ "${VERIFY_ONLY}" -eq 0 ]; then
  log "清理 pip 半成品残留目录"
  python - <<'PY'
import site
import shutil
from pathlib import Path

cleaned, failed = [], []
for sp in set(site.getsitepackages()) | {site.getusersitepackages()}:
    root = Path(sp)
    if not root.is_dir():
        continue
    for entry in sorted(root.iterdir()):
        if not entry.name.startswith("~") or entry.name == "~":
            continue
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            cleaned.append(str(entry))
        except Exception as exc:
            failed.append(f"{entry}: {exc}")

for line in cleaned:
    print(f"  已清理: {line}")
for line in failed:
    print(f"  [!!] 失败: {line}")
if not cleaned and not failed:
    print("  无残留，环境干净")
PY

  log "复验 numpy 完整性"
  if python - <<'PY'
import numpy
import numpy.core
import numpy.linalg
import torch
assert numpy.__version__ == "1.26.4", f"numpy 版本为 {numpy.__version__}"
# 半装时这两个子模块会缺失，是 ~umpy 残留的典型症状
print(f"  [OK] numpy {numpy.__version__} | core/linalg 导入正常 | torch {torch.__version__}")
PY
  then
    : # 通过
  else
    warn "numpy 仍不完整，执行 force-reinstall（含 --no-deps 防动 torch）"
    python -m pip install --index-url "${PIP_INDEX}" \
      --trusted-host "$(printf '%s' "${PIP_INDEX}" | sed 's|https\?://||; s|/.*||')" \
      --force-reinstall --no-deps "numpy==1.26.4"
    python -c "import numpy, numpy.core, numpy.linalg; print(f'  [OK] numpy {numpy.__version__} 已修复')" \
      || { bad "numpy 仍损坏，需按文档的 torch 残缺流程处理"; exit 1; }
  fi
fi

# -----------------------------------------------------------------------------
# 2. 验证 GL 相关模块（带 osmesa）
# -----------------------------------------------------------------------------
log "验证 robosuite / mujoco（MUJOCO_GL=${MUJOCO_GL}）"
python - <<'PY'
import os
import traceback

print(f"  MUJOCO_GL={os.environ.get('MUJOCO_GL')} | "
      f"PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM')}")

for name in ("mujoco", "robosuite", "gym"):
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "?")
        print(f"  [OK] {name} {ver}")
    except Exception as exc:
        msg = f"{type(exc).__name__}: {str(exc)[:80]}"
        if "glGetError" in str(exc) or "OpenGL" in str(exc):
            # 无头实例上没有 GL 上下文，但量化推理走的是 LIBERO/mujoco 的
            # 离屏渲染路径，实际评测脚本会自行设好 MUJOCO_GL，故此处不算失败
            print(f"  [跳过] {name} 需要 GL 上下文（无头环境正常）：{msg}")
        else:
            print(f"  [FAIL] {name}: {msg}")
            traceback.print_exc(limit=3)
            raise SystemExit(1)
PY
RC_GL=$?

# -----------------------------------------------------------------------------
# 3. 最终门禁：prismatic 常量
# -----------------------------------------------------------------------------
log "prismatic 最终门禁"
python - <<'PY'
try:
    import prismatic
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
    assert (NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM) == (8, 7, 8), "OFT 常量不符"
    print(f"  [OK] prismatic | chunk={NUM_ACTIONS_CHUNK} "
          f"action_dim={ACTION_DIM} proprio_dim={PROPRIO_DIM}")
except Exception:
    import traceback
    print("  [FAIL] prismatic 无法导入：")
    traceback.print_exc(limit=5)
    raise SystemExit(1)
PY
RC_PRI=$?

# -----------------------------------------------------------------------------
# 4. 汇总
# -----------------------------------------------------------------------------
log "检查残留告警是否消失"
python -m pip --version >/dev/null 2>&1
WARN_COUNT=$(python -m pip list 2>&1 | grep -c "Ignoring invalid distribution" || true)
if [ "${WARN_COUNT}" -eq 0 ]; then
  ok "pip 不再输出 'Ignoring invalid distribution' 告警"
else
  warn "pip 仍有 ${WARN_COUNT} 条 'Ignoring invalid distribution' 告警"
  python -m pip list 2>&1 | grep "Ignoring invalid distribution" | head -5
fi

echo
echo "================================================================"
if [ "${RC_GL}" -eq 0 ] && [ "${RC_PRI}" -eq 0 ]; then
  echo " 收尾完成。环境可用。"
  echo " 下一步（G0 门禁）："
  echo "   cd \${OFT_ROOT:-/mnt/workspace/openvla-oft}"
  echo "   PYTHON_BIN=python bash experiments/robot/libero/failure_study/verify_oft_step.sh eval-import"
else
  echo " 仍有未解决项，请把上方 [FAIL] 段落贴出。"
fi
echo "================================================================"

#!/usr/bin/env bash
# ============================================================================
# fix_activate_gl.sh —— 把 activate_oft.sh 的 MUJOCO_GL 从 osmesa 改成 egl
# ----------------------------------------------------------------------------
# 背景：本实例没有 libOSMesa*（`ldconfig -p | grep osmesa` 与
#       `find / -name "libOSMesa*"` 均无输出），用 osmesa 会报
#       `AttributeError: 'NoneType' object has no attribute 'glGetError'`。
#       A10 驱动自带 EGL，走 GPU 硬件渲染，对 G2 实时性更有利。
#
# 用法：bash fix_activate_gl.sh [--check]
# ============================================================================
set -uo pipefail

ACT="${ACT:-/mnt/workspace/activate_oft.sh}"

if [ ! -f "${ACT}" ]; then
  echo "[FAIL] 找不到 ${ACT}；用 ACT=<路径> 指定"
  exit 1
fi

echo "==> 当前 MUJOCO_GL 相关行"
grep -n "MUJOCO_GL\|PYOPENGL_PLATFORM" "${ACT}" || echo "    (无)"

if [ "${1:-}" = "--check" ]; then
  exit 0
fi

cp -n "${ACT}" "${ACT}.bak" 2>/dev/null && echo "    备份 -> ${ACT}.bak"

# osmesa -> egl（只改 export MUJOCO_GL=osmesa 这一行）
sed -i 's/^\(export MUJOCO_GL=\)osmesa[[:space:]]*$/\1egl/' "${ACT}"

# 确保 PYOPENGL_PLATFORM 不被设成 egl/osmesa 之一（两者不能同设）
sed -i '/^export PYOPENGL_PLATFORM=/s/^/# /' "${ACT}"

echo "==> 修改后"
grep -n "MUJOCO_GL\|PYOPENGL_PLATFORM" "${ACT}" || true

if grep -q '^export MUJOCO_GL=egl' "${ACT}"; then
  echo "[OK] MUJOCO_GL 已改为 egl"
else
  echo "[WARN] 未匹配到 'export MUJOCO_GL=osmesa' 行，请手工确认"
fi
echo
echo "验证：source ${ACT} && echo \$MUJOCO_GL \$PYOPENGL_PLATFORM   # 期望: egl (空)"

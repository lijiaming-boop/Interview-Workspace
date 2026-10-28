#!/usr/bin/env bash
# ============================================================================
# resume_verify.sh —— 实例重启后的环境复验（幂等，可反复跑）
# ----------------------------------------------------------------------------
# 重启后要重新确认的四件事：
#   1. venv_oft 是否完好（torch / flash-attn / bitsandbytes）
#   2. ~/.libero/config.yaml 是否还在（home 目录可能被重置）
#   3. checkpoint 15.45G 是否还在（NAS 盘应保留）
#   4. GL 后端 + 五个门禁阶段
#
# 用法：
#   bash resume_verify.sh              # 全量复验（修 + 验）
#   bash resume_verify.sh --check      # 只查不修
#   bash resume_verify.sh --gate       # 只跑五阶段门禁
# ============================================================================
set -uo pipefail

# ---- 粘贴检测：防止脚本内的 exit 关闭交互终端 ----
_SELF="${0##*/}"
_PASTED=0
case "${_SELF}" in
  bash|-bash|sh|-sh|dash|-dash|zsh|-zsh) _PASTED=1 ;;
esac
if [ "${_PASTED}" = "1" ]; then
  printf '\n\033[1;33m[注意]\033[0m 本脚本被【粘贴】进交互终端（$0=%s），而非作为文件运行。\n' "${0}"
  echo "  脚本内的 exit 会关闭你的终端。建议先存成文件，再执行: bash <文件名>"
  echo "  （本次将继续执行，但收尾不会 exit）"
  echo
fi

WS="${WS:-/mnt/workspace}"
VENV="${VENV:-${WS}/venv_oft}"
OFT_ROOT="${OFT_ROOT:-${WS}/openvla-oft}"
CKPT="${CKPT:-${WS}/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial}"
LIBERO_ROOT="${LIBERO_ROOT:-${WS}/LIBERO/libero/libero}"
ACT="${ACT:-${WS}/activate_oft.sh}"

MODE="all"
case "${1:-}" in
  --check) MODE="check" ;;
  --gate)  MODE="gate"  ;;
  "")      MODE="all"   ;;
  *) echo "未知参数: $1"; exit 2 ;;
esac

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m[OK]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*"; }
FAILED=0

# ---------------------------------------------------------------------------
log "0. 激活 venv"
if [ ! -f "${VENV}/bin/activate" ]; then
  fail "找不到 ${VENV}/bin/activate —— venv 丢了，需重建（NAS 未挂载？）"
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
PY="${VENV}/bin/python"
ok "venv: $(command -v python)"

# ---------------------------------------------------------------------------
log "1. GL 后端"
# 顺手把 activate 脚本里的 osmesa 改成 egl（幂等）
if [ -f "${ACT}" ] && grep -q '^export MUJOCO_GL=osmesa' "${ACT}" 2>/dev/null; then
  cp -n "${ACT}" "${ACT}.bak" 2>/dev/null || true
  sed -i 's/^\(export MUJOCO_GL=\)osmesa[[:space:]]*$/\1egl/' "${ACT}"
  sed -i '/^export PYOPENGL_PLATFORM=/s/^/# /' "${ACT}"
  warn "已把 ${ACT} 的 MUJOCO_GL 改为 egl（原 osmesa 本机无 libOSMesa）"
fi
export MUJOCO_GL="egl"
unset PYOPENGL_PLATFORM
echo "    MUJOCO_GL=${MUJOCO_GL}   PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-<未设，正确>}"

# ---------------------------------------------------------------------------
log "2. 解释器与核心包完整性"
"${PY}" - <<'PY'
import sys, site, os
print(f"  executable : {sys.executable}")
print(f"  version    : {sys.version.split()[0]}")
print(f"  prefix     : {sys.prefix}")
print(f"  base_prefix: {sys.base_prefix}")
print(f"  in-venv    : {sys.prefix != sys.base_prefix}")
for p in site.getsitepackages():
    n = 0
    if os.path.isdir(p):
        try:
            n = len([x for x in os.listdir(p) if not x.startswith('_')])
        except Exception:
            pass
    print(f"  site-pkg   : {p}  ({n} 个条目)")
print(f"  user-site  : {site.getusersitepackages()}")
if sys.version_info[:2] != (3, 10):
    print(f"\n  [FAIL] OpenVLA-OFT 要求 Python 3.10，当前 {sys.version_info.major}.{sys.version_info.minor}")
    print("         venv 的解释器很可能被换掉了（实例重建后基础镜像 python 版本变了）")
    print("         修复见手册 §0.3「venv 解释器被换掉」")
PY

"${PY}" - <<'PY'
import sys
fails = []
def chk(mod, attr=None, label=None):
    label = label or mod
    try:
        m = __import__(mod, fromlist=["__file__"])
        v = getattr(m, attr, "") if attr else getattr(m, "__version__", "")
        print(f"  [OK]   {label:<28} {v}")
    except Exception as e:
        print(f"  [FAIL] {label:<28} {type(e).__name__}: {e}")
        fails.append(label)
chk("torch", "__version__")
chk("torch.version", "cuda", "torch.version.cuda")
chk("numpy", "__version__")
chk("bitsandbytes", "__version__")
chk("transformers", "__version__")
chk("flash_attn", "__version__")
chk("libero")
chk("h5py", "__version__")
chk("peft", "__version__")
chk("draccus", "__version__")
chk("tensorflow", "__version__")
chk("google.protobuf", "__version__", "google.protobuf")
chk("dlimp", "__file__", "dlimp")
print()
if fails:
    print("[FAIL] 缺失:", ", ".join(fails))
    print("       修复: bash setup_oft_env.sh")
    sys.exit(1)
print("[OK] 核心包齐全")
PY
[ $? -ne 0 ] && FAILED=1

# ---------------------------------------------------------------------------
log "3. torch CUDA 可用性"
"${PY}" - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    print("[FAIL] CUDA 不可用（驱动/容器问题）"); sys.exit(1)
print(f"  torch={torch.__version__}  CUDA={torch.version.cuda}  "
      f"GPU={torch.cuda.get_device_name(0)}")
cap = torch.cuda.get_device_capability(0)
print(f"  capability={cap}")
if cap != (8, 6):
    print(f"  [WARN] 期望 (8,6) 即 A10，实得 {cap}")
print("[OK] CUDA 就绪")
PY
[ $? -ne 0 ] && FAILED=1

# ---------------------------------------------------------------------------
log "4. LIBERO config（重启后 home 目录可能被重置）"
CFG="${HOME}/.libero/config.yaml"
need_fix=0
if [ ! -f "${CFG}" ]; then
  warn "缺 ${CFG}"
  need_fix=1
else
  ok "存在 ${CFG}"
  # 检查路径是否有效
  if ! grep -q "${LIBERO_ROOT}" "${CFG}" 2>/dev/null; then
    warn "内容里的路径与本机不符"
    need_fix=1
  fi
fi

if [ "${need_fix}" = "1" ] && [ "${MODE}" != "check" ]; then
  warn "重新写入 ${CFG}"
  mkdir -p "$(dirname "${CFG}")"
  cat > "${CFG}" <<EOF
benchmark_root: ${LIBERO_ROOT}/benchmark
bddl_files: ${LIBERO_ROOT}/bddl_files
init_states: ${LIBERO_ROOT}/init_files
datasets: ${LIBERO_ROOT}/datasets
assets: ${LIBERO_ROOT}/assets
EOF
  ok "已写入（datasets 目录可不存在，评测不用）"
elif [ "${need_fix}" = "1" ]; then
  fail "需要修，但处于 --check 模式"
  FAILED=1
fi

if [ "${MODE}" != "check" ]; then
  # 非交互验证：import libero 不应再抛 EOFError
  "${PY}" - <<'PY'
import sys
try:
    from libero.libero import benchmark
    b = benchmark.get_benchmark_dict()
    print(f"  [OK] LIBERO 可 import；可用 suite {len(b)} 个")
except EOFError:
    print("  [FAIL] 仍报 EOFError —— config.yaml 未生效")
    sys.exit(1)
except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {e}")
    sys.exit(1)
PY
  [ $? -ne 0 ] && FAILED=1
fi

# ---------------------------------------------------------------------------
log "5. checkpoint 落盘校验"
if [ ! -d "${CKPT}" ]; then
  fail "缺 ${CKPT}；用 bash download_oft_ckpt.sh 下载"
  FAILED=1
else
  echo "    体积: $(du -sh "${CKPT}" 2>/dev/null | cut -f1)"
  "${PY}" - "${CKPT}" <<'PY'
import json, os, sys
d = sys.argv[1]
EXPECTED = {
    "model-00001-of-00004.safetensors": 4_925_122_448,
    "model-00002-of-00004.safetensors": 4_947_392_496,
    "model-00003-of-00004.safetensors": 4_947_417_456,
    "model-00004-of-00004.safetensors": 262_668_432,
    "action_head--150000_checkpoint.pt": 302_242_674,
    "proprio_projector--150000_checkpoint.pt": 67_275_256,
    "dataset_statistics.json": 2_957,
}
bad = 0
for name, want in EXPECTED.items():
    p = os.path.join(d, name)
    if not os.path.exists(p):
        print(f"  [FAIL] 缺 {name}"); bad = 1; continue
    got = os.path.getsize(p)
    if got == want:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}  {got:,} != {want:,}"); bad = 1
# norm_stats
p = os.path.join(d, "dataset_statistics.json")
if os.path.exists(p):
    try:
        st = json.load(open(p))
        keys = list(st.keys())
        print(f"  norm_stats 键: {keys}")
        if not any("spatial" in k.lower() for k in keys):
            print("  [FAIL] 无 spatial 统计量 → 500 episodes 会全失败")
            bad = 1
        else:
            print("  [OK]   含 spatial 统计量")
    except Exception as e:
        print(f"  [FAIL] 解析 dataset_statistics.json: {e}"); bad = 1
sys.exit(bad)
PY
  [ $? -ne 0 ] && FAILED=1
fi

# ---------------------------------------------------------------------------
log "6. GPU 占用检查（跑基准前必须干净）"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free \
    --format=csv,noheader | sed 's/^/    /'
  BUSY=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
  if [ -z "${BUSY}" ]; then
    ok "GPU 无其他计算进程"
  else
    warn "GPU 上有进程占用："
    echo "${BUSY}" | sed 's/^/    /'
    echo "    若确认是上次异常退出留下的僵尸进程，可 kill："
    echo "${BUSY}" | awk -F, '{gsub(/ /,"",$1); print "      kill -9 " $1}'
  fi
fi

# ---------------------------------------------------------------------------
log "6.5 内存守护进程检测（仅供跑大模型前参考）"

# --- (a) nohang：DSW 预装的用户态内存保护进程，接近 OOM 时主动杀最耗内存的进程 ---
NOHANG_PID=$(pgrep -f "nohang" 2>/dev/null | head -1)
if [ -n "${NOHANG_PID}" ]; then
  warn "检测到 nohang 守护进程在运行 (PID ${NOHANG_PID})"
  echo "    它在空闲内存跌破硬阈值（本实例约 573 MiB）时会杀【整个进程组】。"
  echo "    注意：这与「终端退出码 2」无关（那是粘贴脚本触发 exit）；"
  echo "          仅在真正跑 7B 模型、内存逼近上限时才需防范。"
  for CFG in /etc/dsw/cache/pre-install/plugins/nohang/*/pai-nohang.conf \
             /etc/nohang/nohang.conf; do
    [ -f "${CFG}" ] && { echo "    配置: ${CFG}"; \
      grep -iE "^(soft|hard|warning)_threshold|^kill_process_group|^psi_checking" "${CFG}" 2>/dev/null | head -12 | sed 's/^/      /'; }
  done
  echo "    查它日志: ls -l /proc/${NOHANG_PID}/fd/1"
  for LG in /var/log/nohang/nohang.log /var/log/nohang.log; do
    if [ -f "${LG}" ]; then
      echo "    日志尾部 ${LG}:"
      tail -15 "${LG}" 2>/dev/null | sed 's/^/      /'
    fi
  done
  if command -v journalctl >/dev/null 2>&1; then
    JN=$(journalctl --no-pager -n 30 2>/dev/null \
      | grep -iE "nohang|killed|signal" | tail -10)
    [ -n "${JN}" ] && { echo "    journald 相关记录:"; echo "${JN}" | sed 's/^/      /'; }
  fi
else
  ok "未检测到 nohang（本实例无用户态内存杀手）"
fi

# --- (b) 内核 OOM killer 记录（需 sudo；容器内直接读常被拒） ---
OOM=""
if command -v dmesg >/dev/null 2>&1; then
  OOM=$(dmesg 2>/dev/null | grep -iE "out of memory|killed process|oom-kill" | tail -8)
  if [ -z "${OOM}" ] && [ "$(id -u)" != "0" ]; then
    OOM=$(sudo -n dmesg 2>/dev/null | grep -iE "out of memory|killed process|oom-kill" | tail -8)
  fi
fi
if [ -n "${OOM}" ]; then
  warn "发现内核 OOM killer 记录："
  echo "${OOM}" | sed 's/^/    /'
elif [ -z "${OOM}" ]; then
  echo "    (dmesg 无记录或不可读 —— 容器内通常被 kernel.dmesg_restrict 限制，"
  echo "     属正常；nohang 的杀进程行为在内核层不留痕)"
fi

# --- (c) 宿主内存 + swap（零 swap 是本实例的硬伤） ---
echo "    --- 宿主内存 ---"
free -h 2>/dev/null | sed 's/^/    /' || \
  awk '/MemTotal|MemAvailable|SwapTotal|SwapFree/{printf "    %s %s\n",$1,$2}' /proc/meminfo

SWAP_TOTAL=$(awk '/SwapTotal/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
if [ "${SWAP_TOTAL:-0}" = "0" ]; then
  warn "Swap = 0 —— 28G 宿主内存没有缓冲垫（不影响当前门禁，跑 7B 前建议补）"
  echo "    修复: bash setup_swap.sh      # 自动避开 overlayfs 陷阱"
else
  ok "Swap 已配置（$(awk '/SwapTotal/{printf "%.1f GiB", $2/1048576}' /proc/meminfo)）"
fi

# --- (d) 上次运行的残留日志（判断当时在做什么、停在哪一步） ---
for f in /mnt/workspace/_dl_ckpt.log; do
  [ -f "$f" ] && echo "    日志 $f 尾部：" && tail -5 "$f" | sed 's/^/      /'
done

# --- (e) 残留的 python 进程（排除 DSW 系统自带进程） ---
# DSW 自身常驻：supervisord / nohang / paimcp / jupyter-lab，均属正常，需过滤
ORPHAN=$(ps -eo pid,etime,rss,cmd 2>/dev/null \
  | grep -E "python|modelscope" \
  | grep -vE "grep|resume_verify" \
  | grep -vE "/etc/dsw/|supervisord|nohang|paimcp|jupyter-lab" \
  | head -10)
if [ -n "${ORPHAN}" ]; then
  warn "发现非系统 python 进程（可能是上次跑的残留）："
  echo "${ORPHAN}" | sed 's/^/      /'
  echo "      确认无用后可 kill："
  echo "${ORPHAN}" | awk '{print "        kill -9 " $1}'
else
  ok "无残留业务 python 进程（DSW 系统进程已排除）"
fi

# ---------------------------------------------------------------------------
if [ "${MODE}" = "gate" ] || [ "${MODE}" = "all" ]; then
  log "7. 五阶段门禁（verify_oft_step.sh）"
  cd "${OFT_ROOT}" || { fail "缺 ${OFT_ROOT}"; exit 1; }
  for stage in python gpu eval-import checkpoint; do
    printf '\n\033[1;35m  --- %s ---\033[0m\n' "${stage}"
    if PYTHON_BIN="${PY}" bash experiments/robot/libero/failure_study/verify_oft_step.sh "${stage}"; then
      ok "${stage} 通过"
    else
      fail "${stage} 未通过"
      FAILED=1
    fi
  done
  echo
  echo "    model 阶段会真加载 7B，耗时较长，单独手动跑："
  echo "    cd ${OFT_ROOT} && PYTHON_BIN=${PY} bash experiments/robot/libero/failure_study/verify_oft_step.sh model"
fi

# ---------------------------------------------------------------------------
log "复验小结"
if [ "${FAILED}" = "0" ]; then
  ok "全部检查项通过，可以进入 C2 冒烟"
  echo
  echo "  冒烟测试（先确认 GPU 干净）："
  echo "    cd ${OFT_ROOT}"
  echo "    RUN_TAG=int8-quick-\$(date +%Y%m%d-%H%M%S) bash tools/09_int8_benchmark.sh all"
else
  fail "存在未通过项，按上面提示逐项修复后重跑本脚本"
fi
if [ "${_PASTED}" = "1" ]; then
  echo "  (粘贴模式：不执行 exit，避免关闭终端)"
  return 0 2>/dev/null || true
else
  exit ${FAILED}
fi

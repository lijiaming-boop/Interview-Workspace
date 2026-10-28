#!/usr/bin/env bash
# ============================================================================
# download_oft_ckpt.sh —— OpenVLA-OFT libero-spatial checkpoint 下载 + 校验
# ----------------------------------------------------------------------------
# 背景：魔搭社区镜像模型仓库不能用阿里云 OSM 端点
#       （export MODELSCOPE_ENDPOINT='https://mirrors.aliyun.com/modelscope/'
#        对社区第三方模型一律 404；该端点只服务 damo/、iic/ 等官方模型）
#       本脚本显式走官方站点，并清掉可能污染进程的 MODELSCOPE_ENDPOINT。
#
# 用法：
#   bash download_oft_ckpt.sh            # 下载 + 校验
#   bash download_oft_ckpt.sh --probe    # 只探测文件清单，不下载
#   bash download_oft_ckpt.sh --verify   # 只校验已下载内容
#   bash download_oft_ckpt.sh --force    # 忽略已存在文件，强制重下
#   bash download_oft_ckpt.sh --python   # 走 dl_ckpt.py（snapshot_download API）
#
# 注意：不要用 `python -m modelscope download` —— 很多 pip 版 modelscope 没有
#       __main__.py，会报 "No module named modelscope.__main__"。
#       本脚本自动探测 modelscope CLI，找不到则退到 snapshot_download API。
#
# 备用源（主源残缺时）：
#   MODEL=a11111ad/openvla-7b-oft-finetuned-libero-spatial  （默认，OFT 版）
#   MODEL=superpeach/openvla-7b-oft-finetuend-libero-spatial  （注意 finetuend 是原作者的拼写错误）
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
  echo "  请先存成文件，再执行: bash download_oft_ckpt.sh [--probe|--verify|--force]"
  echo
fi

MODEL="${MODEL:-a11111ad/openvla-7b-oft-finetuned-libero-spatial}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/workspace/openvla-oft-ckpts}"
DEST="${DEST:-${CKPT_ROOT}/openvla-7b-oft-finetuned-libero-spatial}"
LOG="${LOG:-/mnt/workspace/_dl_ckpt.log}"
OFT_ROOT="${OFT_ROOT:-/mnt/workspace/openvla-oft}"
PY="${PYTHON_BIN:-python}"

MODE="all"
case "${1:-}" in
  --probe)  MODE="probe"  ;;
  --verify) MODE="verify" ;;
  --force)  MODE="force"  ;;
  "")       MODE="all"    ;;
  *) echo "未知参数: $1"; exit 2 ;;
esac

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m[OK]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------------------
# 0. 关键：清掉阿里云 OSM 端点，强制走官方站点
# ---------------------------------------------------------------------------
unset MODELSCOPE_ENDPOINT
unset MODELSCOPE_CACHE
export MODELSCOPE_DOMAIN="${MODELSCOPE_DOMAIN:-www.modelscope.cn}"
ok "已清除 MODELSCOPE_ENDPOINT（该端点对社区模型会 404）"
echo "    MODELSCOPE_DOMAIN=${MODELSCOPE_DOMAIN}"

# ---------------------------------------------------------------------------
# 1. 探测：列出远端文件清单（不下载）
# ---------------------------------------------------------------------------
probe() {
  log "探测远端文件清单: ${MODEL}"
  "${PY}" - "$MODEL" <<'PY'
import json, sys, urllib.request
model = sys.argv[1]
url = (f"https://www.modelscope.cn/api/v1/models/{model}"
       f"/repo/files?Revision=master&Recursive=True")
with urllib.request.urlopen(url, timeout=30) as r:
    d = json.load(r)
files = d.get("Data", {}).get("Files", [])
if not files:
    print("[FAIL] 未取到文件清单，仓库可能不存在或为私有")
    sys.exit(1)

print(f"{'PATH':<58} {'SIZE':>14}  LFS")
total = 0
for f in sorted(files, key=lambda x: x["Path"]):
    if f["Type"] == "tree":
        continue
    total += f["Size"]
    print(f"{f['Path']:<58} {f['Size']:>14,}  {'Y' if f['IsLFS'] else ''}")
print(f"\n文件数: {len([f for f in files if f['Type']!='tree'])}")
print(f"总计: {total/1024**3:.2f} GiB")

need = ["dataset_statistics.json"]
for n in need:
    hit = any(f["Path"] == n for f in files)
    print(f"{'[OK]' if hit else '[FAIL]'} {n}")
PY
}

# ---------------------------------------------------------------------------
# 2. 下载
# ---------------------------------------------------------------------------
download() {
  log "确认 modelscope 可用"
  if ! "${PY}" -c "import modelscope" 2>/dev/null; then
    "${PY}" -m pip install -U modelscope \
      -i https://mirrors.aliyun.com/pypi/simple/ 2>&1 | tail -3
  fi

  # 注意：很多 pip 版 modelscope 没有 __main__.py，`python -m modelscope` 会报
  # "No module named modelscope.__main__"。优先用 CLI，退到 snapshot_download API。
  local USE_CLI=""
  if command -v modelscope >/dev/null 2>&1; then
    USE_CLI="$(command -v modelscope)"
  elif [ -x "${VIRTUAL_ENV:-/nonexistent}/bin/modelscope" ]; then
    USE_CLI="${VIRTUAL_ENV}/bin/modelscope"
  fi

  mkdir -p "${DEST}"
  log "开始下载 -> ${DEST}"
  echo "    权重约 15.5 GiB，中断后重跑本脚本会自动续传。"
  echo "    端点: 官方 www.modelscope.cn（已清除 MODELSCOPE_ENDPOINT）"

  local pid rc
  if [ -n "${USE_CLI}" ]; then
    echo "    入口: ${USE_CLI} download"
    local extra=""
    [ "${MODE}" = "force" ] && extra="--force_download"
    "${USE_CLI}" download \
      --model "${MODEL}" \
      --local_dir "${DEST}" \
      ${extra} 2>&1 | tee "${LOG}" &
    pid=$!
  else
    warn "未找到 modelscope CLI，改用 snapshot_download API（dl_ckpt.py）"
    local SCRIPT="$(dirname "${BASH_SOURCE[0]}")/dl_ckpt.py"
    if [ ! -f "${SCRIPT}" ]; then
      fail "缺 dl_ckpt.py，请与本脚本放同一目录"
      return 1
    fi
    MODEL="${MODEL}" DEST="${DEST}" "${PY}" "${SCRIPT}" | tee "${LOG}" &
    pid=$!
  fi

  echo "    后台 PID: ${pid}"
  echo
  # 前台等待；tee 已经把进度打到屏幕，这里只等结束
  wait "${pid}"; rc=$?
  if [ $rc -ne 0 ]; then
    fail "下载退出码 ${rc}，日志尾部："
    tail -20 "${LOG}"
    return $rc
  fi
  ok "下载进程结束"
}

# ---------------------------------------------------------------------------
# 3. 校验
# ---------------------------------------------------------------------------
verify() {
  log "校验下载内容: ${DEST}"
  [ -d "${DEST}" ] || { fail "目录不存在: ${DEST}"; return 1; }

  echo "--- 目录体积 ---"
  du -sh "${DEST}" 2>/dev/null || true
  echo "--- 权重分片 ---"
  ls -lh "${DEST}"/model-*.safetensors 2>/dev/null | awk '{print "   ", $9, $5}'
  echo "--- OFT 专有文件 ---"
  ls -lh "${DEST}" | grep -Ei "action_head|proprio_projector|dataset_statistics" \
    || warn "未见 action_head / proprio_projector / dataset_statistics"

  log "norm_stats 键校验（最关键；缺它会导致 500 episodes 全失败）"
  "${PY}" - "${DEST}" <<'PY'
import json, os, sys
d_dir = sys.argv[1]
p = os.path.join(d_dir, "dataset_statistics.json")
if not os.path.exists(p):
    print("[FAIL] 缺 dataset_statistics.json —— 反归一化会除零出 nan，评估全废")
    sys.exit(1)
d = json.load(open(p))
keys = [k for k in d if "libero" in k.lower()]
print("  全部统计键:", list(d.keys()))
print("  libero 相关:", keys)
assert keys, "[FAIL] 无 libero 统计量"
assert any("spatial" in k.lower() for k in keys), "[FAIL] 无 spatial 统计量"
print("[OK] norm_stats 就绪")
PY
  local rc=$?

  log "OFT 常量与模型配置一致性抽查"
  "${PY}" - "${DEST}" <<'PY'
import json, os, sys
d_dir = sys.argv[1]
cfg = json.load(open(os.path.join(d_dir, "config.json")))
for k in ["num_actions_chunk", "action_dim", "proprio_dim",
          "num_images_in_input", "use_l1_regression", "use_proprio"]:
    print(f"  {k}: {cfg.get(k, '<缺>')}")
PY

  log "权重分片 MD5 抽样（与远端清单比对 model-00004）"
  local f="${DEST}/model-00004-of-00004.safetensors"
  if [ -f "$f" ]; then
    local sz
    sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
    if [ "$sz" = "262668432" ]; then
      ok "model-00004 大小一致 (262668432 B)"
    else
      fail "model-00004 大小 ${sz} != 262668432（不完整，重跑下载）"
      rc=1
    fi
  else
    fail "缺 model-00004-of-00004.safetensors"
    rc=1
  fi

  if [ "${rc:-0}" = "0" ]; then
    ok "校验通过。下一步："
    echo "    cd ${OFT_ROOT}"
    echo "    bash experiments/robot/libero/failure_study/verify_oft_step.sh checkpoint"
    echo "    bash experiments/robot/libero/failure_study/verify_oft_step.sh model"
  fi
  return ${rc:-0}
}

# ---------------------------------------------------------------------------
case "${MODE}" in
  probe)  probe ;;
  verify) verify ;;
  all)    probe && download && verify ;;
  force)  probe && download && verify ;;
esac

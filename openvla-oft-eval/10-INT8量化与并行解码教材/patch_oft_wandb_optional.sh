#!/usr/bin/env bash
# =============================================================================
# patch_oft_wandb_optional.sh —— 把 OFT 里所有裸 `import wandb` 改为可选导入
#
# 背景：wandb 在 OFT 里只用于可选的日志上报（tracker_type == "wandb" 分支、
#       cfg.use_wandb 开关），但代码在**模块顶层无条件 import**。
#       一旦 wandb 的深层依赖缺失（dockerpycreds / pydantic_core / ...），
#       整个 `import prismatic` 链条就会炸，哪怕根本不启用 wandb。
#
# 处置：改成 try/except ImportError → wandb = None。
#       所有实际调用点都在条件分支内，因此零风险。
#
# 用法：
#   bash patch_oft_wandb_optional.sh           # 应用补丁（幂等）
#   bash patch_oft_wandb_optional.sh --check   # 只检查，不修改
#   OFT_ROOT=/path/to/openvla-oft bash patch_oft_wandb_optional.sh
# =============================================================================
set -uo pipefail

OFT_ROOT="${OFT_ROOT:-/mnt/workspace/openvla-oft}"
CHECK_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --check) CHECK_ONLY=1 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "未知参数: ${arg}" >&2; exit 2 ;;
  esac
done

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m[OK]\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m[!!]\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$*"; }

[ -d "${OFT_ROOT}" ] || { bad "找不到 ${OFT_ROOT}"; exit 1; }
cd "${OFT_ROOT}" || exit 1

log "扫描裸 import wandb"
# 找出所有"行首 import wandb"的 py 文件（排除 .git / __pycache__）
python - <<'PY'
import re
from pathlib import Path

root = Path(".")
hits = []
for p in root.rglob("*.py"):
    if any(part in {".git", "__pycache__"} for part in p.parts):
        continue
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    for i, line in enumerate(text.splitlines(), 1):
        if re.match(r"^import wandb\s*$", line):
            hits.append(f"{p}:{i}")

if hits:
    print(f"  发现 {len(hits)} 处：")
    for h in hits:
        print(f"    {h}")
else:
    print("  未发现裸 import wandb（可能已打补丁）")
PY

if [ "${CHECK_ONLY}" -eq 1 ]; then
  log "检查模式：不修改任何文件"
  exit 0
fi

log "应用可选导入补丁"
python - <<'PY'
import re
from pathlib import Path

OLD = "import wandb\n"
NEW = """try:
    import wandb
except ImportError:  # wandb 仅在启用对应 tracker 时需要；缺它不影响评测
    wandb = None
"""

patched, skipped, manual = [], [], []

for p in sorted(Path(".").rglob("*.py")):
    if any(part in {".git", "__pycache__"} for part in p.parts):
        continue
    try:
        src = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue

    # 只处理"行首且独占一行"的 import wandb
    if not re.search(r"^import wandb\s*$", src, flags=re.M):
        continue

    if "wandb = None" in src:
        skipped.append(str(p))
        continue

    n = len(re.findall(r"^import wandb\s*$", src, flags=re.M))
    if n != 1:
        # 多处出现，或缩进形式，保守跳过并提示人工处理
        manual.append(f"{p} ({n} 处)")
        continue

    p.write_text(re.sub(r"^import wandb\s*$", NEW.rstrip("\n"), src,
                        count=1, flags=re.M), encoding="utf-8")
    patched.append(str(p))

for f in patched:
    print(f"  已打补丁: {f}")
for f in skipped:
    print(f"  已是可选导入: {f}")
for f in manual:
    print(f"  [!!] 需人工处理: {f}")
if not (patched or skipped or manual):
    print("  无需改动")
PY

log "验证：不依赖 wandb 也能导入 prismatic 链路"
if python - <<'PY'
import sys

# 临时屏蔽 wandb，模拟它不可用的情况
class _Blocker:
    def find_module(self, name, path=None):
        if name == "wandb" or name.startswith("wandb."):
            return self

    def load_module(self, name):
        raise ImportError(f"blocked: {name}")


sys.meta_path.insert(0, _Blocker())

try:
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
    assert (NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM) == (8, 7, 8)
    print(f"  [OK] 屏蔽 wandb 后 prismatic 仍可用 "
          f"chunk={NUM_ACTIONS_CHUNK} action_dim={ACTION_DIM} proprio_dim={PROPRIO_DIM}")
except Exception as exc:
    print(f"  [FAIL] {type(exc).__name__}: {exc}")
    raise SystemExit(1)
PY
then
  ok "补丁生效"
else
  bad "补丁未生效，请把上方输出贴出"
  exit 1
fi

echo
echo "================================================================"
echo " wandb 可选化完成。"
echo " 下一步（G0 门禁）："
echo "   cd ${OFT_ROOT}"
echo "   PYTHON_BIN=python bash experiments/robot/libero/failure_study/verify_oft_step.sh eval-import"
echo "================================================================"

#!/usr/bin/env bash
# Install only the packages reported missing by verify_oft_runtime.sh.
# The script always returns success so a missing package never hides later checks.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFT_ROOT="${OFT_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
REPORT_PATH="${REPORT_PATH:-${OFT_ROOT}/dependency_report.json}"
LIBERO_DIR="${LIBERO_DIR:-}"
DRY_RUN="${DRY_RUN:-0}"
CHECKER="${SCRIPT_DIR}/verify_oft_runtime.sh"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[FAIL] Python not found: ${PYTHON_BIN}"
  exit 0
fi

if [[ ! -d "${OFT_ROOT}" || ! -f "${CHECKER}" ]]; then
  echo "[FAIL] OpenVLA-OFT repository or verifier is missing."
  exit 0
fi

# The verifier emits a complete JSON list and deliberately returns zero even
# when dependencies are absent.  Continue if an unexpected interpreter error
# occurs so this wrapper also never stops the caller's job.
"${CHECKER}" --deps-only --python "${PYTHON_BIN}" --report "${REPORT_PATH}" || true

export DRY_RUN
"${PYTHON_BIN}" - "${REPORT_PATH}" "${OFT_ROOT}" "${LIBERO_DIR}" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

report_path, root_text, libero_text = map(Path, sys.argv[1:])
root = root_text.resolve()
dry_run = os.environ.get("DRY_RUN") == "1"

try:
    missing = {item["import"] for item in json.loads(report_path.read_text(encoding="utf-8"))["missing"]}
except Exception as exc:
    print(f"[FAIL] Cannot read dependency report {report_path}: {type(exc).__name__}: {exc}")
    raise SystemExit(0)

libero_candidates = [Path(str(libero_text))] if str(libero_text) not in {"", "."} else []
libero_candidates += [root / name for name in ("LIBERO", "LIBERO-upstream")]
libero_candidates += [root.parent / name for name in ("LIBERO", "LIBERO-upstream")]
libero_candidates += [Path("/mnt/workspace/LIBERO")]
libero_dir = next((path for path in libero_candidates if (path / "setup.py").is_file()), None)

install = {
    "accelerate": ["accelerate"],
    "bddl": ["bddl"],
    "cloudpickle": ["cloudpickle"],
    "diffusers": ["diffusers==0.30.3"],
    "dlimp": ["--no-deps", str(root / "vendor" / "dlimp_openvla-040105d.tar.gz")],
    "draccus": ["draccus==0.8.0"],
    "easydict": ["easydict"],
    "einops": ["einops"],
    "gym": ["gym"],
    "huggingface_hub": ["huggingface-hub"],
    "imageio": ["imageio[ffmpeg]"],
    "imageio_ffmpeg": ["imageio-ffmpeg"],
    "json_numpy": ["json-numpy"],
    "jsonlines": ["jsonlines"],
    "mujoco": ["mujoco"],
    "peft": ["peft==0.11.1"],
    "PIL": ["Pillow"],
    "requests": ["requests"],
    "robosuite": ["robosuite==1.4.1"],
    "safetensors": ["safetensors"],
    "sentencepiece": ["sentencepiece==0.1.99"],
    "tensorflow": ["tensorflow==2.15.0"],
    "tensorflow_datasets": ["tensorflow-datasets==4.9.3"],
    "tensorflow_graphics": ["tensorflow-graphics==2021.12.3"],
    "timm": ["timm==0.9.10"],
    "tokenizers": ["tokenizers==0.19.1"],
    "tqdm": ["tqdm"],
    "transformers": ["git+https://github.com/moojink/transformers-openvla-oft.git"],
    "wandb": ["wandb"],
}
if libero_dir:
    install["libero"] = ["-e", str(libero_dir)]

# PyTorch's three wheels must stay matched.  Only install the trio when any
# member is actually absent; version-only differences remain in the report.
torch_missing = missing & {"torch", "torchvision", "torchaudio"}
if torch_missing:
    index_url = os.environ.get("TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu121")
    install["torch"] = ["--index-url", index_url, "torch==2.2.0", "torchvision==0.17.0", "torchaudio==2.2.0"]

handled = set()
for name in sorted(missing):
    if name in {"torchvision", "torchaudio"} and torch_missing:
        continue
    args = install.get(name)
    if not args:
        if name == "libero":
            print("[MANUAL] LIBERO checkout not found. Set LIBERO_DIR=/path/to/LIBERO and rerun.")
        else:
            print(f"[MANUAL] No install recipe for {name}; see {report_path}.")
        continue
    key = tuple(args)
    if key in handled:
        continue
    handled.add(key)
    command = [sys.executable, "-m", "pip", "install", *args]
    print("[INSTALL]", " ".join(command))
    if dry_run:
        print(f"[DRY RUN] {name}: skipped")
        continue
    try:
        result = subprocess.run(command, check=False)
    except OSError as exc:
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
    else:
        print(f"[{'OK' if result.returncode == 0 else 'FAIL'}] {name}: pip exit {result.returncode}")
PY

echo "[INFO] Rechecking after attempted installs..."
"${CHECKER}" --deps-only --python "${PYTHON_BIN}" --report "${REPORT_PATH}" || true
echo "[INFO] Final report: ${REPORT_PATH}"
exit 0

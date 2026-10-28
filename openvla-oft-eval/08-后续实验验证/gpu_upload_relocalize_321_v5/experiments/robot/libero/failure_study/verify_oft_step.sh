#!/usr/bin/env bash
# Run one OpenVLA-OFT/LIBERO startup stage at a time.  No Conda activation,
# package enumeration, installation, or environment mutation is performed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFT_ROOT="${OFT_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
CKPT_DIR="${CKPT_DIR:-/mnt/workspace/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STEP="${1:-all}"

usage() {
  cat <<'EOF'
Usage: verify_oft_step.sh {python|gpu|eval-import|checkpoint|model|all}

Stages:
  python       Python interpreter and repository location.
  gpu          PyTorch, CUDA, BF16, and Torch--NumPy interoperability.
  eval-import  Import the LIBERO evaluation entry point.
  checkpoint   Check the local checkpoint's required files.
  model        Load the 7B model plus its action head and proprio projector.
  all          Run the five stages above in order, stopping at the first failure.

Examples:
  PYTHON_BIN=python3 bash experiments/robot/libero/failure_study/verify_oft_step.sh python
  PYTHON_BIN=python3 bash experiments/robot/libero/failure_study/verify_oft_step.sh gpu
  PYTHON_BIN=python3 CKPT_DIR=/path/to/checkpoint \
    bash experiments/robot/libero/failure_study/verify_oft_step.sh model
EOF
}

fail() {
  echo "[FAIL] $1" >&2
  exit 1
}

run_python() {
  "${PYTHON_BIN}" - <<'PY'
import sys
assert sys.version_info[:2] == (3, 10), f"OpenVLA-OFT expects Python 3.10, found {sys.version.split()[0]}"
print(f"[OK] Python {sys.version.split()[0]} at {sys.executable}")
PY
}

run_gpu() {
  "${PYTHON_BIN}" - <<'PY'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
assert torch.cuda.is_bf16_supported(), "the selected GPU does not support bfloat16"
torch.zeros(1).numpy()
device = torch.cuda.current_device()
print(f"[OK] torch={torch.__version__}; CUDA={torch.version.cuda}; GPU={torch.cuda.get_device_name(device)}")
PY
}

run_eval_import() {
  "${PYTHON_BIN}" - <<'PY'
from experiments.robot.libero.run_libero_eval import GenerateConfig
print(f"[OK] LIBERO evaluation entry point imported; default open-loop={GenerateConfig.num_open_loop_steps}")
PY
}

run_checkpoint() {
  [[ -d "${CKPT_DIR}" ]] || fail "checkpoint directory not found: ${CKPT_DIR}"
  CKPT_DIR="${CKPT_DIR}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

checkpoint = Path(os.environ["CKPT_DIR"])
for name in ("config.json", "dataset_statistics.json"):
    assert (checkpoint / name).is_file(), f"missing {checkpoint / name}"
stats = json.loads((checkpoint / "dataset_statistics.json").read_text(encoding="utf-8"))
assert "libero_spatial" in stats or "libero_spatial_no_noops" in stats, "LIBERO-Spatial normalization key is absent"
weights = [path for path in checkpoint.rglob("*") if path.is_file() and path.suffix in {".bin", ".pt", ".safetensors"}]
assert weights, "no model weight file (.bin, .pt, or .safetensors) found"
print(f"[OK] checkpoint={checkpoint}; weight_files={len(weights)}")
PY
}

run_model() {
  CKPT_DIR="${CKPT_DIR}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.openvla_utils import get_action_head, get_proprio_projector
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.constants import PROPRIO_DIM

checkpoint = Path(os.environ["CKPT_DIR"])
assert checkpoint.is_dir(), f"checkpoint directory not found: {checkpoint}"
AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor, exist_ok=True)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction, exist_ok=True)
model = AutoModelForVision2Seq.from_pretrained(
    checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).eval().to("cuda:0")
model.vision_backbone.set_num_images_in_input(2)
cfg = SimpleNamespace(
    pretrained_checkpoint=str(checkpoint), use_l1_regression=True, use_diffusion=False,
    num_diffusion_steps_train=50, num_diffusion_steps_inference=50,
)
action_head = get_action_head(cfg, llm_dim=model.llm_dim)
proprio_projector = get_proprio_projector(cfg, llm_dim=model.llm_dim, proprio_dim=PROPRIO_DIM)
print(f"[OK] loaded 7B VLA; llm_dim={model.llm_dim}; action_head={next(action_head.parameters()).device}")
del proprio_projector, action_head, model
torch.cuda.empty_cache()
PY
}

run_step() {
  local name="$1"
  echo "==> ${name}"
  case "${name}" in
    python) run_python ;;
    gpu) run_gpu ;;
    eval-import) run_eval_import ;;
    checkpoint) run_checkpoint ;;
    model) run_model ;;
    *) usage; exit 2 ;;
  esac
}

[[ -d "${OFT_ROOT}" ]] || fail "OpenVLA-OFT repository not found: ${OFT_ROOT}"
command -v "${PYTHON_BIN}" >/dev/null || fail "Python executable not found: ${PYTHON_BIN}"
cd "${OFT_ROOT}"

if [[ "${STEP}" = "all" ]]; then
  for stage in python gpu eval-import checkpoint model; do
    run_step "${stage}"
  done
else
  run_step "${STEP}"
fi

echo "[OK] requested verification completed"

#!/usr/bin/env bash
# Validate the active Python, LIBERO dependencies, and a local OpenVLA-OFT checkpoint.
# No Conda activation is performed: the script uses PYTHON_BIN (or `python`) as-is.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFT_ROOT="${OFT_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
CKPT_DIR="${CKPT_DIR:-/mnt/workspace/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOAD_MODEL=1

usage() {
  cat <<'EOF'
Usage: verify_oft_runtime.sh [--fast] [--checkpoint PATH] [--python PATH]

  --fast             Check dependencies and checkpoint files only; do not load 7B weights.
  --checkpoint PATH  Local OpenVLA-OFT checkpoint directory.
  --python PATH      Python executable already configured for this environment.

Examples:
  bash experiments/robot/libero/failure_study/verify_oft_runtime.sh
  bash experiments/robot/libero/failure_study/verify_oft_runtime.sh --fast
  CKPT_DIR=/mnt/workspace/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial \
    bash experiments/robot/libero/failure_study/verify_oft_runtime.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fast) LOAD_MODEL=0 ;;
    --checkpoint) CKPT_DIR="$2"; shift ;;
    --python) PYTHON_BIN="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ -d "${OFT_ROOT}" ]] || { echo "[FAIL] OpenVLA-OFT repo not found: ${OFT_ROOT}" >&2; exit 1; }
[[ -d "${CKPT_DIR}" ]] || { echo "[FAIL] Checkpoint not found: ${CKPT_DIR}" >&2; exit 1; }
command -v "${PYTHON_BIN}" >/dev/null || { echo "[FAIL] Python not found: ${PYTHON_BIN}" >&2; exit 1; }

cd "${OFT_ROOT}"
export OFT_ROOT CKPT_DIR PYTHON_BIN LOAD_MODEL
echo "[INFO] repo=${OFT_ROOT}"
echo "[INFO] checkpoint=${CKPT_DIR}"
echo "[INFO] python=$(command -v "${PYTHON_BIN}")"

"${PYTHON_BIN}" - <<'PY'
import importlib
import json
import os
import sys
from pathlib import Path

repo = Path(os.environ["OFT_ROOT"])
checkpoint = Path(os.environ["CKPT_DIR"])
load_model = os.environ["LOAD_MODEL"] == "1"

print(f"[OK] Python {sys.version.split()[0]} at {sys.executable}")
for name in ("torch", "transformers", "draccus", "libero", "robosuite", "mujoco"):
    module = importlib.import_module(name)
    print(f"[OK] import {name} {getattr(module, '__version__', '(version unavailable)')}")

import torch
if not torch.cuda.is_available():
    raise SystemExit("[FAIL] torch.cuda.is_available() is False; OpenVLA-OFT LIBERO evaluation needs a CUDA GPU.")
device = torch.cuda.current_device()
print(f"[OK] CUDA {torch.version.cuda}; GPU {device}: {torch.cuda.get_device_name(device)}")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("[FAIL] Current GPU does not support bfloat16 required by this checkpoint loader.")
print("[OK] bfloat16 supported")

config_path = checkpoint / "config.json"
stats_path = checkpoint / "dataset_statistics.json"
if not config_path.is_file():
    raise SystemExit(f"[FAIL] Missing {config_path}")
if not stats_path.is_file():
    raise SystemExit(f"[FAIL] Missing {stats_path}")
config = json.loads(config_path.read_text(encoding="utf-8"))
stats = json.loads(stats_path.read_text(encoding="utf-8"))
weight_files = [p for p in checkpoint.rglob("*") if p.is_file() and p.suffix in {".safetensors", ".bin", ".pt"}]
if not weight_files:
    raise SystemExit("[FAIL] No .safetensors, .bin, or .pt weight file found in checkpoint")
if "libero_spatial" not in stats and "libero_spatial_no_noops" not in stats:
    raise SystemExit("[FAIL] dataset_statistics.json lacks a LIBERO-Spatial normalization key")
print(f"[OK] checkpoint config model_type={config.get('model_type')!r}; {len(weight_files)} weight files found")

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
if (NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM) != (8, 7, 8):
    raise SystemExit(f"[FAIL] Expected LIBERO constants (8, 7, 8), got {(NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM)}")
print("[OK] LIBERO action shape: 8 x 7; proprio dimension: 8")

if not load_model:
    print("[OK] Fast validation complete (model weights were not loaded).")
    raise SystemExit(0)

# Register local classes and load without calling get_vla(), which may synchronize
# repository files into the checkpoint. Validation must not mutate the checkpoint.
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor, exist_ok=True)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction, exist_ok=True)

model = AutoModelForVision2Seq.from_pretrained(
    checkpoint,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).eval().to("cuda:0")
model.vision_backbone.set_num_images_in_input(2)
print(f"[OK] VLA weights loaded; llm_dim={model.llm_dim}; vision inputs=2")

from types import SimpleNamespace
from experiments.robot.openvla_utils import get_action_head, get_proprio_projector

cfg = SimpleNamespace(
    pretrained_checkpoint=str(checkpoint),
    use_l1_regression=True,
    use_diffusion=False,
    num_diffusion_steps_train=50,
    num_diffusion_steps_inference=50,
)
action_head = get_action_head(cfg, llm_dim=model.llm_dim)
proprio_projector = get_proprio_projector(cfg, llm_dim=model.llm_dim, proprio_dim=PROPRIO_DIM)
print(f"[OK] action head and proprio projector loaded on {next(action_head.parameters()).device}")

del proprio_projector, action_head, model
torch.cuda.empty_cache()
print("[OK] Full environment and model validation complete.")
PY

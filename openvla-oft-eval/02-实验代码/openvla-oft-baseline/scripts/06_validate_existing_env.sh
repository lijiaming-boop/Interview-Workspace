#!/usr/bin/env bash
# Read-only validation for an existing ModelScope GPU environment.
set -uo pipefail

OFT_ROOT="${OFT_ROOT:-/mnt/workspace/openvla-oft}"
CKPT_DIR="${CKPT_DIR:-/mnt/workspace/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial}"
FAILURES=0

ok()   { printf '[OK]   %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*"; FAILURES=$((FAILURES + 1)); }

printf '== System ==\n'
uname -a
printf 'python: %s\n' "$(command -v python 2>/dev/null || echo NOT_FOUND)"
python --version 2>&1 || true
printf 'repo: %s\ncheckpoint: %s\n' "${OFT_ROOT}" "${CKPT_DIR}"
df -h /mnt/workspace 2>/dev/null || df -h "${OFT_ROOT}" 2>/dev/null || true
printf 'MUJOCO_GL=%s\nPYOPENGL_PLATFORM=%s\n' "${MUJOCO_GL:-UNSET}" "${PYOPENGL_PLATFORM:-UNSET}"
[ -f "${HOME}/.libero/config.yaml" ] \
  && ok "LIBERO config: ${HOME}/.libero/config.yaml" \
  || warn 'LIBERO config is absent; the first foreground smoke run will ask for a writable data path'

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  fail 'nvidia-smi is unavailable'
fi

if ! command -v python >/dev/null 2>&1; then
  fail 'python is unavailable'
elif [ ! -d "${OFT_ROOT}/.git" ]; then
  fail "OpenVLA-OFT repository not found at ${OFT_ROOT}"
else
  printf '\n== Repository ==\n'
  git -C "${OFT_ROOT}" branch --show-current || true
  git -C "${OFT_ROOT}" log -1 --oneline || true
  for path in \
    experiments/robot/libero/run_libero_eval.py \
    experiments/robot/libero/failure_study/run_simulation_study.sh \
    experiments/robot/libero/failure_study/analyze_action_chunks.py; do
    [ -f "${OFT_ROOT}/${path}" ] && ok "found ${path}" || fail "missing ${path}"
  done

  printf '\n== Python and CUDA ==\n'
  (
    cd "${OFT_ROOT}" || exit 1
    python - <<'PY'
import importlib
import sys

required = ["torch", "numpy", "draccus", "transformers", "tensorflow", "gym", "robosuite", "dlimp"]
failed = False

print("Python:", sys.version.replace("\n", " "))
if sys.version_info[:2] != (3, 10):
    print("FAIL: Python 3.10 is required for the reproducible stack")
    failed = True

versions = {}
for name in required:
    try:
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "unknown")
        print(f"{name}: {versions[name]}")
    except Exception as exc:
        print(f"FAIL: import {name}: {exc}")
        failed = True

if importlib.util.find_spec("libero") is None:
    print("FAIL: LIBERO package is not installed")
    failed = True
else:
    print("libero: installed")

if "numpy" in versions and versions["numpy"].split(".", 1)[0] == "2":
    print("FAIL: NumPy 2.x is incompatible with this LIBERO stack")
    failed = True

expected_prefixes = {
    "torch": "2.2.0",
    "numpy": "1.24.4",
    "transformers": "4.40.",
    "tensorflow": "2.15.",
    "gym": "0.21.",
    "robosuite": "1.4.1",
}
for name, prefix in expected_prefixes.items():
    version = versions.get(name)
    if version is not None and not version.startswith(prefix):
        print(f"FAIL: {name} {version} does not match required {prefix}*")
        failed = True

try:
    import torch
    print("PyTorch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("FAIL: CUDA is unavailable to PyTorch")
        failed = True
    else:
        props = torch.cuda.get_device_properties(0)
        memory_gib = props.total_memory / 1024**3
        print("GPU:", torch.cuda.get_device_name(0))
        print(f"VRAM: {memory_gib:.1f} GiB")
        print("BF16 supported:", torch.cuda.is_bf16_supported())
        if memory_gib < 16:
            print("FAIL: less than 16 GiB VRAM")
            failed = True
        if not torch.cuda.is_bf16_supported():
            print("FAIL: BF16 is unsupported")
            failed = True
except Exception as exc:
    print("FAIL: CUDA inspection:", exc)
    failed = True

try:
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
    print("OFT constants:", NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM)
    if (NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM) != (8, 7, 8):
        print("FAIL: expected OFT LIBERO constants 8, 7, 8")
        failed = True
except Exception as exc:
    print("FAIL: importing OFT constants:", exc)
    failed = True

raise SystemExit(1 if failed else 0)
PY
  ) && ok 'Python imports, CUDA, and OFT constants' || fail 'Python/CUDA validation failed'

  printf '\n== Source checks ==\n'
  (
    cd "${OFT_ROOT}" || exit 1
    python -m py_compile \
      experiments/robot/libero/run_libero_eval.py \
      experiments/robot/libero/failure_study/analyze_action_chunks.py \
      experiments/robot/libero/failure_study/grasp_verification.py \
      experiments/robot/libero/failure_study/test_grasp_verification.py && \
    python experiments/robot/libero/failure_study/test_grasp_verification.py
  ) && ok 'Python source and grasp rule' || fail 'source validation failed'

  shell_failed=0
  while IFS= read -r script; do
    bash -n "${script}" || shell_failed=1
  done < <(find "${OFT_ROOT}/experiments/robot/libero/failure_study" -maxdepth 1 -name '*.sh' -type f | sort)
  [ "${shell_failed}" -eq 0 ] && ok 'failure-study shell syntax' || fail 'shell syntax validation failed'
fi

printf '\n== Checkpoint ==\n'
if [ ! -d "${CKPT_DIR}" ]; then
  fail "checkpoint directory not found: ${CKPT_DIR}"
else
  [ -f "${CKPT_DIR}/config.json" ] && ok 'config.json' || fail 'missing config.json'
  [ -f "${CKPT_DIR}/dataset_statistics.json" ] && ok 'dataset_statistics.json' || fail 'missing dataset_statistics.json'
  find "${CKPT_DIR}" -maxdepth 1 -name '*.safetensors' -print -quit | grep -q . \
    && ok 'model safetensors' || fail 'missing model safetensors'
  find "${CKPT_DIR}" -maxdepth 1 -type f -iname '*action_head*' -print -quit | grep -q . \
    && ok 'action head weights' || fail 'missing action head weights'
  find "${CKPT_DIR}" -maxdepth 1 -type f -iname '*proprio_projector*' -print -quit | grep -q . \
    && ok 'proprio projector weights' || fail 'missing proprio projector weights'
fi

printf '\n== Result ==\n'
if [ "${FAILURES}" -eq 0 ]; then
  ok 'current environment is ready for the failure-study smoke run'
  exit 0
fi

fail "${FAILURES} blocking check(s) found; do not start the long simulation yet"
exit 1

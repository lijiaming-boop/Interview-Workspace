#!/usr/bin/env bash
# =============================================================================
# _common.sh —— 阿里云 PWI 公共配置（可用环境变量覆盖）
# =============================================================================
set -euo pipefail

if [ -z "${PWI_WORKSPACE:-}" ]; then
  for candidate in /mnt/workspace /root/openvla_oft_workspace; do
    if [ -d "${candidate}/openvla-oft" ]; then PWI_WORKSPACE="${candidate}"; break; fi
  done
fi
export PWI_WORKSPACE="${PWI_WORKSPACE:-/mnt/workspace}"
export OFT_ROOT="${OFT_ROOT:-${PWI_WORKSPACE}/openvla-oft}"
export LIBERO_ROOT="${LIBERO_ROOT:-${PWI_WORKSPACE}/LIBERO}"
export CKPT_DIR="${CKPT_DIR:-${PWI_WORKSPACE}/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial}"
export CONDA_ENV="${CONDA_ENV:-oft}" # 仅兼容旧 conda 环境；PWI 优先使用 venv_oft
export GPU_ID="${GPU:-0}"
export RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d-%H%M%S')}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export NO_GCE_CHECK=true

# 所有结果写到同一 RUN_TAG 目录：$OFT_ROOT/experiments/failure_study_runs/<tag>
export RESULT_BASE="${OUTPUT_ROOT:-${OFT_ROOT}/experiments/failure_study_runs}/${RUN_TAG}"

# 必须已经应用 0001-Add-full-LIBERO-failure-study-pipeline.patch
require_full_patch() {
  for f in experiments/robot/libero/failure_study/analyze_action_chunks.py \
           experiments/robot/libero/failure_study/summarize_targeted_eval.py; do
    if [ ! -f "${OFT_ROOT}/${f}" ]; then
      echo "!! 缺少 ${f} —— 先在 ${OFT_ROOT} 应用 0001-Add-full-LIBERO-failure-study-pipeline.patch" >&2
      exit 1
    fi
  done
}

require_checkpoint() {
  [ -d "${CKPT_DIR}" ] || { echo "!! 缺少 checkpoint: ${CKPT_DIR}" >&2; exit 1; }
}

require_env() {
  local activate_file="${OFT_ACTIVATE:-${PWI_WORKSPACE}/activate_oft.sh}"
  if [ -f "${activate_file}" ]; then
    # shellcheck disable=SC1090
    source "${activate_file}"
  elif [ -f "${PWI_WORKSPACE}/venv_oft/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${PWI_WORKSPACE}/venv_oft/bin/activate"
  elif command -v conda >/dev/null 2>&1 && conda env list | grep -qw "${CONDA_ENV}"; then
    # shellcheck disable=SC1090
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
  fi
  export PYTHONPATH="${OFT_ROOT}:${LIBERO_ROOT}:${PYTHONPATH:-}"
  python - <<'PY'
import importlib
import os
import sys

training = os.environ.get("REQUIRE_TRAINING_STACK") == "1"
required = ["torch", "numpy", "draccus", "transformers", "libero", "robosuite", "mujoco", "gym"]
if training:
    required += ["tensorflow", "tensorflow_datasets", "tensorflow_graphics", "dlimp", "h5py", "peft"]
modules = {}
for name in required:
    try:
        modules[name] = importlib.import_module(name)
    except Exception as exc:
        raise SystemExit(f"[FAIL] import {name}: {exc}")

torch, numpy = modules["torch"], modules["numpy"]
if training and sys.version_info[:2] != (3, 10):
    raise SystemExit(f"[FAIL] 训练环境要求 Python 3.10，当前 {sys.version.split()[0]}")
if numpy.__version__ != "1.26.4":
    raise SystemExit(f"[FAIL] PWI 训练栈要求 NumPy 1.26.4，当前 {numpy.__version__}")
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise SystemExit("[FAIL] 当前 PyTorch 无可用 CUDA/BF16")
if training and not (torch.__version__.startswith("2.2.0") or torch.__version__.startswith("2.5.1")):
    raise SystemExit(f"[FAIL] PWI 只接受文档验证过的 torch 2.2.0 或 2.5.1，当前 {torch.__version__}")
if training:
    expected_cuda = "12.1" if torch.__version__.startswith("2.2.0") else "12.4"
    if not str(torch.version.cuda).startswith(expected_cuda):
        raise SystemExit(f"[FAIL] torch {torch.__version__} 应配 CUDA {expected_cuda}，当前 {torch.version.cuda}")
if training and not modules["transformers"].__version__.startswith("4.40."):
    raise SystemExit(f"[FAIL] 需要 OpenVLA-OFT transformers 4.40.x fork，当前 {modules['transformers'].__version__}")
expected = {"robosuite": "1.4.1", "mujoco": "3.3.2", "gym": "0.25.2"}
if training:
    expected.update({"tensorflow": "2.15.", "tensorflow_datasets": "4.9.3", "peft": "0.11.1"})
for name, prefix in expected.items():
    version = getattr(modules[name], "__version__", "")
    if version and not version.startswith(prefix):
        raise SystemExit(f"[FAIL] {name} 应为 {prefix}*，当前 {version}")
if training:
    import google.protobuf
    if google.protobuf.__version__ != "4.25.3":
        raise SystemExit(f"[FAIL] protobuf 应为 4.25.3，当前 {google.protobuf.__version__}")
try:
    import flash_attn
except ModuleNotFoundError:
    pass  # PWI 默认走 SDPA，不要求 flash-attn。
except Exception as exc:
    raise SystemExit(f"[FAIL] flash-attn 与当前 torch/CUDA ABI 不兼容，请卸载后走 SDPA: {exc}")
print(f"[OK] Python {sys.version.split()[0]} | torch {torch.__version__} | CUDA {torch.version.cuda} | numpy {numpy.__version__}")
print(f"[OK] GPU {torch.cuda.get_device_name(0)} | BF16=True | training_stack={training}")
PY
}

study_run_eval() { # label targets open_loop wrist dump_cam dump_actions motion_thr freq [seed] [run_root]
  local label="$1" targets="$2" open_loop="$3" wrist="$4" dump_cam="$5" dump_actions="$6" motion_thr="$7" freq="$8"
  local seed="${9:-7}"
  local run_root="${10:-${RESULT_BASE}/${label}}"
  mkdir -p "${run_root}"
  printf 'label=%s\ntask_suite=%s\nseed=%s\ntargets=%s\nopen_loop=%s\nwrist=%s\nmotion_threshold=%s\nfreq=%s\nckpt=%s\n' \
    "${label}" "libero_spatial" "${seed}" "${targets}" "${open_loop}" "${wrist}" "${motion_thr}" "${freq}" "${CKPT_DIR}" \
    > "${run_root}/run_config.txt"
  (
    cd "${OFT_ROOT}"
    python experiments/robot/libero/run_libero_eval.py \
      --pretrained_checkpoint "${CKPT_DIR}" \
      --task_suite_name libero_spatial --seed "${seed}" \
      --num_open_loop_steps "${open_loop}" \
      --target_episodes "${targets}" \
      --wrist_image_weight "${wrist}" \
      --dump_camera_inputs "${dump_cam}" \
      --camera_dump_dir "${run_root}/cameras" \
      --dump_action_chunks "${dump_actions}" \
      --action_dump_dir "${run_root}/actions" \
      --target_motion_replan_threshold_m "${motion_thr}" \
      --control_freq "${freq}" \
      --run_id_note "${label}"
  ) 2>&1 | tee "${run_root}/console.log"

  if [ "${dump_actions}" = "True" ]; then
    python "${OFT_ROOT}/experiments/robot/libero/failure_study/analyze_action_chunks.py" \
      --root "${run_root}/actions" \
      --output-dir "${run_root}/analysis"
  fi
  echo "[OK] ${label}: ${run_root}"
}

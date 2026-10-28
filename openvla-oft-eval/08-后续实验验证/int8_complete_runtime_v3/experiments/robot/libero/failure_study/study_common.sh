#!/usr/bin/env bash

STUDY_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_REPO_ROOT="$(cd "${STUDY_SCRIPT_DIR}/../../../.." && pwd)"
STUDY_CKPT_DIR="${CKPT_DIR:-/mnt/workspace/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial}"
STUDY_GPU_ID="${GPU:-0}"
STUDY_RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d-%H%M%S')}"
STUDY_OUTPUT_ROOT="${OUTPUT_ROOT:-${STUDY_REPO_ROOT}/experiments/failure_study_runs/${STUDY_RUN_TAG}}"

export CUDA_VISIBLE_DEVICES="${STUDY_GPU_ID}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

study_require_runtime() {
  if [ ! -d "${STUDY_CKPT_DIR}" ]; then
    echo "Missing checkpoint: ${STUDY_CKPT_DIR}" >&2
    return 1
  fi
  if ! python -c 'import draccus, torch, transformers, libero' >/dev/null 2>&1; then
    echo "The active Python is not the OpenVLA-OFT environment." >&2
    echo "Use the Python interpreter configured for this runtime, then rerun this script." >&2
    echo "Current Python: $(command -v python)" >&2
    return 1
  fi
  mkdir -p "${STUDY_OUTPUT_ROOT}"
}

study_run_eval() {
  local label="$1"
  local targets="$2"
  local open_loop_steps="$3"
  local wrist_weight="$4"
  local dump_cameras="$5"
  local dump_actions="$6"
  local motion_threshold="$7"
  local control_freq="$8"
  local verify_grasp="${9:-False}"
  local grasp_close_hold_steps="${GRASP_CLOSE_HOLD_STEPS:-2}"
  local grasp_open_recovery_steps="${GRASP_OPEN_RECOVERY_STEPS:-4}"
  local grasp_max_retries="${GRASP_MAX_RETRIES:-3}"
  local run_root="${STUDY_OUTPUT_ROOT}/${label}"

  mkdir -p "${run_root}"
  printf '%s\n' \
    "label=${label}" \
    "targets=${targets}" \
    "open_loop_steps=${open_loop_steps}" \
    "wrist_image_weight=${wrist_weight}" \
    "target_motion_replan_threshold_m=${motion_threshold}" \
    "control_freq=${control_freq}" \
    "verify_grasp=${verify_grasp}" \
    "grasp_close_hold_steps=${grasp_close_hold_steps}" \
    "grasp_open_recovery_steps=${grasp_open_recovery_steps}" \
    "grasp_max_retries=${grasp_max_retries}" \
    "checkpoint=${STUDY_CKPT_DIR}" \
    > "${run_root}/run_config.txt"

  (
    cd "${STUDY_REPO_ROOT}"
    python experiments/robot/libero/run_libero_eval.py \
      --pretrained_checkpoint "${STUDY_CKPT_DIR}" \
      --task_suite_name libero_spatial \
      --seed 7 \
      --num_open_loop_steps "${open_loop_steps}" \
      --target_episodes "${targets}" \
      --wrist_image_weight "${wrist_weight}" \
      --dump_camera_inputs "${dump_cameras}" \
      --camera_dump_dir "${run_root}/cameras" \
      --dump_action_chunks "${dump_actions}" \
      --action_dump_dir "${run_root}/actions" \
      --target_motion_replan_threshold_m "${motion_threshold}" \
      --verify_grasp "${verify_grasp}" \
      --grasp_close_hold_steps "${grasp_close_hold_steps}" \
      --grasp_open_recovery_steps "${grasp_open_recovery_steps}" \
      --grasp_max_retries "${grasp_max_retries}" \
      --control_freq "${control_freq}" \
      --run_id_note "${label}"
  ) 2>&1 | tee "${run_root}/console.log"

  if [ "${dump_actions}" = "True" ]; then
    python "${STUDY_SCRIPT_DIR}/analyze_action_chunks.py" \
      --root "${run_root}/actions" \
      --output-dir "${run_root}/analysis"
  fi
}

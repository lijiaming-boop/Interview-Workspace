#!/usr/bin/env bash
set -euo pipefail

# Targeted follow-up experiments for the five LIBERO-Spatial baseline failures.
# Initial-state indices are zero-based and were recovered from the baseline log:
#   233 -> task 4, state 32
#   378 -> task 7, state 27
#   321 -> task 6, state 20
#   230 -> task 4, state 29
#   286 -> task 5, state 35

MODE="${1:-all}"
OFT_ROOT="${OFT_ROOT:-/mnt/workspace/openvla-oft}"
CKPT_DIR="${CKPT_DIR:-/mnt/workspace/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial}"
GPU_ID="${GPU:-0}"
DIAG_ROOT="${DIAG_ROOT:-${OFT_ROOT}/experiments/targeted_diagnostics}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

if [ ! -d "${OFT_ROOT}" ]; then
  echo "Missing OpenVLA-OFT directory: ${OFT_ROOT}" >&2
  exit 1
fi
if [ ! -d "${CKPT_DIR}" ]; then
  echo "Missing checkpoint directory: ${CKPT_DIR}" >&2
  exit 1
fi

run_eval() {
  local label="$1"
  local targets="$2"
  local open_loop_steps="$3"
  local wrist_weight="$4"
  local dump_cameras="$5"

  echo "Running ${label}: targets=${targets}, open_loop=${open_loop_steps}, wrist_weight=${wrist_weight}"
  (
    cd "${OFT_ROOT}"
    python experiments/robot/libero/run_libero_eval.py \
      --pretrained_checkpoint "${CKPT_DIR}" \
      --task_suite_name libero_spatial \
      --seed 7 \
      --num_open_loop_steps "${open_loop_steps}" \
      --target_episodes "${targets}" \
      --wrist_image_weight "${wrist_weight}" \
      --dump_camera_inputs "${dump_cameras}" \
      --camera_dump_dir "${DIAG_ROOT}/${label}" \
      --run_id_note "${label}"
  ) 2>&1 | tee "${DIAG_ROOT}/${label}.console.log"
}

mkdir -p "${DIAG_ROOT}"

case "${MODE}" in
  openloop8_control)
    run_eval "openloop8-control" "4:32,6:20,7:27" 8 1.0 False
    ;;
  openloop4)
    run_eval "openloop4-first4-replan" "4:32,6:20,7:27" 4 1.0 False
    ;;
  camera_control)
    run_eval "camera-control-w1" "4:29,5:35" 8 1.0 True
    ;;
  camera_w125)
    run_eval "camera-weight-w1p25" "4:29,5:35" 8 1.25 True
    ;;
  all)
    run_eval "openloop8-control" "4:32,6:20,7:27" 8 1.0 False
    run_eval "openloop4-first4-replan" "4:32,6:20,7:27" 4 1.0 False
    run_eval "camera-control-w1" "4:29,5:35" 8 1.0 True
    run_eval "camera-weight-w1p25" "4:29,5:35" 8 1.25 True
    ;;
  *)
    echo "Usage: $0 {openloop8_control|openloop4|camera_control|camera_w125|all}" >&2
    exit 2
    ;;
esac

echo "Completed. Logs and camera dumps: ${DIAG_ROOT}"

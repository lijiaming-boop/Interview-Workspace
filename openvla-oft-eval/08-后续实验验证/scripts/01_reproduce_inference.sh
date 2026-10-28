#!/usr/bin/env bash
# =============================================================================
# 01_reproduce_inference.sh —— 推理侧对照组复现（对应已归档的 4 组实验）
# 为什么：仿真侧所有结论都要与同一 checkpoint 的推理侧结果对照。
#         重跑一遍确认在应用 full-patch 后成功率/行为与历史记录一致
#         （历史：chunk8 下 233/378 失败、321 成功；chunk4 下 233/378 仍失败；
#          腕部权重 1.0→1.25 对 230/286 无影响）。
# 输出：$RESULT_BASE/<tag>/inference-*/console.log + cameras/
# 用法：bash 01_reproduce_inference.sh [all|chunk8_control|chunk4|camera_control|camera_w125]
# =============================================================================
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-all}"
source "${SELF_DIR}/_common.sh"

run_infer() { # label targets open_loop wrist dump_cam
  local label="$1" targets="$2" open_loop="$3" wrist="$4" dump_cam="$5"
  local run_root="${RESULT_BASE}/inference-${label}"
  mkdir -p "${run_root}"
  printf 'label=%s targets=%s open_loop=%s wrist=%s camera_dump=%s ckpt=%s\n' \
    "${label}" "${targets}" "${open_loop}" "${wrist}" "${dump_cam}" "${CKPT_DIR}" \
    > "${run_root}/run_config.txt"
  (
    cd "${OFT_ROOT}"
    python experiments/robot/libero/run_libero_eval.py \
      --pretrained_checkpoint "${CKPT_DIR}" \
      --task_suite_name libero_spatial --seed 7 \
      --num_open_loop_steps "${open_loop}" \
      --target_episodes "${targets}" \
      --wrist_image_weight "${wrist}" \
      --dump_camera_inputs "${dump_cam}" \
      --camera_dump_dir "${run_root}/cameras" \
      --run_id_note "inference-${label}"
  ) 2>&1 | tee "${run_root}/console.log"
  echo "[OK] ${label}: ${run_root}/console.log"
}

case "${MODE}" in
  chunk8_control) run_infer "chunk8-control" "4:32,6:20,7:27" 8 1.0 False ;;
  chunk4)         run_infer "chunk4-first4"    "4:32,6:20,7:27" 4 1.0 False ;;
  camera_control) run_infer "camera-w1"        "4:29,5:35"       8 1.0 True ;;
  camera_w125)    run_infer "camera-w1p25"     "4:29,5:35"       8 1.25 True ;;
  all)
    run_infer "chunk8-control" "4:32,6:20,7:27" 8 1.0 False
    run_infer "chunk4-first4"  "4:32,6:20,7:27" 4 1.0 False
    run_infer "camera-w1"      "4:29,5:35"       8 1.0 True
    run_infer "camera-w1p25"   "4:29,5:35"       8 1.25 True
    ;;
  *) echo "Usage: $0 {all|chunk8_control|chunk4|camera_control|camera_w125}" >&2; exit 2 ;;
esac

echo "========== 验收 =========="
cat <<'EOF'
对照历史记录（inference_experiments_tidy.csv）：
  1. chunk8:  233(4:32)=Failure, 321(6:20)=Success, 378(7:27)=Failure
  2. chunk4:  233=Failure, 321=Success, 378=Failure   （说明单纯降 chunk 无效）
  3. camera w1 vs w1.25: 230(4:29)=Failure, 286(5:35)=Success 不变
若重跑结果与上述一致，说明环境可复现，可以进入仿真侧实验（02~05）。
检查 camera dump：python -c "from PIL import Image; import glob,os;[print(f, Image.open(f).size) for f in glob.glob('$RESULT_BASE/inference-*/cameras/task_04/state_29/*.png')[:3]]"
EOF
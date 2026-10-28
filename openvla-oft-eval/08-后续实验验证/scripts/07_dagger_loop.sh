#!/usr/bin/env bash
# =============================================================================
# 07_dagger_loop.sh —— 阿里云 PWI · DAgger 针对性增训（top drawer）
#
# 为什么这样做：
#   推理/仿真侧的干预都不改权重——空抓/抓不到这类问题本质是策略行为缺陷，
#   若仿真侧恢复机制增益有限，就回到“在策略失败的状态上补专家修正”：
#   DAgger。先由 06 sweep 定位失败 state，在其上收集专家示教（人工/脚本
#   teleop 的 correct demo），与原始 LIBERO 数据混合构建 RLDS，LoRA 微调
#   官方 checkpoint（关键：显存 ≥25GB，batch_size=1 + grad accum）。
#
# 流程（每个失败 state 一个 demo 目录）：
#   1. 06 输出 dagger_collection_manifest.csv → 在 LIBERO GUI 下用
#      collect_demonstration.py 录制专家轨迹（scripts/collect_demonstration.py）。
#   2. 把 demo 以 RLDS/LIBERO 格式整理进 ${DAGGER_RLDS_ROOT}
#      （参考 LIBERO scripts/ 的 RLDS 转换脚本），dataset_name 见下。
#   3. 先做数据/权重硬检查，再从 150K OFT 动作头正确续训 200 步。
#   4. 用保存的 LoRA/合并权重回到 01~04 复评 4:29 / 4:32 / 6:20 / 7:27。
#
# 用法：
#   DAGGER_RLDS_ROOT=/path/to/rlds VLA_PATH=/path/to/spatial \
#     RUN_ROOT=/mnt/workspace/openvla-oft-training-runs/top-drawer-dagger \
#     DATASET_NAME=libero_spatial_plus_top_drawer_dagger \
#     SOURCE_HDF5=/path/to/expert.hdf5 EXTRA_STEPS=200 bash 07_dagger_loop.sh
# 验收：finetune 后重新评测失败状态合集，成功率 > 微调前；同时 4 个对照
#       任务成功率不得明显回退（无灾难性遗忘）。
# =============================================================================
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SELF_DIR}/_common.sh"

DAGGER_RLDS_ROOT="${DAGGER_RLDS_ROOT:-${PWI_WORKSPACE}/datasets/rlds}"
SOURCE_HDF5="${SOURCE_HDF5:?请设置 SOURCE_HDF5（前检要核验示教质量）}"
VLA_PATH="${VLA_PATH:-${CKPT_DIR}}"
RUN_ROOT="${RUN_ROOT:-${PWI_WORKSPACE}/openvla-oft-training-runs/top-drawer-dagger}"
DATASET_NAME="${DATASET_NAME:-libero_spatial_plus_top_drawer_dagger}"
BASE_DATASET="${BASE_DATASET:-libero_spatial_no_noops}"
CORRECTION_DATASET="${CORRECTION_DATASET:-libero_spatial_top_drawer_dagger}"
BASE_STEP="${BASE_STEP:-150000}"
EXTRA_STEPS="${EXTRA_STEPS:-200}"
MAX_STEPS="$((BASE_STEP + EXTRA_STEPS))"
SAVE_FREQ="${SAVE_FREQ:-50}"
MIN_DEMOS="${MIN_DEMOS:-50}"
MIN_VRAM_MIB="${MIN_VRAM_MIB:-22000}"
RUN_ID="${RUN_ID:-top-drawer-dagger-safe}"

export REQUIRE_TRAINING_STACK=1
require_checkpoint; require_env
[ -d "${DAGGER_RLDS_ROOT}" ] || { echo "!! 缺少 RLDS 根目录: ${DAGGER_RLDS_ROOT}" >&2; exit 1; }

python "${SELF_DIR}/07_retrain_preflight.py" \
  --checkpoint "${VLA_PATH}" --data-root "${DAGGER_RLDS_ROOT}" --oft-root "${OFT_ROOT}" \
  --base-step "${BASE_STEP}" --base-dataset "${BASE_DATASET}" \
  --correction-dataset "${CORRECTION_DATASET}" --mixture "${DATASET_NAME}" \
  --source-hdf5 "${SOURCE_HDF5}" --min-demos "${MIN_DEMOS}"

VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if [ "${VRAM_MIB}" -lt "${MIN_VRAM_MIB}" ] && [ "${ALLOW_LOW_VRAM:-0}" != "1" ]; then
  echo "!! GPU 显存 ${VRAM_MIB}MiB < ${MIN_VRAM_MIB}MiB（官方 batch=1 约需 25GB）" >&2
  echo "   请使用 ≥32GB 显卡，或 ALLOW_LOW_VRAM=1 自担风险继续。" >&2
  exit 1
fi

expect_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "!! 缺少命令 $1" >&2; exit 1; } }
expect_cmd torchrun

export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p "${RUN_ROOT}"

cd "${OFT_ROOT}"
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "${VLA_PATH}" \
  --data_root_dir "${DAGGER_RLDS_ROOT}" \
  --dataset_name "${DATASET_NAME}" \
  --run_root_dir "${RUN_ROOT}" \
  --use_l1_regression True --use_diffusion False --use_film False \
  --num_images_in_input 2 --use_proprio True \
  --batch_size 1 --grad_accumulation_steps "${GRAD_ACCUMULATION:-4}" \
  --learning_rate "${LEARNING_RATE:-5e-6}" \
  --num_steps_before_decay "$((EXTRA_STEPS + 1))" \
  --max_steps "${MAX_STEPS}" --save_freq "${SAVE_FREQ}" \
  --shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE:-5000}" --save_latest_checkpoint_only False \
  --merge_lora_during_training False --image_aug True \
  --resume True --resume_step "${BASE_STEP}" \
  --lora_rank 32 --wandb_entity local \
  --wandb_project openvla-oft-top-drawer-dagger --run_id_override "${RUN_ID}"

echo "========== 验收（微调完成后执行） =========="
cat <<EOF
1) 候选 checkpoint：${RUN_ROOT}/${RUN_ID}--${MAX_STEPS}_chkpt
2) 在评测 GPU 上合并（动作头/proprio 文件已在同目录）：
   python vla-scripts/merge_lora_weights_and_save.py \\
     --base_checkpoint ${VLA_PATH} \\
     --lora_finetuned_checkpoint_dir ${RUN_ROOT}/${RUN_ID}--${MAX_STEPS}_chkpt
3) 先复评目标失败 state，再跑 top drawer 50 states，最后跑 spatial 500 episodes。
4) 仅当目标集提升、top drawer 不低于 98%、全套不低于 99%，才保留候选；否则回滚基座。
EOF

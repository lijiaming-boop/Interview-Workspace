#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

DAGGER_RLDS_ROOT="${DAGGER_RLDS_ROOT:?Set DAGGER_RLDS_ROOT to the directory containing base and correction RLDS datasets}"
VLA_PATH="${VLA_PATH:-/mnt/workspace/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial}"
RUN_ROOT="${RUN_ROOT:-/mnt/workspace/openvla-oft-training-runs/top-drawer-dagger}"
DATASET_NAME="${DATASET_NAME:-libero_spatial_plus_top_drawer_dagger}"
MAX_STEPS="${MAX_STEPS:-5000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-8}"
MIN_VRAM_MIB="${MIN_VRAM_MIB:-25000}"

if ! python -c 'import torch, tensorflow, tensorflow_datasets, draccus' >/dev/null 2>&1; then
  echo "The active Python does not contain the OpenVLA-OFT training dependencies." >&2
  exit 1
fi
if [ ! -d "${DAGGER_RLDS_ROOT}" ]; then
  echo "Missing RLDS root: ${DAGGER_RLDS_ROOT}" >&2
  exit 1
fi
if [ ! -d "${VLA_PATH}" ]; then
  echo "Missing starting checkpoint: ${VLA_PATH}" >&2
  exit 1
fi

VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if [ "${VRAM_MIB}" -lt "${MIN_VRAM_MIB}" ] && [ "${ALLOW_LOW_VRAM:-0}" != "1" ]; then
  echo "GPU memory is ${VRAM_MIB} MiB; the official batch-size-1 estimate is about 25 GB." >&2
  echo "Use a >=32 GB GPU, or set ALLOW_LOW_VRAM=1 to attempt the run at your own risk." >&2
  exit 1
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${RUN_ROOT}"

cd "${REPO_ROOT}"
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "${VLA_PATH}" \
  --data_root_dir "${DAGGER_RLDS_ROOT}" \
  --dataset_name "${DATASET_NAME}" \
  --run_root_dir "${RUN_ROOT}" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 1 \
  --grad_accumulation_steps "${GRAD_ACCUMULATION}" \
  --learning_rate "${LEARNING_RATE}" \
  --num_steps_before_decay 4000 \
  --max_steps "${MAX_STEPS}" \
  --save_freq "${SAVE_FREQ}" \
  --shuffle_buffer_size 10000 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity local \
  --wandb_project openvla-oft-top-drawer-dagger \
  --run_id_note top-drawer-dagger

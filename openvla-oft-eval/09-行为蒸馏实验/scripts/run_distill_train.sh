#!/usr/bin/env bash
set -euo pipefail

cd /mnt/workspace/openvla-oft
source /mnt/workspace/venv_oft/bin/activate
export PYTHONPATH="/mnt/workspace/openvla-oft:/mnt/workspace/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python vla-scripts/finetune_teacher_distill.py \
  --base_checkpoint /mnt/workspace/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial \
  --data_root_dir /mnt/workspace/datasets/rlds_head_only_clean_20260906 \
  --dataset_name libero_spatial_top_drawer_expert \
  --output_dir /mnt/workspace/runs/topdrawer_teacher_distill_v1 \
  --batch_size 1 \
  --grad_accumulation_steps 4 \
  --learning_rate 2e-6 \
  --max_steps 100 \
  --save_steps 25,50,100 \
  --expert_weight 1.0 \
  --kd_weight 2.0 \
  --correction_kd_weight 0.25 \
  --correction_threshold 0.12 \
  --parameter_anchor_weight 0.1 \
  --image_aug False \
  --seed 7 \
  2>&1 | tee /mnt/workspace/topdrawer_teacher_distill_v1.log

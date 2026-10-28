#!/usr/bin/env bash
set -euo pipefail

cd /mnt/workspace/openvla-oft
source /mnt/workspace/venv_oft/bin/activate
export PYTHONPATH="/mnt/workspace/openvla-oft:/mnt/workspace/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SOURCE_H5=/mnt/workspace/libero_datasets_hf/libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5
CORRECTION_H5=/mnt/workspace/correction_data/topdrawer_failure_demos29_32_steps45_90.hdf5
DATA_ROOT=/mnt/workspace/datasets/rlds_topdrawer_failure_correction_2932

python tools/make_failure_correction_hdf5.py \
  --input "$SOURCE_H5" \
  --output "$CORRECTION_H5" \
  --demos 29 32 \
  --start 45 \
  --end 91

python -m tools.convert_libero_hdf5_to_rlds \
  --hdf5 "$CORRECTION_H5" \
  --output_dir "$DATA_ROOT"

python vla-scripts/finetune_teacher_distill.py \
  --base_checkpoint /mnt/workspace/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial \
  --data_root_dir "$DATA_ROOT" \
  --dataset_name libero_spatial_top_drawer \
  --output_dir /mnt/workspace/runs/topdrawer_failure_correction_v1 \
  --batch_size 1 \
  --grad_accumulation_steps 4 \
  --learning_rate 3e-6 \
  --max_steps 75 \
  --save_steps 25,50,75 \
  --expert_weight 2.0 \
  --kd_weight 0.5 \
  --correction_kd_weight 0.0 \
  --correction_threshold 0.08 \
  --parameter_anchor_weight 0.02 \
  --image_aug False \
  --seed 7 \
  2>&1 | tee /mnt/workspace/topdrawer_failure_correction_v1.log

#!/bin/bash

EXP_NAME="openpi_05_$(date +%Y%m%d_%H%M%S)_boundaries_but_less"
echo "Experiment name: $EXP_NAME"

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi05_single_task/openpi_05_20251109_221546/37000/ \
    outputs/checkpoints/pi05_single_task/openpi_05_20251109_221546/37000/

CUDA_VISIBLE_DEVICES=4,5 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMNIGIBSON_NO_SIGNALS=1 uv run scripts/train_val.py pi05_single_task_focus_on_boundaries_but_less \
    --exp_name="$EXP_NAME" \
    --overwrite \
    --batch_size=64 \
    --weight_loader.params_path=outputs/checkpoints/pi05_single_task/openpi_05_20251109_221546/37000/params \
    --num_train_steps=50000 \
    --val_log_interval=2000

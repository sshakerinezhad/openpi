#!/bin/bash

    # --overwrite \
    # --exp_name="openpi_$(date +%Y%m%d_%H%M%S)" \
    
    # 2>&1 | tee "openpi_20250919_180225_resume/$(date +%Y%m%d_%H%M%S).log"

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_val.py pi05_b1k \
#     --exp_name="openpi_05_$(date +%Y%m%d_%H%M%S)" \
#     --batch_size=16 \
#     --num_train_steps=100000

# EXP_NAME="openpi_05_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
# EXP_NAME="openpi_05_20251023_221110"
EXP_NAME="openpi_05_20251024_234149"
echo "Experiment name: $EXP_NAME"

START_STEP=$(ls outputs/checkpoints/pi05_b1k/$EXP_NAME/ 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
START_STEP=${START_STEP:-0}
echo "Starting step: $START_STEP"
END_STEP=12000
STEP_SIZE=3000

for (( STEP=$START_STEP; STEP<$END_STEP; STEP+=$STEP_SIZE )); do
    NEXT_STEP=$((STEP + STEP_SIZE))
    if (( NEXT_STEP > END_STEP )); then
        NEXT_STEP=$END_STEP
    fi
    LOG_FILE="logs/${EXP_NAME}_step${STEP}_to_${NEXT_STEP}.log"
    echo "Training from step $STEP to $NEXT_STEP"
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.93 uv run scripts/train_val.py pi05_b1k \
        --exp_name="$EXP_NAME" \
        --resume \
        --batch_size=256 \
        --weight_loader.params_path=/workspace/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20251023_221110/28000/params \
        --num_train_steps=$NEXT_STEP \
        2>&1 | tee $LOG_FILE
done

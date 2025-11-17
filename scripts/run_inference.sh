#!/bin/bash

# Ensure logs directory exists
mkdir -p logs

# Get current timestamp in YYYYMMDD_HHMMSS format
timestamp=$(date +"%Y%m%d_%H%M%S")

# Compose log file path
log_file="logs/${timestamp}.log"

# deactivate
# source .venv/bin/activate
kill $(lsof -ti:8080)

# Launch inference server, logging both stdout and stderr to log_file and screen
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/serve_b1k.py \
    --port 8080 \
    policy:checkpoint \
    --policy.config=pi05_b1k_inference_final \
    --policy.dir=/workspace/openpi/outputs/checkpoints/openpi_05_20251115_045832/36000/ \
    2>&1 | tee "$log_file"

#!/bin/bash

deactivate
source .venv/bin/activate
kill $(lsof -ti:8080)
CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/serve_b1k.py \
    --port 8080 \
    policy:checkpoint \
    --policy.config=pi05_b1k_inference_final \
    --policy.dir=/workspace/openpi/outputs/checkpoints/pi05_b1k_oversample_mbts/openpi_05_20251115_045832/36000/

#!/bin/bash

deactivate
source .venv/bin/activate
kill $(lsof -ti:8000)
XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/serve_b1k.py \
    --port 8000 \
    --task_name=turning_on_radio \
    policy:checkpoint \
    --policy.config=pi05_b1k \
    --policy.dir=/workspace/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20250929_205856/49999/
    # --policy.config=pi0_b1k_base \
    # --policy.dir=gs://openpi-assets/checkpoints/pi0_base/

#!/bin/bash

deactivate
source .venv/bin/activate
kill $(lsof -ti:8000)
XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/serve_b1k.py \
    --port 8000 \
    policy:checkpoint \
    --policy.config=pi0_b1k \
    --policy.dir=outputs/checkpoints/pi0_b1k/openpi_20250919_180225/99999/
    # --policy.config=pi0_b1k_base \
    # --policy.dir=gs://openpi-assets/checkpoints/pi0_base/

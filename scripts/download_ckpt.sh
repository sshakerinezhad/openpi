#!/bin/bash

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi05_b1k/openpi_05_20250929_205856/8000/ \
    outputs/checkpoints/pi05_b1k/openpi_05_20250929_205856/8000/

#!/bin/bash

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi05_b1k/openpi_05_20251031_192101/5000/ \
    /root/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20251031_192101/5000/

#!/bin/bash

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi05_b1k_oversample_mbts/openpi_05_20251115_045832/36000/ \
    /workspace/openpi/outputs/checkpoints/openpi_05_20251115_045832/36000/

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi05_b1k_oversample_psor/psor_openpi_05_20251116_062730/27000/ \
    /workspace/openpi/outputs/checkpoints/psor_openpi_05_20251116_062730/27000/

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi05_b1k_oversample_biw/openpi_05_20251115_071839/15000/ \
    /workspace/openpi/outputs/checkpoints/openpi_05_20251115_071839/15000/

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi05_b1k_oversample_chop_an_onion/chop_an_onion_openpi_05_20251116_220711/9000/ \
    /workspace/openpi/outputs/checkpoints/chop_an_onion_openpi_05_20251116_220711/9000/

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/placeholder/cw_openpi_05_20251116_072941/15000/ \
    /workspace/openpi/outputs/checkpoints/cw_openpi_05_20251116_072941/15000/

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/placeholder/hee_openpi_05_20251116_064228/18000/ \
    /workspace/openpi/outputs/checkpoints/hee_openpi_05_20251116_064228/18000/

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/placeholder/ltc_openpi_05_20251116_073405/15000/ \
    /workspace/openpi/outputs/checkpoints/ltc_openpi_05_20251116_073405/15000/

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/placeholder/openpi_05_20251115_072623/21000/ \
    /workspace/openpi/outputs/checkpoints/openpi_05_20251115_072623/21000/

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/placeholder/sfb_openpi_05_20251116_065743/24000/ \
    /workspace/openpi/outputs/checkpoints/sfb_openpi_05_20251116_065743/24000/

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/placeholder/sft_openpi_05_20251116_070631/21000/ \
    /workspace/openpi/outputs/checkpoints/sft_openpi_05_20251116_070631/21000/

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/placeholder/cupaf_openpi_05_20251116_073015/18000/ \
    /workspace/openpi/outputs/checkpoints/cupaf_openpi_05_20251116_073015/18000/

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/placeholder/openpi_05_20251115_050323/9000/ \
    /workspace/openpi/outputs/checkpoints/openpi_05_20251115_050323/9000/

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/placeholder/rkf_openpi_05_20251116_220634/3000/ \
    /workspace/openpi/outputs/checkpoints/rkf_openpi_05_20251116_220634/3000/

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/placeholder/pahd_openpi_05_20251116_073515/3000/ \
    /workspace/openpi/outputs/checkpoints/pahd_openpi_05_20251116_073515/3000/

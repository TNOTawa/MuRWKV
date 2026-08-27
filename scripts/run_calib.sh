#!/bin/bash
# R1 calibration: 200 steps on the 120-track subset (throughput + val cost).
# GPU sampling runs alongside (scripts/sample_gpu.sh).
set -e
cd /root/MuRWKV
export PYTHONPATH=src
DATA=/root/autodl-tmp/data/slakh2100_16k_from_flac
python3 -m murwkv.training.train \
  --exp results/slakh_r1_calib \
  --data-root "$DATA" --splits \
  --tracks $(cat /tmp/slakh_r1_train_tracks.txt) \
  --steps 200 --max-tokens-per-chunk 4096 --units 4 \
  --seed 42 --log-every 10 --save-every 500 \
  --val-every 100 --val-limit 16 \
  > /tmp/calib.log 2>&1
echo "CALIB EXIT=$?" >> /tmp/calib.log
grep -v "cpp_extension\|TORCH_CUDA_ARCH\|CUDA runtime" /tmp/calib.log | tail -40
#!/bin/bash
# R2 official run launcher (fixed protocol; keep in sync with REPORT_R2.md §3).
# usage: bash scripts/run_r2_train.sh
set -e
cd /root/MuRWKV
export PYTHONPATH=src
# variable window shapes (T=11k..32k) fragment the default CUDA caching
# allocator (measured: step time degrades 0.8s -> 14.5s within 500 steps);
# expandable segments keep VA mapping growth O(1) and the step time flat.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec python -m murwkv.training.train \
  --exp results/slakh_r2_carry \
  --splits \
  --data-root /root/autodl-tmp/data/slakh2100_16k_from_flac \
  --units 16 --val-units 4 \
  --val-every 500 --val-limit 16 \
  --carry-seg 2048 \
  --noise-p 0.15 --noise-anneal 500 \
  --steps 5000 --warmup-steps 200 \
  --log-every 25 --save-every 1000 \
  --max-tokens-per-chunk 4096 --seed 42

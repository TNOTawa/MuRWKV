#!/bin/bash
# Paired held-out eval on the sealed 60-track test split, 3 parallel workers
# (model is ~1 GB VRAM; the 32 GB card handles 3 without contention).
# Usage: bash scripts/run_eval_test.sh <exp_dir> <ckpt>
set -e
cd /root/MuRWKV
export PYTHONPATH=src
EXP="$1"; CKPT="$2"
DATA=/root/autodl-tmp/data/slakh2100_16k_from_flac
python3 - <<'EOF'
import json
m = json.load(open('results/splits/slakh2100_subset_r1.json'))
te = m['test']
import os
shards = [te[i::3] for i in range(3)]
for i, s in enumerate(shards):
    open(f'/tmp/eval_shard_{i}.txt', 'w').write('\n'.join(s))
    print(f'shard{i}: {len(s)} tracks')
EOF
PIDS=()
for i in 0 1 2; do
  python3 -m murwkv.eval.eval_heldout \
    --exp "${EXP}_eval$i" --ckpt "$CKPT" \
    --data-root "$DATA" --splits \
    --tracks $(cat /tmp/eval_shard_$i.txt) \
    --split test --mode both --max-tokens 4096 --compile \
    > /tmp/eval_shard_$i.log 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
echo "ALL EVAL WORKERS DONE"
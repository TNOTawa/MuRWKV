#!/bin/bash
# Paired held-out eval on the sealed 60-track test split — R2 budget round.
# Same protocol as scripts/run_eval_test.sh (R1): --compile, --max-tokens 4096,
# both arms (continuous/reset), data root slakh2100_16k_from_flac, round-robin
# shards over the canonical manifest. Worker count raised 3 -> 5 for the hard
# GPU budget; worker count affects wall time only, never outputs (per-track
# rows are worker-independent; decode is deterministic per protocol).
# Usage: bash scripts/run_r2_eval_test.sh <exp_dir> <ckpt> [n_workers]
set -e
cd /root/MuRWKV
export PYTHONPATH=src
EXP="$1"; CKPT="$2"; N="${3:-5}"
DATA=/root/autodl-tmp/data/slakh2100_16k_from_flac
python3 - "$N" <<'EOF'
import json, sys
n = int(sys.argv[1])
m = json.load(open('results/splits/slakh2100_subset_r1.json'))
te = m['test']
assert len(te) == 60
for i in range(n):
    s = te[i::n]
    open(f'/tmp/r2_eval_shard_{i}.txt', 'w').write('\n'.join(s))
    print(f'shard{i}: {len(s)} tracks')
EOF
PIDS=()
for i in $(seq 0 $((N-1))); do
  python3 -u -m murwkv.eval.eval_heldout \
    --exp "${EXP}_eval$i" --ckpt "$CKPT" \
    --data-root "$DATA" --splits \
    --tracks $(cat /tmp/r2_eval_shard_$i.txt) \
    --split test --mode both --max-tokens 4096 --compile \
    > /tmp/r2_eval_shard_$i.log 2>&1 &
  PIDS+=($!)
done
echo "worker pids: ${PIDS[*]}" | tee /tmp/r2_eval_pids.txt
for p in "${PIDS[@]}"; do wait "$p"; done
echo "ALL EVAL WORKERS DONE"

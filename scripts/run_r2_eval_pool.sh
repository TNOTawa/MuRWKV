#!/bin/bash
# Hardened sharded eval pool driver (R2 budget round).
# - fine shards (few tracks each) so a crashed worker loses little;
# - shard completion = <exp>_eval<i>/eval/test/{continuous,reset}.json exist
#   AND both cover exactly the shard's tracks;
# - re-check + retry loop (up to MAX_PASSES) makes the pool self-healing;
# - up to NWorkers shards run concurrently.
# Usage: bash scripts/run_r2_eval_pool.sh <exp> <ckpt> <tracks_per_shard> <n_workers>
set -u
cd /root/MuRWKV
export PYTHONPATH=src
EXP="$1"; CKPT="$2"; TPS="${3:-2}"; NW="${4:-5}"
DATA=/root/autodl-tmp/data/slakh2100_16k_from_flac
MAX_PASSES=4
log(){ echo "[$(date +%H:%M:%S)] $*"; }

mapfile -t TE < <(python3 -c "import json; print('\n'.join(json.load(open('results/splits/slakh2100_subset_r1.json'))['test']))")
N=${#TE[@]}
NS=$(( (N + TPS - 1) / TPS ))
log "pool: $N tracks -> $NS shards of <=$TPS tracks, $NW workers, exp=$EXP ckpt=$CKPT"

shard_file(){ local i=$1; local f=/tmp/r2pool_shard_$i.txt; : > "$f"; for ((j=i; j<N; j+=NS)); do echo "${TE[j]}" >> "$f"; done; }
shard_done(){
  local i=$1 f
  for mode in continuous reset; do
    f="${EXP}_eval${i}/eval/test/${mode}.json"
    [ -f "$f" ] || return 1
    python3 - "$f" "$i" <<'EOF' || return 1
import json, sys
rows = {r["track"] for r in json.load(open(sys.argv[1]))}
want = set(open(f"/tmp/r2pool_shard_{sys.argv[2]}.txt").read().split())
sys.exit(0 if rows == want else 1)
EOF
  done
  return 0
}

run_shard(){
  local i=$1
  shard_file "$i"
  log "shard$i START tracks: $(tr '\n' ' ' < /tmp/r2pool_shard_$i.txt)"
  python3 -u -m murwkv.eval.eval_heldout \
    --exp "${EXP}_eval${i}" --ckpt "$CKPT" \
    --data-root "$DATA" --splits \
    --tracks $(cat /tmp/r2pool_shard_$i.txt) \
    --split test --mode both --max-tokens 4096 --compile \
    >> "/tmp/r2pool_shard_${i}.log" 2>&1
  local rc=$?
  if shard_done "$i"; then log "shard$i DONE rc=$rc"; else log "shard$i INCOMPLETE rc=$rc"; fi
}

for ((i=0; i<NS; i++)); do shard_file "$i"; done

for pass in $(seq 1 $MAX_PASSES); do
  PENDING=()
  for ((i=0; i<NS; i++)); do shard_done "$i" || PENDING+=("$i"); done
  log "pass $pass: ${#PENDING[@]} pending shards: ${PENDING[*]:-none}"
  [ ${#PENDING[@]} -eq 0 ] && break
  for s in "${PENDING[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$NW" ]; do wait -n; done
    run_shard "$s" &
  done
  wait
done

DONE=0; for ((i=0; i<NS; i++)); do shard_done "$i" && DONE=$((DONE+1)); done
log "FINAL: $DONE/$NS shards complete"
[ "$DONE" -eq "$NS" ] && { log "ALL EVAL SHARDS DONE"; exit 0; }
exit 1

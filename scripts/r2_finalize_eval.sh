#!/bin/bash
# Robust R2 eval finalization (generation-agnostic).
# Waits until (a) no eval_heldout worker remains and (b) the union of all
# results/slakh_r2_carry_eval*/eval/test rows covers the sealed 60-track
# manifest, then merges the union (asserting duplicate rows are identical),
# and produces the paired report + R1-vs-R2 cross-round comparison.
set -e
cd /root/MuRWKV
export PYTHONPATH=src

covered() {
python3 - <<'EOF'
import json, glob
man = set(json.load(open('results/splits/slakh2100_subset_r1.json'))['test'])
done = set()
for p in glob.glob('results/slakh_r2_carry_eval*/eval/test/continuous.json'):
    done |= {r['track'] for r in json.load(open(p))}
missing = sorted(man - done)
print(f"{len(done & man)}/60 covered; missing: {missing if missing else 'NONE'}")
exit(0 if not missing else 1)
EOF
}

while true; do
  if ! pgrep -f "murwkv.eval.eval_heldout" >/dev/null; then
    if covered; then break; fi
    echo "=== $(date +%H:%M:%S) no workers but pool incomplete — waiting for orchestrator ==="
  fi
  sleep 30
done
echo "=== pool complete $(date +%H:%M:%S) ==="

python3 - <<'EOF'
import json, glob, os
man = set(json.load(open('results/splits/slakh2100_subset_r1.json'))['test'])
rows = {}
for p in sorted(glob.glob('results/slakh_r2_carry_eval*/eval/test/continuous.json')):
    for r in json.load(open(p)):
        if r['track'] in rows:
            assert rows[r['track']] == r, f"non-deterministic duplicate row for {r['track']} in {p}"
        rows[r['track']] = r
assert set(rows) == man, f"final union != manifest: {set(rows) ^ man}"
os.makedirs('results/slakh_r2_carry/eval/test', exist_ok=True)
json.dump([rows[t] for t in sorted(rows)], open('results/slakh_r2_carry/eval/test/continuous.json','w'), indent=2)
print(f"union continuous: {len(rows)} tracks, duplicates consistent")
EOF
python3 - <<'EOF'
import json, glob, os
man = set(json.load(open('results/splits/slakh2100_subset_r1.json'))['test'])
rows = {}
for p in sorted(glob.glob('results/slakh_r2_carry_eval*/eval/test/reset.json')):
    for r in json.load(open(p)):
        if r['track'] in rows:
            assert rows[r['track']] == r, f"non-deterministic duplicate row for {r['track']} in {p}"
        rows[r['track']] = r
assert set(rows) == man
json.dump([rows[t] for t in sorted(rows)], open('results/slakh_r2_carry/eval/test/reset.json','w'), indent=2)
print(f"union reset: {len(rows)} tracks, duplicates consistent")
EOF

python scripts/report_paired_eval.py --exp results/slakh_r2_carry | tee results/slakh_r2_carry/eval/paired_report_r2.txt
echo "=== paired report done ==="
python scripts/compare_r1_r2.py --exp-a results/slakh_r1 --exp-b results/slakh_r2_carry \
  --out results/slakh_r2_carry/eval/r1_vs_r2.json | tee results/slakh_r2_carry/eval/r1_vs_r2_report.txt
echo "=== cross-round comparison done ==="

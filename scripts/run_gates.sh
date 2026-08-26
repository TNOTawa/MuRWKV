#!/usr/bin/env bash
# Gate pipeline runner (BabySlakh path). Usage: bash scripts/run_gates.sh
set -e
ROOT=/root/MuRWKV
DATA=/root/autodl-tmp/data/babyslakh/babyslakh_16k
export PYTHONPATH=$ROOT/src

echo "== GATE 1 =="
python $ROOT/tests/test_gate1_data.py $DATA

echo "== GATE 3: 1-song overfit =="
python -m murwkv.training.train --exp $ROOT/results/gate3_overfit --data-root $DATA \
    --tracks Track00001 --units 4 --steps 4000 --lr-init 6e-4 --warmup-steps 100 --seed 42 \
    --log-every 20 --save-every 1000

echo "== GATE 3 eval on the overfit track =="
python -m murwkv.eval.eval_heldout --exp $ROOT/results/gate3_overfit \
    --ckpt $ROOT/results/gate3_overfit/final.pt --data-root $DATA --mode both \
    --tracks Track00001

echo "== GATE 4: 10-song overfit =="
python -m murwkv.training.train --exp $ROOT/results/gate4_overfit --data-root $DATA \
    --tracks Track00001 Track00002 Track00003 Track00004 Track00005 Track00006 Track00007 Track00008 Track00009 Track00010 \
    --units 4 --steps 6000 --lr-init 6e-4 --warmup-steps 200 --seed 42 \
    --log-every 20 --save-every 1000

echo "== GATE 4 eval on held-out sanity tracks =="
python -m murwkv.eval.eval_heldout --exp $ROOT/results/gate4_overfit \
    --ckpt $ROOT/results/gate4_overfit/final.pt --data-root $DATA --split valid --mode both

echo "GATES DONE"
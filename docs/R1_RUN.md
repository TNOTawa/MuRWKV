# R1 runbook — Slakh2100 subset, first generalization round (2026-08-27 late)

All commands run with `PYTHONPATH=src` from the repo root, RTX 5090, torch
2.8.0+cu128, BF16, official RWKV-7 clampw kernel. Protocol constraints
(10h-run supplement): no scheduled sampling, no arch changes, no 2D frontend;
test split sealed; checkpoint selection by validation loss only.

## 0. Frozen artifacts (committed before training)

| Artifact | Content |
|---|---|
| `results/splits/slakh2100_subset_r1.json` | train 120 / val 20 / test 60, seed 42, cap 4096; Gate-1 passed (0 truncation, 9628 chunks, 6.56M tokens) |
| `results/splits/slakh2100_subset_r1_stats.json` | per-track + totals |
| `results/slakh_r1/split.json` | exp-local copy derived from the manifest (frozen) |

Corpus facts (docs/DATA.md): indexable 1288/270/151 (`Track00846` corrupt,
excluded in the manifest); mixes carry a ~5s reverb tail (rule −0.5..12s);
chunk token cap 4096 (corpus max chunk 2725); all tracks 16k mono.

## 1. G5-v2 (mechanism review — not a gate for R1)

```bash
python -m murwkv.eval.memory_probe_v2 --exp results/gate5_probe_v2 \
  --data-root /root/autodl-tmp/data/babyslakh/babyslakh_16k \
  --ckpt results/gate4_overfit_v2/final.pt --device cuda
```
Frozen G4 21.9M AMT; track-level split; test = the 4 tracks G4 never trained
on; metadata-verified Guitar/Piano stems (all 20 tracks); bit-identical neutral
chunks; official learned decay; ridge probes on h_last-only / S-only / h_last+S.

## 2. R1 training

```bash
# calibration (throughput + val cost) — 200 steps, no selection
python -m murwkv.training.train --exp results/slakh_r1_calib \
  --data-root /root/autodl-tmp/data/slakh2100_16k_from_flac --splits \
  --tracks $(cat /tmp/slakh_r1_train_tracks.txt) --steps 200 \
  --max-tokens-per-chunk 4096 --units 4 --seed 42 --log-every 10

# full run (steps decided after calibration; final selection by val loss)
python -m murwkv.training.train --exp results/slakh_r1 \
  --data-root /root/autodl-tmp/data/slakh2100_16k_from_flac --splits \
  --tracks $(cat /tmp/slakh_r1_train_tracks.txt) \
  --max-tokens-per-chunk 4096 --units 4 --seed 42 --steps N \
  --val-every 1000 --val-limit 32 --save-every 500 --log-every 10
```
The trainer never sees the test split; `best_val.pt` is the sole selection
criterion (val = 20 tracks from the official validation split).

## 3. Paired held-out evaluation (60 test tracks, sealed)

```bash
python -m murwkv.eval.eval_heldout --exp results/slakh_r1 \
  --ckpt results/slakh_r1/best_val.pt \
  --data-root /root/autodl-tmp/data/slakh2100_16k_from_flac --splits \
  --split test --mode both --max-tokens 4096
python scripts/report_paired_eval.py --exp results/slakh_r1
```
`reset` = per-5s-unit state+conv-carry+shift-lead zeroed; everything else
identical (checkpoint, inputs, greedy decoding). No teacher forcing.

## 4. Verdicts (strict)

- Level 3 PASS: non-trivial held-out F1 with valid MIDI / 0 truncation.
- Level 4 PASS: Level 3 holds AND paired bootstrap CI of
  `continuous − reset` (onset F1) excludes 0; mean positive on held-out tracks.
  Otherwise: "positive trend, not established" (CI crosses 0) or "no evidence".
- Train-track continuous≫reset does NOT count toward Level 4.
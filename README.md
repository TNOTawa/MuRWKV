# MuRWKV — Pure RWKV-7 Automatic Music Transcription

**10-hour from-scratch research run. Full results: [`REPORT_10H.md`](REPORT_10H.md).
Gate ledger: [`results/gates.json`](results/gates.json) (G0–G6 all PASS).**

## What

MuRWKV is a from-scratch, **pure RWKV-7** (no attention of any kind) sequence model
that transcribes monaural music audio into MIDI. A single unified RWKV stream
consumes log-Mel audio frames and emits MT3-style MIDI event tokens:

```
AUDIO(t0..t1) → MIDI(t0..t1) → AUDIO(t1..t2) → MIDI(t1..t2) → ...
```

The recurrent state is **never reset** between 5-second chunks — the model must
maintain musical state (instruments, key, tempo, sustained notes) directly from
the audio history it has heard.

## Answers up front

1. **What is MuRWKV?** A 21.9M-parameter pure-RWKV-7 model that learns
   Audio→MIDI transcription from scratch (random init, no pretrained weights).
2. **Is it pure RWKV?** Yes. Recurrent backbone = official RWKV-7 "Goose"
   time-mixing/FFN (train_temp reference, plus the vendored official clampw
   CUDA kernel). No self-attention, no cross-attention, no MLA/KDA/Mamba.
   Log-Mel + linear/causal-conv frontend is an acoustic frontend, not a second
   sequence architecture.
3. **Pretrained weights?** None. All parameters randomly initialized with the
   official RWKV-7 init scheme. No foundation-model features, no distillation.
   The MuScriptor-medium checkpoint is used **only** as a comparison baseline
   (separate isolated venv, not loaded by MuRWKV).
4. **Public data?** BabySlakh 16k (Zenodo 4603870, CC-BY-4.0, MD5 verified).
   Fixed train/valid/test splits are saved in every experiment dir; 10 training
   songs excluded all held-out tracks.
5. **Level reached in the 10h run:** **Level 2 (pipeline + audio state) with
   strong Level-1 evidence and a partial Level-4 signal** — see REPORT.
   G0–G6 all PASS. Held-out note F1 ≈ 0.01 (10-song memorizer; generalization
   needs Slakh-scale data — documented as the Level-3 gap).
6. **Continuous vs reset:**
   * memory probe: continuous 100% vs reset 48% (chance) — the state carries
     remote acoustic identity through 2 identical neutral chunks;
   * train tracks: continuous onset F1 0.579 vs reset 0.081 (Track00008:
     0.991 vs 0.063), fewer instrument switches/flicker with continuous.
7. **Listening MIDI:** `artifacts/listening/<track>/` — `gt.mid`,
   `murwkv_continuous.mid`, `murwkv_reset.mid`, `muscriptor.mid` (baseline).
   Open them in any DAW/MuseScore.
8. **Checkpoints:** `results/gate3_overfit_v2/final.pt` (1-song overfit),
   `results/gate4_overfit_v2/final.pt` (10-song overfit) — also on the private
   HF repo **`TNOT/MuRWKV-R0`** (`murwkv_r0_gate4.pt`, 43.8 MB, bf16:
   `hf download TNOT/MuRWKV-R0 murwkv_r0_gate4.pt --include murwkv_r0_gate4.pt`).
   GitHub push was blocked by a token without write scope (commits are local —
   see `REPORT_10H.md` §Final Git/HF State).
9. **Reproduce:** `REPORT_10H.md` §Reproduction (clean-clone commands).

## Repo layout

```
src/murwkv/model/rwkv7.py        RWKV-7 core (official math/init; CUDA kernel hook)
src/murwkv/model/murwkv_model.py unified audio+MIDI stream model + loss protocol
src/murwkv/tokenizer.py          MT3_FULL_PLUS tokenizer + tie/open-note protocol
src/murwkv/data/babyslakh.py     BabySlakh dataset (mel cache, chunk windows)
src/murwkv/training/train.py     trainer (official param groups/LR/BF16)
src/murwkv/eval/infer.py         true-recurrent streaming transcriber
src/murwkv/eval/eval_heldout.py  Gate-6 evaluation (continuous vs reset)
src/murwkv/eval/memory_probe.py  Gate-5 controlled audio-memory probe
tests/                           QA: parity, tokenizer, alignment, gate1, smoke
scripts/                         plots, downloads, baseline, environment record
```
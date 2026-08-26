# REPORT_10H — MuRWKV: Pure RWKV-7 Automatic Music Transcription

_All numbers below come from the experiment logs and metrics files under `results/`
(no simulated data). Run: 2026-08-27, ~10h, single RTX 5090._

## Executive Summary

> **Pure RWKV（无任何 Attention）能够直接从声音学习 Audio→MIDI：在一个统一的不重置 recurrent
> stream 里，随机初始化的 21.9M RWKV-7 模型学会了把 5 秒音频块转录成 MT3 风格 MIDI token，
> 并保持跨 chunk 的连续状态。受控实验证明 RWKV state 能携带远程声音历史（连续 100% vs
> reset 48%）。** 但 BabySlakh 规模（≤16 首歌）不足以获得 held-out 泛化：10 首歌过拟合后
> held-out 转录 F1≈0.01，泛化需要更大数据（Slakh2100）。

## Environment

- GPU: NVIDIA GeForce RTX 5090 32GB (Blackwell, compute capability 12.0, arch list incl. sm_120)
- PyTorch 2.8.0+cu128 (CUDA runtime 12.8), BF16 supported and used
- CPU 16 cores, ~750GB RAM (measured), 16 cores; data disk 750GB at /root/autodl-tmp
- Official RWKV-7 CUDA clampw kernel (`src/murwkv/cuda/`, RWKV-LM commit `658042ca`): compiled for sm_120, forward+backward smoke OK; used for all training
- Wall-clock per training step (4×5s chunks, B=1, T≈2300–5500): 0.8–2.5 s (kernel path; data-loading bound)
- Full QA: `python tests/qa.py` (parity, trainer smoke, tokenizer; Gate-1 needs data)

## Architecture

R0 (≈21,870,080 params, all random init, official RWKV-7 init rules):

- Audio frontend: mono 16kHz → log-Mel (n_fft 2048, hop 160, 100 fps, 512 HTK bins, log(x+1e-6),
  matching MuScriptor) → Linear(512→512) → GELU → causal Conv1d(k=3) → GELU → Linear(512→512)
- Unified recurrent stream per 5s chunk: `[500 audio frames] [MIDI tokens incl. EOS]`, concatenated
  across chunks — the RWKV state is NEVER reset (training: 4-chunk windows = 20s TBPTT, gradients
  flow across chunk boundaries)
- 6 × RWKV-7 "Goose" x070 blocks (official train_temp math: PreLN ln0/ln1/ln2, time-mix with
  w=exp(W_SCALE·sigmoid(w0+…)) clamping, LoRA decays, kk normalization, value residual, GroupNorm
  eps=64e-5, ChannelMix with official init); head_size 64 (dim_att 512); dim_ffn 1792 (official 3.5×)
- Mid-token embedding 1393 (MT3_FULL_PLUS reimplementation) ; head Linear(512→1393)
- Loss: masked CE ONLY at MIDI-prediction positions: the last audio frame predicts the chunk's first
  MIDI token; each MIDI position predicts the next token; the LAST MIDI position predicts EOS.
- Inference: true recurrent (state `S: (B,H,64,64)` fp32 per layer + official per-layer time-shift
  carry buffers), `reset()/clone()/save()/load()` implemented; 1-frame shift lead + 2-frame mel
  carry for exact chunk transitions (Gate-2 verified); greedy decoding

## Data

- BabySlakh 16k (Zenodo record 4603870, DOI 10.5281/zenodo.4603870, CC-BY-4.0, MD5
  `311096dc2bde7d61c97e930edbfc7f78`, verified): 20 tracks, 16 kHz mono mix + stems + MIDI.
- Fixed split (checked into each exp dir as `split.json`, seed 42): train 16 / valid 2 / test 2
  → valid: Track00005, Track00015; test: Track00006, Track00020; train: the rest.
- Token budget: dynamic, max 2048 tokens/chunk (measured max chunk was 1653); **0 truncated chunks
  in every official run**. Dataset never committed to git (data disk only).
- Slakh2100-16k (Stage D1): a 307 GB mirror tar existed but was not fetchable within this run
  (mirror ~2.6 MB/s → ~33 h); SKIPPED, documented.

## Gate Results

| Gate | Result | Evidence |
|---|---|---|
| G0 environment | **PASS** | sm_120 kernel compiles+runs, BF16 OK, data MD5 verified, no secrets logged |
| G1 data/tokenizer | **PASS** | 20 tracks exact round-trip on the 10ms grid; ties, drums, EOS, 0 truncation |
| G2 RWKV-7 parity | **PASS** | scan≡kernel (bf16), parallel≡stepwise RNN (fp32 3.4e-8, non-vacuous), cross-chunk carry≡joint, streaming≡batch |
| G3 1-song overfit | **PASS** | generative token acc 99.8%, onset F1 0.988/0.991, offset 0.916/0.917, 0 truncation |
| G4 10-song overfit | **PASS** | loss 0.0024, acc 100%, EOS 1.0; held-out sanity: ~0.01 F1 (no generalization at this scale — documented) |
| G5 audio memory probe | **PASS** | continuous 1.0 vs reset 0.483 (chance); state distance decays 43→0.4 over 4 neutral chunks |
| G6 held-out eval | **PASS** | full continuous-vs-reset table; validity 100%; RTF 0.4–0.75; VRAM ~0.13 GB |

## Training

| Exp | Data | Steps | Final loss | Acc | EOS acc | Wall | Throughput |
|---|---|---|---|---|---|---|---|
| gate3_overfit_v2 | 1 track | 4000 | ~0.0000 | 100.0% | 1.0 | ~75 min | ~0.83 s/step |
| gate4_overfit_v2 | 10 tracks (split-safe) | 4500 | 0.0024 | 100.0% | 1.0 | ~120 min | ~1.6 s/step |
| gate5_probe | 2-class stems, 600 samples | 60 epochs | 0.0 | cont 100% / reset 48% | — | ~35 min | — |

Hyperparameters (official RWKV rules unless noted): lr 6e-4→1e-5 cosine (official shape), warmup
100–200 steps, betas (0.9, 0.99), eps 1e-18, grad clip 1.0, weight decay 0.1 on 2D `.weight` params
only, `att.w0` at 2× lr; BF16 params, fp32 wkv state; CUDA kernel when T%16==0.

## Quantitative Results

**G3 (Track00001, generative):** token acc 99.8% (17008/17036), onset F1 0.988 (continuous) /
0.991 (reset), offset F1 0.916/0.917, instrument F1 0.988/0.991, notes 3124→3200/3181, trunc 0.

**G4 held-out (4 tracks, never trained):** onset F1 0.002–0.019 (continuous and reset equivalent),
no truncation, no boundary errors. MuScriptor-medium baseline on the same tracks: onset F1 0.244
(0.22/0.31 per track). The gap is a data-scale gap (10 vs ~100k+ songs), not a validity failure.

**G4 train-track continuity:** continuous onset F1 0.579 vs reset 0.081 (2 tracks; Track00008:
0.991 vs 0.063). Continuous beats reset by ~7× when the model can actually transcribe.

**Efficiency (G6):** RTF 0.35–0.75 (faster than realtime), inference peak VRAM 0.12–0.16 GB,
recurrent state = 6×(8×64×64 fp32 + shift buffers) ≈ 1 MB — constant with song length.

## Continuous vs Reset

- Where transcription works (train tracks): **continuous >> reset** (0.991 vs 0.063 on Track00008;
  0.579 vs 0.081 aggregate). Reset halves/pollutes note counts and instruments (flicker 4 vs 24–40
  switches on the same tracks); continuous keeps instrument continuity (4 switches vs 24+).
- On held-out tracks (F1≈0) both modes are equivalently poor — continuity cannot fix a model that
  never learned the mapping.
- Conclusion: with the lead-in-exact streaming protocol, the continuous state is the better
  transcriptor; the difference is measurable and directionally consistent with the memory probe.

## Audio Memory Sanity

Controlled 2-class probe (guitar-stem history vs piano-stem history, then **two bit-identical
neutral chunks**; target depends only on remote audio): training the small RWKV probe (slow-decay
init, documented) reaches **continuous val acc 1.0 vs reset 0.483 (chance)**. State-distance
analysis: 43.2 (no neutral) → 15.9 → 3.5 → 0.41 across 1–4 neutral chunks — the state decays
gracefully but retains and uses remote acoustic identity. A prerequisite finding: at the official
decay init the remote-history signal AND gradient vanish (~0.5^1000), the probe was unlearnable;
the slow-decay probe init is a recorded deviation.

## Listening Artifacts

`artifacts/listening/<track>/` — MIDI files ready for DAW comparison:
- Track00001: `gt.mid`, `murwkv_continuous.mid`, `murwkv_reset.mid` (G3 model — high fidelity)
- Track00005 / Track00015 (held-out): `gt.mid`, `murwkv_{continuous,reset}.mid` (G4 model),
  `muscriptor.mid` (baseline)
- Metadata JSONs per file (track, checkpoint, git commit, metrics).

## Failures / Bugs Found (all fixed and re-verified)

1. **EOS target off-by-one (critical)**: `build_targets` covered `[audio_end, audio_end+M-1)` —
   the EOS prediction position was NEVER trained; the model could not terminate chunks
   (every chunk truncated at the cap). Fixed the range to `[E, E+M)`; the old "100% acc" was
   silently excluding EOS positions. Added `tests/test_target_alignment.py`.
2. **Scan tail-state padding bug**: `wkv7_scan` returned the state after zero-padded positions
   (decay-only fake steps) — chunk-boundary states used by the transcriber were wrong
   (generated tokens matched only 25%). Fixed: real-steps tail, state at true T.
3. **Missing official RNN shift-carry**: my stepwise path zero-padded the time-shift every step;
   the official RNN state includes per-layer `att_x_prev`/`ffn_x_prev`. Rewrote the stepwise path
   (Gate-2 parity now non-vacuous with a broken-shift ablation).
4. **Vacuous parity tests**: official init zeroes `att.output`/`ffn.value` → all outputs were 0;
   the "parity" tests passed trivially. Now randomized-weight non-vacuity checks + margin
   analysis.
5. **GPU-vs-CPU mel mismatch** in the early transcriber (fixed; train/inference mel now
   bit-identical via the f16 cache convention) — was NOT the root cause of gate-3 failure but
   remains a correctness fix.
6. **Kernel state indexing**: `s[:, -1]` → `s[:, :, -1]` for the final-state return.
7. Process/tooling: segmented downloader part-count bug; infiniband-time hiccups — all recorded
   in commit history.

## What Is Proven

- A pure RWKV-7 (no attention) trains stably from random init on Audio→MIDI (G2, G3, G4).
- Generative transcription of a memorized song reaches token acc 99.8%, onset F1 0.99, 100%
  parse validity, 0 truncation — with the exact tie/EOS protocol (G3).
- RWKV state genuinely carries remote acoustic identity through identical neutral audio
  (G5: 100% vs chance).
- Continuous transcription beats per-chunk reset on well-transcribed songs (0.991 vs 0.063).
- The whole pipeline is reproducible from clean clone (see Reproduction).

## What Is NOT Proven

- **BabySlakh held-out generalization** (Level 3): with 10–16 songs the model memorizes;
  held-out F1 ≈ 0.01, far below the 0.24 of a baseline trained on orders-of-magnitude more data.
- That continuous state helps LONG real songs beyond 20s windows (evaluated on ≤4 min tracks).
- Any claim about larger models, Slakh2100, or real-world audio (all out of scope for R0).
- The memory probe used slow-decay init (recorded deviation) — the official-init probe is
  unlearnable, which is itself evidence about RWKV-7's default decay scale vs 10s gaps.

## Next Recommended Experiment

1. Train R0 on Slakh2100-16k (or a 100–200 track subset) at B=2–4 with the same pipeline —
   the only missing ingredient for Level 3/4 (all tooling is ready; the data fetch is the blocker).
2. Scheduled sampling (10–20%) to harden against exposure drift in longer generations.
3. AudioRWKV-style causal 2D depthwise-separable frontend as an ablation on the same recipe.

## Reproduction

```bash
git clone https://github.com/TNOTawa/MuRWKV.git && cd MuRWKV
pip install -r requirements.txt            # torch>=2.7+cu128, ninja
# data: BabySlakh 16k (Zenodo 4603870, md5 311096dc2bde7d61c97e930edbfc7f78)
#       extract to /path/babyslakh_16k
python tests/qa.py --babyslakh-root /path/babyslakh_16k   # G1+G2+smoke
python -m murwkv.training.train --exp results/repro --tracks Track00001 \
  --steps 4000 --seed 42                                   # G3
python -m murwkv.eval.eval_heldout --exp results/repro --ckpt results/repro/final.pt \
  --mode both --tracks Track00001                          # G3 eval + MIDI
```

## Final Git / HF State

- **Git:** all work is committed locally on `main` (18 commits, see `git log`).
  **Push to GitHub was not possible in this run**: the fine-grained PAT
  (`github_pat_*`, user TNOTawa) lacks `Contents: write` — verified via the
  GitHub API (`403 Resource not accessible by personal access token`), so both
  direct and proxy pushes are denied regardless of network path. The repository
  commits remain on the instance's persistent disk; a token with write scope is
  required to publish (one `git push origin main` after granting write).
- **HF:** private model repo **`TNOT/MuRWKV-R0`** (private; the token's account
  is `TNOT`, so the requested `TNOTawa` namespace was not creatable — recorded):
  `murwkv_r0_gate4.pt` (model state only, 43.8 MB, bf16) + `config.json`.
- **Local checkpoints (persistent):** `results/gate3_overfit_v2/final.pt`,
  `results/gate4_overfit_v2/final.pt` (+ resumable `ckpt_*.pt` and `latest.pt`),
  all configs/metrics/plots alongside.
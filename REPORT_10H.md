# REPORT_10H — MuRWKV: Pure RWKV-7 Automatic Music Transcription

_All numbers below come from the experiment logs and metrics files under `results/`
(no simulated data). Run: 2026-08-27, ~10h, single RTX 5090._

## Executive Summary

> **Pure RWKV（无任何 Attention）能够直接从声音学习 Audio→MIDI：在一个统一的不重置 recurrent
> stream 里，随机初始化的 21.9M RWKV-7 模型学会了把 5 秒音频块转录成 MT3 风格 MIDI token，
> 并保持跨 chunk 的连续状态。受控实验证明 RWKV state 能携带远程声音/声源身份（连续 100% vs
> reset 48%，见 Erratum：这是 track/source-level 记忆，尚不能证明跨曲目乐器泛化）。**
> 但 BabySlakh 规模（≤16 首歌）不足以获得 held-out 泛化：10 首歌过拟合后 held-out 转录
> F1≈0.01，泛化需要更大数据（Slakh2100）。本报告末尾附有第一轮评审后的措辞修正（Erratum）。

## Post-Review Erratum (Round 1)

科学结论收紧（实现与工程验证不变；评审未推翻核心结论）：

1. **99.8% token acc / 0.99 onset F1 仅证明"背下了一首歌"**：是 G3 单曲过拟合
   （Track00001，训练曲目）的生成质量度量，不是通用转录能力。后续引用必须标
   `one-song memorized training track`。
2. **G5 "guitar vs piano identity" 降级**：G5 的 100% vs 48% 证明的是
   *remote acoustic/source identity retention*（远程声源/曲目身份存在并可用于分类），
   不能证明跨曲目的乐器级泛化——数据每类只有一首歌、未按 metadata 验证 stem、
   sample-level 划分有泄漏风险。详见 §Audio Memory Sanity。
3. **"partial Level-4 signal" 撤回**：continuous 0.579 vs reset 0.081 只发生在
   **已背下来的训练歌曲**上（held-out 双臂都 ≈0.01）；改述为
   *train-sequence continuity signal; Level 4 (persistent-state 泛化连续性) not yet established*。
4. **训练时间跨度澄清**：所谓 20s TBPTT 实现为"4 个 5s chunk 的 window 一次并行训练"
   （window 间 shuffle，无整曲 state 贯穿）；推理可无限 recurrent，但"无限推理"≠"训练过无限尺度"。
5. **环境溯源修正**：原 `results/environment.json` 的 `cpu_cores=128 / ram=791.2GB`
   与报告"16 cores / 750GB"都读自宿主机（容器配额未记录）。`scripts/record_environment.py`
   已改为读取 cgroup v2 `cpu.max` / `memory.max`（v1 回退），并显式记录容器配额 + 宿主机参考值。
6. **G5-v2（无泄漏冻结探针）已实现，待 GPU 运行**：冻结 G4 AMT checkpoint、只训 ridge
   linear probe、按 track 划分 train/val/test（test=G4 从未训练过的 4 首歌）、
   按 Slakh metadata 验证 stem 乐器、official learned decay（无 `w0` 偏差）。
   代码 `src/murwkv/eval/memory_probe_v2.py`，测试 `tests/test_probe_v2.py`，
   数据制品（split/stems/samples）已在 `results/gate5_probe_v2/` 生成。

## Environment

- GPU: NVIDIA GeForce RTX 5090 32GB (Blackwell, compute capability 12.0, arch list incl. sm_120)
- PyTorch 2.8.0+cu128 (CUDA runtime 12.8), BF16 supported and used
- CPU/RAM provenance: see **Erratum #5** — the original run recorded host values
  (128 cores / 791 GB in `results/environment.json`, "16 cores" in the report);
  the actual container quota was not recorded at the time. The recorder now reads
  cgroup v2 (`cpu.max` → quota cores, `memory.max` → limit GB) with host counts as
  reference only (regenerated `results/environment.json` on the no-GPU container:
  quota 0.5 cores / 2.1 GB, host 128 cores / 791.2 GB).
- Official RWKV-7 CUDA clampw kernel (`src/murwkv/cuda/`, RWKV-LM commit `658042ca`): compiled for sm_120, forward+backward smoke OK; used for all training
- Wall-clock per training step (4×5s chunks, B=1, T≈2300–5500): 0.8–2.5 s (kernel path; data-loading bound)
- Full QA: `python tests/qa.py` (parity, trainer smoke, tokenizer; Gate-1 needs data)

## Architecture

R0 (≈21,870,080 params, all random init, official RWKV-7 init rules):

- Audio frontend: mono 16kHz → log-Mel (n_fft 2048, hop 160, 100 fps, 512 HTK bins, log(x+1e-6),
  matching MuScriptor) → Linear(512→512) → GELU → causal Conv1d(k=3) → GELU → Linear(512→512)
- Unified recurrent stream per 5s chunk: `[500 audio frames] [MIDI tokens incl. EOS]`, concatenated
  across chunks — the RWKV state is NEVER reset. Training spans 4-chunk windows
  (= 20s TBPTT: one window trained at a time, windows shuffled; gradients flow across the
  chunk boundaries INSIDE a window; NOT full-song state carry — Erratum #4).
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
| G3 1-song overfit | **PASS** | generative token acc 99.8% (single memorized training track Track00001), onset F1 0.988/0.991, offset 0.916/0.917, 0 truncation |
| G4 10-song overfit | **PASS** | loss 0.0024, acc 100%, EOS 1.0; held-out sanity: ~0.01 F1 (no generalization at this scale — documented); continuous>reset only on memorized train tracks |
| G5 audio memory probe | **PASS (as source-identity)** | continuous 1.0 vs reset 0.483 (chance); but scored as remote *source/track* identity — instrument-level cross-track generalization NOT established (see Erratum #2); leak-free G5-v2 ready |
| G6 held-out eval | **PASS** | full continuous-vs-reset table; validity 100%; RTF 0.4–0.75; VRAM ~0.13 GB; held-out arms equivalent at F1≈0 |

## Training

| Exp | Data | Steps | Final loss | Acc | EOS acc | Wall | Throughput |
|---|---|---|---|---|---|---|---|
| gate3_overfit_v2 | 1 track, memorized | 4000 | ~0.0000 | 100.0% | 1.0 | ~75 min | ~0.83 s/step |
| gate4_overfit_v2 | 10 tracks (split-safe) | 4500 | 0.0024 | 100.0% | 1.0 | ~120 min | ~1.6 s/step |
| gate5_probe | 2-class stems (1 song/class), 600 samples | 60 epochs | 0.0 | cont 100% / reset 48% | — | ~35 min | — |
| gate5_probe_v2 | leak-free frozen-AMT probe (code ready, GPU pending) | ridge | — | — | — | — | — |

Hyperparameters (official RWKV rules unless noted): lr 6e-4→1e-5 cosine (official shape), warmup
100–200 steps, betas (0.9, 0.99), eps 1e-18, grad clip 1.0, weight decay 0.1 on 2D `.weight` params
only, `att.w0` at 2× lr; BF16 params, fp32 wkv state; CUDA kernel when T%16==0.

## Quantitative Results

**G3 (Track00001 — ONE-SONG MEMORIZED TRAINING TRACK, not generalization):**
token acc 99.8% (17008/17036), onset F1 0.988 (continuous) / 0.991 (reset),
offset F1 0.916/0.917, instrument F1 0.988/0.991, notes 3124→3200/3181, trunc 0.
Every citation of these numbers must carry the "memorized training track" label
(Erratum #1).

**G4 held-out (4 tracks, never trained):** onset F1 0.002–0.019 (continuous and reset equivalent),
no truncation, no boundary errors. MuScriptor-medium baseline on the same tracks: onset F1 0.244
(0.22/0.31 per track). The gap is a data-scale gap (10 vs ~100k+ songs), not a validity failure.

**G4 train-track continuity (MEMORIZED SEQUENCES ONLY — Erratum #3):**
continuous onset F1 0.579 vs reset 0.081 (2 tracks; Track00008:
0.991 vs 0.063). Continuous beats reset by ~7× when the model can actually transcribe —
but this is a memorized-sequence effect; on held-out tracks both arms are ≈0.01, so it is
**not** evidence that a persistent state improves generalizable AMT continuity (Level 4 not
established).

**Efficiency (G6):** RTF 0.35–0.75 (faster than realtime), inference peak VRAM 0.12–0.16 GB,
recurrent state = 6×(8×64×64 fp32 + shift buffers) ≈ 1 MB — constant with song length.

## Continuous vs Reset

- Where transcription works (MEMORIZED train tracks): **continuous >> reset** (0.991 vs 0.063 on Track00008;
  0.579 vs 0.081 aggregate). Reset halves/pollutes note counts and instruments (flicker 4 vs 24–40
  switches on the same tracks); continuous keeps instrument continuity (4 switches vs 24+).
- On held-out tracks (F1≈0) both modes are equivalently poor — continuity cannot fix a model that
  never learned the mapping.
- **Scope guard (Erratum #3):** all of the above is about memorized training sequences. The claim
  "persistent state improves generalizable AMT continuity" is NOT established — it can only be
  tested once a model transcribes held-out songs non-trivially (next stage: Slakh subset).

## Audio Memory Sanity

**Corrected interpretation (Erratum #2).** The controlled 2-class probe
(guitar-stem history vs piano-stem history, then **two bit-identical neutral chunks**; target
depends only on remote audio) reached **continuous val acc 1.0 vs reset 0.483 (chance)** with a
separately-trained small RWKV probe (slow-decay init `w0 -= 6`, a recorded deviation). State
distance: 43.2 (no neutral) → 15.9 → 3.5 → 0.41 across 1–4 neutral chunks.

What this DOES prove:
> **RWKV recurrent state can learn to retain and use remote acoustic SOURCE identity through
> two identical neutral chunks** (the classifier reads it after the neutral window).

What it does NOT prove (and why):
> Instrument-level (guitar-vs-piano) generalization across tracks: the probe used ONE song per
> class, took `stems[0]` without verifying Slakh metadata, and split train/val at the SAMPLE
> level (crops of the same two songs in both) — the classifier could just have learned
> "sounds like Track00001 vs Track00003" (track/source identity), and the official-decay init
> was artificially slowed. The slow-decay finding itself (remote signal + gradient vanish under
> the official init) stands as recorded.

**G5-v2 (leak-free, ready — needs GPU):** `python -m murwkv.eval.memory_probe_v2` freezes the
G4 21.9M AMT checkpoint, trains only an exact ridge (linear) probe on its recurrent state
(h_last + last-layer S), uses ALL 20 BabySlakh tracks with metadata-verified Guitar/Piano stems,
splits at TRACK level with probe-test = the 4 tracks G4 never trained on, and uses the official
learned decay (no `w0` bias). Arms: continuous / reset / lead1 (1-frame view bound), plus a
state-distance decay table. Data-side artifacts are already generated in
`results/gate5_probe_v2/`; tests in `tests/test_probe_v2.py`.

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
- Generative transcription of a MEMORIZED training song reaches token acc 99.8%, onset F1 0.99,
  100% parse validity, 0 truncation — with the exact tie/EOS protocol (G3). (Memorization-level
  evidence only; it does show the mapping is learnable end-to-end.)
- RWKV state genuinely carries remote acoustic SOURCE identity through identical neutral audio
  (G5: 100% vs chance; claim scope per Erratum #2).
- Continuous transcription beats per-chunk reset on memorized train songs (0.991 vs 0.063).
- The whole pipeline is reproducible from clean clone (see Reproduction).

## What Is NOT Proven

- **BabySlakh held-out generalization** (Level 3): with 10–16 songs the model memorizes;
  held-out F1 ≈ 0.01, far below the 0.24 of a baseline trained on orders-of-magnitude more data.
- Instrument-level (cross-track) memory generalization — the G5 claim the probe design could
  not support; G5-v2 is built to test it.
- That persistent state helps generalization-level AMT continuity ("Level 4") — the
  continuous-vs-reset gap is so far only a memorized-sequence effect.
- That continuous state helps LONG real songs beyond 20s windows (evaluated on ≤4 min tracks;
  training optimized the 20s horizon, not full-song state carry).
- Any claim about larger models, Slakh2100, or real-world audio (all out of scope for R0).
- The memory probe used slow-decay init (recorded deviation) — the official-init probe is
  unlearnable, which is itself evidence about RWKV-7's default decay scale vs 10s gaps; the
  frozen-AMT probe (G5-v2) is the official-decay measurement.

## Next Recommended Experiment

1. Run G5-v2 (code + data artifacts ready): `python -m murwkv.eval.memory_probe_v2` — establishes
   whether the frozen G4 AMT state carries instrument-level (cross-track) identity under the
   official learned decay, with a leak-free track-level split.
2. Train R0 on a Slakh2100-16k subset (100–200 tracks) at B=2–4 with the same pipeline — the only
   missing ingredient for Level 3/4 (all tooling is ready; the 307 GB download is in progress).
   Only when held-out F1 becomes non-trivial does the continuous-vs-reset comparison regain the
   right to answer the Level-4 question.
3. Scheduled sampling (10–20%) to harden against exposure drift in longer generations.
4. AudioRWKV-style causal 2D depthwise-separable frontend as an ablation on the same recipe.
5. If long-horizon continuity is the target, raise the training horizon (longer windows /
   cross-window state carry + TBPTT) — "infinite recurrent inference" ≠ "trained at that scale".

## Reproduction

```bash
git clone https://github.com/TNOTawa/MuRWKV.git && cd MuRWKV
pip install -r requirements.txt            # torch>=2.7+cu128, ninja
# data: BabySlakh 16k (Zenodo 4603870, md5 311096dc2bde7d61c97e930edbfc7f78)
#       extract to /path/babyslakh_16k
python tests/qa.py --babyslakh-root /path/babyslakh_16k   # G1+G2+smoke+probe-v2
python -m murwkv.training.train --exp results/repro --tracks Track00001 \
  --steps 4000 --seed 42                                   # G3
python -m murwkv.eval.eval_heldout --exp results/repro --ckpt results/repro/final.pt \
  --mode both --tracks Track00001                          # G3 eval + MIDI
python -m murwkv.eval.memory_probe_v2 --exp results/gate5_probe_v2 \
  --data-root /path/babyslakh_16k --ckpt results/gate4_overfit_v2/final.pt \
  --device cuda                                            # G5-v2 leak-free probe
```

## Final Git / HF State

- **Git:** all work is committed locally on `main` (31 commits incl. the post-review
  revision, see `git log`).
  **Push to GitHub was not possible in this run**: the fine-grained PAT
  (`github_pat_*`, user TNOTawa) lacks `Contents: write` — verified via the
  GitHub API (`403 Resource not accessible by personal access token`), so both
  direct and proxy pushes are denied regardless of network path. The repository
  commits remain on the instance's persistent disk; a token with write scope is
  required to publish (one `git push origin main` after granting write).
- **HF:** private model repo **`TNOT/MuRWKV`** (renamed from `TNOT/MuRWKV-R0`;
  the token's account is `TNOT`, so the requested `TNOTawa` namespace was not
  creatable — recorded): `murwkv_r0_gate4.pt` (model state only, 43.8 MB,
  bf16) + `config.json`. R1/R2 checkpoints were later mirrored into the same
  repo (see REPORT_R2.md §Checkpoint persistence).
- **Local checkpoints (persistent):** `results/gate3_overfit_v2/final.pt`,
  `results/gate4_overfit_v2/final.pt` (+ resumable `ckpt_*.pt` and `latest.pt`),
  all configs/metrics/plots alongside.
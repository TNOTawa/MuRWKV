# REPORT_R2 — exposure / state-distribution training round (Slakh, 2026-08-28)

Setup: RTX 5090 32 GB, torch 2.8.0+cu128, BF16, official RWKV-7 clampw kernel
(+ `forward_init` extension, see below), container quota 16 cores / 98.8 GB
(cgroup). Protocol constraints honored: test split sealed (identical frozen
manifest to R1: 120 train / 20 val / 60 official-test), checkpoint selection
by validation loss only, val criterion held at the R1 protocol (4-unit
windows, fresh state, clean teacher-forced monolithic forward — `--val-units
4`), no architecture changes (the RWKV-7 math is unchanged; only the kernel's
initial-state seeding is new, parity-tested).

## 1. Question

R1 established that teacher-forced validation learning emerged (val loss
1.593 → 1.330 on 20 unseen tracks) but failed to translate into free-running
held-out transcription (onset F1 0.010–0.026). The candidate mechanism
(REPORT_R1 §7): an **exposure/state-distribution mismatch** — R1 training gave
the model ≈20 s lifetimes from a fresh state with ground-truth teacher-forced
MIDI history, while continuous inference runs minutes of self-generated MIDI
through accumulated state. R2 tests the reviewer's two architecture-neutral
levers, applied together:

- **A — noisy history (scheduled-sampling proxy).** MIDI *input* tokens are
  corrupted with annealed probability p ∈ [0, 0.15] (uniform over event ids
  ≥ 3; audio and PAD positions never corrupted; loss targets stay clean GT).
  Exposes the state to erroneous MIDI history of the kind its own free-running
  output produces.
- **B — cross-window state-carry training.** The training window is extended
  from 4×5 s to 16×5 s (80 s) and processed in ~2048-token parallel passes
  with the RWKV state carried (detached, truncated BPTT) across passes and the
  previous real position's embedding re-fed as shift lead — the exact
  continuity protocol continuous inference applies at every chunk boundary
  (Gate-2-verified lead+carry protocol). The audio frontend runs once per
  window, so the causal-conv context is exact across pass boundaries.

## 2. Implementation notes (recorded deviations)

- **Kernel `forward_init`** (vendored official clampw kernel, new op beside
  the untouched `forward`): seeds the per-thread state with an initial S
  (B,H,N,N). The official backward is reused unchanged — it reconstructs
  stateT from chunk-boundary states, so input gradients account for the
  propagated seed while the gradient INTO the seed is discarded (detached
  carry semantics), which is exactly truncated BPTT.
- **State-convention fix (bug found and fixed during R2 bring-up):** the
  official kernel's chunk-state buffer holds **S transposed** (element
  (c,i,j) = S[j,i]; verified against the python scan to 2e-8), so the CUDA
  autograd wrappers previously returned the final state in a different
  convention than `wkv7_scan`/`RWKVState`. This was unobservable before R2
  (kernel-saved states were never carried across calls — all init_state
  consumers used the scan path). Both wrappers now return the final state in
  the row convention; a chained A→B continuation parity test would have
  caught the original bug and is now part of Gate 2.
- Parity (Gate 2 additions, all PASS): kernel-with-init vs scan (fwd + bwd,
  no grad into carry); segmented-carry vs joint forward (first segment exact;
  all positions within bf16 tolerance, 2.4e-4; final state equal); trainer
  carry+noise smoke (memorized batch, loss 7.4 → 2.2 with 30 % input noise).

## 3. Run

| config | value |
|---|---|
| run | 5,000 steps, B=1, 16×5s windows (80 s) → 8–16 passes of 2047+1 tokens, seed 42 |
| exposure | ~1.7× the R1 token budget, at 4× R1's state lifetime (R1: 12k steps × 4 units) |
| levers | carry-seg 2048 + noise-p 0.15 (annealed over 500 steps) |
| val criterion | masked CE on 20 val tracks, **val windows = R1's (4 units)**, 16 windows/eval, every 500 steps |
| selection | best_val.pt solely by val loss; test never evaluated during training |
| data | identical frozen manifest `results/splits/slakh2100_subset_r1.json` |

Budget / accounting notes (honest record):

- A 12,000-step attempt of the same protocol was launched first and killed at
  step ~3,600. The kill was based on a **mislabeled trainer timer** (the log's
  "s/step" printed the time for the whole `log_every` interval, 50 steps, not
  one step): the run was in fact healthy at ~0.3 s/step (checkpoint save
  cadence: 1,000 steps ≈ 285 s) with val still improving (1.4091 @3500). That
  attempt hit **3 NaN-loss steps (3140, 3141, 3147)** — caught by the R1-era
  NaN guard: step skipped, no optimizer update, `nan_checkpoint.pt` dumped;
  noise_p at those steps = 0.15 (past the anneal horizon), grad-norm n/a
  (aborted before clipping; last successful logged gnorm ≈ 0.96). Its log is
  preserved at `results/slakh_r2_carry/train_12000_abandoned.log` and its
  checkpoints were discarded (different schedule; not a protocol artifact).
- The official R2 artifact is the 5,000-step protocol below: **0 NaN steps,
  5,000/5,000 optimizer updates completed** (NaN-skipped steps are excluded
  from update counts by construction — each abort `continue`s before
  `backward`/`opt.step()`), best val **1.3994 @ step 4000**, final val
  1.4010 @ 5000. Wall: ~25 min of stepping (true steady ~0.29–0.35 s/step)
  + ~8 min tokenization.
- Trainer fix shipped: the log now prints true per-step time
  (`dt/log_every`). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set
  by `scripts/run_r2_train.sh` as a harmless safety for variable-shape
  windows (the suspected allocator fragmentation was a misread of the same
  timer bug; no fragmentation was observed).

Val curve (R1-protocol criterion, clean 4-unit windows, 16/eval):

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | **4000** | 4500 | 5000 |
|---|---|---|---|---|---|---|---|---|---|---|
| val loss | 1.6947 | 1.5395 | 1.4474 | 1.4133 | 1.4069 | 1.4074 | 1.4025 | **1.3994** | 1.4008 | 1.4010 |

For orientation only (teacher-forced, never the decisive metric): R1 reached
1.3300 @6000 with 3× more optimizer steps at 4× shorter lifetimes; R2's val
per unit-visit is comparable (R2 sees 4× the tokens/step), and the plateau
shape differs (R1 overfit past its best by +0.10; R2 is flat within ±0.002).

Hard-budget context (2026-08-28): the account's remaining GPU budget forced
the round to "interpretable conclusion" priority. Training itself had already
completed (5,000/5,000 steps, 09:33:44–09:57:41), so the remaining budget went
to evaluation, not more training. Before opening the official test pool, the
frozen-protocol observation item was run: a **free-running continuous/reset
diagnostic on a pre-fixed validation-only subset** — the 6 tracks
`sorted(valid)[:6]` (Track00038…00224, fixed before any evaluation, not
result-selected), both arms, R1's decode protocol. **Secondary diagnostic
only**: it did not participate in checkpoint selection and triggered no
training change.

## 4. Val-only free-running diagnostic (observation item 1)

| pooled over 6 fixed val tracks (micro F1, macro in parens) | onset F1 | offset F1 | inst F1 | pred/GT | trunc | bnd err | inst switches |
|---|---|---|---|---|---|---|---|
| best_val (4000) continuous | 0.0323 (0.0399) | 0.0296 | 0.0323 | 0.70 | 2 | 0 | 28 |
| best_val (4000) reset | 0.0270 (0.0291) | 0.0209 | 0.0270 | 3.30 | 20 | 0 | 176 |
| final (5000) continuous | **0.0493** (0.0551) | 0.0453 | 0.0493 | 0.83 | 0 | 0 | 23 |
| final (5000) reset | 0.0267 (0.0313) | 0.0179 | 0.0267 | 3.11 | 29 | 0 | 119 |

Per-track (best_val): continuous > reset on 4/6 tracks (Track00050 +0.070,
Track00148 +0.027; Track00111 −0.022, Track00038 −0.013); instrument
switches are lower for continuous on **every** track (0–9 vs 17–44); zero
boundary errors everywhere.

Reading (two points, both central to the round):

1. **Teacher-forced val ↔ free-running misalignment is real and signed.**
   TF-val is flat-to-worse 4000→5000 (1.3994 → 1.4010), yet free-running
   continuous improved 0.0399 → 0.0551 macro onset F1 (+38 % rel.) with
   truncation 2 → 0 and flicker 28 → 23. TF-val cannot rank these
   checkpoints; free-running can — the auxiliary metric under-reports the
   axis this round actually cares about.
2. **The continuous/reset relation inverted vs R1.** On unseen tracks R2's
   continuous arm *beats* reset (0.0399 vs 0.0291 at 4000; 0.0551 vs 0.0313
   at 5000) while staying conservative in note count (0.70–0.83×) — in R1
   continuous was the collapsing arm (test pooled 0.010 vs reset 0.026,
   overproducing 2.5×), with 845 boundary errors; R2 has **zero** boundary
   errors in every arm/checkpoint of this diagnostic.

## 5. Official test — paired held-out evaluation (60 sealed tracks)

Protocol identical to R1 §3: same sealed 60-track official-test split
(byte-identical manifest), both arms continuous/reset, same greedy stepwise
decode with the Gate-2 carry protocol, `--compile`, `--max-tokens 4096`,
`data root slakh2100_16k_from_flac`. Deviation: **5 parallel shards instead
of R1's 3** — a wall-time-only change (per-track decode is independent and
deterministic; worker count cannot alter outputs). Checkpoint:
`best_val.pt` = step 4000 (selected by val loss only).

### 5.1 R2 best_val: continuous vs reset (60 tracks)

Pooled (n=60 sealed tracks, protocol identical to R1):

| metric | continuous | reset | paired Δ (cont − reset) | 95% CI | tracks Δ>0 |
|---|---|---|---|---|---|
| onset F1 | **0.0230** | 0.0227 | **+0.0003** | [−0.0040, +0.0046] | 28/60 |
| offset F1 | 0.0164 | 0.0151 | +0.0013 | [−0.0017, +0.0043] | 29/60 |
| instrument F1 | 0.0230 | 0.0227 | +0.0003 | [−0.0040, +0.0046] | 28/60 |
| pred/GT note ratio | **0.974** | 3.516 | −2.542 | [−2.842, −2.247] | 1/60 |
| instrument switches | **3.42** | 36.50 | −33.08 | [−36.17, −30.05] | 0/60 |
| truncated chunks (total) | **2** | 141 | — | — | — |
| boundary errors (total) | **0** | 0 | — | — | — |

Protocol verdict (per the R1 rule: mean Δ>0 AND CI excludes 0 → established):
onset F1 Δ = +0.0003 with CI crossing zero → **"positive trend, not
established"**. The headline is the sign flip plus the hygiene columns: in
R1 the continuous arm was the *broken* one (F1 0.010 vs reset 0.026, CI
significantly negative, 1,246 truncations, 845 boundary errors, pred ratio
2.51); in R2 the continuous arm is statistically **tied** with reset while
being the calm, protocol-clean arm (ratio 0.97 ≈ exact note budget, 2
truncations, 0 boundary errors).

### 5.2 Cross-round paired comparison: R1 best_val vs R2 best_val

Same 60 tracks, per-track paired deltas, 10k bootstrap, seed 0
(`results/slakh_r2_carry/eval/r1_vs_r2.json`):

| comparison | metric | R1 | R2 | Δ | 95% CI | tracks improved |
|---|---|---|---|---|---|---|
| **continuous** | onset F1 | 0.0100 | **0.0230** | **+0.0130** | **[+0.0076, +0.0187]** | **43/60** |
| continuous | offset F1 | 0.0063 | 0.0164 | +0.0101 | [+0.0055, +0.0148] | 40/60 |
| continuous | pred/GT ratio | 2.51 | 0.97 | −1.54 | [−2.01, −1.09] | 7/60 (lower better) |
| continuous | totals | trunc 1246, bnd 845, sw 510 | **trunc 2, bnd 0, sw 205** | — | — | — |
| reset | onset F1 | 0.0262 | 0.0227 | −0.0035 | [−0.0065, −0.0004] | 23/60 |
| reset | totals | trunc 441, bnd 0, sw 2463 | trunc 141, bnd 0, sw 2190 | — | — | — |

### 5.3 R2 latest (step 5000) — deliberately NOT evaluated on test

The free-running advantage of the final checkpoint (§4: 0.0493 vs 0.0323 on
the val diagnostic) is recorded as an observation. The sealed test pool is
evaluated **once per round, for the single protocol-selected checkpoint**
(best_val), as in R1 — evaluating a second checkpoint would spend the pool
twice and turn the test into a selection set. If a future round wants a
test-side comparison of checkpoints, it must pre-register that choice before
the pool is opened.

### 5.4 Verdict on the round's core question

**Yes at mechanism level, no at product level.** The question R2 was built
to answer — "does exposure/state-distribution matching alleviate the R1
continuous-mode collapse without touching the architecture?" —

- **Alleviated (significant, CI-positive):** the continuous arm improves
  onset F1 0.0100 → 0.0230 (Δ +0.0130, 95% CI [+0.0076, +0.0187], 43/60
  tracks), and its failure signature is eliminated: truncated chunks
  1246 → 2, boundary errors 845 → 0, note ratio 2.51 → 0.97, switches
  8.5 → 3.4. The sticky-attractor behavior REPORT_R1 §4 documented no
  longer occurs on the sealed pool.
- **Continuous ≥ reset now holds as a trend, not yet established:** the
  paired Δ moved from significantly negative (R1 −0.0162, CI
  [−0.0206, −0.0120]) to +0.0003 (CI [−0.0040, +0.0046]). Level 4 stays
  "positive trend, not established" — the arms are tied, not ordered.
- **Cost, where expected:** the reset arm (which never exercises carried
  state) is slightly worse (−0.0035, CI [−0.0065, −0.0004]); optimization
  pressure went to the carried-state regime. The val diagnostic (§4) shows
  the same assignment: continuous is the calm/accurate arm on val tracks.
- **Level 3 remains NOT PASS:** absolute onset F1 0.023 vs the ~0.24
  data-scale baseline reference. Exposure matching was the reviewer's
  explicitly-first lever; data scale is the deferred next one, and this
  round's result is consistent with capacity/data — not state handling —
  being the remaining bottleneck.

Confounds, stated: levers A+B were applied together (documented in §1;
ablation deferred); 5,000 steps vs R1's 12,000 optimizer steps at ~4×
tokens/step (≈1.7× R1's total token exposure); single seed; teacher-forced
val is near-identical between rounds' best checkpoints (1.3300 vs 1.3994 —
R2 slightly worse on the R1 criterion), so the free-running gain is not
explained by a teacher-forced val gain.

## 6. Budget ledger and protocol audit

- Budget context (recorded by the round driver): ≈4 h of GPU budget from
  09:58. Training consumed none of it (the 5,000-step run had finished at
  09:57:41; total run wall ≈ 24 min stepping ≈ 0.287 s/step + tokenization).
- Evaluation wall: val-diagnostic best_val 10:04–10:28 (6 fixed val tracks);
  val-diagnostic latest/final 10:25–10:38 (`final.pt` ≡ `latest.pt` verified
  bit-identical, 0/244 tensor diffs); official-test pool (best_val)
  10:11:47–12:21. A first pool attempt (5×12-track shards) lost all workers
  to an external process kill at ~10:40 (cause not directly observable; the
  parallel session states it was intentional, to get observation item 1 —
  the val-only diagnostic of BOTH candidate checkpoints — on record before
  the test pool opened; no OOM: cgroup `oom_kill 0`; 19/120 track-modes'
  log-only rows were lost — per-track JSONs are written at worker exit).
  Two relaunches of `scripts/run_r2_eval_pool.sh` followed (a shard-
  completion bookkeeping bug, then a malformed worker command — both fixed;
  validly completed shards were detected and skipped, not recomputed); the
  final driver wave ran 11:26–12:21 and all 30 shards verified valid. The
  two val diagnostics overlap the pool's wall-clock; neither influenced
  checkpoint selection (best_val was fixed by val loss at train time) and
  the latest checkpoint was never evaluated on test (§5.3). Test pool for
  `latest`: **not run**. No new training, no new experiments, no protocol
  changes: this round ends with a complete conclusion from a
  fully-evaluated single-checkpoint test pool.
- Checkpoint hygiene: `best_val.pt` (4000), `latest.pt`/`final.pt` (5000),
  `ckpt_00{1..5}000.pt` preserved in `results/slakh_r2_carry/` (gitignored);
  eval JSONs, paired reports, plots, listening MIDIs and logs committed.
- Boundary errors = chunk-prologue violations (chunk ended without the
  required `tie` token; open notes force-closed — `tokenizer.py` §Decoder).
  Truncation = chunk generation hit the 4096-token cap without EOS.
  Instrument flicker = adjacent-5s-chunk changes of the predicted program
  set (`inst_switches`).
- Val criterion untouched: teacher-forced, R1 windows (4 units), clean
  inputs — never corrupted, never carry-mode (train/val asymmetry is the
  design, matching R1's criterion for comparability).

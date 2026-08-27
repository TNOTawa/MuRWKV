# REPORT_R1 — Slakh2100 first generalization round (10h GPU session, 2026-08-28)

Setup: RTX 5090 32 GB, torch 2.8.0+cu128, BF16, official RWKV-7 clampw kernel,
container quota 16 cores / 92 GB (cgroup). Protocol constraints honored: no
scheduled sampling, no architecture changes, test split sealed, checkpoint
selection by validation loss only.

## 1. G5-v2 — mechanism review on the frozen G4 AMT model (corrected arms)

Design (unchanged, leak-free): G4 21.9M checkpoint fully frozen; track-level
split; probe TEST = the 4 BabySlakh tracks the G4 AMT never trained on;
Guitar/Piano stems verified against Slakh metadata (all 20 tracks);
bit-identical neutral chunks; official learned decay (no `w0` bias);
linear ridge probes on three feature sets with identical split/selection.

During the first official run a construction bug was found and fixed: the
audio frontend was sliced post-hoc, leaking 2 history mel frames through the
causal-conv context into the reset arm (reset scored 0.69 — impossible for a
bit-identical-neutral arm). The frontend now runs per arm.

| arm (test split, n=64) | h_last+S | h_last only | S only |
|---|---|---|---|
| continuous (20 s history) | 0.750 [0.632, 0.840] | **0.766** [0.649, 0.853] | 0.750 |
| lead1 (3 frames visible) | 0.562 [0.441, 0.677] | 0.562 | 0.562 |
| reset (neutral only) | **0.500** [0.381, 0.619] | 0.500 | 0.500 |
| history (no neutrals) | 0.781 [0.666, 0.865] | 0.703 | 0.781 |

State norm distance (continuous, test tracks): 12.8 (0 neutrals) → 2.6 (2) →
1.6 (4); classifier accuracy stays 0.75 after 2 neutral chunks.

**Verdict (rule 8 wording): PASS** — instrument-related acoustic information
is linearly recoverable across unseen tracks from the frozen AMT recurrent
representation (reset exact chance; lead1 ≈ chance bounds the short-horizon
channel; the readout h_last alone suffices). This does NOT by itself
demonstrate AMT continuity.

## 2. Slakh R1 — training

Manifest (frozen, seed 42, committed before training):
`results/splits/slakh2100_subset_r1.json` — train 120 / val 20 from corpus
train, **test 60 from the OFFICIAL corpus test split** (protocol correction
2026-08-28: the first draft mixed corpus-validation tracks into the test
pool; train/val byte-identical; Gate-1 re-validated: 0 truncation, 9,908
chunks, 6.54 M tokens). Chunk cap 4096 (corpus max chunk 2725).

| config | value |
|---|---|
| run | 12,000 steps, B=1, 4×5s windows, seed 42 |
| speed | calibration 1.3–1.5 s/step; steady 0.7–0.9 s/step |
| val criterion | masked CE on 20 val tracks (16 windows/eval, every 1000 steps) |
| selection | **best_val.pt = step 6000, val loss 1.3300** (sole criterion; test never evaluated during training) |
| final | step 12000: train loss 0.412 / acc 87.9 %; val 1.4306 |

Val curve: 1.593 (1k) → 1.330 (6k, best) → 1.431 (12k) — mild overfit tail
after 6k; final/latest checkpoints preserved alongside best_val.

## 3. Held-out paired evaluation (60 official-test tracks, sealed)

Both arms: same checkpoint (best_val), same inputs, same greedy decoding,
no teacher forcing; continuous = state/conv-carry/shift-lead never reset,
reset = all zeroed per 5 s unit. Stepwise decode was torch.compiled for both
arms identically (0.39 ms/token vs 3.65 ms eager; measured 2026-08-28).

Pooled (60 tracks; 3 shards merged):

| metric | continuous | reset | Δ (cont−reset) | paired CI95 | tracks Δ>0 |
|---|---|---|---|---|---|
| onset F1 | 0.0100 | 0.0262 | −0.0162 | [−0.0206, −0.0120] | 9/60 |
| onset+offset F1 | 0.0063 | 0.0175 | −0.0112 | [−0.0150, −0.0075] | 10/60 |
| instrument F1 | 0.0100 | 0.0262 | −0.0162 | [−0.0206, −0.0120] | 9/60 |
| instrument switches | 8.5 | 41.1 | −32.6 | [−36.4, −28.8] | 0/60 |
| pred/GT note ratio | 2.51 | 4.95 | −2.44 | [−2.87, −2.00] | 8/60 |
| RTF | 0.589 | 0.614 | −0.025 | [−0.095, +0.043] | 31/60 |
| truncated chunks (total) | 1246 | 441 | +805 | — | — |
| boundary errors (total) | 845 | 0 | +845 | — | — |

Duration quartiles (test tracks: 235.6 / 264.5 / 300.4 s):

| duration bin | n | Δ onset F1 | Δ switches | Δ note ratio |
|---|---|---|---|---|
| short | 15 | −0.0201 | −19.6 | −1.34 |
| mid | 30 | −0.0155 | −31.6 | −2.66 |
| long | 15 | −0.0138 | −47.4 | −3.08 |

The F1 delta is negative in every quartile (no continuity advantage with
length); the behavioral stability advantages of continuous (fewer instrument
switches, less overproduction) grow with track length but exist at
garbage-level transcription, and continuous additionally pays higher
truncation (1246 vs 441 chunks) and 845 boundary errors (reset: 0).

## 4. Verdicts (strict rules)

- **G5-v2: PASS (mechanism claim only)** — see §1; does not certify AMT
  continuity by itself.
- **Level 3: NOT PASS.** Held-out onset F1 0.010–0.026 with 2.5–5× note
  overproduction and 1,687 truncated chunks: no non-trivial Audio→MIDI
  generalization emerged from the 10→120 song expansion. (Expected per rule
  9; this says nothing about whether Pure RWKV suits AMT.)
- **Level 4: "no evidence" (NOT established).** Prerequisite Level 3 not met;
  the paired onset-F1 CI excludes 0 on the NEGATIVE side. The
  continuous-vs-reset difference on held-out tracks is a *behavioral*
  difference (stability vs truncation/boundary errors), not a transcription
  advantage. Train-track continuous≫reset does not count.

**Answer to the round's question (rule 9):** at 120 songs the model still
memorizes; no quantifiable held-out generalization, and recurrent continuity
does not yet show a transcription-level benefit on unseen songs. The next
levers (rule 3 ablations; larger corpus; EOS/truncation stability —
continuous-mode chunk truncation is an R1-specific failure mode worth a
scheduled-sampling ablation) are recorded for round 2.

## 5. Protocol audit

- Manifest corrected for test-only pool; train/val byte-identical to the
  original submission; Gate-1 re-run (0 truncation).
- Test sealed during training: the trainer has no test data path; the test
  set was first opened by the final eval.
- Checkpoint selection: best_val.pt only (val = 20 manifest tracks).
- Compiled decode is symmetric across arms (paired design preserved);
  torch.compile path verified on a real track (0 truncation, sane tokens).
- Nothing was rerun "because results were bad": the single re-run of eval was
  caused by the test-pool protocol fix + compile enablement.
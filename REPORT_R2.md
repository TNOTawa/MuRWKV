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

Budget note: a carry step costs ~3–5 s (measured with synced phase timers:
forward through the seeded-kernel passes 0.2–0.9 s; backward ~2.5–5 s, i.e.
~5× the forward — the kernel backward does more work per sequential step).
12,000 steps were attempted first and abandoned at step ~3,600 for budget
(val was still improving; 3 NaN-loss steps at 3140–3147 were caught by the
R1-era NaN guard — step skipped, no optimizer update, diagnostic dumped —
and val continued improving to 1.4091 @3500). The official R2 run is the
5,000-step protocol above.

(val curve + results filled in below after the run)

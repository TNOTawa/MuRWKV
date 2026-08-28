# NEXT_OPTIONS — ranked next-stage candidates after the R2 postmortem

**Status:** ranked options only; nothing here has been executed. All anchors
refer to `R2_POSTMORTEM.md` (§), `REPORT_R2.md`, and the frozen split
`results/slakh_r2_carry/split.json` (120 train / 20 val / 60 official-test,
sealed).

**Ranking principle:** expected movement on the dominant error mass
(§2: 81.3 % of pitched GT content never proposed; pitched recall 0.75 %)
per unit cost, with attribution options ranked by decision value. Data
facts used: the corpus on disk (`slakh2100_16k_from_flac`) holds
**1289 official-train / 270 validation / 151 test** tracks; the frozen
R1/R2 split uses 120 train + 20 val (the val tracks are drawn from the
same official train pool) + 60 official-test — leaving **≈1149 unused
train-side tracks** with the sealed pool untouched. R2's training wall
was ~24 min (5000 steps, 0.287 s/step); **GPU cost is dominated by
evaluation, not stepping.**

| rank | option | closes | anchor | GPU cost |
|---|---|---|---|---|
| N1 | **R3: data-scale round** (train 120 → 500–1000 tracks, longer schedule) | the "capacity/coverage/supervision regime" hypothesis (§9) | §6, §2d, §9 | hours (data prep + eval dominate) |
| N2 | **Free-running selection criterion** (pre-registered) | §4/§4.1 rank inversion — checkpoint choice is decoupled from the deployed mode | §4.1 | ~4–7 h (val-diag scaled to 20 tracks × 3–4 ckpts) |
| N3 | **G5-v2 state probe + state dumps on the R2 checkpoint** | §5 "U": does R2's state carry decodable identity; §3's unmeasurable state dynamics | §5, §3 | ~1 h |
| N4 | **A/B ablation** (A-only vs B-only short runs) | §5 "H": what actually removed the collapse | §5, REPORT_R2 §5.4 | ~1–2 h (training ≈ 24 min/arm) |
| N5 | **Bottleneck-targeted supervision levers** (factorial arms inside R3) | §2f/§2g/§2h/§7.1 error structure (tie/EOS, register, rare groups) | §7.1 | folded into N1 |

---

## N1 — R3: data-scale round (primary)

- **Change one thing:** train-side data (120 → 500–1000 tracks drawn only
  from the ≈1149 unused official-train tracks; val 20 and sealed test 60
  remain frozen), same R2 protocol (levers A+B, carry-seg 2048, noise
  0.15), schedule scaled to the data (≈20–30 k steps; R2's ≈6.7 M seen
  tokens was ~2 epochs — R3 should keep repeats ≈ 2 and grow unique
  tokens instead).
- **Why first:** §6 rules out complexity-distribution shift (KS all ≫ 0.05)
  and leaves volume as the cause *consistent with every measured signature*
  (81 % content absent, 4 groups carry all matches, diffuse per-position
  token error §7.1); MuScriptor reaches ~0.24 on the *same* synthetic domain
  at scale. It is the only lever whose reference effect size is ≈ 8×.
- **Decision rule (pre-register):** if pitched-instrument recall does not
  at least ~2× at ≥ 4× unique-token exposure, volume alone is excluded as
  the binding constraint and N5/capacity levers take priority.
- **Risks:** shallow group coverage does not automatically improve with
  more tracks (guitar is in only 53 % of the current train tracks — sample
  the expansion stratified by group coverage); eval wall grows with the
  pool (keep 60 test tracks; they are already the wall).

## N2 — Free-running selection criterion (must land before or with N1)

- §4.1 shows no teacher-forced family (fresh, carried, noisy) recovers the
  free-running ranking; §4 shows the official criterion picked the *worse*
  free-running checkpoint (4000 over 5000). R3 at 20–30 k steps makes this
  selection problem larger, not smaller.
- **Pre-registered change:** official selector = free-running val diagnostic
  on **all 20 val tracks** (not 6), both arms, at 3–4 scheduled checkpoints;
  selection metric = macro onset F1 (continuous), tie-breakers: truncation
  → boundary errors → zero-match rate. TF-val is demoted to a recorded
  secondary diagnostic.
- **Cost estimate:** the 6-track × 2-ckpt val-diag took ~24 min (REPORT_R2
  §6); 20 tracks × 3–4 ckpts ≈ 4–7 GPU-h per round — the price of a sane
  selector.
- **Optional research item (not the selector):** a free-running *NLL*
  surrogate (score the model's own sampled rollouts under its own next-token
  distribution) — cheap, but unproven; do not adopt without a retrospective
  check against §4's recorded checkpoints (4000/5000 must order correctly).

## N3 — G5-v2 probe + state dumps on the R2 checkpoint (do this first in calendar order)

- G5-v2 was only ever run on the frozen **G4** model (§5); the R2
  checkpoint's state has never been probed. One GPU pass of
  `memory_probe_v2.py` with the R2 checkpoint (instrument-identity linear
  readout from carried history vs reset) converts §5's biggest Unknown into
  evidence either way.
- In the same session, dump state norms / state distances during one
  val-diag pass — §3's persistence analysis is error-level *only* because
  no state dumps exist for R2.
- ~1 GPU-h, no training, no protocol change; result decides whether N4's
  ablation is even worth running (if R2's state carries nothing decodable,
  "state benefit" attribution loses its mechanism target).

## N4 — A/B ablation (A-only, B-only)

- Two short runs (R2 protocol minus one lever each, 5000 steps, ≈ 24 min
  stepping each), evaluated on **val** with the paired protocol — the
  collapse signature (trunc / boundary / degenerate-chunk rate / zero-match
  runs, §3) is measurable on val without spending the sealed pool. Only a
  *pre-registered* surprise earns a test claim (REPORT_R2 §5.3 discipline).
- Converts §5's three H rows into E or refuted before R3 scales an
  unattributed recipe.

## N5 — Bottleneck-targeted supervision levers (arms inside R3, not a separate round)

Each targets a measured second-order-but-sharp signature; test as
pre-registered factorial arms of R3, one decision each:

- **tie/EOS supervision** — §2f: 0.9 % of predicted pitched notes > 0.5 s
  (GT 26 %), 34 % ≤ 50 ms; §2h: ~247-token emission template. Up-weight
  velocity-0/tie tokens or add a duration-aware loss term.
- **register/pitch re-weighting** — §2g: output squeezed low (pred mean
  pitch 44–49 vs GT 54–59); §7.1: shift+pitch = 62.6 % of TF error mass.
  Register-balanced loss or an auxiliary pitch-class head.
- **rare-group exposure** — §2d: voice/organ/synth/flute emitted 0 times;
  program distribution is a habit. Program-token balancing or
  group-conditioned sampling in training.
- **Expectation cap:** §7.1/§7.2 say these move F1 only multiplicatively
  with per-position accuracy — they are complements to N1, not substitutes.

---

## Explicitly rejected for now (with the anchor that rejects them)

- **Decode/grammar repairs** — §7.2: one token error costs ≈ 1 note, no
  chain reaction; half of pitch/shift errors damage nothing.
- **Boundary hygiene** — §2g: interior vs boundary recall 3.6 % vs 4.0 %;
  nothing left to fix.
- **More/louder history noise (dose-response of A)** — §7.1: accuracy drops
  under the training noise; §4.1: noisy-history loss is +0.23 worse at every
  step; no evidence of an unused robustness lever here.
- **Bigger model without more data** — §7.1's diffuse per-position error and
  §6's volume evidence point at supervision/data first; capacity is tested
  by R3's outcome, not by adding parameters on 120 tracks.
- **Re-opening the sealed test pool** for checkpoint comparisons — REPORT_R2
  §5.3: one pre-registered evaluation per round.

## Metric hygiene decision to pre-register with the next round

The float boundary effect (`provenance_check.csv`: 14 % of matched pairs sit
exactly at |Δonset| = 5 ticks and are silently dropped) should be resolved
*before* R3 reports: either adopt integer-tick tolerance officially
(micro F1 0.0264 → 0.0377 for R2-continuous; uniform across arms/rounds, no
verdict changes) or keep the float path and document the rule. Carrying the
ambiguity into a third round would multiply the places that need a footnote.

## Suggested sequencing

1. **N3** (hours) — freezes the state-probe fact base before anything scales.
2. **N2** (paper change + one val-diag calibration pass) and **N4**
   (≈ 1–2 h) — can overlap; both are attribution/protocol hygiene.
3. **N1** with **N5** arms pre-registered — the round that actually attacks
   the 81 % content-absent mass.

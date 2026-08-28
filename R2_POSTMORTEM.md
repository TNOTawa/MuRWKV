# R2_POSTMORTEM — why the collapse is gone but held-out F1 is still ~0.03

**Mode:** offline, CPU-only post-analysis of existing artifacts. No GPU was
used; no training, inference, or test-pool data was regenerated; no R1/R2
original result file was modified. All new outputs live in
`results/r2_postmortem/` (+ this file, `NEXT_OPTIONS.md`, and the scripts in
`scripts/r2_postmortem/`).

**Question:** R2 eliminated R1's free-running collapse (onset F1 0.0100→0.0230
on the sealed pool, truncation 1246→2, boundary errors 845→0, note ratio
2.51→0.97) yet remains far below the ~0.24 data-scale baseline reference.
**Where did the remaining ~97 % of errors come from, mechanically?**

---

## 0. What artifacts this analysis is built on (and their limits)

| input | used for | limit |
|---|---|---|
| `results/{slakh_r1,slakh_r2_carry}/eval/test/*.json` | all cross-round comparisons (official numbers, untouched) | per-chunk token counts only; no note lists |
| `artifacts/listening/<track>/murwkv_{continuous,reset}.mid` | **all note-level taxonomy** | these are R2 `best_val.pt` (step 4000) outputs of the official pool, provenance-verified for all 60×2 rows (metrics match the official JSONs; `provenance_check.csv`). **R1's listening MIDIs were overwritten by R2's eval runs, so no R1 note-level taxonomy exists** — R1 enters only through official per-track aggregates |
| `results/slakh_r2_valdiag/` | step-4000 vs step-5000 free-running ranking (6 fixed val tracks) | n=6 tracks, secondary diagnostic by protocol |
| `results/slakh_r{1,2}_carry/metrics.{csv,json}`, `train.log` | training/val trajectories | val criterion sampled 16 windows/eval |
| `results/slakh_r2_carry/split.json` + data disk GT | data audit (200 tracks re-tokenized on CPU) | — |
| R2 checkpoints on CPU (float32) | teacher-forced token probe + carry-mode val proxy (post-hoc diagnostic only) | CPU float32 differs slightly from GPU bf16; used only for ranking questions |

**Metric caveat found on day one (documented, not hidden):** canonical GT and
decoder output both live on the 10 ms grid. 14 % of matched pairs sit at
exactly |Δonset| = 5 ticks (50 ms), where the official float subtraction
(`|0.49-0.44| = 0.050000…04 > 0.05`) silently drops the pair. Re-matching with
integer ticks (same tolerance semantics) raises R2-continuous matched pairs
26 282 → 36 317 (micro F1 0.0264 → 0.0377; `provenance_check.csv`). All three
aggregations of the same reality:

| R2 continuous onset F1 | value | definition |
|---|---|---|
| 0.0230 | macro (mean of 60 per-track F1) — the number quoted in REPORT_R2 | official |
| 0.0264 | micro (pooled counts; = `continuous_agg.json`) | official |
| 0.0377 | micro, tick-exact tolerance boundary (this round) | recomputed |

Everything below uses tick-exact matching for *taxonomy* (exact integer
arithmetic) and official numbers for *round-level claims*. No conclusion in
REPORT_R1/R2 changes; the boundary effect is uniform across arms and rounds.
It does mean the honest headline is "F1 ≈ 0.02–0.04", not one number.

---

## 1. R1 → R2 improvement, decomposed

From official per-track rows only (`improvement_per_track.csv`,
`improvement_groups.csv`, `improvement_pooled.json`).

**1a. Precision vs recall (Shapley attribution on the pooled micro P/R):**

| | R1 continuous | R2 continuous | Shapley share of ΔF1 = +0.0167 |
|---|---|---|---|
| precision | 0.0071 | 0.0275 (×3.9) | **+0.0128 (76 %)** |
| recall | 0.0157 | 0.0255 (×1.6) | **+0.0039 (24 %)** |

The gain is a **precision story**: R2 stopped emitting 2.2× the GT note budget
and came down to 0.93×. It is *not* a "hears more notes" story — recall rose
only 1.6× and is still 0.026.

**1b. What remains after the note count is fixed.** Section 2 shows the
remaining error mass: of 354 242 GT notes, 288 053 (81 %) have *no* predicted
note with the same (program, pitch) within ±0.5 s — the content is simply
never transcribed, not merely shifted.

**1c. Truncation 1246→2 and boundary errors 845→0 vs the F1 gain.** Grouping
tracks by R1 collapse severity (mean ΔF1 per group):

| group (R1 continuous truncations) | n | mean R1 F1 | mean R2 F1 | mean ΔF1 |
|---|---|---|---|---|
| 0 (never collapsed) | 15 | 0.0181 | 0.0247 | +0.0066 |
| 1–10 | 8 | 0.0218 | 0.0302 | +0.0085 |
| >10 (collapsed) | 37 | 0.0042 | 0.0208 | **+0.0167** |

Two readings, both kept: (a) collapsed tracks improved most, so removing
truncation/boundary failures *is* the dominant part of the gain; (b) the
15 never-collapsed tracks still gained +0.0066 (and overproducer tracks
[R1 ratio > 1.5] gained +0.0118 vs +0.0151 for the rest), so a genuine
conservatism/quality effect exists beyond de-catastrophizing. The de-garbling
itself contributes little *directly* to F1: note-level taxonomy shows cap-
truncation destroyed relatively few GT-note opportunities (truncated chunks
were 1246/−-scale events; the 81 % "no candidate" misses are spread over all
chunks). I.e. **de-garbling raised the ceiling from ~0.01 to ~0.03; it did not
move the ceiling to 0.24.**

**1d. Instrument flicker: more stable, but "more correct" is unmeasurable by
the official metric.** Two facts:

* instrument switches 8.5→3.4 per track (continuous); but in **all 120
  official rows**, `n_inst_match == n_matched` — the matcher penalizes program
  mismatch by 10³×tolerance, so *every* matched pair has the right program and
  instrument F1 is mathematically identical to onset F1. The official metric
  cannot say whether the calmer instrument stream is more *accurate*.
* On the note level, program-swaps are rare among predictions: 1252/329 140
  (0.4 %) of continuous predictions sit at a GT onset+pitch but under another
  program group. When the model emits a concrete note event it almost always
  uses *a* plausible program; its problem is not choosing the wrong label —
  it is emitting the wrong (pitch, time) content entirely (Section 2).
* Spearman across tracks: −Δswitches vs ΔF1 ρ = 0.13 (negligible). 46/60
  tracks reduced flicker; 32 of those also improved F1, 14 got worse.
  **Stability and correctness are largely decorrelated in R2.**

**1e. Who improved, by features (tercile means, `improvement_groups.csv`):**

* improved most: high GT note-count (+0.0242), high density (+0.0212), short
  median GT duration (+0.0227), long tracks (+0.0188), high drum share
  (+0.0176). I.e. **dense, percussive, long tracks** — precisely the tracks
  R1's garbling hurt most.
* degraded: 17/60 tracks (max −0.031). No feature cleanly predicts degradation
  at this sample size; degradations concentrate in tracks where R1's reset-like
  overproduction happened to hit more GT notes (median-duration-mid tercile is
  the only group with negative median Δ).

---

## 2. Error taxonomy — where the remaining errors live

Tick-exact matching, R2 continuous arm, 60 tracks (354 242 GT / 329 140 pred
notes). `error_taxonomy.csv`, `taxonomy_pooled.csv`, `strata_errors.csv`,
`taxonomy_bottleneck.png`.

**2a. GT notes (recall losses), one primary cause each:**

| why is a GT note missed | notes | % of GT |
|---|---|---|
| **no plausible prediction anywhere near (content absent)** | **288 053** | **81.3 %** |
| right (program,pitch) predicted within ±0.5 s (timing error) | 30 843 | 8.7 % |
| same key somewhere else in the track | 18 786 | 5.3 % |
| octave neighbour at same onset | 2 086 | 0.6 % |
| right onset+pitch, wrong program | 1 602 | 0.5 % |

**2b. Predicted notes (precision losses):**

| what kind of wrong | notes | % of pred |
|---|---|---|
| right key, wrong time | 150 168 | 45.6 % |
| wrong pitch near some GT onset | 157 487 | 47.8 % |
| octave | 953 | 0.3 % |
| program swap | 1 252 | 0.4 % |
| nothing GT-like within 0.25 s ("hallucination") | 6 408 | 1.9 % |

R2's predictions are *not* noise scattered in silence: 93 % sit near real GT
activity. They are a **dense, roughly on-grid, wrong-pitch/percussive texture**
laid over the music.

**2c. Drums vs pitched — the single sharpest split:**

| | GT notes | matched | recall | precision |
|---|---|---|---|---|
| drums | 113 273 | 11 074 | **9.8 %** | 8.7 % |
| pitched | 240 969 | 1 798 | **0.75 %** | 0.89 % |

Drum transcription works at a completely different (still low) level. Pitched
transcription — the actual AMT task — is essentially unsolved at this scale.

**2d. By instrument group (continuous; `strata_errors.csv` program rows):**

| group | GT notes | recall | pred notes | precision |
|---|---|---|---|---|
| drums | 113 273 | 0.098 | 127 775 | 0.087 |
| acoustic_piano | 42 326 | 0.023 | 80 727 | 0.012 |
| electric_bass | 20 698 | 0.017 | 35 912 | 0.010 |
| clean_electric_guitar | 36 646 | 0.010 | 15 178 | 0.025 |
| electric_piano | 26 013 | 0.002 | 9 991 | 0.005 |
| acoustic_guitar | 44 307 | **0.0005** | 46 300 | 0.0005 |
| voice / organ / synth_pad / flutes / synth_lead / … | ~30 k | **0.000** | **0** | — |

Four programs (drums, piano, bass, one guitar group) carry effectively all
matched notes; voice, organ, synths, flutes are **never emitted at all**; the
model predicts 46 k "acoustic_guitar" notes of which 22 matched. The output
program distribution is a habit, not a reading of the mix.

**2e. Onset timing of the pairs that do match:** mean |Δ| = 2.4 ticks (24 ms),
median 2; only 20 % land on the exact tick, and the histogram is flat to the
tolerance edge (`onset_deltas.csv`, `timing_duration.png`). Even "correct"
notes are quantization-coarse.

**2f. Duration/offset — the tie protocol is barely used:** GT pitched notes
have 26 % longer than 0.5 s; predictions **0.9 %**; 34 % of predicted pitched
notes are ≤ 50 ms staccato ticks. Of the few matched pitched notes, only 23 %
have an offset within ±0.1 s (67 % are too short). The model opens notes and
re-triggers/closes them early; sustained-note structure (the `tie` prologue and
velocity-0 off events) is essentially absent from free-running output.

**2g. Strata (`strata_errors.csv`):**

* register: recall 9.7 % for pitches 0–47 vs 0.5–0.9 % for 48+ — output is
  squeezed into the low register (mean pred pitch 44–49 vs GT 54–59 in every
  audited track).
* GT duration: short notes recalled 7.2 %, notes ≥ 1 s **0.4 %**.
* density: sparse passages (2–5 notes/s) 1.0 % vs dense (≥ 20/s) 3.6 %.
* polyphony: flat 3.4–3.8 % across 1→≥7 concurrent notes (not the axis).
* chunk boundary vs interior: 4.0 % vs 3.6 % — **no meaningful boundary
  penalty remains** (R2's protocol fix did its job; R1's 845 boundary errors
  are gone and the residual boundary/interior gradient is small).

**2h. Token-budget template (free-running emission habit).** R2 continuous
emits a nearly constant number of tokens per 5 s chunk regardless of content:
median pred/GT chunk-token ratio 0.49, and e.g. Track01876 emits exactly 247
tokens in 28 of 49 chunks and 243 in 11 more (R1 overproduced 4.1×). The
model regulates output density with a **stereotyped emission template**, not
with a reading of the audio.

**Answer to "the bottleneck has moved where?":** from generation collapse to
**pitched-instrument content selection**: which pitch, at which exact tick,
on which instrument, for how long. Timing-quantization, boundary, and
count-budget problems are now second-order (≈ 9 % timing-near misses +
0.4 % program swaps + small boundary gradient) compared to the 81 % of GT
content that is never proposed at all in the right key.

---

## 3. Free-running error propagation (no re-inference)

Definitions and outputs: `propagation_summary.json`, `hazard_curves.csv`,
`propagation_runs.png`. Note-level series come from the listening MIDIs
(R2 only); token-level degeneracy (chunk ≥ 95 % of the 4096 cap, or ≤ 20
tokens) is measurable for **both rounds and arms** from official rows.
"Zero-match chunk" = ≥ 20 GT notes, 0 matched.

| arm | rate | P(bad\|bad) | P(bad\|good) | persistence ratio | perm. null p95 | mean bad run (chunks) |
|---|---|---|---|---|---|---|
| R1 continuous (token-degenerate) | **46.0 %** | 0.932 | 0.058 | **16.2** | 1.07 | — |
| R1 reset (token-degenerate) | 20.0 % | 0.307 | 0.173 | 1.8 | 1.14 | — |
| R2 continuous (token-degenerate) | **6.9 %** | 0.700 | 0.022 | 32.3 | 1.63 | — |
| R2 reset (token-degenerate) | 12.1 % | 0.332 | 0.092 | 3.6 | 1.26 | — |
| R2 continuous (zero-match) | 43.1 % | 0.752 | 0.183 | 4.1 | 1.08 | **6.1 (≈ 30 s)** |
| R2 reset (zero-match) | 23.7 % | 0.560 | 0.138 | 4.1 | 1.13 | 2.1 (≈ 11 s) |

* **All arms show significant clustering** (permutation p < 0.001 in every
  arm; null ratios ≈ 1.0): hard passages cluster in the *audio*, so even the
  reset arm has persistence. The state-attributable part is the *difference
  between arms*.
* **R1's sticky attractor, quantified:** 46 % of chunks degenerate and P(next
  degenerate | degenerate) = 0.93 — once collapsed, the continuous arm stayed
  collapsed (runs of dozens of chunks; 1246 truncations).
* **R2 is recoverable, not clean:** catastrophic degeneracy dropped 46 % →
  6.9 %, but zero-match episodes still average **6.1 chunks (≈ 30 s) in
  continuous vs 2.1 (≈ 11 s) in reset**, and the hazard stays high
  (P(bad at k+8 | 8 bad) = 0.88). Self-recovery exists (max runs ≈ 12–13
  chunks) but is slow.
* **not measurable from current artifacts:** true state-level dynamics (state
  norms, attractor geometry) for the R2 checkpoint — G5-v2-style probes were
  run on the frozen *G4* model, not on R2; no state dumps exist for the eval
  runs. Everything above is *error-level* persistence, which is the observable
  proxy, not a state measurement.

---

## 4. Teacher-forced val vs free-running misalignment (post-hoc; official R2 result untouched)

From `metrics.json` / `metrics.csv` / `slakh_r2_valdiag/` (+ CPU proxy below):

* Train loss kept improving to the end (window means: 1.081 @3500–4000 →
  1.042 @4500–5000; train acc 65.9 %→67.0 %) while R1-protocol TF-val was
  flat (1.4025 → 1.3994 → 1.4010). The model was still learning something;
  the TF-val criterion stopped seeing it.
* **Rank inversion (already documented, now quantified):** TF-val ranks
  4000 (1.3994) < 5000 (1.4010); free-running val-diag ranks 5000 ≫ 4000
  (macro 0.0551 vs 0.0399 continuous; truncation 0 vs 2). Per-track val-diag
  shows why the TF signal is useless locally: the free-running gain 4000→5000
  is dominated by two tracks flipping from dead to alive (Track00111
  0.001→0.058) while TF-val moved −0.0016. **Correlation between TF-val and
  free-running is not estimable from 2 checkpoints; only the sign of the
  inversion is established.** n=6 val tracks also makes the free-running
  ranking itself noisy (Track00038: 0.025→0.010 went the other way).
* `val_eos_acc` on the tiny 16-window sample swings 0.16–0.42 with no trend —
  not a usable signal at this sample size.
* **CPU-computable proxies (this round, `ckpt_proxy.csv`):** recomputed
  carry-mode (lever-B protocol) and noisy-history (lever-A protocol) val loss
  on the 6 fixed val tracks for steps 3000/4000/5000 — see §4.1 for the
  outcome; this is the candidate *fully-offline* selection proxy family.

### 4.1 CPU carry/noisy val proxy (result)

Fully offline post-hoc recomputation (`ckpt_proxy.py` → `ckpt_proxy.csv`,
`ckpt_proxy.json`; bf16 CPU, 2 GB cgroup budget): the lever-B carry criterion
(`forward_gpt_carry`, seg 2048, detached carry + shift lead) and the lever-A
noisy-history variant (p = 0.15), both over the 6 fixed val-diag tracks ×
2 units (12 units, 4 113 loss positions), at steps 3000/4000/5000:

| variant | 3000 | 4000 | 5000 | ordering | matches free-running? |
|---|---|---|---|---|---|
| carry_clean | 1.38469 | **1.38427** | 1.38556 | 4000 ≺ 3000 ≺ 5000 | **no** — 5000 ranked worst |
| carry_noisy | 1.61168 | 1.62162 | 1.61845 | 3000 ≺ 5000 ≺ 4000 | **no** — 4000 ranked worst |
| fresh_clean | 1.38407 | 1.38410 | 1.38529 | 3000 ≺ 4000 ≺ 5000 | **no** — 5000 ranked worst |

Reference points: the official frozen criterion (GPU TF-val, R1 protocol)
ranks 4000 (1.3994) ahead of 5000 (1.4010); the free-running val-diag ranks
5000 ≫ 4000 (macro onset F1 0.0551 vs 0.0399, continuous).

* **No TF-family criterion recovers the free-running ranking.** Carried,
  noisy, and fresh variants all rank 5000 last or middle; the free-running
  gain 4000→5000 (+38 % relative macro F1) is invisible to every teacher-
  forced loss. Within this family there is **no offline selection
  candidate** for free-running quality; the (GPU) free-running val-diag is
  the only criterion that ranked 5000 correctly — and it is n = 6.
* **Sanity check passes for the TF family:** carry_clean on CPU reproduces
  the official criterion's direction (4000 ahead of 5000). Absolute levels
  differ (1.384 vs 1.399 GPU) as expected from the different window plan,
  dtype, and 6-track sample; only the ordering is comparable.
* **Carry vs fresh under clean history is a null effect:** Δ ≤ 6·10⁻⁴ at
  every step (4000: 1.38427 vs 1.38410). With GT history the carried state
  contributes ≈ nothing to next-token loss — the TF view is blind to what
  the state does in free-running (consistent with §4's decoupling and §5's
  "state buys stability, not TF-measured transcription").
* **The noise gap is the only large effect:** carry_noisy costs +0.23 loss
  (acc −6 pp) at every step — the model is not robust even to its own
  training-matched noise in loss terms (mirrors §7.1). Its internal
  ordering (3000 best, 4000 worst) agrees with neither other criterion;
  at Δ ≤ 0.01 over 12 units it is noise-dominated (no per-unit variance
  was recorded, so no CI is available without a re-run; the clean-variant
  ≤ 1.3·10⁻³ step differences should be read as ties).

**§4.1 verdict:** the rank inversion is not a sampling accident of one
criterion — it is a property of the TF-loss family. Any next round that
trains past a flat TF-val plateau must pre-register a free-running
selection signal (NEXT_OPTIONS, option N2); otherwise checkpoint choice
stays decoupled from the deployed inference mode.

---

## 5. Does R2 "rely on state", or merely tolerate noisy history? (A+B were applied together — no ablation exists)

| claim | status | evidence |
|---|---|---|
| **Evidence** | | |
| The *combination* (carry-trained 80 s windows + noisy history) removed R1's continuous-arm collapse on the sealed pool | **E** | official paired rows: trunc 1246→2, bnd 845→0, ratio 2.51→0.97, ΔF1 +0.0130 CI[+0.0076,+0.0187] (43/60 tracks) |
| R2's continuous arm behaves calmly *by construction of the training regime* (4× lifetime exposure + noisy input), while its reset arm lost a little | **E** | official: reset Δ −0.0035 CI[−0.0065,−0.0004]; flicker 36.5 vs 3.4; the val-diag arms order the same way on 6/6 tracks for switches |
| Carried state transmits *something* instrument-related across chunks in this model family | **E (R1's G5-v2, different checkpoint)** | G5-v2: linear readout of instrument identity 0.75–0.77 from 20 s history vs 0.50 reset on the frozen **G4** AMT model — mechanism exists in the architecture, but was **never run on the R2 checkpoint** |
| Continuous ≈ reset on transcription (tied, not ordered) | **E** | paired Δ +0.0003 CI[−0.0040,+0.0046] |
| **Hypothesis (plausible, not separated)** | | |
| Lever A (noisy-history training) is what made free-running robust to self-generated history | **H** | mechanistically matches the collapse signature; no A-only arm was trained |
| Lever B (state-carry training) is what made carried state non-poisonous (slow-burn rather than attractor) | **H** | zero-match runs still 3× longer in continuous than reset (30 s vs 11 s) — state still carries errors; but catastrophic lock-in is gone |
| The residual error state is *audio-driven* (hard passages cluster), state adds run-length on top | **H** | reset arms also cluster (ratios 1.8–4.1, perm p<0.001); continuous adds mean run 6.1 vs 2.1 chunks |
| **Unknown** | | |
| Whether persistent state helps *transcription* (vs only stability) on held-out music | **U** | arms tied on F1; n=60 CI includes 0 |
| Whether the R2 checkpoint's state carries decodable instrument/track identity (G5-v2 on R2) | **U** | probe not run on R2 (GPU round closed) |
| A/B attribution, dose-response of noise_p, sensitivity to carry-seg length | **U** | no ablations exist |
| Whether step 5000's free-running advantage replicates on test | **U** | test pool deliberately spent once (best_val only) |

Bottom line: R2 demonstrated that the *combination* fixes the collapse; it
**cannot** say persistent state itself is beneficial — that attribution
requires the ablation that was explicitly deferred.

---

## 6. Data scale / coverage audit (120/20/60, re-tokenized on CPU)

`data_audit_tracks.csv`, `group_coverage.csv`, `split_compare.csv`,
`train_test_ks.json`, `data_audit_*.png`.

* Corpus: median 5.5 k notes/track, 22–25 notes/s, mean polyphony ≈ 4.8,
  29–31 % drums, 8–9 instrument groups/track, ~30 k tokens/track.
  Train ≈ 3.6 M tokens total (R2 saw ≈ 6.7 M tokens over 5 k steps incl.
  repeats — the model saw the equivalent of ~2 epochs).
* **Train vs test distributions match on every complexity feature**
  (KS: n_notes p=0.88, density p=0.10, polyphony p=0.98, drum share p=0.64,
  median duration p=0.81, groups p=0.99). Only *track length* differs
  (KS p=0.0009; official corpus test tracks are longer — by design).
  → **complexity/distribution mismatch is ruled out** as the low-F1 cause.
* Group coverage is balanced but *shallow*: drums in 100 % of tracks;
  acoustic_guitar only in 53 % of train tracks; brass 23 %, synth_lead 16 %.
  Note shares agree between train and test to within ~2 % for all major
  groups. 4 rare singleton programs appear in test but not train (negligible
  note counts).
* Verdict on the four candidate causes:
  * data volume: **consistent with the evidence** (81 % of pitched GT content
    never proposed; 4 program groups carry all correct notes; 120 tracks /
    ~3.6 M tokens is far below every published AMT regime);
  * instrument coverage: **contributing** (rare groups never emitted — with
    0.1–1.5 % note shares they had ~0 training mass), but the *common* groups
    (guitar 13 % of notes!) also fail, so coverage alone is not the cause;
  * synthetic domain: no direct evidence either way this round (baseline
    MuScriptor reaches 0.24 on the *same* synthetic data — so domain is
    survivable at scale);
  * complexity distribution shift: **ruled out** (KS above).

---

## 7. Token-level vs note-level gap

Two CPU instruments: (a) a teacher-forced token probe of `best_val.pt`
(6 F1-stratified test tracks × units 1–4, exact training plan, clean and
noisy-history inputs; `token_probe.csv`); (b) a single-token corruption
cascade through the real decoder (`cascade.csv`).

### 7.1 Teacher-forced token accuracy

Clean-GT teacher forcing, 6 F1-stratified test tracks × units 1–4, 11 239
loss positions (`token_probe.csv`, `token_probe_summary.json`). Per-unit
fresh-state condition (no left context — a *harder* setting than training;
the composition below, not the absolute level, is the finding):

| target class | share of positions | token error rate | share of all errors |
|---|---|---|---|
| shift (TIME) | 30.4 % | 45.3 % | **33.4 %** |
| pitch | 27.5 % | 43.7 % | **29.2 %** |
| program | 11.3 % | 57.2 % | 15.8 % |
| velocity (on/off) | 26.1 % | 22.6 % | 14.3 % |
| drum | 4.3 % | 66.5 % | 6.9 % |
| tie / EOS | 0.4 % | ~43 % | 0.4 % |

* **No single failing token class — the error is diffuse:** overall token
  error 41.2 %, and **24/24 probed units diverge from GT immediately**
  (median first-error position = 0; the tie prologue's program token is the
  most common first mistake). Per-track token accuracy spans only 46–68 %
  — even the best track is wrong every third token.
* **shift + pitch dominate the error mass (62.6 %)** and their errors are
  in-class (wrong tick value, wrong pitch — 99 % of pitch-position argmaxes
  are still pitch tokens), exactly matching the note taxonomy's "wrong pitch
  near right onset" / 24 ms timing quantization.
* **Teacher-forced accuracy → note recall, quantitatively:** a pitched
  note-on event needs shift+program+velocity+pitch all right:
  P ≈ 0.55·0.43·0.77·0.56 ≈ 0.10; with its note-off pair ≈ 0.043 —
  bracketing the measured note recall (0.036 tick-exact / 0.023 official).
  **The note-level collapse is already present as per-position token
  uncertainty, not created by decoding.**
* **Noise robustness did not materialize at the token level:** overall
  accuracy drops 58.8 % → 54.2 % under the training noise (p = 0.15). The
  noisy-history lever bought free-running stability (§3, §5) without making
  the model better at predicting tokens through corrupted context.

### 7.2 One-token-error damage amplification

Single same-class token replacements decoded through the real MT3 decoder
(`cascade.csv`, n = 668):

| corrupted class | damage mean | p50 | p90 | zero-damage share | worst observed |
|---|---|---|---|---|---|
| program | 2.14 | 2 | 6 | 41 % | 10 notes (sticky program relabels the whole run) |
| drum | 2.00 | 2 | 2 | 0 % | 2 |
| velocity | 1.00 | 1 | 1 | 0 % | 1 |
| pitch | 0.83 | 0 | 2 | 58 % | 2 |
| shift | 0.81 | 0.5 | 2 | 50 % | 2 |

**No grammar chain reaction.** One token error costs ≈ 1 note (worst class:
program, which propagates through the sticky-program run). Half of all
pitch/shift errors damage nothing. The MT3 event grammar is locally
self-repairing; there is no "one bad token destroys the chunk" effect to fix.

### 7.3 Free-running side of the same story

* emission template: ~247-token chunks regardless of content (§2h) — EOS
  timing is a habit, not a decision;
* duration profile: 34 % of pitched notes ≤ 50 ms, 0.9 % > 0.5 s (§2f) —
  velocity-0 (note-off) decisions are where sustained music dies;
* zero-match chunks 43 % while predictions sit *near* GT 93 % of the time —
  the stream is "musically shaped" but key/pitch-selection is wrong at
  scale, which no local grammar error can explain.

**Q7 verdict: no minority token-class error dominates the failure.** The
"token→note gap" is arithmetic, not catastrophic: ~45 % in-class errors on
the two highest-mass classes (shift, pitch) compose multiplicatively into
the ~3 % note recall. Any fix that does not raise per-position pitch/time
accuracy (i.e., does not change what the model *represents* about the audio)
cannot move note F1 materially — grammar repair, decode tricks, or boundary
hygiene are already exhausted (§2g).

---

## 8. Structured audit of listening samples (fixed rules, no cherry-picking)

**Honesty note (also stamped in every CSV row): this is a scripted structural
audit of note arrays — no human listening happened in this offline round.**
Selection is mechanical from official per-track ΔF1 (R2−R1, continuous):
top-3 improved / top-3 degraded / 3 least-changed (disjoint)
(`track_case_audit.csv`).

| rule | tracks | ΔF1 |
|---|---|---|
| improved_top3 | Track01895, Track02047, Track02019 | +0.060, +0.060, +0.055 |
| degraded_top3 | Track01906, Track01876, Track01902 | −0.031, −0.026, −0.022 |
| changed_least3 | Track01952, Track02029, Track01907 | +0.000, −0.000, −0.001 |

Mechanical observations shared by all nine (details per track in the CSV):

* **漏声部 (missing voices):** every track's top GT groups include
  guitar/e-piano/organ-type parts that are absent or ≥10× under-represented
  in predictions; voice/synth groups are emitted zero times.
* **节奏错位 (rhythm):** coverage is high (86–99 % of chunks contain
  predictions; longest empty gap ≤ 25 s), and predictions sit near GT onsets
  — the *pulse* is roughly present; there is no gross tempo drift.
* **音高近似但音符结构错:** pitch-histogram overlap with GT is only
  0.05–0.31 and mean pitch drops by ~9 semitones in every track — output
  lives in a low-register, drum-heavy band; "right rhythm, wrong pitches"
  is the dominant texture.
* **program 合理性:** no wrong-instrument chaos — the model uses a narrow,
  self-consistent program set (drums/piano/bass/guitar-ish); the failure is
  absence of the other instruments, not absurd labeling.
* **长时间保持错误模式:** even the best improved track (Track01895) has a
  163-note repeated single-key run (73 % of its notes inside stuck runs);
  Track01902 shows 159-note runs with 24 % of notes stuck. Long error runs
  at note level coexist with the chunk-level 30 s zero-match runs of §3.

---

## 9. Final mechanism verdicts

| sentence | verdict | evidence |
|---|---|---|
| **R2 修复了 catastrophic free-running collapse** | **SUPPORTED** | trunc 1246→2, bnd 845→0, note ratio 2.51→0.93, degenerate-chunk rate 46 %→6.9 %, ΔF1 CI positive; zero-match runs now terminate (max ≈ 13 chunks) instead of locking in. Caveat: errors still persist ~30 s per episode in the continuous arm — recoverable, not clean. |
| **R2 提高了 held-out transcription** | **SUPPORTED** (continuous arm) | +0.0130 macro / +0.0166 micro ΔF1, CI excludes 0, 43/60 tracks improved; 5547→9036 matched notes. The reset arm got slightly worse (−0.0035) — the gain is specific to the carried regime. |
| **R2 证明 persistent state 本身有益** | **NOT SUPPORTED** | A+B applied jointly by design; no ablation; continuous vs reset on F1 is a statistical tie; G5-v2 state-probe was never run on the R2 checkpoint. §5 table separates the one proved combination-effect from three plausible-but-unproven attributions. |
| **R2 已接近实用 AMT** | **NOT SUPPORTED** | pitched-instrument recall 0.75 % (reset 2.4 %); 81 % of GT notes never proposed; vs ~0.24 baseline reference. What R2 achieved is *protocol health* (no truncation, no boundary errors, sane budgets), which is the precondition for, not the arrival at, usable AMT. |

**Mechanism summary in one paragraph.** R2's training-matched exposure turned
the continuous stream from a degenerating attractor (46 % degenerate chunks,
overproduction 2.2×, garbled boundaries) into a *conservative, punctual,
template-driven* emitter: it now paces itself (0.93× note budget, ~247-token
chunks), stays near real musical activity (93 % of predictions within 0.25 s
of a GT note), and almost never garbles chunk boundaries. But note *content
selection* — pitch, exact tick, instrument, duration — was never learned to
generalization: four program groups carry all correct notes, sustained notes
are absent, and 81 % of pitched GT content is simply never proposed. That is
a capacity/coverage/supervision regime problem, not a streaming-protocol
problem: at 120 tracks the model learned *how to talk* (grammar, budgets,
boundaries, drums) but not *what to say* (polyphonic pitched content).

---

## 10. Next-stage candidates (ranked, not executed)

Full detail, decision rules, and rejected options: **`NEXT_OPTIONS.md`**.
Ranked by expected movement on the dominant error mass per cost:

| rank | option | closes | anchor |
|---|---|---|---|
| N1 | **R3: data-scale round** — train 120 → 500–1000 of the ≈1149 unused official-train tracks, longer schedule, protocol frozen | the capacity/coverage/supervision-regime hypothesis | §6, §9 |
| N2 | **Free-running selection criterion**, pre-registered (20 val tracks, both arms, 3–4 ckpts) | §4.1: no TF-family criterion selects for free-running quality | §4, §4.1 |
| N3 | **G5-v2 probe + state dumps on the R2 checkpoint** (~1 GPU-h; do first in calendar order) | §5's biggest Unknown; §3's unmeasurable state dynamics | §3, §5 |
| N4 | **A/B ablation** (A-only / B-only, ~24 min stepping each, val-scored) | §5's H rows (what removed the collapse) | §5 |
| N5 | **Bottleneck-targeted supervision levers** (tie/EOS, register, rare groups) as factorial arms inside R3 | §2f/§2g/§2d error structure | §7.1 |

Rejected for now: decode/grammar repairs (§7.2), boundary hygiene (§2g),
more history-noise dose (§7.1, §4.1), bigger model on 120 tracks (§6),
re-opening the sealed pool (REPORT_R2 §5.3). One pre-registration is due
before R3: resolve the float/tick matcher ambiguity officially
(provenance_check.csv).

---

## Appendix: outputs produced this round

| file | content |
|---|---|
| `results/r2_postmortem/summary.json` | machine-readable rollup of everything above |
| `results/r2_postmortem/error_taxonomy.csv` | per (track, mode, category) note-level taxonomy |
| `results/r2_postmortem/track_case_audit.csv` | 9-track × 2-mode structured audit |
| `results/r2_postmortem/provenance_check.csv` | official-vs-recomputed metric deltas (MIDI round-trip + tick boundary) |
| `results/r2_postmortem/improvement_per_track.csv`, `improvement_groups.csv`, `improvement_pooled.json` | Q1 |
| `results/r2_postmortem/strata_errors.csv`, `duration_profile.csv`, `onset_deltas.csv`, `offset_deltas.csv`, `taxonomy_pooled.csv`, `chunk_level.csv` | Q2/Q7 |
| `results/r2_postmortem/propagation_summary.json`, `hazard_curves.csv` | Q3 |
| `results/r2_postmortem/data_audit_tracks.csv`, `group_coverage.csv`, `split_compare.csv`, `train_test_ks.json` | Q6 |
| `results/r2_postmortem/token_probe.csv`, `token_probe_summary.json`, `cascade.csv`, `cascade_summary.json`, `ckpt_proxy.csv`, `ckpt_proxy.json` | Q4/Q7 |
| `results/r2_postmortem/*.png` | figures named in the sections above |
| `scripts/r2_postmortem/*.py` | every number above regenerable from committed artifacts |

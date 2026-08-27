# AGENTS.md

> **This file is a map, not a manual.**
> Keep it short, stable, project-specific, and mechanically actionable.
> Put branch-specific knowledge behind pointers; put enforceable rules into tests, CI, linters, architecture checks, or tool permissions.

## Sources of truth

- The current task/spec describes intended behavior.
- Existing code and tests describe current behavior and compatibility obligations.
- Project docs describe architecture, data semantics, and rationale.
- Local/nested `AGENTS.md` files may add subtree-specific facts and commands.
- If these sources conflict, do not silently pick the easiest one. Follow an explicitly documented authority rule; otherwise surface the conflict as a decision.

## Default work protocol

1. **Investigate** — inspect the relevant implementation, tests, docs, and existing call sites before assuming behavior.
2. **Verify need** — establish whether a code change is actually required. A verified **no-change** result is a valid successful outcome.
3. **Resolve uncertainty** — look up facts yourself. Ask only about consequential decisions that remain unresolved after investigation.
4. **Change narrowly** — make the smallest coherent change; reuse existing abstractions and single sources of truth.
5. **Verify independently** — run focused checks while developing, then the project QA gate before completion.
6. **Report evidence** — state what changed, what was run, what passed, what could not be run, and any remaining risk.

## Engineering invariants

- Preserve user data, backward compatibility, and stable identities unless the task explicitly changes them.
- Confirm unfamiliar APIs, units, encodings, indexes, lifetimes, and boundary semantics from code/docs/tests or a minimal experiment.
- Keep dependency direction and architecture boundaries intact.
- Prefer deterministic behavior for generators, compilers, serializers, exporters, and reports when the domain allows it.
- Treat tests, golden files, quality thresholds, and CI rules as part of the verification system. Changes to them require task intent and explicit justification.
- Prefer rules that can be mechanically checked over rules that merely ask the Agent to “be careful”.
- An inability to run a required check is a reported limitation, not an implicit pass.

## Read on demand

- **Material product/architecture ambiguity remains, or the user asks to stress-test a plan / invokes a grilling workflow**  
  → read `docs/agents/DECISIONS.md`.
- **Adding/changing tests, QA, coverage, mutation testing, golden files, or deciding whether work is complete**  
  → read `docs/agents/QUALITY.md`.
- **Changing CI, permissions, sandboxes, architecture checks, agent infrastructure, or instruction layout**  
  → read `docs/agents/HARNESS.md`.
- **Need the evidence behind these rules**  
  → read `docs/agents/RESEARCH.md`.
- **Working inside a subtree with its own `AGENTS.md`**  
  → read that file and treat it as the local delta; do not duplicate root guidance into it.

## Project map

- **Project:** MuRWKV — pure RWKV-7 automatic music transcription (research, 10h run).
- **Purpose:** answer whether a random-init, attention-free RWKV-7 can learn Audio→MIDI
  from log-Mel directly and maintain musical state across 5s chunks without resets.
- **Main entry point:** `src/murwkv/training/train.py` (train), `src/murwkv/eval/eval_heldout.py` (evaluate),
  `src/murwkv/eval/infer.py` (streaming transcription), `src/murwkv/eval/memory_probe_v2.py`
  (Gate 5-v2 leak-free memory probe: frozen AMT ckpt + linear ridge, track-level split).
- **Architecture map:** `src/murwkv/model/rwkv7.py` (RWKV-7 core, official math/init;
  vendored official clampw CUDA kernel in `src/murwkv/cuda/`), `src/murwkv/model/murwkv_model.py`
  (unified audio+MIDI stream model), `src/murwkv/tokenizer.py` (MT3_FULL_PLUS, tie protocol),
  `src/murwkv/data/babyslakh.py` (dataset). See `REPORT_10H.md` (10h run, incl. the
  post-review Erratum: G5 scope, Level-4 claim, environment provenance) and
  `REPORT_R1.md` (Slakh2100 generalization round: G5-v2 official, R1 training,
  paired held-out verdicts — Level 3 not reached, Level 4 no evidence).
- **Build/setup:** `pip install -r requirements.txt`; no build step (CUDA kernel auto-compiles
  on first import; needs ninja).
- **Fast relevant tests:** `tests/test_tokenizer.py`, `tests/test_rwkv7_parity.py` (Gate 2),
  `tests/test_train_smoke.py`, `tests/test_gate1_data.py <babyslakh_root>` (Gate 1),
  `tests/test_probe_v2.py [babyslakh_root] [amt_ckpt]` (Gate 5-v2 design contract; CPU-safe).
- **Full QA command:** `python tests/qa.py` — runs every test in `tests/` that is runnable
  on the current machine (needs CUDA for parity/smoke; Gate1 needs extracted BabySlakh).
- **Release/build artifact:** listening MIDI artifacts under `artifacts/listening/<track>/`;
  checkpoints in `results/<exp>/` (+ private HF repo if writable).
- **Network:** for large remote fetches, `source /etc/network_turbo` (AutoDL
  academic proxy) first — measured ~35-48 MB/s vs ~2-4 MB/s direct
  (see docs/DATA.md).
- **Critical data/compatibility notes:** BabySlakh 16k (Zenodo 4603870, CC-BY-4.0) lives on
  the data disk (`/root/autodl-tmp/data`), never in git. No pretrained weights may be loaded
  for MuRWKV (random init only). Token truncation of target chunks is a pipeline bug
  (must be 0 in official runs). Continuous inference requires the lead-in protocol
  (conv carry 2 frames + shift lead 1 frame + state carry) verified in Gate 2.
  Derived `*.npz` caches (probe features, mel) are gitignored — regenerable via the
  probe/train commands; committed sha256 manifests document expected content
  (e.g. `results/gate5_probe_v2/feats/manifest.json`).

## Completion

Completion is based on **observable evidence**, not Agent confidence.

A normal completion report should be compact and factual, for example:

```text
Changed: ...
Focused checks: ...
Full QA: ...
Golden/determinism: ...
Not run / remaining risk: ...
```

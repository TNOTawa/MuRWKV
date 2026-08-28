# Third-party code / data provenance

Everything here is recorded with exact sources and licenses. MuRWKV uses NO
pretrained weights (all model parameters are randomly initialized).

## Reference repositories (read for specification; independent reimplementation)

| Repo | Commit | License | Used for |
|---|---|---|---|
| BlinkDL/RWKV-LM | `658042ca30222715c1d3ab662a3c556824dc6618` | Apache-2.0 | RWKV-7 "Goose" math, init, optimizer groups, LR schedule; official clampw CUDA kernel (`RWKV-v7/train_temp/cuda/rwkv7_clampw.{cu,cpp}`) vendored in `src/murwkv/cuda/` with attribution. |
| Jiayu-Xiong/AudioRWKV | `034146707e9cae0bfee2b9a347ff8bbfbb2f6f0e` | BSD-3-Clause | Reference only (audio frontend ideas; no code copied). |
| muscriptor/muscriptor | `e34b397bf0584e67bfd81dc591c390e6dcb03350` | MIT (LICENSE present at the pinned commit) | Reference for tokenizer/event semantics and log-Mel conventions; no code copied / independently reimplemented in `src/murwkv/tokenizer.py`. |

RWKV-7 CUDA kernel files vendored under `src/murwkv/cuda/` are copied verbatim
from RWKV-LM (Apache-2.0). Everything else in this repo is original code
written for MuRWKV (Apache-2.0, see `LICENSE`).

## Data

| Dataset | Source | DOI / URL | License | Checksum (MD5) |
|---|---|---|---|---|
| BabySlakh 16k | Zenodo record 4603870 | DOI 10.5281/zenodo.4603870 — `babyslakh_16k.tar.gz` | CC-BY-4.0 | `311096dc2bde7d61c97e930edbfc7f78` |

BabySlakh is the first 20 tracks of Slakh2100 (itself CC-BY-4.0, synthesized
from the Lakh MIDI corpus with public-domain soundfonts). Slakh2100-16k (via
`choihy/slakh2100_yourmt3_16k` on HuggingFace) is considered for Stage D1 only.

## Tokenizer provenance

MT3 vocabulary semantics: magenta/mt3 (Apache-2.0). MT3_FULL_PLUS group map
definition: muscriptor (MIT at the pinned commit `e34b397…` — independent
reimplementation). The Slakh MIDI program → MT3_FULL_PLUS group mapping is defined in
`src/murwkv/tokenizer.py` (`GROUP_PROGRAM_MAP`).
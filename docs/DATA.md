# Dataset inventory (data disk /root/autodl-tmp/data)

| Corpus | Path | Size | Contents | Use |
|---|---|---|---|---|
| BabySlakh 16k | `babyslakh/babyslakh_16k/` | 1.7 GB | 20 tracks: 16k mono mix + stems + `all_src.mid` + metadata.yaml | R0 gates (done) |
| Slakh2100 FLAC-redux | `slakh2100_flac_redux/data/{train,validation,test}/Track*` | 18.7 GB | original Slakh2100: `mix.flac` (44.1 kHz mono) + `all_src.mid` per track; 1289/270/151 tracks | full-corpus AMT training |
| Slakh2100-16k (resampled from FLAC-redux) | `slakh2100_16k_from_flac/{train,validation,test}/Track*/mix.wav` (+ `all_src.mid`) | 13.7 GB | 44.1k → 16 kHz mono via `resample_poly(160, 441)`; 1709 tracks, verified 16 kHz decode-clean | **preferred MuRWKV full-corpus training input** (original MIDI → MT3_FULL_PLUS incl. ties) |
| Slakh2100-16k (YourMT3+ preprocessed) | `slakh_yourmt3_16k/slakh2100_yourmt3_16k.tar` | 307 GB | 16 kHz mixes + YourMT3+ note metadata (NO original MIDI) | YourMT3-style training / source-sep; indexed by `yourmt3_indexes/` |
| yourmt3 indexes | `slakh_yourmt3_16k/yourmt3_indexes/` | 7 MB | per-split file lists (JSON dict) | corpus indexing |
| MuScriptor medium/small baseline | `muscriptor_medium/`, `muscriptor_small/` | 1.2 + 0.4 GB | gated HF checkpoints | baseline only (isolated venv) |

Corruption note: 39 tracks of the FLAC-redux mirror were truncated/corrupt on
first download; 38 recovered by re-download, `Track00846` is corrupt on the
mirror side and excluded (`results/splits/slakh2100_flac_excluded.json`).

## Notes for the next (GPU) round

1. **Use the resampled 16k corpus (`slakh2100_16k_from_flac`) as the MuRWKV
   training input**: original `all_src.mid` per track (our MT3_FULL_PLUS
   tokenizer incl. ties works unchanged) and 16 kHz mono `mix.wav` — the
   existing `BabySlakh(root, splits=True)` loader indexes it directly
   (train/validation/test = 1289/270/151; split lists in
   `results/splits/slakh2100_flac.json`, exclusion in
   `results/splits/slakh2100_flac_excluded.json`). The FLAC-redux archive is
   kept as the lossless 44.1 kHz source.
2. **The 307 GB tar is NOT extracted** (extraction would double disk usage to
   ~610 GB). When needed, extract per-track on demand:
   `tar -xf slakh2100_yourmt3_16k.tar -C slakh_preprocessed slakh2100_yourmt3_16k/train/TrackXXXXX`
   or extract the whole train split when the remaining disk allows.
3. The `_16k` (YourMT3+) corpus contains note metadata instead of MIDI —
   usable for a YourMT3-compatible training path, but MuRWKV's tie/EOS
   protocol is designed against the original MIDI (use corpus 1).
4. BabySlakh stays the debug corpus; gate tests reference it.
"""Extract an R1 eval subset matched to a given track list (budget round).

    python scripts/extract_r1_subset.py --tracks t1 t2 ... [--out results/slakh_r1_budget_subset]

Writes <out>/eval/test/{continuous,reset}.json containing only the given
tracks from results/slakh_r1/eval/test/. Pure offline; deterministic.
"""
from __future__ import annotations

import argparse
import json
import os

SRC = "results/slakh_r1/eval/test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="+", required=True)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default="results/slakh_r1_budget_subset")
    args = ap.parse_args()
    want = set(args.tracks)
    os.makedirs(os.path.join(args.out, "eval", "test"), exist_ok=True)
    for mode in ("continuous", "reset"):
        rows = json.load(open(os.path.join(args.src, f"{mode}.json")))
        sel = sorted((r for r in rows if r["track"] in want), key=lambda r: r["track"])
        got = {r["track"] for r in sel}
        assert got == want, f"missing in R1 {mode}: {sorted(want - got)}"
        with open(os.path.join(args.out, "eval", "test", f"{mode}.json"), "w") as f:
            json.dump(sel, f, indent=2)
        print(f"[subset] {mode}: {len(sel)} tracks -> {args.out}")


if __name__ == "__main__":
    main()

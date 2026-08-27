"""Merge per-shard eval JSONs into the canonical results/<exp>/eval/test/ dir.

    python scripts/merge_eval_shards.py --exp results/slakh_r1 --shards 3

Each shard exp is <exp>_eval<i> (split 'test'); this merges the per-track
rows of {continuous,reset}.json into <exp>/eval/test/ and writes the
per-mode aggregate files.
"""
from __future__ import annotations

import argparse
import json
import os

from murwkv.eval.metrics import TrackMetrics, aggregate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--shards", type=int, default=3)
    args = ap.parse_args()
    exp = args.exp.rstrip("/")
    out_dir = os.path.join(exp, "eval", "test")
    os.makedirs(out_dir, exist_ok=True)
    for mode in ("continuous", "reset"):
        rows = []
        for i in range(args.shards):
            p = os.path.join(f"{exp}_eval{i}", "eval", "test", f"{mode}.json")
            if not os.path.exists(p):
                print(f"SKIP missing {p}")
                continue
            rows += json.load(open(p))
        rows.sort(key=lambda r: r["track"])
        dual = [r for r in rows if r["track"] in {x["track"] for x in rows}]
        with open(os.path.join(out_dir, f"{mode}.json"), "w") as f:
            json.dump(rows, f, indent=2)
        ms = []
        for r in rows:
            m = TrackMetrics(track=r["track"])
            for k, v in r.items():
                if hasattr(m, k):
                    setattr(m, k, v)
            ms.append(m)
        agg = aggregate(ms)
        with open(os.path.join(out_dir, f"{mode}_agg.json"), "w") as f:
            json.dump(agg, f, indent=2)
        print(f"[merge] {mode}: {len(rows)} tracks")
        print(f"[merge] {mode} AGG:", {k: round(v, 4) if isinstance(v, float) else v for k, v in agg.items()})


if __name__ == "__main__":
    main()
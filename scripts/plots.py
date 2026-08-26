"""Generate report plots from real experiment data (PNG + underlying CSV).

    python scripts/plots.py results/<exp> [probe json ...]
"""
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def gk(d, k, default=None):
    return d.get(k, default)


def plot_training(exp_dir):
    mc = os.path.join(exp_dir, "metrics.csv")
    if not os.path.exists(mc):
        print("no metrics.csv", exp_dir)
        return
    with open(mc) as f:
        rows = list(csv.DictReader(f))
    steps = [int(r["step"]) for r in rows]
    loss = [float(r["loss"]) for r in rows]
    acc = [float(r["acc"]) * 100 for r in rows]
    gnorm = [float(r["gnorm"]) for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(steps, loss)
    ax[0, 0].set_title("train loss vs step")
    ax[0, 1].plot(steps, acc)
    ax[0, 1].set_title("token accuracy % vs step")
    ax[1, 0].plot(steps, gnorm)
    ax[1, 0].set_title("grad norm vs step")
    ax[1, 1].axis("off")
    out = os.path.join(exp_dir, "plots")
    os.makedirs(out, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "training.png"), dpi=110)
    plt.close(fig)
    print("wrote", os.path.join(out, "training.png"))


def plot_eval_cr(exp_dir):
    """continuous vs reset aggregate bars."""
    ev = os.path.join(exp_dir, "eval")
    if not os.path.isdir(ev):
        return
    data = {}
    for mode in ("continuous", "reset"):
        p = os.path.join(ev, "valid", f"{mode}_agg.json")
        if os.path.exists(p):
            data[mode] = json.load(open(p))
    if not data:
        return
    keys = ["onset_f1", "offset_f1", "inst_f1"]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
    for i, k in enumerate(keys):
        vals = [data.get(m, {}).get(k, 0) for m in ("continuous", "reset")]
        ax[i].bar(["continuous", "reset"], vals)
        ax[i].set_title(k)
        ax[i].set_ylim(0, 1)
    fig.tight_layout()
    out = os.path.join(exp_dir, "plots")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "continuous_vs_reset.png"), dpi=110)
    plt.close(fig)
    print("wrote", os.path.join(out, "continuous_vs_reset.png"))


def plot_probe(probe_dir):
    p = os.path.join(probe_dir, "probe_metrics.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    rows = d["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot([r["epoch"] for r in rows], [r["cont_acc"] for r in rows], label="continuous")
    ax[0].plot([r["epoch"] for r in rows], [r["reset_acc"] for r in rows], label="reset")
    ax[0].axhline(0.5, ls="--", c="gray", label="chance")
    ax[0].legend()
    ax[0].set_title("probe accuracy (continuous vs reset)")
    sp = os.path.join(probe_dir, "probe_state_distance.json")
    if os.path.exists(sp):
        sd = json.load(open(sp))
        ax[1].plot([r["n_neutral"] for r in sd], [r["dist"] for r in sd], marker="o")
        ax[1].set_title("inter-class state distance vs neutral chunks")
    fig.tight_layout()
    out = os.path.join(probe_dir, "plots")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "probe.png"), dpi=110)
    plt.close(fig)
    print("wrote", os.path.join(out, "probe.png"))


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if "probe" in arg or os.path.exists(os.path.join(arg, "probe_metrics.json")):
            plot_probe(arg)
        else:
            plot_training(arg)
            plot_eval_cr(arg)
    print("PLOTS DONE")
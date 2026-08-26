"""MuRWKV QA gate — runs every machine-runnable test in tests/.

    python tests/qa.py [--no-cuda] [--babyslakh-root PATH]

Exit code 0 iff all runnable tests pass; anything skipped (no GPU / no data)
is reported explicitly as a limitation, never as a pass.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TESTS = [
    "test_tokenizer.py",
    "test_rwkv7_parity.py",  # requires CUDA
    "test_train_smoke.py",  # requires CUDA
    "test_gate1_data.py",  # requires extracted BabySlakh
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--babyslakh-root", default="/root/autodl-tmp/data/babyslakh/babyslakh_16k")
    args = ap.parse_args()

    cuda_ok = not args.no_cuda
    if cuda_ok:
        try:
            import torch

            cuda_ok = torch.cuda.is_available()
        except Exception:
            cuda_ok = False
    data_ok = os.path.isdir(args.babyslakh_root) and any(
        d.startswith("Track") for d in os.listdir(args.babyslakh_root)
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    failures = []
    for t in TESTS:
        if t == "test_rwkv7_parity.py" and not cuda_ok:
            print(f"SKIP {t} (no CUDA)")
            continue
        if t == "test_train_smoke.py" and not cuda_ok:
            print(f"SKIP {t} (no CUDA)")
            continue
        if t == "test_gate1_data.py" and not data_ok:
            print(f"SKIP {t} (no BabySlakh at {args.babyslakh_root})")
            continue
        cmd = [sys.executable, os.path.join(HERE, t)]
        if t == "test_gate1_data.py":
            cmd.append(args.babyslakh_root)
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        print(r.stdout[-1200:], end="")
        if r.returncode != 0:
            print(r.stderr[-1200:], file=sys.stderr)
            failures.append(t)
        else:
            print(f"PASS {t}")
    if failures:
        print(f"QA FAILED: {failures}")
        sys.exit(1)
    print("QA PASS (skipped tests reported above are explicit limitations)")
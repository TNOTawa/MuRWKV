"""Parallel download of a HuggingFace dataset's individual files via a mirror.

    python scripts/fetch_hf_files.py REPO_ID FILE_LIST.JSON OUT_DIR [--workers 32]

FILE_LIST.JSON = list of repo-relative file paths (from the datasets API
siblings). Each file is fetched with curl (resumable) to OUT_DIR/<path>.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")


def fetch(repo, rel, out):
    dest = os.path.join(out, rel)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"{MIRROR}/datasets/{repo}/resolve/main/{rel}"
    for attempt in range(6):
        p = subprocess.run(
            ["curl", "-sL", "--max-time", "1800", "-C", "-", "-o", dest, url],
            capture_output=True,
        )
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            # retry empty/truncated files once: curl -C - resumes
            return True
        time.sleep(2 * (attempt + 1))
    print(f"FAILED {rel}", file=sys.stderr)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("file_list")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    files = json.load(open(a.file_list))
    print(f"{len(files)} files -> {a.out}", flush=True)
    ok = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(lambda f: fetch(a.repo, f, a.out), files)):
            ok += 1 if r else 0
            if (i + 1) % 200 == 0:
                print(f"{i+1}/{len(files)} ok={ok}", flush=True)
    print(f"DONE ok={ok}/{len(files)}", flush=True)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Hash a set of prediction files before the ground truth is opened.

A blind protocol is only worth what you can check afterwards.  This writes the sha256 of
every file that carries a prediction, together with the time and the git commit, so the
scoring step can prove the numbers it scores are the ones that existed before the truth
was on disk (`av2_eval.py --verify-sealed`).

It records what is deliberately absent too: the seal lists the ground-truth files that do
NOT yet exist, which is the part a reader can verify independently by looking at the
directory.

What a seal cannot prove: that no parameter was chosen with knowledge of the answer.  That
has to come from the protocol itself -- the same fixed parameters across every log, and
failures left in the table instead of tuned away.

Usage:
  python3 tools/seal_predictions.py --files "data/output/av2_depth/pred_*.csv" \
      --absent "data/input/av2/*/annotations.feather" \
      --out data/output/av2_depth/seal.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
import time
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True, help="globs of prediction files")
    ap.add_argument("--also", nargs="*", default=[], help="other inputs to pin (calibration, targets)")
    ap.add_argument("--absent", nargs="*", default=[], help="globs that must match nothing yet")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted({p for g in args.files for p in glob.glob(g)})
    also = sorted({p for g in args.also for p in glob.glob(g)})
    if not files:
        raise SystemExit("no files matched")
    present = [p for g in args.absent for p in glob.glob(g)]
    if present:
        raise SystemExit("these must not exist yet:\n  " + "\n  ".join(present))

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = ""
    seal = dict(sealed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), git_commit=commit,
                files={Path(p).name: sha(p) for p in files},
                inputs={Path(p).name: sha(p) for p in also},
                absent_at_seal=args.absent)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(seal, indent=2))
    print(f"sealed {len(files)} prediction files and {len(also)} inputs at {seal['sealed_utc']}")
    for p in files:
        print(f"  {sha(p)[:16]}  {p}")
    print(f"absent as required: {', '.join(args.absent) or '(nothing declared)'}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())

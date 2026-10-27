#!/usr/bin/env python3
"""comma2k19 Chunk_9, tenth and eleventh versions (docs/DEPTH_DASH_V11_PREREG.md sections 3 and 4): after both seals.

  python3 -B tools/v11/score/score_chunk9.py check --v10-seal S10 --v11-seal S11
      every prediction file named in the two seals is in its folder with the sealed sha256, all 210 segments are there, and
      both seal files are committed in a commit that is on origin/main (git fetch first). Reads no truth.
  python3 -B tools/v11/score/score_chunk9.py truth --v10-seal S10 --v11-seal S11          (only after Terry agrees in writing)
      check again; then the radar structure check (tools/c2k19_radar_check.py: processed radar = CAN LONG_DIST + 2.70 m on
      every segment, else stop: no truth); then tools/c2k19_extract.py truth for every segment -> data/output/c2k19_truth/.
  python3 -B tools/v11/score/score_chunk9.py score --v10-seal S10 --v11-seal S11 [--out DIR]
      check again; every truth file of the batch must be newer than both seals; then collect.py (batch C9: v10 = the sealed
      tenth-version folder, v11f = the sealed eleventh-version folder), f03_report.py and expectations.py into DIR.
"""
import argparse
import calendar
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = Path("/mnt/d/depth_work/chunk9/run_v11")
V10_DIR, V11_DIR = RUN / "v10", RUN / "v11"
RUN_LIST = REPO / "docs/DEPTH_DASH_V11_CHUNK9/run_list_final.txt"
ZIP = Path("/mnt/d/depth_work/chunk9/Chunk_9.zip")
TRUTH = REPO / "data/output/c2k19_truth"
RANGES = HERE / "ranges_cmbA.json"          # the development ranges file the parameters were made from (f03_report.py input)
PY = "/home/terry/venvs/depthbench/bin/python"
ENV = dict(os.environ, LD_PRELOAD=os.path.expanduser("~/miniconda3/lib/libstdc++.so.6"), OMP_NUM_THREADS="2",
           MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def segments():
    out = []
    for l in RUN_LIST.read_text().splitlines():
        if l.strip():
            d, r, s, ch, _ = l.strip().split("|")
            out.append((d, r, s, ch, f"{d}_{r}_{s}"))
    return out


def git(*args):
    return subprocess.run(["git", "-C", str(REPO)] + list(args), capture_output=True, text=True)


def check(a):
    """Seals against the files, and the seal commits on GitHub. Returns the two seal times (UTC epoch seconds)."""
    tags = [x[4] for x in segments()]
    if len(tags) != 210:
        raise SystemExit(f"run list holds {len(tags)} segments, not 210")
    times = []
    for seal_path, folder, sfx in ((a.v10_seal, V10_DIR, "_cars_v10.json"), (a.v11_seal, V11_DIR, "_cars_v11f.json")):
        seal = json.loads(Path(seal_path).read_text())
        files = seal["files"]
        for t in tags:
            if t + sfx not in files:
                raise SystemExit(f"{seal_path}: {t}{sfx} is not sealed")
        bad = [n for n, h in files.items() if not (folder / n).exists() or sha(folder / n) != h]
        if bad:
            raise SystemExit(f"{folder}: {len(bad)} file(s) missing or changed since the seal, e.g. {bad[:3]}")
        rel = Path(seal_path).resolve().relative_to(REPO)
        c = git("log", "--diff-filter=A", "--format=%H", "--", str(rel)).stdout.split()
        if not c:
            raise SystemExit(f"{rel} is not committed")
        if git("fetch", "-q", "origin", "main").returncode != 0:
            raise SystemExit("git fetch origin main failed (network?): cannot confirm the seal is on GitHub")
        if git("merge-base", "--is-ancestor", c[-1], "origin/main").returncode != 0:
            raise SystemExit(f"{rel}: its commit {c[-1][:8]} is not on origin/main yet (push first)")
        ts = calendar.timegm(time.strptime(seal["sealed_utc"], "%Y-%m-%dT%H:%M:%SZ"))
        times.append(ts)
        print(f"{rel}: {len(files)} files match, sealed {seal['sealed_utc']}, commit {c[-1][:8]} on origin/main")
    return times


def truth(a):
    check(a)
    link = REPO / "data/input/comma2k19/Chunk_9.zip"
    if not link.exists():
        link.symlink_to(ZIP)
        print(f"linked {link} -> {ZIP}")
    rc = RUN / "radar_check.json"
    subprocess.run([PY, "-B", str(REPO / "tools/c2k19_radar_check.py"), "--list", str(RUN_LIST), "--out", str(rc)],
                   cwd=str(REPO), env=ENV, check=True)
    if json.loads(rc.read_text())["failing"]:
        raise SystemExit("radar structure check failed: no truth made (write an addendum first)")
    made = 0
    for d, r, s, ch, t in segments():
        if not (TRUTH / t / "radar.csv").exists():
            subprocess.run([PY, "-B", str(REPO / "tools/c2k19_extract.py"), "truth", "--zip", str(ZIP), "--seg", f"{d}|{r}|{s}",
                            "--out", str(TRUTH)], cwd=str(REPO), env=ENV, check=True, stdout=subprocess.DEVNULL)
        made += (TRUTH / t / "radar.csv").exists() and (TRUTH / t / "ego.csv").exists()
    print(f"truth made {time.strftime('%F %T')}: {made} / 210")


def score(a):
    times = check(a)
    newest_seal = max(times)
    for *_, t in segments():
        for f in ("radar.csv", "ego.csv"):
            p = TRUTH / t / f
            if not p.exists():
                raise SystemExit(f"{p}: no truth")
            if p.stat().st_mtime <= newest_seal:
                raise SystemExit(f"{p} is older than a seal: not scored")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def run(*cmd):
        subprocess.run([PY, "-B"] + [str(c) for c in cmd], cwd=str(REPO), env=ENV, check=True)
    run(HERE / "collect.py", "--batch", "C9", "--out", out / "scores", f"v10={V10_DIR}", f"v11f={V11_DIR}")
    run(HERE / "f03_report.py", "--scores", out / "scores", "--ranges", RANGES, "--out", out / "report")
    run(HERE / "expectations.py", "--scores", out / "scores", "--report", out / "report", "--batch", "C9",
        "--out", out / "expectations.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "truth", "score"))
    ap.add_argument("--v10-seal", required=True)
    ap.add_argument("--v11-seal", required=True)
    ap.add_argument("--out", default="/mnt/d/depth_work/chunk9/score_v11")
    a = ap.parse_args()
    dict(check=check, truth=truth, score=score)[a.cmd](a)


if __name__ == "__main__":
    main()

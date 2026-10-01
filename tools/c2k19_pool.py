#!/usr/bin/env python3
"""Candidate pool of a comma2k19 chunk for a blind test, from the zip directory only (2026-10-01, third version).

Reads nothing but the names in the zip's central directory: no frame, no CAN, no pose, no radar. Every route of the
chunk is then searched for, by name, in the text files and the git history of this repository and of the earlier
shared repository; the report says where each name was found, so a route whose data was ever used (an output, an
evaluation, a truth file, a log of a run) can be dropped whole, as docs/DEPTH_DASH_V2_PREREG.md 3.1 did.

  python3 tools/c2k19_pool.py --zip data/input/comma2k19/Chunk_2.zip --report pool_report.json
"""
import argparse
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = Path.home() / "Vehicle-Distance-Forensics"
TEXT = {".json", ".csv", ".txt", ".md", ".py", ".sh", ".log", ".yaml", ".yml", ".tsv", ".html", ".ipynb"}
SKIP_DIRS = {".git", "cache", "frames", "__pycache__", "node_modules", "venv", ".venv"}


def segments(zpath):
    """dongle|route|segment from the zip directory (names like Chunk_2/<dongle>|<route>/<seg>/...)."""
    seen = set()
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            m = re.search(r"([0-9a-f]{16})\|(\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})/(\d+)/", n)
            if m:
                seen.add((m.group(1), m.group(2), int(m.group(3))))
    return sorted(seen)


def text_files(base):
    for dp, dns, fns in os.walk(base, followlinks=True):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for f in fns:
            p = Path(dp) / f
            if p.suffix.lower() in TEXT:
                try:
                    if p.stat().st_size < 200_000_000:
                        yield p
                except OSError:
                    pass


def git_text(repo):
    """Every message and patch in the repository's history, all branches (names only matter)."""
    r = subprocess.run(["git", "-C", str(repo), "log", "--all", "-p", "--format=%H %s"], capture_output=True, text=True,
                       errors="replace")
    return r.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    segs = segments(a.zip)
    routes = sorted({(d, r) for d, r, _ in segs})
    print(f"{a.zip}: {len(segs)} segments, {len(routes)} routes")
    names = {r for _, r in routes}
    hits = {r: [] for r in names}
    pat = re.compile("|".join(re.escape(r) for r in names))
    for base in (ROOT, OLD):
        for p in text_files(base):
            try:
                s = p.read_text(errors="replace")
            except OSError:
                continue
            for r in set(pat.findall(s)):
                hits[r].append(str(p))
        g = git_text(base)
        for r in set(pat.findall(g)):
            hits[r].append(f"git history of {base}")
    rep = dict(zip=str(a.zip), n_segments=len(segs), n_routes=len(routes),
               routes=[dict(dongle=d, route=r, segments=sorted(s for dd, rr, s in segs if (dd, rr) == (d, r)),
                            found_in=sorted(set(hits[r]))) for d, r in routes])
    Path(a.report).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    clean = [x for x in rep["routes"] if not x["found_in"]]
    print(f"routes never named anywhere: {len(clean)} ({sum(len(x['segments']) for x in clean)} segments)")
    for x in rep["routes"]:
        if x["found_in"]:
            print(f"  {x['dongle']}|{x['route']} ({len(x['segments'])} seg): {len(x['found_in'])} places, e.g. {x['found_in'][:3]}")


if __name__ == "__main__":
    main()

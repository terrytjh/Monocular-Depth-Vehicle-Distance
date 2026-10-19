#!/usr/bin/env python3
"""Tenth version (docs/DEPTH_DASH_V10_PREREG.md): write tags_{main,local_day,night}.txt with only the camera-k
fallback segments of a tools/v10_post.py output folder, from its v10_run.json (each segment's path "camera k day" /
"camera k night" and its group). Prediction files only: no truth, radar, CAN or pose file is read. Writes only the
three lists into DEST_DIR. The lists are written before the seal and sealed with the tenth version's files; the scorer
(tools/v10_score.py) then scores the camera-k segments apart with --batch DEST_DIR.

  python3 tools/v10_camk_tags.py OUT_V10_DIR DEST_DIR
"""
import json
import sys
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit(__doc__)
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
r = json.loads((src / "v10_run.json").read_text())
dst.mkdir(parents=True, exist_ok=True)
groups = {"main": [], "local_day": [], "night": []}
for tag, row in sorted(r["segments"].items()):
    if row.get("path") in ("camera k day", "camera k night"):
        groups[row["group"]].append(tag)
for g, tags in groups.items():
    (dst / f"tags_{g}.txt").write_text("".join(t + "\n" for t in tags))
print({g: len(t) for g, t in groups.items()})

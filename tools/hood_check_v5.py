#!/usr/bin/env python3
"""Post-hoc (truth open, not a registered analysis): how much of the gate-off own-lane error is the own car's hood
detected as a car. Hood box = bottom at >= 97% of the frame height and width > 80% of the frame width (same rule as
the Chunk_4 check in v6dev/gross). Main group, fifth and fourth versions, gate off, inside the reach.

Moved here on 2026-10-03 from data/output/dash_scale/blind_v5/hood_check.py (where it ran after the seal); only the
repository root changed, and a re-run writes byte-identical output.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import depth_dash_multicar as MC  # noqa: E402
assert Path(MC.__file__).resolve().parent == ROOT / "tools"
B = ROOT / "data/output/dash_scale/blind_v5"
TRUTH = ROOT / "data/output/c2k19_truth"
MC.PARAMS["radar_processing_m"] = 2.70
MC.PARAMS["sanity_gate"] = 1e9
W, H = 1164, 874
tags = [t.strip() for t in open(B / "tags_main.txt") if t.strip()]


def summ(e):
    e = np.asarray(e)
    return dict(n=int(len(e)), median_abs_pct=round(float(np.median(np.abs(e))), 2) if len(e) else None,
                bias_pct=round(float(np.median(e)), 2) if len(e) else None,
                gross_gt25_pct=round(100 * float(np.mean(np.abs(e) > 25)), 1) if len(e) else None)


res = {}
for lab, sfx in (("v5", "_cars_v5.json"), ("v4", "_cars_v4.json")):
    files = {t: B / f"{t}{sfx}" for t in tags if (B / f"{t}{sfx}").exists()}
    MC.check_sealed(str(B / "seal.json"), [str(f) for f in files.values()])
    keep, hood, hood_segs = {0: [], 1: []}, {0: [], 1: []}, {0: set(), 1: set()}
    for t, f in files.items():
        c = json.loads(f.read_text())
        if c.get("status") == "refused":
            continue
        box = {(s["t"], o["id"]): o["box"] for s in c["samples"] for o in s["objects"]}
        _, pairs, _ = MC.c2k19_score(c, TRUTH / t, return_rows=True)
        for p in pairs:
            if p["far"] or p["lane"] not in (-1, 0, 1):
                continue
            ln = 0 if p["lane"] == 0 else 1
            b = box.get((p["t"], p["id"]))
            is_hood = b is not None and b[3] >= 0.97 * H and (b[2] - b[0]) > 0.8 * W
            (hood if is_hood else keep)[ln].append(p["err"])
            if is_hood:
                hood_segs[ln].add(t)
    for ln, name in ((0, "own lane"), (1, "next lanes")):
        res[f"{lab}, {name}"] = dict(all=summ(keep[ln] + hood[ln]), hood_boxes=summ(hood[ln]), hood_segments=len(hood_segs[ln]),
                                     without_hood=summ(keep[ln]))
for k, v in res.items():
    print(k, json.dumps(v, ensure_ascii=False))
(B / "hood_check.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))

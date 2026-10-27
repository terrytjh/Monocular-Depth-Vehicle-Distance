# Copy of /mnt/d/depth_work/v11merge_20261024/score/s_lib.py for scoring comma2k19 Chunk_9 (docs/DEPTH_DASH_V11_PREREG.md); changed: imports v11common from this folder; frame counts through v11common.frames_dir (Chunk_9's decoded frames).
"""Scoring helpers for the merged eleventh-version build (v11m). READS TRUTH: development batches only, after every
prediction file of the run was written (v11m_post.py reads none).

Shared rules D1-D13 as re-implemented in /mnt/d/depth_work/decisions_20261018/g1/rescore_lib.py (S rules: every return
with d_cam > 1 m enters the matching, only in-scope ones -- |pose ego + v_rel| >= 7.2 km/h, |lat| <= 5.49 m -- are scored,
duplicates merged, vehicle-width gate 0.8-4.5 m, no distance gate; second car truth = radar - 0.70 m, centre column 615
(589 before 2018-06-25--22-19-02), first car - 0.33 m, 598), called exactly as the tenth version's development scorer
(/mnt/d/depth_work/v11dev_20261020/scripts/v11common.py score). The matching never uses the predicted distance, so a
reading without a number stays in the matching and only drops out of the distance statistics (as a blank dist_m).
Relative-speed truth = the frame's radar relative speed (rel_truth_inst; shared rules). Ego truth = mean pose speed of
the second's 20 frames; error = max(0, value) - truth (as tools/v10_score.py).
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v11common as C  # noqa: E402

L = C.L
REPO = C.REPO
M = Path("/mnt/d/depth_work/v11merge_20261024")
BASE = C.BASE
BINS = (("le30", 0.0, 30.0), ("30-60", 30.0, 60.0), ("60-100", 60.0, 100.0), ("gt100", 100.0, math.inf))
BIN_NAMES = [b[0] for b in BINS]
DEV = ("C6", "C7a", "C7b")
ALL = DEV + ("C8",)
HARD = 4.0
SENT = 1e9


def tbin(d):
    for name, lo, hi in BINS:
        if lo < d <= hi:
            return name
    return "le30"


def n_frames_of(tag):
    return len(list(C.frames_dir(tag).glob("*.jpg")))


def truth_counts(tag, n_frames):
    """In-scope truth per truth-distance bin on the 0.2 s grid (every 4th frame) of the segment."""
    cars = dict(fps=20.0, samples=[dict(fi=fi, t=fi / 20.0, objects=[]) for fi in range(0, n_frames, 4)])
    out, prev = {}, 0
    off, cx = C.cam_params(tag)
    for name, lo, hi in BINS:
        L.CAM_CX = cx
        n = L.score_seg(cars, tag, dict(L.S_DEFAULT, off=off), hi if math.isfinite(hi) else 1e12)[1]
        out[name] = n - prev
        prev = n
    return out


def pairs_of(tag, cars, number_rule):
    """Matched in-scope pairs (any truth distance) of one cars file. number_rule(d) -> True if the reading carries a
    number. Readings without one keep their place in the matching (sentinel), as a blank dist_m."""
    c = dict(fps=cars["fps"], samples=[])
    keep = {}
    for s in cars["samples"]:
        objs = []
        for o in s["objects"]:
            d = o.get("dist")
            has = d is not None and number_rule(d)
            keep[(s["fi"], o["id"])] = (o, has)
            objs.append(dict(id=o["id"], box=o["box"], far=bool(o.get("far")), dist=float(d) if has else SENT,
                             rel_kmh=o.get("rel_kmh"), win=o.get("win")))
        c["samples"].append(dict(fi=s["fi"], t=s["t"], objects=objs))
    off, cx = C.cam_params(tag)
    L.CAM_CX = cx
    pairs, _, _ = L.score_seg(c, tag, dict(L.S_DEFAULT, off=off), 1e12)
    out = []
    for p in pairs:
        o, has = keep[(p["fi"], p["id"])]
        out.append(dict(fi=p["fi"], id=p["id"], truth_id=p["truth_id"], truth=p["truth"], lane=p["lane"],
                        dist=o["dist"] if has else None, rel=o.get("rel_kmh"), rel_truth_inst=p.get("rel_truth_inst"),
                        obj=o))
    return out


def v10_number(d):
    return 0.0 < d <= 60.0


def v11_number(d):
    return d > 0.0


def ego_truth(tag):
    return C.ego_truth(tag)


def accel(et):
    return {k: (et[k + 1] - et[k - 1]) / 2.0 for k in et if (k - 1) in et and (k + 1) in et}


def sign_p(good, bad):
    n = good + bad
    if n == 0:
        return None
    return float(sum(math.comb(n, k) for k in range(good, n + 1)) / 2 ** n)


def wcsv(path, rows, cols):
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r.get(c) is None else r.get(c) for c in cols])

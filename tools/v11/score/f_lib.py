# Copy of /mnt/d/depth_work/v11final_20261024/score/f_lib.py for scoring comma2k19 Chunk_9 (docs/DEPTH_DASH_V11_PREREG.md); changed: imports s_lib from this folder.
"""Scoring helpers for the eleventh version's final build (v11f). READS TRUTH: development batches only (C6, C7a, C7b;
C8 reference), after every prediction file of the run was written (v11f_post.py reads none).

Shared rules D1-D13 as re-implemented in /mnt/d/depth_work/decisions_20261018/g1/rescore_lib.py (S rules), called exactly as
the merged build's scorer /mnt/d/depth_work/v11merge_20261024/score/s_lib.py (itself = the tenth version's development
scorer): second car truth = radar - 0.70 m, centre column 615 (589 before 2018-06-25--22-19-02), first car - 0.33 m, 598.
The matching never uses the predicted distance: a reading without a number keeps its place in the matching and only drops
out of the distance statistics. Relative-speed truth = the frame's radar relative speed (rel_truth_inst); absolute truth =
pose ego + radar relative speed of the frame (abs_truth_ms of the shared rules); ego truth = mean pose speed of the second.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s_lib as S  # noqa: E402

L, C = S.L, S.C
SENT = S.SENT
BINS, BIN_NAMES, tbin = S.BINS, S.BIN_NAMES, S.tbin
DEV, ALL = S.DEV, S.ALL
HARD = S.HARD


WIDTH = None          # (lo, hi) m: the vehicle-width gate of the matching; None = the shared rules' (0.8, 4.5)


def set_width_from_config(path):
    """Sensitivity runs: the matching's width gate from a (dev copy of the) shared config's match_width_m."""
    global WIDTH
    import json
    c = json.loads(Path(path).read_text())
    WIDTH = tuple(float(x) for x in c["match_width_m"])
    return WIDTH


def pairs_of(tag, cars, number_rule):
    """= s_lib.pairs_of, plus the absolute-speed truth of the pair (and the width gate of WIDTH when set)."""
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
    opt = dict(L.S_DEFAULT, off=off)
    if WIDTH is not None:
        opt["width"] = WIDTH
    pairs, _, _ = L.score_seg(c, tag, opt, 1e12)
    out = []
    for p in pairs:
        o, has = keep[(p["fi"], p["id"])]
        out.append(dict(fi=p["fi"], id=p["id"], truth_id=p["truth_id"], truth=p["truth"], lane=p["lane"],
                        dist=o["dist"] if has else None, rel=o.get("rel_kmh"), rel_truth_inst=p.get("rel_truth_inst"),
                        rel_truth_win=p.get("rel_truth"), abs_truth=3.6 * p["abs_truth_ms"], obj=o))
    return out

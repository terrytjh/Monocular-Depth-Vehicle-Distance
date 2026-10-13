#!/usr/bin/env python3
"""Seventh version scoring (docs/DEPTH_DASH_V7_PREREG.md section 3), comma2k19, corrected truth (radar - 0.33 m).

Run only after the seal: every prediction file it reads is first checked against the seal (sha256), before any truth
is read. The registered second to sixth versions and marking geometry are scored by score-batch as before; this adds
the seventh version and each of its steps, on the same segments:

  v6  the registered sixth version (*_cars_v6.json)
  S1  + hood boxes not measured (*_cars_v6h.json)
  S2  + the camera's k: check, and fill-in where the current method refused (*_cars_ck.json)
  S3  + the same-class ruler beyond the reach (*_cars_ckf.json)
  v7  + relative speeds of the two methods averaged = the seventh version (*_cars_v7.json)

Per road group (tags_main / tags_local_day / tags_night of the batch) and step: segments with an output; distances of
the own lane and the next lanes inside the reach, with the scorer's 50 % gate and without it; readings beyond the reach;
by true distance (0-10 ... 60-80 m) in % and in metres; relative / absolute / ego speed (score-batch's pooled numbers).
Paired, on the same readings: S1 vs v6 (no gate, own lane, per segment), S2's filled-in segments, S3 vs S2 (beyond the
reach: same sample, car and radar target), v7 vs S3 (relative speed: same sample, car and radar target); one-sided sign
tests over segments with >= 10 paired readings.

  python3 tools/v7_score.py --batch DIR --seal DIR/seal.json --out DIR/score_v7.json
  python3 tools/v7_score.py --batch DIR --dev-opened data/output/dash_scale/blind_v4 --out X.json    # opened data only
"""
import argparse
import contextlib
import io
import json
import sys
import tempfile
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import depth_dash_multicar as M  # noqa: E402

TRUTH = ROOT / "data/output/c2k19_truth"
STEPS = [("v6", "_cars_v6.json"), ("S1", "_cars_v6h.json"), ("S2", "_cars_ck.json"), ("S3", "_cars_ckf.json"),
         ("v7", "_cars_v7.json")]
GROUPS = ("main", "local_day", "night")
BINS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 60), (60, 80)]
MIN_PAIRED = 10


def tags_of(batch, g):
    p = batch / f"tags_{g}.txt"
    return [l.strip() for l in open(p) if l.strip() and not l.startswith("#")] if p.exists() else []


def summ(rows):
    if not rows:
        return dict(n=0)
    e = np.array([r["err"] for r in rows])
    m = np.array([abs(r["dist"] - r["truth"]) for r in rows])
    return dict(n=int(len(e)), median_abs_pct=round(float(np.median(np.abs(e))), 2), bias_pct=round(float(np.median(e)), 2),
                gt25_pct=round(float(100 * np.mean(np.abs(e) > 25)), 1), median_abs_m=round(float(np.median(m)), 2))


def load(batch, t, sfx):
    p = batch / f"{t}{sfx}"
    if not p.exists():
        return None
    c = json.loads(p.read_text())
    return None if c.get("status") == "refused" else c


def pairs_of(c, t, gate):
    M.PARAMS["sanity_gate"] = 0.5 if gate else 1e9
    _, pr, _ = M.c2k19_score(c, TRUTH / t, return_rows=True)
    M.PARAMS["sanity_gate"] = 0.5
    return [dict(p, seg=t) for p in pr if p["lane"] in (-1, 0, 1)]


def pooled_speeds(batch, tags, sfx, seal):
    lst = Path(tempfile.mkstemp(suffix=".txt")[1])
    lst.write_text("\n".join(tags) + "\n")
    tmp = Path(tempfile.mkstemp(suffix=".json")[1])
    with contextlib.redirect_stdout(io.StringIO()):
        M.cmd_score_batch(Namespace(list=str(lst), cars_dir=str(batch), suffix=sfx, refused_from="_run.json",
                                    truth_root=str(TRUTH), verify_sealed=seal or "", out=str(tmp), truth_offset="corrected"))
    p = json.loads(tmp.read_text())["pooled"]
    lst.unlink()
    tmp.unlink()
    return {k: {x: p[k].get(x) for x in ("n", "mae_kmh", "bias_kmh", "coverage_95", "guess0_mae_kmh") if x in p[k]}
            for k in ("relative_speed", "absolute_speed", "ego_per_bin")}


def sign_test(per_seg):
    """per_seg: [(error of the new, error of the old)] per segment; smaller is better."""
    better = sum(1 for a, b in per_seg if a < b)
    worse = sum(1 for a, b in per_seg if a > b)
    n = better + worse
    return dict(segments=len(per_seg), better=better, worse=worse, ties=len(per_seg) - n,
                p_one_sided=float(binomtest(better, n, 0.5, alternative="greater").pvalue) if n else None)


def key(p):
    return (p["seg"], round(p["t"], 3), p["id"], p["truth_id"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--seal", help="the batch's seal.json (blind segments)")
    ap.add_argument("--dev-opened", help="an opened batch (its seal.json lists these segments and their truth exists): no seal check")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    B = Path(a.batch)
    M.PARAMS["radar_processing_m"] = 2.70
    alltags = sorted({t for g in GROUPS for t in tags_of(B, g)})
    if a.dev_opened:
        sealed = json.loads((Path(a.dev_opened) / "seal.json").read_text())["files"]
        for t in alltags:
            if f"{t}_dash.json" not in sealed or not (TRUTH / t / "radar.csv").exists():
                raise SystemExit(f"{t}: not a sealed-and-opened segment of {a.dev_opened} -- refusing to score")
        M.C2K19_DEV.update(alltags)
        seal = None
    else:
        if not a.seal:
            raise SystemExit("--seal is required (or --dev-opened for opened data)")
        files = [str(B / f"{t}{sfx}") for t in alltags for _, sfx in STEPS if (B / f"{t}{sfx}").exists()]
        M.check_sealed(a.seal, files)                     # before any truth is read
        seal = a.seal
    res = {}
    for g in GROUPS:
        tags = tags_of(B, g)
        if not tags:
            continue
        r = res[g] = dict(segments=len(tags), steps={}, paired={})
        rows = {}                                         # step -> {"gate": [...], "nogate": [...]}, cars loaded
        have = {}
        for lab, sfx in STEPS:
            rows[lab] = dict(gate=[], nogate=[])
            have[lab] = set()
            for t in tags:
                c = load(B, t, sfx)
                if c is None:
                    continue
                have[lab].add(t)
                rows[lab]["gate"] += pairs_of(c, t, True)
                rows[lab]["nogate"] += pairs_of(c, t, False)
            gp, ng = rows[lab]["gate"], rows[lab]["nogate"]
            r["steps"][lab] = dict(
                output=len(have[lab]),
                gate=dict(own_lane=summ([p for p in gp if not p["far"] and p["lane"] == 0]),
                          next_lanes=summ([p for p in gp if not p["far"] and p["lane"] != 0]),
                          inside_reach=summ([p for p in gp if not p["far"]]),
                          beyond_reach=summ([p for p in gp if p["far"]])),
                nogate=dict(own_lane=summ([p for p in ng if not p["far"] and p["lane"] == 0]),
                            next_lanes=summ([p for p in ng if not p["far"] and p["lane"] != 0]),
                            beyond_reach=summ([p for p in ng if p["far"]])),
                by_true_distance_gate={f"{lo}-{hi} m": summ([p for p in gp if lo <= p["truth"] < hi]) for lo, hi in BINS},
                speeds=pooled_speeds(B, tags, sfx, seal))
        # S1 vs v6: no gate, own lane inside the reach, per segment
        per = []
        for t in have["v6"] & have["S1"]:
            e0 = [abs(p["err"]) for p in rows["v6"]["nogate"] if p["seg"] == t and not p["far"] and p["lane"] == 0]
            e1 = [abs(p["err"]) for p in rows["S1"]["nogate"] if p["seg"] == t and not p["far"] and p["lane"] == 0]
            if len(e0) >= MIN_PAIRED and len(e1) >= MIN_PAIRED:
                per.append((float(np.median(e1)), float(np.median(e0))))
        r["paired"]["S1_vs_v6_nogate_own_lane"] = sign_test(per)
        # S2's filled-in segments: refused by S1, an output from S2
        filled = sorted(have["S2"] - have["S1"])
        fr = [p for p in rows["S2"]["gate"] if p["seg"] in set(filled) and not p["far"]]
        r["paired"]["S2_filled_in"] = dict(segments=len(filled), inside_reach_gate=summ(fr),
                                           inside_reach_nogate=summ([p for p in rows["S2"]["nogate"]
                                                                     if p["seg"] in set(filled) and not p["far"]]))
        # S2 vs S1 on the segments both kept (the camera-k check can swap a segment's source)
        swapped = sorted(t for t in have["S1"] & have["S2"]
                         if json.loads((B / f"{t}_cars_ck.json").read_text()).get("next_candidate", {}).get("source") != "current")
        r["paired"]["S2_swapped_to_camera_k"] = dict(segments=len(swapped), names=swapped)
        # S3 vs S2: beyond the reach, the same sample, car and radar target
        for lab_new, lab_old, name, field in (("S3", "S2", "S3_vs_S2_beyond_reach", "dist"), ("v7", "S3", "v7_vs_S3_relative_speed", "rel")):
            old = {key(p): p for p in rows[lab_old]["gate"] if (p["far"] if field == "dist" else p.get("rel") is not None)}
            both = [(p, old[key(p)]) for p in rows[lab_new]["gate"]
                    if key(p) in old and (p["far"] if field == "dist" else p.get("rel") is not None)]
            if field == "dist":
                err = lambda p: abs(p["err"])
            else:
                err = lambda p: abs(p["rel"] - p["rel_truth"])
            seg = defaultdict(list)
            for pn, po in both:
                seg[pn["seg"]].append((err(pn), err(po)))
            per = [(float(np.median([x for x, _ in v])) if field == "dist" else float(np.mean([x for x, _ in v])),
                    float(np.median([y for _, y in v])) if field == "dist" else float(np.mean([y for _, y in v])))
                   for v in seg.values() if len(v) >= MIN_PAIRED]
            stat = (lambda xs: round(float(np.median(xs)), 2)) if field == "dist" else (lambda xs: round(float(np.mean(xs)), 3))
            r["paired"][name] = dict(readings=len(both), new=stat([err(pn) for pn, _ in both]) if both else None,
                                     old=stat([err(po) for _, po in both]) if both else None,
                                     statistic="median |error| %" if field == "dist" else "MAE km/h", **sign_test(per))
    Path(a.out).write_text(json.dumps(res, indent=1))
    for g, r in res.items():
        print(f"===== {g} ({r['segments']} segments)")
        for lab, x in r["steps"].items():
            f = lambda d: f"{d.get('median_abs_pct', '-'):>5}% {d.get('median_abs_m', '-'):>5} m ({d.get('n', 0):5d})"
            sp = x["speeds"]
            print(f"  {lab:3s} output {x['output']:3d}  own {f(x['gate']['own_lane'])}  next {f(x['gate']['next_lanes'])}"
                  f"  no gate own {f(x['nogate']['own_lane'])}  beyond {f(x['gate']['beyond_reach'])}"
                  f"  rel {sp['relative_speed'].get('mae_kmh')}  abs {sp['absolute_speed'].get('mae_kmh')}  ego {sp['ego_per_bin'].get('mae_kmh')}")
        for k, v in r["paired"].items():
            print(f"  {k}: " + ", ".join(f"{kk} {vv}" for kk, vv in v.items() if kk != "names"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare a changed method with the current one on an OPENED comma2k19 batch (default Chunk_4, sealed and scored on
2026-10-02; --batch data/output/dash_scale/blind_v3 for Chunk_2). A development comparison, not a blind test: every
segment is checked to be in the batch's seal.json with its truth made.

  python3 tools/improve_dev.py --base DIR --new DIR [--batch DIR] [--groups main,local_day,night] --out X.json

Per road group: segments with an output, distances inside the reach (own lane / next lanes) with the scorer's 50 %
sanity gate (as the registered scores) and without it (every reading a user would see), the readings beyond the reach,
relative / absolute / ego speed (fifth version), and the sixth version's far readings. Corrected truth (radar - 0.33 m).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import depth_dash_multicar as M  # noqa: E402

SRC = ROOT / "data/output/dash_scale/blind_v4"
TRUTH = ROOT / "data/output/c2k19_truth"


def opened_check(tags, src):
    seal = json.loads((src / "seal.json").read_text())["files"]
    for t in tags:
        if f"{t}_dash.json" not in seal or not (TRUTH / t / "radar.csv").exists():
            raise SystemExit(f"{t}: not a sealed-and-opened segment of {src.name} -- refusing to score")


def summ(e):
    e = np.asarray(e, float)
    if not len(e):
        return dict(n=0)
    return dict(n=int(len(e)), median_abs_pct=round(float(np.median(np.abs(e))), 2), bias_pct=round(float(np.median(e)), 2),
                gt25_pct=round(float(100 * np.mean(np.abs(e) > 25)), 1))


def mae(rows, key="err_kmh"):
    v = [abs(r[key]) for r in rows if r.get(key) is not None]
    return dict(n=len(v), mae_kmh=round(float(np.mean(v)), 2) if v else None)


def score_dir(d, tags, suffix):
    out = dict(segments=len(tags), output=0)
    for gate in ("gate", "nogate"):
        M.PARAMS["sanity_gate"] = 0.5 if gate == "gate" else 1e9
        own, nxt, far = [], [], []
        n_out = 0
        for t in tags:
            f = d / f"{t}{suffix}"
            if not f.exists():
                continue
            c = json.loads(f.read_text())
            if c.get("status") == "refused":
                continue
            n_out += 1
            _, pairs, _ = M.c2k19_score(c, TRUTH / t, return_rows=True)
            for p in pairs:
                if p["lane"] not in (-1, 0, 1):
                    continue
                (far if p["far"] else own if p["lane"] == 0 else nxt).append(p["err"])
        out["output"] = n_out
        out[gate] = dict(own_lane=summ(own), next_lanes=summ(nxt), beyond_reach=summ(far))
    M.PARAMS["sanity_gate"] = 0.5
    return out


def speeds(d, tags, suffix):
    """Pooled speeds as score-batch reports them (gate on)."""
    from argparse import Namespace
    import contextlib
    import io
    import tempfile
    lst = Path(tempfile.mkstemp(suffix=".txt")[1])
    lst.write_text("\n".join(tags) + "\n")
    tmp = Path(tempfile.mkstemp(suffix=".json")[1])
    with contextlib.redirect_stdout(io.StringIO()):
        M.cmd_score_batch(Namespace(list=str(lst), cars_dir=str(d), suffix=suffix, refused_from="_run.json",
                                    truth_root=str(TRUTH), verify_sealed="", out=str(tmp), truth_offset="corrected"))
    p = json.loads(tmp.read_text())["pooled"]
    lst.unlink(); tmp.unlink()
    return {k: {x: p[k].get(x) for x in ("n", "mae_kmh", "bias_kmh", "coverage_95")} for k in
            ("relative_speed", "absolute_speed", "ego_per_bin")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--batch", default=str(SRC), help="the opened batch: its seal.json and tags_*.txt")
    ap.add_argument("--groups", default="main,local_day,night")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    M.PARAMS["radar_processing_m"] = 2.70                       # corrected truth
    res = {}
    for g in a.groups.split(","):
        src = Path(a.batch)
        tags = [l.strip() for l in open(src / f"tags_{g}.txt") if l.strip() and not l.startswith("#")]
        opened_check(tags, src)
        M.C2K19_DEV.update(tags)                                  # opened_check stands in for the seal check of new files
        res[g] = {}
        for lab, d in (("base", Path(a.base)), ("new", Path(a.new))):
            res[g][lab] = dict(v5=score_dir(d, tags, "_cars_v5.json"), v6=score_dir(d, tags, "_cars_v6.json"),
                               speeds_v5=speeds(d, tags, "_cars_v5.json"))
    Path(a.out).write_text(json.dumps(res, indent=1))
    for g, r in res.items():
        print(f"===== {g}")
        for lab in ("base", "new"):
            v5, v6, sp = r[lab]["v5"], r[lab]["v6"], r[lab]["speeds_v5"]
            f = lambda x: f"{x.get('median_abs_pct', '-'):>5} ({x.get('n', 0):5d}, >25% {x.get('gt25_pct', '-'):>5})"
            print(f"  {lab:4s} output {v5['output']:3d}/{v5['segments']}  gate own {f(v5['gate']['own_lane'])} next {f(v5['gate']['next_lanes'])}"
                  f" | no gate own {f(v5['nogate']['own_lane'])} next {f(v5['nogate']['next_lanes'])}"
                  f" | v6 far {f(v6['gate']['beyond_reach'])}")
            print(f"       rel {sp['relative_speed']['mae_kmh']} ({sp['relative_speed']['n']})  abs {sp['absolute_speed']['mae_kmh']}"
                  f"  ego {sp['ego_per_bin']['mae_kmh']} ({sp['ego_per_bin']['n']})")


if __name__ == "__main__":
    main()

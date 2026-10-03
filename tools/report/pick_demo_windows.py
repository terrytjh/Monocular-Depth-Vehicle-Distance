#!/usr/bin/env python3
"""Showcase windows for the demo (post-hoc, on a scored blind batch): the 10 s windows where the distance, the other
cars' speed and the ego speed all agree best with the truth, with enough on screen. Not a result: a demo choice.

Per window (sliding by 1 s): distance pairs inside the reach (the scorer's 50 % gate), at least MIN_D, median |error|;
other cars' absolute speed, at least MIN_V, median |error|; ego speed seconds, at least MIN_E, mean |error|.
Score = dist% / 5 + speed km/h / 4 + ego km/h / 3 (each over a typical value); the best window per route.

  python3 tools/report/pick_demo_windows.py --batch data/output/dash_scale/blind_v5 --top 5
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import depth_dash_multicar as M  # noqa: E402

W, MIN_D, MIN_V, MIN_E = 10.0, 20, 10, 8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--suffix", default="_cars_v6.json")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    B = ROOT / a.batch
    M.PARAMS["radar_processing_m"] = 2.70                          # corrected truth
    tags = [t.strip() for t in open(B / "tags_main.txt") if t.strip()]
    sealed = {}
    for sf in ("seal.json", "seal_v6.json"):
        if (B / sf).exists():
            sealed.update(json.loads((B / sf).read_text())["files"])
    import hashlib
    best = {}
    for t in tags:
        f = B / f"{t}{a.suffix}"
        if not f.exists():
            continue
        if sealed.get(f.name) != hashlib.sha256(f.read_bytes()).hexdigest():
            raise SystemExit(f"{f.name}: not the sealed file")
        c = json.loads(f.read_text())
        if c.get("status") == "refused":
            continue
        _, pairs, ego = M.c2k19_score(c, ROOT / "data/output/c2k19_truth" / t, return_rows=True)
        d = [(p["t"], abs(p["err"])) for p in pairs if not p["far"] and abs(p["err"]) <= 100 * M.PARAMS["sanity_gate"]]
        v = [(p["t"], abs(p["abs"] - p["abs_truth"])) for p in pairs if p.get("abs") is not None and not p["far"]]
        e = [(r["t0"], abs(r["kmh"] - r["truth"])) for r in ego if r.get("kmh") is not None and r.get("truth") is not None]
        route = t.rsplit("_", 1)[0]
        for s0 in np.arange(0, 60 - W + 1, 1.0):
            win = lambda xs: [x for tt, x in xs if s0 <= tt < s0 + W]   # noqa: E731
            dd, vv, ee = win(d), win(v), win(e)
            if len(dd) < MIN_D or len(vv) < MIN_V or len(ee) < MIN_E:
                continue
            dm, vm, em = float(np.median(dd)), float(np.median(vv)), float(np.mean(ee))
            score = dm / 5 + vm / 4 + em / 3
            row = dict(tag=t, start=float(s0), score=round(score, 3), dist_median_pct=round(dm, 2), dist_n=len(dd),
                       car_speed_median_kmh=round(vm, 2), speed_n=len(vv), ego_mae_kmh=round(em, 2), ego_n=len(ee))
            if route not in best or score < best[route]["score"]:
                best[route] = row
    rows = sorted(best.values(), key=lambda r: r["score"])[:a.top]
    print("segment                                      start  dist%  (n)   car km/h (n)   ego km/h (n)")
    for r in rows:
        print(f"{r['tag']}  {r['start']:4.0f}s  {r['dist_median_pct']:5.2f} ({r['dist_n']:3d})  "
              f"{r['car_speed_median_kmh']:5.2f} ({r['speed_n']:3d})  {r['ego_mae_kmh']:5.2f} ({r['ego_n']:2d})")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

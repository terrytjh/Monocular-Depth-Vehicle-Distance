#!/usr/bin/env python3
"""Third version, registered extra scoring (docs/DEPTH_DASH_V3_PREREG.md section 4), main group only, after the seal:
  1. distance with the 50 % sanity gate switched off (the pessimistic bound), for v2 and marking geometry;
  2. v3 vs v2 paired: ego speed on the seconds where both have a reading (MAE each; per segment with >= 10 paired
     seconds which is smaller; one-sided sign test), and absolute speed on the windows both have.

Moved here on 2026-10-03 from data/output/dash_scale/blind_v3/score_extra.py (where it ran after the seal); only the
repository root changed, and a re-run writes byte-identical output.
"""
import json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import depth_dash_multicar as MC
assert Path(MC.__file__).resolve().parent == ROOT / "tools"
B = ROOT / "data/output/dash_scale/blind_v3"; TRUTH = ROOT / "data/output/c2k19_truth"
tags = [t.strip() for t in open(B / "tags_main.txt") if t.strip()]

def rows(sfx):
    files = {t: B / f"{t}{sfx}" for t in tags if (B / f"{t}{sfx}").exists()}
    MC.check_sealed(str(B / "seal.json"), [str(f) for f in files.values()])
    out = {}
    for t, f in files.items():
        c = json.loads(f.read_text())
        if c.get("status") == "refused":
            continue
        out[t] = MC.c2k19_score(c, TRUTH / t, return_rows=True)
    return out

res = {}
gate = MC.PARAMS["sanity_gate"]
MC.PARAMS["sanity_gate"] = 1e9
for lab, sfx in (("v2", "_cars.json"), ("c", "_cars_c.json")):
    pr = [p for _, ps, _ in rows(sfx).values() for p in ps if not p["far"]]
    d = {}
    for name, lanes in (("own lane", (0,)), ("next lanes", (-1, 1))):
        e = np.array([p["err"] for p in pr if p["lane"] in lanes])
        d[name] = dict(n=int(len(e)), median_abs_pct=round(float(np.median(np.abs(e))), 2) if len(e) else None,
                       bias_pct=round(float(np.median(e)), 2) if len(e) else None,
                       gross_gt25_pct=round(100 * float(np.mean(np.abs(e) > 25)), 1) if len(e) else None)
    res[f"gate off, {lab}"] = d
MC.PARAMS["sanity_gate"] = gate

r2, r3 = rows("_cars.json"), rows("_cars_v3.json")
both, segs, absd = [], [], []
for t in set(r2) & set(r3):
    e2 = {round(e["t0"], 2): e for e in r2[t][2] if e.get("kmh") is not None and e.get("truth") is not None}
    e3 = {round(e["t0"], 2): e for e in r3[t][2] if e.get("kmh") is not None and e.get("truth") is not None}
    k = sorted(set(e2) & set(e3))
    a2 = [abs(e2[x]["kmh"] - e2[x]["truth"]) for x in k]; a3 = [abs(e3[x]["kmh"] - e3[x]["truth"]) for x in k]
    both += list(zip(a2, a3))
    if len(k) >= 10:
        segs.append((np.mean(a2), np.mean(a3)))
    p2 = {(p["t"], p["id"]): p for p in r2[t][1] if p.get("abs") is not None}
    p3 = {(p["t"], p["id"]): p for p in r3[t][1] if p.get("abs") is not None}
    absd += [(abs(p2[q]["abs"] - p2[q]["abs_truth"]), abs(p3[q]["abs"] - p3[q]["abs_truth"])) for q in set(p2) & set(p3)]
better = sum(1 for a, b in segs if b < a); n = len(segs)
p = sum(math.comb(n, i) for i in range(better, n + 1)) / 2 ** n if n else None
b = np.array(both); ab = np.array(absd)
res["v3 vs v2, ego speed"] = dict(paired_seconds=len(b), mae_v2=round(float(b[:, 0].mean()), 2), mae_v3=round(float(b[:, 1].mean()), 2),
                                  segments=n, v3_better_segments=better, sign_test_one_sided_p=p)
res["v3 vs v2, absolute speed"] = dict(paired_windows=len(ab), mae_v2=round(float(ab[:, 0].mean()), 2),
                                       mae_v3=round(float(ab[:, 1].mean()), 2)) if len(ab) else {}
(B / "score_extra.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
print(json.dumps(res, indent=1, ensure_ascii=False))

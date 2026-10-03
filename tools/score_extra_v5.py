#!/usr/bin/env python3
"""Fifth version, registered extra scoring (docs/DEPTH_DASH_V5_PREREG.md section 3; same as the fourth version plus v5), after the seal. Corrected truth
(comma2k19 processed radar range - 0.33 m, bearing - 1.18 m) throughout:
  1. distance with the 50 % sanity gate switched off (main group), every method;
  2. distance on the development definition (radar pairs with truth < 30 m, whatever our own reach says), every method
     alone and on the pairs it shares with the fourth version;
  3. paired, main group: v4 vs v3 and v4 vs v2 distance (per segment median |err| on shared pairs inside both reaches,
     segments with >= 20 shared pairs, one-sided sign test); v4 vs v3 and v4 vs v2 relative and absolute speed on
     shared windows; v3 vs v2 ego speed (the third version's test again, on the second car); v4 ego == v3 ego (checked);
  4. segments with output, per group and method.

Moved here on 2026-10-03 from data/output/dash_scale/blind_v5/score_extra.py (where it ran after the seal); only the
repository root changed, and a re-run writes byte-identical output.
"""
import json
import math
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
SFX = {"v2": "_cars.json", "v3": "_cars_v3.json", "v4": "_cars_v4.json", "v5": "_cars_v5.json", "c": "_cars_c.json"}
GROUPS = {g: [t.strip() for t in open(B / f"tags_{g}.txt") if t.strip()] for g in ("main", "local_day", "night")}


def rows(tags, sfx):
    files = {t: B / f"{t}{sfx}" for t in tags if (B / f"{t}{sfx}").exists()}
    MC.check_sealed(str(B / "seal.json"), [str(f) for f in files.values()])
    out = {}
    for t, f in files.items():
        c = json.loads(f.read_text())
        if c.get("status") == "refused":
            continue
        out[t] = MC.c2k19_score(c, TRUTH / t, return_rows=True)
    return out


def rnd(x, n=2):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), n)


def dist_summary(pairs):
    d = {}
    for name, lanes in (("own lane", (0,)), ("next lanes", (-1, 1))):
        e = np.array([p["err"] for p in pairs if p["lane"] in lanes])
        d[name] = dict(n=int(len(e)), median_abs_pct=rnd(np.median(np.abs(e))) if len(e) else None,
                       bias_pct=rnd(np.median(e)) if len(e) else None,
                       gross_gt25_pct=rnd(100 * np.mean(np.abs(e) > 25), 1) if len(e) else None)
    return d


def sign_p(better, n):
    return sum(math.comb(n, i) for i in range(better, n + 1)) / 2 ** n if n else None


res = {}
main = GROUPS["main"]

# 1. gate off
gate = MC.PARAMS["sanity_gate"]
MC.PARAMS["sanity_gate"] = 1e9
for lab, sfx in SFX.items():
    pr = [p for _, ps, _ in rows(main, sfx).values() for p in ps if not p["far"]]
    res[f"1 gate off, {lab}"] = dist_summary(pr)
MC.PARAMS["sanity_gate"] = gate

# rows with the gate on, main group
R = {m: rows(main, s) for m, s in SFX.items()}
K = {m: {(t, p["t"], p["truth_id"]): p for t, (_, ps, _) in R[m].items() for p in ps} for m in R}

# 2. development definition: truth < 30 m
for m in R:
    near = [p for p in K[m].values() if p["truth"] < 30 and p["lane"] in (-1, 0, 1)]
    res[f"2 truth < 30 m, {m} alone"] = dist_summary(near)
    if m != "v4":
        sh = [k for k in K[m] if k in K["v4"] and K[m][k]["truth"] < 30 and K[m][k]["lane"] in (-1, 0, 1)]
        res[f"2 truth < 30 m, shared {m} and v4"] = {m: dist_summary([K[m][k] for k in sh]),
                                                     "v4": dist_summary([K["v4"][k] for k in sh])}


# 3. paired comparisons
def paired_distance(a, b):
    sh = [k for k in K[a] if k in K[b] and not K[a][k]["far"] and not K[b][k]["far"] and K[a][k]["lane"] in (-1, 0, 1)]
    by_seg = {}
    for k in sh:
        by_seg.setdefault(k[0], []).append(k)
    segs = [(np.median([abs(K[a][k]["err"]) for k in ks]), np.median([abs(K[b][k]["err"]) for k in ks]))
            for ks in by_seg.values() if len(ks) >= 20]
    better = sum(1 for x, y in segs if x < y)
    ea, eb = np.array([K[a][k]["err"] for k in sh]), np.array([K[b][k]["err"] for k in sh])
    return dict(shared_pairs=len(sh), **{f"median_abs_{a}": rnd(np.median(np.abs(ea))) if len(sh) else None,
                                         f"median_abs_{b}": rnd(np.median(np.abs(eb))) if len(sh) else None,
                                         f"bias_{a}": rnd(np.median(ea)) if len(sh) else None,
                                         f"bias_{b}": rnd(np.median(eb)) if len(sh) else None},
                segments=len(segs), **{f"{a}_better_segments": better}, sign_test_one_sided_p=sign_p(better, len(segs)))


def paired_speed(a, b, key):
    ka = {(t, p["t"], p["id"]): p for t, (_, ps, _) in R[a].items() for p in ps if p.get(key) is not None}
    kb = {(t, p["t"], p["id"]): p for t, (_, ps, _) in R[b].items() for p in ps if p.get(key) is not None}
    sh = sorted(set(ka) & set(kb))
    if not sh:
        return dict(windows=0)
    ea = np.array([abs(ka[k][key] - ka[k][key + "_truth"]) for k in sh])
    eb = np.array([abs(kb[k][key] - kb[k][key + "_truth"]) for k in sh])
    return dict(windows=len(sh), **{f"mae_{a}": rnd(ea.mean()), f"mae_{b}": rnd(eb.mean())})


for a, b in (("v4", "v3"), ("v4", "v2"), ("v3", "v2"), ("c", "v4"), ("v5", "v4")):
    res[f"3 distance paired, {a} vs {b}"] = paired_distance(a, b)
for a, b in (("v4", "v3"), ("v4", "v2"), ("v5", "v4")):
    res[f"3 relative speed paired, {a} vs {b}"] = paired_speed(a, b, "rel")
    res[f"3 absolute speed paired, {a} vs {b}"] = paired_speed(a, b, "abs")

both, segs = [], []
for t in set(R["v2"]) & set(R["v3"]):
    e2 = {round(e["t0"], 2): e for e in R["v2"][t][2] if e.get("kmh") is not None and e.get("truth") is not None}
    e3 = {round(e["t0"], 2): e for e in R["v3"][t][2] if e.get("kmh") is not None and e.get("truth") is not None}
    k = sorted(set(e2) & set(e3))
    a2 = [abs(e2[x]["kmh"] - e2[x]["truth"]) for x in k]
    a3 = [abs(e3[x]["kmh"] - e3[x]["truth"]) for x in k]
    both += list(zip(a2, a3))
    if len(k) >= 10:
        segs.append((np.mean(a2), np.mean(a3)))
bb = np.array(both)
better = sum(1 for x, y in segs if y < x)
res["3 ego speed paired, v3 vs v2"] = dict(paired_seconds=len(bb), mae_v2=rnd(bb[:, 0].mean()) if len(bb) else None,
                                           mae_v3=rnd(bb[:, 1].mean()) if len(bb) else None, segments=len(segs),
                                           v3_better_segments=better, sign_test_one_sided_p=sign_p(better, len(segs)))
diffs, n_cmp = [], 0
for t in set(R["v3"]) & set(R["v4"]):
    e3 = {round(e["t0"], 2): e["kmh"] for e in R["v3"][t][2] if e.get("kmh") is not None}
    e4 = {round(e["t0"], 2): e["kmh"] for e in R["v4"][t][2] if e.get("kmh") is not None}
    k = set(e3) | set(e4)
    n_cmp += len(k)
    diffs += [abs(e3[x] - e4[x]) if x in e3 and x in e4 else float("inf") for x in k]
res["3 ego speed v4 == v3"] = dict(seconds_compared=n_cmp, max_abs_diff_kmh=max(diffs) if diffs else None)


# fifth version: ego speed v5 vs v4 on the same seconds (segments with >= 10 paired seconds, one-sided sign test)
both, segs = [], []
for t in set(R["v4"]) & set(R["v5"]):
    e4 = {round(e["t0"], 2): e for e in R["v4"][t][2] if e.get("kmh") is not None and e.get("truth") is not None}
    e5 = {round(e["t0"], 2): e for e in R["v5"][t][2] if e.get("kmh") is not None and e.get("truth") is not None}
    k = sorted(set(e4) & set(e5))
    a4 = [abs(e4[x]["kmh"] - e4[x]["truth"]) for x in k]
    a5 = [abs(e5[x]["kmh"] - e5[x]["truth"]) for x in k]
    both += list(zip(a4, a5))
    if len(k) >= 10:
        segs.append((np.mean(a4), np.mean(a5)))
bb = np.array(both)
better = sum(1 for x, y in segs if y < x)
res["3 ego speed paired, v5 vs v4"] = dict(paired_seconds=len(bb), mae_v4=rnd(bb[:, 0].mean()) if len(bb) else None,
                                           mae_v5=rnd(bb[:, 1].mean()) if len(bb) else None, segments=len(segs),
                                           v5_better_segments=better, sign_test_one_sided_p=sign_p(better, len(segs)))
n4 = sum(1 for t in GROUPS["main"] if (B / f"{t}_cars_v4.json").exists() and json.loads((B / f"{t}_cars_v4.json").read_text()).get("status") != "refused")
n5r = sum(1 for t in GROUPS["main"] if (B / f"{t}_cars_v5.json").exists() and json.loads((B / f"{t}_cars_v5.json").read_text()).get("status") == "refused")
res["3 scale check refusals, main"] = dict(v4_with_output=n4, v5_refused_by_check=n5r, fraction=rnd(n5r / n4, 3) if n4 else None)

# 4. segments with output
cnt = {}
for g, tags in GROUPS.items():
    for m, s in SFX.items():
        ok = 0
        for t in tags:
            f = B / f"{t}{s}"
            if f.exists() and json.loads(f.read_text()).get("status") != "refused":
                ok += 1
        cnt[f"{g}, {m}"] = f"{ok}/{len(tags)}"
res["4 segments with output"] = cnt

(B / "score_extra.json").write_text(json.dumps(res, indent=1))
for k, v in res.items():
    print(k, json.dumps(v))

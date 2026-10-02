#!/usr/bin/env python3
"""Fifth version: combine the fourth-version depth output (*_cars_v4.json, ground-calibrated car ruler) with the
marking-geometry output (*_cars_c.json) of the same segment and the same tracks (2026-10-02, development on opened data).

1. Scale check. The ground ruler fits D_g = a z + b, D_g from the marking run. If the marking run locked to the wrong
   dash period, a is off by about a factor 2 while the depth model's own dash ruler k is not. So the segment is refused
   (distances and speeds) when the median a is outside [1/G_A, G_A] (the 1.5 of the second version's k guard: on opened
   data the worst segments had BOTH rulers locked to the half period, k and a 1.6-1.85, distances +64..+88 %), or, when
   the dash ruler gave k, when a / k is outside [1/G_K, G_K].
2. Ego speed. The depth method's ego speed (local ruler) and marking geometry's ego speed have nearly independent
   errors (correlation 0.12-0.18 on opened data), so each 1 s bin where both have a reading reports their mean, with
   sigma = 0.5 * hypot(sigma_depth, sigma_marking) and the 95 % range 1.96 * M_EGO * sigma. A bin with only the depth
   reading keeps it as it was; marking geometry alone is not used (it is the method that locks wrong at night).
3. Absolute speed of other cars = the window's ego speed (the mean of the two methods' window ego speeds when both
   exist) + the fourth version's relative speed; range 1.96 * hypot(M_REL * sigma_rel, M_EGO * sigma_ego_window).
Distances and relative speeds are the fourth version's, unchanged.

  python3 tools/v5_combine.py --depth X_cars_v4.json --marking X_cars_c.json --out X_cars_v5.json
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

V5 = dict(g_k=1.3, g_a=1.5, m_ego=2.0, m_rel=3.4)


def median_a(cars):
    a = [s["ground_fit"]["a"] for s in cars.get("samples", []) if (s.get("ground_fit") or {}).get("ok")]
    return float(np.median(a)) if a else None


def combine(dep, mrk, p=V5):
    out = copy.deepcopy(dep)
    out["v5"] = dict(params=dict(p), method="fifth version: scale check + mean ego of depth and marking geometry")
    if dep.get("status") == "refused":
        return out
    a, k = median_a(dep), dep.get("k")
    out["v5"].update(median_a=a, k=k)
    if a is None:
        bad = "no accepted ground fit"
    elif not 1 / p["g_a"] <= a <= p["g_a"]:
        bad = f"a = {a:.2f} outside [1/{p['g_a']}, {p['g_a']}]"
    elif k and not 1 / p["g_k"] <= a / k <= p["g_k"]:
        bad = f"a / k = {a / k:.2f} outside [1/{p['g_k']}, {p['g_k']}]"
    else:
        bad = None
    if bad:
        return dict(status="refused", reason=f"marking scale inconsistent with the depth model: {bad}", v5=out["v5"],
                    params=dep.get("params"), fps=dep.get("fps"))
    m_ok = mrk is not None and mrk.get("status") != "refused"
    em = {round(e["t0"]): e for e in (mrk.get("ego", []) if m_ok else []) if e.get("kmh") is not None}
    used = 0
    for e in out.get("ego", []):
        m = em.get(round(e["t0"]))
        if e.get("kmh") is None or m is None:
            continue
        e["kmh_depth"], e["kmh_marking"] = e["kmh"], m["kmh"]
        e["kmh"] = 0.5 * (e["kmh"] + m["kmh"])
        e["sigma_kmh"] = 0.5 * float(np.hypot(e["sigma_kmh"], m["sigma_kmh"]))
        e["ci95_kmh"] = 1.96 * p["m_ego"] * e["sigma_kmh"]
        used += 1
    om = {}
    if m_ok:
        for s in mrk.get("samples", []):
            for o in s.get("objects", []):
                if o.get("ego_win_kmh") is not None:
                    om[(round(s["t"], 3), o["id"])] = o
    n_abs = 0
    for s in out.get("samples", []):
        for o in s.get("objects", []):
            if o.get("abs_kmh") is None or o.get("ego_win_kmh") is None:
                continue
            m = om.get((round(s["t"], 3), o["id"]))
            if m is None:
                continue
            e = 0.5 * (o["ego_win_kmh"] + m["ego_win_kmh"])
            es = 0.5 * float(np.hypot(o["ego_win_sigma_kmh"], m["ego_win_sigma_kmh"]))
            o["abs_kmh_depth_ego"] = o["abs_kmh"]
            o["ego_win_kmh"], o["ego_win_sigma_kmh"] = e, es
            o["abs_kmh"] = e + o["rel_kmh"]
            o["abs_ci95_kmh"] = 1.96 * float(np.hypot(p["m_rel"] * o["rel_sigma_kmh"], p["m_ego"] * es))
            n_abs += 1
    out["v5"].update(ego_bins_combined=used, abs_windows_combined=n_abs)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--marking", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dep = json.loads(Path(a.depth).read_text())
    mrk = json.loads(Path(a.marking).read_text()) if Path(a.marking).exists() else None
    res = combine(dep, mrk)
    Path(a.out).write_text(json.dumps(res, indent=1))
    v = res.get("v5", {})
    print(f"wrote {a.out}: {res.get('status', 'ok')} {res.get('reason', '')} ego bins combined {v.get('ego_bins_combined')}, "
          f"abs windows {v.get('abs_windows_combined')}")


if __name__ == "__main__":
    main()

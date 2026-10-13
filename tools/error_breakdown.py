#!/usr/bin/env python3
"""Where the errors and refusals of a scored comma2k19 batch come from (post-hoc analysis of OPENED data: it reads the
truth, so nothing here is a blind result and nothing here may feed a prediction).

  1. Refusals: per road group, which ruler refused (depth dash ruler / marking geometry), and how far the best line of
     each was from the 3 % confidence gate (how many segments a looser gate would have let through).
  2. Distance errors inside the reach (lanes -1..1, corrected truth, the scorer's bearing pairing) split by: true
     distance, lane, whether our lane agrees with the radar's, box cut by the image edge, box partly hidden behind a nearer
     box, box height, vehicle class, how well the ground fit held (residual spread), our ego speed. Each with the 50 %
     sanity gate (as scored) and without it (the gross errors the gate hides).
  3. Ego speed per second by speed.

  python3 tools/error_breakdown.py --batch data/output/dash_scale/blind_v4 --suffix _cars_v4.json --out X.json
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import depth_dash_multicar as M

GROUPS = ("main", "local_day", "night")
CLS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def stat(e):
    e = np.asarray(e, float)
    if not len(e):
        return dict(n=0)
    return dict(n=int(len(e)), median_abs_pct=round(float(np.median(np.abs(e))), 2), bias_pct=round(float(np.median(e)), 2),
                gt10_pct=round(float(100 * np.mean(np.abs(e) > 10)), 1), gt25_pct=round(float(100 * np.mean(np.abs(e) > 25)), 1))


def refusals(B, tags, suffix):
    out = Counter(); near = Counter()
    for t in tags:
        f = B / f"{t}{suffix}"
        if f.exists() and json.loads(f.read_text()).get("status") != "refused":
            out["output"] += 1
            continue
        d = json.loads((B / f"{t}_dash.json").read_text()) if (B / f"{t}_dash.json").exists() else {}
        r = json.loads((B / f"{t}_run.json").read_text()) if (B / f"{t}_run.json").exists() else {}
        dr, rr = d.get("status") == "refused", r.get("status") == "refused"
        if f.exists() and not (dr or rr):
            out["refused by the scale check"] += 1          # fifth version and later
            continue
        out["both rulers refused" if dr and rr else "marking geometry refused" if rr else "dash ruler refused" if dr
            else "other"] += 1
        for name, j in (("dash", d), ("marking", r)):
            ci = [s.get("ci_half") for s in (j.get("sides") or {}).values() if s.get("ci_half") is not None and s.get("n", 0) >= 20]
            best = min(ci) if ci else None
            for lim in (0.04, 0.05):
                if best is not None and best <= lim:
                    near[f"{name} best line CI <= {int(lim * 100)}%"] += 1
    return dict(out), dict(near)


def occluded(o, objs):
    """Partly hidden: another box whose bottom is lower (a nearer car) covers >= 30 % of this box's width."""
    x0, y0, x1, y1 = o["box"]
    for p in objs:
        if p is o:
            continue
        a0, b0, a1, b1 = p["box"]
        if b1 > y1 + 2 and b0 < y1 and min(x1, a1) - max(x0, a0) >= 0.3 * (x1 - x0):
            return True
    return False


def factors(cars, width, ego_by_bin):
    """(t, id) -> the factors of that reading."""
    rsd = [s["ground_fit"]["rsd"] for s in cars["samples"] if (s.get("ground_fit") or {}).get("ok")]
    rsd_hi = np.percentile(rsd, 75) if rsd else None
    out = {}
    for s in cars["samples"]:
        g = s.get("ground_fit") or {}
        for o in s["objects"]:
            x0, y0, x1, y1 = o["box"]
            ego = ego_by_bin.get(int(s["t"] // 1))
            out[(round(s["t"], 3), o["id"])] = {
                "box cut by the image edge": "yes" if x0 <= 4 or x1 >= width - 4 else "no",
                "partly hidden by a nearer car": "yes" if occluded(o, s["objects"]) else "no",
                "box height": "< 30 px" if y1 - y0 < 30 else "30-60 px" if y1 - y0 < 60 else ">= 60 px",
                "class": CLS.get(o.get("cls"), "other"),
                "ground fit spread": "none" if not g.get("ok") else "top quarter" if rsd_hi is not None and g["rsd"] > rsd_hi else "lower 3/4",
                "our ego speed": "none" if ego is None else "< 60 km/h" if ego < 60 else "60-90 km/h" if ego < 90 else ">= 90 km/h",
            }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--suffix", default="_cars_v4.json")
    ap.add_argument("--truth-root", default="data/output/c2k19_truth")
    ap.add_argument("--width", type=int, default=1164)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    M.PARAMS["radar_processing_m"] = 2.70                       # corrected truth (radar - 0.33 m)
    B = Path(a.batch)
    res = {}
    for grp in GROUPS:
        tags = [l.strip() for l in open(B / f"tags_{grp}.txt") if l.strip() and not l.startswith("#")]
        ref, near = refusals(B, tags, a.suffix)
        rows = {"gate": [], "nogate": []}
        ego_rows = []
        for t in tags:
            f = B / f"{t}{a.suffix}"
            if not f.exists():
                continue
            cars = json.loads(f.read_text())
            if cars.get("status") == "refused":
                continue
            ego_by_bin = {int(e["t0"]): e["kmh"] for e in cars.get("ego", []) if e.get("kmh") is not None}
            fac = factors(cars, a.width, ego_by_bin)
            for gate_name, gate in (("gate", 0.5), ("nogate", 1e9)):
                M.PARAMS["sanity_gate"] = gate
                _, pairs, eg = M.c2k19_score(cars, Path(a.truth_root) / t, return_rows=True)
                for q in pairs:
                    if q["lane"] not in (-1, 0, 1) or q["far"]:
                        continue
                    f_ = dict(fac.get((round(q["t"], 3), q["id"]), {}))
                    f_["true distance"] = f"{int(q['truth'] // 5 * 5)}-{int(q['truth'] // 5 * 5) + 5} m"
                    f_["lane"] = {0: "own", -1: "left", 1: "right"}[q["lane"]]
                    f_["our lane = radar lane"] = "yes" if q.get("our_lane") == q["lane"] else "no"
                    rows[gate_name].append((q["err"], f_))
                if gate_name == "gate":
                    ego_rows += [(e["kmh"] - e["truth"], e["truth"]) for e in eg
                                 if e.get("kmh") is not None and e.get("truth") is not None]
            M.PARAMS["sanity_gate"] = 0.5
        g = dict(refusals=ref, refused_but_near_the_gate=near, segments=len(tags),
                 distance_all=dict(gate=stat([e for e, _ in rows["gate"]]), nogate=stat([e for e, _ in rows["nogate"]])),
                 distance_by={})
        keys = sorted({k for _, f_ in rows["gate"] for k in f_})
        for k in keys:
            g["distance_by"][k] = {}
            for v in sorted({f_.get(k) for _, f_ in rows["gate"] if f_.get(k) is not None}):
                g["distance_by"][k][v] = dict(gate=stat([e for e, f_ in rows["gate"] if f_.get(k) == v]),
                                              nogate=stat([e for e, f_ in rows["nogate"] if f_.get(k) == v]))
        # what the gross errors (|err| > 25 %, no gate) are made of, against all readings
        gross = [f_ for e, f_ in rows["nogate"] if abs(e) > 25]
        g["gross_errors_share"] = {k: {v: dict(all_pct=round(100 * np.mean([f_.get(k) == v for _, f_ in rows["nogate"]]), 1),
                                                gross_pct=round(100 * np.mean([f_.get(k) == v for f_ in gross]), 1) if gross else None)
                                       for v in g["distance_by"].get(k, {})} for k in keys}
        g["n_gross_nogate"] = len(gross)
        er = np.array(ego_rows) if ego_rows else np.zeros((0, 2))
        g["ego_by_speed"] = {lab: stat(100 * er[m, 0] / np.maximum(er[m, 1], 1)) | dict(mae_kmh=round(float(np.mean(np.abs(er[m, 0]))), 2) if m.any() else None)
                             for lab, m in (("< 60 km/h", er[:, 1] < 60), ("60-90 km/h", (er[:, 1] >= 60) & (er[:, 1] < 90)),
                                            (">= 90 km/h", er[:, 1] >= 90))} if len(er) else {}
        res[grp] = g
        print(f"\n[{grp}] {len(tags)} segments: {ref} | near the gate: {near}")
        print(f"  distance inside the reach: gated {g['distance_all']['gate']} | no gate {g['distance_all']['nogate']}")
        for k in keys:
            print(f"  {k}: " + "; ".join(f"{v} {d['gate'].get('median_abs_pct')}% (n {d['gate']['n']}, >25% no-gate {d['nogate'].get('gt25_pct')}%)"
                                        for v, d in g["distance_by"][k].items()))
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

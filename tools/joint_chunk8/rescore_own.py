"""Independent re-score of the Chunk_7 first-half dry runs under the settled shared rules (audit, 2026-10-19).

Written from mddbench/comma2k19.py (truth sampling, slot linking, rounding as written to objects.csv / ego.csv),
mddbench/score.py (prepare_truth, merge_duplicates, score_segment, summarize) and mddbench/match.py (match_frame),
with configs/eval_comma2k19.json values and the registration's matching centre 615 (582 as sensitivity).
Does not import mddbench or rescore_lib. Opened data only (Chunk_7 first half = blind_v8 tags). Truth values are
computed in memory and never written; only summary numbers are written.

  python rescore_own.py            -> rescore_own.json next to this folder
"""
from __future__ import annotations

import csv
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = Path("/home/terry/Monocular-Depth-Vehicle-Distance")
V8 = REPO / "data/output/dash_scale/blind_v8"
ZIP = REPO / "data/input/comma2k19/Chunk_7.zip"
TRUTH_T = REPO / "data/output/c2k19_truth"
OUT = Path("/mnt/d/depth_work/audit_chunk8_20261019/dryrun/rescore_own.json")
RUNS = {"run (S7+S8)": Path("/mnt/d/depth_work/v9dry/run"), "keep (KEEP_A, S8=0)": Path("/mnt/d/depth_work/v9dry/keep")}

OFF = {"99c94dc769b5d96e": -0.70, "b0c9d2329ad1606b": -0.33}
EMPTY_MIN = 258.6
FX = 910.0
MOVING, HALF, ADJ, MIN_D, DUP_D, DUP_LAT = 7.2, 1.83, 5.49, 1.0, 1.0, 0.5
KW, PX, DW, WMIN, WMAX = 0.75, 10.0, 0.01, 0.8, 4.5
NEAR = 30.0
LINK_GAP, LINK_DD, LINK_DLAT = 2, 2.5, 1.2
BLOCKED = 1e9


def r(x, nd):  # the value as written with f"{x:.{nd}f}" and read back
    return float(f"{x:.{nd}f}")


# ---------------- truth, shared way (in memory) ----------------
def npy(z, name):
    return np.load(io.BytesIO(z.read(name)), allow_pickle=False)


def truth_zip(z, tag):
    dongle, date, num = tag.split("_")
    base = f"Chunk_7/{dongle}|{date}/{int(num)}/"
    ft = npy(z, base + "global_pose/frame_times")
    fv = npy(z, base + "global_pose/frame_velocities")
    rt = npy(z, base + "processed_log/CAN/radar/t")
    rv = npy(z, base + "processed_log/CAN/radar/value")
    ego = {i: r(float(v), 3) for i, v in enumerate(np.linalg.norm(np.asarray(fv, np.float64), axis=1) * 3.6)}
    rows = []
    for slot in np.unique(rv[:, 5]):
        k = rv[:, 5] == slot
        ts, vals = rt[k], rv[k]
        o = np.argsort(ts, kind="stable")
        ts, vals = ts[o], vals[o]
        idx = np.searchsorted(ts, ft)
        for i, tf in enumerate(ft):
            cand = [j for j in (idx[i] - 1, idx[i]) if 0 <= j < len(ts)]
            if not cand:
                continue
            j = min(cand, key=lambda j: abs(ts[j] - tf))
            if abs(ts[j] - tf) > 0.05:
                continue
            rows.append([i, int(slot), float(vals[j, 0]), float(vals[j, 1]), float(vals[j, 2])])
    rows.sort(key=lambda q: (q[0], q[1]))
    ids = link(rows)
    obj = [(fi, tid, r(d, 2), r(la, 2), r(v * 3.6, 3)) for (fi, _, d, la, v), tid in zip(rows, ids)]
    return ego, obj, len(ft)


def link(rows):
    ids, last = [], {}
    for fi, slot, dist, lat, _ in rows:
        prev = last.get(slot)
        if prev is None:
            k = 0
        else:
            pf, pd_, pl, pk = prev
            same = fi - pf - 1 <= LINK_GAP and abs(dist - pd_) < LINK_DD and abs(lat - pl) < LINK_DLAT
            k = pk if same else pk + 1
        last[slot] = (fi, dist, lat, k)
        ids.append(f"{slot}-{k}")
    return ids


def truth_terry(tag):
    """Variant: Terry's existing truth files (radar.csv: each track's latest return within 0.05 s before the frame)."""
    d = TRUTH_T / tag
    ego = {int(q["frame_idx"]): float(q["pose_speed_ms"]) * 3.6 for q in csv.DictReader(open(d / "ego.csv"))}
    rows = [[int(q["frame_idx"]), int(q["track_id"]), float(q["d_rel_m"]), float(q["y_rel_m"]), float(q["v_rel_ms"])]
            for q in csv.DictReader(open(d / "radar.csv"))]
    rows.sort(key=lambda q: (q[0], q[1]))
    ids = link(rows)
    obj = [(fi, tid, d_, la, v * 3.6) for (fi, _, d_, la, v), tid in zip(rows, ids)]
    return ego, obj, None


def prepare(ego, obj, off):
    """score.prepare_truth: drop empty slots, camera distance, d > 1 m, merge duplicates, scope and lane."""
    t = []
    for fi, tid, dn, la, rel in obj:
        if not dn < EMPTY_MIN:
            continue
        e = ego.get(fi, float("nan"))
        dc = dn + off
        if not dc > MIN_D:
            continue
        t.append(dict(fi=fi, tid=tid, d=dc, lat=la, rel=rel, abs=e + rel))
    t.sort(key=lambda q: (q["fi"], q["tid"]))           # stable: frame, then truth_id as a string
    keep = [True] * len(t)
    s = 0
    for e_ in range(1, len(t) + 1):
        if e_ < len(t) and t[e_]["fi"] == t[s]["fi"]:
            continue
        for i in range(s + 1, e_):
            for j in range(s, i):
                if keep[j] and abs(t[i]["d"] - t[j]["d"]) <= DUP_D and abs(t[i]["lat"] - t[j]["lat"]) <= DUP_LAT:
                    keep[i] = False
                    break
        s = e_
    t = [q for q, k in zip(t, keep) if k]
    for q in t:
        q["in_scope"] = bool(abs(q["abs"]) >= MOVING and abs(q["lat"]) <= ADJ)   # NaN abs -> False
        q["lane"] = 0 if abs(q["lat"]) <= HALF else int(np.sign(q["lat"]))
    return t


def match(boxes, truths, cx):
    """match.match_frame: centre within 0.75 w + 10 px of the truth column, implied width 0.8-4.5 m, Hungarian."""
    if not boxes or not truths:
        return []
    x0 = np.array([b["x0"] for b in boxes], float); x1 = np.array([b["x1"] for b in boxes], float)
    cu, w = (x0 + x1) / 2.0, np.maximum(x1 - x0, 1.0)
    d = np.array([q["d"] for q in truths], float); la = np.array([q["lat"] for q in truths], float)
    u = cx - FX * la / d
    du = cu[:, None] - u[None, :]
    iw = d[None, :] * w[:, None] / FX
    gate = (np.abs(du) <= (KW * w + PX)[:, None]) & (iw >= WMIN) & (iw <= WMAX)
    cost = np.where(gate, np.abs(du) / w[:, None] + DW * d[None, :], BLOCKED)
    rr, cc = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(rr, cc) if cost[i, j] < BLOCKED]


def preds(cars, cut60):
    """run.py convert(): every object with its box; dist as written (cut60: readings over 60 m left blank)."""
    frames, boxes = [], {}
    for s in cars["samples"]:
        fi = int(s["fi"]); frames.append(fi)
        for o in s["objects"]:
            dist = o.get("dist")
            dist = float("nan") if dist is None or (cut60 and dist > 60.0) else float(dist)
            rel = o.get("rel_kmh")
            boxes.setdefault(fi, []).append(dict(x0=float(o["box"][0]), x1=float(o["box"][2]), dist=dist,
                                                 rel=float("nan") if rel is None else float(rel)))
    return set(frames), boxes


def score_seg(tag, cars, truth, cx, cut60):
    ego, obj, _ = truth
    t = prepare(ego, obj, OFF[tag.split("_")[0]])
    frames, boxes = preds(cars, cut60)
    t = [q for q in t if q["fi"] in frames]
    by = {}
    for q in t:
        by.setdefault(q["fi"], []).append(q)
    pairs = []
    for fi in sorted(boxes):
        tt = by.get(fi)
        if not tt:
            continue
        for i, j in match(boxes[fi], tt, cx):
            q = tt[j]
            if not q["in_scope"]:
                continue
            b = boxes[fi][i]
            pairs.append(dict(seg=tag, fi=fi, d_truth=q["d"], d_pred=b["dist"], rel_truth=q["rel"], rel_pred=b["rel"], lane=q["lane"]))
    n_scope = sum(q["in_scope"] for q in t)
    n_scope30 = sum(q["in_scope"] and q["d"] <= NEAR for q in t)
    return pairs, n_scope, n_scope30


def summarize(pairs, n_scope, n_scope30):
    P = pairs
    near = [p for p in P if math.isfinite(p["d_pred"]) and p["d_truth"] <= NEAR]
    if not near:
        return {"n": 0}
    e = np.array([(p["d_pred"] - p["d_truth"]) / p["d_truth"] * 100 for p in near])
    m = np.array([abs(p["d_pred"] - p["d_truth"]) for p in near])
    def bin_m(lo, hi):
        b = [abs(p["d_pred"] - p["d_truth"]) for p in P if math.isfinite(p["d_pred"]) and lo < p["d_truth"] <= hi]
        return round(float(np.median(b)), 3) if b else None
    pn = [p for p in P if p["d_truth"] <= NEAR and math.isfinite(p["rel_pred"]) and math.isfinite(p["rel_truth"])]
    pn_d = [p for p in pn if math.isfinite(p["d_pred"]) and p["d_pred"] <= 60.0]
    far_any = [p for p in P if math.isfinite(p["d_pred"]) and p["d_pred"] > 60.0 and p["d_truth"] <= NEAR]
    matched30 = sum(1 for p in P if p["d_truth"] <= NEAR)
    return dict(n=len(near), med_pct=round(float(np.median(np.abs(e))), 3), med_m=round(float(np.median(m)), 3),
                bias_pct=round(float(np.median(e)), 3), gt25_pct=round(100 * float(np.mean(np.abs(e) > 25)), 2),
                bins_m={"0-10": bin_m(0, 10), "10-20": bin_m(10, 20), "20-30": bin_m(20, 30)},
                rel_mae_shared=round(float(np.mean([abs(p["rel_pred"] - p["rel_truth"]) for p in pn])), 3) if pn else None,
                rel_n_shared=len(pn),
                rel_mae_with_dist_le60=round(float(np.mean([abs(p["rel_pred"] - p["rel_truth"]) for p in pn_d])), 3) if pn_d else None,
                rel_n_with_dist_le60=len(pn_d),
                near_pairs_with_pred_over_60m=len(far_any),
                found_pct_all=round(100 * len(P) / n_scope, 2) if n_scope else None,
                found_pct_le30=round(100 * matched30 / n_scope30, 2) if n_scope30 else None)


def main():
    groups = {g: (V8 / f"tags_{g}.txt").read_text().split() for g in ("main", "local_day", "night")}
    z = zipfile.ZipFile(ZIP)
    truth = {"zip": {}, "terry": {}}
    for g, tags in groups.items():
        for t in tags:
            truth["zip"][t] = truth_zip(z, t)
            truth["terry"][t] = truth_terry(t)
    print("truth ready", flush=True)
    res = {}
    for name, folder in RUNS.items():
        cars = {}
        for tags in groups.values():
            for t in tags:
                c = json.loads((folder / f"{t}_cars_v9.json").read_text())
                cars[t] = None if c.get("status") == "refused" else c
        for tv in ("zip", "terry"):
            for cx in (615.0, 582.0):
                for cut60 in (False, True):
                    for g, tags in groups.items():
                        P, ns, ns30, nout = [], 0, 0, 0
                        for t in tags:
                            if cars[t] is None:
                                continue
                            nout += 1
                            p, a, b = score_seg(t, cars[t], truth[tv][t], cx, cut60)
                            P += p; ns += a; ns30 += b
                        key = f"{name} | truth {tv} | cx {cx:.0f} | {'over-60 blank' if cut60 else 'as run.py writes'} | {g}"
                        res[key] = dict(segments=len(tags), with_output=nout, **summarize(P, ns, ns30))
                        print(key, res[key], flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

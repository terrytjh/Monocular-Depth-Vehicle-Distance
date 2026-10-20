#!/usr/bin/env python3
"""Rule-variant re-scoring of Terry's sealed, already-opened comma2k19 predictions (decisions D1-D4).

Read-only on both repos and on data/: reads *_cars_*.json (checked against the batch seal.json first) and the existing
truth radar.csv / ego.csv under data/output/c2k19_truth/ (opened segments only); writes only into this folder.

Two scoring frameworks, each with switches:

  T  = Terry's registered scorer, tools/depth_dash_multicar.py c2k19_score with radar_processing_m = 2.70 (truth = radar
       - 0.33 m, bearing range = radar - 1.18 m), as used by tools/v8_score.py / v9_score.py: only moving (|v_rel + v_CAN|
       > 2 m/s) returns inside the three 12-ft lanes enter the matching; the 50 % distance gate is inside the matching.
  S  = a re-implementation (own code, not mddbench) of the shared draft rules of docs/SPEC.md section 5 / configs/
       eval_comma2k19.json applied to Terry's boxes: every return with d_cam > 1 m enters the matching, only in-scope ones
       (|pose ego + v_rel| >= 7.2 km/h, |lat| <= 5.49 m) are scored; duplicate returns (1 m / 0.5 m) merged; the
       vehicle-width gate 0.8-4.5 m; no distance gate (option: post-hoc 50 % drop, or 50 % inside the matching).

Both: Terry's boxes and distances; readings he outputs beyond 60 m carry no number (v8/v9 rule; dropped after the
matching, as a blank dist_m would be); relative-speed truth = Terry's window average (the D8 question is not handled here).
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = Path("/home/terry/Monocular-Depth-Vehicle-Distance")
DS = REPO / "data/output/dash_scale"
TRUTH = REPO / "data/output/c2k19_truth"
CAM_F, CAM_CX = 910.0, 582.0
TERRY_LANE_HALF = 3.6576 / 2
NO_NUMBER_M = 60.0

T_DEFAULT = dict(name="T", off=-0.33, bearing="terry", mov_ms=2.0, mov_src="can", mov_ge=False, prefilter=True,
                 gate="match", gate_frac=0.5, width=None, dup=None, min_d=0.0,
                 lane_half=TERRY_LANE_HALF, adj_max=3 * TERRY_LANE_HALF)
S_DEFAULT = dict(name="S", off=-0.33, bearing="offset", mov_ms=7.2 / 3.6, mov_src="pose", mov_ge=True, prefilter=False,
                 gate="none", gate_frac=0.5, width=(0.8, 4.5), dup=(1.0, 0.5), min_d=1.0,
                 lane_half=1.83, adj_max=5.49)


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tags_of(batch: Path, group: str) -> list[str]:
    p = batch / f"tags_{group}.txt"
    return [l.strip() for l in open(p) if l.strip() and not l.startswith("#")] if p.exists() else []


def drive_of(tag: str) -> str:
    return tag.rsplit("_", 1)[0]


_SEAL = {}


def load_cars(batch: Path, tag: str, sfx: str):
    """The cars file if it exists and matches the batch seal (refuse otherwise); None when refused / absent."""
    p = batch / f"{tag}{sfx}"
    if not p.exists():
        return None
    if batch not in _SEAL:
        _SEAL[batch] = json.loads((batch / "seal.json").read_text())["files"]
    if _SEAL[batch].get(p.name) != sha256(p):
        raise SystemExit(f"{p}: not in the seal or changed -- refusing")
    c = json.loads(p.read_text())
    return None if c.get("status") == "refused" else c


_TRUTH = {}


def load_truth(tag: str):
    if tag in _TRUTH:
        return _TRUTH[tag]
    d = TRUTH / tag
    if not (d / "radar.csv").exists():
        raise SystemExit(f"{tag}: no opened truth -- refusing")
    rows = list(csv.DictReader(open(d / "ego.csv")))
    can = np.array([float(r["can_speed_ms"]) for r in rows])
    pose = np.array([float(r["pose_speed_ms"]) for r in rows])
    by_f = defaultdict(list)
    for r in csv.DictReader(open(d / "radar.csv")):
        by_f[int(r["frame_idx"])].append((int(r["track_id"]), float(r["d_rel_m"]), float(r["y_rel_m"]), float(r["v_rel_ms"])))
    _TRUTH[tag] = (can, pose, dict(by_f))
    return _TRUTH[tag]


def terry_lane(y, half, adj):
    if abs(y) <= half:
        return 0
    return (-1 if y > 0 else 1) if abs(y) <= adj else None


def score_seg(cars: dict, tag: str, o: dict, near_max: float = 30.0):
    """(pairs, n_truth_in_scope_le_near, n_found_le_near) for one segment under option set o."""
    can, pose, by_f = load_truth(tag)
    fps = cars["fps"]
    ego_src = can if o["mov_src"] == "can" else pose
    off = o["off"]
    # Terry's relative-speed truth uses the slot history of moving in-lane returns (as in c2k19_score)
    by_track = {}
    for f, rets in by_f.items():
        if f >= len(can):
            continue
        for rid, d, y, v in rets:
            if abs(v + can[f]) > 2.0 and d > 0 and terry_lane(y, TERRY_LANE_HALF, 3 * TERRY_LANE_HALF) is not None:
                by_track.setdefault(rid, {})[f] = (d, v)
    pairs, n_truth, n_found = [], 0, 0
    for s in cars["samples"]:
        fi, objs = s["fi"], s["objects"]
        if fi >= len(can):
            continue
        cand = []
        for rid, d, y, v in by_f.get(fi, []):
            if d <= 0 or d + off <= o["min_d"]:
                continue
            a = (v + ego_src[fi]) if o.get("mov_signed") else abs(v + ego_src[fi])
            moving = a >= o["mov_ms"] if o["mov_ge"] else a > o["mov_ms"]
            lane = terry_lane(y, o["lane_half"], o["adj_max"])
            ins = moving and lane is not None
            if o["prefilter"] and not ins:
                continue
            cand.append([rid, d, y, v, ins, lane])
        if o["dup"] is not None and len(cand) > 1:
            cand.sort(key=lambda r: r[0])
            keep = []
            for r in cand:
                if any(abs(r[1] - k[1]) <= o["dup"][0] and abs(r[2] - k[2]) <= o["dup"][1] for k in keep):
                    continue
                keep.append(r)
            cand = keep
        n_truth += sum(1 for r in cand if r[4] and r[1] + off <= near_max)
        if not objs or not cand:
            continue
        C = np.full((len(objs), len(cand)), 1e6)
        U = np.zeros_like(C)
        for i, ob in enumerate(objs):
            x0, _, x1, _ = ob["box"]
            wb, ub = max(1.0, x1 - x0), (x0 + x1) / 2
            for j, (rid, d, y, v, ins, lane) in enumerate(cand):
                dt = d + off
                db = d - 1.18 if o["bearing"] == "terry" else dt
                U[i, j] = ub - (CAM_CX - CAM_F * y / db)
                ok = abs(U[i, j]) <= 0.75 * wb + 10
                if ok and o["width"] is not None:
                    iw = dt * wb / CAM_F
                    ok = o["width"][0] <= iw <= o["width"][1]
                if ok and o["gate"] == "match":
                    ok = abs(ob["dist"] - dt) <= o["gate_frac"] * dt
                if ok:
                    C[i, j] = abs(U[i, j]) / wb + 0.01 * dt
        for i, j in zip(*linear_sum_assignment(C)):
            if C[i, j] >= 1e6:
                continue
            rid, d, y, v, ins, lane = cand[j]
            if not ins:
                continue
            ob, dt = objs[i], d + off
            if dt <= near_max:
                n_found += 1
            p = dict(seg=tag, drive=drive_of(tag), t=s["t"], fi=fi, id=ob["id"], truth_id=rid, lane=lane, far=ob["far"],
                     dist=ob["dist"], truth=dt, err=100 * (ob["dist"] - dt) / dt, du=float(U[i, j]),
                     abs_truth_ms=float(v + pose[fi]), oncoming=bool((v + pose[fi]) < 0))
            if ob.get("rel_kmh") is not None:
                f0, f1 = ob["win"]
                vs = [vv for ff, (dd, vv) in by_track.get(rid, {}).items()
                      if f0 <= ff <= f1 and abs(dd - (d + v * (ff - fi) / fps)) <= 2.0]
                vt = float(np.mean(vs)) if vs else v
                p.update(rel=ob["rel_kmh"], rel_truth=3.6 * vt, rel_truth_inst=3.6 * v)
            pairs.append(p)
    if o["gate"] == "post":
        for p in pairs:
            p["post_dropped"] = abs(p["err"]) > 100 * o["gate_frac"]
    return pairs, n_truth, n_found


def score_batch(batch: Path, tags: list[str], sfx: str, o: dict, near_max: float = 30.0):
    rows, n_truth, n_found, n_out = [], 0, 0, 0
    for t in tags:
        c = load_cars(batch, t, sfx)
        if c is None:
            continue
        n_out += 1
        pr, nt, nf = score_seg(c, t, o, near_max)
        rows += pr
        n_truth += nt
        n_found += nf
    no_number = sum(1 for p in rows if p["dist"] > NO_NUMBER_M)
    rows = [p for p in rows if p["dist"] <= NO_NUMBER_M]
    return dict(rows=rows, n_truth=n_truth, n_found=n_found, segments=len(tags), with_output=n_out, no_number=no_number)


def boot_ci(groups, stat, n=2000, seed=20261018):
    """95 % interval of stat(pooled array) when whole clusters (segments or drives) are drawn with replacement."""
    ks = [k for k, v in groups.items() if len(v)]
    if len(ks) < 2:
        return None
    rng = np.random.default_rng(seed)
    arrs = [groups[k] for k in ks]
    vals = [stat(np.concatenate([arrs[i] for i in rng.integers(0, len(ks), len(ks))])) for _ in range(n)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 2), round(float(hi), 2)]


def summ(rows, ci=False, nboot=2000, cluster="seg"):
    """n, median |error| % and m, signed median (bias) % and m, share > 25 %; optional cluster-bootstrap 95 % intervals."""
    if not rows:
        return dict(n=0)
    e = np.array([p["err"] for p in rows])
    m = np.array([p["dist"] - p["truth"] for p in rows])
    out = dict(n=int(len(e)), drives=len({p["drive"] for p in rows}), segs=len({p["seg"] for p in rows}),
               median_abs_pct=round(float(np.median(np.abs(e))), 2), bias_pct=round(float(np.median(e)), 2),
               gt25_pct=round(float(100 * np.mean(np.abs(e) > 25)), 1), median_abs_m=round(float(np.median(np.abs(m))), 2),
               bias_m=round(float(np.median(m)), 2))
    if ci:
        ge, gm = defaultdict(list), defaultdict(list)
        for p in rows:
            ge[p[cluster]].append(abs(p["err"]))
            gm[p[cluster]].append(abs(p["dist"] - p["truth"]))
        ge = {k: np.array(v) for k, v in ge.items()}
        gm = {k: np.array(v) for k, v in gm.items()}
        out[f"ci95_pct_by_{cluster}"] = boot_ci(ge, np.median, nboot)
        out[f"ci95_m_by_{cluster}"] = boot_ci(gm, np.median, nboot)
    return out


def rel_summ(rows):
    r = [p for p in rows if p.get("rel") is not None]
    if not r:
        return dict(n=0)
    e = np.array([p["rel"] - p["rel_truth"] for p in r])
    return dict(n=len(r), mae_kmh=round(float(np.mean(np.abs(e))), 2),
                guess0_kmh=round(float(np.mean(np.abs([p["rel_truth"] for p in r]))), 2))

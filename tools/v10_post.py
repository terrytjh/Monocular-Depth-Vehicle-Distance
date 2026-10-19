#!/usr/bin/env python3
"""Tenth version (v10) = the ninth version + D1 + C2 + C4 + C5, as a post-processing step on the ninth version's own
per-segment output files (2026-10-19; changes found and chosen on opened development data only: Chunk_6, Chunk_7).

No truth is read: only the batch folder's prediction files of each segment, the batch's group lists (tags_main.txt,
tags_local_day.txt, tags_night.txt), the camera's A (car_a.json, learned by tools/car_a.py from earlier sealed
predictions) and the parameters file (tools/v10_params.json). No depth model is run. Refused segments stay refused.

  D1 (distance, camera-k fallback segments: next_candidate.source "camera k day" / "camera k night", i.e. the depth
     method's own scale was refused or disagreed with the camera k and the ninth version reads the cars with the camera k
     only): the camera-k measurement (*_cars_kc.json by day, *_cars_kn.json at night) is averaged, for every reading inside
     its reach, with marking geometry read with the camera's A and the segment's own lane-line crossing row:
        d = (1 - w) d_camera_k + w A_camera / (y_bottom - vp_row),
     vp_row from the nearest window of *_run.json with both lines within 1 s (as measure-c), y_bottom - vp_row >= 3 px;
     the marking half Theil-Sen smoothed along each track (+-4 samples that have a value, >= 3); A_camera = median of the
     same camera's drive medians in car_a.json leaving out the segment's own drive (the ninth version's S7 rule). Then the
     ninth version's far-ruler step again (tools/v6_far_ruler.combine: own-car ruler, same-class ruler from 3 cars,
     apply_from none), so far readings follow the new near distances. Identity check first: that step on the unchanged
     camera-k file must reproduce the ninth version's distances, otherwise the segment keeps the ninth version's.
  C2 (ego speed): each second that has a ninth-version value gets the value at that second of a straight line fitted
     over the seconds within +-h s that have a value (>= 3 of them, else their mean), weighted 1 / max(95 % half-width,
     0.5 km/h)^2 when the group's parameters say so; h per group. 95 % half-width = factor (group, source) x the ninth
     version's (source: both methods averaged / depth alone). Uses up to h seconds after the second (after-the-fact
     measurement). The ninth version's value is kept beside it (kmh_v9).
  C4 (relative speed): each method's slope over the longest complete window among the half-widths of the parameters
     (samples of 0.2 s; complete = every sample present, none beyond the method's reach, no step > 15 %), then the
     registered rule: the mean of the two methods where the ninth version averaged them, depth alone otherwise. The
     depth series is the one the ninth version's relative speed was fitted to (*_cars_v4h.json dist_raw on the current
     path, the camera-k measurement on the camera-k path), the marking series the marking cars file of the combine step
     (*_cars_cca.json for the ninth version, *_cars_c.json for the seventh). sigma = hypot(slope se, |v| x ruler scale
     sigma), the mean's sigma = 0.5 hypot; 95 % half-width = 1.96 x 3.4 x sigma x rel_range_factor. Identity check
     first: the registered +-0.8 s fit must reproduce the ninth version's value, otherwise the reading keeps it.
  C5 (relative speed, main group only): readings on the current path that the ninth version reports from depth alone
     also get marking geometry's slope when its series is complete over the window counting samples beyond its reach
     whose own distance is <= D m, and report the mean of the two.
  Absolute speed = the tenth version's ego speed of the reading's second (floor(frame / fps)) + the tenth version's
     relative speed, where both exist; 95 % half-width = hypot of the two half-widths. The ninth version's values are kept
     beside every changed value (*_v9), every changed value lists its steps in v10_steps, and the file carries a "v10"
     provenance block (parameters file sha256, input files sha256, counts per step). Deterministic.

  python3 tools/v10_post.py --batch DIR --out DIR [--suffix _cars_v9.json] [--marking-suffix _cars_cca.json]
                            [--tags-dir DIR] [--car-a car_a.json] [--params tools/v10_params.json] [--seal seal.json]
  (development check on Chunk_6: --suffix _cars_v7.json --marking-suffix _cars_c.json --car-a <a car_a.json>)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v6_far_ruler as V6  # noqa: E402

GROUPS = ("main", "local_day", "night")
FAR_KEYS = ("dist_v5", "dist_own_car", "dist_class", "far_ruler", "dist_kc", "dist_mg_ca")
M_REL_Z = 1.96
TOL_KMH, TOL_M = 1e-6, 1e-9


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Batch:
    """The batch folder's files, read once each, sha256 recorded; optionally checked against a seal (files by name)."""

    def __init__(self, folder: Path, seal: Path | None):
        self.folder = folder
        self.read = {}
        self.seal = None
        if seal is not None:
            sj = json.loads(seal.read_text())
            self.seal = dict(sj.get("files") or {})

    def load(self, name: str):
        p = self.folder / name
        if "c2k19_truth" in str(p.resolve()):
            raise SystemExit(f"{p}: truth path -- refusing")
        if not p.exists():
            return None
        raw = p.read_bytes()
        h = sha256(raw)
        if self.seal is not None and self.seal.get(name) != h:
            raise SystemExit(f"{p}: not in the seal or changed -- refusing")
        self.read[name] = h
        return json.loads(raw)


def usable(c):
    return c is not None and c.get("status") != "refused"


# ------------------------------------------------------------------ D1: marking geometry with the camera A

def camera_A(car_a: dict, tag: str):
    cam, drive = tag.split("_")[0], tag.split("_")[1]
    c = car_a.get("cameras", {}).get(cam)
    if not c:
        return None
    vals = [v for d, v in c["drive_median_A"].items() if d != drive]
    return float(st.median(vals)) if vals else None


def nearest_geometry(windows, fi, fps, max_s):
    """= tools/marking_geometry.nearest_geometry: the nearest window with a horizon row and both lines, within max_s."""
    ok = [wd for wd in windows if wd.get("vp_row") is not None and wd.get("left") and wd.get("right")]
    if not ok:
        return None
    wd = min(ok, key=lambda x: abs(x["frame"] - fi))
    return wd if abs(wd["frame"] - fi) / fps <= max_s else None


def mg_dist(windows, fi, fps, A, box, max_s, min_rows):
    wd = nearest_geometry(windows, fi, fps, max_s)
    if wd is None:
        return None
    dy = box[3] - wd["vp_row"]
    return A / dy if dy >= min_rows else None


def theil_sen(t, d):
    t, d = np.asarray(t, float), np.asarray(d, float)
    i, j = np.triu_indices(len(t), 1)
    dt = t[j] - t[i]
    ok = dt != 0
    s = float(np.median((d[j] - d[i])[ok] / dt[ok]))
    return s, float(np.median(d - s * t))


def smooth_track_values(samples, vals, half, min_n):
    """vals {(j, id): d} on sample index j -> Theil-Sen value at each sample over +-half samples that have a value."""
    by_id = defaultdict(dict)
    for (j, i), d in vals.items():
        by_id[i][j] = d
    out = {}
    for i, seq in by_id.items():
        for j, d in seq.items():
            nb = [k for k in range(j - half, j + half + 1) if k in seq]
            if len(nb) < min_n:
                out[(j, i)] = d
                continue
            tt = np.array([samples[k]["t"] for k in nb]) - samples[j]["t"]
            out[(j, i)] = theil_sen(tt, np.array([seq[k] for k in nb]))[1]
    return out


def far_step(cars, P):
    fr = P["D1"]["far_ruler"]
    return V6.combine(cars, fr["frame_width"], V6.V6, fr["apply_from"], fr["class_min_tracks"])


def step_d1(out, kc, run, A, P, info):
    """Changes out's distances in place (camera-k segment)."""
    p = P["D1"]
    ref = far_step(kc, P)
    dv = {(s["fi"], o["id"]): o for s in out["samples"] for o in s["objects"]}
    worst = 0.0
    for s in ref["samples"]:
        for o in s["objects"]:
            v = dv.get((s["fi"], o["id"]))
            if v is None or (v.get("dist") is None) != (o.get("dist") is None):
                worst = math.inf
            elif o.get("dist") is not None:
                worst = max(worst, abs(v["dist"] - o["dist"]))
    n_ref = sum(len(s["objects"]) for s in ref["samples"])
    n_out = sum(len(s["objects"]) for s in out["samples"])
    info.update(identity_max_abs_m=worst, identity_readings=n_ref)
    if worst > TOL_M or n_ref != n_out:
        info.update(applied=False, why="the far-ruler step on the camera-k file does not reproduce the ninth version")
        return
    if A is None:
        info.update(applied=False, why="no camera A for this camera")
        return
    raw = {}
    windows = run.get("windows", []) if run else []          # the lane-line windows (as the candidate: any run status)
    for j, s in enumerate(kc["samples"]):
        for o in s["objects"]:
            d = mg_dist(windows, s["fi"], kc["fps"], A, o["box"], p["geometry_max_s"], p["min_rows_below_horizon"])
            if d is not None:
                raw[(j, o["id"])] = d
    mg = smooth_track_values(kc["samples"], raw, p["smooth_half_samples"], p["smooth_min_n"]) if p["smooth"] else raw
    k2 = copy.deepcopy(kc)
    n_avg = 0
    for j, s in enumerate(k2["samples"]):
        for o in s["objects"]:
            if o.get("far") or o.get("dist") is None:
                continue
            m = mg.get((j, o["id"]))
            if m is None:
                continue
            if p["gate"] is not None and not (1 / p["gate"] <= m / o["dist"] <= p["gate"]):
                continue
            o["dist_kc"], o["dist_mg_ca"] = o["dist"], m
            o["dist"] = (1 - p["w"]) * o["dist"] + p["w"] * m
            n_avg += 1
    new = far_step(k2, P)
    nv = {(s["fi"], o["id"]): o for s in new["samples"] for o in s["objects"]}
    n_near = n_far = 0
    for s in out["samples"]:
        for o in s["objects"]:
            q = nv[(s["fi"], o["id"])]
            if q.get("dist") is None or o.get("dist") is None or abs(q["dist"] - o["dist"]) <= TOL_M:
                continue
            o["dist_v9"] = o["dist"]
            o["dist"] = q["dist"]
            for k in FAR_KEYS:
                if k in q:
                    o[k] = q[k]
                else:
                    o.pop(k, None)
            tagged = "D1" if "dist_mg_ca" in q and not o.get("far") else "D1-far-ruler"
            o.setdefault("v10_steps", []).append(tagged)
            n_near += tagged == "D1"
            n_far += tagged != "D1"
    info.update(applied=True, camera_A=A, readings_with_marking=len(raw), readings_averaged=n_avg,
                readings_changed_near=n_near, readings_changed_far=n_far, far_ruler_v6=new.get("v6"))


# ------------------------------------------------------------------ C2: ego speed, local weighted straight line

def step_c2(out, group, P, info):
    p = P["C2"][group]
    ego = [e for e in out.get("ego", []) if e.get("kmh") is not None]
    if not ego:
        info.update(seconds=0)
        return
    sec = np.array([int(round(e["t0"])) for e in ego])
    if len(set(sec.tolist())) != len(sec):
        raise SystemExit("duplicate ego seconds")
    v = np.array([e["kmh"] for e in ego], float)
    ci = np.array([e["ci95_kmh"] for e in ego], float)
    new = np.empty(len(ego))
    for i in range(len(ego)):
        nb = np.where(np.abs(sec - sec[i]) <= p["h_s"])[0]
        u = (sec[nb] - sec[i]).astype(float)
        y = v[nb]
        wt = 1.0 / np.maximum(ci[nb], p["min_halfwidth_kmh"]) ** 2 if p["weighted"] else np.ones(len(nb))
        if len(nb) >= p["min_seconds_for_line"]:
            W = np.sqrt(wt)
            X = np.vstack([np.ones_like(u), u]).T * W[:, None]
            new[i] = float(np.linalg.lstsq(X, y * W, rcond=None)[0][0])
        else:
            new[i] = float(np.sum(wt * y) / np.sum(wt))
    for e, x in zip(ego, new):
        src = "both" if e.get("kmh_marking") is not None else "depth"
        f = p["range_factor"][src]
        e["kmh_v9"], e["sigma_kmh_v9"], e["ci95_kmh_v9"] = e["kmh"], e["sigma_kmh"], e["ci95_kmh"]
        e["kmh"] = float(x)
        e["sigma_kmh"] = f * e["sigma_kmh_v9"] if e.get("sigma_kmh_v9") is not None else None
        e["ci95_kmh"] = f * e["ci95_kmh_v9"]
        e["v10_source"] = src
        e["v10_steps"] = ["C2"]
    info.update(seconds=len(ego), h_s=p["h_s"], weighted=p["weighted"])


# ------------------------------------------------------------------ C4 / C5: relative speed

class Series:
    """One method's per-sample distances by track (key: dist_raw for the depth series, dist for marking)."""

    def __init__(self, cars, key, max_jump):
        self.t = [s["t"] for s in cars["samples"]]
        self.fi = [s["fi"] for s in cars["samples"]]
        self.max_jump = max_jump
        self.idx, self.by_id = {}, {}
        for j, s in enumerate(cars["samples"]):
            for o in s["objects"]:
                d = o.get(key) if o.get(key) is not None else o.get("dist")
                self.by_id.setdefault(o["id"], {})[j] = (d, bool(o["far"]))
                self.idx[(s["fi"], o["id"])] = j

    def fit(self, fi, tid, hw, dmax=None):
        """(slope m/s, se m/s, first frame, last frame) or None (= track_speeds' completeness rule)."""
        j = self.idx.get((fi, tid))
        if j is None:
            return None
        seq = self.by_id[tid]
        grp = [seq.get(i) for i in range(j - hw, j + hw + 1)]
        if any(g is None or g[0] is None for g in grp):
            return None
        if any(g[1] and (dmax is None or g[0] > dmax) for g in grp):
            return None
        dd = np.array([g[0] for g in grp], float)
        if np.any(np.abs(np.diff(dd)) / dd[:-1] > self.max_jump):
            return None
        t = np.array([self.t[i] for i in range(j - hw, j + hw + 1)])
        tc = t - t.mean()
        sxx = float(np.sum(tc ** 2))
        rel = float(np.sum(tc * (dd - dd.mean())) / sxx)
        icpt = float(dd.mean())
        resid = dd - (rel * tc + icpt)
        se = float(np.sqrt(np.sum(resid ** 2) / max(1, len(dd) - 2)) / np.sqrt(sxx))
        return rel, se, self.fi[j - hw], self.fi[j + hw]

    def longest(self, fi, tid, hws, dmax=None):
        best = None
        for hw in sorted(hws):
            f = self.fit(fi, tid, hw, dmax)
            if f is not None:
                best = (3.6 * f[0], 3.6 * f[1], hw, f[2], f[3])
        return best


def step_rel(out, src, group, dep, mrk, P, info):
    p4, p5 = P["C4"], P["C5"]
    fac = P["rel_range_factor"]
    zm = M_REL_Z * p4["m_rel"]
    Sd = Series(dep, "dist_raw", p4["max_jump"]) if usable(dep) else None
    Sm = Series(mrk, "dist", p4["max_jump"]) if (src == "current" and usable(mrk)) else None
    if src == "current":
        sc_d = float(dep.get("rel_scale_sigma", 0.0)) if usable(dep) else 0.0
    else:
        sc_d = float(dep.get("k_camera_sigma", 0.0)) if usable(dep) else 0.0
    sc_m = float(mrk.get("ruler_scale_sigma", 0.0)) if Sm is not None else 0.0
    c5 = group in p5["groups"] and p5["D_m"] is not None
    n = n_long = n_ext = n_bad = 0
    for s in out["samples"]:
        for o in s["objects"]:
            if o.get("rel_kmh") is None:
                continue
            n += 1
            fi, tid = s["fi"], o["id"]
            averaged = o.get("rel_kmh_marking") is not None
            o["rel_kmh_v9"], o["rel_sigma_kmh_v9"], o["rel_ci95_kmh_v9"] = o["rel_kmh"], o.get("rel_sigma_kmh"), o.get("rel_ci95_kmh")
            if "win" in o:
                o["win_v9"] = list(o["win"])
            # identity: the registered +-0.8 s fits reproduce the ninth version's value
            fd4 = Sd.longest(fi, tid, (4,)) if Sd else None
            fm4 = Sm.longest(fi, tid, (4,)) if (Sm and averaged) else None
            if averaged:
                ok = (fd4 is not None and fm4 is not None and abs(fd4[0] - o["rel_kmh_depth"]) <= TOL_KMH
                      and abs(fm4[0] - o["rel_kmh_marking"]) <= TOL_KMH)
            else:
                ok = fd4 is not None and abs(fd4[0] - o["rel_kmh"]) <= TOL_KMH
            if not ok:
                n_bad += 1
                if o.get("rel_ci95_kmh") is not None:
                    o["rel_ci95_kmh"] = fac * o["rel_ci95_kmh"]
                o["v10_steps"] = o.get("v10_steps", []) + ["rel-identity-failed-kept-v9"]
                continue
            fd = Sd.longest(fi, tid, p4["halfwidths_samples"])
            fm = Sm.longest(fi, tid, p4["halfwidths_samples"]) if (Sm and averaged) else None
            ext = False
            if c5 and not averaged and src == "current" and Sm is not None:
                fx = Sm.longest(fi, tid, p4["halfwidths_samples"], dmax=p5["D_m"])
                if fx is not None:
                    fm, ext = fx, True
            vd, sed = fd[0], fd[1]
            sd = float(np.hypot(sed, abs(vd) * sc_d))
            if fm is not None:
                vm = fm[0]
                sm = float(np.hypot(fm[1], abs(vm) * sc_m))
                v = 0.5 * vd + 0.5 * vm
                sig = float(np.sqrt(0.25 * sd ** 2 + 0.25 * sm ** 2))
                o["rel_kmh_depth"], o["rel_kmh_marking"] = vd, vm
                o["rel_hw_marking"] = fm[2]
                f0, f1 = min(fd[3], fm[3]), max(fd[4], fm[4])
            else:
                v, sig = vd, sd
                f0, f1 = fd[3], fd[4]
            o["rel_hw_depth"] = fd[2]
            o["rel_kmh"], o["rel_sigma_kmh"], o["rel_ci95_kmh"] = float(v), sig, fac * zm * sig
            o["win"] = [f0, f1]
            steps = []
            if fd[2] > 4 or (fm is not None and fm[2] > 4):
                steps.append("C4")
                n_long += 1
            if ext:
                o["rel_marking_extended_to_m"] = p5["D_m"]
                steps.append("C5")
                n_ext += 1
            steps.append("rel-range-factor")
            o["v10_steps"] = o.get("v10_steps", []) + steps
    info.update(readings=n, readings_longer_window=n_long, readings_marking_extended=n_ext, identity_failed=n_bad,
                c5_applied=c5, scale_sigma_depth=sc_d, scale_sigma_marking=sc_m)


def step_abs(out, info):
    ego = {int(round(e["t0"])): e for e in out.get("ego", []) if e.get("kmh") is not None}
    fps = out["fps"]
    n = n_v9 = 0
    for s in out["samples"]:
        for o in s["objects"]:
            if o.get("rel_kmh") is None:
                continue
            o["abs_kmh_v9"], o["abs_ci95_kmh_v9"] = o.get("abs_kmh"), o.get("abs_ci95_kmh")
            n_v9 += o.get("abs_kmh") is not None
            b = int(math.floor(s["fi"] / fps + 1e-9))
            e = ego.get(b)
            if e is None or o.get("rel_ci95_kmh") is None or e.get("ci95_kmh") is None:
                o["abs_kmh"] = None
                o.pop("abs_ci95_kmh", None)
                continue
            o["abs_kmh"] = float(e["kmh"] + o["rel_kmh"])
            o["abs_ci95_kmh"] = float(np.hypot(e["ci95_kmh"], o["rel_ci95_kmh"]))
            o["abs_ego_second"] = b
            o["v10_steps"] = o.get("v10_steps", []) + ["abs"]
            n += 1
    info.update(readings=n, readings_ninth_version=n_v9)


# ------------------------------------------------------------------ per segment

def process(tag, group, B, args, P, psha, car_a):
    base = B.load(f"{tag}{args.suffix}")
    if base is None:
        return None, dict(status="absent")
    out = copy.deepcopy(base)
    prov = dict(version="v10", method="ninth version + D1 + C2 + C4 + C5 (tools/v10_post.py)", params_file=args.params,
                params_sha256=psha, group=group, base_file=f"{tag}{args.suffix}")
    if base.get("status") == "refused":
        prov.update(status="refused (kept)", inputs={f"{tag}{args.suffix}": B.read[f"{tag}{args.suffix}"]})
        out["v10"] = prov
        return out, dict(status="refused")
    nc = base.get("next_candidate") or {}
    src = nc.get("source", "current")
    if src not in ("current", "camera k day", "camera k night"):
        raise SystemExit(f"{tag}: unknown source {src!r}")
    names = [f"{tag}{args.suffix}"]
    steps = {}
    if src.startswith("camera k"):
        kname = f"{tag}_cars_{'kc' if nc.get('day') else 'kn'}.json"
        kc = B.load(kname)
        run = B.load(f"{tag}_run.json")
        names += [kname, f"{tag}_run.json"]
        if not usable(kc):
            raise SystemExit(f"{tag}: camera-k path without a usable {kname}")
        steps["D1"] = {}
        step_d1(out, kc, run, camera_A(car_a, tag), P, steps["D1"])
        dep, mrk = kc, None
    else:
        steps["D1"] = dict(applied=False, why="not a camera-k segment")
        dep = B.load(f"{tag}_cars_v4h.json")
        mrk = B.load(f"{tag}{args.marking_suffix}")
        names += [f"{tag}_cars_v4h.json"] + ([f"{tag}{args.marking_suffix}"] if mrk is not None else [])
    steps["C2"] = {}
    step_c2(out, group, P, steps["C2"])
    steps["C4_C5"] = {}
    step_rel(out, src, group, dep, mrk, P, steps["C4_C5"])
    steps["abs"] = {}
    step_abs(out, steps["abs"])
    prov.update(path=src, steps=steps, inputs={k: B.read[k] for k in names if k in B.read})
    out["v10"] = prov
    return out, dict(status="ok", path=src, d1=steps["D1"].get("applied"), rel_identity_failed=steps["C4_C5"]["identity_failed"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True, help="folder with the per-segment files")
    ap.add_argument("--out", required=True, help="folder for <tag>_cars_v10.json and v10_run.json")
    ap.add_argument("--suffix", default="_cars_v9.json", help="input version file (development: _cars_v7.json)")
    ap.add_argument("--marking-suffix", default="_cars_cca.json", help="marking cars of the combine step (v7: _cars_c.json)")
    ap.add_argument("--tags-dir", default=None, help="folder of tags_{main,local_day,night}.txt (default: --batch)")
    ap.add_argument("--car-a", default=None, help="the camera A (default: <batch>/car_a.json)")
    ap.add_argument("--params", default=str(Path(__file__).resolve().parent / "v10_params.json"))
    ap.add_argument("--seal", default=None, help="check every per-segment file against this seal (files by name)")
    args = ap.parse_args()
    bdir, odir = Path(args.batch), Path(args.out)
    tdir = Path(args.tags_dir) if args.tags_dir else bdir
    craw = Path(args.car_a or bdir / "car_a.json").read_bytes()
    praw = Path(args.params).read_bytes()
    P = json.loads(praw)["values"]
    if P is None:
        raise SystemExit("the parameters file has no values")
    psha = sha256(praw)
    car_a = json.loads(craw)
    B = Batch(bdir, Path(args.seal) if args.seal else None)
    groups, gfiles = {}, {}
    for g in GROUPS:
        p = tdir / f"tags_{g}.txt"
        raw = p.read_bytes()
        gfiles[p.name] = sha256(raw)
        for line in raw.decode().splitlines():
            t = line.strip()
            if t and not t.startswith("#"):
                if t in groups:
                    raise SystemExit(f"{t} in two groups")
                groups[t] = g
    rl = bdir / "run_list.txt"
    if rl.exists():
        listed = {"_".join(l.split("|")[:3]) for l in rl.read_text().splitlines() if l.strip()}
        if listed != set(groups):
            raise SystemExit(f"run_list.txt and the group lists differ: {sorted(listed ^ set(groups))[:5]}")
    odir.mkdir(parents=True, exist_ok=True)
    summary = dict(version="v10", batch=str(bdir), suffix=args.suffix, marking_suffix=args.marking_suffix,
                   params_file=args.params, params_sha256=psha, car_a=str(args.car_a or bdir / "car_a.json"),
                   car_a_sha256=sha256(craw), group_lists=gfiles, seal=args.seal, segments={})
    for tag in sorted(groups):
        res, row = process(tag, groups[tag], B, args, P, psha, car_a)
        if res is not None:
            body = json.dumps(res, indent=1)
            (odir / f"{tag}_cars_v10.json").write_text(body)
            row["sha256"] = sha256(body.encode())
        summary["segments"][tag] = dict(group=groups[tag], **row)
    summary["inputs_read"] = dict(sorted(B.read.items()))
    cnt = defaultdict(int)
    for r in summary["segments"].values():
        cnt[(r["status"], r.get("path"), r.get("d1"))] += 1
    summary["counts"] = {f"{k[0]}|{k[1]}|D1 {k[2]}": v for k, v in sorted(cnt.items(), key=str)}
    (odir / "v10_run.json").write_text(json.dumps(summary, indent=1))
    bad = sum(r.get("rel_identity_failed", 0) for r in summary["segments"].values())
    print(f"wrote {sum(1 for r in summary['segments'].values() if 'sha256' in r)} files to {odir}; {summary['counts']}; "
          f"relative-speed identity failures {bad}")


if __name__ == "__main__":
    main()

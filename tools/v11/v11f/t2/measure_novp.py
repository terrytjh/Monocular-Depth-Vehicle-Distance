#!/usr/bin/env python3
"""[v11f copy, 2026-10-24: unchanged except that t2lib takes its root and batch definitions from the environment]
T2 (b): car readings on the measurement sample frames that had no DA3 map ("no-vp frames"), added to the tenth-version
output of each segment without changing anything that is already there.

Why the frames had no map: depth_dash_scale.run computes (and caches) depth only at frames whose nearest lane-line window
has a vp_row; depth_dash_multicar.py measure reads depth only from that cache and gives no car on a frame without a map.
After tools/prefill_da3.py has written the missing maps into a new cache, this step reads each new frame exactly as
cmd_measure would (same box filter with --drop-hood, the depth 6 % of the box height above its bottom at the box centre,
5 x 5 median of the half-resolution map) and converts it with what the segment already has:

  path "current" (the segment's own ruler; ground calibration):
     d_raw = a z + b with the hood measurement's ground fit of THIS sample (fitted over the road rows of the sampled frames
     within +-2.5 s, which the measurement computed for every sample, with or without a map) when it was accepted
     (--fit own, default: exactly what the measurement would have done had the map existed but given no road rows);
     --fit nearest: else the accepted fit of the nearest sample within --fit-max-gap-s (flagged with the gap);
     --camk-fallback on: else the camera's k (k_camera.json by day / k_night.json at night, not on local roads, as
     next_combine's fallback), d_raw = k_cam z (flagged).
  path "camera k day / night" (the segment reads every car with the camera k): d_raw = k_camera z (the kc / kn file's k).
  Then, as the registered chain would for that reading (pipeline of the batch, values frozen from the existing files):
     ts16  Theil-Sen along the track over +-4 samples that have a raw reading (existing dist_raw + new), >= 3, else raw;
           far = d_raw > 30 m (the ruler's reach);
     S5    (ninth-version chain, not on daytime local roads) inside the reach, the mean with the camera-A marking distance of
           the same car in the same sample (*_cars_cca.json);
     far   the far ruler (own-car K from the track's 10-25 m readings, else the same-class K from >= 3 cars; K, K_class
           computed from the existing readings only, exactly as next_combine / v10 computed them);
     S6    (ninth-version chain, daytime non-local) the depth half of an averaged reading from MoGe-2 (*_cars_v5m.json);
     S8    (ninth-version chain, daytime local) the in-reach distance from MoGe-2 (*_cars_v5m8.json);
     D1    (camera-k path, when v10 applied it) inside the reach 0.25 d + 0.75 A_cam / (y_bottom - vp_row) with the marking
           run's nearest kept window within 1 s, the marking half Theil-Sen smoothed along the track (+-4 samples, >= 3) over
           the existing and new marking values; far readings with the far ruler of v10's D1 step (K from its averaged
           near readings).
  New readings carry no relative / absolute speed (the existing speeds are untouched), lane None on a frame without a
  vp_row (cmd_measure's own rule), and "t2_novp": {ruler, fit_fi, fit_gap_s, vp_reason, steps}.

Every existing reading, ego speed and field is kept byte for byte (checked on writing); refused segments stay refused.
Identity test (--identity-test N, CPU only, no new map needed): N sample frames per segment that HAVE a map and readings
are treated as new: their readings are removed from the existing sets, read again through the new cache's links and this
code, and must come back with the same distance (and far flag, raw distance, lane) -- this checks the whole chain.

  python -B measure_novp.py --batch C7b --cache-root <new cache root> --out <dir> [--fit own|nearest] [--camk-fallback on]
  python -B measure_novp.py --batch C7b --cache-root <root> --out <dir> --identity-test 3
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2lib as T  # noqa: E402

import depth_dash_multicar as M  # noqa: E402  (repository tools: pure functions only)
_V11X_GL = __import__("os").environ.get("V11X_GL") == "1"   # v11extra (a) GL switch (off: unchanged)
import depth_dash_scale as D  # noqa: E402
import v10_post as V10  # noqa: E402
import v6_far_ruler as V6  # noqa: E402

P10 = json.loads((T.TOOLS / "v10_params.json").read_text())["values"]
REACH = M.GROUND_RULER["d_max"]           # 30 m: the ground ruler's reach (and the camera k's, PARAMS k_camera_reach)
HALF, MIN_N = M.DIST_SMOOTH["ts16"]["half"], M.DIST_SMOOTH["ts16"]["min_n"]
WIDTH = 1164                              # next_combine WIDTH / v10 far_ruler frame_width
TOL = 1e-9


# ------------------------------------------------------------------ inputs

def load_seg(batch, tag, group, road, day):
    B = T.BATCHES[batch]
    F = Path(B["folder"])
    S = dict(batch=batch, tag=tag, group=group, road=road, day=day, pipeline=B["pipeline"])
    S["final_path"] = Path(B["v10"]) / f"{tag}_cars_v10.json"
    S["final_raw"] = S["final_path"].read_bytes() if S["final_path"].exists() else None
    S["final"] = json.loads(S["final_raw"]) if S["final_raw"] else None
    S["src"] = T.source_of(S["final"])
    S["inputs"] = {}

    def ld(name, folder=F):
        p = Path(folder) / name
        x = T.jload(p, batch)
        if x is not None:
            S["inputs"][str(p)] = T.sha_file(p)
        return x

    S["dash"] = ld(f"{tag}_dash.json")
    S["run"] = ld(f"{tag}_run.json")
    S["tracks"] = (ld(f"{tag}_tracks.json") or {}).get("frames", {})
    if S["src"] == "refused":
        return S
    S["fps"] = float(S["final"]["fps"])
    if S["src"] == "current":
        S["depth"] = ld(f"{tag}_cars_v4h.json")                      # hood measurement (ground fits, dist_raw)
        # far-ruler K: v9 = next_combine's far step on work/ca_reldist, whose near readings *_cars_v9a.json keeps unchanged;
        # v7 = the far step on work/hood_rel, whose distances are the hood measurement's
        S["kfile"] = S["depth"] if B["pipeline"] == "v7" else ld(f"{tag}_cars_v9a.json")
        S["s5"] = None
        if B["pipeline"] == "v9" and not (road == "local" and day):
            mk = ld(f"{tag}{B['msfx']}")
            S["s5"] = mk if mk is not None and mk.get("status") != "refused" else None
        S["s6"] = S["s8"] = None
        if B["pipeline"] == "v9":
            m6, m8 = ld(f"{tag}_cars_v5m.json"), ld(f"{tag}_cars_v5m8.json")
            S["s6"] = m6 if m6 is not None and m6.get("status") != "refused" else None
            S["s8"] = m8 if m8 is not None and m8.get("status") != "refused" else None
        kd, kn = ld("k_camera.json"), ld("k_night.json")
        kc = ((kd if day else kn) or {}).get("segments", {}).get(tag)
        S["k_fallback"] = None if (kc is None or road == "local") else float(kc["k_camera"])
    else:
        nc = S["final"].get("next_candidate") or {}
        S["depth"] = ld(f"{tag}_cars_{'kc' if nc.get('day') else 'kn'}.json")
        S["k_cam"] = float(S["depth"]["k_camera"])
        S["reach_cam"] = float(S["depth"].get("k_camera_reach") or REACH)
        d1 = ((S["final"].get("v10") or {}).get("steps") or {}).get("D1") or {}
        S["d1_applied"] = bool(d1.get("applied"))
        car_a = json.loads(Path(B["car_a"]).read_text())
        S["inputs"][str(B["car_a"])] = T.sha_file(B["car_a"])
        S["A_cam"] = V10.camera_A(car_a, tag)
        if S["d1_applied"] and abs(S["A_cam"] - d1["camera_A"]) > 1e-9:
            raise SystemExit(f"{tag}: camera A {S['A_cam']} differs from v10's {d1['camera_A']}")
        S["kfile"] = S["depth"]
    return S


# ------------------------------------------------------------------ helpers (each = the registered tool's rule)

def usable(b):
    return b[0] > V6.V6["edge_px"] and b[2] < WIDTH - V6.V6["edge_px"]


def far_rulers(cars):
    """(K per track id, K per class) = v6_far_ruler.combine's own-car and same-class rulers (class_min_tracks 3)."""
    near = defaultdict(list)
    for s in cars["samples"]:
        for o in s["objects"]:
            d = o.get("dist")
            if d is not None and not o.get("far") and V6.V6["near_m"][0] <= d <= V6.V6["near_m"][1] and usable(o["box"]):
                near[o["id"]].append(d * (o["box"][2] - o["box"][0]))
    K = {i: float(np.median(v)) for i, v in near.items() if len(v) >= V6.V6["min_near"]}
    cls = {o["id"]: o.get("cls") for s in cars["samples"] for o in s["objects"]}
    byc = defaultdict(list)
    for i, k in K.items():
        byc[cls.get(i)].append(k)
    Kc = {c: float(np.median(v)) for c, v in byc.items() if c is not None and len(v) >= P10["D1"]["far_ruler"]["class_min_tracks"]}
    return K, Kc


def apply_far(o, K, Kc):
    """v6_far_ruler.combine for one reading flagged far (apply_from None); returns the step name or None."""
    if o.get("dist") is None or not o.get("far"):
        return None
    if not usable(o["box"]) or (o["id"] not in K and o.get("cls") not in Kc):
        return None
    w = o["box"][2] - o["box"][0]
    if o["id"] in K:
        size = K[o["id"]] / w
        o["dist_v5"], o["dist_own_car"] = o["dist"], size
        o["far_ruler"] = "own-car"
    else:
        size = Kc[o["cls"]] / w
        o["dist_v5"], o["dist_class"] = o["dist"], size
        o["far_ruler"] = "same-class"
    o["dist"] = float(np.sqrt(o["dist"] * size))
    return "far-" + o["far_ruler"]


def obj_map(cars, key="dist", near_only=False):
    out = {}
    for s in (cars or {}).get("samples", []):
        for o in s.get("objects", []):
            if o.get(key) is not None and not (near_only and o.get("far")):
                out[(s["fi"], o["id"])] = o[key]
    return out


def ts_at(samples_t, seq, j):
    """Theil-Sen value at sample j over the track's values seq {sample index: value} within +-HALF (>= MIN_N), else seq[j]."""
    nb = [k for k in range(j - HALF, j + HALF + 1) if k in seq]
    if len(nb) < MIN_N:
        return seq[j]
    tt = np.array([samples_t[k] for k in nb]) - samples_t[j]
    return M.theil_sen(tt, np.array([seq[k] for k in nb]))[1]


def read_z(depth, box, h, w):
    """= cmd_measure: depth 6 % of the box height above its bottom, at the box centre, 5 x 5 median (half resolution)."""
    dh, dw = depth.shape
    r = M.PARAMS["patch"] // 2
    x0, y0, x1, y1 = box[:4]
    yy = int((y1 - M.PARAMS["read_up"] * (y1 - y0)) * dh / h)
    xx = int((x0 + x1) / 2 * dw / w)
    return float(np.median(depth[max(0, yy - r): yy + r + 1, max(0, xx - r): xx + r + 1]))


# ------------------------------------------------------------------ one segment

def process(S, cache_seg: Path, args, new_set=None, removed=None):
    """Adds readings on the frames of new_set ({fi}) to a copy of the final file and returns (out, rows, info).
    removed: {(fi, id)} of existing readings that the identity test took out (their raw values are not used)."""
    fin = S["final"]
    out = copy.deepcopy(fin)
    info = dict(tag=S["tag"], group=S["group"], source=S["src"], new_frames=0, frames_without_map=0, boxes=0, readings=0,
                by_ruler=defaultdict(int), by_step=defaultdict(int), no_fit=0, nonpositive=0)
    if S["src"] == "refused" or not new_set:
        return out, [], info
    if getattr(args, "paths", "all") == "current" and S["src"] != "current":
        info["skipped_path"] = S["src"]                      # variant own_cur: no new reading on a camera-k segment
        return out, [], info
    removed = removed or set()
    frames = T.list_frames(S["batch"], S["tag"])
    import cv2
    h, w = cv2.imread(str(frames[0])).shape[:2]
    bon = T.bonnet_row(S["batch"], S["tag"], S["run"], frames)
    dash = S["dash"] or {}
    step = int(dash.get("step") or 2)
    wins = dash.get("windows") or []
    samples = out["samples"]
    j_of = {s["fi"]: j for j, s in enumerate(samples)}
    st = [s["t"] for s in samples]
    dep = S["depth"]
    dep_j = {s["fi"]: j for j, s in enumerate(dep["samples"])}
    if [s["fi"] for s in dep["samples"]] != [s["fi"] for s in samples]:
        raise SystemExit(f"{S['tag']}: sample grids of the depth measurement and the final file differ")
    # raw distances of the existing readings, by track (sample index -> raw)
    raw = defaultdict(dict)
    for j, s in enumerate(dep["samples"]):
        for o in s["objects"]:
            if (s["fi"], o["id"]) in removed:
                continue
            raw[o["id"]][j] = o.get("dist_raw", o["dist"])
    # rulers frozen from the existing readings
    if S["src"] == "current":
        K, Kc = far_rulers(S["kfile"])
        fits = [s.get("ground_fit") or {} for s in dep["samples"]]
        ok_j = [j for j, f in enumerate(fits) if f.get("ok")]
        s5 = obj_map(S["s5"]) if S["s5"] is not None else None
        s6 = obj_map(S["s6"], near_only=True) if S["s6"] is not None else None
        s8 = obj_map(S["s8"], near_only=True) if S["s8"] is not None else None
    else:
        p1 = P10["D1"]
        rw = (S["run"] or {}).get("windows", []) if S["run"] else []
        mraw = {}
        if S["d1_applied"]:
            for j, s in enumerate(dep["samples"]):
                for o in s["objects"]:
                    if (s["fi"], o["id"]) in removed:
                        continue
                    m = V10.mg_dist(rw, s["fi"], dep["fps"], S["A_cam"], o["box"], p1["geometry_max_s"], p1["min_rows_below_horizon"])
                    if m is not None:
                        mraw[(j, o["id"])] = m
            # v10's D1 k2 (from the full camera-k file) gives the far ruler of the averaged near readings
            full = {}
            for j, s in enumerate(dep["samples"]):
                for o in s["objects"]:
                    m = V10.mg_dist(rw, s["fi"], dep["fps"], S["A_cam"], o["box"], p1["geometry_max_s"], p1["min_rows_below_horizon"])
                    if m is not None:
                        full[(j, o["id"])] = m
            mg_full = V10.smooth_track_values(dep["samples"], full, p1["smooth_half_samples"], p1["smooth_min_n"]) if p1["smooth"] else full
            k2 = copy.deepcopy(dep)
            for j, s in enumerate(k2["samples"]):
                for o in s["objects"]:
                    if o.get("far") or o.get("dist") is None:
                        continue
                    m = mg_full.get((j, o["id"]))
                    if m is not None:
                        o["dist"] = (1 - p1["w"]) * o["dist"] + p1["w"] * m
            K, Kc = far_rulers(k2)
        else:
            K, Kc = far_rulers(dep)
    # pass 1: raw readings on the new frames
    pend = []
    for fi in sorted(new_set):
        j = j_of.get(fi)
        if j is None:
            continue
        stem = frames[fi].stem
        p = cache_seg / f"{stem}.npy"
        if not p.exists():
            info["frames_without_map"] += 1
            continue
        info["new_frames"] += 1
        cand = T.candidates(S["tracks"].get(stem, []), h, w, bon)
        if not cand:
            continue
        for _k in range(5):                    # v11extra: /mnt/d short reads under load; a re-read returns the same data
            try:
                depth = np.load(p).astype(np.float32)
                break
            except (ValueError, OSError):
                if _k == 4:
                    raise
                __import__("time").sleep(2.0)
        wd = D.lines_for(dict(windows=wins), fi / step) if wins else {}
        reason = T.vp_reason(wd) if wins else "no_windows"
        for x0, y0, x1, y1, conf, cls, tid in cand:
            info["boxes"] += 1
            z = read_z(depth, (x0, y0, x1, y1), h, w)
            prov = dict(vp_reason=reason)
            if S["src"] == "current":
                f = fits[dep_j[fi]]
                fs = None
                if _V11X_GL:                       # v11extra (a) GL: the side fit of the same sample for a car in lane -1 / +1
                    side = {-1: "left", 1: "right"}.get(M.lane_of((x0, y0, x1, y1), wd))
                    fs = (dep["samples"][dep_j[fi]].get(f"ground_fit_{side}") or {}) if side else None
                if f.get("ok") and fs and fs.get("ok"):
                    a_, b_, ruler, gap, ffi = fs["a"], fs["b"], f"own_{side}", 0.0, fi
                elif f.get("ok"):
                    a_, b_, ruler, gap, ffi = f["a"], f["b"], "own", 0.0, fi
                elif args.fit == "nearest" and ok_j:
                    jj = min(ok_j, key=lambda k: (abs(k - dep_j[fi]), k))
                    gap = abs(dep["samples"][jj]["fi"] - fi) / S["fps"]
                    if gap <= args.fit_max_gap_s:
                        a_, b_, ruler, ffi = fits[jj]["a"], fits[jj]["b"], "nearest", dep["samples"][jj]["fi"]
                    else:
                        a_ = None
                else:
                    a_ = None
                if a_ is None and args.camk_fallback == "on" and S["k_fallback"] is not None:
                    d = S["k_fallback"] * z
                    ruler, gap, ffi = "camera_k_fallback", None, None
                elif a_ is None:
                    info["no_fit"] += 1
                    continue
                else:
                    d = a_ * z + b_
                if not (np.isfinite(d) and d > 0):
                    info["nonpositive"] += 1
                    continue
                far = bool(d > REACH)
            else:
                d = S["k_cam"] * z
                ruler, gap, ffi = "camera_k_segment", None, None
                far = bool(d > S["reach_cam"])
                if not np.isfinite(d):
                    info["nonpositive"] += 1
                    continue
            prov.update(ruler=ruler, fit_fi=ffi, fit_gap_s=None if gap is None else round(gap, 3))
            o = dict(id=int(tid), box=[x0, y0, x1, y1], conf=conf, cls=int(cls), lane=M.lane_of((x0, y0, x1, y1), wd),
                     dist=float(d), far=far)
            pend.append((j, o, prov))
            raw[int(tid)][j] = float(d)
    # pass 2: ts16, then the chain
    rows = []
    for j, o, prov in pend:
        steps = ["T2-novp", prov["ruler"], "ts16"]
        o["dist_raw"] = o["dist"]
        o["dist"] = float(ts_at(st, raw[o["id"]], j))
        fi = samples[j]["fi"]
        if S["src"] == "current":
            if s5 is not None and not o["far"]:
                m = s5.get((fi, o["id"]))
                if m is not None:
                    o["dist_depth"], o["dist_marking"] = o["dist"], m
                    o["dist"] = 0.5 * (o["dist"] + m)
                    steps.append("S5")
            stp = apply_far(o, K, Kc)
            if stp:
                steps.append(stp)
            if s6 is not None and "dist_depth" in o and "dist_marking" in o:
                m = s6.get((fi, o["id"]))
                if m is not None:
                    o["dist_depth_da3"], o["dist_depth"] = o["dist_depth"], m
                    o["dist"] = 0.5 * (m + o["dist_marking"])
                    steps.append("S6")
            if s8 is not None and not o["far"] and "dist_marking" not in o:
                m = s8.get((fi, o["id"]))
                if m is not None:
                    o["dist_depth_da3"], o["dist"] = o["dist"], m
                    steps.append("S8")
        else:
            if S["d1_applied"]:
                m_new = V10.mg_dist(rw, fi, S["fps"], S["A_cam"], o["box"], p1["geometry_max_s"], p1["min_rows_below_horizon"])
                if m_new is not None:
                    mraw[(j, o["id"])] = m_new
            prov["_d1"] = True
        rows.append((j, o, prov, steps))
    if S["src"] != "current" and S["d1_applied"]:
        mseq = defaultdict(dict)
        for (jj, i), v in mraw.items():
            mseq[i][jj] = v
        for j, o, prov, steps in rows:
            if o["far"] or o.get("dist") is None:
                continue
            seq = mseq.get(o["id"], {})
            if j not in seq:
                continue
            m = float(ts_at(st, seq, j)) if p1["smooth"] else seq[j]
            o["dist_kc"], o["dist_mg_ca"] = o["dist"], m
            o["dist"] = (1 - p1["w"]) * o["dist"] + p1["w"] * m
            steps.append("D1")
    if S["src"] != "current":
        for j, o, prov, steps in rows:
            stp = apply_far(o, K, Kc)
            if stp:
                steps.append(stp)
    recs = []
    for j, o, prov, steps in rows:
        prov.pop("_d1", None)
        o["t2_novp"] = dict(prov, steps=steps)
        samples[j]["objects"].append(o)
        info["readings"] += 1
        info["by_ruler"][prov["ruler"]] += 1
        for sname in steps[3:]:
            info["by_step"][sname] += 1
        recs.append(dict(fi=samples[j]["fi"], id=o["id"], dist=o["dist"], dist_raw=o["dist_raw"], far=o["far"], lane=o["lane"],
                         ruler=prov["ruler"], fit_gap_s=prov["fit_gap_s"], vp_reason=prov["vp_reason"], steps="+".join(steps[3:])))
    info["by_ruler"], info["by_step"] = dict(info["by_ruler"]), dict(info["by_step"])
    return out, recs, info


def check_untouched(fin, out):
    """Every existing sample, object (all fields) and every other key of the final file is unchanged."""
    a = {k: v for k, v in fin.items() if k != "samples"}
    b = {k: v for k, v in out.items() if k not in ("samples", "t2")}
    if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
        return False
    if "samples" not in fin:                                # a refused segment: nothing else to compare
        return "samples" not in out
    for s0, s1 in zip(fin["samples"], out["samples"]):
        if {k: v for k, v in s0.items() if k != "objects"} != {k: v for k, v in s1.items() if k != "objects"}:
            return False
        old = s1["objects"][:len(s0["objects"])]
        if json.dumps(old) != json.dumps(s0["objects"]):
            return False
        if any("t2_novp" not in o for o in s1["objects"][len(s0["objects"]):]):
            return False
    return len(fin["samples"]) == len(out["samples"])


# ------------------------------------------------------------------ identity test

def identity_pick(S, cache_seg, n, seed, have=None):
    """Up to n sample frames with readings in the final file and a pre-existing map: a link of the new cache (have None),
    or a map of the original cache that existed at measurement (have = those stems; read-only use)."""
    fin = S["final"]
    frames = T.list_frames(S["batch"], S["tag"])
    good = []
    for s in fin["samples"]:
        if not s["objects"]:
            continue
        stem = frames[s["fi"]].stem
        if have is None and not os.path.islink(cache_seg / f"{stem}.npy"):
            continue
        if have is not None and stem not in have:
            continue
        good.append(s["fi"])
    rng = random.Random(seed + int(hashlib.sha256(S["tag"].encode()).hexdigest()[:8], 16))
    return sorted(rng.sample(good, min(n, len(good))))


def identity_compare(fin, out, picks):
    """Compare the re-added readings with the removed originals on the picked frames."""
    res = dict(frames=len(picks), orig=0, back=0, missing=0, extra=0, max_abs_dist=0.0, max_abs_raw=0.0, far_diff=0, lane_diff=0)
    by = {s["fi"]: s for s in out["samples"]}
    for s in fin["samples"]:
        if s["fi"] not in picks:
            continue
        orig = {o["id"]: o for o in s["objects"]}
        back = {o["id"]: o for o in by[s["fi"]]["objects"] if "t2_novp" in o}
        res["orig"] += len(orig)
        res["back"] += len(back)
        res["missing"] += len(set(orig) - set(back))
        res["extra"] += len(set(back) - set(orig))
        for i in set(orig) & set(back):
            a, b = orig[i], back[i]
            res["max_abs_dist"] = max(res["max_abs_dist"], abs(a["dist"] - b["dist"]))
            res["max_abs_raw"] = max(res["max_abs_raw"], abs(a.get("dist_raw", a["dist"]) - b["dist_raw"]))
            res["far_diff"] += a["far"] != b["far"]
            res["lane_diff"] += a.get("lane") != b.get("lane")
    return res


def strip_frames(S, picks):
    """The segment as if the picked frames had no map: their readings leave the final file (kept for comparison)."""
    S2 = dict(S)
    fin = copy.deepcopy(S["final"])
    removed = set()
    for s in fin["samples"]:
        if s["fi"] in picks:
            removed |= {(s["fi"], o["id"]) for o in s["objects"]}
            s["objects"] = []
    S2["final"] = fin
    return S2, removed


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True, choices=list(T.BATCHES))
    ap.add_argument("--cache-root", required=True, help="the new cache root of prefill_da3.py (<root>/<batch>/<tag>/)")
    ap.add_argument("--out", required=True, help="folder inside the track folder for <tag>_cars_v10nv.json")
    ap.add_argument("--fit", choices=("own", "nearest"), default="own")
    ap.add_argument("--fit-max-gap-s", type=float, default=2.5)
    ap.add_argument("--camk-fallback", choices=("off", "on"), default="off")
    ap.add_argument("--paths", choices=("all", "current"), default="all",
                    help="run3: 'current' = new readings only on segments read with their own ruler (not on the camera-k "
                         "fallback segments; variant own_cur)")
    ap.add_argument("--scope", choices=("cars", "all"), default="cars", help="the prefill's scope (its FRAMES_<scope>.txt)")
    ap.add_argument("--identity-test", type=int, default=0, help="CPU check: N frames per segment with maps, re-read")
    ap.add_argument("--identity-maps", choices=("links", "original"), default="links",
                    help="identity test: read the maps through the new cache's links, or from the batch's own cache (read only)")
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--max-segments", type=int, default=None)
    a = ap.parse_args()
    if not T.inside_t2(a.out):
        raise SystemExit(f"--out {a.out} is outside the track folder: refusing")
    if a.identity_test and (a.fit != "own" or a.camk_fallback != "off" or a.paths != "all"):
        raise SystemExit("the identity test checks the default rules only (--fit own, --camk-fallback off, --paths all)")
    os.nice(10)
    B = T.BATCHES[a.batch]
    G = T.tags_and_groups(a.batch)
    road, grey = T.road_of(a.batch), T.grey_of(a.batch)
    tags = sorted(G)
    if a.tags:
        tags = [t for t in tags if t in set(a.tags)]
    tags = tags[:a.max_segments]
    odir = Path(a.out)
    odir.mkdir(parents=True, exist_ok=True)
    root = Path(a.cache_root) / a.batch
    summary = dict(version="T2 no-vp readings (measure_novp.py)", batch=a.batch, fit=a.fit, fit_max_gap_s=a.fit_max_gap_s,
                   camk_fallback=a.camk_fallback, paths=a.paths, identity_test=a.identity_test, cache_root=str(Path(a.cache_root)),
                   script_sha256=T.sha_file(__file__), lib_sha256=T.sha_file(Path(__file__).with_name("t2lib.py")),
                   v10_params_sha256=T.sha_file(T.TOOLS / "v10_params.json"), started=T.now(), segments={})
    allrecs = []
    for tag in tags:
        T.wait_memory(2500)
        S = load_seg(a.batch, tag, G[tag], road.get(tag, "?"), T.is_day(grey, tag))
        cseg = root / tag
        if S["final"] is None:
            summary["segments"][tag] = dict(status="no final file")
            continue
        if a.identity_test:
            have = None
            if a.identity_maps == "original":
                cseg = Path(B["da3"]) / tag                          # read-only: np.load of maps that existed at measurement
                have = T.maps_at_measurement(a.batch, tag)[0]
            if S["src"] == "refused" or not cseg.is_dir():
                summary["segments"][tag] = dict(status="skipped", source=S["src"])
                continue
            picks = identity_pick(S, cseg, a.identity_test, a.seed, have)
            S2, removed = strip_frames(S, set(picks))
            out, recs, info = process(S2, cseg, a, set(picks), removed)
            cmp_ = identity_compare(S["final"], out, set(picks))
            ok = (cmp_["missing"] == 0 and cmp_["extra"] == 0 and cmp_["max_abs_dist"] <= TOL and cmp_["max_abs_raw"] <= TOL
                  and cmp_["far_diff"] == 0 and cmp_["lane_diff"] == 0)
            summary["segments"][tag] = dict(status="identity " + ("ok" if ok else "FAILED"), source=S["src"], picks=picks,
                                            **cmp_, info=info)
            (odir / f"{tag}_cars_v10id.json").write_text(json.dumps(out))
            for r in recs:
                allrecs.append(dict(tag=tag, group=G[tag], source=S["src"], **r))
            continue
        have, late, _ = T.maps_at_measurement(a.batch, tag)
        frames = T.list_frames(a.batch, tag)
        every = T.sample_every(float((S["dash"] or {}).get("fps") or 20.0))
        new = {fi for fi in range(0, len(frames), every) if frames[fi].stem not in have
               and (cseg / f"{frames[fi].stem}.npy").exists() and not os.path.islink(cseg / f"{frames[fi].stem}.npy")}
        out, recs, info = process(S, cseg, a, new)
        planned = root / f"FRAMES_{a.scope}.txt"
        if planned.exists():                                          # frames the prefill planned but did not write
            want = {int(l.split()[1]) for l in planned.read_text().splitlines() if l.startswith(tag + " ")}
            info["planned"] = len(want)
            info["planned_without_map"] = sum(1 for fi in want if fi not in new)
        if S["final"] is not None and not check_untouched(S["final"], out):
            raise SystemExit(f"{tag}: an existing reading or field changed -- refusing to write")
        out["t2"] = dict(step="T2 no-vp readings", fit=a.fit, fit_max_gap_s=a.fit_max_gap_s, camk_fallback=a.camk_fallback, paths=a.paths,
                         base=str(S["final_path"]), base_sha256=T.sha256b(S["final_raw"]), inputs=S["inputs"], counts=info)
        (odir / f"{tag}_cars_v10nv.json").write_text(json.dumps(out))
        summary["segments"][tag] = dict(status="ok", **info)
        for r in recs:
            allrecs.append(dict(tag=tag, group=G[tag], source=S["src"], **r))
    summary["ended"] = T.now()
    name = "identity" if a.identity_test else "run"
    (odir / f"novp_{name}.json").write_text(json.dumps(summary, indent=1, default=str))
    import csv
    cols = ["tag", "group", "source", "fi", "id", "dist", "dist_raw", "far", "lane", "ruler", "fit_gap_s", "vp_reason", "steps"]
    with open(odir / f"novp_{name}_readings.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r in allrecs:
            wr.writerow({k: r.get(k) for k in cols})
    segs = summary["segments"].values()
    if a.identity_test:
        okn = sum(1 for s in segs if s["status"] == "identity ok")
        bad = [t for t, s in summary["segments"].items() if s["status"] == "identity FAILED"]
        tot = sum(s.get("orig", 0) for s in segs)
        mx = max([s.get("max_abs_dist", 0.0) for s in segs] or [0.0])
        print(f"{a.batch} identity test: {okn} segments ok, {len(bad)} failed {bad[:5]}; readings re-read {tot}; "
              f"max |d dist| {mx:.3g} m", flush=True)
    else:
        n = sum(s.get("readings", 0) for s in segs)
        print(f"{a.batch}: {len(tags)} segments, {n} new readings", flush=True)


if __name__ == "__main__":
    main()

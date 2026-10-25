#!/usr/bin/env python3
"""Eleventh version, final build (v11f, 2026-10-24): one post-processing step on the tenth version's per-segment output
files that applies every decision Terry took on 2026-10-24 (development only so far; not registered).

No truth is read (every path containing "c2k19_truth" is refused). No depth model, no detector, no GPU (torch,
ultralytics and the depth-model wrappers are blocked at import). Deterministic. Every input file's sha256 is recorded.

Inputs (all predictions of the batch, never truth): a batch definition JSON (--batch-def: batch folder, group lists, ninth-
version suffix, marking suffix, camera-A file, seal, tenth-version folder, frames root, T2 folders, aux folders), the
parameters file (v11f_params.json: values + where each came from) and the per-camera hood contour file.

Tiers, decision 1 (no whole-segment refusal; every reading carries tier, tier_class, tier_label = tier_class:group):
  t1   own dash ruler + marking geometry (the tenth version's current path);
  t2k  same-camera constants: the tenth version's camera-k path (camera k + D1 average with the camera A);
  t2f  segments the tenth version refuses, in EVERY group: the camera-k measurement (<tag>_cars_kc.json by day, _kn.json at
       night; grey level >= 50 = day) through the tenth version's camera-k branch (V6 combine, D1, C2, C4/C5, abs) with D1
       weight 0.6 on daytime local roads (track T4) and 0.75 otherwise (= the tenth version);
  t3   licence-plate ruler: placeholder, never gives a reading;
  t4d  DA3 metric with the nominal focal ("uncalibrated": z at the registered read point, = camera-k dist_raw / k_camera)
       only for a segment that has neither t1 / t2k nor a usable camera-k measurement (segment level).
  Each tier class x group x read-distance bin has its own 95 % range (asymmetric by default, symmetric beside it).
Other steps, in this order:
  T2   (decision 6) when the batch definition names T2 folders: the tenth-version file with T2's own_cur readings (sample
       frames without a DA3 map at measurement, read from the new every-frame maps; own-ruler segments only) and the
       relative / absolute speeds those readings make possible (rel_novp.py) replaces the tenth-version file as the base
       (checked: its base_sha256 is the tenth-version file read here).
  hood (decision 7) per-camera pooled hood contour (comma2k19: >= 20 development segments of the same camera period):
       box centre at or below the contour -> not a car (the object is removed, listed in v11f.removed_not_car);
       box bottom within tol_px of the contour (lowest contour row over the box's columns), of a masked strip's top or
       of the frame bottom -> the box is kept with its speeds, no distance (dist_v10 keeps it, "hood" says why).
       The measurement's own bonnet rule (bottom >= 0.97 h and width > 0.8 w, --drop-hood) stays as it is.
  K4   (decision 4) covered >= 0.30 by a nearer box (bottom lower by > 5 px) or within 3 px of the left / right edge:
       no distance, speeds kept (tools/v11_post.step_k4 with the development-only leave-one-batch thresholds).
  rel  (decision 5) "simple": main-group own-ruler readings with a tenth-version relative speed take relsimple.py's
       r_up25_reg_mg_0.5 (read 25 % up the box, registered calibration, MoGe-2 where the segment has it, mean with marking
       0.5) after the identity column r_reg_reg_da3_0.5 reproduces the tenth version's value (<= 1e-6 km/h); the 95 %
       half-width is kept.
  ego  (decision 3) segments without any tenth-version ego second: main group MH (track T5: marking pair speeds with the
       camera A, run passed -> MS, run refused -> MC re-run; >= 3 pairs per second; C2 of the main group, h 6 s,
       weighted; fixed 95 % half-width from the range table); night K2-ego (camera-k night) and E1 (own ruler, dash ruler
       refused); local_day E1 (tools/v11_post rules; night range factors = the conservative ones by default).
  abs  absolute speed = ego of the reading's second + relative speed, 95 % half-width = hypot of the two, recomputed for
       every reading (identical to the tenth version where neither changed); new ego seconds get absolute speeds too.
  > 60 m  output, flagged far60; <= 0 m blanked (dist_le0).
  2026-10-24 (Terry's answers): T2 readings carry tier class t1n (own ruler, no lane pair at that frame) and use their own
       95 % range cells in the daytime-local and night groups (the t1 cells in the main group); every range of the
       parameters file is widened until every leave-one-batch fold covers >= 95 % (asymmetric distance limits, lower and
       upper separately); the daytime-local E1 ego factor and the MH half-width likewise (--local-range).

  python3 -B v11f_post.py tier4 --batch-def B.json --out AUX [--force-cache TAG ...]
  python3 -B v11f_post.py mc    --batch-def B.json --out AUX
  python3 -B v11f_post.py post  --batch-def B.json --aux AUX --out DIR [--params v11f_params.json]
                                [--t2 on|off] [--rel simple|off] [--night-range conservative|t1] [--k4 lobo1pp|draft]
                                [--hood on|off] [--range asym|sym] [--no-camera-constants]
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

for _m in ("torch", "ultralytics", "depth_backends", "moge_backend"):
    sys.modules[_m] = None          # any import of these raises ImportError: no depth model, no detector, no GPU
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

REPO = Path("/home/terry/Monocular-Depth-Vehicle-Distance")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "tools"))

import os as _os_nv11
if _os_nv11.environ.get("NV11_TOOLS"):
    sys.path.insert(0, _os_nv11.environ["NV11_TOOLS"])   # night_v11: arm override (patched depth_dash_multicar.py only)
import v10_post as V10  # noqa: E402
import v11_post as V11  # noqa: E402

V6 = V10.V6
sha256 = V10.sha256
VERSION = "v11f"
GROUPS = ("main", "local_day", "night")
BINS = (("le30", 0.0, 30.0), ("30-60", 30.0, 60.0), ("60-100", 60.0, 100.0), ("gt100", 100.0, math.inf))


def _rb_retry(p):                                   # /mnt/d returns ENOMEM / short reads under load; a re-read returns the same bytes
    for _k in range(5):
        try:
            return Path(p).read_bytes()
        except OSError:
            if _k == 4:
                raise
            __import__("time").sleep(2.0)


def refuse_truth(p):
    if "c2k19_truth" in str(Path(p).resolve()):
        raise SystemExit(f"{p}: truth path -- refusing")


def read_plain(p) -> bytes:
    refuse_truth(p)
    return _rb_retry(p)


def pred_bin(d):
    for name, lo, hi in BINS:
        if lo < d <= hi:
            return name
    return "le30" if d <= 0 else "gt100"


def group_lists(tdir: Path):
    groups, gfiles = {}, {}
    for g in GROUPS:
        p = Path(tdir) / f"tags_{g}.txt"
        raw = read_plain(p)
        gfiles[p.name] = sha256(raw)
        for line in raw.decode().splitlines():
            t = line.strip()
            if t and not t.startswith("#"):
                if t in groups:
                    raise SystemExit(f"{t} in two groups")
                groups[t] = g
    return groups, gfiles


def seal_input(bdir: Path, seal, name: str):
    raw = read_plain(Path(bdir) / name)
    if seal:
        h = (json.loads(Path(seal).read_text()).get("inputs") or {}).get(name)
        if h is not None and h != sha256(raw):
            raise SystemExit(f"{bdir / name}: differs from the seal inputs -- refusing")
    return raw


def load_bdef(p):
    raw = read_plain(p)
    return json.loads(raw), sha256(raw)


def load_retry(p: Path, tries=4):
    for k in range(tries):
        try:
            return np.load(p).astype(np.float32)
        except (ValueError, OSError):
            if k == tries - 1:
                raise
            time.sleep(2.0)


# ------------------------------------------------------------------ tier 4: the model's own metres (DA3, nominal focal)

def read_point(depth, box, h, w, read_up, patch):
    """= tools/depth_dash_multicar.py cmd_measure: 5 x 5 median at read_up of the box height above its bottom, centre."""
    dh, dw = depth.shape
    x0, y0, x1, y1 = box[:4]
    r = patch // 2
    yy = int((y1 - read_up * (y1 - y0)) * dh / h)
    xx = int((x0 + x1) / 2 * dw / w)
    v = depth[max(0, yy - r): yy + r + 1, max(0, xx - r): xx + r + 1]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else None


def tier4_from_cache(BD, tag, B, P):
    """Tier-4 rows of a segment WITHOUT a usable camera-k measurement: the measurement's box filter (cmd_measure, --drop-hood)
    on the detector's tracks at every sample frame, z read from the DA3 map (the T2 every-frame cache first, else the batch
    cache), u_da3 = z (the DA3 metric depth with the nominal focal). Lane is not set (no lane-line window is used)."""
    import cv2
    import depth_dash_multicar as M
    import depth_dash_scale as D
    tr = B.load(f"{tag}_tracks.json")
    run = B.load(f"{tag}_run.json")
    if tr is None:
        return None
    fd = Path(BD["frames_root"]) / tag / "frames"
    refuse_truth(fd)
    frames = D.list_frames(fd)
    h, w = cv2.imread(str(frames[0])).shape[:2]
    bon = int(run["bonnet_row"]) if run and run.get("bonnet_row") is not None else int(D.static_rows(frames)[0])
    dash = B.load(f"{tag}_dash.json") or {}
    fps = float(dash.get("fps") or (run or {}).get("fps") or 20.0)
    every = max(1, int(round(M.PARAMS["sample_s"] * fps)))
    roots = [Path(r) / tag for r in (BD.get("t2_cache"), BD.get("da3")) if r]
    rows, samples, n_map = {}, [], 0
    frames_tr = tr.get("frames", {})
    for fi in range(0, len(frames), every):
        stem = frames[fi].stem
        cand = [b for b in frames_tr.get(stem, []) if not M.is_hood(b, h, w) and b[6] >= 0 and b[4] >= M.PARAMS["min_conf"]
                and b[3] < bon - M.PARAMS["bonnet_px"]]
        cand = M.dedup(cand, M.PARAMS["dedup_iou"])
        samples.append(fi)
        if not cand:
            continue
        mp = next((r / f"{stem}.npy" for r in roots if (r / f"{stem}.npy").exists()), None)
        if mp is None:
            continue
        refuse_truth(mp)
        depth = load_retry(mp)
        n_map += 1
        for x0, y0, x1, y1, conf, cls, tid in cand:
            z = read_point(depth, (x0, y0, x1, y1), h, w, P["read_up"], P["patch"])
            rows[f"{fi}|{int(tid)}"] = dict(u_da3=z, box=[x0, y0, x1, y1], cls=int(cls), conf=conf, src="cache")
    return dict(rows=rows, samples=samples, fps=fps, n_frames=len(frames), maps_read=n_map,
                roots=[str(r) for r in roots])


def cmd_tier4(a):
    BD, bsha = load_bdef(a.batch_def)
    P = json.loads(read_plain(a.params))["values"]["tier4"]
    odir = Path(a.out) / "tier4"
    odir.mkdir(parents=True, exist_ok=True)
    B = V10.Batch(Path(BD["folder"]), Path(BD["seal"]) if BD.get("seal") else None)
    groups, _ = group_lists(BD["tags"])
    summ = dict(tool="v11f_post.py tier4", batch_def=a.batch_def, batch_def_sha256=bsha, params=P, segments={})
    for tag in sorted(groups):
        fo = odir / f"{tag}.json"
        if fo.exists() and not a.force:
            summ["segments"][tag] = dict(status="exists", sha256=sha256(fo.read_bytes()))
            continue
        res = dict(tag=tag, rows={}, files={}, source=None)
        if tag not in set(a.force_cache or []):
            for kname in (f"{tag}_cars_kc.json", f"{tag}_cars_kn.json"):
                kc = B.load(kname)
                if not V10.usable(kc):
                    continue
                k = float(kc["k_camera"])
                res["files"][kname] = dict(sha256=B.read[kname], k_camera=k)
                for s in kc["samples"]:
                    for o in s["objects"]:
                        key = f"{s['fi']}|{o['id']}"
                        if key in res["rows"]:
                            continue
                        raw = o.get("dist_raw", o.get("dist"))
                        res["rows"][key] = dict(u_da3=None if raw is None else float(raw) / k, box=o["box"][:4], src=kname,
                                                cls=o.get("cls"), conf=o.get("conf"))
                res["source"] = kname
                break
        if res["source"] is None:
            c = tier4_from_cache(BD, tag, B, P)
            if c is not None:
                res.update(rows=c["rows"], source="cache", cache=dict(samples=c["samples"], fps=c["fps"], n_frames=c["n_frames"],
                                                                      maps_read=c["maps_read"], roots=c["roots"]))
        body = json.dumps(res)
        fo.write_text(body)
        summ["segments"][tag] = dict(status="written", source=res["source"], rows=len(res["rows"]), sha256=sha256(body.encode()))
        print(f"tier4 {tag}: {len(res['rows'])} boxes from {res['source']}", flush=True)
    summ["inputs_read"] = dict(sorted(B.read.items()))
    (Path(a.out) / "tier4_run.json").write_text(json.dumps(summ, indent=1))


# ------------------------------------------------------------------ MC: marking's speed step re-run with the camera A (= v11m)

def camera_A_rel_sd(car_a, tag):
    cam, drive = tag.split("_")[0], tag.split("_")[1]
    c = (car_a.get("cameras") or {}).get(cam)
    if not c:
        return None
    v = np.array([x for d, x in c["drive_median_A"].items() if d != drive], float)
    return float(np.std(v, ddof=1) / np.median(v)) if len(v) >= 2 else 0.0


def find_mc(tag, dirs):
    for d in dirs:
        p = Path(d) / f"{tag}_camA.json"
        if p.exists():
            return p
    return None


def cmd_mc(a):
    sys.path.insert(0, str(HERE))
    import marking_camA as MCA
    import depth_dash_scale as D
    BD, _ = load_bdef(a.batch_def)
    B = V10.Batch(Path(BD["folder"]), Path(BD["seal"]) if BD.get("seal") else None)
    V = V10.Batch(Path(BD["v10"]), None)
    groups, _ = group_lists(BD["tags"])
    car_a = json.loads(read_plain(BD["car_a"]))
    od = Path(a.out) / "mc"
    od.mkdir(parents=True, exist_ok=True)
    root = Path(BD.get("run_frames_root") or REPO)
    for tag, g in sorted(groups.items()):
        if g != "main":
            continue
        c10 = V.load(f"{tag}_cars_v10.json")
        if c10 is None or any(e.get("kmh") is not None for e in c10.get("ego") or []):
            continue
        run = B.load(f"{tag}_run.json")
        if run is None or run.get("status") == "ok":
            continue
        if find_mc(tag, (BD.get("mc_cache") or []) + [str(od)]):
            continue
        A = V10.camera_A(car_a, tag)
        if A is None:
            continue
        fd = root / run["frames_dir"]
        refuse_truth(fd)
        frames = D.list_frames(fd)
        if len(frames) != run["n_frames"]:
            raise SystemExit(f"{fd}: {len(frames)} frames, run has {run['n_frames']}")
        t0 = time.time()
        sp = MCA.recompute(run, frames, A)
        res = dict(tag=tag, mode="camA", A_cam=A, run_status=run.get("status"), n_frames=run["n_frames"], fps=run["fps"],
                   step=run["step"], speeds=sp, seconds=round(time.time() - t0, 1), run_sha256=B.read[f"{tag}_run.json"],
                   made_by="v11f_post.py mc (marking_camA.recompute)")
        (od / f"{tag}_camA.json").write_text(json.dumps(res))
        print(f"mc {tag}: {len(sp)} pairs, {sum(1 for x in sp if x['kmh'] is not None)} kept, {res['seconds']} s", flush=True)


# ------------------------------------------------------------------ main-group ego MH (track T5; = v11m)

def bins_of(pairs, nb, min_n):
    by = {}
    for t_s, v in pairs:
        if v is None:
            continue
        b = int(math.floor(t_s))
        if 0 <= b < nb:
            by.setdefault(b, []).append(float(v))
    out = {}
    for b, v in by.items():
        if len(v) >= min_n:
            x = np.asarray(v)
            out[b] = (float(np.median(x)), int(len(x)), float(np.std(x, ddof=1)))
    return out


def mh_seconds(run, mc, A_cam, A_sd, pm):
    fps = run.get("fps") or 20.0
    nb = int(run["n_frames"] / fps // 1.0)
    if run.get("status") == "ok" and run.get("speeds"):
        rul = {int(x["frame"]): x for x in run.get("rulers") or []}
        side = []
        for x in run["speeds"]:
            if x.get("kmh") is None:
                continue
            r = rul.get(int(x["frame_i"]))
            if r is None:
                continue
            vs = [ps["kmh"] * A_cam / r["A_side"][sd] for sd, ps in (x.get("per_side") or {}).items()
                  if ps.get("kmh") is not None and (r.get("A_side") or {}).get(sd)]
            if vs:
                side.append((x["t_s"], float(np.mean(vs))))
        st, lab = bins_of(side, nb, pm["min_pairs"]), "MS"
    elif mc is not None:
        st, lab = bins_of([(x["t_s"], x["kmh"]) for x in mc["speeds"]], nb, pm["min_pairs"]), "MC"
    else:
        return None, {}
    out = {}
    for b, (med, n, sd) in st.items():
        sig = float(math.hypot(pm["median_se_factor"] * sd / math.sqrt(n), med * (A_sd or 0.0)))
        out[b] = (med, n, pm["z"] * pm["sigma_mult"] * sig, sig)
    return lab, out


def step_mh(out, run, mc, A_cam, A_sd, P10, pm, info):
    lab, secs = mh_seconds(run, mc, A_cam, A_sd, pm)
    info.update(source=lab, camera_A=A_cam, camera_A_rel_sd=A_sd)
    if not secs:
        info.update(applied=False, why="no second with enough marking speed pairs" if lab else "no marking speeds",
                    seconds_added=0)
        return 0
    ks = sorted(secs)
    tmp = dict(ego=[dict(t0=float(k), kmh=secs[k][0], sigma_kmh=None, ci95_kmh=secs[k][2]) for k in ks])
    Pc = {"C2": {"main": dict(P10["C2"]["main"], range_factor={"both": 1.0, "depth": 1.0})}}
    V10.step_c2(tmp, "main", Pc, {})
    hw = pm.get("hw_fixed_kmh")
    new = []
    for k, e in zip(ks, tmp["ego"]):
        raw, n, hw9, sig = secs[k]
        new.append(dict(t0=float(k), n=n, kmh=float(e["kmh"]), sigma_kmh=None if hw is None else hw / 1.96,
                        ci95_kmh=hw, kmh_v11_raw=raw, sigma_kmh_v11_raw=sig, ci95_kmh_v11_raw=hw9, camera_A=A_cam,
                        v11_source=f"marking, camera A, main-group fill ({lab})", v11_steps=[f"MH-{lab}", "C2-main", "range"],
                        tier=2))
    kept = [e for e in out.get("ego") or [] if int(round(e["t0"])) not in {int(k) for k in ks}]
    if any(e.get("kmh") is not None for e in kept):
        raise SystemExit("MH on a segment that has ego values")
    out["ego"] = sorted(kept + new, key=lambda e: float(e["t0"]))
    info.update(applied=True, seconds_added=len(new), hw_fixed_kmh=hw, negative_values=sum(1 for e in new if e["kmh"] < 0))
    return len(new)


# ------------------------------------------------------------------ tier-2 fill of refused segments, tier-4 fill

def fill_camk(kc, run, A, group, day, w, P10, info):
    """The tenth version's camera-k branch on the camera-k measurement (= track T4 t4lib.build_camk + v10_camk)."""
    v9 = V6.combine(kc, 1164, V6.V6, None, 3)
    src = "camera k " + ("day" if day else "night")
    v9["next_candidate"] = dict(source=src, day=day, k_camera=kc.get("k_camera"))
    P = copy.deepcopy(P10)
    P["D1"]["w"] = w
    out = copy.deepcopy(v9)
    for k, f in (("D1", lambda i: V10.step_d1(out, kc, run, A, P, i)), ("C2", lambda i: V10.step_c2(out, group, P, i)),
                 ("C4_C5", lambda i: V10.step_rel(out, src, group, kc, None, P, i)), ("abs", lambda i: V10.step_abs(out, i))):
        info[k] = {}
        f(info[k])
    return out, src


def fill_tier4(t4, base_k):
    """Tier-4 segment: every box of the tier-4 rows with its DA3 nominal-focal metres (no camera constant)."""
    rows = t4["rows"]
    if base_k is not None:
        fps, nf, grid = base_k["fps"], base_k["n_frames"], [(s["fi"], s["t"]) for s in base_k["samples"]]
    else:
        c = t4.get("cache") or {}
        fps, nf = c["fps"], c["n_frames"]
        grid = [(fi, fi / fps) for fi in c["samples"]]
    by = defaultdict(list)
    for key, r in rows.items():
        fi, tid = key.split("|")
        by[int(fi)].append((int(tid), r))
    out = dict(fps=fps, n_frames=nf, ego=[], samples=[], status="ok")
    for fi, t in grid:
        objs = []
        for tid, r in sorted(by.get(fi, [])):
            if r.get("u_da3") is None:
                continue
            objs.append(dict(id=tid, box=r["box"], conf=r.get("conf"), cls=r.get("cls"), lane=None, dist=float(r["u_da3"]),
                             far=False, tier4_model="da3"))
        out["samples"].append(dict(fi=fi, t=t, objects=objs))
    return out


# ------------------------------------------------------------------ hood contour (decision 7)

def hood_limit(contours, tag, width, height):
    """(limit per column, why per column, key) for the segment's camera: min(hood contour, masked-strip top, frame bottom)."""
    cam, drive = tag.split("_")[0], tag.split("_")[1]
    key = None
    for k, c in (contours.get("cameras") or {}).items():
        if c.get("dongle") == cam and c.get("drive_from", "") <= drive < c.get("drive_to", "9999"):
            key = k
            break
    hood = np.full(width, float(height))
    ov = np.full(width, float(height))
    if key is not None:
        c = contours["cameras"][key]
        if c.get("width") != width or c.get("height") != height:
            raise SystemExit(f"{tag}: contour {key} is for {c.get('width')} x {c.get('height')}, frames are {width} x {height}")
        hood = np.array([height if (v is None or not np.isfinite(v)) else v for v in c["y_hood"]], float)
        if c.get("overlay_top") is not None:
            ov = np.array(c["overlay_top"], float)
    lim = np.minimum(np.minimum(hood, ov), float(height))
    why = np.where(hood <= np.minimum(ov, height) - 0.5, "hood", np.where(ov < height - 0.5, "overlay", "bottom"))
    return lim, why, key


def classify(box, lim, why, tol):
    """= /mnt/d/depth_work/hood_20261024/scripts/box_effect.classify: below_* / touch_* / clear."""
    x0, y0, x1, y1 = box[:4]
    w = len(lim)
    a_, b_ = int(np.clip(np.floor(x0), 0, w - 1)), int(np.clip(np.ceil(x1), 0, w - 1))
    cx = int(np.clip(round((x0 + x1) / 2), 0, w - 1))
    if (y0 + y1) / 2 >= lim[cx]:
        return f"below_{why[cx]}"
    k = a_ + int(np.argmin(lim[a_:b_ + 1]))
    if y1 >= lim[k] - tol:
        return f"touch_{why[k]}"
    return "clear"


def step_hood(out, lim, why, tol, info):
    n = defaultdict(int)
    removed = []
    for s in out["samples"]:
        keep = []
        for o in s["objects"]:
            c = classify(o["box"], lim, why, tol)
            n[c] += 1
            if c == "below_hood":
                removed.append(dict(fi=s["fi"], id=o["id"], box=o["box"][:4], dist=o.get("dist")))
                continue
            if c != "clear":
                o["hood"] = c
                if o.get("dist") is not None:
                    o["dist_v10"] = o["dist"]
                    o["dist"] = None
                    o["v11_steps"] = o.get("v11_steps", []) + ["hood"]
                    n["blanked"] += 1
            keep.append(o)
        s["objects"] = keep
    info.update(dict(sorted(n.items())), removed_not_car=len(removed))
    return removed


# ------------------------------------------------------------------ ranges

def range_lookup(cells, tclass, group, b):
    if not cells:
        return None, None
    for key in (f"{tclass}|{group}|{b}", f"{tclass}|{group}|*", f"{tclass}|*|{b}", f"{tclass}|*|*"):
        if cells.get(key) is not None:
            return cells[key], key
    return None, None


def range_class(tc, group):
    """t1n readings (T2 frames) have their own 95 % range in the daytime-local and night groups (Terry, 2026-10-24); in the
    main group they share the t1 range."""
    if tc == "t1n":
        return "t1n" if group in ("local_day", "night") else "t1"
    return tc


def step_tiers(out, tclass_default, group, t4rows, R, primary, info):
    n = defaultdict(int)
    asym, sym = (R or {}).get("asym") or {}, (R or {}).get("sym") or {}
    for s in out["samples"]:
        for o in s["objects"]:
            r = t4rows.get(f"{s['fi']}|{o['id']}") or {}
            o["u_da3"] = r.get("u_da3")
            d = o.get("dist")
            if d is not None and d <= 0:
                o["dist_le0"] = d
                o["dist"] = None
                o["v11_steps"] = o.get("v11_steps", []) + ["le0-blank"]
                n["blank_le0"] += 1
                d = None
            tc = "t4d" if o.get("tier4_model") == "da3" else tclass_default
            if tc == "t1" and "t2_novp" in o:
                tc = "t1n"                      # own ruler, read on a T2 every-frame map (no lane pair at that frame)
            rc = range_class(tc, group)
            o["tier"] = int(tc[1])
            o["tier_class"] = tc
            o["tier_label"] = f"{tc}:{group}"
            o["dist_range_class"] = rc
            if tc.startswith("t4"):
                o["uncalibrated"] = True
            if "t2_novp" in o:
                o["every_frame_depth"] = True
            if d is None:
                n["no_distance"] += 1
                continue
            o["far60"] = bool(d > 60.0)
            b = pred_bin(d)
            qa, ka = range_lookup(asym, rc, group, b)
            hs, ks = range_lookup(sym, rc, group, b)
            o["dist_lo95_asym"] = None if qa is None else d * qa[0]
            o["dist_hi95_asym"] = None if qa is None else d * qa[1]
            o["dist_lo95_sym"] = None if hs is None else max(0.0, d * (1 - hs))
            o["dist_hi95_sym"] = None if hs is None else d * (1 + hs)
            p = "asym" if primary == "asym" else "sym"
            o["dist_lo95"], o["dist_hi95"] = o[f"dist_lo95_{p}"], o[f"dist_hi95_{p}"]
            o["dist_range_cell"] = ka if primary == "asym" else ks
            n[f"{tc}|{b}"] += 1
            n["far60"] += o["far60"]
            n["no_range"] += o["dist_lo95"] is None
    info.update(dict(sorted(n.items())))


# ------------------------------------------------------------------ relative speed "simple", absolute speed

def step_rel_simple(out, rows, info):
    ego_new = 0
    n = n_id_bad = n_nan = 0
    for s in out["samples"]:
        for o in s["objects"]:
            if o.get("rel_kmh") is None or "t2_rel" in o:
                continue
            r = rows.get((s["fi"], o["id"]))
            if r is None or r[1] is None:
                n_nan += 1
                continue
            if r[0] is None or abs(r[0] - o["rel_kmh"]) > 1e-6:
                n_id_bad += 1
                continue
            o["rel_kmh_v10"] = o["rel_kmh"]
            o["rel_kmh"] = float(r[1])
            o["rel_simple_mg"] = bool(r[2])
            o["v11_steps"] = o.get("v11_steps", []) + ["rel-simple"]
            n += 1
    info.update(changed=n, identity_failed_kept=n_id_bad, no_value_kept=n_nan, new_ego=ego_new)


def read_relsimple(p: Path):
    raw = read_plain(p)
    rows = {}
    for r in csv.DictReader(gzip.decompress(raw).decode().splitlines()):
        f = lambda x: None if x in (None, "") else float(x)
        rows[(int(r["fi"]), int(r["id"]))] = (f(r["r_reg_reg_da3_0.5"]), f(r["r_up25_reg_mg_0.5"]), int(r.get("mg_used") or 0))
    return rows, sha256(raw)


def rel_class(o, fill):
    """the source of a relative speed for its range: t2f (refused-segment fill), t2rel (made possible by T2 readings), v10."""
    if fill:
        return "t2f"
    return "t2rel" if "t2_rel" in o else "v10"


def step_range_mult(out, group, fill, M, info):
    rel, ego = M.get("rel") or {}, M.get("ego") or {}
    n = defaultdict(int)
    for s in out["samples"]:
        for o in s["objects"]:
            if o.get("rel_ci95_kmh") is None:
                continue
            c = rel_class(o, fill)
            o["rel_range_class"] = c
            m = rel.get(f"{group}|{c}")
            if m is not None and m != 1.0:
                o["rel_ci95_kmh_unwidened"] = o["rel_ci95_kmh"]
                o["rel_ci95_kmh"] = o["rel_ci95_kmh"] * m
                n[f"rel|{c}"] += 1
    m = ego.get(group)
    for e in out.get("ego") or []:
        if e.get("kmh") is None or e.get("ci95_kmh") is None or "v11_steps" in e:
            continue
        e["ego_range_class"] = "v10"
        if m is not None and m != 1.0:
            e["ci95_kmh_unwidened"] = e["ci95_kmh"]
            e["ci95_kmh"] = e["ci95_kmh"] * m
            if e.get("sigma_kmh") is not None:
                e["sigma_kmh"] = e["sigma_kmh"] * m
            n["ego|v10"] += 1
    info.update(dict(n), rel_mult={k: v for k, v in rel.items() if k.startswith(group + "|")}, ego_mult=m)


def step_abs_all(out, info):
    ego = {int(round(e["t0"])): e for e in out.get("ego", []) if e.get("kmh") is not None}
    fps = out["fps"]
    n = n_changed = n_new_ego = n_removed = 0
    for s in out["samples"]:
        for o in s["objects"]:
            if o.get("rel_kmh") is None:
                continue
            b = int(math.floor(s["fi"] / fps + 1e-9))
            e = ego.get(b)
            old, old_ci = o.get("abs_kmh"), o.get("abs_ci95_kmh")
            if e is None or o.get("rel_ci95_kmh") is None or e.get("ci95_kmh") is None:
                new, new_ci = None, None
            else:
                new = float(e["kmh"] + o["rel_kmh"])
                new_ci = float(np.hypot(e["ci95_kmh"], o["rel_ci95_kmh"]))
            if new is not None:
                n += 1
            if new != old or new_ci != old_ci:
                o["abs_kmh_v10"], o["abs_ci95_kmh_v10"] = old, old_ci
                o["abs_kmh"] = new
                if new_ci is None:
                    o.pop("abs_ci95_kmh", None)
                else:
                    o["abs_ci95_kmh"] = new_ci
                    o["abs_ego_second"] = b
                o["v11_steps"] = o.get("v11_steps", []) + ["abs"]
                n_changed += 1
                n_removed += new is None and old is not None
            if new is not None and "v11_steps" in e:
                o["abs_from_new_ego_second"] = True
                n_new_ego += 1
    info.update(readings=n, changed=n_changed, from_new_ego_seconds=n_new_ego, removed=n_removed)


# ------------------------------------------------------------------ per segment

def process(tag, group, ctx, a):
    V, B, P, P10, BD = ctx["V"], ctx["B"], ctx["P"], ctx["P10"], ctx["BD"]
    name10 = f"{tag}_cars_v10.json"
    c10 = V.load(name10)
    if c10 is None:
        return None, dict(status="absent")
    blk = c10.get("v10") or {}
    if blk.get("group") != group:
        raise SystemExit(f"{tag}: v10 group {blk.get('group')!r} vs lists {group!r}")
    if blk.get("params_sha256") != P["v10_params_sha256"]:
        raise SystemExit(f"{tag}: v10 parameters {blk.get('params_sha256')}")
    name9 = f"{tag}{BD['sfx']}"
    v9 = B.load(name9)
    if v9 is None:
        raise SystemExit(f"{tag}: no {name9}")
    if c10.get("status") != "refused" and (blk.get("inputs") or {}).get(name9) != B.read.get(name9):
        raise SystemExit(f"{tag}: the tenth-version file was not made from this batch's {name9}")
    steps = {}
    reads = {name10: V.read[name10], name9: B.read[name9]}
    # ---- T2 base
    base = c10
    t2_used = False
    if a.t2 == "on" and BD.get("t2_rel") and c10.get("status") != "refused":
        p2 = Path(BD["t2_rel"]) / f"{tag}_cars_v10nvr.json"
        if p2.exists():
            raw2 = read_plain(p2)
            t2 = json.loads(raw2)
            if (t2.get("t2") or {}).get("base_sha256") != V.read[name10]:
                raise SystemExit(f"{tag}: the T2 file was not made from this tenth-version file")
            if (t2.get("t2") or {}).get("paths") != "current":
                raise SystemExit(f"{tag}: the T2 file is not the own_cur variant")
            base = t2
            t2_used = True
            reads[f"t2/{p2.name}"] = sha256(raw2)
            steps["T2"] = dict(used=True, counts=(t2.get("t2") or {}).get("counts"),
                               new_readings=sum(1 for s in t2["samples"] for o in s["objects"] if "t2_novp" in o),
                               new_rel=sum(1 for s in t2["samples"] for o in s["objects"] if "t2_rel" in o))
        else:
            steps["T2"] = dict(used=False, why="no T2 file for this segment")
    t4p = Path(a.aux) / "tier4" / f"{tag}.json"
    t4raw = read_plain(t4p) if t4p.exists() else None
    t4 = json.loads(t4raw) if t4raw else {"rows": {}}
    t4rows = t4["rows"]
    if t4raw:
        reads[f"aux/tier4/{tag}.json"] = sha256(t4raw)
    A = V10.camera_A(ctx["car_a"], tag) if not a.no_camera_constants else None
    fill = None
    src10 = blk.get("path")
    if c10.get("status") == "refused" or (a.no_camera_constants and (src10 or "").startswith("camera k")):
        day = (ctx["grey"].get(tag) or 0.0) >= P["fill"]["day_grey_min"]
        kname = f"{tag}_cars_{'kc' if day else 'kn'}.json"
        kc = B.load(kname)
        if not V10.usable(kc):
            kname = f"{tag}_cars_{'kn' if day else 'kc'}.json"
            kc = B.load(kname)
            day = kname.endswith("_kc.json")
        reads.update({k: B.read[k] for k in (kname,) if k in B.read})
        if not a.no_camera_constants and V10.usable(kc):
            run = B.load(f"{tag}_run.json")
            if run is not None:
                reads[f"{tag}_run.json"] = B.read[f"{tag}_run.json"]
            w = P["fill"]["w_local_day"] if (group == "local_day" and day) else P["fill"]["w_other"]
            steps["fill"] = dict(tier=2, kind="camera-k measurement + D1 (tenth version's camera-k branch)", file=kname,
                                 day=day, w=w, grey=ctx["grey"].get(tag))
            out, src = fill_camk(kc, run, A, group, day, w, P10, steps["fill"])
            tdef = "t2f"
        elif t4rows:
            out = fill_tier4(t4, kc if V10.usable(kc) else None)
            src, tdef = "tier 4", "t4d"
            steps["fill"] = dict(tier=4, kind="uncalibrated DA3 metres (nominal focal)", rows_from=t4.get("source"))
        else:
            out, src, tdef = copy.deepcopy(c10), None, None
            steps["fill"] = dict(tier=None, why="no camera-k measurement and no tier-4 values")
        for k in ("params", "dash", "tracks"):
            if k in v9 and k not in out:
                out[k] = v9[k]
        out["status"] = "ok" if src else "refused"
        out["v10_status"], out["v10_reason"] = c10.get("status"), c10.get("reason")
        out["v10"] = blk
        if src:
            out["fps"] = out.get("fps") or (kc or {}).get("fps") or 20.0
            out["n_frames"] = out.get("n_frames") or (kc or {}).get("n_frames")
        fill = src
    else:
        out = copy.deepcopy(base)
        src = (v9.get("next_candidate") or {}).get("source", "current")
        if src != src10:
            raise SystemExit(f"{tag}: source {src!r} vs v10 path {src10!r}")
        tdef = "t1" if src == "current" else "t2k"
    if out.get("status") == "refused":
        out[VERSION] = dict(version=VERSION, group=group, status="refused (no tier available)", steps=steps, inputs=reads,
                            params_sha256=ctx["params_sha256"])
        return out, dict(status="refused", path=None)
    # ---- hood (before K4: a non-car box must not blank the cars behind it)
    removed = []
    if a.hood == "on":
        steps["hood"] = {}
        w_, h_ = ctx["frame_size"]
        lim, why, key = hood_limit(ctx["contours"], tag, w_, h_)
        steps["hood"]["contour"] = key
        removed = step_hood(out, lim, why, P["hood"]["tol_px"], steps["hood"])
    # ---- K4
    steps["K4"] = {}
    V11.step_k4(out, ctx["K4"], steps["K4"])
    # ---- relative speed variant
    if a.rel == "simple" and group == "main" and src == "current" and fill is None:
        rp = Path(a.aux) / "relsimple" / f"{tag}.csv.gz"
        if rp.exists():
            rows, h = read_relsimple(rp)
            reads[f"aux/relsimple/{tag}.csv.gz"] = h
            steps["rel_simple"] = {}
            step_rel_simple(out, rows, steps["rel_simple"])
        else:
            steps["rel_simple"] = dict(changed=0, why="no relsimple file")
    # ---- ego
    eg = steps["ego"] = {}
    for e in out.get("ego") or []:
        if e.get("kmh") is not None and "tier" not in e:
            e["tier"] = 1 if src == "current" else 2
    has = any(e.get("kmh") is not None for e in out.get("ego") or [])
    added, rule, why = 0, None, None
    if has:
        why = "the tenth version has ego values"
    elif a.no_camera_constants:
        why = "no camera constants (simulation): no camera-A ego"
    elif group == "main":
        rule = "MH"
        run = B.load(f"{tag}_run.json")
        if run is not None:
            reads[f"{tag}_run.json"] = B.read[f"{tag}_run.json"]
        mc = None
        if run is not None and run.get("status") != "ok":
            mp = find_mc(tag, (BD.get("mc_cache") or []) + [str(Path(a.aux) / "mc")])
            if mp is not None:
                mraw = read_plain(mp)
                mc = json.loads(mraw)
                reads[f"mc/{mp.name}"] = sha256(mraw)
                if A is None or abs(mc["A_cam"] - A) > 1e-9:
                    raise SystemExit(f"{tag}: re-run camera A {mc.get('A_cam')} != {A}")
                if mc.get("run_sha256") not in (None, B.read[f"{tag}_run.json"]):
                    raise SystemExit(f"{tag}: the MC re-run was made from another {tag}_run.json")
                eg["mc_file"] = str(mp)
        if run is None:
            why = "no marking run"
        elif A is None:
            why = "no camera A"
        else:
            added = step_mh(out, run, mc, A, camera_A_rel_sd(ctx["car_a"], tag), P10, P["ego_main_MH"], eg)
            if not added:
                why = eg.get("why")
    else:
        pe = ctx["PE"]
        rule = V11.choose_rule(group, src, pe)
        if rule is None:
            why = f"no new-ego rule for {src!r} in {group}"
        elif rule == "E1":
            dash = B.load(f"{tag}_dash.json")
            if dash is not None:
                reads[f"{tag}_dash.json"] = B.read[f"{tag}_dash.json"]
            eg["dash_status"] = "absent" if dash is None else dash.get("status")
            if dash is not None and dash.get("status") == "ok":
                why = "the depth dash ruler passed"
            elif any(e.get("kmh") is not None for e in v9.get("ego") or []):
                why = "the ninth-version file has ego values"
        if rule is not None and why is None:
            run = B.load(f"{tag}_run.json")
            if run is not None:
                reads[f"{tag}_run.json"] = B.read[f"{tag}_run.json"]
            eg["run_status"] = "absent" if run is None else run.get("status")
            if run is None or run.get("status") != pe["run_status"]:
                why = "the marking run did not pass"
            elif A is None:
                why = "no camera A"
            else:
                added = V11.step_ego(out, group, rule, run, A, pe, eg)
                for e in out.get("ego") or []:
                    if "v11_steps" in e:
                        e["tier"] = 2
                if not added:
                    why = eg.get("why")
    eg.update(rule=rule, applied=bool(added), why=why, seconds_added=added)
    # ---- every range widened until every leave-one-batch fold covers >= 95 % (Terry 2026-10-24): relative speed per
    #      (group, source) and the tenth version's own ego seconds per group, by a multiplier >= 1 of their half-width
    steps["range_mult"] = {}
    step_range_mult(out, group, fill, P.get("range_mult") or {}, steps["range_mult"])
    # ---- absolute speeds (every reading; new ego seconds included)
    steps["abs"] = {}
    step_abs_all(out, steps["abs"])
    # ---- tiers, > 60 m, <= 0 m, ranges
    steps["tiers"] = {}
    step_tiers(out, tdef, group, t4rows, ctx["R"], a.range, steps["tiers"])
    out[VERSION] = dict(version=VERSION, method="v11f_post.py post", params_sha256=ctx["params_sha256"],
                        switches=dict(t2=a.t2, rel=a.rel, night_range=a.night_range, local_range=a.local_range, k4=a.k4,
                                      hood=a.hood, range=a.range, no_camera_constants=bool(a.no_camera_constants)),
                        group=group, path=src, fill=fill, base="v10 + T2 own_cur" if t2_used else "v10",
                        tier3="licence-plate ruler not built (placeholder, no reading)", removed_not_car=removed,
                        steps=steps, inputs=reads)
    return out, dict(status="ok", path=src, fill=fill, t2=t2_used, k4_blanked=steps["K4"]["blanked"],
                     hood_blanked=(steps.get("hood") or {}).get("blanked", 0), hood_removed=len(removed),
                     ego_rule=rule if added else None, ego_seconds_added=added)


def cmd_post(a):
    BD, bsha = load_bdef(a.batch_def)
    bdir, vdir, odir = Path(BD["folder"]), Path(BD["v10"]), Path(a.out)
    if odir.resolve() in (bdir.resolve(), vdir.resolve()):
        raise SystemExit("--out must be its own folder")
    praw = read_plain(a.params)
    P = json.loads(praw)["values"]
    P10raw = read_plain(REPO / "tools/v10_params.json")
    if sha256(P10raw) != P["v10_params_sha256"]:
        raise SystemExit("tools/v10_params.json changed")
    P10 = json.loads(P10raw)["values"]
    P11raw = read_plain(REPO / "tools/v11_params.json")
    if sha256(P11raw) != P["v11_t1_params_sha256"]:
        raise SystemExit("tools/v11_params.json changed")
    P11 = json.loads(P11raw)["values"]
    PE = copy.deepcopy(P11["ego"])
    if a.night_range == "conservative":
        cons = P["ego_night_conservative"]["range_factor"]
        if cons is None:
            raise SystemExit("conservative night factors not set")
        PE["range_factor"]["night"] = dict(cons)
    if a.local_range == "conservative":
        lc = (P.get("ego_local_conservative") or {}).get("range_factor")
        if lc is None:
            raise SystemExit("conservative local-day factor not set")
        PE["range_factor"]["local_day"] = dict(lc)
    K4 = dict(P11["K4"]) if a.k4 == "draft" else dict(P["K4_lobo1pp"])
    groups, gfiles = group_lists(BD["tags"])
    craw = read_plain(BD["car_a"])
    seal = BD.get("seal")
    graw = seal_input(bdir, seal, "grey.json")
    hraw = read_plain(P["hood"]["contour_file"])
    if sha256(hraw) != P["hood"]["contour_sha256"]:
        raise SystemExit("hood contour file changed")
    import cv2
    tags_sorted = sorted(groups)
    fr0 = sorted((Path(BD["frames_root"]) / tags_sorted[0] / "frames").glob("*.jpg"))[0]
    im = cv2.imread(str(fr0))
    ctx = dict(V=V10.Batch(vdir, Path(BD["v10_seal"]) if BD.get("v10_seal") else None), B=V10.Batch(bdir, Path(seal) if seal else None),
               P=P, P10=P10, PE=PE, K4=K4, BD=BD, car_a=json.loads(craw), grey=json.loads(graw),
               R=(P.get("ranges") or {}).get("distance"), params_sha256=sha256(praw), contours=json.loads(hraw),
               frame_size=(im.shape[1], im.shape[0]))
    odir.mkdir(parents=True, exist_ok=True)
    summary = dict(version=VERSION, batch_def=str(a.batch_def), batch_def_sha256=bsha, batch=str(bdir), v10=str(vdir),
                   params_file=str(a.params), params_sha256=sha256(praw), v10_params_sha256=sha256(P10raw),
                   v11_t1_params_sha256=sha256(P11raw), car_a_sha256=sha256(craw), grey_sha256=sha256(graw),
                   hood_contour_sha256=sha256(hraw), group_lists=gfiles,
                   switches=dict(t2=a.t2, rel=a.rel, night_range=a.night_range, local_range=a.local_range, k4=a.k4,
                                 hood=a.hood, range=a.range, no_camera_constants=a.no_camera_constants), k4_values=K4, segments={},
                   started=time.strftime("%F %T"))
    for tag in tags_sorted:
        res, row = process(tag, groups[tag], ctx, a)
        if res is not None:
            body = json.dumps(res, indent=None)
            (odir / f"{tag}_cars_{VERSION}.json").write_text(body)
            row["sha256"] = sha256(body.encode())
        summary["segments"][tag] = dict(group=groups[tag], **row)
    summary["inputs_read"] = dict(sorted(ctx["B"].read.items()))
    summary["v10_inputs_read"] = dict(sorted(ctx["V"].read.items()))
    cnt = defaultdict(int)
    for r in summary["segments"].values():
        cnt[f"{r['group']}|{r['status']}|{r.get('path')}|fill {r.get('fill')}|t2 {r.get('t2')}|ego {r.get('ego_rule')}"] += 1
    summary["counts"] = dict(sorted(cnt.items()))
    summary["ended"] = time.strftime("%F %T")
    (odir / f"{VERSION}_run.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {sum(1 for r in summary['segments'].values() if 'sha256' in r)} files to {odir}")
    for k, v in summary["counts"].items():
        print("  ", k, v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tier4")
    t.add_argument("--batch-def", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--params", default=str(HERE / "v11f_params.json"))
    t.add_argument("--force", action="store_true")
    t.add_argument("--force-cache", nargs="*", default=None, help="tests: these segments take the cache path")
    m = sub.add_parser("mc")
    m.add_argument("--batch-def", required=True)
    m.add_argument("--out", required=True)
    p = sub.add_parser("post")
    p.add_argument("--batch-def", required=True)
    p.add_argument("--aux", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--params", default=str(HERE / "v11f_params.json"))
    p.add_argument("--t2", choices=("on", "off"), default="on")
    p.add_argument("--rel", choices=("simple", "off"), default="simple")
    p.add_argument("--night-range", choices=("conservative", "t1"), default="conservative")
    p.add_argument("--local-range", choices=("conservative", "t1"), default="conservative",
                   help="daytime-local E1 ego range factor: the all-folds >= 95 %% value (default) or track T1's")
    p.add_argument("--k4", choices=("lobo1pp", "draft"), default="lobo1pp")
    p.add_argument("--hood", choices=("on", "off"), default="on")
    p.add_argument("--range", choices=("asym", "sym"), default="asym")
    p.add_argument("--no-camera-constants", action="store_true")
    a = ap.parse_args()
    {"tier4": cmd_tier4, "mc": cmd_mc, "post": cmd_post}[a.cmd](a)


if __name__ == "__main__":
    main()

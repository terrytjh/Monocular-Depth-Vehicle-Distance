#!/usr/bin/env python3
"""Every car in view, not just the one ahead: distance from the depth model x k, followed with a tracker.

2026-09-30. It continues depth_dash_scale.py: the scale k, the tracked ego-lane lines and the ego speed come
from a dash-scale result, nothing here re-measures them. docs/DEPTH_DASH_V2_PREREG.md (draft) registers it.

  track    YOLOv8m-seg (official Ultralytics weights; car, bus, truck) on every frame, all boxes at det_conf
           stored, with ByteTrack ids (Zhang et al., ECCV 2022; Ultralytics' default bytetrack.yaml).
  retrack  the association again on the stored boxes, no detector run: --tracker botsort.yaml (BoT-SORT,
           Aharon et al. 2022; Ultralytics' default, camera-motion compensation on, no appearance model).
           The trackers only give ids; the boxes are the detector's. Image only.
  measure  every 0.2 s, for each tracked box on a frame that has a depth map:
             lane      0 = ego lane, -1 / +1 = the lane left / right of it, from where the bottom centre of the
                       box falls against the tracked ego-lane lines at that image row, one lane = the ego lane's
                       width in pixels at that row; None when the window has no line pair
             distance  k x depth read 6 % of the box height above its bottom, at the box centre
             far       distance beyond the ruler's reach: the 95th percentile of the far ends of the dash cycles
                       that set k. Beyond it k is an extrapolation: distance only, flagged, no speed.
             speeds    relative = slope of the distance over 9 samples of one track (1.6 s), only when all nine
                       are inside the ruler's reach and the distance moves <= 15 % per sample; absolute = ego speed
                       (median of the dash-scale readings in that window) + relative
             ranges    every speed carries a 95 % range: +-1.96 x mult x sigma, sigma from the fit (slope standard
                       error, median standard error) and the ruler's own scale uncertainty, mult set on open clips
           and the ego speed per second (median of the dash-scale readings, with its 95 % range).
           Boxes whose bottom reaches the bonnet rows (depth_dash_scale.static_rows) are skipped: YOLO calls the
           ego vehicle's own bonnet a car. Reads frames, cached depth maps, the dash-scale result and the tracks
           only; no answer file.
  score    comma2k19: moving radar returns (|relative speed + CAN speed| > 2 m/s) in the three lanes (legal
           12 ft lanes) are matched to our boxes frame by frame by bearing, not by distance, so a wrong distance
           still counts; distance error by lane, reach and legal cycles, how many radar cars inside the ruler's
           reach were found, lane agreement, relative / absolute speed against the radar (averaged over the same
           window) and ego speed per second against the pose speed, with the coverage of the 95 % ranges.
           Segments other than the two open ones are scored only with --verify-sealed.
  score-av2  Argoverse 2: our boxes vs projected lidar cuboids (IoU, never distance): ids that move to another
           car, cars that get a new id; with --cars the same distance / speed report against the lidar and the
           poses. Never downloads; logs outside the open four only with --verify-sealed.

  python3 tools/depth_dash_multicar.py track   --frames DIR --out tracks_bt.json
  python3 tools/depth_dash_multicar.py retrack --tracks tracks_bt.json --tracker botsort.yaml --out tracks.json
  python3 tools/depth_dash_multicar.py measure --frames DIR --tracks tracks.json --dash result.json \\
          --depth-cache DIR --out cars.json
  python3 tools/depth_dash_multicar.py score   --cars cars.json --truth data/output/c2k19_truth/<tag> [--out s.json]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

import depth_dash_scale as D

ROOT = Path(__file__).resolve().parents[1]
PARAMS = dict(
    yolo="checkpoints/yolov8m-seg.pt",
    classes=[2, 5, 7],       # car, bus, truck
    det_conf=0.1,            # boxes handed to the tracker (its second stage uses the low-score ones)
    min_conf=0.4,            # a box is measured only at the threshold of the 2026-09-29 relspeed registration
    tracker="bytetrack.yaml",  # what `track` runs while it stores every box; `retrack --tracker botsort.yaml` then
                             # re-associates the stored boxes (BoT-SORT: fewer restarts on the open AV2 logs)
    sample_s=0.2,            # measurement spacing
    read_up=0.06,            # depth read this fraction of the box height above its bottom
    patch=5,                 # depth median over patch x patch (half-resolution map)
    bonnet_px=2,
    dedup_iou=0.7,           # 2026-09-30 evening: two boxes of one frame overlapping by IoU > this are one car (YOLO's NMS
                             # is per class, so a pickup comes out as both car and truck with two track ids); the more
                             # confident box is kept. 0 = off (outputs before this addition)
    reach_pct=95,            # ruler's reach: this percentile of the measured cycles' far ends
    car_ruler="single",      # 2026-09-30 evening, the ruler the CARS are read with: "single" = the one k (default; all earlier
                             # outputs), "far" = k from the cycles whose near end is >= car_ruler_from legal cycles away (where
                             # the cars are; the depth model compresses more further out), the single k with fewer than
                             # car_ruler_min of them. The ego speed always keeps the single k (it is measured on the near cycles).
    car_ruler_from=1.0,
    car_ruler_min=10,
    k_guard=None,            # 2026-09-30 evening, off by default: refuse a ruler whose k is off the depth model's own metres by more
                             # than this factor (k > g or k < 1/g). Every correct k on the open clips is 0.78-1.39; a ruler that
                             # took half the legal cycle (a local road run with the freeway cycle, night reflectors) gives 2x
                             # (1.68-1.93 on the development segments). Registered runs pass --k-guard 1.5.
    half_win=4,              # relative speed over 2 * half_win + 1 samples = 1.6 s (2026-09-30; runs before 12:29 used 2)
    max_jump=0.15,
    min_ego=3,
    # reported 95 % ranges = +-1.96 x mult x sigma; mult set on the opened comma2k19 clips (docs/TERRY_DEV_LOG.md)
    rel_sigma_mult=3.4,      # 2026-09-30 19:40, the 30 daytime-freeway development segments: pooled 3.29 (single ruler) / 3.36
                             # (car ruler "far"), leaving one drive out covers 92-97 % of it. (13:00, seg10 + seg21: 3.8)
    ego_sigma_mult=2.5,      # same segments: pooled 2.49, leave-one-drive-out 95 / 95 / 95 %. (13:00, seg10 + seg21: 2.6)
    ego_bin_s=1.0,           # ego speed reported per this many seconds (median of the dash-scale readings) ...
    ego_min_n=3,             # ... when the bin has at least this many readings
    # scoring
    radar_offset_m=2.37,     # comma2k19 radar ranges start at the front of the car (openpilot v0.5.7 radar_interface):
                             # camera 1.52 m behind the radar (RADAR_TO_CAMERA) + tyre contact line 0.85 m ahead of the
                             # rear face (av2_offset_m); fixed in advance, never fitted to the radar
    radar_processing_m=0.0,  # 2026-10-02: comma2k19's processed radar range already includes openpilot's RDR_TO_LDR = 2.70 m
                             # (decoded from the dataset's own raw_can: LONG_DIST + 2.700 on every message of both cars).
                             # 0.0 keeps the second / third versions' registered scoring; --truth corrected subtracts 2.70 from
                             # every radar range before the offsets above, i.e. truth = range - 0.33 m, bearing range - 1.18 m
                             # (score-batch --truth-offset corrected).
    av2_offset_m=0.85,       # tyre contact line vs the lidar cuboid's rear face: ray through our box's bottom centre
                             # (AV2 factory calibration) meeting the ground under the car, open AV2 logs, rear face
                             # <= 25 m: median +0.85 m, n 343, per log -0.4..+1.0 (dev/contact_offset_av2.py, 12:50)
    moving_ms=2.0,
    lane_m=3.6576,           # 12 ft, the US freeway lane width: own lane |y| <= lane / 2, the next lanes out to 1.5 lanes
    radar_left_positive=True,  # with y positive to the left the radar bearings fall on our boxes (open clips)
    bearing_gate=0.75,       # radar-to-box matching: |column difference| <= bearing_gate x box width + 10 px ...
    sanity_gate=0.5,         # ... and |our distance - radar distance| <= this x radar distance: past that the pair is two
                             # different cars on one bearing (the radar often misses the nearer one), counted as not found
)


# ------------------------------------------------------------------ track (image only)

def cmd_track(a):
    from ultralytics import YOLO
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml
    model = YOLO(str(ROOT / PARAMS["yolo"]))
    frames = D.list_frames(Path(a.frames))
    _, osd = D.static_rows(frames)
    tracker = BYTETracker(IterableSimpleNamespace(**YAML.load(check_yaml(PARAMS["tracker"]))))
    per = {}
    for n, f in enumerate(frames):
        img = D.black_out(cv2.imread(str(f)), osd)
        r = model.predict(img, verbose=False, conf=PARAMS["det_conf"], classes=PARAMS["classes"], device=a.device)[0]
        boxes = r.boxes.cpu().numpy()
        ids = [-1] * len(boxes)
        if len(boxes):
            for t in tracker.update(boxes, img):
                ids[int(t[7])] = int(t[4])
        per[f.stem] = [[*map(float, b), float(s), int(c), i] for b, s, c, i in zip(boxes.xyxy, boxes.conf, boxes.cls, ids)]
        if (n + 1) % 200 == 0:
            print(f"{n + 1}/{len(frames)}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(params=PARAMS, frames_dir=str(a.frames), osd_boxes=osd, frames=per)))
    print("wrote", a.out)



def frames_dir_of(tr):
    """The frames directory a tracks file was made from; older files stored it relative to where they ran."""
    for cand in (Path(tr["frames_dir"]), ROOT / tr["frames_dir"], ROOT / "tools" / tr["frames_dir"]):
        if cand.is_dir():
            return cand.resolve()
    raise SystemExit(f"frames directory not found: {tr['frames_dir']}")


def cmd_retrack(a):
    """ByteTrack again on the detections stored by `track` (all boxes at det_conf), with other tracker
    settings; no detector run. --track-buffer is ByteTrack's own "frames a lost track is kept"."""
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml
    src = json.loads(Path(a.tracks).read_text())
    frames = D.list_frames(frames_dir_of(src))
    h, w = cv2.imread(str(frames[0])).shape[:2]
    cfg = YAML.load(check_yaml(a.tracker))
    cfg["track_buffer"] = int(a.track_buffer)
    if a.match_thresh is not None:
        cfg["match_thresh"] = float(a.match_thresh)
    if a.fuse_score is not None:
        cfg["fuse_score"] = a.fuse_score == "true"
    if cfg["tracker_type"] == "botsort":
        from ultralytics.trackers.bot_sort import BOTSORT
        cfg["with_reid"] = False
        tracker = BOTSORT(IterableSimpleNamespace(**cfg))
    else:
        tracker = BYTETracker(IterableSimpleNamespace(**cfg))
    per = {}
    for f in frames:
        rows = src["frames"].get(f.stem, [])
        ids = [-1] * len(rows)
        img = cv2.imread(str(f)) if cfg["tracker_type"] == "botsort" else None    # BoT-SORT's camera-motion step reads the frame
        if rows:
            boxes = Boxes(np.array([r[:6] for r in rows], dtype=np.float32), (h, w))
            for t in tracker.update(boxes, img):
                ids[int(t[7])] = int(t[4])
        elif img is not None:
            tracker.update(Boxes(np.zeros((0, 6), dtype=np.float32), (h, w)), img)
        per[f.stem] = [[*r[:6], i] for r, i in zip(rows, ids)]
    used = {k: cfg[k] for k in ("tracker_type", "track_buffer", "match_thresh", "fuse_score")}
    Path(a.out).write_text(json.dumps(dict(params=dict(PARAMS, tracker_used=used), frames_dir=str(frames_dir_of(src)),
                                           osd_boxes=src.get("osd_boxes"), frames=per)))
    print("wrote", a.out)


# ------------------------------------------------------------------ measure (no answers)

def ruler_reach(res, pct=PARAMS["reach_pct"]):
    ends = [(x["z_near"] + x["period"]) * res["k"] for x in res["periods"] if x["side"] in res.get("lines_used", [])]
    return float(np.percentile(ends, pct)) if ends else None


def lane_of(box, wd):
    """0 ego, -1 left, +1 right, 2 / -2 further out; None without a line pair."""
    L, R = wd.get("left"), wd.get("right")
    if not (wd.get("vp_row") and L and R):
        return None
    x0, y0, x1, y1 = box[:4]
    cx = (x0 + x1) / 2
    xl, xr = L["a"] * y1 + L["b"], R["a"] * y1 + R["b"]
    if xr - xl <= 1:
        return None
    return int(np.clip(np.floor((cx - xl) / (xr - xl)), -2, 2))


def ruler_scale_sigma(res):
    """One standard deviation of the ruler's scale, relative: its 95 % CI half-width / 1.96, mean over the lines used."""
    cis = [e["ci_half"] for sd, e in res.get("sides", {}).items() if sd in res.get("lines_used", []) and "ci_half" in e]
    return float(np.mean(cis)) / 1.96 if cis else 0.0


def ego_bins(res, n_frames, fps, scale_rel, p=PARAMS):
    """Ego speed per p['ego_bin_s'] seconds: the median of the dash-scale readings whose time falls in the bin (at
    least p['ego_min_n'] of them). sigma = the median's standard error (1.253 s / sqrt n) with the ruler's scale
    uncertainty; the reported 95 % range is +-1.96 x ego_sigma_mult x sigma."""
    out = []
    for b in range(int(n_frames / fps // p["ego_bin_s"])):
        t0 = b * p["ego_bin_s"]
        got = np.array([s["kmh"] for s in res["speeds"] if s["kmh"] is not None and t0 <= s["t_s"] < t0 + p["ego_bin_s"]])
        row = dict(t0=t0, n=int(len(got)), kmh=None, sigma_kmh=None, ci95_kmh=None)
        if len(got) >= p["ego_min_n"]:
            med = float(np.median(got))
            sig = float(np.hypot(1.253 * np.std(got, ddof=1) / np.sqrt(len(got)), med * scale_rel))
            row.update(kmh=med, sigma_kmh=sig, ci95_kmh=1.96 * p["ego_sigma_mult"] * sig)
        out.append(row)
    return out


def track_speeds(samples, ego_speeds, scale_rel, hw, m_rel, m_ego):
    """Relative / absolute speed per track, inside the ruler's reach only (fills the objects in place);
    returns the objects grouped by track id. ego_speeds: the pairwise ego readings (frame_i, frame_j, kmh)."""
    by_id = {}
    for j, s in enumerate(samples):
        for o in s["objects"]:
            by_id.setdefault(o["id"], {})[j] = o
    for tid, seq in by_id.items():
        for j, o in seq.items():
            grp = [seq.get(i) for i in range(j - hw, j + hw + 1)]
            if o["far"] or any(g is None or g["far"] for g in grp):
                continue
            if any(abs(b["dist"] - a_["dist"]) / a_["dist"] > PARAMS["max_jump"] for a_, b in zip(grp[:-1], grp[1:])):
                continue
            t = np.array([samples[i]["t"] for i in range(j - hw, j + hw + 1)])
            dd = np.array([g["dist"] for g in grp])
            tc = t - t.mean()
            rel, icpt = np.polyfit(tc, dd, 1)
            rel = float(rel)
            # uncertainty: the slope's standard error from the fit residuals, and the ruler's scale uncertainty
            # acting on the whole value
            resid = dd - (rel * tc + icpt)
            se = float(np.sqrt(np.sum(resid ** 2) / max(1, len(dd) - 2)) / np.sqrt(np.sum(tc ** 2)))
            o["win"] = [samples[j - hw]["fi"], samples[j + hw]["fi"]]
            o["rel_kmh"] = 3.6 * rel
            o["rel_sigma_kmh"] = 3.6 * float(np.hypot(se, abs(rel) * scale_rel))
            o["rel_ci95_kmh"] = 1.96 * m_rel * o["rel_sigma_kmh"]
            f0, f1 = o["win"]
            ego = np.array([sp["kmh"] for sp in ego_speeds if sp["kmh"] is not None and sp["frame_i"] >= f0 and sp["frame_j"] <= f1])
            o["abs_kmh"] = None
            if len(ego) >= PARAMS["min_ego"]:
                e_med = float(np.median(ego))
                e_sig = float(np.hypot(1.253 * np.std(ego, ddof=1) / np.sqrt(len(ego)), e_med * scale_rel))
                o["ego_win_kmh"], o["ego_win_sigma_kmh"] = e_med, e_sig
                o["abs_kmh"] = e_med + o["rel_kmh"]
                o["abs_ci95_kmh"] = 1.96 * float(np.hypot(m_rel * o["rel_sigma_kmh"], m_ego * e_sig))
    return by_id


# 2026-10-01, the ruler the EGO speed is read with (measure --ego-ruler local; off by default, so every output of the second
# version is unchanged byte for byte). Kept out of PARAMS for that reason: PARAMS is written into every cars file.
EGO_RULER = dict(
    w_s=2.5,                 # cycles within +-this many seconds of a reading ...
    n_min=20,                # ... at least this many of them, else the clip's k (both values: method C's registered local ruler)
    guard=1.5,               # k_local / k outside [1/guard, guard] -> the clip's k for that reading. A window can lock onto
                             # half the legal cycle (seen in the cycles alone on a development segment: k_local / k up to
                             # 1.84), which doubles k_local; ordinary drift on the development segments is 0.92-1.09
                             # (p5-p95). The factor is the registered k guard's.
)


def local_ruler_speeds(res, w=EGO_RULER["w_s"], n_min=EGO_RULER["n_min"], guard=EGO_RULER["guard"]):
    """2026-10-01. The dash run's pairwise ego readings with a ruler that follows the clip in time: each reading is
    k x shift / time, so it is multiplied by k_local(t) / k, where k_local = legal cycle / median of the cycles of the
    accepted lines measured within +-w s of the reading (at least n_min of them, and k_local / k within the guard;
    otherwise the clip's k is kept). The edges themselves were matched with the clip's k (depth_dash_scale.edge_speeds);
    only the conversion to km/h follows k_local. The depth model's scale drifts over seconds (second version, blind test: per-second ego errors correlated
    0.60 / 0.45 / 0.34 at 1 / 2 / 3 s); one k for the whole clip cannot follow that. Uses nothing but the clip."""
    k, L, step, fps = res["k"], res["cycle_m"], res["step"], res["fps"]
    cyc = [(c["n"] * step / fps, c["period"]) for c in res["periods"] if c["side"] in res.get("lines_used", [])]
    t = np.array([a for a, _ in cyc], float)
    v = np.array([b for _, b in cyc], float)
    out = []
    for s in res["speeds"]:
        s = dict(s)
        if s["kmh"] is not None:
            m = np.abs(t - s["t_s"]) <= w
            if m.sum() >= n_min:
                r = (L / float(np.median(v[m]))) / k
                if 1 / guard <= r <= guard:
                    s["kmh"] = s["kmh"] * r
        out.append(s)
    return out


def car_ruler_factor(cycles, L, mode, p=PARAMS):
    """cycles: (near end in metres by the single ruler, period in the ruler's own units) of the cycles that set it.
    Returns (factor on the single ruler, number of far cycles): 1.0 for mode "single" or too few far cycles."""
    if mode == "single" or not len(cycles):
        return 1.0, None
    c = np.asarray(cycles, float)
    far = c[c[:, 0] >= p["car_ruler_from"] * L, 1]
    if len(far) < p["car_ruler_min"]:
        return 1.0, int(len(far))
    return float(np.median(c[:, 1]) / np.median(far)), int(len(far))


def dedup(boxes, iou_max):
    """Of two boxes overlapping by IoU > iou_max keep the more confident one; order otherwise unchanged."""
    if not iou_max:
        return boxes
    keep = []
    for b in sorted(boxes, key=lambda b: -b[4]):
        if all(_iou(b[:4], k[:4]) <= iou_max for k in keep):
            keep.append(b)
    ids = {id(b) for b in keep}
    return [b for b in boxes if id(b) in ids]


def cmd_measure(a):
    res = json.loads(Path(a.dash).read_text())
    if res.get("status") != "ok":
        print(f"{a.dash}: the dash-scale result was refused ({res.get('reason', '')[:80]}); no distances")
        return
    tr = json.loads(Path(a.tracks).read_text())["frames"]
    frames = D.list_frames(Path(a.frames))
    h, w = cv2.imread(str(frames[0])).shape[:2]
    bonnet = D.static_rows(frames)[0]
    k, fps, step = res["k"], res["fps"], res["step"]
    g = a.k_guard if a.k_guard is not None else PARAMS["k_guard"]
    if g and not (1 / g <= k <= g):
        why = f"k {k:.3f} is off the depth model's own metres by more than x{g} (half or double legal cycle?)"
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(dict(status="refused", reason=why, k=k, k_guard=g, dash=str(a.dash))))
        print(f"{a.out}: refused -- {why}")
        return
    mode = a.car_ruler or PARAMS["car_ruler"]
    r_car, n_far = car_ruler_factor([(x["z_near"] * k, x["period"]) for x in res["periods"]
                                     if x["side"] in res.get("lines_used", [])], res["cycle_m"], mode)
    k_car = k * r_car
    reach = ruler_reach(res)
    if reach is not None:
        reach *= r_car
    every = max(1, int(round(PARAMS["sample_s"] * fps)))
    cache = Path(a.depth_cache)
    samples = []                                   # (fi, t, [objects])
    for fi in range(0, len(frames), every):
        stem = frames[fi].stem
        wd = D.lines_for(dict(windows=res["windows"]), fi / step)
        objs = []
        cand = [b for b in tr.get(stem, []) if b[6] >= 0 and b[4] >= PARAMS["min_conf"]
                and b[3] < bonnet - PARAMS["bonnet_px"]]
        cand = dedup(cand, a.dedup_iou if a.dedup_iou is not None else PARAMS["dedup_iou"])
        if cand:
            try:
                depth = D.load_depth(cache, frames[fi])
            except FileNotFoundError:
                depth = None
            if depth is not None:
                dh, dw = depth.shape
                r = PARAMS["patch"] // 2
                for x0, y0, x1, y1, conf, cls, tid in cand:
                    yy = int((y1 - PARAMS["read_up"] * (y1 - y0)) * dh / h)
                    xx = int((x0 + x1) / 2 * dw / w)
                    z = float(np.median(depth[max(0, yy - r): yy + r + 1, max(0, xx - r): xx + r + 1]))
                    d = k_car * z
                    objs.append(dict(id=int(tid), box=[x0, y0, x1, y1], conf=conf, cls=int(cls),
                                     lane=lane_of((x0, y0, x1, y1), wd), dist=d,
                                     far=bool(reach is None or d > reach)))
        samples.append(dict(fi=fi, t=fi / fps, objects=objs))
    hw = int(a.half_win) if getattr(a, "half_win", None) else PARAMS["half_win"]
    scale_rel = ruler_scale_sigma(res)
    ego_mode = getattr(a, "ego_ruler", None) or "single"
    if ego_mode == "local":
        res = dict(res, speeds=local_ruler_speeds(res))
    by_id = track_speeds(samples, res["speeds"], scale_rel, hw, PARAMS["rel_sigma_mult"], PARAMS["ego_sigma_mult"])
    out = dict(params=dict(PARAMS, half_win=hw, dedup_iou=a.dedup_iou if a.dedup_iou is not None else PARAMS["dedup_iou"]), dash=str(a.dash), tracks=str(a.tracks), k=k, cycle_m=res["cycle_m"],
               ruler_reach_m=reach, ruler_scale_sigma=scale_rel, fps=fps, n_frames=len(frames),
               ego=ego_bins(res, len(frames), fps, scale_rel), samples=samples)
    if mode != "single":
        out.update(car_ruler=mode, k_car=k_car, car_ruler_far_cycles=n_far)
    if ego_mode != "single":
        out.update(ego_ruler=ego_mode, ego_ruler_params=EGO_RULER)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out))
    n_obj = sum(len(s["objects"]) for s in samples)
    lanes = {}
    for s in samples:
        for o in s["objects"]:
            lanes[o["lane"]] = lanes.get(o["lane"], 0) + 1
    print(f"wrote {a.out}: k {k:.4f}, ruler reach {reach:.1f} m, {len(samples)} samples, {n_obj} car readings, "
          f"{len(by_id)} tracks, by lane {dict(sorted(lanes.items(), key=lambda x: (x[0] is None, x[0] or 0)))}")


# ------------------------------------------------------------------ measure, method C (lane markings only)

# 2026-09-30 evening. The spatial method C (marking_geometry.py) on the same tracked boxes: no depth model at all.
# Everything after the distance -- lanes, the reach flag, speeds, ranges, ego bins, scoring -- is the code above.
PARAMS_C = dict(
    min_rows=3.0,            # box bottom at least this many rows below the horizon (marking_geometry min_rows)
    geom_max_s=1.0,          # nearest kept horizon window within this many seconds (marking_geometry geom_max_s)
    rel_sigma_mult=3.4,      # 2026-09-30 19:40, the 30 daytime-freeway development segments: pooled 3.35, leave-one-drive-out
                             # 91 / 95 / 98 %
    ego_sigma_mult=1.9,      # same: pooled 1.84, leave-one-drive-out 93 / 97 / 94 %
)


def marking_reach(res, pct=PARAMS["reach_pct"]):
    """Same definition as ruler_reach: the pct-th percentile of the far ends of the dash cycles that set A. Cycles are
    stored in the nominal layout Z0 = A0 * u, so metres = Z0 * A / A0."""
    f = res["A"] / res["A0"]
    ends = [(c["z_near"] + c["period_m"]) * f for c in res.get("cycles", []) if c["side"] in res.get("lines_used", [])]
    return float(np.percentile(ends, pct)) if ends else None


def cmd_measure_c(a):
    """Method C distances for every tracked box: d = A_local / (y_bottom - y_h), y_bottom = the box's bottom row (the
    tyre on the road; the reading the AV2 contact offset was measured for), y_h and A_local from the marking run."""
    import marking_geometry as MG
    res = json.loads(Path(a.marking).read_text())
    if res.get("status") != "ok":
        print(f"{a.marking}: the marking run was refused ({res.get('reason', '')[:80]}); no distances")
        return
    pc = PARAMS_C
    tr = json.loads(Path(a.tracks).read_text())["frames"]
    frames = D.list_frames(Path(a.frames))
    bonnet = D.static_rows(frames)[0]
    fps = res["fps"]
    mode = a.car_ruler or PARAMS["car_ruler"]
    f = res["A"] / res["A0"]
    r_car, n_far = car_ruler_factor([(c["z_near"] * f, c["period_m"]) for c in res.get("cycles", [])
                                     if c["side"] in res.get("lines_used", [])], res["cycle_m"], mode)
    reach = marking_reach(res)
    if reach is not None:
        reach *= r_car
    rul = res["rulers"]
    rf = np.array([r["frame"] for r in rul])
    every = max(1, int(round(PARAMS["sample_s"] * fps)))
    samples = []
    for fi in range(0, len(frames), every):
        stem = frames[fi].stem
        objs = []
        wd = MG.nearest_geometry(res["windows"], fi, fps, pc["geom_max_s"])
        cand = [b for b in tr.get(stem, []) if b[6] >= 0 and b[4] >= PARAMS["min_conf"]
                and b[3] < bonnet - PARAMS["bonnet_px"]]
        cand = dedup(cand, a.dedup_iou if a.dedup_iou is not None else PARAMS["dedup_iou"])
        if wd is not None and cand:
            A_loc = rul[int(np.argmin(np.abs(rf - fi)))]["A_local"] * r_car
            for x0, y0, x1, y1, conf, cls, tid in cand:
                dy = y1 - wd["vp_row"]
                if dy < pc["min_rows"]:
                    continue
                d = A_loc / dy
                objs.append(dict(id=int(tid), box=[x0, y0, x1, y1], conf=conf, cls=int(cls),
                                 lane=lane_of((x0, y0, x1, y1), wd), dist=d,
                                 far=bool(reach is None or d > reach)))
        samples.append(dict(fi=fi, t=fi / fps, objects=objs))
    hw = int(a.half_win) if getattr(a, "half_win", None) else PARAMS["half_win"]
    scale_rel = ruler_scale_sigma(res)
    by_id = track_speeds(samples, res["speeds"], scale_rel, hw, pc["rel_sigma_mult"], pc["ego_sigma_mult"])
    out = dict(method="c", params=dict(PARAMS, half_win=hw, dedup_iou=a.dedup_iou if a.dedup_iou is not None else PARAMS["dedup_iou"]), params_c=pc, marking=str(a.marking), tracks=str(a.tracks),
               A=res["A"], cycle_m=res["cycle_m"], ruler_reach_m=reach, ruler_scale_sigma=scale_rel, fps=fps,
               n_frames=len(frames), ego=ego_bins(res, len(frames), fps, scale_rel, p=dict(PARAMS, **pc)),
               samples=samples)
    if mode != "single":
        out.update(car_ruler=mode, A_car_factor=r_car, car_ruler_far_cycles=n_far)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out))
    n_obj = sum(len(s["objects"]) for s in samples)
    print(f"wrote {a.out}: A {res['A']:.1f}, ruler reach {reach:.1f} m, {len(samples)} samples, {n_obj} car readings, "
          f"{len(by_id)} tracks")


# ------------------------------------------------------------------ scoring, shared

def check_sealed(seal_path, paths):
    """Refuse to score unless every prediction file is in the seal with the same sha256."""
    seal = json.loads(Path(seal_path).read_text())
    for p in paths:
        h = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        if seal.get("files", {}).get(Path(p).name) != h:
            raise SystemExit(f"{Path(p).name}: not in the seal, or changed since it was sealed -- refusing to score")
    print(f"sealed predictions verified ({len(paths)} files, sealed {seal.get('sealed_utc')})")


def lane_of_y(y_left):
    """Lane from the lateral offset (m, left positive) and the legal lane width: 0 own lane, -1 / +1 the lane
    left / right of it, None further out."""
    half = PARAMS["lane_m"] / 2
    if abs(y_left) <= half:
        return 0
    return (-1 if y_left > 0 else 1) if abs(y_left) <= 3 * half else None


def dist_summary(e):
    e = np.asarray(e, float)
    if not len(e):
        return dict(n=0)
    return dict(n=int(len(e)), median_abs_pct=float(np.median(np.abs(e))), bias_pct=float(np.median(e)),
                gross_gt25_pct=float(100 * np.mean(np.abs(e) > 25)))


def speed_summary(err, ci=None, base=None):
    err = np.asarray(err, float)
    if not len(err):
        return dict(n=0)
    d = dict(n=int(len(err)), mae_kmh=float(np.mean(np.abs(err))), bias_kmh=float(np.mean(err)),
             median_abs_kmh=float(np.median(np.abs(err))))
    if ci is not None:
        ci = np.asarray(ci, float)
        d["coverage_95"] = float(np.mean(np.abs(err) <= ci))
        d["ci95_median_kmh"] = float(np.median(ci))
    if base is not None:
        d["guess0_mae_kmh"] = float(np.mean(np.abs(np.asarray(base, float))))
    return d


def report(r):
    """Shared text report of a score dict (comma2k19 or Argoverse 2)."""
    print(f"ruler reach {r['ruler_reach_m']:.1f} m, cycle {r['cycle_m']} m, truth offset +{r['offset_m']} m")
    if "bearing_offset_px" in r:
        print(f"radar bearing check: median column offset of matched returns {r['bearing_offset_px']:+.1f} px")
    print(f"cars in the three lanes inside the ruler's reach: {r['found']['truth_inside_reach']}, "
          f"matched to one of our boxes {r['found']['matched']} ({r['found']['share']:.0%})")
    for lab, d in r["distance"].items():
        if d.get("n"):
            print(f"  distance {lab:<28} n {d['n']:4d}  median |err| {d['median_abs_pct']:4.1f}%  bias {d['bias_pct']:+5.1f}%"
                  f"  |err| > 25% {d['gross_gt25_pct']:4.1f}%")
    if r.get("lane_agreement") is not None:
        print(f"  our lane = true lane for {r['lane_agreement']:.0%} of matched cars ({r['lane_pairs']} with a lane)")
    for lab in ("relative_speed", "absolute_speed"):
        d = r[lab]
        if d.get("n"):
            extra = f"  guess 0 {d['guess0_mae_kmh']:.2f}" if "guess0_mae_kmh" in d else ""
            print(f"  {lab.replace('_', ' '):<15} n {d['n']:4d}  MAE {d['mae_kmh']:.2f} km/h  bias {d['bias_kmh']:+.2f}{extra}"
                  f"  95% range +-{d['ci95_median_kmh']:.1f} covers {d['coverage_95']:.0%}")
    e = r["ego_per_bin"]
    if e.get("n"):
        print(f"  ego speed per {r['ego_bin_s']:.0f} s  {e['n']} of {r['ego_bins_total']} bins  MAE {e['mae_kmh']:.2f} km/h  "
              f"bias {e['bias_kmh']:+.2f}  95% range +-{e['ci95_median_kmh']:.1f} covers {e['coverage_95']:.0%}"
              + (f"  (vs CAN: MAE {r['ego_vs_can']['mae_kmh']:.2f}, bias {r['ego_vs_can']['bias_kmh']:+.2f})" if r.get("ego_vs_can") else ""))


def assemble(pairs, ego_rows, found, total, reach, cyc, off, bin_s, extra=None):
    """Score dict from matched pairs (inside the three lanes) and per-bin ego rows."""
    inside = [p for p in pairs if not p["far"]]
    r = dict(ruler_reach_m=reach, cycle_m=cyc, offset_m=off, ego_bin_s=bin_s,
             found=dict(truth_inside_reach=total, matched=found, share=found / max(1, total)),
             distance={"own lane, inside reach": dist_summary([p["err"] for p in inside if p["lane"] == 0]),
                       "next lanes, inside reach": dist_summary([p["err"] for p in inside if p["lane"] in (-1, 1)]),
                       "beyond reach (flagged)": dist_summary([p["err"] for p in pairs if p["far"]])},
             distance_by_cycles={str(b) if b < 3 else "3+": dist_summary([p["err"] for p in inside if p["cyc_bin"] == b])
                                 for b in range(4)})
    lp = [p for p in pairs if p["our_lane"] is not None]
    r["lane_agreement"] = float(np.mean([p["our_lane"] == p["lane"] for p in lp])) if lp else None
    r["lane_pairs"] = len(lp)
    rs = [p for p in pairs if p.get("rel") is not None]
    r["relative_speed"] = speed_summary([p["rel"] - p["rel_truth"] for p in rs], [p["rel_ci95"] for p in rs],
                                        [p["rel_truth"] for p in rs])
    r["relative_speed"]["share_of_inside_pairs_with_output"] = len(rs) / max(1, len(inside))
    ab = [p for p in pairs if p.get("abs") is not None]
    r["absolute_speed"] = speed_summary([p["abs"] - p["abs_truth"] for p in ab], [p["abs_ci95"] for p in ab])
    got = [e for e in ego_rows if e["kmh"] is not None and e.get("truth") is not None]
    r["ego_per_bin"] = speed_summary([e["kmh"] - e["truth"] for e in got], [e["ci95_kmh"] for e in got])
    r["ego_bins_total"] = len(ego_rows)
    if extra:
        r.update(extra)
    return r


# ------------------------------------------------------------------ score, comma2k19

def _c2k19_dev():
    """Segments whose answers may be opened without a seal: seg10 / seg21, the four pilot segments of 2026-09-30,
    and the development side of docs/DEPTH_DASH_V2_SPLIT.txt. The blind side of that file is not in this set."""
    dev = {"b0c9d2329ad1606b_2018-07-30--13-44-30_10", "b0c9d2329ad1606b_2018-08-15--09-01-03_21",
           "b0c9d2329ad1606b_2018-08-02--16-41-38_13", "b0c9d2329ad1606b_2018-08-17--12-07-08_35",
           "99c94dc769b5d96e_2018-05-01--10-47-27_26", "99c94dc769b5d96e_2018-05-13--12-07-39_24"}
    split = ROOT / "docs/DEPTH_DASH_V2_SPLIT.txt"
    if split.exists():
        part = split.read_text().split("[blind")[0].split("\n")
        dev |= {"_".join(l.strip().split("|")[:3]) for l in part if l.count("|") >= 2}
    return dev


C2K19_DEV = _c2k19_dev()
# The comma2k19 camera, from openpilot's public source (common/transformations/camera.py, EON: focal 910 px on
# 1164 x 874, principal point at the centre) and the radar 1.52 m ahead of the camera (RADAR_TO_CAMERA). Used
# only to match radar returns to boxes by their bearing -- never in a measurement.
C2K19_CAM = dict(f=910.0, cx=582.0, radar_to_camera_m=1.52)


def c2k19_score(cars, tdir, return_rows=False):
    """Our boxes vs the car's radar, frame by frame. Moving returns only (|v_rel + v_CAN| > moving_ms), in the
    three lanes by the legal lane width. Matched by bearing, not by distance, so a wrong distance still counts:
    predicted column u = cx - f * y / (d + radar_to_camera), cost |u_box - u| / box width + 0.01 x true distance
    (along one bearing the camera sees the nearer car), gate |u_box - u| <= bearing_gate x width + 10 px; only a
    pair more than sanity_gate (50 %) apart in distance is refused, as two different cars on one bearing.
    Relative-speed truth = the radar's relative speed averaged over the same window and the same radar track;
    ego truth = the pose speed (CAN reported alongside). No id-switch count: the radar's track ids are 16 slots
    that get reused for other cars, so they cannot say whether our ids stayed on one car (AV2 can)."""
    from scipy.optimize import linear_sum_assignment
    P = PARAMS
    rows = list(csv.DictReader(open(tdir / "ego.csv")))
    can = np.array([float(r["can_speed_ms"]) for r in rows])
    pose = np.array([float(r["pose_speed_ms"]) for r in rows])
    sgn = 1.0 if P["radar_left_positive"] else -1.0
    radar, by_track = {}, {}
    for r in csv.DictReader(open(tdir / "radar.csv")):
        f, d, y, v = int(r["frame_idx"]), float(r["d_rel_m"]), sgn * float(r["y_rel_m"]), float(r["v_rel_ms"])
        if f < len(can) and abs(v + can[f]) > P["moving_ms"] and d > 0 and lane_of_y(y) is not None:
            radar.setdefault(f, []).append((int(r["track_id"]), d, y, v))
            by_track.setdefault(int(r["track_id"]), {})[f] = (d, v)
    off, reach, fps, cam = P["radar_offset_m"] - P["radar_processing_m"], cars["ruler_reach_m"] or 0.0, cars["fps"], \
        dict(C2K19_CAM, radar_to_camera_m=C2K19_CAM["radar_to_camera_m"] - P["radar_processing_m"])
    cyc = cars.get("cycle_m") or json.loads(Path(cars["dash"]).read_text())["cycle_m"]
    pairs, total, found, du_signed = [], 0, 0, []
    for s in cars["samples"]:
        fi, rets, objs = s["fi"], radar.get(s["fi"], []), s["objects"]
        inside = {i for i, r in enumerate(rets) if r[1] + off <= reach}
        total += len(inside)
        if not objs or not rets:
            continue
        C = np.full((len(objs), len(rets)), 1e6)
        U = np.zeros_like(C)
        for i, o in enumerate(objs):
            x0, _, x1, _ = o["box"]
            wb, ub = max(1.0, x1 - x0), (x0 + x1) / 2
            for jj, (rid, d, y, v) in enumerate(rets):
                U[i, jj] = ub - (cam["cx"] - cam["f"] * y / (d + cam["radar_to_camera_m"]))
                if abs(U[i, jj]) <= P["bearing_gate"] * wb + 10 and abs(o["dist"] - (d + off)) <= P["sanity_gate"] * (d + off):
                    C[i, jj] = abs(U[i, jj]) / wb + 0.01 * (d + off)
        for i, jj in zip(*linear_sum_assignment(C)):
            if C[i, jj] >= 1e6:
                continue
            o, (rid, d, y, v) = objs[i], rets[jj]
            dt = d + off
            found += jj in inside
            du_signed.append(U[i, jj])
            p = dict(t=s["t"], id=o["id"], truth_id=rid, lane=lane_of_y(y), our_lane=o["lane"], far=o["far"],
                     dist=o["dist"], truth=dt, err=100 * (o["dist"] - dt) / dt, cyc_bin=int(min(3, dt // cyc)))
            if o.get("rel_kmh") is not None:
                f0, f1 = o["win"]
                # the radar's 16 track ids are reused slots: average only while the same slot stays on this car
                # (within 2 m of where it would be at its own relative speed)
                vs = [vv for ff, (dd, vv) in by_track.get(rid, {}).items()
                      if f0 <= ff <= f1 and abs(dd - (d + v * (ff - fi) / fps)) <= 2.0]
                vt = float(np.mean(vs)) if vs else v
                p.update(rel=o["rel_kmh"], rel_truth=3.6 * vt, rel_ci95=o["rel_ci95_kmh"])
                if o.get("abs_kmh") is not None:
                    p.update(abs=o["abs_kmh"], abs_truth=3.6 * (float(np.mean(pose[f0:f1 + 1])) + vt),
                             abs_ci95=o["abs_ci95_kmh"])
            pairs.append(p)
    ego_rows, can_err = [], []
    for e in cars.get("ego", []):
        a0, a1 = int(round(e["t0"] * fps)), int(round((e["t0"] + P["ego_bin_s"]) * fps))
        e = dict(e, truth=3.6 * float(np.mean(pose[a0:a1])) if a1 <= len(pose) else None)
        if e["kmh"] is not None and e["truth"] is not None:
            can_err.append(e["kmh"] - 3.6 * float(np.mean(can[a0:a1])))
        ego_rows.append(e)
    r = assemble(pairs, ego_rows, found, total, cars["ruler_reach_m"], cyc, off, P["ego_bin_s"],
                 extra=dict(bearing_offset_px=float(np.median(du_signed)) if du_signed else float("nan"),
                            ego_vs_can=speed_summary(can_err)))
    return (r, pairs, ego_rows) if return_rows else r


def cmd_score(a):
    tdir = Path(a.truth)
    if tdir.name not in C2K19_DEV:
        if not a.verify_sealed:
            raise SystemExit(f"{tdir.name} is not a development segment: its predictions must be sealed (--verify-sealed)")
        check_sealed(a.verify_sealed, [a.cars])
    cars = json.loads(Path(a.cars).read_text())
    if cars.get("status") == "refused":
        print(f"{tdir.name}: refused -- {cars.get('reason', '')}")
        return
    r = c2k19_score(cars, tdir)
    print(tdir.name)
    report(r)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=1))



def cmd_score_batch(a):
    """comma2k19, many segments at once: every segment of --list is reported, a segment with no cars file (ruler refused,
    or measurement missing) is counted as refused with its reason; the pooled summary counts every matched pair once.
    Blind segments need --verify-sealed (all their cars files are checked against the seal before any truth is read)."""
    if getattr(a, "truth_offset", "registered") == "corrected":
        PARAMS["radar_processing_m"] = 2.70
    tags = [l.strip() for l in open(a.list) if l.strip() and not l.startswith("#")]
    cdir = Path(a.cars_dir)
    have = {t: cdir / f"{t}{a.suffix}" for t in tags if (cdir / f"{t}{a.suffix}").exists()}
    blind = [t for t in have if t not in C2K19_DEV]
    if blind:
        if not a.verify_sealed:
            raise SystemExit(f"{len(blind)} segments are not development segments: --verify-sealed is required")
        check_sealed(a.verify_sealed, [str(have[t]) for t in tags if t in have])
    per, pairs, ego, found, total, refused = {}, [], [], 0, 0, {}
    for t in tags:
        if t not in have:
            dj = cdir / f"{t}{a.refused_from}" if a.refused_from else None
            why = json.loads(dj.read_text()).get("reason", "") if dj is not None and dj.exists() else "no result file"
            refused[t] = why[:160]
            continue
        cars = json.loads(have[t].read_text())
        if cars.get("status") == "refused":
            refused[t] = cars.get("reason", "")[:160]
            continue
        r, pr, eg = c2k19_score(cars, Path(a.truth_root) / t, return_rows=True)
        per[t] = r
        pairs += [dict(p, seg=t) for p in pr]
        ego += [dict(e, seg=t) for e in eg]
        found += r["found"]["matched"]
        total += r["found"]["truth_inside_reach"]
    pooled = assemble(pairs, ego, found, total, None, "per segment", PARAMS["radar_offset_m"] - PARAMS["radar_processing_m"],
                      PARAMS["ego_bin_s"])
    seg_med = [np.median(np.abs([p["err"] for p in pairs if p["seg"] == t and not p["far"] and p["lane"] in (-1, 0, 1)]))
               for t in per if sum(1 for p in pairs if p["seg"] == t and not p["far"] and p["lane"] in (-1, 0, 1)) >= 10]
    out = dict(segments=len(tags), scored=len(per), refused=refused, pooled=pooled,
               per_segment_median_abs_pct=dict(n=len(seg_med), median=float(np.median(seg_med)) if seg_med else None,
                                                within_3=int(np.sum(np.array(seg_med) <= 3)), within_5=int(np.sum(np.array(seg_med) <= 5))),
               per_segment=per)
    print(f"{len(tags)} segments: {len(per)} scored, {len(refused)} refused")
    d = pooled["distance"]
    for lab in ("own lane, inside reach", "next lanes, inside reach", "beyond reach (flagged)"):
        e = d[lab]
        if e.get("n"):
            print(f"  distance {lab:28s} n {e['n']:5d}  median |err| {e['median_abs_pct']:5.1f}%  bias {e['bias_pct']:+5.1f}%")
    sm = out["per_segment_median_abs_pct"]
    if sm["n"]:
        print(f"  per segment (>= 10 pairs inside reach): median {sm['median']:.1f}%, <= 3 % in {sm['within_3']}/{sm['n']}, "
              f"<= 5 % in {sm['within_5']}/{sm['n']}")
    for lab, key in (("relative speed", "relative_speed"), ("absolute speed", "absolute_speed"), ("ego per 1 s", "ego_per_bin")):
        e = pooled[key]
        if e.get("n"):
            extra = f"  guess 0 {e['guess0_mae_kmh']:.2f}" if "guess0_mae_kmh" in e else ""
            print(f"  {lab:15s} n {e['n']:5d}  MAE {e['mae_kmh']:5.2f} km/h  bias {e['bias_kmh']:+5.2f}{extra}  "
                  f"95% range covers {100 * e.get('coverage_95', float('nan')):.0f}%")
    print(f"  found {found}/{total} radar cars inside reach; lane agreement {pooled['lane_agreement']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))

# ------------------------------------------------------------------ score, Argoverse 2

AV2_OPEN = {"c865c156", "cf5aaa11", "dc9077b9", "f668074d"}     # development logs; the four sealed ones stay closed


def av2_cuboids(log_dir: Path, stems, allow_sealed=False):
    """Per frame stem: vehicle cuboids projected into the front camera (lidar annotations, track uuid) with the
    forward distance from the camera to the cuboid's nearest face and the cuboid centre's lateral offset (m, left
    positive); and the ego speed per frame from the poses. Refuses a log whose annotations are not on disk, and a
    log outside the development set unless its predictions were verified against a seal: this never downloads."""
    import pandas as pd
    import av2_eval as E
    ann_p = log_dir / "annotations.feather"
    if not ((log_dir.name[:8] in AV2_OPEN or allow_sealed) and ann_p.exists()):
        raise SystemExit(f"{log_dir.name}: not an open development log (or sealed predictions) with annotations on disk; refusing")
    meta = json.loads((log_dir / "pinhole_meta.json").read_text())
    ann = pd.read_feather(ann_p)
    ann = ann[ann.category.isin({"REGULAR_VEHICLE", "LARGE_VEHICLE", "BUS", "BOX_TRUCK", "TRUCK", "SCHOOL_BUS",
                                 "ARTICULATED_BUS", "TRUCK_CAB", "VEHICULAR_TRAILER"})]
    poses = pd.read_feather(log_dir / "city_SE3_egovehicle.feather").drop_duplicates("timestamp_ns")
    poses = poses.sort_values("timestamp_ns").reset_index(drop=True)
    ptime = poses.timestamp_ns.values
    keep = np.r_[True, np.diff(ptime) >= 1_000_000]          # some logs carry poses 1 ns apart
    pt, pxyz = ptime[keep], poses[["tx_m", "ty_m", "tz_m"]].values[keep]
    pspeed = np.linalg.norm(np.gradient(pxyz, (pt - pt[0]) * 1e-9, axis=0), axis=1)     # m/s at each pose time
    extr = pd.read_feather(log_dir / "calibration/egovehicle_SE3_sensor.feather")
    extr = extr[extr.sensor_name == "ring_front_center"].iloc[0]
    cam_T_ego = np.linalg.inv(E.SE3((extr.qw, extr.qx, extr.qy, extr.qz), (extr.tx_m, extr.ty_m, extr.tz_m)))
    K = np.array([[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]])
    fr = pd.read_csv(log_dir / "pinhole_frames.csv")
    ts_of = dict(zip(fr.frame_idx.astype(str), fr.timestamp_ns))
    ann_ts = np.unique(ann.timestamp_ns.values)

    def pose_at(ts):
        r = poses.iloc[int(np.argmin(np.abs(ptime - ts)))]
        return E.SE3((r.qw, r.qx, r.qy, r.qz), (r.tx_m, r.ty_m, r.tz_m))
    out, speed = {}, {}
    for stem in stems:
        ts = ts_of.get(stem)
        cubs = []
        if ts is not None:
            speed[stem] = float(np.interp(ts, pt, pspeed))
            ats = int(ann_ts[np.argmin(np.abs(ann_ts - ts))])
            if abs(ats - ts) <= 60e6:
                M = np.linalg.inv(pose_at(ts)) @ pose_at(ats)
                for _, a in ann[ann.timestamp_ns == ats].iterrows():
                    C = np.c_[E.corners(a), np.ones(8)].T
                    Ce = (M @ C).T[:, :3]
                    Ch = (cam_T_ego @ M @ C).T[:, :3]
                    if (Ch[:, 2] <= 0.5).any():
                        continue
                    uv = (K @ Ch.T).T
                    uv = uv[:, :2] / uv[:, 2:3]
                    cubs.append(dict(box=[float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())],
                                     gt=float(Ce[:, 0].min()) - float(extr.tx_m), lat=float(Ce[:, 1].mean() - extr.ty_m),
                                     uuid=str(a.track_uuid)))
        out[stem] = cubs
    return out, speed


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def cmd_score_av2(a):
    """Every frame: our tracked boxes vs projected lidar cuboids (clipped to the image), Hungarian on IoU >= 0.5 --
    matching never looks at distance. "wrong car": one of our ids matched to cuboid A and later to cuboid B;
    "restart": one cuboid matched to one of our ids and later to another. With --cars also distance (offset fixed
    in advance, av2_offset_m: tyre contact vs the cuboid's rear face), relative speed (truth: slope of the lidar distance
    of the same cuboid over the same window), absolute speed and ego speed per bin (truth: the poses)."""
    from scipy.optimize import linear_sum_assignment
    tr = json.loads(Path(a.tracks).read_text())
    if a.verify_sealed:
        check_sealed(a.verify_sealed, [a.tracks] + ([a.cars] if a.cars else []))
    fdir = frames_dir_of(tr)
    log_dir = next(Path(ROOT / "data/input/av2").glob(fdir.parent.name[:8] + "*"))
    frames = D.list_frames(fdir)
    h, w = cv2.imread(str(frames[0])).shape[:2]
    bonnet = D.static_rows(frames)[0]
    gt, speed = av2_cuboids(log_dir, [f.stem for f in frames], allow_sealed=bool(a.verify_sealed))
    clip = lambda b: [max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3])]
    seq_by_ours, seq_by_gt, n_match, n_gt = {}, {}, 0, 0
    match_at = {}
    for j, f in enumerate(frames):
        ours = [b for b in tr["frames"].get(f.stem, []) if b[6] >= 0 and b[4] >= PARAMS["min_conf"] and b[3] < bonnet - PARAMS["bonnet_px"]]
        cubs = [c for c in gt.get(f.stem, []) if c["gt"] <= a.max_range]
        cb = [clip(c["box"]) for c in cubs]
        keep = [i for i, b in enumerate(cb) if (b[2] - b[0]) * (b[3] - b[1]) >= 400]
        n_gt += len(keep)
        if not ours or not keep:
            continue
        C = np.array([[1 - _iou(o[:4], cb[i]) for i in keep] for o in ours])
        for oi, kj in zip(*linear_sum_assignment(C)):
            if C[oi, kj] <= 0.5:
                o, c = ours[oi], cubs[keep[kj]]
                n_match += 1
                seq_by_ours.setdefault(o[6], []).append((j, c["uuid"]))
                seq_by_gt.setdefault(c["uuid"], []).append((j, o[6]))
                match_at[(f.stem, o[6])] = c
    wrong = sum(sum(1 for (_, x), (_, y) in zip(v[:-1], v[1:]) if x != y) for v in seq_by_ours.values())
    restart = sum(sum(1 for (_, x), (_, y) in zip(v[:-1], v[1:]) if x != y) for v in seq_by_gt.values())
    tracks_wrong = sum(any(x != y for (_, x), (_, y) in zip(v[:-1], v[1:])) for v in seq_by_ours.values())
    print(f"{log_dir.name[:8]}: frames {len(frames)}, visible cuboids <= {a.max_range:.0f} m: {n_gt}, matched boxes {n_match} "
          f"({n_match / max(1, n_gt):.0%}); our tracks matched {len(seq_by_ours)}, cars matched {len(seq_by_gt)}")
    print(f"   wrong car (our id moves to another car): {wrong} times, in {tracks_wrong} of {len(seq_by_ours)} tracks "
          f"= {wrong / max(1, n_match) * 100:.2f} per 100 matched boxes")
    gaps = [y[0] - x[0] for v in seq_by_gt.values() for x, y in zip(v[:-1], v[1:]) if x[1] != y[1]]
    gap_txt = (f"; frames between the old id's last match and the new id's first: median {np.median(gaps):.0f}, "
               f"<= 2 frames {np.mean(np.array(gaps) <= 2):.0%}") if gaps else ""
    print(f"   restart (a car gets a new id of ours): {restart} times over {len(seq_by_gt)} cars{gap_txt}")
    idr = dict(frames=len(frames), cuboids=n_gt, matched_boxes=n_match, wrong_car=wrong, restart=restart,
               tracks_matched=len(seq_by_ours), cars_matched=len(seq_by_gt))
    if not a.cars:
        return
    if not Path(a.cars).exists():
        print("   distance: no measurement (dash-scale refused on this log)")
        if a.out:
            Path(a.out).write_text(json.dumps(dict(id_tracking=idr, refused=True), indent=1))
        return
    cars = json.loads(Path(a.cars).read_text())
    stem_of = {fi: frames[fi].stem for fi in range(len(frames))}
    off, reach, fps = PARAMS["av2_offset_m"], cars["ruler_reach_m"] or 0.0, cars["fps"]
    cyc = cars["cycle_m"]
    t_of = {s["fi"]: s["t"] for s in cars["samples"]}
    pairs, total, found = [], 0, 0
    for s in cars["samples"]:
        stem = stem_of[s["fi"]]
        truth_in = [c for c in gt.get(stem, []) if lane_of_y(c["lat"]) is not None and c["gt"] + off <= reach]
        total += len(truth_in)
        seen = set()
        for o in s["objects"]:
            c = match_at.get((stem, o["id"]))
            if c is None or lane_of_y(c["lat"]) is None:
                continue
            dt = c["gt"] + off
            found += c in truth_in and c["uuid"] not in seen
            seen.add(c["uuid"])
            p = dict(t=s["t"], id=o["id"], truth_id=c["uuid"], lane=lane_of_y(c["lat"]), our_lane=o["lane"], far=o["far"],
                     dist=o["dist"], truth=dt, err=100 * (o["dist"] - dt) / dt, cyc_bin=int(min(3, dt // cyc)))
            if o.get("rel_kmh") is not None:
                f0, f1 = o["win"]
                ts, ds = [], []
                for fi in range(f0, f1 + 1):
                    if fi in t_of:
                        cc = [x for x in gt.get(stem_of[fi], []) if x["uuid"] == c["uuid"]]
                        if cc:
                            ts.append(t_of[fi])
                            ds.append(cc[0]["gt"])
                if len(ts) >= 3:
                    vt = float(np.polyfit(np.array(ts) - np.mean(ts), ds, 1)[0])
                    p.update(rel=o["rel_kmh"], rel_truth=3.6 * vt, rel_ci95=o["rel_ci95_kmh"])
                    ego_true = [speed[stem_of[fi]] for fi in range(f0, f1 + 1) if stem_of.get(fi) in speed]
                    if o.get("abs_kmh") is not None and ego_true:
                        p.update(abs=o["abs_kmh"], abs_truth=3.6 * (float(np.mean(ego_true)) + vt), abs_ci95=o["abs_ci95_kmh"])
            pairs.append(p)
    ego_rows = []
    for e in cars.get("ego", []):
        a0, a1 = int(round(e["t0"] * fps)), int(round((e["t0"] + PARAMS["ego_bin_s"]) * fps))
        v = [speed[stem_of[fi]] for fi in range(a0, min(a1, len(frames))) if stem_of.get(fi) in speed]
        ego_rows.append(dict(e, truth=3.6 * float(np.mean(v)) if v else None))
    r = assemble(pairs, ego_rows, found, total, cars["ruler_reach_m"], cyc, off, PARAMS["ego_bin_s"], extra=dict(id_tracking=idr))
    report(r)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("track")
    t.add_argument("--frames", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--device", default="0")
    m = sub.add_parser("measure")
    for x in ("--frames", "--tracks", "--dash", "--depth-cache", "--out"):
        m.add_argument(x, required=True)
    m.add_argument("--car-ruler", choices=("single", "far"), default=None,
                   help="ruler the cars are read with (default PARAMS car_ruler = single)")
    m.add_argument("--k-guard", type=float, default=None, help="refuse k outside [1/g, g] (default PARAMS k_guard = off)")
    m.add_argument("--dedup-iou", type=float, default=None, help="default PARAMS dedup_iou; 0 reproduces older outputs")
    m.add_argument("--half-win", type=int, default=None,
                   help="relative speed over 2 * half_win + 1 samples (default PARAMS half_win = 4, i.e. 1.6 s)")
    m.add_argument("--ego-ruler", choices=("single", "local"), default=None,
                   help="2026-10-01: ruler the ego speed is read with (default single = the second version); "
                        "local = k from the cycles within +-2.5 s of each reading (local_ruler_speeds)")
    mc = sub.add_parser("measure-c", help="method C (lane markings only) distances on the same tracks")
    for x in ("--frames", "--tracks", "--marking", "--out"):
        mc.add_argument(x, required=True)
    mc.add_argument("--half-win", type=int, default=None)
    mc.add_argument("--car-ruler", choices=("single", "far"), default=None,
                   help="ruler the cars are read with (default PARAMS car_ruler = single)")
    mc.add_argument("--dedup-iou", type=float, default=None)
    s = sub.add_parser("score")
    s.add_argument("--cars", required=True)
    s.add_argument("--truth", required=True)
    s.add_argument("--verify-sealed", default="", help="seal json; required for any segment outside the two open ones")
    s.add_argument("--out", default="")
    sb = sub.add_parser("score-batch", help="comma2k19: many segments, per segment and pooled")
    sb.add_argument("--list", required=True, help="text file, one segment tag per line")
    sb.add_argument("--cars-dir", required=True)
    sb.add_argument("--suffix", default="_cars.json")
    sb.add_argument("--refused-from", default="_dash.json", help="where a refused segment's reason is read (same dir)")
    sb.add_argument("--truth-root", default=str(ROOT / "data/output/c2k19_truth"))
    sb.add_argument("--verify-sealed", default="")
    sb.add_argument("--out", default="")
    sb.add_argument("--truth-offset", choices=("registered", "corrected"), default="registered",
                    help="corrected: remove the 2.70 m that comma2k19's processed radar range already includes")
    rt = sub.add_parser("retrack")
    rt.add_argument("--tracks", required=True)
    rt.add_argument("--out", required=True)
    rt.add_argument("--track-buffer", type=int, default=30)
    rt.add_argument("--tracker", default=PARAMS["tracker"], help="bytetrack.yaml or botsort.yaml (Ultralytics built-ins)")
    rt.add_argument("--match-thresh", type=float, default=None)
    rt.add_argument("--fuse-score", choices=("true", "false"), default=None)
    v = sub.add_parser("score-av2")
    v.add_argument("--tracks", required=True)
    v.add_argument("--cars", default=None)
    v.add_argument("--max-range", type=float, default=60.0)
    v.add_argument("--verify-sealed", default="", help="seal json; required for any log outside the four open ones")
    v.add_argument("--out", default="")
    a = ap.parse_args()
    dict(track=cmd_track, retrack=cmd_retrack, measure=cmd_measure, score=cmd_score, **{"measure-c": cmd_measure_c, "score-batch": cmd_score_batch}, **{"score-av2": cmd_score_av2})[a.cmd](a)


if __name__ == "__main__":
    main()

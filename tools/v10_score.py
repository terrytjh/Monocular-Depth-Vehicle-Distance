#!/usr/bin/env python3
"""Tenth version scoring (docs/DEPTH_DASH_V10_PREREG.md), comma2k19: v9d, v9 and v10 of one batch, after the seal.

Run only after the seal. Every prediction file it scores is first checked against its version's seal (sha256, in the
format of tools/seal_predictions.py: the file's name under "files" or "inputs"), a sealed file that is missing is
refused too, and every truth file must be newer than the seal -- all before any truth is read. --no-seal-check is
accepted only for opened development segments (listed in the seal of one of the opened batch folders below and with
their truth on disk), and even then a seal.json in a version's folder must cover that version's files. A seal is
keyed by file name, so two versions whose files share a name (v9 and v9d both write <tag>_cars_v9.json, in different
work folders) need separate seal files.

Main table: the settled shared rules (shared repo, docs/SPEC.md section 5 and configs/eval_comma2k19.json, decisions
D1-D12 settled 2026-10-18), re-implemented here (own code; the S rules of the development re-scoring), on Terry's boxes:

  truth          processed radar distance + the car's offset: second car (99c94dc769b5d96e) - 0.70 m, first car
                 (b0c9d2329ad1606b) - 0.33 m; returns at >= 258.6 m (empty radar slots) removed; d_cam > 1 m
  in scope       moving (|pose ego + radar relative speed| >= 7.2 km/h) and |lat| <= 5.49 m (own lane |lat| <= 1.83 m)
  duplicates     two returns of one frame within 1 m and 0.5 m are one car (the lower track id is kept)
  matching       by image position only: column u = cx - f * lat / d_cam (f 910 px), |u_box - u| <= 0.75 x box width
                 + 10 px, car width d_cam x box width / f in 0.8-4.5 m, cost |u_box - u| / width + 0.01 x d_cam,
                 Hungarian; every return enters, a box matched to an out-of-scope return is not scored; no distance gate
  centre column  second car 615 px (589 for drives before the 2018-06-25--22-19-02 remount), first car 598 px;
                 --columns overrides per drive (the per-drive vanishing-point rule, fixed before the seal);
                 582 px re-matched as the sensitivity
  no number      readings the method puts beyond 60 m carry no number (dropped after the matching)
  distance       true distance <= 30 m: median |error| % and m, signed median (bias), share > 25 %, found rate;
                 0-10 / 10-20 / 20-30 / 30-60 m (lo < truth <= hi); 95 % intervals by drawing whole segments
                 (2000 draws, seed 20261016); truth-offset sensitivity on the same pairs
  speeds         relative speed vs the radar's per-frame value and absolute speed vs pose ego + radar relative, on the
                 scored readings with true distance <= 30 m (MAE, bias, guess 0, slope, correlation, 95 % coverage);
                 ego speed per fixed 1 s bin of frame_idx (the bin's readings, >= 3, against the mean pose speed of the
                 same frames; negative values count as 0 as in the shared format), per frame MAE and 95 % coverage,
                 coverage = bins with a value / all 1 s bins of the group's segments (refused segments included)
  hard accel.    reported apart, no threshold: the 1 s bins whose true acceleration is >= 4 km/h per s in size,
                 a = (v(b + 1) - v(b - 1)) / 2 with v(b) = the mean pose speed of all frames of bin b (truth only, so
                 only here; undefined for the first and last full bin of a segment); per-second MAE on those bins and
                 the paired comparison on the same bins

Secondary table: the ninth version's registered rules (tools/v9_score.py: c2k19_score of tools/depth_dash_multicar.py
with truth = radar - 0.33 m for both cars, moving returns in the three 12-ft lanes, the 50 % distance gate; readings
beyond 60 m carry no number; speeds pooled as score-batch pools them).

Paired, on the same readings (same segment, frame, box and radar target, true distance <= 30 m) and the same 1 s bins:
NEW vs OLD for each --pair (default v10 vs v9d, v9d vs v9): pooled numbers on the intersection, the 95 % interval of the
difference (whole segments drawn), and one-sided sign tests (NEW better) over segments and over drives with >= 10 paired
readings (distance: median |error|; speeds: MAE).

Groups: tags_main.txt / tags_local_day.txt / tags_night.txt of --batch (written before predicting); --labels checks them
against the batch's label table (segment_id,road_class,light) and refuses if they differ. Segments whose lowest true ego
speed is >= 60 km/h are also reported apart (that uses the truth, so only here).

  # a blind batch: each version's own folder, file suffix and seal
  python3 tools/v10_score.py --batch DIR_WITH_TAGS \\
      --version v9d DIR_V9D _cars_v9.json SEAL_V9D --version v9 DIR_V9 _cars_v9.json SEAL_V9 \\
      --version v10 DIR_V10 _cars_v10.json SEAL_V10 --labels LABELS.csv --out score_v10.json
  # one folder, files <tag>_cars_<label>.json, one seal (default DIR/seal.json)
  python3 tools/v10_score.py --batch DIR --versions v9 v10 --seal DIR/seal_v10.json --out X.json
  # opened development data only
  python3 tools/v10_score.py --batch data/output/dash_scale/blind_v9 --version v9 data/output/dash_scale/blind_v9 \\
      _cars_v9.json data/output/dash_scale/blind_v9/seal.json --version v9d DEV_V9D _cars_v9.json none --no-seal-check --out X.json
"""
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data/output/dash_scale"
TRUTH = ROOT / "data/output/c2k19_truth"
GROUPS = ("main", "local_day", "night")

# Opened batches: scored, their answers read (results in docs/DEPTH_DASH_V{2..9}_RESULTS.md). Their seal.json lists each
# segment's <tag>_dash.json; only these segments may be scored with --no-seal-check.
OPENED_BATCHES = ("blind_v2", "blind_v3", "blind_v4", "blind_v5", "blind_v7", "blind_v8", "blind_v9")

CAR1, CAR2 = "b0c9d2329ad1606b", "99c94dc769b5d96e"
REMOUNT = "2018-06-25--22-19-02"          # second car: camera remounted between the 21:14 and 22:19 drives of that day
SETTLED = dict(
    fx=910.0,
    cx={CAR1: 598.0, CAR2: 615.0},
    cx_car2_before_remount=589.0,
    cx_sensitivity=582.0,
    truth_dist_offset_m={CAR1: -0.33, CAR2: -0.70},
    truth_dist_offset_sensitivity_m={CAR2: [-0.33, -0.56, -0.70, -0.89, -1.2, 0.1],
                                     CAR1: [0.04, -0.19, -0.33, -0.52, -0.83, 0.47]},
    truth_empty_slot_min_native_m=258.6,
    moving_min_abs_kmh=7.2,
    lane_half_width_m=1.83,
    adjacent_max_abs_lat_m=5.49,
    min_truth_dist_m=1.0,
    truth_duplicate_dist_m=1.0,
    truth_duplicate_lat_m=0.5,
    match_k_width=0.75,
    match_px=10.0,
    match_dist_weight=0.01,
    match_width_m=[0.8, 4.5],
    near_max_m=30.0,
    dist_bins_m=[0, 10, 20, 30, 60],
    no_number_beyond_m=60.0,
    ego_bin_s=1.0,
    ego_min_readings_per_bin=3,
    group_min_truth_ego_kmh=60.0,
    min_pairs_per_segment=10,
    bootstrap=dict(n=2000, seed=20261016),
)
# this registration's own report line (not a shared rule): ego seconds of hard acceleration or braking, by the truth
HARD_ACCEL_KMH_PER_S = 4.0
# the keys that must equal the shared configuration file when --shared-config is given
SHARED_KEYS = ("truth_dist_offset_m", "truth_dist_offset_sensitivity_m", "truth_empty_slot_min_native_m",
               "moving_min_abs_kmh", "lane_half_width_m", "adjacent_max_abs_lat_m", "min_truth_dist_m",
               "truth_duplicate_dist_m", "truth_duplicate_lat_m", "match_k_width", "match_px", "match_dist_weight",
               "match_width_m", "near_max_m", "ego_bin_s", "ego_min_readings_per_bin", "group_min_truth_ego_kmh",
               "min_pairs_per_segment")


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def drive_of(tag: str) -> str:
    return tag.rsplit("_", 1)[0]


def car_of(tag: str) -> str:
    return tag.split("_", 1)[0]


def tags_of(d: Path, g: str) -> list[str]:
    p = d / f"tags_{g}.txt"
    return [l.strip() for l in open(p) if l.strip() and not l.startswith("#")] if p.exists() else []


class Version:
    def __init__(self, label, folder, suffix, seal):
        self.label, self.dir, self.suffix = label, Path(folder), suffix
        self.seal = None if seal in (None, "", "none", "-") else Path(seal)

    def path(self, tag):
        return self.dir / f"{tag}{self.suffix}"

    def load(self, tag):
        p = self.path(tag)
        if not p.exists():
            return None
        c = json.loads(p.read_text())
        return None if c.get("status") == "refused" else c


# ------------------------------------------------------------------ checks before any truth is read

def parse_utc(s):
    return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))


def verify_against(seal_path: Path, v: Version, tags, what="seal"):
    seal = json.loads(seal_path.read_text())
    ent = dict(seal.get("inputs") or {})
    ent.update(seal.get("files") or {})
    n = 0
    for t in tags:
        p = v.path(t)
        if p.exists():
            if ent.get(p.name) != sha256(p):
                raise SystemExit(f"{v.label}: {p} is not in the {what} {seal_path}, or changed since -- refusing to score")
            n += 1
        elif p.name in ent:
            raise SystemExit(f"{v.label}: {p.name} is in the {what} {seal_path} but missing -- refusing to score")
    return seal, n


def check_seals(versions, alltags, no_seal_check, troot):
    info = {}
    if no_seal_check:
        opened = set()
        for b in OPENED_BATCHES:
            sp = DS / b / "seal.json"
            if sp.exists():
                opened |= {k[: -len("_dash.json")] for k in json.loads(sp.read_text()).get("files", {}) if k.endswith("_dash.json")}
        for t in alltags:
            if t not in opened:
                raise SystemExit(f"--no-seal-check: {t} is not a segment of an opened development batch "
                                 f"({', '.join(OPENED_BATCHES)}) -- refusing")
            if not (troot / t / "radar.csv").exists():
                raise SystemExit(f"--no-seal-check: {t} has no opened truth -- refusing")
    for v in versions:
        if v.seal is None:
            if not no_seal_check:
                raise SystemExit(f"{v.label}: no seal given -- refusing (only --no-seal-check on opened development data scores without one)")
            own = v.dir / "seal.json"
            if own.exists():
                _, n = verify_against(own, v, alltags, "folder's seal")
                info[v.label] = dict(seal=str(own), files_verified=n, note="--no-seal-check; the folder's seal.json covers the files")
            else:
                info[v.label] = dict(seal=None, files_verified=0, note="--no-seal-check (opened development data)")
            continue
        seal, n = verify_against(v.seal, v, alltags)
        if not seal.get("sealed_utc"):
            raise SystemExit(f"{v.seal}: no sealed_utc -- refusing")
        info[v.label] = dict(seal=str(v.seal), seal_sha256=sha256(v.seal), sealed_utc=seal["sealed_utc"], files_verified=n)
        print(f"{v.label}: sealed predictions verified ({n} files, sealed {seal['sealed_utc']})")
        if not no_seal_check:
            t_seal = parse_utc(seal["sealed_utc"])
            for t in alltags:
                for name in ("radar.csv", "ego.csv"):
                    p = troot / t / name
                    if not p.exists():
                        raise SystemExit(f"{t}: truth {name} missing -- create it after the seal, then score")
                    if os.stat(p).st_mtime <= t_seal:
                        raise SystemExit(f"{p} is older than the seal of {v.label} ({seal['sealed_utc']}) -- refusing")
    return info


def check_labels(tags_dir, labels_csv):
    """The groups as tools/run.py of the shared method folder writes them from the labels: night if light = night,
    else local_day if road_class = ca_local, else main."""
    want = {g: set() for g in GROUPS}
    for r in csv.DictReader(open(labels_csv)):
        g = "night" if r["light"] == "night" else ("local_day" if r["road_class"] == "ca_local" else "main")
        want[g].add(r["segment_id"])
    for g in GROUPS:
        have = set(tags_of(tags_dir, g))
        if have != want[g]:
            raise SystemExit(f"tags_{g}.txt differs from {labels_csv}: {len(have - want[g])} extra, {len(want[g] - have)} missing")
    return {g: len(want[g]) for g in GROUPS}


def check_shared_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg.get("_pending"):
        raise SystemExit(f"{path}: pending items {cfg['_pending']} -- refusing")
    plain = lambda x: {k: v for k, v in x.items() if not k.startswith("_")} if isinstance(x, dict) else x
    bad = [k for k in SHARED_KEYS if json.dumps(plain(cfg.get(k)), sort_keys=True) != json.dumps(SETTLED[k], sort_keys=True)]
    if cfg.get("camera_for_matching", {}).get("fx") != SETTLED["fx"]:
        bad.append("camera_for_matching.fx")
    if bad:
        raise SystemExit(f"{path} differs from this scorer's settled rules in {bad} -- refusing")
    return dict(path=str(path), sha256=sha256(Path(path)), keys_checked=len(SHARED_KEYS) + 1)


# ------------------------------------------------------------------ truth

_TRUTH = {}


def load_truth(troot, tag):
    if tag not in _TRUTH:
        d = troot / tag
        rows = list(csv.DictReader(open(d / "ego.csv")))
        can = np.array([float(r["can_speed_ms"]) for r in rows])
        pose = np.array([float(r["pose_speed_ms"]) for r in rows])
        by_f = defaultdict(list)
        for r in csv.DictReader(open(d / "radar.csv")):
            by_f[int(r["frame_idx"])].append((int(r["track_id"]), float(r["d_rel_m"]), float(r["y_rel_m"]), float(r["v_rel_ms"])))
        _TRUTH[tag] = (can, pose, dict(by_f))
    return _TRUTH[tag]


def column_of(drive, overrides):
    if drive in overrides:
        return float(overrides[drive])
    car, when = drive.split("_", 1)
    if car == CAR2:
        return SETTLED["cx_car2_before_remount"] if when < REMOUNT else SETTLED["cx"][CAR2]
    if car == CAR1:
        return SETTLED["cx"][CAR1]
    raise SystemExit(f"{drive}: unknown car -- no matching column")


# ------------------------------------------------------------------ settled shared rules

def lane_settled(y):
    if abs(y) <= SETTLED["lane_half_width_m"]:
        return 0
    return (-1 if y > 0 else 1) if abs(y) <= SETTLED["adjacent_max_abs_lat_m"] else None


def settled_segment(cars, tag, troot, cx):
    """(scored pairs, in-scope truth <= 30 m, of them matched, ego bins, ego frames) of one segment."""
    S = SETTLED
    can, pose, by_f = load_truth(troot, tag)
    car = car_of(tag)
    off = S["truth_dist_offset_m"][car]
    mov = S["moving_min_abs_kmh"] / 3.6
    near = S["near_max_m"]
    pairs, n_truth, n_found = [], 0, 0
    for s in cars["samples"]:
        fi, objs = s["fi"], s["objects"]
        if fi >= len(pose):
            continue
        cand = []
        for rid, d, y, v in by_f.get(fi, []):
            if d >= S["truth_empty_slot_min_native_m"] or d <= 0 or d + off <= S["min_truth_dist_m"]:
                continue
            lane = lane_settled(y)
            cand.append([rid, d, y, v, abs(v + pose[fi]) >= mov and lane is not None, lane])
        if len(cand) > 1:
            cand.sort(key=lambda r: r[0])
            keep = []
            for r in cand:
                if any(abs(r[1] - k[1]) <= S["truth_duplicate_dist_m"] and abs(r[2] - k[2]) <= S["truth_duplicate_lat_m"] for k in keep):
                    continue
                keep.append(r)
            cand = keep
        n_truth += sum(1 for r in cand if r[4] and r[1] + off <= near)
        if not objs or not cand:
            continue
        C = np.full((len(objs), len(cand)), 1e6)
        for i, ob in enumerate(objs):
            x0, _, x1, _ = ob["box"]
            wb, ub = max(1.0, x1 - x0), (x0 + x1) / 2
            for j, (rid, d, y, v, ins, lane) in enumerate(cand):
                dt = d + off
                du = ub - (cx - S["fx"] * y / dt)
                if abs(du) <= S["match_k_width"] * wb + S["match_px"] and \
                        S["match_width_m"][0] <= dt * wb / S["fx"] <= S["match_width_m"][1]:
                    C[i, j] = abs(du) / wb + S["match_dist_weight"] * dt
        for i, j in zip(*linear_sum_assignment(C)):
            if C[i, j] >= 1e6 or not cand[j][4]:
                continue
            rid, d, y, v, ins, lane = cand[j]
            ob, dt = objs[i], d + off
            if dt <= near:
                n_found += 1
            if ob["dist"] > S["no_number_beyond_m"]:
                continue
            p = dict(seg=tag, drive=drive_of(tag), car=car, fi=fi, id=ob["id"], truth_id=rid, lane=lane, far=bool(ob.get("far")),
                     dist=ob["dist"], truth=dt, err=100 * (ob["dist"] - dt) / dt)
            if ob.get("rel_kmh") is not None:
                p.update(rel=ob["rel_kmh"], rel_ci95=ob.get("rel_ci95_kmh"), rel_truth=3.6 * v)
            if ob.get("abs_kmh") is not None:
                p.update(abs=ob["abs_kmh"], abs_ci95=ob.get("abs_ci95_kmh"), abs_truth=3.6 * (pose[fi] + v))
            pairs.append(p)
    # ego speed: the method's 1 s value written to each of its sampled frames in that second
    fps = cars.get("fps") or 20.0
    nb = int(round(fps * S["ego_bin_s"]))
    ego = {int(round(e["t0"] / S["ego_bin_s"])): e for e in cars.get("ego") or [] if e.get("kmh") is not None}
    frames, by_bin = [], defaultdict(list)
    for s in cars["samples"]:
        fi = s["fi"]
        e = ego.get(fi // nb)
        if e is None or fi >= len(pose):
            continue
        val = max(0.0, float(e["kmh"]))
        fr = dict(seg=tag, drive=drive_of(tag), bin=fi // nb, fi=fi, kmh=val, truth=3.6 * pose[fi], ci95=e.get("ci95_kmh"))
        frames.append(fr)
        by_bin[fi // nb].append(fr)
    # true acceleration of each bin (truth only): central difference of the all-frame mean pose speed per bin
    vsec = [3.6 * float(np.mean(pose[b * nb:(b + 1) * nb])) for b in range(len(pose) // nb)]
    acc = {b: (vsec[b + 1] - vsec[b - 1]) / 2.0 for b in range(1, len(vsec) - 1)}
    bins = [dict(seg=tag, drive=drive_of(tag), bin=b, kmh=float(np.mean([f["kmh"] for f in fs])),
                 truth=float(np.mean([f["truth"] for f in fs])), n=len(fs), acc=acc.get(b))
            for b, fs in sorted(by_bin.items()) if len(fs) >= S["ego_min_readings_per_bin"]]
    return pairs, n_truth, n_found, bins, frames


def hard_accel(b):
    return b.get("acc") is not None and abs(b["acc"]) >= HARD_ACCEL_KMH_PER_S


def boot(rows, value, stat=np.median, cluster="seg", n=None, seed=None):
    """95 % interval of stat over the pooled values when whole clusters are drawn with replacement."""
    n = n or SETTLED["bootstrap"]["n"]
    seed = SETTLED["bootstrap"]["seed"] if seed is None else seed
    by = {}
    for r in rows:
        by.setdefault(r[cluster], []).append(value(r))
    g = [np.array(x) for x in by.values()]
    if len(g) < 2:
        return None
    rng = np.random.default_rng(seed)
    out = [stat(np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])) for _ in range(n)]
    return [round(float(x), 2) for x in np.percentile(out, [2.5, 97.5])]


def dist_summ(rows, ci=False):
    if not rows:
        return dict(n=0)
    e = np.array([p["err"] for p in rows])
    m = np.array([p["dist"] - p["truth"] for p in rows])
    out = dict(n=int(len(e)), segments=len({p["seg"] for p in rows}), drives=len({p["drive"] for p in rows}),
               median_abs_pct=round(float(np.median(np.abs(e))), 2), median_abs_m=round(float(np.median(np.abs(m))), 2),
               bias_pct=round(float(np.median(e)), 2), bias_m=round(float(np.median(m)), 2),
               gt25_pct=round(float(100 * np.mean(np.abs(e) > 25)), 1))
    if ci:
        out["ci95_pct_by_segment"] = boot(rows, lambda p: abs(p["err"]))
        out["ci95_m_by_segment"] = boot(rows, lambda p: abs(p["dist"] - p["truth"]))
    return out


def speed_summ(err, ci=None, truth=None, pred=None):
    err = np.asarray(err, float)
    if not len(err):
        return dict(n=0)
    d = dict(n=int(len(err)), mae_kmh=round(float(np.mean(np.abs(err))), 3), bias_kmh=round(float(np.mean(err)), 3),
             median_abs_kmh=round(float(np.median(np.abs(err))), 3))
    if ci is not None:
        ok = [(abs(a), c) for a, c in zip(err, ci) if c is not None]
        if ok:
            d["coverage_95"] = round(float(np.mean([a <= c for a, c in ok])), 4)
            d["ci95_median_kmh"] = round(float(np.median([c for _, c in ok])), 3)
            d["with_range"] = len(ok)
    if truth is not None:
        tr = np.asarray(truth, float)
        d["guess0_mae_kmh"] = round(float(np.mean(np.abs(tr))), 3)
        if pred is not None and len(tr) >= 3 and np.std(tr) > 0:
            pr = np.asarray(pred, float)
            d["slope"] = round(float(np.polyfit(tr, pr, 1)[0]), 3)
            d["corr"] = round(float(np.corrcoef(tr, pr)[0, 1]), 3)
    return d


def rel_summ(rows):
    r = [p for p in rows if p.get("rel") is not None]
    return speed_summ([p["rel"] - p["rel_truth"] for p in r], [p["rel_ci95"] for p in r], [p["rel_truth"] for p in r],
                      [p["rel"] for p in r])


def abs_summ(rows):
    r = [p for p in rows if p.get("abs") is not None]
    return speed_summ([p["abs"] - p["abs_truth"] for p in r], [p["abs_ci95"] for p in r])


def min_truth_ego_kmh(troot, tag):
    return 3.6 * float(np.min(load_truth(troot, tag)[1]))


def settled_group(v, tags, troot, overrides, sensitivity=True):
    S = SETTLED
    rows, bins, frames, n_truth, n_found, have, n_bins_all = [], [], [], 0, 0, [], 0
    rows582, nt582, nf582 = [], 0, 0
    for t in tags:
        n_bins_all += int(len(load_truth(troot, t)[1]) // int(round(20.0 * S["ego_bin_s"])))
        c = v.load(t)
        if c is None:
            continue
        have.append(t)
        pr, nt, nf, eb, ef = settled_segment(c, t, troot, column_of(drive_of(t), overrides))
        rows += pr
        bins += eb
        frames += ef
        n_truth += nt
        n_found += nf
        if sensitivity:
            pr, nt, nf, _, _ = settled_segment(c, t, troot, S["cx_sensitivity"])
            rows582 += pr
            nt582 += nt
            nf582 += nf
    near = [p for p in rows if p["truth"] <= S["near_max_m"]]
    edges = S["dist_bins_m"]
    r = dict(segments=len(tags), with_output=len(have),
             distance=dict(near30=dict(dist_summ(near, ci=True), found_pct=round(100 * n_found / n_truth, 1) if n_truth else None,
                                       truth_in_scope=n_truth),
                           own_lane=dist_summ([p for p in near if p["lane"] == 0]),
                           next_lanes=dist_summ([p for p in near if p["lane"] != 0]),
                           by_true_distance={f"{lo}-{hi} m": dist_summ([p for p in rows if lo < p["truth"] <= hi])
                                             for lo, hi in zip(edges[:-1], edges[1:])},
                           beyond_60m_true=dist_summ([p for p in rows if p["truth"] > edges[-1]]),
                           flagged_far_near30=dist_summ([p for p in near if p["far"]])),
             relative_speed_near30=rel_summ(near), absolute_speed_near30=abs_summ(near),
             ego=dict(per_second=speed_summ([b["kmh"] - b["truth"] for b in bins]),
                      per_frame=speed_summ([f["kmh"] - f["truth"] for f in frames], [f["ci95"] for f in frames]),
                      seconds_with_value=len(bins), seconds_all=n_bins_all,
                      coverage_pct=round(100 * len(bins) / n_bins_all, 1) if n_bins_all else None,
                      per_second_hard_accel=dict(speed_summ([b["kmh"] - b["truth"] for b in bins if hard_accel(b)]),
                                                 min_abs_accel_kmh_per_s=HARD_ACCEL_KMH_PER_S)))
    # lowest true ego speed >= 60 km/h (truth-based, so only at scoring)
    fast = {t for t in tags if min_truth_ego_kmh(troot, t) >= S["group_min_truth_ego_kmh"]}
    r["min_truth_ego_ge60"] = dict(segments=len(fast), distance_near30=dist_summ([p for p in near if p["seg"] in fast]),
                                   relative_speed_near30=rel_summ([p for p in near if p["seg"] in fast]),
                                   ego_per_second=speed_summ([b["kmh"] - b["truth"] for b in bins if b["seg"] in fast]))
    r["by_car"] = {c: dist_summ([p for p in near if p["car"] == c]) for c in sorted({p["car"] for p in near})}
    if sensitivity:
        near582 = [p for p in rows582 if p["truth"] <= S["near_max_m"]]
        r["sensitivity_column_582"] = dict(dist_summ(near582), found_pct=round(100 * nf582 / nt582, 1) if nt582 else None,
                                           truth_in_scope=nt582)
        sc = []
        for k in range(len(S["truth_dist_offset_sensitivity_m"][CAR2])):
            sh = []
            for p in near:
                dt = p["truth"] + S["truth_dist_offset_sensitivity_m"][p["car"]][k] - S["truth_dist_offset_m"][p["car"]]
                sh.append(dict(p, truth=dt, err=100 * (p["dist"] - dt) / dt))
            sc.append(dict({c: S["truth_dist_offset_sensitivity_m"][c][k] for c in (CAR2, CAR1)}, **dist_summ(sh)))
        r["sensitivity_truth_offset_same_pairs"] = sc
    return r, dict(rows=near, bins=bins)


# ------------------------------------------------------------------ the ninth version's registered rules

def registered_group(v, tags, troot):
    sys.path.insert(0, str(ROOT / "tools"))
    import depth_dash_multicar as M
    M.PARAMS["radar_processing_m"] = 2.70
    M.PARAMS["sanity_gate"] = 0.5
    allp, ego, have = [], [], 0
    for t in tags:
        c = v.load(t)
        if c is None:
            continue
        have += 1
        c = dict(c, cycle_m=c.get("cycle_m") or 1e9)     # cycle only bins an unreported table
        _, pr, eg = M.c2k19_score(c, troot / t, return_rows=True)
        allp += [dict(p, seg=t, drive=drive_of(t)) for p in pr if p["lane"] in (-1, 0, 1)]
        ego += [dict(e, seg=t) for e in eg]
    pooled = M.assemble(allp, ego, 0, 0, None, "per segment", 0.0, M.PARAMS["ego_bin_s"])
    gp = [p for p in allp if p["dist"] <= SETTLED["no_number_beyond_m"]]
    near = [p for p in gp if p["truth"] <= 30]
    sp = {k: {x: (round(pooled[k][x], 4) if isinstance(pooled[k].get(x), float) else pooled[k].get(x))
              for x in ("n", "mae_kmh", "bias_kmh", "coverage_95", "guess0_mae_kmh") if x in pooled[k]}
          for k in ("relative_speed", "absolute_speed", "ego_per_bin")}
    r = dict(segments=len(tags), with_output=have, near30_gate=dist_summ(near),
             own_lane=dist_summ([p for p in near if p["lane"] == 0]), next_lanes=dist_summ([p for p in near if p["lane"] != 0]),
             by_true_distance={f"{lo}-{hi} m": dist_summ([p for p in gp if lo <= p["truth"] < hi])
                               for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 60))},
             readings_without_number=len(allp) - len(gp), speeds=sp)
    return r, dict(rows=near)


# ------------------------------------------------------------------ paired

def sign_test(per):
    better = sum(1 for a, b in per if a < b)
    worse = sum(1 for a, b in per if a > b)
    n = better + worse
    return dict(units=len(per), better=better, worse=worse, ties=len(per) - n,
                p_one_sided=float(binomtest(better, n, 0.5, alternative="greater").pvalue) if n else None,
                p_two_sided=float(binomtest(better, n, 0.5).pvalue) if n else None)


def paired_block(both, err_new, err_old, stat):
    """both: [(new row, old row)]; err_*: row -> error used by stat (median |error| or MAE)."""
    if not both:
        return dict(n=0)
    en = np.array([err_new(a) for a, _ in both], float)
    eo = np.array([err_old(b) for _, b in both], float)
    out = dict(n=len(both), new=round(float(stat(en)), 3), old=round(float(stat(eo)), 3))
    mp = SETTLED["min_pairs_per_segment"]
    for unit in ("seg", "drive"):
        u = defaultdict(list)
        for k, (a, _) in enumerate(both):
            u[a[unit]].append(k)
        idx = [np.array(x) for x in u.values()]
        if unit == "seg" and len(idx) >= 2:
            rng = np.random.default_rng(SETTLED["bootstrap"]["seed"])
            diffs = []
            for _ in range(SETTLED["bootstrap"]["n"]):
                sel = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
                diffs.append(stat(en[sel]) - stat(eo[sel]))
            out["ci95_difference_by_segment"] = [round(float(x), 3) for x in np.percentile(diffs, [2.5, 97.5])]
        per = [(float(stat(en[x])), float(stat(eo[x]))) for x in idx if len(x) >= mp]
        out[f"sign_test_by_{'segment' if unit == 'seg' else 'drive'}"] = sign_test(per)
    return out


def mae(x):
    return float(np.mean(np.abs(x)))


def medabs(x):
    return float(np.median(np.abs(x)))


def paired_settled(new, old):
    key = lambda p: (p["seg"], p["fi"], p["id"], p["truth_id"])
    o = {key(p): p for p in old["rows"]}
    both = [(p, o[key(p)]) for p in new["rows"] if key(p) in o]
    rel = [(a, b) for a, b in both if a.get("rel") is not None and b.get("rel") is not None]
    ab = [(a, b) for a, b in both if a.get("abs") is not None and b.get("abs") is not None]
    ob = {(b["seg"], b["bin"]): b for b in old["bins"]}
    eg = [(a, ob[(a["seg"], a["bin"])]) for a in new["bins"] if (a["seg"], a["bin"]) in ob]
    return dict(
        distance_pct=paired_block(both, lambda p: p["err"], lambda p: p["err"], medabs),
        distance_m=paired_block(both, lambda p: p["dist"] - p["truth"], lambda p: p["dist"] - p["truth"], medabs),
        relative_speed_kmh=paired_block(rel, lambda p: p["rel"] - p["rel_truth"], lambda p: p["rel"] - p["rel_truth"], mae),
        absolute_speed_kmh=paired_block(ab, lambda p: p["abs"] - p["abs_truth"], lambda p: p["abs"] - p["abs_truth"], mae),
        ego_per_second_kmh=paired_block(eg, lambda b: b["kmh"] - b["truth"], lambda b: b["kmh"] - b["truth"], mae),
        ego_per_second_hard_accel_kmh=paired_block([(a, b) for a, b in eg if hard_accel(a)], lambda b: b["kmh"] - b["truth"],
                                                   lambda b: b["kmh"] - b["truth"], mae),
        readings_new=len(new["rows"]), readings_old=len(old["rows"]), seconds_new=len(new["bins"]), seconds_old=len(old["bins"]))


def paired_registered(new, old):
    key = lambda p: (p["seg"], round(p["t"], 3), p["id"], p["truth_id"])
    o = {key(p): p for p in old["rows"]}
    both = [(p, o[key(p)]) for p in new["rows"] if key(p) in o]
    return dict(distance_pct=paired_block(both, lambda p: p["err"], lambda p: p["err"], medabs),
                distance_m=paired_block(both, lambda p: p["dist"] - p["truth"], lambda p: p["dist"] - p["truth"], medabs))


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True, help="folder with tags_main.txt / tags_local_day.txt / tags_night.txt")
    ap.add_argument("--version", nargs=4, action="append", default=[], metavar=("LABEL", "DIR", "SUFFIX", "SEAL"),
                    help="a version: its folder, file suffix (e.g. _cars_v9.json) and seal ('none' only with --no-seal-check)")
    ap.add_argument("--versions", nargs="*", default=[], help="labels whose files are <tag>_cars_<label>.json in --batch")
    ap.add_argument("--seal", default=None, help="seal for --versions (default <batch>/seal.json)")
    ap.add_argument("--pair", nargs=2, action="append", default=[], metavar=("NEW", "OLD"),
                    help="paired comparison (default v10 v9d and v9d v9, when present)")
    ap.add_argument("--no-seal-check", action="store_true", help="opened development segments only")
    ap.add_argument("--labels", default=None, help="label table (segment_id,road_class,light) the tags files must agree with")
    ap.add_argument("--columns", default=None, help="JSON {drive: matching centre column}, fixed before the seal")
    ap.add_argument("--shared-config", default=None, help="the shared configs/eval_comma2k19.json: refuse unless it equals the rules here")
    ap.add_argument("--truth-root", default=str(TRUTH))
    ap.add_argument("--boot-seed", type=int, default=None, help=f"default {SETTLED['bootstrap']['seed']} (the shared configuration)")
    ap.add_argument("--no-registered", action="store_true", help="skip the secondary table")
    ap.add_argument("--no-sensitivity", action="store_true", help="skip the 582 px re-matching and the offset scenarios")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    B, troot = Path(a.batch), Path(a.truth_root)
    versions = [Version(*x) for x in a.version]
    versions += [Version(l, B, f"_cars_{l}.json", a.seal or str(B / "seal.json")) for l in a.versions]
    if not versions:
        raise SystemExit("no version given (--version or --versions)")
    labels = [v.label for v in versions]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"duplicate version labels {labels}")
    if a.boot_seed is not None:
        SETTLED["bootstrap"]["seed"] = a.boot_seed
    groups = {g: tags_of(B, g) for g in GROUPS}
    alltags = sorted({t for g in GROUPS for t in groups[g]})
    if not alltags:
        raise SystemExit(f"{B}: no tags_*.txt")
    overlap = [t for g in GROUPS for h in GROUPS if g < h for t in set(groups[g]) & set(groups[h])]
    if overlap:
        raise SystemExit(f"segments in two groups: {overlap[:3]}")
    meta = dict(scorer="tools/v10_score.py", run_local=time.strftime("%Y-%m-%d %H:%M:%S %Z"), batch=str(B),
                truth_root=str(troot), groups={g: len(t) for g, t in groups.items()}, settled_rules=SETTLED,
                no_seal_check=a.no_seal_check)
    if a.labels:
        meta["labels"] = dict(path=a.labels, sha256=sha256(Path(a.labels)), groups=check_labels(B, a.labels))
    if a.shared_config:
        meta["shared_config"] = check_shared_config(a.shared_config)
    overrides = json.loads(Path(a.columns).read_text()) if a.columns else {}
    if overrides:
        meta["columns_override"] = dict(path=a.columns, sha256=sha256(Path(a.columns)), drives=overrides)
    meta["columns_used"] = {d: column_of(d, overrides) for d in sorted({drive_of(t) for t in alltags})}
    # ---- before any truth is read
    meta["seals"] = check_seals(versions, alltags, a.no_seal_check, troot)
    for t in alltags:
        if not (troot / t / "radar.csv").exists() or not (troot / t / "ego.csv").exists():
            raise SystemExit(f"{t}: no truth under {troot} -- refusing")
    pairs = [tuple(p) for p in a.pair] or [p for p in (("v10", "v9d"), ("v9d", "v9")) if p[0] in labels and p[1] in labels]
    for n_, o_ in pairs:
        if n_ not in labels or o_ not in labels:
            raise SystemExit(f"--pair {n_} {o_}: not both among {labels}")

    res = dict(meta=meta, settled={}, registered={})
    for g in GROUPS:
        tags = groups[g]
        if not tags:
            continue
        st, keep = {}, {}
        for v in versions:
            st[v.label], keep[v.label] = settled_group(v, tags, troot, overrides, sensitivity=not a.no_sensitivity)
        res["settled"][g] = dict(versions=st, paired={f"{n_}_vs_{o_}": paired_settled(keep[n_], keep[o_]) for n_, o_ in pairs})
        if not a.no_registered:
            rg, rk = {}, {}
            for v in versions:
                rg[v.label], rk[v.label] = registered_group(v, tags, troot)
            res["registered"][g] = dict(versions=rg, paired={f"{n_}_vs_{o_}": paired_registered(rk[n_], rk[o_]) for n_, o_ in pairs})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))

    f = lambda d: (f"{d.get('median_abs_pct', '-'):>5}% {d.get('median_abs_m', '-'):>5} m bias {d.get('bias_pct', '-'):>5}% "
                   f">25% {d.get('gt25_pct', '-'):>4} n {d.get('n', 0):5d}")
    for g, r in res["settled"].items():
        print(f"===== {g} ({len(groups[g])} segments), settled shared rules")
        for lab, x in r["versions"].items():
            dd = x["distance"]
            print(f"  {lab:5s} out {x['with_output']:3d} | <= 30 m {f(dd['near30'])} CI {dd['near30'].get('ci95_pct_by_segment')} "
                  f"found {dd['near30'].get('found_pct')}% | "
                  + " ".join(f"{k} {v.get('median_abs_m', '-')}" for k, v in dd["by_true_distance"].items()))
            e = x["ego"]
            print(f"        rel {x['relative_speed_near30'].get('mae_kmh')} (cov {x['relative_speed_near30'].get('coverage_95')}) "
                  f"abs {x['absolute_speed_near30'].get('mae_kmh')} ego/s {e['per_second'].get('mae_kmh')} "
                  f"(frame cov {e['per_frame'].get('coverage_95')}, {e['seconds_with_value']}/{e['seconds_all']} s = {e['coverage_pct']}%)"
                  + (f" | 582: {x['sensitivity_column_582'].get('median_abs_pct')}% found {x['sensitivity_column_582'].get('found_pct')}%"
                     if "sensitivity_column_582" in x else ""))
        for k, p in r["paired"].items():
            for what in ("distance_pct", "relative_speed_kmh", "absolute_speed_kmh", "ego_per_second_kmh",
                         "ego_per_second_hard_accel_kmh"):
                q = p[what]
                if q.get("n"):
                    s1, s2 = q["sign_test_by_segment"], q["sign_test_by_drive"]
                    print(f"  {k} {what}: n {q['n']} {q['old']} -> {q['new']} CI {q.get('ci95_difference_by_segment')} | "
                          f"segments {s1['better']}/{s1['worse']} p {s1['p_one_sided']} | drives {s2['better']}/{s2['worse']} p {s2['p_one_sided']}")
    for g, r in res["registered"].items():
        print(f"===== {g}, the ninth version's registered rules (radar - 0.33 m, 50 % gate)")
        for lab, x in r["versions"].items():
            sp = x["speeds"]
            print(f"  {lab:5s} out {x['with_output']:3d} | <= 30 m {f(x['near30_gate'])} | rel {sp['relative_speed'].get('mae_kmh')} "
                  f"abs {sp['absolute_speed'].get('mae_kmh')} ego {sp['ego_per_bin'].get('mae_kmh')}")
        for k, p in r["paired"].items():
            q = p["distance_pct"]
            if q.get("n"):
                print(f"  {k}: n {q['n']} {q['old']} -> {q['new']} | segments {q['sign_test_by_segment']['better']}/"
                      f"{q['sign_test_by_segment']['worse']} p {q['sign_test_by_segment']['p_one_sided']}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

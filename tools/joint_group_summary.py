#!/usr/bin/env python3
"""Joint blind test (docs/JOINT_CHUNK8_PREREG.md, addendum 1 point 14): the registered groups from ONE run of the shared
scorer (`python -m mddbench score`), computed from that run's per-row files exactly as the scorer computes its own groups.

Written on 2026-10-20, before any truth of comma2k19 Chunk_8 existed. The shared scorer groups only by single
road_class/light labels, by car and by lowest true ego speed (results.json "by_group"). The registered groups are
  main       ca_highway/day + unknown/day   (Chunk_8: 126 segments; expectations 1-5 of section 3.1)
  local_day  ca_local/day                   (43; equals the scorer's own by_group.label["ca_local/day"]; expectation 8)
  night      light night, any road class    (39; expectation 9)
Main and night are taken here from the per-row files of the same scoring run (pairs.csv, ego_frames.csv, ego_bins.csv),
segment by segment, by the labels in segment.json, as addendum 1 point 14 registers. Labels are read as the scorer reads
them: f"{road_class}/{light}" from each listed segment's segment.json, a missing field = unknown (on Chunk_8 written
before prediction from splits/batches/comma2k19_chunk8_labels.csv and sealed by sha256); or from a labels csv
(segment_id,road_class,light); given both, they must agree. Other groups: --group NAME=RC/LIGHT[,...] (fnmatch
patterns, e.g. */night) or --group-file NAME=FILE (explicit segment ids, one per line).

For each group, the rows of the group's segments are selected and every number of the scorer's summarize is computed
again in the same way (own code, no import of the shared package; it follows the shared repository's mddbench/score.py
summarize, by_group, offset_sensitivity, _per_segment and mddbench/metrics.py dist_stats, speed_stats,
cluster_bootstrap, read at 3138e60, scoring code last changed in 32bf7ec, D13). The config is the scoring run's own
(results.json "config"):
  distance        pairs with a predicted distance; near = truth <= near_max_m (30 m), far > 30 m; own lane
                  (truth_lane 0) / adjacent lanes; truth bins (lo, hi] of dist_bins_m; unflagged / flagged by the method;
                  the 50 % gate (reported only, with the number dropped); per set n, median |e| %, median e % (bias),
                  p90 |e| %, share |e| > 25 %, median |pred - truth| m, and the 95 % coverage only when the method states
                  a half-width (method B states none); e = (pred - truth) / truth x 100
  speeds          relative and absolute speed on ALL pairs of the truth band (with or without a distance, so a row
                  whose distance was left empty still counts): MAE, bias, median, p90, Pearson r, slope of pred on
                  truth, 95 % coverage as a fraction over rows with a half-width, guess-0 MAE (relative only)
  ego speed       1 s bins: MAE etc. over the bins with a value (>= 3 readings, truth of the same frames); coverage
                  = bins with a value / all truth bins of the group's segments, refused, missing and invalid segments
                  included; share within 3 km/h; per frame MAE etc. and 95 % coverage
  intervals       near distance median %, near relative MAE, ego per-second MAE: whole clusters (segments) drawn with
                  replacement among the clusters that have values for the metric, bootstrap.n = 2,000 draws, a fresh
                  numpy default_rng(bootstrap.seed = 20261016) for each metric and group, clusters in sorted order, the
                  statistic on the pooled values, linear 2.5 / 97.5 percentiles; None with fewer than 2 clusters
  group extras    segments, with output (status ok and valid), refused / missing / invalid, found rate = pairs / the
                  in-scope truth of the group's segments with output (all distances; as by_group), per-segment
                  medians, the truth-offset scenarios on the group's pairs, the matching columns and offsets seen
Rounding as the scorer: 4 decimals; found rate and the ego percentages 2; per-segment medians 3. The row files are
read with pandas' round-trip number parser, so every value is the double the scorer had in memory (the default parser
is one unit in the last place off for about a quarter of them, enough to move pairs across a bin edge).

Self-check, every run, before anything is written: the same code on all listed segments must reproduce the top level
of results.json (summarize keys, detection, per_segment, distance_offset_sensitivity), and on each single label and
each car it must reproduce results.json by_group.label / by_group.car, field by field, with 0 differences; otherwise
nothing is written and the exit status is 1 (--allow-mismatch writes anyway, marked). The lowest-true-ego-speed groups
are not checked: their membership needs each segment's truth over all frames, which no output file holds.

The 582 px matching sensitivity: the scorer re-runs the whole matching with cx 582 in memory and writes only
whole-set numbers (results.json matching_sensitivity: matched, found rate, distance, relative speed); there is no
per-row file, no group breakdown, no interval. So from a scoring run this program can give NO per-group 582 numbers;
it copies the whole-set block and says so in the output. Per-group 582 numbers (expectation 7) and the 30 m found
rate (its denominator, the in-scope truth within 30 m, is in no output file) come from tools/joint_own_score.py.
Development data only: a second scoring run with a config copy that differs only in camera_for_matching (cx 582, no
per-car or per-drive columns) does write per-row 582 files; with --sensitivity-run cx_582=DIR this program checks that
run against the main run (same config otherwise, same segments, identical ego files, whole-set numbers equal to the
main run's matching_sensitivity block) and reports the groups for it. On a blind batch the scorer refuses such a run
(the config hash must equal the seal's).

Reads only results.json, pairs.csv, ego_frames.csv and ego_bins.csv of the scoring folder, the segment.json files (or
the labels csv) and, when given, the scored list and the seal (whose segment.json hashes must match; required when the
run was scored with a seal). Never truth folders, zips or radar files. Refuses a folder without results.json.
Outputs (--out and --md), checked before anything is computed and again just before writing: a file that exists is
never overwritten; refused inside the scoring folder, a --sensitivity-run folder, the segments root or the predictions
folder that results.json names; inside any git work tree (the shared repository, this repository or any other); and
inside any folder of one segment (segment.json, or meta.json + ego.csv + objects.csv) or any folder holding such
folders, i.e. a segments, prediction or truth folder, sealed or not, given or not (folders compared after resolving
links). Run it in the scorer's environment (~/venvs/depthbench: Python 3.13, numpy 2.5, pandas 3.0) so that every
number is computed by the same library code.

The registered expectations of section 3.1 (with addenda 1-3; registered for v9d) are evaluated on the reported,
rounded numbers when the groups are the registered ones: 1-5 main, 8 local_day, 9 night. 6 and 7 are not computed here.

  ~/venvs/depthbench/bin/python tools/joint_group_summary.py --score SCORE_DIR --segments SEGMENTS_ROOT [--labels LABELS.csv]
      [--segment-list SCORED_LIST.txt] [--seal SEAL.json] [--expect-count main=126 --expect-count local_day=43 ...]
      [--group NAME=RC/LIGHT[,RC/LIGHT...] ...] [--group-file NAME=IDS.txt ...] [--sensitivity-run cx_582=DIR]
      [--allow-mismatch] --out OUT.json [--md OUT.md]
  (without --group / --group-file: the three registered groups; registered expectations are evaluated only for them)
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROW_FILES = ("pairs.csv", "ego_frames.csv", "ego_bins.csv")
SCORER_FILES = ("results.json", "summary.md") + ROW_FILES
# kept as text: the scorer's reader keeps track_id, truth_id, flag and cls as text; ids and the car are text too
TEXT_COLS = ("segment_id", "cluster", "truth_id", "track_id", "flag", "cls", "car")
NEED_COLS = {
    "pairs.csv": ("segment_id", "cluster", "truth_lane", "d_truth", "d_pred", "dist_ci95_m", "rel_truth", "rel_pred",
                  "rel_ci95_kmh", "abs_truth", "abs_pred", "abs_ci95_kmh", "flag", "car", "truth_offset_m",
                  "match_cx_px"),
    "ego_frames.csv": ("segment_id", "cluster", "ego_kmh", "ego_ci95_kmh", "ego_truth"),
    "ego_bins.csv": ("segment_id", "cluster", "pred", "truth"),
}
SUMMARY_KEYS = ("distance", "relative_speed", "absolute_speed", "ego_speed", "bootstrap_95ci")
SCORER_CODE = "shared repository mddbench/score.py and mddbench/metrics.py at 3138e60 (scoring code last changed in 32bf7ec)"

# addendum 1 point 14
REGISTERED_GROUPS = {"main": ("ca_highway/day", "unknown/day"), "local_day": ("ca_local/day",), "night": ("*/night",)}
REGISTERED_SETTINGS = {"bootstrap": {"n": 2000, "seed": 20261016}, "bootstrap_cluster": "segment", "near_max_m": 30.0}
# section 3.1 (addenda 1-3 keep every threshold): (number, group, field, rule, bound, registered wording)
EXPECTATIONS = (
    ("1", "main", ("distance", "near_le_30m", "median_abs_pct"), "<=", 4.8, "median |error|, truth <= 30 m: <= 4.8 %"),
    ("1", "main", ("distance", "near_le_30m", "median_abs_m"), "<=", 0.95, "and <= 0.95 m"),
    ("1", "main", ("distance", "near_le_30m", "bias_pct"), "abs<=", 3.0, "median signed error within +-3 %"),
    ("2", "main", ("distance", "by_truth_distance_m", "10-20", "median_abs_m"), "<=", 0.75, "10-20 m: <= 0.75 m"),
    ("2", "main", ("distance", "by_truth_distance_m", "20-30", "median_abs_m"), "<=", 1.15, "20-30 m: <= 1.15 m"),
    ("3", "main", ("with_output_fraction",), ">=", 0.90, "segments with output >= 90 % of the group"),
    ("4", "main", ("relative_speed", "near", "mae"), "<=", 2.3, "relative speed, truth <= 30 m: MAE <= 2.3 km/h"),
    ("4", "main", ("relative_speed", "near", "coverage_95"), "within", (0.90, 0.99), "95 % range covers 90-99 %"),
    ("5", "main", ("ego_speed", "per_second", "mae"), "<=", 3.5, "ego speed per second: MAE <= 3.5 km/h"),
    ("5", "main", ("ego_speed", "per_frame", "coverage_95"), "within", (0.90, 0.99), "per-frame 95 % range covers 90-99 %"),
    ("5", "main", ("ego_speed", "per_second_coverage_pct"), "report", None, "seconds covered (refused included): reported"),
    ("8", "local_day", ("distance", "near_le_30m", "median_abs_pct"), "<=", 5.0, "median |error|, truth <= 30 m: <= 5.0 %"),
    ("9", "night", ("distance", "near_le_30m", "median_abs_pct"), "<=", 7.0, "median |error|, truth <= 30 m: <= 7.0 %"),
)
NOT_HERE = {"6": "v9d against v9 on the same readings: tools/joint_own_score.py (one shared seal per method, so v9 is "
                 "not scored by the shared scorer)",
            "7": "615 against 582 in the main group (30 m found rate, share > 25 %): tools/joint_own_score.py (the "
                 "scorer writes no per-row or per-group 582 output)"}


# ---------------------------------------------------------------- reading

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_list(path: Path) -> list[str]:
    """One id per line; blank and '#' lines skipped; a UTF-8 byte-order mark tolerated (as the scorer's lists)."""
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


def read_rows(path: Path, need: tuple) -> pd.DataFrame:
    """As the scorer's reader: only blank cells are missing ('NA' stays text), id and flag columns stay text.

    Numbers are parsed with pandas' round-trip parser. The scorer computes its own groups from the values in memory and
    writes each one as the shortest text that gives back the same double; pandas' default parser reads about a quarter
    of those texts one unit in the last place off (7,111 of 31,579 d_truth values of the development run), which moves
    pairs across a bin edge in the truth-offset scenarios (24 pairs at 60 m there). The round-trip parser reads every
    value exactly.
    """
    try:
        head = pd.read_csv(path, nrows=0).columns
    except pd.errors.EmptyDataError:
        raise SystemExit(f"refused: {path} is empty; the scorer always writes a header row") from None
    lack = [c for c in need if c not in head]
    if lack:
        raise SystemExit(f"refused: {path} lacks columns {lack}; not a file of the expected scorer")
    df = pd.read_csv(path, dtype={c: str for c in TEXT_COLS if c in head}, keep_default_na=False, na_values=[""],
                     float_precision="round_trip")
    if len(df) == 0:
        df = df.astype({c: float for c in df.columns if c not in TEXT_COLS})
    return df


def car_of(seg: dict) -> str:
    """The recording car as the scorer takes it: the device id at the start of source.route."""
    return str((seg.get("source") or {}).get("route", "")).split("|")[0]


# ---------------------------------------------------------------- the scorer's numbers, computed the same way

def r4(x) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def dist_stats(pred, truth, ci95=None) -> dict:
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    ok = np.isfinite(pred) & np.isfinite(truth) & (truth > 0)
    if not ok.any():
        return {"n": 0}
    p, t = pred[ok], truth[ok]
    e = (p - t) / t * 100.0
    out = {"n": int(len(e)), "median_abs_pct": r4(np.median(np.abs(e))), "bias_pct": r4(np.median(e)),
           "p90_abs_pct": r4(np.percentile(np.abs(e), 90)), "gross_pct": r4(np.mean(np.abs(e) > 25) * 100),
           "median_abs_m": r4(np.median(np.abs(p - t)))}
    if ci95 is not None:
        c = np.asarray(ci95, float)[ok]
        has = np.isfinite(c)
        if has.any():
            out["coverage_95"] = r4(np.mean(np.abs(p - t)[has] <= c[has]))
            out["n_with_ci"] = int(has.sum())
    return out


def speed_stats(pred, truth, ci95=None, guess_zero: bool = False) -> dict:
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    ok = np.isfinite(pred) & np.isfinite(truth)
    if len(pred) == 0 or not ok.any():
        return {"n": 0}
    p, t = pred[ok], truth[ok]
    err = p - t
    out = {"n": int(len(err)), "mae": r4(np.mean(np.abs(err))), "bias": r4(np.mean(err)),
           "median_abs": r4(np.median(np.abs(err))), "p90_abs": r4(np.percentile(np.abs(err), 90))}
    if len(err) >= 3 and np.std(t) > 0 and np.std(p) > 0:
        out["corr"] = r4(np.corrcoef(p, t)[0, 1])
        out["slope"] = r4(np.mean((t - t.mean()) * (p - p.mean())) / np.var(t))
    if ci95 is not None:
        c = np.asarray(ci95, float)[ok]
        has = np.isfinite(c)
        if has.any():
            out["coverage_95"] = r4(np.mean(np.abs(err[has]) <= c[has]))
            out["n_with_ci"] = int(has.sum())
    if guess_zero:
        out["guess0_mae"] = r4(np.mean(np.abs(t)))
    return out


def cluster_interval(values_by_cluster: dict, stat, n: int, seed: int) -> list | None:
    """95 % interval of stat(pooled values) when whole clusters are drawn with replacement."""
    keys = [k for k, v in values_by_cluster.items() if len(v)]
    if len(keys) < 2:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        vals.append(stat(np.concatenate([values_by_cluster[keys[i]] for i in pick])))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def by_cluster(df: pd.DataFrame) -> dict:
    return {k: g for k, g in df.groupby("cluster")} if len(df) else {}


def dset(df: pd.DataFrame) -> dict:
    return dist_stats(df["d_pred"], df["d_truth"], df["dist_ci95_m"])


def summarize(P: pd.DataFrame, EF: pd.DataFrame, EB: pd.DataFrame, cfg: dict, intervals: bool = True) -> dict:
    near_max = cfg["near_max_m"]
    flagged = P["flag"].notna() & (P["flag"].astype(str) != "")
    Pn, Pf = P[P["d_truth"] <= near_max], P[P["d_truth"] > near_max]
    Pd = P[P["d_pred"].notna()]
    near, far = Pd[Pd["d_truth"] <= near_max], Pd[Pd["d_truth"] > near_max]
    edges = cfg["dist_bins_m"]
    by_bin = {f"{lo:g}-{hi:g}": dset(Pd[(Pd["d_truth"] > lo) & (Pd["d_truth"] <= hi)])
              for lo, hi in zip(edges[:-1], edges[1:])}
    gate = cfg["sanity_gate_frac"]
    gated = near[(near["d_pred"] - near["d_truth"]).abs() <= gate * near["d_truth"]]

    def rel(df):
        return speed_stats(df["rel_pred"], df["rel_truth"], df["rel_ci95_kmh"], guess_zero=True)

    def absol(df):
        return speed_stats(df["abs_pred"], df["abs_truth"], df["abs_ci95_kmh"])

    eb_have = EB[EB["pred"].notna()]
    ci = None
    if intervals:
        boot = cfg["bootstrap"]
        ci = {
            "near_distance_median_abs_pct": cluster_interval(
                {k: np.abs((g["d_pred"] - g["d_truth"]) / g["d_truth"] * 100).values for k, g in by_cluster(near).items()},
                np.median, boot["n"], boot["seed"]),
            "near_relative_speed_mae": cluster_interval(
                {k: np.abs((g["rel_pred"] - g["rel_truth"]).dropna().values) for k, g in by_cluster(Pn).items()},
                np.mean, boot["n"], boot["seed"]),
            "ego_per_second_mae": cluster_interval(
                {k: np.abs((g["pred"] - g["truth"]).values) for k, g in by_cluster(eb_have).items()},
                np.mean, boot["n"], boot["seed"]),
        }
    return {
        "distance": {
            f"near_le_{near_max:g}m": dset(near),
            f"far_gt_{near_max:g}m": dset(far),
            "near_own_lane": dset(near[near["truth_lane"] == 0]),
            "near_adjacent_lanes": dset(near[near["truth_lane"] != 0]),
            "by_truth_distance_m": by_bin,
            "near_unflagged": dset(near[~flagged.loc[near.index]]),
            "near_flagged_by_method": dset(near[flagged.loc[near.index]]),
            f"near_sanity_gated_{int(gate * 100)}pct": {**dset(gated), "dropped": int(len(near) - len(gated))},
            "pairs_without_distance": int(P["d_pred"].isna().sum()),
        },
        "relative_speed": {"near": rel(Pn), "far": rel(Pf)},
        "absolute_speed": {"near": absol(Pn), "far": absol(Pf)},
        "ego_speed": {
            "per_second": speed_stats(EB["pred"], EB["truth"]),
            "per_second_coverage_pct": round(100 * EB["pred"].notna().mean(), 2) if len(EB) else None,
            "per_second_within_3kmh_pct": round(100 * float(np.mean(
                np.abs(eb_have["pred"] - eb_have["truth"]) <= 3)), 2) if len(eb_have) else None,
            "per_frame": speed_stats(EF["ego_kmh"], EF["ego_truth"], EF["ego_ci95_kmh"]),
        },
        "bootstrap_95ci": ci,
    }


def group_entry(P, EF, EB, ids: list, in_scope: dict, cfg: dict) -> dict:
    """One entry as the scorer's by_group writes it: summarize + segments + found_pct."""
    Pg = P[P["segment_id"].isin(ids)]
    r = summarize(Pg, EF[EF["segment_id"].isin(ids)], EB[EB["segment_id"].isin(ids)], cfg)
    n_truth = sum(in_scope.get(s, 0) for s in ids)
    r["segments"] = len(ids)
    r["found_pct"] = round(100 * len(Pg) / n_truth, 2) if n_truth else None
    return r


def offset_scenarios(P: pd.DataFrame, cfg: dict) -> list | None:
    """Distance errors under the other truth offsets, on the same pairs (only the truth distance moves)."""
    sens = cfg.get("truth_dist_offset_sensitivity_m")
    if not sens:
        return None
    Pd = P[P["d_pred"].notna()]
    cars = sorted(Pd["car"].astype(str).unique())
    if isinstance(sens, dict):
        lack = [c for c in cars if c not in sens]
        if lack:
            raise SystemExit(f"refused: truth_dist_offset_sensitivity_m has no list for car(s) {lack}")
        table = {c: list(sens[c]) for c in cars}
    else:
        table = {c: list(sens) for c in cars}
    if len({len(v) for v in table.values()}) > 1:
        raise SystemExit("refused: truth_dist_offset_sensitivity_m lists differ in length")
    near_max, edges = cfg["near_max_m"], cfg["dist_bins_m"]
    rows = []
    for k in range(len(next(iter(table.values()))) if table else 0):
        off = {c: float(v[k]) for c, v in table.items()}
        d = Pd.assign(d_truth=Pd["d_truth"] - Pd["truth_offset_m"] + Pd["car"].astype(str).map(off))
        rows.append({"offset_m": off, f"near_le_{near_max:g}m": dset(d[d["d_truth"] <= near_max]),
                     "by_truth_distance_m": {f"{lo:g}-{hi:g}": dset(d[(d["d_truth"] > lo) & (d["d_truth"] <= hi)])
                                             for lo, hi in zip(edges[:-1], edges[1:])}})
    return rows


def per_segment_block(P: pd.DataFrame, rows: list, cfg: dict) -> dict:
    """rows: [{segment_id, n_pairs, truth_in_scope}] of segments with output, in the scorer's order."""
    near = P[(P["d_truth"] <= cfg["near_max_m"]) & P["d_pred"].notna()]
    out = []
    for s in rows:
        g = near[near["segment_id"] == s["segment_id"]]
        med = (float(np.median(np.abs((g["d_pred"] - g["d_truth"]) / g["d_truth"] * 100)))
               if len(g) >= cfg["min_pairs_per_segment"] else None)
        out.append({**s, "near_pairs": int(len(g)), "near_median_abs_pct": None if med is None else round(med, 3)})
    meds = [r["near_median_abs_pct"] for r in out if r["near_median_abs_pct"] is not None]
    return {"rows": out, "segments_with_enough_pairs": len(meds),
            "median_of_segment_medians": round(float(np.median(meds)), 3) if meds else None}


def sensitivity_config(cfg: dict, v: dict) -> dict:
    """The config the scorer re-matches with (score.py matching_sensitivity)."""
    return dict(cfg, camera_for_matching={**cfg["camera_for_matching"], "cx": float(v["cx"]),
                                          "cx_by_car": {}, "cx_by_route": {}})


def sensitivity_entry(Pv, EF, EB, cfg_v: dict, cx: float, n_truth: int) -> dict:
    r = summarize(Pv, EF, EB, cfg_v, intervals=False)
    return {"cx": float(cx), "matched": int(len(Pv)),
            "found_pct": round(100 * len(Pv) / n_truth, 2) if n_truth else None,
            "distance": r["distance"], "relative_speed": r["relative_speed"]}


# ---------------------------------------------------------------- checks and helpers

def diff(mine, theirs, path: str = "") -> list[str]:
    """Every field that differs (exact equality; key order ignored)."""
    if isinstance(mine, dict) and isinstance(theirs, dict):
        out = []
        for k in list(mine) + [k for k in theirs if k not in mine]:
            if k not in mine or k not in theirs:
                out.append(f"{path}/{k}: only in {'this program' if k in mine else 'results.json'}")
            else:
                out += diff(mine[k], theirs[k], f"{path}/{k}")
        return out
    if isinstance(mine, (list, tuple)) and isinstance(theirs, (list, tuple)):
        if len(mine) != len(theirs):
            return [f"{path}: {len(mine)} items here, {len(theirs)} in results.json"]
        out = []
        for i, (a, b) in enumerate(zip(mine, theirs)):
            out += diff(a, b, f"{path}[{i}]")
        return out
    if mine is None or theirs is None:
        return [] if mine is None and theirs is None else [f"{path}: {mine!r} here, {theirs!r} in results.json"]
    return [] if mine == theirs else [f"{path}: {mine!r} here, {theirs!r} in results.json"]


def get_path(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def judge(value, rule: str, bound):
    if rule == "report":
        return "reported"
    if value is None:
        return "no value"
    ok = {"<=": lambda: value <= bound, ">=": lambda: value >= bound, "abs<=": lambda: abs(value) <= bound,
          "within": lambda: bound[0] <= value <= bound[1]}[rule]()
    return "met" if ok else "not met"


def counts(series: pd.Series) -> dict:
    return {f"{k:g}" if isinstance(k, float) else str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().items()}


def ids_sha256(ids) -> str:
    return hashlib.sha256("".join(f"{s}\n" for s in sorted(ids)).encode()).hexdigest()


def json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON: {type(o)}")


# ---------------------------------------------------------------- where the outputs may go

SEGMENT_FILE_SETS = (("segment.json",), ("meta.json", "ego.csv", "objects.csv"))


def is_work_tree(d: Path) -> bool:
    """A git work tree root: .git/HEAD, or a .git file pointing elsewhere (an empty .git folder is not one)."""
    g = d / ".git"
    try:
        if (g / "HEAD").is_file():
            return True
        return g.is_file() and g.read_text(encoding="utf-8", errors="replace").startswith("gitdir:")
    except OSError:
        return False


def segment_files_in(d: Path) -> str | None:
    """The files that make d one segment's folder (segments, predictions or truth), or None."""
    for names in SEGMENT_FILE_SETS:
        try:
            if all((d / n).is_file() for n in names):
                return ", ".join(names)
        except OSError:
            return None
    return None


def output_problem(target: Path, inputs: dict) -> str | None:
    """Why target may not be written, or None. Refused: a place inside a given input folder; inside any git work
    tree (the shared repository, this repository or any other); inside one segment's folder (segment.json, or
    meta.json + ego.csv + objects.csv: a segments, prediction or truth folder) or a folder holding such folders (a
    segments, prediction-version or truth root, sealed or not). Folders are compared after resolving links."""
    real = Path(os.path.realpath(target))
    for what, p in inputs.items():
        rp = Path(os.path.realpath(p))
        if real == rp or rp in real.parents:
            return f"{target} is inside {what} ({p})"
    for d in (real, *real.parents):
        if not d.is_dir():
            continue
        if is_work_tree(d):
            return f"{target} is inside the git work tree {d}; outputs are written outside every repository"
        held = segment_files_in(d)
        if held:
            return f"{target} is inside {d}, one segment's folder ({held})"
        try:
            children = sorted(c for c in d.iterdir() if c.is_dir())
        except OSError:
            continue
        for c in children:
            held = segment_files_in(c)
            if held:
                return (f"{target} is inside {d}, which holds segment folders (e.g. {c.name}: {held}); a segments, "
                        "prediction or truth folder is never written to")
    return None


def parse_pairs(args: list, what: str) -> list[tuple[str, str]]:
    out = []
    for a in args:
        name, eq, val = a.partition("=")
        if not eq or not name.strip() or not val.strip():
            raise SystemExit(f"refused: {what} wants NAME=VALUE, got {a!r}")
        out.append((name.strip(), val.strip()))
    return out


# ---------------------------------------------------------------- labels and groups

def load_labels(listed: list, segments_root: Path | None, labels_csv: Path | None) -> tuple[dict, dict]:
    """{segment_id: {label, car[, sha256]}} and a description of the source."""
    info, src = {}, {}
    if segments_root is not None:
        for s in listed:
            p = segments_root / s / "segment.json"
            if not p.is_file():
                raise SystemExit(f"refused: no {p}")
            seg = json.loads(p.read_text(encoding="utf-8"))
            if seg.get("segment_id", s) != s:
                raise SystemExit(f"refused: {p} names segment {seg.get('segment_id')!r}")
            info[s] = {"label": f"{seg.get('road_class', 'unknown')}/{seg.get('light', 'unknown')}",
                       "car": car_of(seg), "sha256": sha256_file(p)}
        lines = "".join(f"{s} {info[s]['sha256']}\n" for s in sorted(listed))
        src.update(segments_root=str(segments_root),
                   segment_json_set_sha256=hashlib.sha256(lines.encode()).hexdigest(),
                   segment_json_set_rule="sha256 of one line per segment, 'segment_id sha256(segment.json)\\n', ids sorted")
    if labels_csv is not None:
        with open(labels_csv, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        if not rows or not {"segment_id", "road_class", "light"} <= set(rows[0]):
            raise SystemExit(f"refused: {labels_csv} needs the columns segment_id,road_class,light")
        tab = {}
        for r in rows:
            sid = (r["segment_id"] or "").strip()
            if sid in tab:
                raise SystemExit(f"refused: {labels_csv} lists {sid} more than once")
            rc, li = (r["road_class"] or "").strip(), (r["light"] or "").strip()
            if not sid or not rc or not li:
                raise SystemExit(f"refused: {labels_csv} has an empty cell ({r})")
            tab[sid] = f"{rc}/{li}"
        lack = [s for s in listed if s not in tab]
        if lack:
            raise SystemExit(f"refused: {labels_csv} has no label for {len(lack)} listed segments, e.g. {lack[0]}")
        if info:
            differ = [s for s in listed if info[s]["label"] != tab[s]]
            if differ:
                raise SystemExit(f"refused: segment.json and {labels_csv} disagree on {len(differ)} segments, "
                                 f"e.g. {differ[0]}: {info[differ[0]]['label']} vs {tab[differ[0]]}")
        else:
            # without segment.json the car is the device id that starts a comma2k19 segment id
            info = {s: {"label": tab[s], "car": s.split("_")[0]} for s in listed}
        src.update(labels_csv=str(labels_csv), labels_csv_sha256=sha256_file(labels_csv),
                   labels_csv_rows_not_listed=len(set(tab) - set(listed)))
    src["used_for_labels"] = "segment.json" if segments_root is not None else "labels csv"
    src["used_for_car"] = "segment.json source.route" if segments_root is not None else "segment id prefix"
    return info, src


def build_groups(a, listed: list, labels: dict) -> dict:
    """{name: {"definition": {...}, "ids": sorted ids, "registered": bool}}"""
    groups = {}
    for name, val in parse_pairs(a.group, "--group"):
        pats = [p.strip() for p in val.split(",") if p.strip()]
        bad = [p for p in pats if p.count("/") != 1 or "" in p.split("/")]
        if not pats or bad:
            raise SystemExit(f"refused: --group {name}: patterns are ROAD_CLASS/LIGHT (fnmatch), got {bad or val!r}")
        if name in groups:
            raise SystemExit(f"refused: group {name} defined twice")
        ids = sorted(s for s in listed if any(fnmatch.fnmatchcase(labels[s]["label"], p) for p in pats))
        groups[name] = {"definition": {"labels": pats}, "ids": ids,
                        "registered": name in REGISTERED_GROUPS and tuple(pats) == REGISTERED_GROUPS[name]}
    known = set(listed)
    for name, val in parse_pairs(a.group_file, "--group-file"):
        if name in groups:
            raise SystemExit(f"refused: group {name} defined twice")
        p = Path(val)
        ids = read_list(p)
        if len(set(ids)) != len(ids):
            raise SystemExit(f"refused: {p} lists a segment more than once")
        strange = [s for s in ids if s not in known]
        if strange:
            raise SystemExit(f"refused: {p} lists {len(strange)} segments that the scoring run did not list, "
                             f"e.g. {strange[0]}")
        groups[name] = {"definition": {"file": str(p), "file_sha256": sha256_file(p)}, "ids": sorted(ids),
                        "registered": False}
    if not groups:
        for name, pats in REGISTERED_GROUPS.items():
            ids = sorted(s for s in listed if any(fnmatch.fnmatchcase(labels[s]["label"], p) for p in pats))
            groups[name] = {"definition": {"labels": list(pats)}, "ids": ids, "registered": True}
    return groups


# ---------------------------------------------------------------- report

def fmt(x) -> str:
    if x is None:
        return "n/a"
    return f"{x:.10g}" if isinstance(x, float) else str(x)


def fmt_ci(ci) -> str:
    return "n/a" if not ci else f"[{fmt(ci[0])}, {fmt(ci[1])}]"


def render_md(out: dict) -> str:
    sr, sc = out["scoring_run"], out["self_check"]
    near_key = out["near_key"]
    L = ["# Registered groups from one shared scoring run", "",
         f"- scoring folder: `{sr['folder']}` (predictions `{sr['predictions']}`)",
         f"- config sha256 `{sr['provenance'].get('config_sha256')}`; seal `{sr['provenance'].get('seal_file')}`; "
         f"shared repository commit `{(sr['provenance'].get('benchmark_repo') or {}).get('commit')}`",
         f"- labels: {out['labels']['used_for_labels']}"
         + (f", set sha256 `{out['labels']['segment_json_set_sha256']}`" if out["labels"].get("segment_json_set_sha256") else ""),
         f"- self-check against results.json: {sc['differences']} differences "
         f"({', '.join(f'{k} {v}' for k, v in sc['checked'].items())})"
         + ("" if sc["ok"] else " -- MISMATCH, written with --allow-mismatch"),
         f"- program sha256 `{out['program_sha256']}`", "",
         "| Group | Segments | With output | Near n | Near median % [95% CI] | Median m | Bias % | > 25 % | "
         "10-20 m (m) | 20-30 m (m) | Rel. MAE km/h [95% CI] | Rel. cov. | Ego/s MAE km/h [95% CI] | Ego s cov. % | "
         "Frame cov. | Found % |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, g in out["groups"].items():
        m = g["metrics"]
        d, b, ci = m["distance"][near_key], m["distance"]["by_truth_distance_m"], m["bootstrap_95ci"] or {}
        rel, eg = m["relative_speed"]["near"], m["ego_speed"]
        L.append(f"| {name} | {m['segments']} | {g['with_output']} | {d.get('n', 0)} | "
                 f"{fmt(d.get('median_abs_pct'))} {fmt_ci(ci.get('near_distance_median_abs_pct'))} | "
                 f"{fmt(d.get('median_abs_m'))} | {fmt(d.get('bias_pct'))} | {fmt(d.get('gross_pct'))} | "
                 f"{fmt((b.get('10-20') or {}).get('median_abs_m'))} | {fmt((b.get('20-30') or {}).get('median_abs_m'))} | "
                 f"{fmt(rel.get('mae'))} {fmt_ci(ci.get('near_relative_speed_mae'))} | {fmt(rel.get('coverage_95'))} | "
                 f"{fmt(eg['per_second'].get('mae'))} {fmt_ci(ci.get('ego_per_second_mae'))} | "
                 f"{fmt(eg['per_second_coverage_pct'])} | {fmt(eg['per_frame'].get('coverage_95'))} | {fmt(m['found_pct'])} |")
    L.append("")
    ex = out.get("expectations")
    if ex and ex.get("evaluated"):
        L += [f"Registered expectations (section 3.1; registered for v9d; this run scored `{sr['predictions']}`), "
              "on the reported numbers:", "",
              "| # | Group | Field | Value | Rule | Result |", "|---|---|---|---|---|---|"]
        for e in ex["parts"]:
            rule = e["rule"] if e["bound"] is None else f"{e['rule']} {e['bound']}"
            L.append(f"| {e['number']} | {e['group']} | {e['field']} | {fmt(e['value'])} | {rule} | {e['result']} |")
        L += ["", "By number: " + ", ".join(f"{k} {v}" for k, v in ex["by_number"].items()), ""]
    elif ex:
        L += [f"Registered expectations not evaluated: {ex['why']}", ""]
    ms = out["matching_sensitivity"]
    L.append(f"Matching sensitivity: {ms['note']}")
    for name, w in (ms.get("whole_set_from_results_json") or {}).items():
        wd = w["distance"][near_key]
        L.append(f"- whole listed set, {name} ({w['cx']:g} px), from results.json: matched {w['matched']}, found "
                 f"{fmt(w['found_pct'])} % (main matching {fmt(sr['detection'].get('found_pct'))} %), near median "
                 f"{fmt(wd.get('median_abs_pct'))} %, > 25 % {fmt(wd.get('gross_pct'))} %, near relative MAE "
                 f"{fmt(w['relative_speed']['near'].get('mae'))} km/h")
    for name, per in (ms.get("per_group") or {}).items():
        for gname, w in per.items():
            wd = w["distance"][near_key]
            L.append(f"- {gname}, {name} (from --sensitivity-run): matched {w['matched']}, found {fmt(w['found_pct'])} %, "
                     f"near n {wd.get('n', 0)}, near median {fmt(wd.get('median_abs_pct'))} % "
                     f"({fmt(wd.get('median_abs_m'))} m), > 25 % {fmt(wd.get('gross_pct'))} %")
    L += ["", "Not available from the scoring outputs: " + "; ".join(out["not_available"]), ""]
    return "\n".join(L)


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--score", required=True, help="folder written by `python -m mddbench score`")
    ap.add_argument("--segments", default=None, help="segments root: <segment_id>/segment.json")
    ap.add_argument("--labels", default=None, help="labels csv (segment_id,road_class,light); with --segments they must agree")
    ap.add_argument("--segment-list", default=None, help="the scored list (e.g. the batch list); must equal the run's listed segments")
    ap.add_argument("--seal", default=None, help="the seal the run was scored with (required when it was): segment.json hashes must match")
    ap.add_argument("--group", action="append", default=[], metavar="NAME=RC/LIGHT[,RC/LIGHT...]")
    ap.add_argument("--group-file", action="append", default=[], metavar="NAME=FILE")
    ap.add_argument("--expect-count", action="append", default=[], metavar="NAME=N", help="refuse unless the group has N segments")
    ap.add_argument("--sensitivity-run", action="append", default=[], metavar="NAME=DIR",
                    help="development only: a scoring folder of the same predictions with the config's matching sensitivity NAME as its main matching")
    ap.add_argument("--allow-mismatch", action="store_true", help="write even if the self-check finds differences (marked)")
    ap.add_argument("--out", required=True, help="output JSON")
    ap.add_argument("--md", default=None, help="markdown table (default: --out with .md)")
    a = ap.parse_args(argv)

    score_dir = Path(a.score)
    if not (score_dir / "results.json").is_file():
        raise SystemExit(f"refused: {score_dir} has no results.json; give the folder `mddbench score --out` wrote")
    for f in ROW_FILES:
        if not (score_dir / f).is_file():
            raise SystemExit(f"refused: {score_dir} has no {f}")
    if a.segments is None and a.labels is None:
        raise SystemExit("refused: give --segments (segment.json files) or --labels (labels csv) for the groups")

    results = json.loads((score_dir / "results.json").read_text(encoding="utf-8"))
    # ---- outputs: never inside an input, a repository or a segments / prediction / truth folder; never overwritten
    out_json = Path(a.out)
    out_md = Path(a.md) if a.md else out_json.with_suffix(".md")
    if os.path.realpath(out_json) == os.path.realpath(out_md):
        raise SystemExit(f"refused: --out and --md are the same file ({out_json})")
    inputs = {"the scoring folder": score_dir}
    if isinstance(results, dict) and isinstance(results.get("predictions"), str):
        inputs["the predictions scored by this run (results.json)"] = Path(results["predictions"])
    if a.segments:
        inputs["the segments root"] = Path(a.segments)
    for name, d in parse_pairs(a.sensitivity_run, "--sensitivity-run"):
        inputs[f"the scoring folder of --sensitivity-run {name}"] = Path(d)
    for p in (out_json, out_md):
        if os.path.lexists(p):
            raise SystemExit(f"refused: {p} exists; an output is never overwritten (give a new file name)")
        why = output_problem(p, inputs)
        if why:
            raise SystemExit(f"refused: {why}")
    for k in SUMMARY_KEYS + ("config", "segments", "detection", "per_segment", "by_group"):
        if k not in results:
            raise SystemExit(f"refused: {score_dir}/results.json has no '{k}'; not a scoring folder")
    cfg = results["config"]
    near_key = f"near_le_{cfg['near_max_m']:g}m"
    P = read_rows(score_dir / "pairs.csv", NEED_COLS["pairs.csv"])
    EF = read_rows(score_dir / "ego_frames.csv", NEED_COLS["ego_frames.csv"])
    EB = read_rows(score_dir / "ego_bins.csv", NEED_COLS["ego_bins.csv"])

    # ---- the listed segments and their status, from results.json
    segs = results["segments"]
    rows = results["per_segment"]["rows"]
    with_output = [r["segment_id"] for r in rows]
    refused, missing, invalid = dict(segs.get("refused") or {}), list(segs.get("missing") or []), dict(segs.get("invalid") or {})
    parts = [with_output, list(refused), missing, list(invalid)]
    listed = sorted(s for part in parts for s in part)
    if len(set(listed)) != len(listed) or len(listed) != segs.get("listed") or len(with_output) != segs.get("with_output"):
        raise SystemExit("refused: results.json segments (with output, refused, missing, invalid) do not add up to 'listed'")
    status = {s: "with_output" for s in with_output}
    for name, part in (("refused", refused), ("missing", missing), ("invalid", invalid)):
        status.update({s: name for s in part})
    out_set, listed_set = set(with_output), set(listed)
    for name, df, allowed in (("pairs.csv", P, out_set), ("ego_frames.csv", EF, out_set), ("ego_bins.csv", EB, listed_set)):
        extra = set(df["segment_id"]) - allowed
        if extra:
            raise SystemExit(f"refused: {name} has rows of {len(extra)} segments outside results.json, e.g. {sorted(extra)[0]}")
    if a.segment_list:
        want = read_list(Path(a.segment_list))
        if len(set(want)) != len(want) or sorted(want) != listed:
            raise SystemExit(f"refused: {a.segment_list} does not list the same segments as the scoring run "
                             f"({len(want)} vs {len(listed)})")

    labels, label_src = load_labels(listed, Path(a.segments) if a.segments else None, Path(a.labels) if a.labels else None)
    prov = results.get("provenance") or {}
    seal_info = None
    if prov.get("seal_sha256") and not a.seal:
        raise SystemExit("refused: the run was scored with a seal; pass --seal (that seal) so the segment.json files "
                         "used for the groups are proven to be the sealed ones")
    if a.seal:
        if a.segments is None:
            raise SystemExit("refused: --seal checks segment.json hashes and needs --segments")
        seal_path = Path(a.seal)
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        if not isinstance(seal, dict) or "sealed_utc" not in seal or "segments" not in seal:
            raise SystemExit(f"refused: {seal_path} is not a seal record")
        seal_sha = sha256_file(seal_path)
        if prov.get("seal_sha256") and prov["seal_sha256"] != seal_sha:
            raise SystemExit(f"refused: the run was scored with another seal ({prov.get('seal_file')})")
        hashes = seal.get("segment_json_sha256") or {}
        bad = [s for s in listed if hashes.get(s) != labels[s]["sha256"]]
        if bad or sorted(seal.get("segments") or []) != listed:
            raise SystemExit(f"refused: {seal_path}: segments or segment.json hashes differ from the files given "
                             f"({len(bad)} hashes differ)")
        seal_info = {"file": str(seal_path), "sha256": seal_sha, "sealed_utc": seal.get("sealed_utc"),
                     "method": seal.get("method"), "version": seal.get("version"),
                     "segment_json_sha256": "all listed segments equal"}

    # ---- self-check against what the scorer wrote
    differences, checked = [], {}
    in_scope = {r["segment_id"]: r["truth_in_scope"] for r in rows}
    n_pairs = P["segment_id"].value_counts().to_dict()
    whole = summarize(P, EF, EB, cfg)
    differences += diff(whole, {k: results[k] for k in SUMMARY_KEYS}, "whole")
    truth_total = sum(in_scope.values())
    det = {"truth_in_scope": truth_total, "matched": int(len(P)),
           "found_pct": round(100 * len(P) / truth_total, 2) if truth_total else None}
    differences += diff(det, {k: results["detection"].get(k) for k in det}, "detection")
    seg_rows = [{"segment_id": r["segment_id"], "n_pairs": int(n_pairs.get(r["segment_id"], 0)),
                 "truth_in_scope": r["truth_in_scope"]} for r in rows]
    differences += diff(per_segment_block(P, seg_rows, cfg), results["per_segment"], "per_segment")
    differences += diff(offset_scenarios(P, cfg), results.get("distance_offset_sensitivity"), "offset_sensitivity")
    checked["whole set"] = 1
    by_label, by_car = {}, {}
    for s in listed:
        by_label.setdefault(labels[s]["label"], []).append(s)
        by_car.setdefault(labels[s]["car"], []).append(s)
    for dim, members in (("label", by_label), ("car", by_car)):
        theirs = results["by_group"].get(dim) or {}
        if set(members) != set(theirs):
            differences.append(f"by_group/{dim}: groups here {sorted(members)}, in results.json {sorted(theirs)}")
        for g in sorted(set(members) & set(theirs)):
            differences += diff(group_entry(P, EF, EB, members[g], in_scope, cfg), theirs[g], f"by_group/{dim}/{g}")
        checked[f"{dim} groups"] = len(set(members) & set(theirs))

    # ---- development only: a separate scoring run whose main matching is a matching sensitivity
    sens_runs = {}
    sens_cfgs = {v["name"]: v for v in (cfg.get("matching_sensitivity") or [])}
    for name, d in parse_pairs(a.sensitivity_run, "--sensitivity-run"):
        if name not in sens_cfgs:
            raise SystemExit(f"refused: {name} is not a matching sensitivity of this config ({sorted(sens_cfgs)})")
        sdir = Path(d)
        if not (sdir / "results.json").is_file():
            raise SystemExit(f"refused: {sdir} has no results.json")
        sres = json.loads((sdir / "results.json").read_text(encoding="utf-8"))
        if sres.get("predictions") != results.get("predictions"):
            raise SystemExit(f"refused: {sdir} scored {sres.get('predictions')}, the main run {results.get('predictions')}")
        cfg_v = sensitivity_config(cfg, sens_cfgs[name])
        cdiff = diff(cfg_v, sres.get("config"), "config")
        if cdiff:
            raise SystemExit(f"refused: {sdir} was not scored with the {name} config (first difference {cdiff[0]})")
        if [r["segment_id"] for r in sres["per_segment"]["rows"]] != with_output or \
                {k: sres["segments"].get(k) for k in ("listed", "refused", "missing", "invalid")} != \
                {k: segs.get(k) for k in ("listed", "refused", "missing", "invalid")}:
            raise SystemExit(f"refused: {sdir} did not score the same segments with the same status")
        if {r["segment_id"]: r["truth_in_scope"] for r in sres["per_segment"]["rows"]} != in_scope:
            raise SystemExit(f"refused: {sdir} has other in-scope truth counts (scope does not depend on the column)")
        for f in ("ego_frames.csv", "ego_bins.csv"):
            if sha256_file(sdir / f) != sha256_file(score_dir / f):
                raise SystemExit(f"refused: {sdir / f} differs from the main run's (ego speed does not depend on matching)")
        Pv = read_rows(sdir / "pairs.csv", NEED_COLS["pairs.csv"])
        if set(Pv["segment_id"]) - out_set:
            raise SystemExit(f"refused: {sdir}/pairs.csv has rows of segments without output")
        theirs = (results.get("matching_sensitivity") or {}).get(name)
        mine = sensitivity_entry(Pv, EF, EB, cfg_v, sens_cfgs[name]["cx"], truth_total)
        differences += diff(mine, theirs, f"matching_sensitivity/{name}")
        checked[f"{name} run"] = 1
        sens_runs[name] = {"folder": str(sdir), "cfg_v": cfg_v, "P": Pv,
                           "files_sha256": {f: sha256_file(sdir / f) for f in ("results.json",) + ROW_FILES}}

    ok = not differences
    if not ok:
        print(f"self-check: {len(differences)} differences from results.json, e.g.", file=sys.stderr)
        for x in differences[:20]:
            print("  " + x, file=sys.stderr)
        if not a.allow_mismatch:
            print("nothing written (--allow-mismatch writes anyway)", file=sys.stderr)
            return 1

    # ---- the groups
    groups = build_groups(a, listed, labels)
    for name, n in parse_pairs(a.expect_count, "--expect-count"):
        if name not in groups:
            raise SystemExit(f"refused: --expect-count names no group {name}")
        if len(groups[name]["ids"]) != int(n):
            raise SystemExit(f"refused: group {name} has {len(groups[name]['ids'])} segments, expected {n}")
    out_groups = {}
    for name, g in groups.items():
        ids, idset = g["ids"], set(g["ids"])
        Pg = P[P["segment_id"].isin(ids)]
        st = Counter(status[s] for s in ids)
        metrics = group_entry(P, EF, EB, ids, in_scope, cfg)
        n_truth = sum(in_scope.get(s, 0) for s in ids)
        grows = [{"segment_id": r["segment_id"], "n_pairs": int(n_pairs.get(r["segment_id"], 0)),
                  "truth_in_scope": r["truth_in_scope"]} for r in rows if r["segment_id"] in idset]
        psb = per_segment_block(Pg, grows, cfg)
        out_groups[name] = {
            "definition": g["definition"],
            "registered_definition": g["registered"],
            "segments": len(ids),
            "with_output": st["with_output"],
            "with_output_fraction": (st["with_output"] / len(ids)) if ids else None,
            "refused": sorted(s for s in ids if status[s] == "refused"),
            "missing": sorted(s for s in ids if status[s] == "missing"),
            "invalid": sorted(s for s in ids if status[s] == "invalid"),
            "labels": dict(sorted(Counter(labels[s]["label"] for s in ids).items())),
            "metrics": metrics,
            "detection": {"truth_in_scope": n_truth, "matched": int(len(Pg)), "found_pct": metrics["found_pct"]},
            "per_segment": {k: psb[k] for k in ("segments_with_enough_pairs", "median_of_segment_medians")},
            "distance_offset_sensitivity": offset_scenarios(Pg, cfg),
            "match_cx_px_seen": counts(Pg["match_cx_px"]),
            "truth_offset_m_seen": counts(Pg["truth_offset_m"]),
            "segment_ids_sha256": ids_sha256(ids),
            "segment_ids": ids,
        }

    per_group_sens = {}
    for name, sr in sens_runs.items():
        per_group_sens[name] = {gname: sensitivity_entry(sr["P"][sr["P"]["segment_id"].isin(g["ids"])],
                                                         EF[EF["segment_id"].isin(g["ids"])],
                                                         EB[EB["segment_id"].isin(g["ids"])], sr["cfg_v"],
                                                         sens_cfgs[name]["cx"],
                                                         sum(in_scope.get(s, 0) for s in g["ids"]))
                                for gname, g in groups.items()}
    note = ("the scorer re-runs the matching for each matching sensitivity in memory and writes only whole-set numbers "
            "(results.json matching_sensitivity: matched, found rate, distance, relative speed); it writes no per-row "
            "file, no group breakdown and no interval for it, so per-group numbers cannot be derived from this scoring "
            "run; per-group 582 numbers come from tools/joint_own_score.py")
    if per_group_sens:
        note += ("; the per-group numbers below come from a separate development scoring run (--sensitivity-run), "
                 "checked against this run's whole-set block")
    if not sens_cfgs:
        note = "the config has no matching sensitivity"

    # ---- registered expectations (section 3.1)
    settings_ok = {k: cfg.get(k) == v for k, v in REGISTERED_SETTINGS.items()}
    edges = cfg["dist_bins_m"]
    settings_ok["dist_bins_m has 10-20 and 20-30"] = {"10-20", "20-30"} <= {f"{lo:g}-{hi:g}" for lo, hi in
                                                                           zip(edges[:-1], edges[1:])}
    need = sorted({e[1] for e in EXPECTATIONS})
    if not all(n in out_groups and out_groups[n]["registered_definition"] for n in need):
        expectations = {"evaluated": False, "why": "main, local_day and night are not all defined by the registered labels "
                                                   "(addendum 1 point 14); explicit segment files are not evaluated"}
    elif not all(settings_ok.values()):
        expectations = {"evaluated": False, "why": f"the scoring config differs from the registration: {settings_ok}"}
    else:
        parts_out, by_num = [], {}
        for num, gname, field, rule, bound, words in EXPECTATIONS:
            src = out_groups[gname] if field == ("with_output_fraction",) else out_groups[gname]["metrics"]
            value = get_path(src, field)
            res = judge(value, rule, bound)
            parts_out.append({"number": num, "group": gname, "field": ".".join(field), "value": value,
                              "rule": rule, "bound": list(bound) if isinstance(bound, tuple) else bound,
                              "registered": words, "result": res})
            if res != "reported":
                prev = by_num.get(num, "met")
                by_num[num] = "not met" if "not met" in (prev, res) else ("no value" if "no value" in (prev, res) else "met")
        for num, why in NOT_HERE.items():
            by_num[num] = "not computed here"
        expectations = {"evaluated": True, "registered_for": "v9d (docs/JOINT_CHUNK8_PREREG.md section 3.1)",
                        "this_run_predictions": results.get("predictions"),
                        "comparison": "on the reported numbers (rounded as the scorer rounds them); ranges inclusive",
                        "parts": parts_out, "by_number": dict(sorted(by_num.items(), key=lambda kv: int(kv[0]))),
                        "not_computed_here": NOT_HERE}

    out = {
        "program": "tools/joint_group_summary.py",
        "program_sha256": sha256_file(Path(__file__)),
        "purpose": "registered groups of docs/JOINT_CHUNK8_PREREG.md (addendum 1 point 14) from one shared scoring run",
        "written": "2026-10-20, before any truth of comma2k19 Chunk_8 existed",
        "follows": SCORER_CODE,
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
        "near_key": near_key,
        "scoring_run": {
            "folder": str(score_dir),
            "predictions": results.get("predictions"),
            "spec": results.get("spec"),
            "provenance": prov,
            "files_sha256": {f: sha256_file(score_dir / f) for f in ("results.json",) + ROW_FILES},
            "other_files": sorted(p.name for p in score_dir.iterdir() if p.name not in SCORER_FILES),
            "segments": {"listed": len(listed), "with_output": len(with_output), "refused": len(refused),
                         "missing": len(missing), "invalid": len(invalid), "split_status": segs.get("split_status"),
                         "note": segs.get("note")},
            "detection": results["detection"],
            "settings": {k: cfg.get(k) for k in ("near_max_m", "dist_bins_m", "bootstrap", "bootstrap_cluster",
                                                 "truth_dist_offset_m", "camera_for_matching", "matching_sensitivity")},
            "settings_as_registered": settings_ok,
        },
        "labels": {**label_src, "seal": seal_info, "segments_per_label": dict(sorted((k, len(v)) for k, v in by_label.items()))},
        "self_check": {"ok": ok, "differences": len(differences), "first_differences": differences[:50],
                       "checked": checked,
                       "not_checked": "by_group.min_truth_ego: membership needs each segment's lowest true ego speed "
                                      "over all frames, which no output file holds; this program does not read truth"},
        "groups": out_groups,
        "matching_sensitivity": {
            "note": note,
            "whole_set_from_results_json": results.get("matching_sensitivity"),
            "per_group": per_group_sens or None,
            "separate_runs": {k: {"folder": v["folder"], "files_sha256": v["files_sha256"]} for k, v in sens_runs.items()} or None,
        },
        "not_available": [
            "per-group matching sensitivity (582) from this run (see matching_sensitivity)",
            "found rate within 30 m (needs the in-scope truth within 30 m, which no output file holds; "
            "tools/joint_own_score.py)",
            "per-group boxes unmatched / matched to out-of-scope truth (whole-set totals only, results.json detection)",
            "per-group lowest-true-ego-speed split",
        ],
        "expectations": expectations,
    }
    texts = ((out_json, json.dumps(out, ensure_ascii=False, indent=1, default=json_default) + "\n"),
             (out_md, render_md(out)))
    for p, _ in texts:                  # checked again just before writing: nothing may have appeared meanwhile
        if os.path.lexists(p):
            raise SystemExit(f"refused: {p} exists; an output is never overwritten (nothing written)")
        why = output_problem(p, inputs)
        if why:
            raise SystemExit(f"refused: {why} (nothing written)")
    for p, text in texts:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "x", encoding="utf-8") as fh:      # "x": fails instead of overwriting
            fh.write(text)
    for name, g in out_groups.items():
        d = g["metrics"]["distance"][near_key]
        print(f"{name}: {g['segments']} segments, {g['with_output']} with output, near n {d.get('n', 0)}, "
              f"median {fmt(d.get('median_abs_pct'))} % ({fmt(d.get('median_abs_m'))} m)")
    print(f"self-check: {len(differences)} differences; wrote {out_json} and {out_md}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

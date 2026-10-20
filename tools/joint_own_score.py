#!/usr/bin/env python3
"""Own scoring of the joint blind batch (docs/JOINT_CHUNK8_PREREG.md): the settled shared rules, written here without
the shared scorer's code (mddbench is not imported), applied to the truth folders that the shared
`python -m mddbench truth-comma2k19` writes and to the shared-format prediction folders of one or more versions.

Written on 2026-10-20, after method B's seals (v9d with the shared seal, v9 and the tenth version with
tools/seal_predictions.py) and before any truth of comma2k19 Chunk_8 existed; no Chunk_8 truth was built, read or
searched for while it was written. It was checked against the shared scorer on synthetic data and on opened development
data (comma2k19 Chunk_7 second half, 101 segments, v9 and v9d, truth written by truth-comma2k19): its row tables were
byte-identical to those of the scorer's score (at 582 too, against a development copy of the config with column 582),
and its numbers equal to those of score (with its 582 re-matching) and compare on the same data. Registration points
it serves:
  * addendum 1, section C (the fallback way, which addendum 2 point 2 uses for expectations 6 and 7): numbers computed
    by this method's own program from the same truth that the shared truth-comma2k19 writes, marked as not computed by
    the shared scorer;
  * addendum 2, point 2: expectation 6 (v9d against v9 on the same readings: |difference of the median error| <= 0.3
    percentage points), expectation 7 (column 615 against 582 in the main group: a higher found rate, counted on the
    in-scope truth within 30 m as in addendum 1 point 18, and a lower share of errors > 25 %), and v9, which the shared
    scorer cannot score (one shared seal per method and batch) and which is reported only in this repository.
The joint result of v9d at the main column is the shared scorer's (addendum 2 point 1); this program computes it too, as
a check. Grouping the shared scorer's own row outputs (addendum 1 point 14) is a separate program.

Rules (shared docs/SPEC.md section 5 and docs/DECISIONS.md D1-D13). Every number comes from the scoring config
(configs/eval_comma2k19.json of the shared repository); it is refused while it has _pending items or no truth offset.
  truth       radar returns at dist_native_m >= truth_empty_slot_min_native_m (empty slots) are dropped; camera
              distance d = dist_native_m + truth_dist_offset_m of the car (the device id that starts source.route);
              d > min_truth_dist_m; in one frame, a return within truth_duplicate_dist_m and truth_duplicate_lat_m of an
              earlier kept return (order: frame, then truth_id as text) is dropped; in scope = |pose ego + radar
              relative speed| >= moving_min_abs_kmh and |lat| <= adjacent_max_abs_lat_m; lane 0 if |lat| <=
              lane_half_width_m, else the sign of lat (+ left); only the frames listed in the method's ego.csv count
  matching    by image position only, in every frame with boxes, against every prepared return of that frame: column
              u = cx - fx * lat / d; a pair is allowed when |box centre - u| <= match_k_width * box width + match_px and
              d * box width / fx lies in match_width_m; cost |box centre - u| / box width + match_dist_weight * d
              (blocked pairs cost 1e9), one-to-one by scipy's linear_sum_assignment; a box paired with an out-of-scope
              return is not scored; the predicted distance takes no part
  column cx   main matching: cx_by_route[route], else the last cx_by_car[car] entry whose first route time is <= the
              route time (text order), else cx; every matching_sensitivity entry re-runs the whole matching with its
              own cx for every segment (cx_by_car and cx_by_route cleared)
  distance    pairs with a predicted distance; near = truth <= near_max_m: median |error| % and m, signed median
              (bias), 90th percentile, share of |error| > 25 %; truth bins (lo, hi] of dist_bins_m; own and adjacent
              lanes; with and without the method's flag; the sanity_gate_frac gate is reported only
  speeds      relative speed (the radar value of the frame) and absolute speed (pose ego + radar relative speed) on
              every pair of the band (near: truth <= near_max_m; far), with or without a distance: MAE, bias, median,
              90th percentile, correlation, slope of the prediction on the truth, 95 % coverage (share within the
              stated half-width, over the rows that state one) and, for relative speed, the MAE of always answering 0
  ego         fixed bins of round(ego_bin_s * fps) frames of frame_idx; a bin counts with >= ego_min_readings_per_bin
              readings, against the mean pose speed of the same frames; coverage = bins with a value / every truth bin
              of every listed segment (refused, missing and invalid segments included); per frame MAE and 95 % coverage
  found rate  pairs / in-scope truth in the processed frames of the segments with output: all distances, within
              near_max_m (addendum 1 point 18), and per truth bin
  intervals   whole clusters (bootstrap_cluster: segments) drawn with replacement, bootstrap.n draws from a fresh
              numpy default_rng(bootstrap.seed) per metric, the statistic of the pooled values, 2.5 and 97.5
              percentiles: near distance median %, near relative speed MAE, ego per-second MAE
  paired      two versions on the same readings: same segment, frame and truth target, truth <= near_max_m, both with a
              distance (relative speed: both with a speed; ego: the same 1 s bins, both with a value); each version's
              numbers on the intersection, the difference a - b of the unrounded statistics (and of the 4-dp medians),
              a cluster bootstrap interval of the difference, and a two-sided sign test over the segments with >=
              min_pairs_per_segment common readings
Groups (registration section 2 and addendum 1 point 14; road_class and light of segment.json, written before any
prediction and sealed): main = day with road_class ca_highway or unknown; local_day = day with ca_local; night = night,
any road class; all = every listed segment.

Access checks, all before any truth value is read:
  seals       --seal-check takes seal files of two kinds. Shared (mddbench seal: "files" keyed by <segment>/<file>
              under the version folder, with method, version and segments): the folder must hash to it exactly
              (nothing changed, missing or added) and every meta.json must carry the seal's method and version.
              Own (tools/seal_predictions.py: "files" keyed by file name; the shared-format folders are pinned by
              manifest_pred_<v>.txt (as run.py wrote them) and manifest_blank_<v>.txt (after tools/joint_blank_dist.py),
              in sha256sum format, written by `find . -type f | LC_ALL=C sort | xargs -d '\\n' sha256sum` in the
              version folder): a manifest is used only if its sha256 equals the seal's entry (it is looked for in
              --manifest-dir, the seal's folder and that folder's subfolders), and the folder must match it exactly.
              Every version must be covered by at least one seal, or --no-seal must be given (development data only).
              The label of --version NAME=ROOT must equal the version in the folder's meta.json files.
  blind data  a segment whose segment.json source.chunk is blind (Chunk_8, Chunk_9 and Chunk_10 as in the shared
              splits/comma2k19.json, plus any chunk --split-table marks blind; a split table cannot make these three
              dev), or --blind: seals are required; a version covered only by a manifest_pred_* manifest (distances
              not yet left empty, not the registered input) is refused; at least one shared seal must be given, and
              the scored segment list, every segment.json sha256 and the config sha256 must equal what each shared
              seal records; every truth folder must have been built after every given seal (meta.json built_utc,
              else the oldest modification time of its three files). A seal that covers no scored folder (method A's,
              or the tenth version's) only adds its time to that check (and, if shared, its segment list, segment.json
              and config records).
  every seal  truth is built only after every version of both methods is sealed (addendum 2 point 3, shared D13), so
              for a blind chunk the program itself requires every seal registered for it in REGISTERED_SEALS (written
              before any truth of the batch): pinned by sha256, or, for a seal kept in method A's own repository whose
              hash was not known when this was written, by file name (only its sealed_utc is read). A registered seal
              that is not given is refused; only an entry marked waivable can be left out with
              --waive-registered-seal ID, which is recorded in results.json and summary.md. A blind chunk with no
              registered list is refused. Besides, every other seal file (*.json with sealed_utc) in the folder of a
              given seal that names any scored segment must be given too. So the time check covers every seal; that
              the seals were pushed before the truth is not checked here (the push times are on GitHub). With --blind
              on segments of no blind chunk (a development test of the checks) the registered list does not apply.
  truth       every listed segment needs <truth>/<segment>/{meta.json, ego.csv, objects.csv}; meta.json source.chunk
              must exist and equal segment.json source.chunk (a segment.json without a chunk is refused when its truth
              is not in a dev chunk); a blocked chunk is refused.
  outputs     --out is checked before any seal or truth is read and again just before writing: it must not exist or
              be an empty folder (nothing is overwritten; every file is created with exclusive creation), and it is
              refused inside the truth root, the segments root, a version folder or a --manifest-dir; inside any git
              work tree (the shared repository, this repository or any other); and inside any folder of one segment
              (segment.json, or meta.json + ego.csv + objects.csv) or any folder holding such folders, i.e. a
              segments, prediction or truth folder, sealed or not, given or not (folders compared after resolving
              links). So writing the results cannot change a sealed folder.

Outputs in --out: results.json, summary.md (also printed) and, per version, pairs_<version>_<matching>.csv,
ego_frames_<version>.csv and ego_bins_<version>.csv with the shared scorer's columns, so that rows can be compared one
by one with its pairs.csv, ego_frames.csv and ego_bins.csv (pandas' default csv float parser can move the last bit of
a value; read with float_precision="round_trip" for exact values). Segments are processed in the order of
--segment-list (default: the segment folders in name order); the shared scorer pools rows in its list order too, so
give it the same list for numbers equal to the last bit. Run in the shared scorer's environment (~/venvs/depthbench:
Python 3.13, numpy 2.5.3, pandas 3.0.2, scipy 1.17.1).

  # the joint batch (from the repository root), only after every version of both methods is sealed and pushed and
  # the shared truth is built
  ~/venvs/depthbench/bin/python tools/joint_own_score.py \\
      --truth /mnt/d/depth_work/joint/chunk8/truth --segments /mnt/d/depth_work/joint/chunk8/segments \\
      --config /mnt/d/team_repo/configs/eval_comma2k19.json \\
      --segment-list /mnt/d/team_repo/splits/batches/comma2k19_chunk8.txt \\
      --split-table /mnt/d/team_repo/splits/comma2k19.json \\
      --version v9d=/mnt/d/depth_work/joint/chunk8/predictions_blank/B_depth_lane_ruler/v9d \\
      --version v9=/mnt/d/depth_work/joint/chunk8/predictions_blank/B_depth_lane_ruler/v9 \\
      --seal-check /mnt/d/team_repo/seals/B_depth_lane_ruler_v9d_comma2k19_chunk8.json \\
                   docs/seals/joint_chunk8_v9d_seal.json docs/seals/joint_chunk8_v9_seal.json \\
                   /mnt/d/team_repo/seals/A_depth_odometer_v2_comma2k19_chunk8.json docs/seals/blind_v10_seal.json \\
                   <a copy of method A's v2own_comma2k19_chunk8.json> \\
      --out <new folder outside every repository and every segments, prediction or truth folder>
  # (all six registered seals; without method A's v2own file only --waive-registered-seal A_v2own, which is recorded)
  # opened development data without seals
  ~/venvs/depthbench/bin/python tools/joint_own_score.py --truth T --segments S --config C --segment-list IDS \\
      --version v9d=P1 --version v9=P2 --no-seal --out OUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest

PROGRAM = Path(__file__).resolve()
REPO = PROGRAM.parents[1]

BLOCKED = 1e9                                       # cost of a box / return pair that fails a gate
TEXT_COLUMNS = ("track_id", "truth_id", "flag", "cls")
DEFAULT_BLIND_CHUNKS = ("Chunk_8", "Chunk_9", "Chunk_10")      # shared splits/comma2k19.json (3138e60)
# Every seal of a blind batch (docs/JOINT_CHUNK8_PREREG.md addendum 2 point 3, shared docs/DECISIONS.md D13: truth only
# after every version of both methods, including versions sealed only in a method's own repository, is sealed).
# Written 2026-10-20, after the last of these seals and before any truth of Chunk_8; sha256 None = not known then
# (matched by file name, only sealed_utc read); "waivable": may be left out with --waive-registered-seal ID (recorded).
REGISTERED_SEALS = {
    "Chunk_8": (
        {"id": "A_v2", "what": "method A v2, mddbench seal (shared repository 5f50714, 2026-10-19T15:10:09Z)",
         "file": "A_depth_odometer_v2_comma2k19_chunk8.json",
         "sha256": "fb37af3b724ce69dd6c786967daf9b5c1a00f3b7c7dc76e55ea2ae3541fa54af"},
        {"id": "A_v2own", "what": "method A v2own, the hash list kept in method A's own repository "
                                  "(pipeline/seals/v2own_comma2k19_chunk8.json; method A registration, addendum 1)",
         "file": "v2own_comma2k19_chunk8.json", "sha256": None, "waivable": True},
        {"id": "B_v9d_shared", "what": "method B v9d, mddbench seal (shared repository 3138e60, 2026-10-20T03:20:54Z)",
         "file": "B_depth_lane_ruler_v9d_comma2k19_chunk8.json",
         "sha256": "2f394421ac10d48dd2ea86793eb6acb0863a3e0a74bdc2b12d21582a10e638ec"},
        {"id": "B_v9d_own", "what": "method B v9d, tools/seal_predictions.py (9547bc0, 2026-10-20T03:21:25Z)",
         "file": "joint_chunk8_v9d_seal.json",
         "sha256": "58d3c83f4341fb79ffa0b6c7b28ecc3022c87a43b2b9acaac9ee3e67d7852d48"},
        {"id": "B_v9_own", "what": "method B v9, tools/seal_predictions.py (9547bc0, 2026-10-20T03:22:59Z)",
         "file": "joint_chunk8_v9_seal.json",
         "sha256": "9882bf7923801c72951946217c9d2ea717b4b5cd47ff5cb2972191757790b4dd"},
        {"id": "B_v10_own", "what": "method B tenth version, tools/seal_predictions.py (9547bc0, 2026-10-20T03:28:08Z)",
         "file": "blind_v10_seal.json",
         "sha256": "0e03f4d455769b61320cd76eab332382350597930026a8e8a1b9dc5fb75131c9"},
    ),
}
UNPINNED_SEALS = {e["file"]: e for entries in REGISTERED_SEALS.values() for e in entries if e["sha256"] is None}
SEGMENT_ID = re.compile(r"[0-9a-f]{16}_\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}_\d+")
SEGMENT_FILE_SETS = (("segment.json",), ("meta.json", "ego.csv", "objects.csv"))
EXPECTATION_6_MAX_PP = 0.3                          # registration section 3.1, item 6
NO_NUMBER_BEYOND_M = 60.0       # method B gives no number beyond 60 m (registration section 1); only counted here
MANIFEST_NAME = re.compile(r"manifest_(pred|blank)_(.+)\.txt")
MANIFEST_LINE = re.compile(r"([0-9a-f]{64}) [ *](.+)")
LABEL = re.compile(r"[A-Za-z0-9_.-]+")

PAIR_COLUMNS = ["segment_id", "cluster", "frame_idx", "truth_id", "track_id", "truth_lane", "du_px",
                "d_truth", "d_pred", "dist_ci95_m", "lat_truth", "lat_pred", "lane_pred",
                "rel_truth", "rel_pred", "rel_ci95_kmh", "abs_truth", "abs_pred", "abs_ci95_kmh",
                "ego_truth", "flag", "car", "truth_offset_m", "match_cx_px"]
EGO_FRAME_COLUMNS = ["segment_id", "cluster", "frame_idx", "ego_kmh", "ego_ci95_kmh", "flag", "ego_truth"]
BIN_COLUMNS = ["segment_id", "cluster", "bin", "truth_full", "pred", "truth", "n"]

# shared-format prediction files (shared docs/SPEC.md section 3)
META_KEYS = ("spec", "method", "version", "segment_id", "status", "code_commit")
SPEC_MAJOR = "0"
EGO_REQUIRED, EGO_OPTIONAL = ["frame_idx", "ego_kmh"], ["ego_ci95_kmh", "flag"]
OBJ_REQUIRED = ["frame_idx", "track_id", "x0", "y0", "x1", "y1", "dist_m"]
OBJ_OPTIONAL = ["lat_m", "lane", "rel_kmh", "abs_kmh", "dist_ci95_m", "rel_ci95_kmh", "abs_ci95_kmh", "cls", "flag"]

# registered groups: (definition, test on (road_class, light) of segment.json)
GROUPS = {
    "main": ("light day, road_class ca_highway or unknown", lambda rc, li: li == "day" and rc in ("ca_highway", "unknown")),
    "local_day": ("light day, road_class ca_local", lambda rc, li: li == "day" and rc == "ca_local"),
    "night": ("light night, any road_class", lambda rc, li: li == "night"),
    "all": ("every listed segment", lambda rc, li: True),
}


# ----------------------------------------------------------------------------------------------------------- basics
def refuse(message: str):
    raise SystemExit(f"refused: {message}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_ids(path: Path) -> list[str]:
    """One id per line; blank lines and '#' lines are skipped; a UTF-8 byte-order mark is tolerated."""
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def read_table(path: Path) -> pd.DataFrame:
    """A shared-format csv: blank cells are missing, id and flag columns stay text ('007' is not 7), and a header-only
    file gets float columns so that it reads like a file with rows."""
    head = pd.read_csv(path, nrows=0).columns
    df = pd.read_csv(path, dtype={c: str for c in TEXT_COLUMNS if c in head}, keep_default_na=False, na_values=[""])
    if len(df) == 0:
        df = df.astype({c: float for c in df.columns if c not in TEXT_COLUMNS})
    return df


def parse_utc(text) -> datetime:
    s = str(text).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    t = datetime.fromisoformat(s)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc)


def utc_text(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def r4(x):
    """4 decimals, None when not finite (the shared scorer's rounding of every metric)."""
    if x is None:
        return None
    x = float(x)
    return round(x, 4) if np.isfinite(x) else None


def pct2(num: int, den: int):
    return round(100.0 * num / den, 2) if den else None


# -------------------------------------------------------------------------------------------------------- metrics
def dist_stats(pred, truth, ci=None) -> dict:
    """e = (pred - truth) / truth * 100 per pair, over pairs with both finite and truth > 0."""
    p_all, t_all = np.asarray(pred, float), np.asarray(truth, float)
    ok = np.isfinite(p_all) & np.isfinite(t_all) & (t_all > 0)
    if not ok.any():
        return {"n": 0}
    p, t = p_all[ok], t_all[ok]
    e = (p - t) / t * 100.0
    out = {"n": int(e.size), "median_abs_pct": r4(np.median(np.abs(e))), "bias_pct": r4(np.median(e)),
           "p90_abs_pct": r4(np.percentile(np.abs(e), 90)), "gross_pct": r4(np.mean(np.abs(e) > 25) * 100),
           "median_abs_m": r4(np.median(np.abs(p - t)))}
    if ci is not None:
        c = np.asarray(ci, float)[ok]
        has = np.isfinite(c)
        if has.any():
            out["coverage_95"] = r4(np.mean(np.abs(p - t)[has] <= c[has]))
            out["n_with_ci"] = int(has.sum())
    return out


def speed_stats(pred, truth, ci=None, guess_zero: bool = False) -> dict:
    """err = pred - truth over pairs with both finite (km/h)."""
    p_all, t_all = np.asarray(pred, float), np.asarray(truth, float)
    ok = np.isfinite(p_all) & np.isfinite(t_all)
    if p_all.size == 0 or not ok.any():
        return {"n": 0}
    p, t = p_all[ok], t_all[ok]
    err = p - t
    out = {"n": int(err.size), "mae": r4(np.mean(np.abs(err))), "bias": r4(np.mean(err)),
           "median_abs": r4(np.median(np.abs(err))), "p90_abs": r4(np.percentile(np.abs(err), 90))}
    if err.size >= 3 and np.std(t) > 0 and np.std(p) > 0:
        out["corr"] = r4(np.corrcoef(p, t)[0, 1])
        out["slope"] = r4(np.mean((t - t.mean()) * (p - p.mean())) / np.var(t))
    if ci is not None:
        c = np.asarray(ci, float)[ok]
        has = np.isfinite(c)
        if has.any():
            out["coverage_95"] = r4(np.mean(np.abs(err[has]) <= c[has]))
            out["n_with_ci"] = int(has.sum())
    if guess_zero:
        out["guess0_mae"] = r4(np.mean(np.abs(t)))
    return out


def by_cluster(df: pd.DataFrame, values) -> dict[str, np.ndarray]:
    """{cluster: values of its rows, in row order}; values is an array aligned with df's rows."""
    vals = np.asarray(values, float)
    cl = df["cluster"].astype(str).to_numpy()
    out: dict[str, list] = {}
    for k, v in zip(cl, vals):
        out.setdefault(k, []).append(v)
    return {k: np.asarray(v, float) for k, v in out.items()}


def cluster_interval(groups: dict[str, np.ndarray], stat, n: int, seed: int):
    """95 % interval of stat(pooled values) when whole clusters (sorted by name) are drawn with replacement."""
    keys = [k for k in sorted(groups) if len(groups[k])]
    if len(keys) < 2:
        return None
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        draws.append(stat(np.concatenate([groups[keys[i]] for i in pick])))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def summarize(P: pd.DataFrame, EF: pd.DataFrame, EB: pd.DataFrame, cfg: dict) -> dict:
    """The metric block of one set of pairs, ego frames and ego bins (same keys as the shared results.json)."""
    near_max = float(cfg["near_max_m"])
    d_truth = P["d_truth"].astype(float)
    Pn, Pf = P[d_truth <= near_max], P[d_truth > near_max]
    Pd = P[P["d_pred"].notna()]
    near, far = Pd[Pd["d_truth"].astype(float) <= near_max], Pd[Pd["d_truth"].astype(float) > near_max]
    flag = P["flag"]
    flagged = flag.notna() & (flag.astype(str) != "")

    def ds(df):
        return dist_stats(df["d_pred"], df["d_truth"], df["dist_ci95_m"])

    bins = cfg["dist_bins_m"]
    by_bin = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        dt = Pd["d_truth"].astype(float)
        by_bin[f"{lo:g}-{hi:g}"] = ds(Pd[(dt > lo) & (dt <= hi)])
    gate = cfg["sanity_gate_frac"]
    nd_pred, nd_truth = near["d_pred"].astype(float), near["d_truth"].astype(float)
    gated = near[(nd_pred - nd_truth).abs() <= gate * nd_truth]

    def rel(df):
        return speed_stats(df["rel_pred"], df["rel_truth"], df["rel_ci95_kmh"], guess_zero=True)

    def absolute(df):
        return speed_stats(df["abs_pred"], df["abs_truth"], df["abs_ci95_kmh"])

    boot = cfg["bootstrap"]
    pct = np.abs((nd_pred - nd_truth) / nd_truth * 100).to_numpy()
    rel_pairs = Pn[(Pn["rel_pred"].astype(float) - Pn["rel_truth"].astype(float)).notna()]
    rel_err = np.abs((rel_pairs["rel_pred"].astype(float) - rel_pairs["rel_truth"].astype(float)).to_numpy())
    have = EB[EB["pred"].notna()]
    ego_err = np.abs((have["pred"].astype(float) - have["truth"].astype(float)).to_numpy())
    intervals = {
        "near_distance_median_abs_pct": cluster_interval(by_cluster(near, pct), np.median, boot["n"], boot["seed"]),
        "near_relative_speed_mae": cluster_interval(by_cluster(rel_pairs, rel_err), np.mean, boot["n"], boot["seed"]),
        "ego_per_second_mae": cluster_interval(by_cluster(have, ego_err), np.mean, boot["n"], boot["seed"]),
    }
    nm = f"{near_max:g}"
    return {
        "distance": {
            f"near_le_{nm}m": ds(near),
            f"far_gt_{nm}m": ds(far),
            "near_own_lane": ds(near[near["truth_lane"].astype(int) == 0]),
            "near_adjacent_lanes": ds(near[near["truth_lane"].astype(int) != 0]),
            "by_truth_distance_m": by_bin,
            "near_unflagged": ds(near[~flagged.loc[near.index]]),
            "near_flagged_by_method": ds(near[flagged.loc[near.index]]),
            f"near_sanity_gated_{int(gate * 100)}pct": {**ds(gated), "dropped": int(len(near) - len(gated))},
            "pairs_without_distance": int(P["d_pred"].isna().sum()),
        },
        "relative_speed": {"near": rel(Pn), "far": rel(Pf)},
        "absolute_speed": {"near": absolute(Pn), "far": absolute(Pf)},
        "ego_speed": {
            "per_second": speed_stats(EB["pred"], EB["truth"]),
            # numpy's rounding of a numpy float, exactly as the shared scorer rounds this one value
            "per_second_coverage_pct": float(round(100 * EB["pred"].notna().mean(), 2)) if len(EB) else None,
            "per_second_within_3kmh_pct": round(float(100 * np.mean(ego_err <= 3)), 2) if len(have) else None,
            "per_frame": speed_stats(EF["ego_kmh"], EF["ego_truth"], EF["ego_ci95_kmh"]),
        },
        "bootstrap_95ci": intervals,
    }


# ---------------------------------------------------------------------------------------------------------- truth
def car_of(segment: dict) -> str:
    return str((segment.get("source") or {}).get("route", "")).split("|")[0]


def offset_of(cfg: dict, segment: dict) -> float:
    off = cfg["truth_dist_offset_m"]
    if not isinstance(off, dict):
        return float(off)
    car = car_of(segment)
    if car not in off:
        refuse(f"truth_dist_offset_m has no value for car '{car}' ({segment.get('segment_id')})")
    return float(off[car])


def main_column(cam: dict, segment: dict) -> float:
    """Image column straight ahead for the main matching: per route, else per car and mounting period, else cx."""
    route = str((segment.get("source") or {}).get("route", ""))
    per_route = cam.get("cx_by_route") or {}
    if route in per_route:
        return float(per_route[route])
    column = cam["cx"]
    route_time = route.split("|")[-1]
    for first_time, value in (cam.get("cx_by_car") or {}).get(car_of(segment), []):
        if route_time >= first_time:
            column = value
    return float(column)


def prepare_truth(tobj: pd.DataFrame, tego: pd.DataFrame, cfg: dict, offset: float) -> dict:
    """Returns of one segment ready for matching, sorted by (frame, truth_id as text), duplicates merged."""
    ego_at = dict(zip(tego["frame_idx"].astype(int).tolist(), tego["ego_kmh"].astype(float).tolist()))
    fi = tobj["frame_idx"].astype(int).to_numpy() if len(tobj) else np.zeros(0, int)
    native = tobj["dist_native_m"].astype(float).to_numpy()
    lat = tobj["lat_m"].astype(float).to_numpy()
    rel = tobj["rel_kmh"].astype(float).to_numpy()
    tid = np.array([str(x) for x in tobj["truth_id"]], dtype=object)
    ego = np.array([ego_at.get(int(f), np.nan) for f in fi], float)
    d = native + offset
    usable = (native < float(cfg.get("truth_empty_slot_min_native_m", np.inf))) & (d > float(cfg["min_truth_dist_m"]))
    idx = np.flatnonzero(usable)
    idx = np.array(sorted(idx.tolist(), key=lambda i: (int(fi[i]), tid[i])), int)      # stable sort
    keep = []
    dd, dl = float(cfg["truth_duplicate_dist_m"]), float(cfg["truth_duplicate_lat_m"])
    start = 0
    while start < len(idx):
        end = start
        while end < len(idx) and fi[idx[end]] == fi[idx[start]]:
            end += 1
        kept_here = []
        for i in idx[start:end]:
            if any(abs(d[i] - d[j]) <= dd and abs(lat[i] - lat[j]) <= dl for j in kept_here):
                continue
            kept_here.append(i)
        keep += kept_here
        start = end
    k = np.array(keep, int)
    absolute = ego[k] + rel[k]
    scope = (np.abs(absolute) >= float(cfg["moving_min_abs_kmh"])) & (np.abs(lat[k]) <= float(cfg["adjacent_max_abs_lat_m"]))
    lane = np.where(np.abs(lat[k]) <= float(cfg["lane_half_width_m"]), 0, np.sign(lat[k])).astype(int)
    return {"fi": fi[k], "tid": tid[k], "d": d[k], "lat": lat[k], "rel": rel[k], "abs": absolute,
            "scope": scope, "lane": lane}


def truth_bins(tego: pd.DataFrame, bin_len: int, cfg: dict) -> pd.DataFrame:
    """Fixed bins over the whole segment's truth; mean pose speed of all frames of the bin."""
    mean = tego.assign(bin=tego["frame_idx"].astype(int) // bin_len).groupby("bin")["ego_kmh"].mean()
    return pd.DataFrame({"truth_full": mean[mean >= cfg["ego_min_truth_kmh"]]})


# -------------------------------------------------------------------------------------------------- predictions
def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def load_prediction(folder: Path, segment: dict):
    """(status, detail, meta, ego, objects); status is ok, refused, invalid or missing (shared SPEC section 3 and the
    checks of its validator: required keys and columns, whole frame numbers in range, one ego row per frame, boxes
    inside the image, dist_m in (0, 500] or blank, speeds within 400 km/h, ego speed not negative, half-widths not
    negative, whole lane numbers)."""
    if not folder.is_dir():
        return "missing", None, None, None, None
    if not (folder / "meta.json").is_file():
        return "invalid", ["meta.json missing"], None, None, None
    try:
        meta = read_json(folder / "meta.json")
    except ValueError as exc:
        return "invalid", [f"meta.json cannot be read ({exc})"], None, None, None
    if not isinstance(meta, dict):
        return "invalid", ["meta.json is not a JSON object"], None, None, None
    errors = [f"meta.json: key {k} missing" for k in META_KEYS if k not in meta]
    if str(meta.get("spec", "")).split(".")[0] != SPEC_MAJOR:
        errors.append(f"meta.json: spec {meta.get('spec')} is not {SPEC_MAJOR}.x")
    if meta.get("segment_id") != segment.get("segment_id"):
        errors.append(f"meta.json: segment_id {meta.get('segment_id')} is not {segment.get('segment_id')}")
    status = meta.get("status")
    if status == "refused":
        if not meta.get("reason"):
            errors.append("meta.json: a refused segment needs a reason")
        return ("invalid", errors, meta, None, None) if errors else ("refused", meta.get("reason"), meta, None, None)
    if status != "ok":
        errors.append(f"meta.json: status is {status!r}, not ok or refused")
        return "invalid", errors, meta, None, None
    tables = {}
    for name, required, optional in (("ego.csv", EGO_REQUIRED, EGO_OPTIONAL), ("objects.csv", OBJ_REQUIRED, OBJ_OPTIONAL)):
        if not (folder / name).is_file():
            errors.append(f"{name} missing")
            continue
        try:
            df = read_table(folder / name)
        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            errors.append(f"{name} cannot be read ({exc})")
            continue
        lacking = [c for c in required if c not in df]
        if lacking:
            errors.append(f"{name}: required columns missing: {lacking}")
        for c in required + optional:
            if c in df and c not in TEXT_COLUMNS and not is_numeric(df[c]):
                errors.append(f"{name}: column {c} is not numeric")
        for c in optional:
            if c not in df:
                df[c] = np.nan
        tables[name] = df
    if errors:
        return "invalid", errors, meta, None, None
    ego, obj = tables["ego.csv"], tables["objects.csv"]
    n_frames = int(segment["n_frames"])
    for name, df in (("ego.csv", ego), ("objects.csv", obj)):
        f = df["frame_idx"].astype(float)
        if f.isna().any() or (f % 1 != 0).any():
            errors.append(f"{name}: frame_idx must be a whole number on every row")
        elif (f < 0).any() or (f >= n_frames).any():
            errors.append(f"{name}: frame_idx outside 0..{n_frames - 1}")
    if ego["frame_idx"].duplicated().any():
        errors.append("ego.csv: a frame_idx repeats")
    for c in ("track_id", "x0", "y0", "x1", "y1"):
        if obj[c].isna().any():
            errors.append(f"objects.csv: {c} is empty on some row")
    if obj.duplicated(["frame_idx", "track_id"]).any():
        errors.append("objects.csv: (frame_idx, track_id) repeats")
    if not set(obj["frame_idx"].tolist()) <= set(ego["frame_idx"].tolist()):
        errors.append("objects.csv has frames that ego.csv does not list")
    x0, y0, x1, y1 = (obj[c].astype(float) for c in ("x0", "y0", "x1", "y1"))
    if ((x1 <= x0) | (y1 <= y0)).any():
        errors.append("objects.csv: boxes need x1 > x0 and y1 > y0")
    w, h = float(segment["width"]), float(segment["height"])
    if ((x0 < -2) | (y0 < -2) | (x1 > w + 2) | (y1 > h + 2)).any():
        errors.append("objects.csv: boxes outside the image")
    dist = obj["dist_m"].astype(float).dropna()
    if ((dist <= 0) | (dist > 500)).any():
        errors.append("objects.csv: dist_m must be in (0, 500] or blank")
    for c in ("rel_kmh", "abs_kmh"):
        if (obj[c].astype(float).dropna().abs() > 400).any():
            errors.append(f"objects.csv: {c} beyond 400 km/h")
    speed = ego["ego_kmh"].astype(float).dropna()
    if (speed.abs() > 400).any():
        errors.append("ego.csv: ego_kmh beyond 400 km/h")
    if (speed < 0).any():
        errors.append("ego.csv: ego_kmh is negative")
    for name, df, cols in (("ego.csv", ego, ["ego_ci95_kmh"]),
                           ("objects.csv", obj, ["dist_ci95_m", "rel_ci95_kmh", "abs_ci95_kmh"])):
        for c in cols:
            if (df[c].astype(float).dropna() < 0).any():
                errors.append(f"{name}: {c} is negative")
    lane = obj["lane"].astype(float).dropna()
    if (lane % 1 != 0).any():
        errors.append("objects.csv: lane must be a whole number or blank")
    if errors:
        return "invalid", errors, meta, None, None
    return "ok", None, meta, ego, obj


# -------------------------------------------------------------------------------------------------------- scoring
def match_frame(x0, x1, u, d, cfg: dict, fx: float) -> list[tuple[int, int, float]]:
    """One-to-one (box, return, column offset) pairs of one frame, by image position only."""
    if len(x0) == 0 or len(u) == 0:
        return []
    centre, width = (x0 + x1) / 2.0, np.maximum(x1 - x0, 1.0)
    du = centre[:, None] - u[None, :]
    car_width = d[None, :] * width[:, None] / fx
    lo, hi = cfg["match_width_m"]
    allowed = ((np.abs(du) <= (cfg["match_k_width"] * width + cfg["match_px"])[:, None])
               & (car_width >= lo) & (car_width <= hi))
    cost = np.abs(du) / width[:, None] + cfg["match_dist_weight"] * d[None, :]
    cost = np.where(allowed, cost, BLOCKED)
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(du[i, j])) for i, j in zip(rows, cols) if cost[i, j] < BLOCKED]


def score_matching(sid: str, cluster: str, car: str, offset: float, column: float, truth: dict, keep: np.ndarray,
                   tego_at: dict, obj: pd.DataFrame, cfg: dict) -> tuple[list[dict], dict]:
    """Pairs of one segment at one image column; keep marks the prepared returns of processed frames."""
    fx = float(cfg["camera_for_matching"]["fx"])
    idx = np.flatnonzero(keep)
    u = column - fx * truth["lat"][idx] / truth["d"][idx]
    rows_of_frame: dict[int, list[int]] = {}
    for pos, i in enumerate(idx):
        rows_of_frame.setdefault(int(truth["fi"][i]), []).append(pos)
    ofi = obj["frame_idx"].astype(int).to_numpy()
    bx0, bx1 = obj["x0"].astype(float).to_numpy(), obj["x1"].astype(float).to_numpy()
    fields = {c: obj[c].to_numpy() for c in ("track_id", "dist_m", "dist_ci95_m", "lat_m", "lane", "rel_kmh",
                                             "rel_ci95_kmh", "abs_kmh", "abs_ci95_kmh", "flag")}
    boxes_of_frame: dict[int, list[int]] = {}
    for b, f in enumerate(ofi):
        boxes_of_frame.setdefault(int(f), []).append(b)
    pairs, unmatched, out_of_scope = [], 0, 0
    for f in sorted(boxes_of_frame):
        b = np.array(boxes_of_frame[f], int)
        here = rows_of_frame.get(f)
        if not here:
            unmatched += len(b)
            continue
        h = np.array(here, int)
        found = match_frame(bx0[b], bx1[b], u[h], truth["d"][idx[h]], cfg, fx)
        unmatched += len(b) - len(found)
        for i, j, du in found:
            t, k = int(idx[h[j]]), int(b[i])
            if not truth["scope"][t]:
                out_of_scope += 1
                continue
            pairs.append({
                "segment_id": sid, "cluster": cluster, "frame_idx": f, "truth_id": truth["tid"][t],
                "track_id": fields["track_id"][k], "truth_lane": int(truth["lane"][t]), "du_px": round(du, 1),
                "d_truth": float(truth["d"][t]), "d_pred": float(fields["dist_m"][k]),
                "dist_ci95_m": float(fields["dist_ci95_m"][k]), "lat_truth": float(truth["lat"][t]),
                "lat_pred": float(fields["lat_m"][k]), "lane_pred": float(fields["lane"][k]),
                "rel_truth": float(truth["rel"][t]), "rel_pred": float(fields["rel_kmh"][k]),
                "rel_ci95_kmh": float(fields["rel_ci95_kmh"][k]), "abs_truth": float(truth["abs"][t]),
                "abs_pred": float(fields["abs_kmh"][k]), "abs_ci95_kmh": float(fields["abs_ci95_kmh"][k]),
                "ego_truth": tego_at.get(f, np.nan), "flag": fields["flag"][k], "car": car,
                "truth_offset_m": offset, "match_cx_px": column})
    return pairs, {"unmatched_boxes": unmatched, "boxes_on_out_of_scope_truth": out_of_scope}


def ego_rows(sid: str, cluster: str, ego: pd.DataFrame, tego: pd.DataFrame, bin_len: int, cfg: dict):
    """(ego frame rows, ego bins) of a segment with output."""
    truth_at = tego.assign(frame_idx=tego["frame_idx"].astype(int)).set_index("frame_idx")["ego_kmh"]
    e = ego[["frame_idx", "ego_kmh", "ego_ci95_kmh", "flag"]].copy()
    e["frame_idx"] = e["frame_idx"].astype(int)
    e["ego_truth"] = e["frame_idx"].map(truth_at)
    e = e[e["ego_truth"].notna()].copy()
    e["bin"] = e["frame_idx"] // bin_len
    frames = e[e["ego_truth"] >= cfg["ego_min_truth_kmh"]].drop(columns="bin")
    frames.insert(0, "segment_id", sid)
    frames.insert(1, "cluster", cluster)
    read = e.dropna(subset=["ego_kmh"]).groupby("bin").agg(
        pred=("ego_kmh", "mean"), truth=("ego_truth", "mean"), n=("ego_kmh", "size"))
    read = read[read["n"] >= cfg["ego_min_readings_per_bin"]]
    bins = truth_bins(tego, bin_len, cfg).join(read, how="left").reset_index()
    bins.insert(0, "segment_id", sid)
    bins.insert(1, "cluster", cluster)
    return frames[EGO_FRAME_COLUMNS], bins[BIN_COLUMNS]


def empty_bins(sid: str, cluster: str, tego: pd.DataFrame, bin_len: int, cfg: dict) -> pd.DataFrame:
    b = truth_bins(tego, bin_len, cfg).reset_index()
    b.insert(0, "segment_id", sid)
    b.insert(1, "cluster", cluster)
    b["pred"] = b["truth"] = b["n"] = np.nan
    return b[BIN_COLUMNS]


def frame_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=PAIR_COLUMNS) if rows else pd.DataFrame(columns=PAIR_COLUMNS)


def concat(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True)[columns] if frames else pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------------------------------------------- seals
def tree_of(root: Path) -> dict[str, str]:
    """sha256 of every file under root, keyed by its path relative to root (linked folders followed)."""
    out = {}
    for d, _, names in os.walk(root, followlinks=True):
        for n in names:
            p = Path(d) / n
            out[p.relative_to(root).as_posix()] = sha256_file(p)
    return out


def read_manifest(path: Path) -> dict[str, str]:
    """A sha256sum listing ('<hash>  ./<segment>/<file>' per line) as {relative path: hash}."""
    files = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if line.startswith("\\"):
            refuse(f"{path}: escaped file names are not supported ({line[:80]})")
        m = MANIFEST_LINE.fullmatch(line)
        if not m:
            refuse(f"{path}: not a sha256sum line: {line[:80]}")
        name = m.group(2)
        name = name[2:] if name.startswith("./") else name
        if name in files:
            refuse(f"{path}: {name} is listed twice")
        files[name] = m.group(1)
    return files


class Seal:
    """A seal file: shared (mddbench) or own (tools/seal_predictions.py with folder manifests)."""

    def __init__(self, path: Path, manifest_dirs: list[Path]):
        self.path = Path(path)
        if not self.path.is_file():
            refuse(f"seal {self.path} not found")
        self.sha256 = sha256_file(self.path)
        rec = read_json(self.path)
        # a registered seal of another repository whose format was not known: only its time is used
        self.time_only = self.path.name in UNPINNED_SEALS
        if self.time_only:
            entry = UNPINNED_SEALS[self.path.name]
            if not isinstance(rec, dict) or "sealed_utc" not in rec:
                refuse(f"{self.path} ({entry['what']}) has no sealed_utc, so its time cannot be checked; ask its "
                       f"owner for the time, or leave it out with --waive-registered-seal {entry['id']} (recorded)")
        elif not isinstance(rec, dict) or "sealed_utc" not in rec or not isinstance(rec.get("files"), dict):
            refuse(f"{self.path} is not a seal (needs sealed_utc and files)")
        self.rec = rec
        self.sealed = parse_utc(rec["sealed_utc"])
        self.shared = not self.time_only and all(k in rec for k in ("method", "version", "segments"))
        self.trees: list[dict] = []          # {"name", "kind", "files"}: folder listings this seal pins
        self.notes: list[str] = []
        if self.time_only:
            self.notes.append("registered by file name only (sha256 not known when this program was written); "
                              "only its sealed_utc is used")
            return
        if self.shared:
            self.trees.append({"name": f"{self.path.name} (shared seal, {rec['method']} {rec['version']})",
                               "kind": "shared", "files": dict(rec["files"])})
            return
        dirs = list(manifest_dirs) + [self.path.parent] + sorted(p for p in self.path.parent.iterdir() if p.is_dir())
        for name, digest in sorted(rec["files"].items()):
            m = MANIFEST_NAME.fullmatch(name)
            if not m:
                continue
            hit = None
            for d in dirs:
                cand = Path(d) / name
                if cand.is_file() and sha256_file(cand) == digest:
                    hit = cand
                    break
            if hit is None:
                self.notes.append(f"{name}: no file with the sealed sha256 {digest[:16]} in {', '.join(map(str, dirs))}")
                continue
            self.trees.append({"name": f"{self.path.name}: {name} ({hit})",
                               "kind": "before_blanking" if m.group(1) == "pred" else "blanked",
                               "files": read_manifest(hit)})

    def record(self) -> dict:
        out = {"file": os.path.abspath(self.path), "sha256": self.sha256, "sealed_utc": utc_text(self.sealed),
               "kind": "time only" if self.time_only else "shared" if self.shared else "own",
               "pins": [t["name"] for t in self.trees]}
        if self.shared:
            out.update(method=self.rec.get("method"), version=self.rec.get("version"),
                       segments=len(self.rec.get("segments", [])), config_sha256=self.rec.get("config_sha256"))
        if self.notes:
            out["notes"] = self.notes
        return out


def version_label(root: Path, ids: list[str]) -> tuple[set, set]:
    methods, versions = set(), set()
    for s in ids:
        p = root / s / "meta.json"
        if p.is_file():
            try:
                m = read_json(p)
            except ValueError:
                continue
            if isinstance(m, dict):
                methods.add(str(m.get("method")))
                versions.add(str(m.get("version")))
    return methods, versions


def check_versions(versions: dict, seals: list[Seal], ids: list[str], blind: bool) -> dict:
    """Every version folder must match at least one sealed listing exactly; returns what covered each one."""
    report = {}
    for name, root in versions.items():
        tree = tree_of(root)
        covered, misses = [], []
        for seal in seals:
            for t in seal.trees:
                files = t["files"]
                changed = sorted(k for k in files if k in tree and tree[k] != files[k])
                missing = sorted(k for k in files if k not in tree)
                added = sorted(k for k in tree if k not in files)
                if changed or missing or added:
                    misses.append(f"{t['name']}: {len(changed)} changed, {len(missing)} missing, {len(added)} added "
                                  f"(e.g. {(changed + missing + added)[0]})")
                    continue
                if t["kind"] == "shared":
                    wrong = []
                    for k in files:
                        if k.endswith("/meta.json"):
                            m = read_json(root / k)
                            if (m.get("method"), m.get("version")) != (seal.rec.get("method"), seal.rec.get("version")):
                                wrong.append(k)
                    if wrong:
                        refuse(f"{name}: {wrong[0]} is not from {seal.rec.get('method')} {seal.rec.get('version')}, "
                               f"the method and version of {seal.path.name}")
                covered.append({"seal": os.path.abspath(seal.path), "listing": t["name"], "kind": t["kind"],
                                "sealed_utc": utc_text(seal.sealed), "files": len(files)})
        if not covered:
            notes = [f"{seal.path.name}: {n}" for seal in seals for n in seal.notes]
            if not misses:
                misses = [f"none of the {len(seals)} given seals pins a folder listing (no shared seal, no own seal "
                          "with an accepted manifest)"]
            refuse(f"version {name} ({root}) matches none of the given seals:\n  " + "\n  ".join(misses)
                   + ("\n  manifests not accepted / notes:\n    " + "\n    ".join(notes) if notes else ""))
        if blind and all(c["kind"] == "before_blanking" for c in covered):
            refuse(f"version {name} ({root}) is sealed only as run.py wrote it, before the distances were left empty "
                   "(manifest_pred_*); the registered input is the blanked folder (addendum 1 point 4, addendum 3)")
        report[name] = {"root": os.path.abspath(root), "files": len(tree), "covered_by": covered}
    return report


def check_registered_seals(chunks: list[str], seals: list[Seal], waived: list[str]) -> list[dict]:
    """Every seal registered for each blind chunk must be among the given seals (or, if waivable, waived)."""
    by_sha = {s.sha256: s for s in seals}
    by_name = {s.path.name: s for s in seals if s.time_only}
    allowed = {e["id"] for c in chunks for e in REGISTERED_SEALS.get(c, ()) if e.get("waivable")}
    bad = sorted(set(waived) - allowed)
    if bad:
        refuse(f"--waive-registered-seal {bad[0]}: not a waivable registered seal of {', '.join(chunks)} "
               f"(waivable: {sorted(allowed) or 'none'})")
    report = []
    for c in chunks:
        if c not in REGISTERED_SEALS:
            refuse(f"{c} is blind but no seal list is registered for it in this program (REGISTERED_SEALS); every seal "
                   "of the batch is added there and committed before any of its truth is built")
        for e in REGISTERED_SEALS[c]:
            seal = by_sha.get(e["sha256"]) if e["sha256"] else by_name.get(e["file"])
            row = {"chunk": c, "id": e["id"], "what": e["what"], "registered_file": e["file"],
                   "registered_sha256": e["sha256"]}
            if seal is None:
                if e["id"] in waived:
                    report.append({**row, "status": "waived with --waive-registered-seal: its time is NOT checked"})
                    continue
                pin = f", sha256 {e['sha256'][:16]}..." if e["sha256"] else " (matched by file name)"
                refuse(f"the registered seal {e['id']} ({e['what']}; {e['file']}{pin}) is not among --seal-check: the "
                       f"truth of {c} may be built only after every version of both methods is sealed (addendum 2 "
                       "point 3), so every seal of the batch is given and the truth must be newer than all of them"
                       + (f"; it can be left out only with --waive-registered-seal {e['id']}, which is recorded"
                          if e.get("waivable") else ""))
            report.append({**row, "status": "given", "file": os.path.abspath(seal.path), "sha256": seal.sha256,
                           "sealed_utc": utc_text(seal.sealed)})
    return report


def segments_named(rec: dict) -> set:
    """Segment ids a seal record names: its segment list and the leading segment id of each sealed file name."""
    out = {str(x) for x in rec.get("segments") or [] if isinstance(x, str)}
    files = rec.get("files")
    for k in (files if isinstance(files, dict) else {}):
        m = SEGMENT_ID.match(str(k))
        if m:
            out.add(m.group(0))
    return out


def other_batch_seals(seals: list[Seal], ids: list[str]) -> list[str]:
    """Seal files in the folder of a given seal that name a scored segment but were not given (compared by sha256)."""
    given = {s.sha256 for s in seals}
    idset = set(ids)
    found = []
    for folder in sorted({Path(os.path.realpath(s.path)).parent for s in seals}):
        for f in sorted(folder.glob("*.json")):
            if not f.is_file():
                continue
            try:
                rec = read_json(f)
            except (OSError, ValueError):
                continue
            if not isinstance(rec, dict) or "sealed_utc" not in rec:
                continue
            n = len(segments_named(rec) & idset)
            if n and sha256_file(f) not in given:
                found.append(f"{f} (sealed_utc {rec.get('sealed_utc')}, names {n} of the scored segments)")
    return found


# ---------------------------------------------------------------------------------------------- output location
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
    """Why the results folder target may not be written, or None (see the docstring, outputs)."""
    if os.path.lexists(target):
        if not target.is_dir():
            return f"{target} exists and is not a folder"
        first = next(iter(target.iterdir()), None)
        if first is not None:
            return f"{target} is not empty (e.g. {first.name}); results never overwrite anything: give a new folder"
    real = Path(os.path.realpath(target))
    for what, p in inputs.items():
        rp = Path(os.path.realpath(p))
        if real == rp or rp in real.parents:
            return f"{target} is inside {what} ({p})"
    for d in (real, *real.parents):
        if not d.is_dir():
            continue
        if is_work_tree(d):
            return f"{target} is inside the git work tree {d}; results are written outside every repository"
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


# ------------------------------------------------------------------------------------------------------- results
def counts_by_bin(d: np.ndarray, edges) -> dict:
    return {f"{lo:g}-{hi:g}": int(((d > lo) & (d <= hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])}


def group_block(ids: list[str], P: pd.DataFrame, EF: pd.DataFrame, EB: pd.DataFrame, seg_out: dict, cfg: dict,
                with_rows: bool) -> dict:
    """Metrics of one group of listed segments at one matching."""
    idset = set(ids)
    Pg, EFg, EBg = P[P["segment_id"].isin(idset)], EF[EF["segment_id"].isin(idset)], EB[EB["segment_id"].isin(idset)]
    out_ids = [s for s in ids if seg_out[s]["status"] == "ok"]
    truth_all = sum(seg_out[s]["truth_in_scope"] for s in out_ids)
    near_max = float(cfg["near_max_m"])
    truth_near = sum(seg_out[s]["truth_in_scope_near"] for s in out_ids)
    d_truth = Pg["d_truth"].astype(float).to_numpy()
    matched_near = int((d_truth <= near_max).sum())
    edges = cfg["dist_bins_m"]
    truth_bin = {k: sum(seg_out[s]["truth_in_scope_by_bin"][k] for s in out_ids)
                 for k in counts_by_bin(np.zeros(0), edges)}
    pair_bin = counts_by_bin(d_truth, edges)
    res = summarize(Pg, EFg, EBg, cfg)
    near = Pg[Pg["d_pred"].notna() & (Pg["d_truth"].astype(float) <= near_max)]
    e = np.abs((near["d_pred"].astype(float) - near["d_truth"].astype(float)) / near["d_truth"].astype(float) * 100)
    block = {
        "segments": len(ids),
        "with_output": len(out_ids),
        "refused": sum(seg_out[s]["status"] == "refused" for s in ids),
        "missing": sum(seg_out[s]["status"] == "missing" for s in ids),
        "invalid": sum(seg_out[s]["status"] == "invalid" for s in ids),
        "found_pct": pct2(len(Pg), truth_all),
        f"found_le_{near_max:g}m_pct": pct2(matched_near, truth_near),
        "found_counts": {"truth_in_scope": truth_all, "matched": int(len(Pg)),
                         f"truth_in_scope_le_{near_max:g}m": truth_near, f"matched_le_{near_max:g}m": matched_near},
        "found_by_truth_distance_m": {k: {"truth_in_scope": truth_bin[k], "matched": pair_bin[k],
                                          "found_pct": pct2(pair_bin[k], truth_bin[k])} for k in truth_bin},
        "near_gross_count": int((e > 25).sum()),
        **res,
    }
    if with_rows:
        rows = []
        for s in out_ids:
            g = near[near["segment_id"] == s]
            m = (float(np.median(np.abs((g["d_pred"].astype(float) - g["d_truth"].astype(float))
                                        / g["d_truth"].astype(float) * 100)))
                 if len(g) >= cfg["min_pairs_per_segment"] else None)
            rows.append({"segment_id": s, "n_pairs": int((Pg["segment_id"] == s).sum()),
                         "truth_in_scope": seg_out[s]["truth_in_scope"], "near_pairs": int(len(g)),
                         "near_median_abs_pct": None if m is None else round(m, 3)})
        meds = [r["near_median_abs_pct"] for r in rows if r["near_median_abs_pct"] is not None]
        block["per_segment"] = {"rows": rows, "segments_with_enough_pairs": len(meds),
                                "median_of_segment_medians": round(float(np.median(meds)), 3) if meds else None}
    return block


def paired_test(a, b, seg, cluster, stat, min_n: int, boot: dict) -> dict:
    """Sign test over segments with >= min_n readings, and a cluster bootstrap interval of stat(a) - stat(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    seg, cluster = np.asarray(seg, str), np.asarray(cluster, str)
    a_better = b_better = 0
    for s in sorted(set(seg.tolist())):
        k = seg == s
        if k.sum() >= min_n:
            sa, sb = stat(a[k]), stat(b[k])
            a_better += int(sa < sb)
            b_better += int(sa > sb)
    n = a_better + b_better
    groups = {}
    for c in sorted(set(cluster.tolist())):
        k = cluster == c
        groups[c] = np.column_stack([a[k], b[k]])
    keys = list(groups)
    interval = None
    if len(keys) >= 2:
        rng = np.random.default_rng(boot["seed"])
        draws = []
        for _ in range(boot["n"]):
            x = np.concatenate([groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
            draws.append(stat(x[:, 0]) - stat(x[:, 1]))
        interval = [round(float(v), 4) for v in np.percentile(draws, [2.5, 97.5])]
    return {"segments_compared": n, "a_better": a_better, "b_better": b_better,
            "sign_test_p_two_sided": round(float(binomtest(a_better, n, 0.5).pvalue), 5) if n else None,
            "diff_a_minus_b": round(float(stat(a) - stat(b)), 4) if len(a) else None,
            "diff_95ci_cluster_bootstrap": interval}


def same_truth(M: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        x, y = M[f"{c}_a"].astype(float).to_numpy(), M[f"{c}_b"].astype(float).to_numpy()
        if len(M) and not np.allclose(x, y, equal_nan=True):
            refuse(f"the two versions disagree on the truth column {c}")


def paired_block(ids: list[str], PA: pd.DataFrame, PB: pd.DataFrame, EBA: pd.DataFrame, EBB: pd.DataFrame,
                 cfg: dict) -> dict:
    idset = set(ids)
    near_max = float(cfg["near_max_m"])
    boot, min_n = cfg["bootstrap"], cfg["min_pairs_per_segment"]
    key = ["segment_id", "frame_idx", "truth_id"]
    A, B = PA[PA["segment_id"].isin(idset)], PB[PB["segment_id"].isin(idset)]
    M = A.merge(B, on=key, suffixes=("_a", "_b"))
    same_truth(M, ["d_truth", "rel_truth", "abs_truth"])
    M = M[M["d_truth_a"].astype(float) <= near_max]
    out = {"near_max_m": near_max}
    d = M[M["d_pred_a"].notna() & M["d_pred_b"].notna()]
    t = d["d_truth_a"].astype(float)
    ea = ((d["d_pred_a"].astype(float) - t).abs() / t * 100).to_numpy()
    eb = ((d["d_pred_b"].astype(float) - t).abs() / t * 100).to_numpy()
    sa, sb = dist_stats(d["d_pred_a"], t), dist_stats(d["d_pred_b"], t)
    test = paired_test(ea, eb, d["segment_id"], d["cluster_a"], np.median, min_n, boot)
    if sa.get("n") and sb.get("n"):
        test["diff_of_rounded_medians"] = round(sa["median_abs_pct"] - sb["median_abs_pct"], 4)
    out["near_distance"] = {"readings": int(len(d)), "a": sa, "b": sb, "paired_median_abs_pct": test}
    r = M[M["rel_pred_a"].notna() & M["rel_pred_b"].notna()]
    rt = r["rel_truth_a"].astype(float)
    out["near_relative_speed"] = {
        "readings": int(len(r)),
        "a": speed_stats(r["rel_pred_a"], rt, guess_zero=True), "b": speed_stats(r["rel_pred_b"], rt, guess_zero=True),
        "paired_mae": paired_test((r["rel_pred_a"].astype(float) - rt).abs(), (r["rel_pred_b"].astype(float) - rt).abs(),
                                  r["segment_id"], r["cluster_a"], np.mean, min_n, boot)}
    A_, B_ = EBA[EBA["segment_id"].isin(idset)], EBB[EBB["segment_id"].isin(idset)]
    E = A_.merge(B_, on=["segment_id", "bin"], suffixes=("_a", "_b"))
    same_truth(E, ["truth_full"])
    E = E[E["pred_a"].notna() & E["pred_b"].notna()]
    out["ego_per_second"] = {
        "bins": int(len(E)),
        "a": speed_stats(E["pred_a"], E["truth_a"]), "b": speed_stats(E["pred_b"], E["truth_b"]),
        "paired_mae": paired_test((E["pred_a"].astype(float) - E["truth_a"].astype(float)).abs(),
                                  (E["pred_b"].astype(float) - E["truth_b"].astype(float)).abs(),
                                  E["segment_id"], E["cluster_a"], np.mean, min_n, boot)}
    return out


# --------------------------------------------------------------------------------------------------------- report
def fmt(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}".rstrip("0").rstrip(".") if nd else f"{x:.0f}"
    return str(x)


def render(res: dict) -> str:
    cfg_near = res["settings"]["near_max_m"]
    nk = f"near_le_{cfg_near:g}m"
    fk = f"found_le_{cfg_near:g}m_pct"
    lines = ["# Own scoring of the joint batch (not the shared scorer)", "",
             res["not_the_shared_scorer"], "",
             f"Access: {res['access']['mode']} ({res['access']['why']}). Truth built {res['access']['truth_built']}"
             + (f", newer than every given seal (latest {res['access']['latest_seal_utc']})."
                if res["access"]["mode"] == "blind" else "."),
             *[f"WAIVED: registered seal {r['id']} ({r['what']}) was not given; its time is not checked."
               for r in (res["access"]["registered_seals"] or []) if r["status"] != "given"],
             f"Config {res['inputs']['config']} (sha256 {res['inputs']['config_sha256'][:16]}...); "
             f"{res['inputs']['segments_listed']} segments.", ""]
    for name, v in res["access"]["versions"].items():
        cov = "; ".join(f"{c['listing']} [{c['kind']}]" for c in v["covered_by"]) or "no seal check (--no-seal)"
        lines.append(f"- {name}: {v['root']} ({v['files']} files) - {cov}")
    lines.append("")
    for matching, rule in res["matchings"].items():
        lines += [f"## Matching {matching}: {rule}", "",
                  "| version | group | segments (output) | near n | median % | median m | bias % | > 25 % | "
                  "0-10 m | 10-20 m | 20-30 m | found <= 30 m % | found all % | rel MAE (n) | rel cov95 | "
                  "ego/s MAE | ego/s coverage % | ego/frame cov95 |",
                  "|" + "---|" * 18]
        for version, per in res["results"].items():
            for g, b in per[matching].items():
                d, bb = b["distance"][nk], b["distance"]["by_truth_distance_m"]
                rn, eg = b["relative_speed"]["near"], b["ego_speed"]
                lines.append(
                    f"| {version} | {g} | {b['segments']} ({b['with_output']}) | {d.get('n', 0)} | "
                    f"{fmt(d.get('median_abs_pct'))} | {fmt(d.get('median_abs_m'))} | {fmt(d.get('bias_pct'))} | "
                    f"{fmt(d.get('gross_pct'))} | {fmt(bb.get('0-10', {}).get('median_abs_m'))} | "
                    f"{fmt(bb.get('10-20', {}).get('median_abs_m'))} | {fmt(bb.get('20-30', {}).get('median_abs_m'))} | "
                    f"{fmt(b[fk])} | {fmt(b['found_pct'])} | {fmt(rn.get('mae'))} ({rn.get('n', 0)}) | "
                    f"{fmt(rn.get('coverage_95'))} | {fmt(eg['per_second'].get('mae'))} | "
                    f"{fmt(eg['per_second_coverage_pct'])} | {fmt(eg['per_frame'].get('coverage_95'))} |")
        lines.append("")
    for p in res["paired"]:
        lines += [f"## Paired: {p['a']} against {p['b']} (same segment, frame and truth target, truth <= "
                  f"{cfg_near:g} m, both with a distance)", "",
                  f"| matching | group | readings | {p['a']} median % | {p['b']} median % | difference pp | "
                  "of rounded medians | 95 % interval | segments a better / b better | sign test p |",
                  "|" + "---|" * 10]
        for matching, per in p["results"].items():
            for g, b in per.items():
                x = b["near_distance"]
                t = x["paired_median_abs_pct"]
                lines.append(f"| {matching} | {g} | {x['readings']} | {fmt(x['a'].get('median_abs_pct'))} | "
                             f"{fmt(x['b'].get('median_abs_pct'))} | {fmt(t['diff_a_minus_b'])} | "
                             f"{fmt(t.get('diff_of_rounded_medians'))} | {t['diff_95ci_cluster_bootstrap']} | "
                             f"{t['a_better']} / {t['b_better']} | {fmt(t['sign_test_p_two_sided'], 5)} |")
        lines.append("")
    lines += ["## Registered checks", ""]
    for k, v in res["registered_checks"].items():
        lines.append(f"- {k}: {json.dumps(v, ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def registered_checks(res: dict, primary: str, cfg: dict) -> dict:
    out = {}
    near_max = float(cfg["near_max_m"])
    nk, fk = f"near_le_{near_max:g}m", f"found_le_{near_max:g}m_pct"
    pair = next((p for p in res["paired"] if (p["a"], p["b"]) == ("v9d", "v9")), res["paired"][0] if res["paired"] else None)
    if pair and "main" in pair["results"]["main"]:
        x = pair["results"]["main"]["main"]["near_distance"]
        diff = x["paired_median_abs_pct"]["diff_a_minus_b"]
        out["expectation_6"] = {
            "rule": f"{pair['a']} against {pair['b']}, main group, main matching, same readings: |difference of the "
                    f"median error| <= {EXPECTATION_6_MAX_PP} percentage points (difference of the unrounded medians)",
            "readings": x["readings"], "median_a_pct": x["a"].get("median_abs_pct"),
            "median_b_pct": x["b"].get("median_abs_pct"), "difference_pp": diff,
            "difference_of_rounded_medians_pp": x["paired_median_abs_pct"].get("diff_of_rounded_medians"),
            "interval_95": x["paired_median_abs_pct"]["diff_95ci_cluster_bootstrap"],
            "met": None if diff is None else bool(abs(diff) <= EXPECTATION_6_MAX_PP)}
    for s in res["matchings"]:
        if s == "main" or primary not in res["results"]:
            continue
        m, o = res["results"][primary]["main"].get("main"), res["results"][primary][s].get("main")
        if m is None or o is None:
            continue
        cm, co = m["found_counts"], o["found_counts"]
        tk, mk = f"truth_in_scope_le_{near_max:g}m", f"matched_le_{near_max:g}m"
        gm, go = m["distance"][nk], o["distance"][nk]
        higher = cm[mk] > co[mk] if cm[tk] == co[tk] and cm[tk] else None
        lower = (m["near_gross_count"] * go["n"] < o["near_gross_count"] * gm["n"]
                 if gm.get("n") and go.get("n") else None)
        out[f"expectation_7_{s}"] = {
            "rule": f"{primary}, main group: at the main matching, a higher found rate within {near_max:g} m "
                    f"(in-scope truth, addendum 1 point 18) and a lower share of near errors > 25 % than at {s}; "
                    "the all-distance found rate is reported too",
            "found_le_near_pct": {"main": m[fk], s: o[fk]}, "found_all_pct": {"main": m["found_pct"], s: o["found_pct"]},
            "near_gross_pct": {"main": gm.get("gross_pct"), s: go.get("gross_pct")},
            "near_n": {"main": gm.get("n", 0), s: go.get("n", 0)},
            "found_higher": higher, "gross_lower": lower,
            "met": None if higher is None or lower is None else bool(higher and lower)}
    return out


# ----------------------------------------------------------------------------------------------------------- main
def git_state() -> dict:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    try:
        rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10, env=env)
        st = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", str(PROGRAM)],
                            capture_output=True, text=True, timeout=10, env=env)
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "program_uncommitted": None}
    if rev.returncode or st.returncode:
        return {"commit": None, "program_uncommitted": None}
    return {"commit": rev.stdout.strip(), "program_uncommitted": bool(st.stdout.strip())}


def parse_versions(items: list[str]) -> list[tuple[str | None, Path]]:
    out = []
    for item in items:
        name, sep, root = item.partition("=")
        if sep and LABEL.fullmatch(name):
            out.append((name, Path(root)))
        else:
            out.append((None, Path(item)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True, help="truth root written by mddbench truth-comma2k19 (<root>/<segment>/)")
    ap.add_argument("--segments", required=True, help="segments root (<root>/<segment>/segment.json)")
    ap.add_argument("--config", required=True, help="the shared scoring config (configs/eval_comma2k19.json)")
    ap.add_argument("--version", action="append", required=True, metavar="NAME=ROOT",
                    help="a version's shared-format prediction folder (<root>/<segment>/); NAME defaults to meta.json")
    ap.add_argument("--segment-list", help="segment ids, one per line (default: every folder of --segments)")
    ap.add_argument("--seal-check", nargs="+", action="extend", default=[], metavar="SEAL",
                    help="seal files (shared mddbench seals and/or tools/seal_predictions.py seals)")
    ap.add_argument("--manifest-dir", action="append", default=[], help="extra folder to look for sealed manifests in")
    ap.add_argument("--no-seal", action="store_true", help="score without seals (development data only)")
    ap.add_argument("--blind", action="store_true", help="apply the blind checks even if no segment is in a blind chunk")
    ap.add_argument("--waive-registered-seal", action="append", default=[], metavar="ID",
                    help="leave out a registered seal marked waivable (recorded; its time is then not checked)")
    ap.add_argument("--split-table", help="the shared splits/<dataset>.json (chunk status: dev, blind, blocked)")
    ap.add_argument("--pair", nargs=2, action="append", default=[], metavar=("A", "B"),
                    help="paired comparison A against B on the same readings (default: v9d v9 when both are given)")
    ap.add_argument("--primary", help="version of expectation 7 (default: v9d if given, else the first version)")
    ap.add_argument("--out", required=True, help="results folder")
    a = ap.parse_args(argv)
    started = datetime.now(timezone.utc)

    # ---- config, segments, versions (no truth)
    cfg_path = Path(a.config)
    cfg = read_json(cfg_path)
    config_sha256 = sha256_file(cfg_path)
    if cfg.get("_pending"):
        refuse(f"the scoring config still has undecided items: {cfg['_pending']}")
    if cfg.get("truth_dist_offset_m") in (None, {}):
        refuse("truth_dist_offset_m is undecided in the scoring config")
    seg_root, truth_root, out_dir = Path(a.segments), Path(a.truth), Path(a.out)
    ids = read_ids(Path(a.segment_list)) if a.segment_list else sorted(
        p.name for p in seg_root.iterdir() if (p / "segment.json").is_file())
    if not ids:
        refuse("no segments to score")
    twice = sorted({s for s in ids if ids.count(s) > 1})
    if twice:
        refuse(f"segments listed more than once: {twice[:5]}")
    for s in ids:
        if not (seg_root / s / "segment.json").is_file():
            refuse(f"no segment.json for {s} under {seg_root}")
    segments = {s: read_json(seg_root / s / "segment.json") for s in ids}
    for s, seg in segments.items():
        if seg.get("segment_id") != s:
            refuse(f"{seg_root / s / 'segment.json'} says segment_id {seg.get('segment_id')}")

    versions: dict[str, Path] = {}
    for name, root in parse_versions(a.version):
        if not root.is_dir():
            refuse(f"prediction folder {root} not found")
        methods, found = version_label(root, ids)
        if len(found) > 1 or len(methods) > 1:
            refuse(f"{root} holds more than one method version: {sorted(methods)} {sorted(found)}")
        if name is None:
            if not found:
                refuse(f"{root}: no meta.json to take the version name from; give NAME=ROOT")
            name = next(iter(found))
        elif found and name != next(iter(found)):
            refuse(f"--version {name}={root}: the folder's meta.json files say version {next(iter(found))}")
        if name in versions:
            refuse(f"version {name} given twice")
        versions[name] = root
    pairs = [tuple(p) for p in a.pair] or ([("v9d", "v9")] if {"v9d", "v9"} <= set(versions) else [])
    for x, y in pairs:
        if x not in versions or y not in versions or x == y:
            refuse(f"--pair {x} {y}: both must be given versions and differ")
    primary = a.primary or ("v9d" if "v9d" in versions else next(iter(versions)))
    if primary not in versions:
        refuse(f"--primary {primary} is not a given version")

    # ---- the results folder (before any seal or truth is read)
    out_inputs = {"the truth root": truth_root, "the segments root": seg_root,
                  **{f"the folder of version {v}": r for v, r in versions.items()},
                  **{f"--manifest-dir {d}": Path(d) for d in a.manifest_dir}}
    why_not_out = output_problem(out_dir, out_inputs)
    if why_not_out:
        refuse(why_not_out)

    # ---- split status from segment.json (no truth)
    table = None
    if a.split_table:
        table = read_json(Path(a.split_table)).get("chunks") or {}

    def status_of(chunk):
        if chunk is None:
            return "unknown"
        if table is not None:
            if chunk not in table:
                refuse(f"{chunk} is not in {a.split_table}")
            if chunk in DEFAULT_BLIND_CHUNKS and table[chunk]["status"] != "blocked":
                return "blind"              # a split table cannot make Chunk_8, 9 or 10 dev here
            return table[chunk]["status"]
        return "blind" if chunk in DEFAULT_BLIND_CHUNKS else "dev"

    seg_chunk = {s: (seg.get("source") or {}).get("chunk") for s, seg in segments.items()}
    seg_status = {s: status_of(c) for s, c in seg_chunk.items()}
    if any(v == "blocked" for v in seg_status.values()):
        refuse("some segments are in a blocked chunk")
    blind_by_chunk = sorted({seg_chunk[s] for s, v in seg_status.items() if v == "blind"})
    blind = bool(a.blind or blind_by_chunk)
    why = (f"segments of blind chunk(s) {', '.join(blind_by_chunk)}" if blind_by_chunk
           else "--blind given" if a.blind else "no segment in a blind chunk")

    # ---- seals (predictions only)
    if a.no_seal and a.seal_check:
        refuse("--no-seal and --seal-check exclude each other")
    if blind and a.no_seal:
        refuse(f"blind checks apply ({why}); --no-seal is for development data only")
    if not a.seal_check and not a.no_seal:
        refuse("give --seal-check SEAL ... (or --no-seal for development data)")
    seals = [Seal(Path(p), [Path(d) for d in a.manifest_dir]) for p in a.seal_check]
    version_access = (check_versions(versions, seals, ids, blind) if seals else
                      {n: {"root": os.path.abspath(r), "files": sum(len(f) for _, _, f in os.walk(r, followlinks=True)),
                           "covered_by": []} for n, r in versions.items()})
    shared = [s for s in seals if s.shared]
    if blind:
        if not shared:
            refuse("blind data needs at least one shared seal, which records the segment list, every segment.json "
                   "and the scoring config")
        for s in shared:
            if sorted(s.rec.get("segments", [])) != sorted(ids):
                refuse(f"the scored segment list differs from the segments of {s.path.name} (a blind batch is scored whole)")
            want = s.rec.get("segment_json_sha256") or {}
            for sid in ids:
                if want.get(sid) != sha256_file(seg_root / sid / "segment.json"):
                    refuse(f"segment.json of {sid} differs from the one recorded in {s.path.name}")
            if s.rec.get("config_sha256") != config_sha256:
                refuse(f"the scoring config differs from the one recorded in {s.path.name}")
    if a.waive_registered_seal and not blind_by_chunk:
        refuse("--waive-registered-seal applies only to segments of a blind chunk")
    registered = check_registered_seals(blind_by_chunk, seals, a.waive_registered_seal) if blind_by_chunk else []
    if blind_by_chunk:
        others = other_batch_seals(seals, ids)
        if others:
            refuse("seal files next to the given seals name scored segments but were not given (the truth must be "
                   "newer than every seal of the batch):\n  " + "\n  ".join(others[:10]))

    # ---- truth folders: presence, chunk and build time (meta.json only)
    latest_seal = max((s.sealed for s in seals), default=None)
    built, sources = {}, {"meta.json built_utc": 0, "file modification time": 0}
    for s in ids:
        d = truth_root / s
        for f in ("meta.json", "ego.csv", "objects.csv"):
            if not (d / f).is_file():
                refuse(f"truth of {s}: {d / f} not found")
        tmeta = read_json(d / "meta.json")
        tchunk = (tmeta.get("source") or {}).get("chunk")
        if tchunk is None:
            refuse(f"truth meta.json of {s} has no source.chunk")
        if seg_chunk[s] is not None and seg_chunk[s] != tchunk:
            refuse(f"{s} is in {tchunk} by its truth but in {seg_chunk[s]} by segment.json")
        if seg_chunk[s] is None and status_of(tchunk) != "dev":
            refuse(f"the truth of {s} is in {tchunk} ({status_of(tchunk)}) but its segment.json names no chunk; a "
                   "chunk that is not dev is scored only with segment.json files that name it")
        if tmeta.get("built_utc"):
            built[s] = parse_utc(tmeta["built_utc"])
            sources["meta.json built_utc"] += 1
        else:
            built[s] = datetime.fromtimestamp(min((d / f).stat().st_mtime for f in ("meta.json", "ego.csv", "objects.csv")),
                                              timezone.utc)
            sources["file modification time"] += 1
    if blind:
        early = [s for s in ids if not built[s] > latest_seal]
        if early:
            refuse(f"the truth of {early[0]} was built at {utc_text(built[early[0]])}, not after the latest given seal "
                   f"({utc_text(latest_seal)}); {len(early)} segments in all")
    truth_built = (f"{utc_text(min(built.values()))} to {utc_text(max(built.values()))} "
                   f"({', '.join(f'{k}: {v}' for k, v in sources.items() if v)})")

    # ---- truth values and scoring
    cam = cfg["camera_for_matching"]
    sens = [(v["name"], float(v["cx"])) for v in (cfg.get("matching_sensitivity") or [])]
    matchings = {"main": "columns by car and mounting period (cx_by_car), per-route exceptions (cx_by_route), else cx"}
    for nm, cx in sens:
        matchings[nm] = f"column {cx:g} for every segment (cx_by_car and cx_by_route cleared)"
    edges = cfg["dist_bins_m"]
    near_max = float(cfg["near_max_m"])
    rows = {v: {m: [] for m in matchings} for v in versions}
    ego_frames = {v: [] for v in versions}
    ego_bins = {v: [] for v in versions}
    seg_out = {v: {} for v in versions}
    detail = {v: {"refused": {}, "missing": [], "invalid": {}} for v in versions}
    readings = {v: {"with_distance": 0, f"beyond_{NO_NUMBER_BEYOND_M:g}_m": 0} for v in versions}
    extra = {v: {"unmatched_boxes": {m: 0 for m in matchings}, "boxes_on_out_of_scope_truth": {m: 0 for m in matchings}}
             for v in versions}
    columns_used = {m: {} for m in matchings}
    truth_lines = []                  # "<segment> <sha256 meta.json> <sha256 ego.csv> <sha256 objects.csv>"
    for s in ids:
        seg = segments[s]
        cluster = (str((seg.get("source") or {}).get("route", s)) if cfg.get("bootstrap_cluster") == "route" else s)
        truth_lines.append(" ".join([s] + [sha256_file(truth_root / s / f) for f in ("meta.json", "ego.csv", "objects.csv")]))
        tego, tobj = read_table(truth_root / s / "ego.csv"), read_table(truth_root / s / "objects.csv")
        tego = tego.assign(frame_idx=tego["frame_idx"].astype(int))
        car, offset = car_of(seg), offset_of(cfg, seg)
        truth = prepare_truth(tobj, tego, cfg, offset)
        tego_at = dict(zip(tego["frame_idx"].tolist(), tego["ego_kmh"].astype(float).tolist()))
        bin_len = max(1, int(round(float(cfg["ego_bin_s"]) * float(seg["fps"]))))
        columns = {"main": main_column(cam, seg), **{nm: cx for nm, cx in sens}}
        for m, c in columns.items():
            columns_used[m][f"{c:g}"] = columns_used[m].get(f"{c:g}", 0) + 1
        for v, root in versions.items():
            status, why_not, _, ego, obj = load_prediction(root / s, seg)
            if status != "ok":
                seg_out[v][s] = {"status": status}
                if status == "refused":
                    detail[v]["refused"][s] = why_not
                elif status == "missing":
                    detail[v]["missing"].append(s)
                else:
                    detail[v]["invalid"][s] = why_not
                ego_bins[v].append(empty_bins(s, cluster, tego, bin_len, cfg))
                continue
            processed = set(ego["frame_idx"].astype(int).tolist())
            keep = np.array([int(f) in processed for f in truth["fi"]], bool)
            scope = truth["scope"] & keep
            seg_out[v][s] = {"status": "ok", "truth_in_scope": int(scope.sum()),
                             "truth_in_scope_near": int((scope & (truth["d"] <= near_max)).sum()),
                             "truth_in_scope_by_bin": counts_by_bin(truth["d"][scope], edges)}
            dist = obj["dist_m"].astype(float)
            readings[v]["with_distance"] += int(dist.notna().sum())
            readings[v][f"beyond_{NO_NUMBER_BEYOND_M:g}_m"] += int((dist > NO_NUMBER_BEYOND_M).sum())
            for m, c in columns.items():
                p, info = score_matching(s, cluster, car, offset, c, truth, keep, tego_at, obj, cfg)
                rows[v][m] += p
                for k2, n2 in info.items():
                    extra[v][k2][m] += n2
            f, b = ego_rows(s, cluster, ego, tego, bin_len, cfg)
            ego_frames[v].append(f)
            ego_bins[v].append(b)

    # ---- tables and summaries
    why_not_out = output_problem(out_dir, out_inputs)      # again: nothing may have appeared meanwhile
    if why_not_out:
        refuse(why_not_out + " (nothing written)")
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = {s: (str(seg.get("road_class", "unknown")), str(seg.get("light", "unknown"))) for s, seg in segments.items()}
    groups = {g: [s for s in ids if test(*labels[s])] for g, (_, test) in GROUPS.items()}
    P = {v: {m: frame_table(rows[v][m]) for m in matchings} for v in versions}
    EF = {v: concat(ego_frames[v], EGO_FRAME_COLUMNS) for v in versions}
    EB = {v: concat(ego_bins[v], BIN_COLUMNS) for v in versions}
    for v in versions:
        for m in matchings:
            P[v][m].to_csv(out_dir / f"pairs_{v}_{m}.csv", index=False, mode="x")     # "x": never overwrite
        EF[v].to_csv(out_dir / f"ego_frames_{v}.csv", index=False, mode="x")
        EB[v].to_csv(out_dir / f"ego_bins_{v}.csv", index=False, mode="x")
    results = {v: {m: {g: group_block(groups[g], P[v][m], EF[v], EB[v], seg_out[v], cfg, with_rows=(g == "all"))
                       for g in GROUPS}
                   for m in matchings}
               for v in versions}
    paired = [{"a": x, "b": y, "results": {m: {g: paired_block(groups[g], P[x][m], P[y][m], EB[x], EB[y], cfg)
                                               for g in GROUPS}
                                           for m in matchings}}
              for x, y in pairs]
    res = {
        "program": "tools/joint_own_score.py",
        "program_sha256": sha256_file(PROGRAM),
        "repository": git_state(),
        "not_the_shared_scorer": ("Computed by tools/joint_own_score.py of the method B repository, its own "
                                  "implementation of the settled shared rules on the shared truth "
                                  "(docs/JOINT_CHUNK8_PREREG.md, addendum 1 section C and addendum 2 point 2); "
                                  "not computed by the shared scorer (mddbench)."),
        "started_utc": utc_text(started),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "scipy": scipy.__version__},
        "inputs": {"truth": os.path.abspath(truth_root), "segments": os.path.abspath(seg_root),
                   "config": os.path.abspath(cfg_path), "config_sha256": config_sha256,
                   "segment_list": os.path.abspath(a.segment_list) if a.segment_list else None,
                   "segment_list_sha256": sha256_file(Path(a.segment_list)) if a.segment_list else None,
                   "segments_listed": len(ids), "versions": {v: os.path.abspath(r) for v, r in versions.items()},
                   "truth_files_sha256": hashlib.sha256("".join(f"{ln}\n" for ln in sorted(truth_lines)).encode()).hexdigest(),
                   "truth_files_sha256_rule": "sha256 of the lines '<segment> <sha256 of meta.json> <sha256 of ego.csv> "
                                              "<sha256 of objects.csv>', one per listed segment, sorted, each ending in \\n"},
        "access": {"mode": "blind" if blind else "dev", "why": why,
                   "chunks": {str(c): sum(1 for x in seg_chunk.values() if x == c) for c in sorted(set(map(str, seg_chunk.values())))},
                   "split_table": a.split_table, "seals": [s.record() for s in seals],
                   "registered_seals": registered or None,
                   "registered_seals_rule": ("for a blind chunk every seal in REGISTERED_SEALS is given (or waived, "
                                             "recorded), and no other seal file next to a given seal names a scored "
                                             "segment; the truth is newer than every given seal"
                                             if blind_by_chunk else "not applied (no segment in a blind chunk)"),
                   "results_folder_check": ("not inside the truth root, segments root, a version folder or a "
                                            "--manifest-dir, not in a git work tree, not in one segment's folder or a "
                                            "folder holding such folders; empty or new; files created exclusively"),
                   "versions": version_access, "truth_built": truth_built,
                   "latest_seal_utc": utc_text(latest_seal) if latest_seal else None},
        "settings": {"near_max_m": near_max, "matching_columns_used": columns_used,
                     "bootstrap": cfg["bootstrap"], "bootstrap_cluster": cfg.get("bootstrap_cluster")},
        "groups": {g: {"definition": GROUPS[g][0], "segments": len(groups[g])} for g in GROUPS},
        "segments_in_no_registered_group": [s for s in ids if not any(s in groups[g] for g in ("main", "local_day", "night"))],
        "matchings": matchings,
        "segments_by_version": {v: {"with_output": sum(1 for x in seg_out[v].values() if x["status"] == "ok"),
                                    **detail[v]} for v in versions},
        "readings": readings,
        "boxes": extra,
        "results": results,
        "paired": paired,
    }
    res["registered_checks"] = registered_checks(res, primary, cfg)
    text = render(res)
    for name, body in (("results.json", json.dumps(res, ensure_ascii=False, indent=1) + "\n"), ("summary.md", text)):
        with open(out_dir / name, "x", encoding="utf-8") as fh:
            fh.write(body)
    print(text)
    print(f"wrote {out_dir}/results.json, summary.md and the row tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())

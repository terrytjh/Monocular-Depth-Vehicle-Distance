#!/usr/bin/env python3
"""Per-second ego speed of the method against the dashcam's on-screen GPS speed (DOD recordings).

The on-screen speed is a reference, not ground truth: the differences are reported as
"差距" (method minus on-screen), never as errors of either side.

Inputs (all read-only):
  --run     the run folder: ego/<tag>_ego.csv, lists/tags_{main,local_day,night}.txt, lists/only_tags.txt
  --osd     the overlay reader output: <tag>_osd_seconds.csv, <tag>_osd_updates.csv, <tag>_osd_summary.json
  --frames  the masked frames root: <tag>/mask.json (timing_irregular flag)
Outputs (--out): seconds.csv, lag_curve.csv, summary.json, report.md

Rules (written before the comparison was run):
  * method bin b = frames [25 b, 25 b + 25) at 25 fps; no-shift reference r_b = the reader's per-second value (the slot
    covering frame 25 b + 12).
  * steady second: the slots k-2 ... k+2 around the second's slot k are all valid and their values span <= 2 km/h
    (judged on the on-screen speed alone).
  * primary set: main group, steady, no shift, r >= 5 km/h, files not timing_irregular.
  * lag L (secondary): the value of slot k is taken as the speed at tau_k = s_k - L (s_k = first frame / fps);
    g(t) = linear interpolation between consecutive valid slots whose tau gap <= 1.6 s; r_b(L) = mean of g over the
    bin's 25 frame times, defined only if g exists at all of them.  The first slot of a file is not used as an anchor
    when it starts at frame 0 (it began before the file, so its display time is unknown).
    L grid -0.50 ... +2.50 s, step 0.04 s; objective = median |m_raw - r_b(L)| on main-group seconds with r_b(L) >= 5
    and method source v10:both or v10:depth (m_raw = kmh_before_smoothing).  Rotation: chosen on two days, evaluated
    on the third; the reported L is the argmin on all three days.  Check: half-file split (grid -0.5 ... +2.0, 0.1 s).
  * statistics: d = m - r; bias, SD, 95 % limits of agreement (bias +- 1.96 SD), empirical 2.5 / 97.5 %, mean |d| +- SD,
    median |d|, share |d| <= 3 km/h, relative |d| / r for r >= 20 km/h, coverage of the method's 95 % range;
    clip-level bootstrap, 2,000 resamples, seed 20261016.

  python3 tools/dod_compare.py --run RUN --osd OSD --frames FRAMES --out OUT
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

FPS = 25.0
SEED = 20261016
NBOOT = 2000
V10 = ("v10:both", "v10:depth")


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_csv(p):
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_list(p):
    p = Path(p)
    return [x.strip() for x in p.read_text().split() if x.strip()] if p.exists() else []


def trip_and_day(tag):
    m = re.match(r"dod_(\d{4}-\d\d-\d\d)--(\d\d-\d\d-\d\d)_", tag)
    return m.group(1) + "--" + m.group(2), m.group(1)


def load_clip(tag, a):
    ego = {int(r["second"]): r for r in read_csv(Path(a.run) / "ego" / f"{tag}_ego.csv")}
    osd = Path(a.osd)
    secs = {int(r["second"]): r for r in read_csv(osd / f"{tag}_osd_seconds.csv")}
    upd = {int(r["slot"]): r for r in read_csv(osd / f"{tag}_osd_updates.csv")}
    summ = json.loads((osd / f"{tag}_osd_summary.json").read_text())
    mask = json.loads((Path(a.frames) / tag / "mask.json").read_text())
    return ego, secs, upd, summ, mask


def slot_ok(u):
    return u is not None and u.get("valid") == "1" and fnum(u.get("value_kmh")) is not None


def steady(k, upd):
    vals = []
    for j in range(k - 2, k + 3):
        u = upd.get(j)
        if not slot_ok(u):
            return False
        vals.append(float(u["value_kmh"]))
    return max(vals) - min(vals) <= 2.0


def anchors(upd):
    """(s_k, v_k) of valid slots usable as interpolation anchors, in slot order."""
    out = []
    for k in sorted(upd):
        u = upd[k]
        if not slot_ok(u):
            continue
        ff = int(u["first_frame"])
        if ff == 0 and u.get("partial") == "1":
            continue
        out.append((ff / FPS, float(u["value_kmh"])))
    return np.array(out, dtype=float).reshape(-1, 2)


def ref_with_lag(anc, nsec, L):
    """r_b(L) for b = 0 ... nsec-1 (nan where undefined)."""
    r = np.full(nsec, np.nan)
    if len(anc) < 2:
        return r
    tau = anc[:, 0] - L
    v = anc[:, 1]
    ok_seg = np.diff(tau) <= 1.6 + 1e-9
    t = np.arange(nsec * 25) / FPS
    i = np.searchsorted(tau, t, side="right") - 1          # segment index: tau[i] <= t < tau[i+1]
    valid = (i >= 0) & (i < len(tau) - 1)
    i_c = np.clip(i, 0, len(tau) - 2)
    valid &= ok_seg[i_c]
    w = (t - tau[i_c]) / np.maximum(tau[i_c + 1] - tau[i_c], 1e-9)
    g = v[i_c] + w * (v[i_c + 1] - v[i_c])
    g = np.where(valid, g, np.nan).reshape(nsec, 25)
    full = ~np.isnan(g).any(axis=1)
    r[full] = g[full].mean(axis=1)
    return r


def stats(rows, rng_seed=SEED, boot=True):
    if not rows:
        return {"n_seconds": 0, "n_clips": 0}
    d = np.array([x["m"] - x["r"] for x in rows])
    r = np.array([x["r"] for x in rows])
    lo = np.array([x["lo"] if x["lo"] is not None else np.nan for x in rows])
    hi = np.array([x["hi"] if x["hi"] is not None else np.nan for x in rows])
    clips = np.array([x["tag"] for x in rows])
    ad = np.abs(d)
    out = {
        "n_seconds": int(len(d)), "n_clips": int(len(set(clips))),
        "bias": float(d.mean()), "sd": float(d.std(ddof=1)) if len(d) > 1 else None,
    }
    if out["sd"] is not None:
        out["loa"] = [out["bias"] - 1.96 * out["sd"], out["bias"] + 1.96 * out["sd"]]
    out["d_p2_5_p97_5"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    out["mae"] = float(ad.mean())
    out["sd_abs"] = float(ad.std(ddof=1)) if len(d) > 1 else None
    out["median_abs"] = float(np.median(ad))
    out["share_le3"] = float((ad <= 3.0).mean())
    big = r >= 20
    if big.any():
        out["rel_abs_median_r20"] = float(np.median(ad[big] / r[big]))
        out["rel_signed_mean_r20"] = float(np.mean(d[big] / r[big]))
        out["n_r20"] = int(big.sum())
    has = ~np.isnan(lo) & ~np.isnan(hi)
    if has.any():
        inside = (r[has] >= np.maximum(0, lo[has])) & (r[has] <= hi[has])
        out["cover95"] = float(inside.mean())
        out["cover95_n"] = int(has.sum())
    if boot and len(set(clips)) >= 2:
        rng = np.random.default_rng(rng_seed)
        uc = sorted(set(clips))
        idx = defaultdict(list)
        for j, c in enumerate(clips):
            idx[c].append(j)
        idx = [np.array(idx[c]) for c in uc]
        bs = defaultdict(list)
        for _ in range(NBOOT):
            pick = rng.integers(0, len(uc), len(uc))
            jj = np.concatenate([idx[p] for p in pick])
            dd = d[jj]
            bs["bias"].append(dd.mean())
            bs["mae"].append(np.abs(dd).mean())
            bs["median_abs"].append(np.median(np.abs(dd)))
            s = dd.std(ddof=1) if len(dd) > 1 else 0.0
            bs["loa_lo"].append(dd.mean() - 1.96 * s)
            bs["loa_hi"].append(dd.mean() + 1.96 * s)
        out["boot95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
    return out


def band(r):
    if r < 5:
        return "<5"
    if r < 30:
        return "5-30"
    if r < 60:
        return "30-60"
    if r < 90:
        return "60-90"
    return ">=90"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--osd", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lists = Path(a.run) / "lists"
    tags = read_list(lists / "only_tags.txt")
    group = {}
    for g in ("main", "local_day", "night"):
        for t in read_list(lists / f"tags_{g}.txt"):
            group[t] = g

    lag_grid = np.round(np.arange(-0.50, 2.50 + 1e-9, 0.04), 2)
    half_grid = np.round(np.arange(-0.5, 2.0 + 1e-9, 0.1), 2)
    rows = []
    clip_info = {}
    lag_rows = defaultdict(list)   # L -> list of (tag, day, b, half, m_raw, r_L, src, group)
    for tag in tags:
        ego, secs, upd, summ, mask = load_clip(tag, a)
        trip, day = trip_and_day(tag)
        ti = bool(mask.get("timing_irregular"))
        nsec = len(ego)
        anc = anchors(upd)
        n_valid_ref = 0
        n_method = 0
        n_both = 0
        for b in range(nsec):
            e = ego.get(b, {})
            s = secs.get(b)
            m = fnum(e.get("kmh"))
            m_raw = fnum(e.get("kmh_before_smoothing"))
            r = fnum(s.get("kmh")) if s and s.get("valid") == "1" else None
            k = int(s["slot"]) if s and s.get("slot") not in (None, "") else None
            if r is not None:
                n_valid_ref += 1
            if m is not None:
                n_method += 1
            if m is not None and r is not None:
                n_both += 1
            rows.append({
                "tag": tag, "trip": trip, "trip_label": summ.get("trip_label"), "day": day,
                "group": group.get(tag, "?"), "timing_irregular": ti, "second": b,
                "m": m, "m_raw": m_raw, "lo": fnum(e.get("lo95")), "hi": fnum(e.get("hi95")),
                "source": e.get("source") or "", "r": r,
                "steady": (r is not None and k is not None and steady(k, upd)),
            })
        clip_info[tag] = {"trip": trip, "trip_label": summ.get("trip_label"), "day": day, "group": group.get(tag, "?"),
                          "timing_irregular": ti, "seconds": nsec, "ref_valid_seconds": n_valid_ref,
                          "method_seconds": n_method, "both_seconds": n_both}
        if ti or group.get(tag) != "main":
            continue
        for L in sorted(set(lag_grid.tolist()) | set(half_grid.tolist())):
            rL = ref_with_lag(anc, nsec, L)
            for b in range(nsec):
                e = ego.get(b, {})
                m_raw = fnum(e.get("kmh_before_smoothing"))
                src = e.get("source") or ""
                if m_raw is None or src not in V10 or np.isnan(rL[b]) or rL[b] < 5:
                    continue
                lag_rows[L].append((tag, day, b, 0 if b < nsec / 2 else 1, abs(m_raw - rL[b])))

    # ---- lag choice ----
    days = sorted({c["day"] for c in clip_info.values()})

    def objective(L, keep):
        v = [x[4] for x in lag_rows.get(L, []) if keep(x)]
        return (float(np.median(v)) if v else None, len(v))

    curve = []
    for L in lag_grid.tolist():
        rec = {"L": L}
        rec["all"], rec["n_all"] = objective(L, lambda x: True)
        for d in days:
            rec[d], rec["n_" + d] = objective(L, lambda x, d=d: x[1] == d)
        curve.append(rec)

    def argmin(keep, grid):
        best = None
        for L in grid:
            v, n = objective(L, keep)
            if v is not None and (best is None or v < best[1] - 1e-12):
                best = (L, v, n)
        return best

    lag = {"grid": [float(lag_grid[0]), float(lag_grid[-1]), 0.04]}
    best_all = argmin(lambda x: True, lag_grid.tolist())
    lag["argmin_all_days"] = {"L": best_all[0], "median_abs": best_all[1], "n": best_all[2]} if best_all else None
    rot = []
    for d in days:
        bt = argmin(lambda x, d=d: x[1] != d, lag_grid.tolist())
        ev = objective(bt[0], lambda x, d=d: x[1] == d) if bt else (None, 0)
        own = argmin(lambda x, d=d: x[1] == d, lag_grid.tolist())
        ev0 = objective(0.0, lambda x, d=d: x[1] == d)
        rot.append({"held_out_day": d, "L_from_other_days": bt[0] if bt else None,
                    "held_out_median_abs_at_that_L": ev[0], "held_out_n": ev[1],
                    "held_out_median_abs_at_L0": ev0[0], "held_out_own_argmin": own[0] if own else None})
    lag["rotation"] = rot
    Ls = [x["L_from_other_days"] for x in rot if x["L_from_other_days"] is not None]
    lag["rotation_span_s"] = (max(Ls) - min(Ls)) if Ls else None
    lag["stable_rule_span_le_0_3"] = (lag["rotation_span_s"] is not None and lag["rotation_span_s"] <= 0.3 + 1e-9)
    halves = []
    for h in (0, 1):
        bt = argmin(lambda x, h=h: x[3] == h, half_grid.tolist())
        ev = objective(bt[0], lambda x, h=h: x[3] != h) if bt else (None, 0)
        ev0 = objective(0.0, lambda x, h=h: x[3] != h)
        halves.append({"chosen_on_half": h, "L": bt[0] if bt else None, "other_half_median_abs": ev[0],
                       "other_half_n": ev[1], "other_half_median_abs_at_L0": ev0[0]})
    lag["half_file_split"] = halves
    L_star = best_all[0] if best_all else 0.0
    with open(out / "lag_curve.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curve[0].keys()))
        w.writeheader()
        w.writerows(curve)

    # r with the chosen lag, per second
    rL_of = {}
    for tag in tags:
        ego, secs, upd, summ, mask = load_clip(tag, a)
        rL_of[tag] = ref_with_lag(anchors(upd), len(ego), L_star)
    for x in rows:
        v = rL_of[x["tag"]][x["second"]] if x["second"] < len(rL_of[x["tag"]]) else np.nan
        x["r_lag"] = None if np.isnan(v) else float(v)

    with open(out / "seconds.csv", "w", newline="") as fh:
        cols = ["tag", "trip", "trip_label", "day", "group", "timing_irregular", "second", "m", "m_raw", "lo", "hi",
                "source", "r", "steady", "r_lag"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for x in rows:
            w.writerow({c: x[c] for c in cols})

    # ---- sets ----
    def pick(fn, lagged=False):
        sel = []
        for x in rows:
            r = x["r_lag"] if lagged else x["r"]
            if x["m"] is None or r is None or x["timing_irregular"]:
                continue
            y = dict(x, r=r)
            if fn(y):
                sel.append(y)
        return sel

    S = {}
    S["primary"] = stats(pick(lambda x: x["group"] == "main" and x["steady"] and x["r"] >= 5))
    S["main_all_valid_noshift_r5"] = stats(pick(lambda x: x["group"] == "main" and x["r"] >= 5))
    S["main_all_valid_lag_r5"] = stats(pick(lambda x: x["group"] == "main" and x["r"] >= 5, lagged=True))
    S["main_steady_lag_r5"] = stats(pick(lambda x: x["group"] == "main" and x["steady"] and x["r"] >= 5, lagged=True))
    S["main_accelerating_noshift_r5"] = stats(pick(lambda x: x["group"] == "main" and not x["steady"] and x["r"] >= 5))
    S["main_stops_r_lt5"] = stats(pick(lambda x: x["group"] == "main" and x["r"] < 5))
    for g in ("night", "local_day"):
        S[f"{g}_all_valid_noshift_r5"] = stats(pick(lambda x, g=g: x["group"] == g and x["r"] >= 5))
    srcs = sorted({x["source"] for x in rows if x["source"]})
    S["by_source_primary"] = {s: stats(pick(lambda x, s=s: x["group"] == "main" and x["steady"] and x["r"] >= 5
                                            and x["source"] == s)) for s in srcs}
    S["tenth_version_primary"] = stats(pick(lambda x: x["group"] == "main" and x["steady"] and x["r"] >= 5
                                            and x["source"] in V10))
    S["by_band_main_noshift"] = {bd: stats(pick(lambda x, bd=bd: x["group"] == "main" and x["r"] >= 5
                                                and band(x["r"]) == bd)) for bd in ("5-30", "30-60", "60-90", ">=90")}
    S["by_day_primary"] = {d: stats(pick(lambda x, d=d: x["group"] == "main" and x["steady"] and x["r"] >= 5
                                         and x["day"] == d), boot=False) for d in days}
    trips = sorted({x["trip"] for x in rows})
    S["by_trip_primary"] = {t: stats(pick(lambda x, t=t: x["group"] == "main" and x["steady"] and x["r"] >= 5
                                          and x["trip"] == t), boot=False) for t in trips}
    ti_rows = [dict(x) for x in rows if x["timing_irregular"] and x["m"] is not None and x["r"] is not None
               and x["r"] >= 5]
    S["timing_irregular_files_noshift_r5"] = stats(ti_rows, boot=False)

    # ---- coverage ----
    cov = {}

    def cov_of(keep):
        cs = [c for t, c in clip_info.items() if keep(t, c)]
        sec = sum(c["seconds"] for c in cs)
        ref = sum(c["ref_valid_seconds"] for c in cs)
        met = sum(c["method_seconds"] for c in cs)
        both = sum(c["both_seconds"] for c in cs)
        return {"clips": len(cs), "seconds": sec, "ref_valid_seconds": ref, "method_seconds": met,
                "both_seconds": both, "ref_coverage": ref / sec if sec else None,
                "method_coverage_of_ref": both / ref if ref else None,
                "method_coverage_of_all": met / sec if sec else None}

    cov["all"] = cov_of(lambda t, c: True)
    for g in ("main", "night", "local_day"):
        cov[g] = cov_of(lambda t, c, g=g: c["group"] == g)
    cov["by_day"] = {d: cov_of(lambda t, c, d=d: c["day"] == d) for d in days}
    cov["by_trip"] = {tr: cov_of(lambda t, c, tr=tr: c["trip"] == tr) for tr in trips}
    src_count = defaultdict(int)
    src_with_ref = defaultdict(int)
    for x in rows:
        if x["m"] is not None:
            src_count[x["source"]] += 1
            if x["r"] is not None:
                src_with_ref[x["source"]] += 1
    cov["method_seconds_by_source"] = dict(src_count)
    cov["method_seconds_with_ref_by_source"] = dict(src_with_ref)
    cov["clips_without_any_gps_speed"] = sorted(t for t, c in clip_info.items() if c["ref_valid_seconds"] == 0)
    cov["timing_irregular_files"] = sorted(t for t, c in clip_info.items() if c["timing_irregular"])
    excl = defaultdict(int)
    for x in rows:
        if x["m"] is None:
            continue
        if x["r"] is None:
            excl["method value but no valid on-screen value"] += 1
        elif x["timing_irregular"]:
            excl["timing_irregular file"] += 1
        elif x["group"] == "main" and x["r"] >= 5 and not x["steady"]:
            excl["main, r >= 5, not steady (secondary only)"] += 1
        elif x["r"] < 5:
            excl["r < 5 km/h (stops, reported apart)"] += 1
    cov["exclusions_of_method_seconds"] = dict(excl)

    res = {"made": time.strftime("%Y-%m-%d %H:%M:%S"), "run": a.run, "osd": a.osd, "frames": a.frames,
           "fps": FPS, "lag": lag, "L_used_for_lagged_sets": L_star, "sets": S, "coverage": cov,
           "clips": clip_info}
    (out / "summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({"primary": S["primary"], "L": lag["argmin_all_days"], "rotation": rot,
                      "half_file_split": halves, "coverage_all": cov["all"]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

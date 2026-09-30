#!/usr/bin/env python3
"""Per-segment distance error of the blind comma2k19 main group, both methods, as dot histograms.

One dot = one segment: the median of its matched radar pairs inside the ruler's reach, own lane and the next
lanes (at least 10 pairs), exactly the per-segment statistic of depth_dash_multicar score-batch
(per_segment_median_abs_pct). Left panel the median |error|, right panel the median signed error (negative =
our distance shorter than the radar's); one row per method, dots at their exact values, stacked apart where they
would overlap (beeswarm). The pairs come from depth_dash_multicar.c2k19_score after every cars file
has been checked against the seal; the per-lane counts and medians are then compared with the stored score files
(score_main_depth.json / score_main_c.json, key per_segment) and the script stops on any difference.
The two demo segments are ringed.

  python3 tools/report/plot_blind_per_segment.py \\
      --demo 99c94dc769b5d96e_2018-05-05--16-36-30_12:典型 --demo 99c94dc769b5d96e_2018-05-01--08-13-53_21:較差
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_dash_multicar as MC  # noqa: E402
from plot_v2_results import COL, cjk_font  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "data/output/dash_scale/blind_v2"
METHODS = (("depth", "_cars.json", "深度模型 × 法定虛線尺(第二版)"), ("c", "_cars_c.json", "標線幾何法(只用標線)"))
DOT_PX = 8.5        # dot diameter in pixels (beeswarm spacing)
ROW = {"depth": 1.3, "c": 0.0}


def per_segment(tags, sfx, stored, seal, truth_root):
    """{tag: (median |err|, median signed err, n)} for segments with >= 10 pairs, own + next lanes, inside reach."""
    files = {t: B / "pred" / f"{t}{sfx}" for t in tags if (B / "pred" / f"{t}{sfx}").exists()}
    MC.check_sealed(seal, [str(f) for f in files.values()])
    out = {}
    for t, f in files.items():
        cars = json.loads(f.read_text())
        if cars.get("status") == "refused":
            continue
        r, pairs, _ = MC.c2k19_score(cars, Path(truth_root) / t, return_rows=True)
        # same scoring as the stored run: per-lane counts and medians must agree
        for lab in ("own lane, inside reach", "next lanes, inside reach"):
            a, b = r["distance"][lab], stored[t]["distance"][lab]
            if a.get("n") != b.get("n") or (a.get("n") and not np.isclose(a["median_abs_pct"], b["median_abs_pct"])):
                raise SystemExit(f"{t} {lab}: rescored {a} differs from the stored score {b}")
        e = np.array([p["err"] for p in pairs if not p["far"] and p["lane"] in (-1, 0, 1)])
        if len(e) >= 10:
            out[t] = (float(np.median(np.abs(e))), float(np.median(e)), int(len(e)))
    return out


def swarm(ax, vals, y0, half):
    """Beeswarm offsets (data units along y) for the values of one row centred at y0: exact x, the smallest
    alternating offset that keeps DOT_PX between dot centres, never more than half a row out."""
    fig = ax.figure
    fig.canvas.draw()
    bb = ax.get_window_extent()
    (x0, x1), (y_lo, y_hi) = ax.get_xlim(), ax.get_ylim()
    px_x = bb.width / (x1 - x0)
    px_y = bb.height / (y_hi - y_lo)
    placed, ys = [], np.zeros(len(vals))
    for i in np.argsort(vals):
        xp = (vals[i] - x0) * px_x
        for k in range(200):
            off = ((k + 1) // 2) * (1 if k % 2 else -1) * DOT_PX * 0.9
            if all((xp - a) ** 2 + (off - b) ** 2 >= (DOT_PX * 0.98) ** 2 for a, b in placed if abs(xp - a) < DOT_PX):
                break
        placed.append((xp, off))
        ys[i] = y0 + np.clip(off / px_y, -half, half)
    return ys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default=str(B / "tags_main.txt"))
    ap.add_argument("--seal", default=str(B / "seal.json"))
    ap.add_argument("--truth-root", default=str(ROOT / "data/output/c2k19_truth"))
    ap.add_argument("--demo", action="append", default=[], help="TAG:短名 of a demo segment to ring (repeatable)")
    ap.add_argument("--out", default=str(B / "per_segment_errors.png"))
    a = ap.parse_args()
    cjk_font()
    tags = [l.strip() for l in open(a.list) if l.strip() and not l.startswith("#")]
    demo = [d.split(":", 1) for d in a.demo]
    res = {}
    for key, sfx, _ in METHODS:
        sc = json.loads((B / f"score_main_{key}.json").read_text())
        res[key] = per_segment(tags, sfx, sc["per_segment"], a.seal, a.truth_root)
        med = float(np.median([v[0] for v in res[key].values()]))
        want = sc["per_segment_median_abs_pct"]
        if len(res[key]) != want["n"] or not np.isclose(med, want["median"]):
            raise SystemExit(f"{key}: {len(res[key])} segments, median {med} vs stored {want}")
        print(f"{key}: {len(res[key])} segments with >= 10 pairs, median of the per-segment medians {med:.2f} %")
        if key == "depth":                      # the rule the demo segments were picked by
            v = np.array([x[0] for x in res[key].values()])
            names = list(res[key])
            for q in (50, 90):
                target = np.percentile(v, q)
                t = names[int(np.argmin(np.abs(v - target)))]
                print(f"  closest to the {q}th percentile ({target:.2f} %): {t} ({res[key][t][0]:.2f} %)")

    fig, ax = plt.subplots(1, 2, figsize=(16, 7.2), sharey=True, gridspec_kw=dict(wspace=0.06))
    fig.subplots_adjust(left=0.2, right=0.985, top=0.83, bottom=0.2)
    allv = [v for r in res.values() for v in r.values()]
    lims = [(0, np.ceil((max(v[0] for v in allv) + 1) / 5) * 5),
            (np.floor((min(v[1] for v in allv) - 1) / 5) * 5, np.ceil((max(v[1] for v in allv) + 1) / 5) * 5)]
    for c_ in (0, 1):
        ax[c_].set_xlim(*lims[c_])
        ax[c_].set_ylim(-0.9, 2.2)
    ann_dir = {"depth": 1, "c": -1}          # demo labels above the depth row, below the method-C row
    for key, _, name in METHODS:
        tags_k = list(res[key])
        y0 = ROW[key]
        for c_ in (0, 1):
            axx = ax[c_]
            vals = np.array([res[key][t][c_] for t in tags_k])
            ys = swarm(axx, vals, y0, 0.42)
            axx.scatter(vals, ys, s=DOT_PX ** 2 * 0.62, color=COL[key], edgecolor="white", linewidth=0.7, zorder=3)
            med = float(np.median(vals))
            axx.plot([med, med], [y0 - 0.46, y0 + 0.46], color="#222", lw=1.3, ls="--", zorder=2)
            lines = [(med, "中位數")]
            if c_ == 0:
                p90 = float(np.percentile(vals, 90))
                axx.plot([p90, p90], [y0 - 0.46, y0 + 0.46], color="#222", lw=1.3, ls=":", zorder=2)
                lines.append((p90, "第 90 百分位"))
            ty = y0 - 0.5 * ann_dir[key]               # values of the marker lines, on the side away from the labels
            for (x, lab), ha in zip(lines, ("right", "left")):       # median text ends at its line, P90 starts at its line
                axx.text(x + (-0.15 if ha == "right" else 0.15), ty,
                         f"{lab} {x:+.1f}%" if c_ == 1 else f"{lab} {x:.1f}%", ha=ha,
                         va="top" if ann_dir[key] > 0 else "bottom", fontsize=9.5, color="#222",
                         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=4)
            here = sorted((vals[tags_k.index(tg)], ys[tags_k.index(tg)], short) for tg, short in demo if tg in tags_k)
            dx = 0.08 * (lims[c_][1] - lims[c_][0])
            for j, (x, y, short) in enumerate(here):
                axx.scatter([x], [y], s=DOT_PX ** 2 * 2.6, facecolor="none", edgecolor="#111", linewidth=1.8, zorder=5)
                side = (-1 if j == 0 else 1) if len(here) > 1 else 0         # left one to the left, right one to the right
                axx.annotate(f"{short} {x:+.1f}%" if c_ == 1 else f"{short} {x:.1f}%", xy=(x, y),
                             xytext=(x + side * dx, y0 + ann_dir[key] * 0.66), textcoords="data",
                             ha="center", va="center", fontsize=10.5,
                             arrowprops=dict(arrowstyle="-", color="#111", lw=0.9, shrinkB=6),
                             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#111", lw=0.7), zorder=6)
    ax[1].axvline(0, color="#888", lw=1, zorder=1)
    ax[0].set_yticks([ROW[k] for k, _, _ in METHODS])
    ax[0].set_yticklabels([f"{name}\n{len(res[k])} 段" for k, _, name in METHODS], fontsize=11.5)
    for tl, (k, _, _) in zip(ax[0].get_yticklabels(), METHODS):
        tl.set_color(COL[k])
    for axx in ax:
        axx.tick_params(axis="y", length=0)
        axx.grid(axis="x", alpha=0.25)
        axx.spines[["top", "right", "left"]].set_visible(False)
    ax[0].set_xlabel("每段距離誤差的中位數:|我們 - 雷達| ÷ 雷達(%)", fontsize=11)
    ax[1].set_xlabel("同上,帶正負號(%;負 = 我們量得比雷達近)", fontsize=11)
    ax[0].set_title("誤差大小", fontsize=12.5)
    ax[1].set_title("偏向:系統性偏遠(+)或偏近(–)", fontsize=12.5)
    fig.suptitle("盲測主要組:每一段的距離誤差(每點 = 一段;尺的範圍內、本車道與左右一道的雷達配對取中位數,至少 10 筆)\n"
                 "圈起來的是影片的兩段:依深度版誤差排名事先挑的「典型」(最接近中位數)與「較差」(最接近第 90 百分位),"
                 "不是挑畫面好看的", fontsize=12.5, y=0.975)
    fig.text(0.2, 0.035, "虛線 = 該方法各段的中位數;點線 = 第 90 百分位。雷達距離已加固定偏移 2.37 m;"
             "預測在 2026-10-01 01:13(台灣時間)封存後才產生並讀取雷達真值。", fontsize=10, color="#444")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=120)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

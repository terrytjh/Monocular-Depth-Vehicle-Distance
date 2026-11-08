#!/usr/bin/env python3
"""Figures for tools/dod_compare.py (Chinese labels; PNG; for private use).

  python3 tools/dod_compare_plots.py --cmp OUT_OF_dod_compare.py [--font /mnt/c/Windows/Fonts/msjh.ttc]

Reads seconds.csv, lag_curve.csv and summary.json in --cmp and writes --cmp/plots/*.png:
  1 trip_<label>.png      on-screen speed (line) and method speed (points, 95 % range) over time, one trip per figure
  2 bland_altman.png      差距 against the mean of the two, bias and 95 % limits (primary set)
  3 scatter.png           method against on-screen speed with the identity line (primary set)
  4 lag_curve.png         median |差距| against the lag L, per day, with the chosen L
  5 band_box.png          差距 by on-screen speed band (main group, no shift)
  6 coverage_trip.png     seconds with an on-screen value, with a method value, with both, per trip
  7 cover95_source.png    share of seconds whose on-screen value lies in the method's 95 % range, per source
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SRC = [  # fixed order: categorical slots 1-4, plus a marker each (secondary encoding)
    ("v10:both", "第十版:兩法平均", "#2a78d6", "o"),
    ("v10:depth", "第十版:只有深度法", "#eb6834", "s"),
    ("marking, camera A, main-group fill (MC)", "第十一版補位 MC(標線幾何、相機 A)", "#1baf7a", "^"),
    ("marking, camera A, main-group fill (MS)", "第十一版補位 MS(標線幾何、相機 A)", "#eda100", "D"),
]
SRC_OF = {s[0]: s for s in SRC}


def setup(font):
    if font and Path(font).exists():
        font_manager.fontManager.addfont(font)
        name = font_manager.FontProperties(fname=font).get_name()
        plt.rcParams["font.family"] = [name, "DejaVu Sans"]
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
        "axes.unicode_minus": False,
    })


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(cmp):
    rows = []
    with open(Path(cmp) / "seconds.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            for k in ("m", "m_raw", "lo", "hi", "r", "r_lag"):
                r[k] = fnum(r[k])
            r["second"] = int(r["second"])
            r["steady"] = r["steady"] == "True"
            r["timing_irregular"] = r["timing_irregular"] == "True"
            rows.append(r)
    return rows


def primary(rows):
    return [r for r in rows if r["group"] == "main" and r["steady"] and not r["timing_irregular"]
            and r["m"] is not None and r["r"] is not None and r["r"] >= 5]


def legend_sources(ax, present, **kw):
    hs = []
    for key, label, col, mk in SRC:
        if key in present:
            hs.append(ax.scatter([], [], s=28, color=col, marker=mk, label=label, edgecolors=SURFACE, linewidths=0.6))
    return hs


def fig_trip(rows, out, summ):
    trips = sorted({r["trip"] for r in rows})
    for trip in trips:
        tr = [r for r in rows if r["trip"] == trip]
        if not any(r["m"] is not None and r["r"] is not None for r in tr):
            continue
        tags = sorted({r["tag"] for r in tr})
        # clip start in minutes from the file-name time (HHMMSS in the tag after the trip id)
        def clip_t0(tag):
            hms = tag.split("_")[-1].split("-")[0]
            return int(hms[:2]) * 3600 + int(hms[2:4]) * 60 + int(hms[4:6])
        t0 = min(clip_t0(t) for t in tags)
        fig, ax = plt.subplots(figsize=(14, 4.6))
        for tag in tags:
            cr = [r for r in tr if r["tag"] == tag]
            base = clip_t0(tag) - t0
            tt = np.array([(base + r["second"] + 0.5) / 60 for r in cr])
            rr = np.array([r["r"] if r["r"] is not None else np.nan for r in cr])
            ax.plot(tt, rr, color=INK2, linewidth=1.4, zorder=2)
        present = set()
        for key, label, col, mk in SRC:
            pts = [r for r in tr if r["m"] is not None and r["source"] == key]
            if not pts:
                continue
            present.add(key)
            tt = []
            for r in pts:
                tt.append((clip_t0(r["tag"]) - t0 + r["second"] + 0.5) / 60)
            tt = np.array(tt)
            mm = np.array([r["m"] for r in pts])
            lo = np.array([r["lo"] if r["lo"] is not None else np.nan for r in pts])
            hi = np.array([r["hi"] if r["hi"] is not None else np.nan for r in pts])
            ax.vlines(tt, np.maximum(lo, 0), hi, color=col, alpha=0.25, linewidth=1.0, zorder=3)
            ax.scatter(tt, mm, s=10, color=col, marker=mk, zorder=4, linewidths=0)
        h_line = ax.plot([], [], color=INK2, linewidth=1.4, label="疊字 GPS 車速")[0]
        hs = legend_sources(ax, present)
        ax.legend(handles=[h_line] + hs, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=5, frameon=False, fontsize=10)
        lab = tr[0]["trip_label"]
        ax.set_title(f"{lab}({trip})疊字車速與方法自車速(點 = 方法每秒讀值,細直線 = 方法自己的 95% 範圍)", loc="left", fontsize=12)
        ax.set_xlabel("這一趟開始後的分鐘")
        ax.set_ylabel("km/h")
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(out / f"1_trip_{lab}.png", dpi=130)
        plt.close(fig)


def fig_bland_altman(rows, out, summ):
    p = primary(rows)
    st = summ["sets"]["primary"]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    present = set()
    for key, label, col, mk in SRC:
        pts = [r for r in p if r["source"] == key]
        if not pts:
            continue
        present.add(key)
        mean = np.array([(r["m"] + r["r"]) / 2 for r in pts])
        d = np.array([r["m"] - r["r"] for r in pts])
        ax.scatter(mean, d, s=14, color=col, marker=mk, alpha=0.65, linewidths=0, zorder=3)
    for y, txt, ls in ((st["bias"], f"偏差 {st['bias']:+.1f}", "-"), (st["loa"][0], f"下界 {st['loa'][0]:+.1f}", "--"),
                       (st["loa"][1], f"上界 {st['loa'][1]:+.1f}", "--")):
        ax.axhline(y, color=INK, linewidth=1.2, linestyle=ls, zorder=4)
        ax.text(ax.get_xlim()[1], y, " " + txt, va="center", ha="left", fontsize=10, color=INK)
    ax.axhline(0, color=INK2, linewidth=0.8, zorder=2)
    ax.set_xlabel("兩者平均(km/h)")
    ax.set_ylabel("差距 = 方法 − 疊字(km/h)")
    ax.set_title(f"Bland–Altman:主要組、穩定行駛的秒、不平移、疊字 ≥ 5 km/h({st['n_seconds']} 秒,{st['n_clips']} 支片)",
                 loc="left", fontsize=11)
    ax.legend(handles=legend_sources(ax, present), loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "2_bland_altman.png", dpi=130)
    plt.close(fig)


def fig_scatter(rows, out, summ):
    p = primary(rows)
    fig, ax = plt.subplots(figsize=(7, 7))
    present = set()
    top = max(max(r["m"] for r in p), max(r["r"] for r in p)) * 1.05
    ax.plot([0, top], [0, top], color=INK2, linewidth=1.0, zorder=2)
    ax.text(top * 0.97, top * 0.93, "方法 = 疊字", ha="right", fontsize=10, color=INK2)
    for key, label, col, mk in SRC:
        pts = [r for r in p if r["source"] == key]
        if not pts:
            continue
        present.add(key)
        ax.scatter([r["r"] for r in pts], [r["m"] for r in pts], s=14, color=col, marker=mk, alpha=0.65,
                   linewidths=0, zorder=3)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect("equal")
    ax.set_xlabel("疊字 GPS 車速(km/h)")
    ax.set_ylabel("方法自車速(km/h)")
    ax.set_title("方法對疊字(主要組、穩定行駛的秒、不平移)", loc="left", fontsize=11)
    ax.legend(handles=legend_sources(ax, present), loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "3_scatter.png", dpi=130)
    plt.close(fig)


def fig_lag(cmp, out, summ):
    with open(Path(cmp) / "lag_curve.csv", newline="") as fh:
        cur = list(csv.DictReader(fh))
    L = np.array([float(c["L"]) for c in cur])
    fig, ax = plt.subplots(figsize=(8.5, 5))
    days = [k for k in cur[0] if k.startswith("2026-")]
    cols = ["#2a78d6", "#eb6834", "#1baf7a"]
    mks = ["o", "s", "^"]
    for d, col, mk in zip(days, cols, mks):
        y = np.array([fnum(c[d]) if fnum(c[d]) is not None else np.nan for c in cur])
        n = cur[0]["n_" + d]
        ax.plot(L, y, color=col, linewidth=2, marker=mk, markersize=4, markevery=5, label=f"{d}(約 {n} 秒)")
    y = np.array([fnum(c["all"]) for c in cur])
    ax.plot(L, y, color=INK, linewidth=2, label="三天合計")
    Ls = summ["lag"]["argmin_all_days"]["L"]
    ax.axvline(Ls, color=INK, linewidth=1, linestyle="--")
    ax.text(Ls, ax.get_ylim()[1], f" 最小值 L = {Ls:.2f} s", va="top", ha="right", fontsize=10)
    ax.set_xlabel("假設 GPS 車速落後畫面的秒數 L(秒)")
    ax.set_ylabel("|差距| 的中位數(km/h)")
    ax.set_title("時間平移曲線(主要組、第十版讀值、未平滑的每秒值)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "4_lag_curve.png", dpi=130)
    plt.close(fig)


def fig_band(rows, out, summ):
    bands = [("5–30", 5, 30), ("30–60", 30, 60), ("60–90", 60, 90), ("≥ 90", 90, 999)]
    p = [r for r in rows if r["group"] == "main" and not r["timing_irregular"] and r["m"] is not None
         and r["r"] is not None and r["r"] >= 5]
    data, labels = [], []
    for name, a, b in bands:
        d = [r["m"] - r["r"] for r in p if a <= r["r"] < b]
        data.append(d)
        labels.append(f"{name}\n({len(d)} 秒)")
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=labels, whis=(2.5, 97.5), showfliers=False, patch_artist=True, widths=0.5)
    for b in bp["boxes"]:
        b.set_facecolor("#9ec5f4")
        b.set_edgecolor("#1c5cab")
    for k in ("whiskers", "caps", "medians"):
        for x in bp[k]:
            x.set_color("#1c5cab" if k != "medians" else INK)
            x.set_linewidth(1.4)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("疊字 GPS 車速區間(km/h)")
    ax.set_ylabel("差距 = 方法 − 疊字(km/h)")
    ax.set_title("各車速區間的差距(主要組、全部有疊字的秒、不平移;盒 = 中間 50%,鬚 = 2.5–97.5%)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "5_band_box.png", dpi=130)
    plt.close(fig)


def fig_coverage(summ, out):
    bt = summ["coverage"]["by_trip"]
    trips = sorted(bt)
    lab = []
    for t in trips:
        tl = next((c["trip_label"] for c in summ["clips"].values() if c["trip"] == t), t)
        lab.append(f"{tl}\n{t[5:10]} {t[12:17].replace('-', ':')}")
    sec = np.array([bt[t]["seconds"] for t in trips]) / 60
    ref = np.array([bt[t]["ref_valid_seconds"] for t in trips]) / 60
    met = np.array([bt[t]["method_seconds"] for t in trips]) / 60
    both = np.array([bt[t]["both_seconds"] for t in trips]) / 60
    x = np.arange(len(trips))
    w = 0.2
    fig, ax = plt.subplots(figsize=(12, 5))
    for k, (v, name, col) in enumerate(((sec, "影片長度", "#c3c2b7"), (ref, "有疊字車速", "#2a78d6"),
                                       (met, "方法有讀值", "#eb6834"), (both, "兩者都有", "#1baf7a"))):
        ax.bar(x + (k - 1.5) * w, v, width=w - 0.02, color=col, label=name, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=9)
    ax.set_ylabel("分鐘")
    ax.set_title("每一趟的覆蓋(開發三天,225 支片)", loc="left", fontsize=11)
    ax.legend(frameon=False, ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "6_coverage_trip.png", dpi=130)
    plt.close(fig)


def fig_cover95(summ, out):
    bs = summ["sets"]["by_source_primary"]
    names, vals, ns, cols = [], [], [], []
    for key, label, col, mk in SRC:
        s = bs.get(key)
        if not s or not s.get("cover95_n"):
            continue
        names.append(label)
        vals.append(s["cover95"] * 100)
        ns.append(s["cover95_n"])
        cols.append(col)
    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(names))
    ax.barh(y, vals, color=cols, height=0.55, zorder=3)
    for yy, v, n in zip(y, vals, ns):
        ax.text(v + 1, yy, f"{v:.0f}%({n} 秒)", va="center", fontsize=10)
    ax.axvline(95, color=INK, linestyle="--", linewidth=1)
    ax.text(95, len(names) - 0.4, " 名目 95%", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("疊字車速落在方法自己 95% 範圍內的比例(%)")
    ax.set_title("95% 範圍的實際覆蓋(主要組、穩定行駛的秒、不平移)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "7_cover95_source.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cmp", required=True)
    ap.add_argument("--font", default="/mnt/c/Windows/Fonts/msjh.ttc")
    a = ap.parse_args()
    setup(a.font)
    out = Path(a.cmp) / "plots"
    out.mkdir(exist_ok=True)
    rows = load(a.cmp)
    summ = json.loads((Path(a.cmp) / "summary.json").read_text())
    fig_trip(rows, out, summ)
    fig_bland_altman(rows, out, summ)
    fig_scatter(rows, out, summ)
    fig_lag(a.cmp, out, summ)
    fig_band(rows, out, summ)
    fig_coverage(summ, out)
    fig_cover95(summ, out)
    print("\n".join(sorted(str(p) for p in out.glob("*.png"))))


if __name__ == "__main__":
    main()

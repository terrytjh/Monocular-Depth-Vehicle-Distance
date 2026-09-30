#!/usr/bin/env python3
"""Results figure for comma2k19: both methods against the car's radar and pose, on the same segments.

Four panels:
  distance            ours vs radar (+ the fixed offset), cars in the own and next lanes inside the ruler's reach
  error vs distance   median relative error per 5 m of true distance, with the middle half shaded:
                      a flat offset is a scale error, a slope is a geometry (horizon / compression) error
  ego speed           ours per second vs the pose speed
  relative speed      ours vs the radar's, over the same 1.6 s window
Only reads the cars files and the truth made by c2k19_extract.py truth; scoring rules are depth_dash_multicar's.

  python3 tools/report/plot_v2_results.py --list tags.txt --depth DIR:_cars.json --c DIR:_cars_c.json --out fig.png \
      [--verify-sealed seal.json] [--title "..."]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import depth_dash_multicar as MC  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
COL = {"depth": "#1f4e79", "c": "#c0392b"}
NAME = {"depth": "深度模型 × 虛線尺", "c": "方法 C(只用標線)"}


def cjk_font():
    for p in ("/mnt/c/Windows/Fonts/msjh.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if Path(p).exists():
            font_manager.fontManager.addfont(p)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=p).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def collect(tags, spec, truth_root, seal):
    d, sfx = spec.split(":")
    pairs, ego, used = [], [], 0
    files = [Path(d) / f"{t}{sfx}" for t in tags if (Path(d) / f"{t}{sfx}").exists()]
    if seal:
        MC.check_sealed(seal, [str(f) for f in files])
    for f in files:
        cars = json.loads(f.read_text())
        if cars.get("status") == "refused":
            continue
        t = f.name[:-len(sfx)]
        _, pr, eg = MC.c2k19_score(cars, Path(truth_root) / t, return_rows=True)
        pairs += pr
        ego += [e for e in eg if e["kmh"] is not None and e.get("truth") is not None]
        used += 1
    return pairs, ego, used


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True)
    ap.add_argument("--depth", required=True, help="DIR:SUFFIX of the depth version's cars files")
    ap.add_argument("--c", required=True, help="DIR:SUFFIX of method C's cars files")
    ap.add_argument("--truth-root", default=str(ROOT / "data/output/c2k19_truth"))
    ap.add_argument("--verify-sealed", default="")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cjk_font()
    tags = [l.strip() for l in open(a.list) if l.strip() and not l.startswith("#")]
    data = {k: collect(tags, spec, a.truth_root, a.verify_sealed) for k, spec in (("depth", a.depth), ("c", a.c))}

    fig, ax = plt.subplots(1, 4, figsize=(22, 7.0))
    lim = 45
    v_all = np.array([e["truth"] for k in data for e in data[k][1]] or [0, 130])
    vr = (max(0, v_all.min() - 5), v_all.max() + 5)          # ego-speed axes: the truth's range
    for k, (pairs, ego, used) in data.items():
        inside = [p for p in pairs if not p["far"] and p["lane"] in (-1, 0, 1)]
        t = np.array([p["truth"] for p in inside]); o = np.array([p["dist"] for p in inside])
        e = 100 * (o - t) / t
        lab = f"{NAME[k]}({used} 段,{len(inside)} 筆,中位誤差 {np.median(np.abs(e)):.1f}%)"
        n_out = int(np.sum((t > lim) | (o > lim)))
        if n_out:
            lab += f"\n  另 {n_out} 筆的雷達距離超過 {lim} m,在圖框外"
        ax[0].scatter(t, o, s=3, alpha=0.25, color=COL[k], label=lab, rasterized=True)
        bins = np.arange(0, lim + 5, 5)
        mid, med, q1, q3 = [], [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (t >= lo) & (t < hi)
            if m.sum() >= 20:
                mid.append((lo + hi) / 2); med.append(np.median(e[m]))
                q1.append(np.percentile(e[m], 25)); q3.append(np.percentile(e[m], 75))
        ax[1].plot(mid, med, "o-", color=COL[k], label=NAME[k])
        ax[1].fill_between(mid, q1, q3, color=COL[k], alpha=0.15)
        ev = np.array([x["kmh"] for x in ego]); et = np.array([x["truth"] for x in ego])
        n_out = int(np.sum((ev < vr[0]) | (ev > vr[1])))
        ax[2].scatter(et, ev, s=4, alpha=0.3, color=COL[k], rasterized=True,
                      label=f"{NAME[k]}(每秒 MAE {np.mean(np.abs(ev - et)):.2f} km/h,n {len(ev)})"
                            + (f"\n  另 {n_out} 秒的讀值超出圖框(最高 {ev.max():.0f} km/h)" if n_out else ""))
        rs = [p for p in pairs if p.get("rel") is not None]
        rv = np.array([p["rel"] for p in rs]); rt = np.array([p["rel_truth"] for p in rs])
        ax[3].scatter(rt, rv, s=4, alpha=0.3, color=COL[k], rasterized=True,
                      label=f"{NAME[k]}(MAE {np.mean(np.abs(rv - rt)):.2f},全部猜 0 為 {np.mean(np.abs(rt)):.2f})")
    x = np.linspace(0, lim, 2)
    ax[0].plot(x, x, color="#333", lw=1); ax[0].fill_between(x, 0.9 * x, 1.1 * x, color="#999", alpha=0.15, label="±10%")
    ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
    ax[0].set_xlabel("雷達距離 + 固定偏移 2.37(公尺)"); ax[0].set_ylabel("我們量到的距離(公尺)")
    ax[0].set_title("距離(本車道與左右一道,尺的範圍內)")
    ax[1].axhline(0, color="#333", lw=1); ax[1].set_xlabel("真實距離(公尺)"); ax[1].set_ylabel("相對誤差(%)")
    ax[1].set_title("誤差對距離(中位數,陰影 = 中間一半)\n平的偏移 = 尺度錯;斜的 = 幾何錯")
    ax[2].plot(vr, vr, color="#333", lw=1); ax[2].set_xlim(vr); ax[2].set_ylim(vr)
    ax[2].set_xlabel("定位車速(km/h)"); ax[2].set_ylabel("我們的自車速(km/h)"); ax[2].set_title("自車速(每秒)")
    ax[3].plot([-40, 40], [-40, 40], color="#333", lw=1); ax[3].set_xlim(-40, 40); ax[3].set_ylim(-40, 40)
    ax[3].set_xlabel("雷達相對速度(km/h)"); ax[3].set_ylabel("我們的相對速度(km/h)"); ax[3].set_title("他車相對速度(1.6 秒)")
    for i, x_ in enumerate(ax):                  # legends under the axes: none of them covers data
        x_.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), markerscale=1 if i == 1 else 3,
                  frameon=False)
        x_.grid(alpha=0.25)
    if a.title:
        fig.suptitle(a.title, fontsize=13)
    fig.subplots_adjust(left=0.035, right=0.99, top=0.86, bottom=0.27, wspace=0.2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""comma2k19 day-highway segments, development (30 segments of the 52, answers open) next to the second version's blind
test (95 segments): distance and speed errors of both methods against the radar / pose truth, read from the stored
batch scores (depth_dash_multicar.py score-batch). The third version's development numbers (local ego ruler) are added
from v3dev/dev30_guard.json. Nothing is re-scored.

  python3 tools/report/plot_dev_vs_blind.py --out data/output/dash_scale/report_hs/c2k19_dev_vs_blind.png
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/home/terry/Monocular-Depth-Vehicle-Distance")
O = ROOT / "data/output/dash_scale"
SRC = {("dev", "depth"): O / "dev52/batch_dayhw_cars_far.json", ("dev", "marking"): O / "dev52c/batch_dayhw_cars.json",
       ("blind", "depth"): O / "blind_v2/score_main_depth.json", ("blind", "marking"): O / "blind_v2/score_main_c.json"}


def metrics(p):
    s = json.loads(p.read_text())
    q = s["pooled"]
    return dict(scored=s["scored"], segments=s["segments"],
                own=q["distance"]["own lane, inside reach"]["median_abs_pct"],
                own_bias=q["distance"]["own lane, inside reach"]["bias_pct"],
                nxt=q["distance"]["next lanes, inside reach"]["median_abs_pct"],
                nxt_bias=q["distance"]["next lanes, inside reach"]["bias_pct"],
                rel=q["relative_speed"]["mae_kmh"], guess0=q["relative_speed"]["guess0_mae_kmh"],
                abs=q["absolute_speed"]["mae_kmh"], ego=q["ego_per_bin"]["mae_kmh"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT / "tools/report"))
    from plot_v2_results import cjk_font
    cjk_font()

    M = {k: metrics(p) for k, p in SRC.items()}
    v3 = next(r for r in json.loads((O / "v3dev/dev30_guard.json").read_text())["rows"]
              if r["variant"]["name"] == "ego: local k +-2.5 s")
    assert abs(M[("dev", "depth")]["ego"] - 5.64) < 0.01 and abs(v3["ego"] - 4.17) < 0.01   # docs/TERRY_DEV_LOG.md
    bars = [("開發 深度×虛線尺(第二版)", M[("dev", "depth")], "#9ecae1", ""),
            ("開發 深度×虛線尺+局部尺(第三版)", dict(M[("dev", "depth")], abs=v3["abs"], ego=v3["ego"]), "#a1d99b", ""),
            ("開發 標線幾何法", M[("dev", "marking")], "#fcbba1", ""),
            ("盲測 深度×虛線尺(第二版)", M[("blind", "depth")], "#1f77b4", "//"),
            ("盲測 標線幾何法", M[("blind", "marking")], "#d62728", "//")]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.6))
    groups = [("本車道距離", "own"), ("左右一道距離", "nxt")]
    w = .16
    for g, (gl, key) in enumerate(groups):
        for i, (lab, m, c, h) in enumerate(bars):
            if i == 1:
                continue                                    # the third version changes only the ego speed
            x = g + (i - 2) * w
            ax[0].bar(x, m[key], w, color=c, hatch=h, edgecolor="k", lw=.4, label=lab if g == 0 else None)
            ax[0].text(x, m[key] + .3, f"{m[key]:.1f}%", ha="center", fontsize=8)
    ax[0].set_xticks(range(len(groups))), ax[0].set_xticklabels([g[0] for g in groups])
    ax[0].set_ylabel("中位誤差(%)"), ax[0].set_ylim(0, 21)
    ax[0].set_title("他車距離(對雷達,尺的可信範圍內)")
    ax[0].legend(fontsize=8.5, loc="upper left")
    groups = [("他車相對速度", "rel"), ("他車絕對速度", "abs"), ("自車速(每秒)", "ego")]
    for g, (gl, key) in enumerate(groups):
        for i, (lab, m, c, h) in enumerate(bars):
            x = g + (i - 2) * w
            ax[1].bar(x, m[key], w, color=c, hatch=h, edgecolor="k", lw=.4, label=lab if g == 0 else None)
            ax[1].text(x, m[key] + .08, f"{m[key]:.2f}", ha="center", fontsize=7.5, rotation=90)
    ax[1].set_xticks(range(len(groups))), ax[1].set_xticklabels([g[0] for g in groups])
    ax[1].set_ylabel("平均絕對誤差 MAE(km/h)"), ax[1].set_ylim(0, 7.2)
    ax[1].set_title("速度(相對速度對雷達;自車速對定位車速)")
    ax[1].legend(fontsize=8.5, loc="upper left")
    n = {k: f"{v['scored']}/{v['segments']}" for k, v in M.items()}
    fig.suptitle(f"comma2k19 白天高速公路:開發片(答案開過,深度 {n[('dev', 'depth')]} 段、標線 {n[('dev', 'marking')]} 段)"
                 f" vs 第二版盲測(深度 {n[('blind', 'depth')]} 段、標線 {n[('blind', 'marking')]} 段)", fontsize=12)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)
    for lab, m, *_ in bars:
        print(f"{lab:32s} own {m['own']:.1f}% ({m['own_bias']:+.1f})  next {m['nxt']:.1f}% ({m['nxt_bias']:+.1f})  "
              f"rel {m['rel']:.2f} (guess0 {m['guess0']:.2f})  abs {m['abs']:.2f}  ego {m['ego']:.2f}  {m['scored']}/{m['segments']}")


if __name__ == "__main__":
    main()

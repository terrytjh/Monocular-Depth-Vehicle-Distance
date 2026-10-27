#!/usr/bin/env python3
"""The 11 expectations of docs/DEPTH_DASH_V11_PREREG.md section 3.1 on one batch: the eleventh version (v11f) against the tenth
(v10) on the same batch. READS TRUTH only through the step-1 tables (collect.py) and the report (f03_report.py).

  python3 -B expectations.py --scores DIR --report DIR --batch C9 [--out FILE.json]

Definitions (the ones the development numbers of the registration's table 1.3 were computed with):
  found rate / median of |error| %: the report's dist_table.csv (cls all, truth bin le30 or 60-100);
  left / right gap: in the step-1 pairs (group, truth <= 30 m, a distance), median signed error % of the readings in the left
    lane (rlane -1) minus that of the right lane (rlane +1);
  relative speed MAE: rel_table.csv (all readings of each version);
  ego speed on the same seconds: the step-1 ego table, the seconds both versions have (with truth), error = max(0, km/h) - truth;
  ego coverage: ego_table.csv (the summary row of the group);
  95 % range coverage: v11f pairs within 30 m that carry a range (lo_a, hi_a): share with lo_a <= truth <= hi_a.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def num(x):
    return pd.to_numeric(x, errors="coerce")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    b = a.batch
    D = pd.read_csv(Path(a.report) / "dist_table.csv")
    Rt = pd.read_csv(Path(a.report) / "rel_table.csv")
    Et = pd.read_csv(Path(a.report) / "ego_table.csv")
    P = pd.read_csv(Path(a.scores) / b / "pairs.csv.gz", compression="gzip", low_memory=False)
    E = pd.read_csv(Path(a.scores) / b / "ego.csv.gz", compression="gzip")

    def dist(g, tb, v, col):
        x = D[(D.batch == b) & (D.group == g) & (D.version == v) & (D.cls == "all") & (D.tbin == tb)]
        return float(x[col].iloc[0]) if len(x) == 1 else None

    def lr_gap(g, v):
        X = P[(P.version == v) & (P.group == g) & (P.truth <= 30) & P.dist.notna()]
        lane = num(X.rlane)
        return float(X[lane == -1].err_pct.median() - X[lane == 1].err_pct.median())

    def ego_same(g):
        a_ = E[(E.version == "v10") & (E.group == g) & E.truth.notna()].set_index(["seg", "sec"])
        c_ = E[(E.version == "v11f") & (E.group == g) & E.truth.notna()].set_index(["seg", "sec"])
        both = a_.index.intersection(c_.index)
        ea = a_.loc[both, "kmh"].clip(lower=0) - a_.loc[both, "truth"]
        ec = c_.loc[both, "kmh"].clip(lower=0) - c_.loc[both, "truth"]
        return dict(seconds=int(len(both)), v10_mae=float(ea.abs().mean()), v11_mae=float(ec.abs().mean()),
                    v10_bias=float(ea.mean()), v11_bias=float(ec.mean()))

    def ego_row(g):
        x = Et[(Et.batch == b) & (Et.group == g) & Et.source.isna()]
        return x.iloc[0]

    def rel_row(g):
        x = Rt[(Rt.batch == b) & (Rt.group == g) & (Rt.kind == "rel")]
        return x.iloc[0]

    def range_cover(g):
        X = P[(P.version == "v11f") & (P.group == g) & (P.truth <= 30) & P.dist.notna() & P.lo_a.notna() & P.hi_a.notna()]
        return float(100 * ((X.truth >= X.lo_a) & (X.truth <= X.hi_a)).mean()), int(len(X))

    out = []

    def add(k, text, values, met):
        out.append(dict(item=k, expectation=text, values=values, met=bool(met)))

    f10, f11 = dist("main", "le30", "v10", "cov_dist"), dist("main", "le30", "v11f", "cov_dist")
    add(1, "main: found rate within 30 m, v11 - v10 >= 4 points", dict(v10=f10, v11=f11, diff=f11 - f10), f11 - f10 >= 4.0)
    m10, m11 = dist("main", "le30", "v10", "med_pct"), dist("main", "le30", "v11f", "med_pct")
    add(2, "main: median |error| within 30 m, v11 <= v10", dict(v10=m10, v11=m11), m11 <= m10)
    g10, g11 = lr_gap("main", "v10"), lr_gap("main", "v11f")
    add(3, "main: |left - right signed median| smaller in v11", dict(v10=g10, v11=g11), abs(g11) < abs(g10))
    f4, m4 = dist("main", "60-100", "v11f", "cov_dist"), dist("main", "60-100", "v11f", "med_pct")
    add(4, "main 60-100 m: v11 found rate >= 40 % and median |error| <= 10 %",
        dict(v10_found=dist("main", "60-100", "v10", "cov_dist"), v11_found=f4, v11_median=m4), f4 >= 40.0 and m4 <= 10.0)
    r = rel_row("main")
    add(5, "main: relative speed MAE, v11 < v10", dict(v10=float(r.v10_mae), v11=float(r.v11_mae)), r.v11_mae < r.v10_mae)
    e = ego_same("main")
    add(6, "main: ego speed on the same seconds, v11 MAE < v10 and |bias| smaller", e,
        e["v11_mae"] < e["v10_mae"] and abs(e["v11_bias"]) < abs(e["v10_bias"]))
    er = ego_row("main")
    add(7, "main: ego coverage >= 80 %", dict(v10=float(er.v10_cov), v11=float(er.v11_cov)), er.v11_cov >= 80.0)
    c, n = range_cover("main")
    add(8, "main: v11 95 % range coverage within 30 m >= 93 %", dict(cover=c, n=n), c >= 93.0)
    for k, g, gain, slack in ((9, "local_day", 10.0, 1.0), (10, "night", 10.0, 1.5)):
        fa, fb = dist(g, "le30", "v10", "cov_dist"), dist(g, "le30", "v11f", "cov_dist")
        ma, mb = dist(g, "le30", "v10", "med_pct"), dist(g, "le30", "v11f", "med_pct")
        add(k, f"{g}: found rate within 30 m +{gain:g} points or more, median |error| <= v10 + {slack:g} points",
            dict(v10_found=fa, v11_found=fb, diff=fb - fa, v10_median=ma, v11_median=mb), fb - fa >= gain and mb <= ma + slack)
    en = ego_row("night")
    add(11, "night: ego coverage >= 40 %", dict(v10=float(en.v10_cov), v11=float(en.v11_cov)), en.v11_cov >= 40.0)
    other = dict(lr_gap_local_day=dict(v10=lr_gap("local_day", "v10"), v11=lr_gap("local_day", "v11f")),
                 ego_same_local_day=ego_same("local_day"), ego_same_night=ego_same("night"))
    res = dict(batch=b, items=out, met=sum(x["met"] for x in out), other=other)
    for x in out:
        print(f"{x['item']:>2} {'MET    ' if x['met'] else 'NOT MET'} {x['expectation']}: "
              + ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in x["values"].items()))
    print(f"{res['met']} of 11 met")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

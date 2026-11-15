#!/usr/bin/env python3
"""Taiwan dashcam (DOD), twelfth version: the numbers of docs/DOD_V12_BLIND_PREREG.md section 3.1 from two outputs of
tools/dod_compare.py (the eleventh and the twelfth version of the same batch against the overlay speed; no time shift).

Reads only seconds.csv and summary.json of the two comparison folders (they hold the overlay speed: a SCORING tool, run
after the seals). Main group throughout. Definitions:
  all valid seconds   rows of seconds.csv with group main and an overlay value r (every clip, timing-irregular ones too)
  coverage            share of those seconds with a method value m
  MAE / median        of |m - r| over the seconds that have both
  stopped             r < 5 km/h;  moving 30+  r >= 30 km/h
  primary             summary.json sets.primary (main group, steady, r >= 5, no shift): cover95, MAE
  tenth-version       summary.json sets.tenth_version_primary: median |m - r| of the seconds whose source is the tenth version
Uncertainty: bootstrap over clips, 2,000 resamples, seed 20261016 (2.5 / 97.5 percentiles).
The overlay is a reference, not truth: the numbers are differences (差距).

usage: dod_blind_score.py --cmp11 DIR --cmp12 DIR --out FILE.json [--label TEXT]
"""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

THRESHOLDS = {1: 0.40, 2: 4.5, 4: 1.0, 5: 0.30, 6: 6.0, 7: 0.55, 8: 0.70, 9: 3.5}


def load(cmp_dir):
    rows = [x for x in csv.DictReader(open(Path(cmp_dir) / "seconds.csv")) if x["group"] == "main" and x["r"]]
    by = defaultdict(list)
    for x in rows:
        r = float(x["r"])
        by[x["tag"]].append((r, float(x["m"]) if x["m"] else None, x["source"]))
    return by


def stats(by, sel):
    def one(tags):
        n = have = 0
        d = []
        for t in tags:
            for r, m, _ in by[t]:
                if not sel(r):
                    continue
                n += 1
                if m is not None:
                    have += 1
                    d.append(abs(m - r))
        return n, have, (float(np.mean(d)) if d else None), (float(np.median(d)) if d else None)
    tags = sorted(by)
    n, have, mae, med = one(tags)
    rng = np.random.default_rng(20261016)
    cov_b, mae_b = [], []
    for _ in range(2000):
        pick = [tags[i] for i in rng.integers(0, len(tags), len(tags))]
        bn, bh, bm, _ = one(pick)
        if bn:
            cov_b.append(bh / bn)
        if bm is not None:
            mae_b.append(bm)
    pct = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None
    return dict(seconds=n, with_value=have, coverage=have / n if n else None, mae=mae, median_abs=med,
                coverage_ci95=pct(cov_b), mae_ci95=pct(mae_b), clips=len(tags))


def version(cmp_dir):
    by = load(cmp_dir)
    s = json.loads((Path(cmp_dir) / "summary.json").read_text())["sets"]
    pick = lambda d: None if not d or not d.get("n_seconds") else {k: d.get(k) for k in ("n_seconds", "n_clips", "bias", "mae", "median_abs", "share_le3", "cover95")}
    still = sum(1 for t in by for r, m, src in by[t] if src.startswith("stationary") and r >= 5)
    return dict(all_valid=stats(by, lambda r: True), stopped=stats(by, lambda r: r < 5), moving30=stats(by, lambda r: r >= 30),
                primary=pick(s.get("primary")), tenth_version_primary=pick(s.get("tenth_version_primary")),
                by_source_primary={k: pick(v) for k, v in (s.get("by_source_primary") or {}).items()},
                stationary_while_overlay_ge5_seconds=still)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cmp11", required=True)
    ap.add_argument("--cmp12", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    v11, v12 = version(a.cmp11), version(a.cmp12)
    A, S, M = v12["all_valid"], v12["stopped"], v12["moving30"]
    val = {1: A["coverage"], 2: A["mae"], 4: S["mae"], 5: S["coverage"], 6: M["mae"], 7: M["coverage"],
           8: (v12["primary"] or {}).get("cover95"), 9: (v12["tenth_version_primary"] or {}).get("median_abs")}
    upper = {2, 4, 6, 9}                                        # "at most" expectations; the others are "at least"
    exp = {}
    for k, thr in THRESHOLDS.items():
        v = val[k]
        exp[k] = dict(value=v, threshold=thr, kind="<=" if k in upper else ">=",
                      met=None if v is None else bool(v <= thr if k in upper else v >= thr))
    a11 = v11["all_valid"]
    exp[3] = dict(value=dict(mae_v12=A["mae"], mae_v11=a11["mae"], coverage_v12=A["coverage"], coverage_v11=a11["coverage"]),
                  threshold="v12 MAE smaller and v12 coverage higher", kind="both",
                  met=None if None in (A["mae"], a11["mae"]) else bool(A["mae"] < a11["mae"] and A["coverage"] > a11["coverage"]))
    out = dict(label=a.label, cmp11=str(a.cmp11), cmp12=str(a.cmp12), v11=v11, v12=v12,
               expectations={str(k): exp[k] for k in sorted(exp)})
    Path(a.out).write_text(json.dumps(out, indent=1))
    f = lambda x, nd=2: "-" if x is None else f"{x:.{nd}f}"
    print(f"{'':22s}{'v11':>22s}{'v12':>22s}")
    for name, key in (("all valid seconds", "all_valid"), ("stopped (r < 5)", "stopped"), ("moving (r >= 30)", "moving30")):
        x, y = v11[key], v12[key]
        print(f"{name:22s}{'cov ' + f(x['coverage']) + ' MAE ' + f(x['mae']) + ' med ' + f(x['median_abs']):>26s}"
              f"{'cov ' + f(y['coverage']) + ' MAE ' + f(y['mae']) + ' med ' + f(y['median_abs']):>26s}  (n {y['seconds']})")
    for k in sorted(exp):
        e = exp[k]
        v = e["value"] if not isinstance(e["value"], dict) else {q: round(w, 3) for q, w in e["value"].items() if w is not None}
        print(f"  #{k}: {v if isinstance(v, dict) else f(v, 3)}  {e['kind']} {e['threshold']}  ->  {'met' if e['met'] else 'NOT met' if e['met'] is not None else 'no data'}")


if __name__ == "__main__":
    main()

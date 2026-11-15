#!/usr/bin/env python3
"""Taiwan dashcam (DOD), highway blind test: the numbers of docs/DOD_V12_BLIND2_PREREG.md section 3.1 from two outputs of
tools/dod_compare.py (eleventh and twelfth version of the same batch against the overlay speed; no time shift).

Reads only seconds.csv of the two comparison folders (they hold the overlay speed: a SCORING tool, run after the seals).
Main group. The primary set is every second whose overlay speed r is at least --min-speed (default 60 km/h):
  coverage   share of those seconds with a method value m
  MAE        mean |m - r| over the seconds with both; median, bias (mean m - r) beside it
  cover95    share of the seconds with a value whose overlay speed lies inside the method's 95 % range [lo, hi]
  by source  tenth-version seconds (source v10:*), fill seconds (marking, camera A ...), standing-still seconds
Uncertainty: bootstrap over clips, 2,000 resamples, seed 20261016. The overlay is a reference: differences (差距), not errors.

usage: dod_blind2_score.py --cmp11 DIR --cmp12 DIR --out FILE.json [--min-speed 60] [--label TEXT]
"""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

THRESHOLDS = {1: 0.60, 2: 6.0, 4: 5.0, 5: 0.005, 6: 0.70}
MIN_V10 = 30            # expectation 4 needs at least this many tenth-version seconds in the primary set


def load(cmp_dir):
    by = defaultdict(list)
    for x in csv.DictReader(open(Path(cmp_dir) / "seconds.csv")):
        if x["group"] != "main" or not x["r"]:
            continue
        f = lambda k: float(x[k]) if x.get(k) not in (None, "") else None
        by[x["tag"]].append(dict(r=float(x["r"]), m=f("m"), lo=f("lo"), hi=f("hi"), src=x["source"]))
    return by


def kind(src):
    return "v10" if src.startswith("v10") else "stationary" if src.startswith("stationary") else "fill"


def block(by, sel):
    def one(tags):
        n = have = cov_n = cov_ok = 0
        d, b = [], []
        for t in tags:
            for s in by[t]:
                if not sel(s):
                    continue
                n += 1
                if s["m"] is None:
                    continue
                have += 1
                d.append(abs(s["m"] - s["r"]))
                b.append(s["m"] - s["r"])
                if s["lo"] is not None and s["hi"] is not None:
                    cov_n += 1
                    cov_ok += s["lo"] <= s["r"] <= s["hi"]
        return n, have, d, b, cov_n, cov_ok
    tags = sorted(by)
    n, have, d, b, cn, ck = one(tags)
    rng = np.random.default_rng(20261016)
    cb, mb = [], []
    for _ in range(2000):
        pick = [tags[i] for i in rng.integers(0, len(tags), len(tags))]
        bn, bh, bd, *_ = one(pick)
        if bn:
            cb.append(bh / bn)
        if bd:
            mb.append(float(np.mean(bd)))
    pct = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None
    return dict(seconds=n, with_value=have, coverage=have / n if n else None, mae=float(np.mean(d)) if d else None,
                median_abs=float(np.median(d)) if d else None, bias=float(np.mean(b)) if b else None,
                cover95=ck / cn if cn else None, cover95_n=cn, coverage_ci95=pct(cb), mae_ci95=pct(mb),
                clips=sum(1 for t in tags if any(sel(s) for s in by[t])))


def version(cmp_dir, vmin):
    by = load(cmp_dir)
    hw = lambda s: s["r"] >= vmin
    out = dict(primary=block(by, hw), all_valid=block(by, lambda s: True), stopped=block(by, lambda s: s["r"] < 5),
               slow_5_to_min=block(by, lambda s: 5 <= s["r"] < vmin), by_source_primary={})
    for k in ("v10", "fill", "stationary"):
        out["by_source_primary"][k] = block(by, lambda s, k=k: hw(s) and (s["m"] is None or kind(s["src"]) == k))
        n = sum(1 for t in by for s in by[t] if hw(s) and s["m"] is not None and kind(s["src"]) == k)
        out["by_source_primary"][k] = dict(seconds_with_value=n, mae=out["by_source_primary"][k]["mae"],
                                           median_abs=out["by_source_primary"][k]["median_abs"],
                                           bias=out["by_source_primary"][k]["bias"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cmp11", required=True)
    ap.add_argument("--cmp12", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-speed", type=float, default=60.0)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    v11, v12 = version(a.cmp11, a.min_speed), version(a.cmp12, a.min_speed)
    P, P11 = v12["primary"], v11["primary"]
    v10 = v12["by_source_primary"]["v10"]
    still = v12["by_source_primary"]["stationary"]["seconds_with_value"]
    exp = {
        1: dict(value=P["coverage"], kind=">=", threshold=THRESHOLDS[1]),
        2: dict(value=P["mae"], kind="<=", threshold=THRESHOLDS[2]),
        3: dict(value=dict(mae_v12=P["mae"], mae_v11=P11["mae"]), kind="v12 <= v11", threshold="twelfth version not worse than the eleventh"),
        4: dict(value=v10["mae"] if v10["seconds_with_value"] >= MIN_V10 else None, kind="<=", threshold=THRESHOLDS[4],
                seconds=v10["seconds_with_value"]),
        5: dict(value=still / P["seconds"] if P["seconds"] else None, kind="<=", threshold=THRESHOLDS[5], seconds=still),
        6: dict(value=P["cover95"], kind=">=", threshold=THRESHOLDS[6]),
    }
    for k, e in exp.items():
        v = e["value"]
        if k == 3:
            e["met"] = None if None in (v["mae_v12"], v["mae_v11"]) else bool(v["mae_v12"] <= v["mae_v11"])
        else:
            e["met"] = None if v is None else bool(v <= e["threshold"] if e["kind"] == "<=" else v >= e["threshold"])
    out = dict(label=a.label, min_speed=a.min_speed, cmp11=str(a.cmp11), cmp12=str(a.cmp12), v11=v11, v12=v12,
               expectations={str(k): exp[k] for k in sorted(exp)})
    Path(a.out).write_text(json.dumps(out, indent=1))
    f = lambda x, nd=2: "-" if x is None else f"{x:.{nd}f}"
    print(f"{'':26s}{'v11':>34s}{'v12':>34s}")
    for name, key in ((f"overlay >= {a.min_speed:.0f} km/h", "primary"), (f"5 .. {a.min_speed:.0f} km/h", "slow_5_to_min"),
                      ("stopped (< 5)", "stopped"), ("all valid seconds", "all_valid")):
        x, y = v11[key], v12[key]
        g = lambda z: f"n {z['seconds']:5d} cov {f(z['coverage'])} MAE {f(z['mae'])} c95 {f(z['cover95'])}"
        print(f"{name:26s}{g(x):>34s}{g(y):>34s}")
    print("  v12 primary by source:", {k: (v["seconds_with_value"], None if v["mae"] is None else round(v["mae"], 2))
                                      for k, v in v12["by_source_primary"].items()})
    for k in sorted(exp):
        e = exp[k]
        v = e["value"]
        vs = {q: (None if w is None else round(w, 3)) for q, w in v.items()} if isinstance(v, dict) else f(v, 3)
        print(f"  #{k}: {vs}  {e['kind']} {e['threshold']}  ->  {'met' if e['met'] else 'NOT met' if e['met'] is not None else 'no data'}")


if __name__ == "__main__":
    main()

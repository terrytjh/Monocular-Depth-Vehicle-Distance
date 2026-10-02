#!/usr/bin/env python3
"""Score the sixth version (tools/v6_far_ruler.py) on comma2k19 segments (docs/DEPTH_DASH_V6_PREREG.md section 3).

Only what the sixth version changes is scored here: far readings (beyond the depth ruler's reach, lanes -1..1) that got
the car's own ruler. Each is compared with the same reading in the fifth version, paired to the SAME radar target in
both (the 50 % sanity gate is applied before the bearing assignment, so changing far distances can move a few pairs; the
fifth version's own numbers are scored from the fifth-version files, not from these). Corrected truth (radar - 0.33 m).

  python3 tools/v6_score.py --list tags.txt --cars-dir DIR --seal DIR/seal.json --seal-v6 DIR/seal_v6.json --out X.json
"""
import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

import depth_dash_multicar as M

BINS = ((30, 40), (40, 50), (50, 60), (60, 80))


def summary(e):
    e = np.asarray(e, float)
    if not len(e):
        return dict(n=0)
    return dict(n=int(len(e)), median_abs_pct=float(np.median(np.abs(e))), bias_pct=float(np.median(e)),
                gross_gt25_pct=float(100 * np.mean(np.abs(e) > 25)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True)
    ap.add_argument("--cars-dir", required=True)
    ap.add_argument("--seal", required=True, help="seal of the fifth-version files")
    ap.add_argument("--seal-v6", required=True, help="seal of the sixth-version files")
    ap.add_argument("--truth-root", default="data/output/c2k19_truth")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    M.PARAMS["radar_processing_m"] = 2.70                       # corrected truth
    cdir = Path(a.cars_dir)
    tags = [l.strip() for l in open(a.list) if l.strip() and not l.startswith("#")]
    have = [t for t in tags if (cdir / f"{t}_cars_v5.json").exists() and (cdir / f"{t}_cars_v6.json").exists()]
    M.check_sealed(a.seal, [str(cdir / f"{t}_cars_v5.json") for t in have])
    M.check_sealed(a.seal_v6, [str(cdir / f"{t}_cars_v6.json") for t in have])
    rows, far_all, inside_moved, segs_out = [], 0, 0, 0
    for t in have:
        c5 = json.loads((cdir / f"{t}_cars_v5.json").read_text())
        c6 = json.loads((cdir / f"{t}_cars_v6.json").read_text())
        if c5.get("status") == "refused":
            continue
        segs_out += 1
        td = Path(a.truth_root) / t
        _, p5, _ = M.c2k19_score(c5, td, return_rows=True)
        _, p6, _ = M.c2k19_score(c6, td, return_rows=True)
        k5 = {(round(q["t"], 3), q["id"]): q for q in p5 if q["lane"] in (-1, 0, 1)}
        k6 = {(round(q["t"], 3), q["id"]): q for q in p6 if q["lane"] in (-1, 0, 1)}
        ruler = {(round(s["t"], 3), o["id"]) for s in c6.get("samples", []) for o in s["objects"] if o.get("far_ruler")}
        for key, q in k5.items():
            if not q["far"]:
                inside_moved += key not in k6 or abs(k6[key]["err"] - q["err"]) > 1e-9
                continue
            far_all += 1
            r = k6.get(key)
            if key in ruler and r is not None and r["truth_id"] == q["truth_id"]:
                rows.append(dict(seg=t, truth=q["truth"], e5=q["err"], e6=r["err"]))
    tr = np.array([r["truth"] for r in rows])
    e5 = np.array([r["e5"] for r in rows]); e6 = np.array([r["e6"] for r in rows])
    by_seg = defaultdict(list)
    for r in rows:
        by_seg[r["seg"]].append((abs(r["e5"]), abs(r["e6"])))
    segs = {s: v for s, v in by_seg.items() if len(v) >= 10}
    better = sum(np.median([b for _, b in v]) < np.median([x for x, _ in v]) for v in segs.values())
    n = len(segs)
    out = dict(segments_listed=len(tags), segments_with_output=segs_out,
               far_readings=far_all, far_readings_with_ruler=len(rows),
               coverage=len(rows) / far_all if far_all else None,
               v5_same_readings=summary(e5), v6=summary(e6),
               bins={f"{lo}-{hi}": dict(v5=summary(e5[(tr >= lo) & (tr < hi)]), v6=summary(e6[(tr >= lo) & (tr < hi)]))
                     for lo, hi in BINS},
               segments_compared=n, segments_v6_better=int(better),
               sign_test_p_one_sided=float(sum(comb(n, i) for i in range(better, n + 1)) / 2 ** n) if n else None,
               inside_reach_pairs_moved_by_gate=int(inside_moved))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"far readings {far_all}, with the own-car ruler {len(rows)}; v5 {out['v5_same_readings'].get('median_abs_pct')} "
          f"-> v6 {out['v6'].get('median_abs_pct')} (bias {out['v6'].get('bias_pct')}); segments better {better}/{n}")


if __name__ == "__main__":
    main()

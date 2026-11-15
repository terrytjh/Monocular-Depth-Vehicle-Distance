#!/usr/bin/env python3
"""Per-second ego speed of every clip of a DOD batch, from the eleventh-version files (<tag>_cars_v11f.json), with the
95 % range (2026-10-28 plan). Reads predictions only (no answer exists in these folders; truth paths are refused).

One CSV per clip, <out>/<tag>_ego.csv, one row per second b = 0 .. floor(n_frames / fps) - 1 of the clip (second b covers
frames [b * fps, (b + 1) * fps); clip time, not the overlay clock):
  second, kmh, lo95, hi95, half95, sigma_kmh, source, steps, tier, n_pairs, kmh_before_smoothing
  kmh / half95   the eleventh version's value and 95 % half-width (symmetric for ego): lo95 = kmh - half95,
                 hi95 = kmh + half95, both as computed (lo95 may be < 0; a scorer may clip at 0)
  source         v10:both (mean of the depth dash ruler and marking geometry, then C2) | v10:depth (depth alone, then C2)
                 | the eleventh version's fill label (main-group fill MS / MC, own-ruler path E1, camera-k path K2-ego)
  kmh_before_smoothing  the ninth version's per-second value (v10 seconds) or the fill's raw median (fill seconds)
Empty kmh = no reading in that second. Also <out>/ego_summary.json: per clip status, path, group, seconds, seconds with a
value, coverage, sources; and the batch totals.

  python3 -I -B ego_export.py --v11 DIR --out DIR [--tags-dir DIR]
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

COLS = ["second", "kmh", "lo95", "hi95", "half95", "sigma_kmh", "source", "steps", "tier", "n_pairs", "kmh_before_smoothing"]


def f(x, nd=3):
    return "" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v11", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tags-dir", default=None, help="tags_{main,local_day,night}.txt, for the group column")
    a = ap.parse_args()
    src = Path(a.v11)
    if "c2k19_truth" in str(src.resolve()):
        raise SystemExit("truth path refused")
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    group = {}
    if a.tags_dir:
        for g in ("main", "local_day", "night"):
            for t in (Path(a.tags_dir) / f"tags_{g}.txt").read_text().split():
                group[t] = g
    summ, tot = {}, Counter()
    for p in sorted(src.glob("*_cars_v11f.json")):
        tag = p.name[: -len("_cars_v11f.json")]
        d = json.loads(p.read_text())
        fps, nf = d.get("fps"), d.get("n_frames")
        v = d.get("v11f") or {}
        nb = int(nf / fps // 1.0) if fps and nf else 0
        ego = {int(round(e["t0"])): e for e in d.get("ego") or []}
        if not nb and ego:
            nb = max(ego) + 1
        rows, srcs = [], Counter()
        for b in range(nb):
            e = ego.get(b) or {}
            k, hw = e.get("kmh"), e.get("ci95_kmh")
            s = e.get("v11_source") or (f"v10:{e['v10_source']}" if e.get("v10_source") else "")
            if k is not None:
                srcs[s] += 1
            before = e.get("kmh_v9") if e.get("kmh_v9") is not None else e.get("kmh_v11_raw")
            rows.append([b, f(k), f(None if k is None or hw is None else k - hw), f(None if k is None or hw is None else k + hw),
                         f(hw), f(e.get("sigma_kmh")), s if k is not None else "",
                         "+".join(e.get("v11_steps") or e.get("v10_steps") or []) if k is not None else "",
                         e.get("tier", "") if k is not None else "", e.get("n", ""), f(before) if k is not None else ""])
        with open(od / f"{tag}_ego.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(COLS)
            w.writerows(rows)
        nv = sum(srcs.values())
        summ[tag] = dict(status=d.get("status") or "ok", path=v.get("path"), fill=v.get("fill"), group=group.get(tag, v.get("group")),
                         seconds=nb, seconds_with_value=nv, coverage=round(nv / nb, 4) if nb else None, sources=dict(srcs),
                         ego_rule=((v.get("steps") or {}).get("ego") or {}).get("rule"),
                         ego_why=((v.get("steps") or {}).get("ego") or {}).get("why"))
        tot["clips"] += 1
        tot["seconds"] += nb
        tot["seconds_with_value"] += nv
        for k_, n_ in srcs.items():
            tot[f"source|{k_}"] += n_
    out = dict(made_from=str(src), clips=summ, totals=dict(tot),
               coverage=round(tot["seconds_with_value"] / tot["seconds"], 4) if tot["seconds"] else None)
    (od / "ego_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(totals=dict(tot), coverage=out["coverage"])))


if __name__ == "__main__":
    main()

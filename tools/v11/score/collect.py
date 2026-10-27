#!/usr/bin/env python3
# Copy of /mnt/d/depth_work/v11extra_20261025/score/collect.py for scoring comma2k19 Chunk_9 (docs/DEPTH_DASH_V11_PREREG.md); changed: imports f_lib from this folder.
"""v11extra copy (2026-10-25) of v11final score/f01_collect.py: two more pair columns, rlane = the reading's own lane
field and gfit (the ground fit the reading used under GL); otherwise unchanged.

Step 1 of scoring (READS TRUTH, development batches): per batch and version, every matched pair (any truth distance),
the in-scope truth counts per truth-distance bin on every segment's 0.2 s grid, and the ego seconds with their truth.

  python3 -B f01_collect.py --batch C7b --out DIR v10=<dir> v11f=<dir> [v11f_x=<dir> ...]
v10 is read as <tag>_cars_v10.json with the tenth version's number rule (0 < d <= 60 m); every other name as
<tag>_cars_v11f.json with the eleventh version's rule (d > 0, nothing blanked at 60 m).
"""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import f_lib as F  # noqa: E402

S = F.S
PAIR_COLS = ["version", "seg", "drive", "group", "path", "fill", "tier_class", "rcls", "t2new", "hood", "k4", "fi", "id", "truth_id",
             "truth", "tbin", "lane", "rlane", "gfit", "far", "far60", "dist", "dist_any", "err_pct", "err_m", "pbin", "lo_a", "hi_a", "lo_s",
             "hi_s", "u_da3", "rel", "rel_ci95", "rel_ci95_base", "relcls", "rel_truth_inst", "rel_simple", "rel_v10", "t2rel", "abs", "abs_ci95",
             "abs_truth", "abs_new_ego", "abs_v10", "steps"]
EGO_COLS = ["version", "seg", "drive", "group", "path", "fill", "sec", "truth", "acc", "kmh", "ci95", "ci95_base", "tier", "source",
            "raw", "hw9", "rule"]


def load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="sensitivity: read the width gate from this (dev copy of the) shared config")
    ap.add_argument("versions", nargs="+")
    a = ap.parse_args()
    if a.config:
        print("width gate", F.set_width_from_config(a.config), flush=True)
    b = a.batch
    vers = [v.split("=", 1) for v in a.versions]
    od = Path(a.out) / b
    od.mkdir(parents=True, exist_ok=True)
    G = S.C.groups_of(b)
    prs, egs, tcs, sgs = [], [], [], []
    for t in sorted(G):
        g = G[t]
        files = {name: load(Path(d) / (f"{t}_cars_v10.json" if name == "v10" else f"{t}_cars_v11f.json")) for name, d in vers}
        nf = S.n_frames_of(t)
        tcs.append(dict(seg=t, drive=S.L.drive_of(t), group=g, n_frames=nf, **S.truth_counts(t, nf)))
        et = S.ego_truth(t)
        acc = S.accel(et)
        for name, c in files.items():
            if c is None:
                raise SystemExit(f"{b} {t}: no file for {name}")
            refused = c.get("status") == "refused"
            blk = c.get("v11f") or {}
            path = None if refused else (blk.get("path") if name != "v10" else (c.get("v10") or {}).get("path"))
            fill = blk.get("fill") if name != "v10" else None
            sgs.append(dict(version=name, seg=t, group=g, status="refused" if refused else "ok", path=path, fill=fill,
                            truth_secs=len(et)))
            if refused:
                continue
            rule = S.v10_number if name == "v10" else S.v11_number
            for p in F.pairs_of(t, c, rule):
                o = p["obj"]
                d = p["dist"]
                dv = o.get("dist") if o.get("dist") is not None else o.get("dist_v10")
                st = o.get("v11_steps") or []
                prs.append(dict(version=name, seg=t, drive=S.L.drive_of(t), group=g, path=path, fill=fill,
                                tier_class=o.get("tier_class") or ("t1" if path == "current" else "t2k"),
                                rcls=o.get("dist_range_class") or o.get("tier_class") or ("t1" if path == "current" else "t2k"),
                                t2new=int("t2_novp" in o), hood=o.get("hood") or "", k4=int("k4" in o),
                                fi=p["fi"], id=p["id"], truth_id=p["truth_id"], truth=round(p["truth"], 4),
                                tbin=S.tbin(p["truth"]), lane=p["lane"], rlane=o.get("lane"), gfit=o.get("gfit") or "", far=int(bool(o.get("far"))),
                                far60=int(bool(d is not None and d > 60)), dist=d, dist_any=dv,
                                err_pct=None if d is None else 100 * (d - p["truth"]) / p["truth"],
                                err_m=None if d is None else d - p["truth"], pbin=None if d is None else S.tbin(d),
                                lo_a=o.get("dist_lo95_asym"), hi_a=o.get("dist_hi95_asym"), lo_s=o.get("dist_lo95_sym"),
                                hi_s=o.get("dist_hi95_sym"), u_da3=o.get("u_da3"), rel=p["rel"], rel_ci95=o.get("rel_ci95_kmh"),
                                rel_ci95_base=o.get("rel_ci95_kmh_unwidened", o.get("rel_ci95_kmh")),
                                relcls=o.get("rel_range_class") or ("t2f" if fill else ("t2rel" if "t2_rel" in o else "v10")),
                                rel_truth_inst=p["rel_truth_inst"], rel_simple=int("rel-simple" in st),
                                rel_v10=o.get("rel_kmh_v10"), t2rel=int("t2_rel" in o), abs=o.get("abs_kmh"),
                                abs_ci95=o.get("abs_ci95_kmh"), abs_truth=round(p["abs_truth"], 4),
                                abs_new_ego=int(bool(o.get("abs_from_new_ego_second"))),
                                abs_v10=o.get("abs_kmh_v10") if "abs" in st else o.get("abs_kmh"), steps="+".join(st)))
            for e in c.get("ego") or []:
                if e.get("kmh") is None:
                    continue
                s = int(round(e["t0"]))
                egs.append(dict(version=name, seg=t, drive=S.L.drive_of(t), group=g, path=path, fill=fill, sec=s,
                                truth=et.get(s), acc=acc.get(s), kmh=e["kmh"], ci95=e.get("ci95_kmh"),
                                ci95_base=e.get("ci95_kmh_unwidened", e.get("ci95_kmh")),
                                tier=e.get("tier", 1), source=e.get("v11_source") or e.get("v10_source"),
                                raw=e.get("kmh_v11_raw"), hw9=e.get("ci95_kmh_v11_raw"),
                                rule=(e.get("v11_steps") or [""])[0]))
        S.L._TRUTH.clear()
        print(b, t, g, flush=True)
    S.wcsv(od / "pairs.csv.gz", prs, PAIR_COLS)
    S.wcsv(od / "ego.csv.gz", egs, EGO_COLS)
    S.wcsv(od / "truth_counts.csv.gz", tcs, ["seg", "drive", "group", "n_frames"] + S.BIN_NAMES)
    S.wcsv(od / "segs.csv.gz", sgs, ["version", "seg", "group", "status", "path", "fill", "truth_secs"])
    print(b, "pairs", len(prs), "ego", len(egs))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""[v11f copy, 2026-10-24: unchanged except that t2lib takes its root and batch definitions from the environment]
T2 (b2, optional): relative and absolute speeds that the no-vp readings make possible, by the tenth version's own rules.

Input: the output of measure_novp.py (<tag>_cars_v10nv.json, or the identity test's <tag>_cars_v10id.json). A reading gets a
relative speed here only when it has none and the registered window of +-4 samples (tools/depth_dash_multicar.track_speeds:
all nine samples of the track present, none beyond the reach, no step > 15 %) is complete on the depth series WITH the
no-vp readings and contains at least one of them -- i.e. exactly the readings whose relative speed the missing maps blocked
(new readings, and existing readings next to them). Readings that already have a relative speed keep it unchanged.

The value follows tools/v10_post.py step_rel for that reading (C4: the longest complete window among the half-widths of
tools/v10_params.json; the mean with marking geometry where marking geometry has a relative speed of the same car in the same
sample, which is where the ninth version's v5_combine --rel-average averages; C5 in the main group: a depth-only reading on
the current path takes marking geometry's slope with samples beyond its reach counted up to 40 m; 95 % half-width =
1.96 x 3.4 x sigma x rel_range_factor) and the absolute speed tools/v10_post.py step_abs (the tenth version's ego of the
reading's second + the relative speed). Depth series: the hood measurement's dist_raw (*_cars_v4h.json) on the current
path, the camera-k measurement on the camera-k path (as v10), plus the no-vp readings' dist_raw / far.

Check mode (--check, CPU, no new map): on an identity-test output of measure_novp.py (readings of picked frames removed and
re-added as no-vp readings, without speeds) every re-added reading must get back exactly the tenth version's relative and
absolute speed when the original had one, and none when it had none.

  python -B rel_novp.py --batch C7b --in-dir <measure_novp out>/C7b --out <dir> [--in-suffix _cars_v10nv.json]
  python -B rel_novp.py --batch C7b --in-dir <identity out>/C7b --in-suffix _cars_v10id.json --out <dir> --check
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2lib as T  # noqa: E402
import v10_post as V10  # noqa: E402

P10 = json.loads((T.TOOLS / "v10_params.json").read_text())["values"]
TOL = 1e-9


def dep_file(batch, tag, base):
    F = Path(T.BATCHES[batch]["folder"])
    src = T.source_of(base)
    if src == "current":
        return T.jload(F / f"{tag}_cars_v4h.json", batch), src
    nc = base.get("next_candidate") or {}
    return T.jload(F / f"{tag}_cars_{'kc' if nc.get('day') else 'kn'}.json", batch), src


def process(batch, tag, group, cars):
    """Adds speeds to cars (in place); returns counts."""
    info = dict(new_readings=0, rel_added_new=0, rel_added_existing=0, marking_identity_failed=0, no_window=0)
    src = T.source_of(cars)
    if src == "refused":
        return info
    dep, src = dep_file(batch, tag, cars)
    mrk = T.jload(Path(T.BATCHES[batch]["folder"]) / f"{tag}{T.BATCHES[batch]['msfx']}", batch) if src == "current" else None
    new = {(s["fi"], o["id"]): o for s in cars["samples"] for o in s["objects"] if "t2_novp" in o}
    info["new_readings"] = len(new)
    if not new:
        return info
    aug = copy.deepcopy(dep)
    by_fi = {s["fi"]: s for s in aug["samples"]}
    for (fi, tid), o in new.items():
        s = by_fi[fi]
        s["objects"] = [x for x in s["objects"] if x["id"] != tid]          # identity test: the stripped original
        s["objects"].append(dict(id=tid, box=o["box"], dist=o["dist_raw"], dist_raw=o["dist_raw"], far=bool(o["far"])))
    p4, p5 = P10["C4"], P10["C5"]
    fac, zm = P10["rel_range_factor"], V10.M_REL_Z * p4["m_rel"]
    Sd = V10.Series(aug, "dist_raw", p4["max_jump"])
    Sm = V10.Series(mrk, "dist", p4["max_jump"]) if (src == "current" and V10.usable(mrk)) else None
    sc_d = float(dep.get("rel_scale_sigma", 0.0)) if src == "current" else float(dep.get("k_camera_sigma", 0.0))
    sc_m = float(mrk.get("ruler_scale_sigma", 0.0)) if Sm is not None else 0.0
    mrel = {(s["fi"], o["id"]): o["rel_kmh"] for s in (mrk or {}).get("samples", []) for o in s["objects"]
            if o.get("rel_kmh") is not None} if Sm is not None else {}
    c5 = group in p5["groups"] and p5["D_m"] is not None
    new_j = defaultdict(set)
    j_of = {s["fi"]: j for j, s in enumerate(aug["samples"])}
    for (fi, tid) in new:
        new_j[tid].add(j_of[fi])
    ego = {int(round(e["t0"])): e for e in cars.get("ego", []) if e.get("kmh") is not None}
    fps = cars["fps"]
    for s in cars["samples"]:
        for o in s["objects"]:
            if o.get("rel_kmh") is not None or o.get("dist") is None:
                continue
            fi, tid = s["fi"], o["id"]
            j = j_of[fi]
            f4 = Sd.fit(fi, tid, 4)
            if f4 is None or not any(k in new_j[tid] for k in range(j - 4, j + 5)):
                info["no_window"] += 1
                continue
            averaged = (fi, tid) in mrel
            fd = Sd.longest(fi, tid, p4["halfwidths_samples"])
            fm, ext = None, False
            if averaged:
                fm4 = Sm.longest(fi, tid, (4,))
                if fm4 is None or abs(fm4[0] - mrel[(fi, tid)]) > 1e-6:
                    info["marking_identity_failed"] += 1
                    continue
                fm = Sm.longest(fi, tid, p4["halfwidths_samples"])
            elif c5 and src == "current" and Sm is not None:
                fx = Sm.longest(fi, tid, p4["halfwidths_samples"], dmax=p5["D_m"])
                if fx is not None:
                    fm, ext = fx, True
            vd, sed = fd[0], fd[1]
            sd = float(np.hypot(sed, abs(vd) * sc_d))
            if fm is not None:
                vm = fm[0]
                sm = float(np.hypot(fm[1], abs(vm) * sc_m))
                v = 0.5 * vd + 0.5 * vm
                sig = float(np.sqrt(0.25 * sd ** 2 + 0.25 * sm ** 2))
                o["rel_kmh_depth"], o["rel_kmh_marking"], o["rel_hw_marking"] = vd, vm, fm[2]
                f0, f1 = min(fd[3], fm[3]), max(fd[4], fm[4])
            else:
                v, sig = vd, sd
                f0, f1 = fd[3], fd[4]
            o["rel_hw_depth"] = fd[2]
            o["rel_kmh"], o["rel_sigma_kmh"], o["rel_ci95_kmh"] = float(v), sig, fac * zm * sig
            o["win"] = [f0, f1]
            if ext:
                o["rel_marking_extended_to_m"] = p5["D_m"]
            b = int(math.floor(fi / fps + 1e-9))
            e = ego.get(b)
            o["abs_kmh"] = None
            if e is not None and e.get("ci95_kmh") is not None:
                o["abs_kmh"] = float(e["kmh"] + o["rel_kmh"])
                o["abs_ci95_kmh"] = float(np.hypot(e["ci95_kmh"], o["rel_ci95_kmh"]))
                o["abs_ego_second"] = b
            o["t2_rel"] = dict(window_new_samples=sum(1 for k in range(j - fd[2], j + fd[2] + 1) if k in new_j[tid]),
                               averaged=fm is not None, c5=ext)
            if "t2_novp" in o:
                info["rel_added_new"] += 1
            else:
                info["rel_added_existing"] += 1
    return info


def check(batch, tag, cars_id, base):
    """Re-added readings vs the tenth version's originals: (n compared, n mismatched, max |d rel|, max |d abs|)."""
    orig = {(s["fi"], o["id"]): o for s in base["samples"] for o in s["objects"]}
    n = bad = 0
    dmax = amax = 0.0
    for s in cars_id["samples"]:
        for o in s["objects"]:
            if "t2_novp" not in o:
                continue
            a = orig.get((s["fi"], o["id"]))
            if a is None or "rel-identity-failed-kept-v9" in (a.get("v10_steps") or []):
                continue
            n += 1
            ra, rb = a.get("rel_kmh"), o.get("rel_kmh")
            if (ra is None) != (rb is None):
                bad += 1
                continue
            if ra is not None:
                dmax = max(dmax, abs(ra - rb))
                xa, xb = a.get("abs_kmh"), o.get("abs_kmh")
                if (xa is None) != (xb is None):
                    bad += 1
                    continue
                if xa is not None:
                    amax = max(amax, abs(xa - xb))
                if abs(ra - rb) > 1e-6 or (xa is not None and abs(xa - xb) > 1e-6):
                    bad += 1
    return n, bad, dmax, amax


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True, choices=list(T.BATCHES))
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--in-suffix", default="_cars_v10nv.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tags", nargs="*", default=None)
    a = ap.parse_args()
    if not T.inside_t2(a.out):
        raise SystemExit("--out outside the track folder")
    G = T.tags_and_groups(a.batch)
    tags = sorted(G) if not a.tags else [t for t in sorted(G) if t in set(a.tags)]
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    summ = dict(batch=a.batch, in_dir=a.in_dir, in_suffix=a.in_suffix, check=a.check, started=T.now(), segments={})
    tot = defaultdict(float)
    for tag in tags:
        p = Path(a.in_dir) / f"{tag}{a.in_suffix}"
        if not p.exists():
            continue
        T.wait_memory(2500)
        cars = json.loads(p.read_text())
        before = json.dumps(cars, sort_keys=True)
        info = process(a.batch, tag, G[tag], cars)
        if a.check:
            base = T.jload(Path(T.BATCHES[a.batch]["v10"]) / f"{tag}_cars_v10.json")
            n, bad, dmax, amax = check(a.batch, tag, cars, base) if T.source_of(base) != "refused" else (0, 0, 0.0, 0.0)
            info.update(compared=n, mismatched=bad, max_abs_d_rel=dmax, max_abs_d_abs=amax)
            tot["max_abs_d_rel"] = max(tot["max_abs_d_rel"], dmax)
            tot["max_abs_d_abs"] = max(tot["max_abs_d_abs"], amax)
        else:
            (od / f"{tag}_cars_v10nvr.json").write_text(json.dumps(cars))
        for k, v in info.items():
            if k not in ("max_abs_d_rel", "max_abs_d_abs"):
                tot[k] += v
        summ["segments"][tag] = info
        if before == json.dumps(cars, sort_keys=True) and info["rel_added_new"] + info["rel_added_existing"]:
            raise SystemExit(f"{tag}: counts and file disagree")
    summ["totals"] = dict(tot)
    summ["ended"] = T.now()
    (od / f"rel_{'check' if a.check else 'run'}_{a.batch}.json").write_text(json.dumps(summ, indent=1))
    print(f"{a.batch}: {dict(tot)}", flush=True)
    if a.check and (tot.get("mismatched", 0) or tot.get("rel_added_existing", 0)):
        raise SystemExit(f"{a.batch}: relative-speed check failed: {int(tot.get('mismatched', 0))} re-added readings differ, "
                         f"{int(tot.get('rel_added_existing', 0))} existing readings would gain a speed the ninth version did not give")


if __name__ == "__main__":
    main()

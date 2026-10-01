#!/usr/bin/env python3
"""Development only (2026-10-01, toward a third version): a ruler that follows the clip in time.

The blind test of the second version left the depth method's ego speed at 5.8 km/h per second, and the error lasts
seconds rather than flickering from frame to frame (per-second errors correlated 0.60 / 0.45 / 0.34 at 1 / 2 / 3 s;
the spread of the readings inside one second is no larger than method C's). That is what a scale that drifts over
seconds would do. This script re-scores OPENED segments with variants that change one thing at a time and runs
everything else through the registered code:

  ego speed  the dash run's pairwise readings, each multiplied by k_local(t) / k (a reading is k x shift / time, so
             linear in k); k_local = legal cycle / median of the cycles measured within +-W s (at least n_min of them,
             otherwise the clip's k). Optionally method C's jump filter (a reading more than 15 % from the median of
             the readings within +-0.5 s is dropped), or method C's own readings in place of the depth ones.
  distances  every car reading multiplied by k_car_local(t) / k_car, the car ruler (far cycles, as registered)
             measured within +-W s.
  then       relative / absolute speed (track_speeds), ego bins (ego_bins) and the comma2k19 scoring (c2k19_score),
             unchanged.

The variant that changes nothing must reproduce the stored scores; the script stops if it does not. Variants run on
the development segments only (daytime freeway of the development 52). The scored blind groups take --registered-only
(the reproduction check, nothing else), and every prediction file read there is checked against the seal first.

  python3 tools/scale_drift_dev.py eval --set dev
  python3 tools/scale_drift_dev.py eval --set blind_main --registered-only --out data/output/dash_scale/v3dev/blind_main.json
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

import depth_dash_multicar as DM

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "data/output/dash_scale"
TRUTH = ROOT / "data/output/c2k19_truth"

SETS = {
    # the 30 daytime-freeway segments of the development 52 (all run with 14.63 m); the registered k guard is applied
    # here because the stored development measurements predate it
    "dev": dict(tags=O / "dev52/tags_dayhw.txt", dash=O / "dev52/{t}_dash.json", cars=O / "dev52/{t}_cars_far.json",
                run=O / "dev52c/{t}_run.json", cars_c=O / "dev52c/{t}_cars_far.json", seal=None, k_guard=1.5,
                stored=O / "dev52/batch_dayhw_cars_far.json"),
    "blind_main": dict(tags=O / "blind_v2/tags_main.txt", dash=O / "blind_v2/pred/{t}_dash.json",
                       cars=O / "blind_v2/pred/{t}_cars.json", run=O / "blind_v2/pred/{t}_run.json",
                       cars_c=O / "blind_v2/pred/{t}_cars_c.json", seal=O / "blind_v2/seal.json", k_guard=None,
                       stored=O / "blind_v2/score_main_depth.json"),
    "blind_local": dict(tags=O / "blind_v2/tags_local_day.txt", dash=O / "blind_v2/pred/{t}_dash.json",
                        cars=O / "blind_v2/pred/{t}_cars.json", run=O / "blind_v2/pred/{t}_run.json",
                        cars_c=O / "blind_v2/pred/{t}_cars_c.json", seal=O / "blind_v2/seal.json", k_guard=None,
                        stored=O / "blind_v2/score_local_day_depth.json"),
}

# One change at a time from the registered run ("registered"), then the combination. The ego ruler (+-2.5 s, >= 20
# cycles of both lines pooled) is method C's registered local ruler for distances; the jump filter (15 % within
# +-0.5 s) is method C's registered one; the car ruler's minimum (10 far cycles) is the registered car_ruler_min.
# All fixed before this script was run; nothing is swept. The edges are still matched with the clip's k (the pairing
# windows in depth_dash_scale.edge_speeds); only the conversion to km/h follows k_local. Each variant computes from
# the video alone: the depth model and the legal dash cycles of the same clip. "reference" rows are not candidates.
VARIANTS = [
    dict(name="registered"),
    dict(name="ego: jump filter", jump=True),
    dict(name="ego: local k +-2.5 s", ego_w=2.5),
    dict(name="ego: local k +-2.5 s + jump", ego_w=2.5, jump=True),
    dict(name="dist: local k_car +-2.5 s", dist_w=2.5),
    dict(name="all three", ego_w=2.5, jump=True, dist_w=2.5),
    dict(name="reference: method C ego readings", ego="c"),
]
EGO_NMIN = DM.EGO_RULER["n_min"]   # cycles needed inside the window for a local k (method C's registered local ruler: 20)
DIST_NMIN = 10         # far cycles needed for a local car ruler (the registered car_ruler_min for the clip)
JUMP = dict(half_s=0.5, dev=0.15)   # method C's registered jump filter
COMPUTED = ("win", "rel_kmh", "rel_sigma_kmh", "rel_ci95_kmh", "abs_kmh", "abs_ci95_kmh", "ego_win_kmh",
            "ego_win_sigma_kmh")


def local_k(cycles, L, w, n_min):
    """cycles: (time s, period in model units). Returns t -> L / median period within +-w s, or None."""
    t = np.array([a for a, _ in cycles], float)
    v = np.array([b for _, b in cycles], float)

    def f(ts):
        m = np.abs(t - ts) <= w
        return L / float(np.median(v[m])) if m.sum() >= n_min else None
    return f


def jump_filter(speeds, half_s=JUMP["half_s"], dev=JUMP["dev"]):
    t = np.array([s["t_s"] for s in speeds if s["kmh"] is not None])
    v = np.array([s["kmh"] for s in speeds if s["kmh"] is not None])
    out = []
    for s in speeds:
        if s["kmh"] is not None:
            m = np.abs(t - s["t_s"]) <= half_s
            if m.sum() >= 3:
                md = float(np.median(v[m]))
                if md > 0 and abs(s["kmh"] - md) > dev * md:
                    s = dict(s, kmh=None)
        out.append(s)
    return out


def ego_speeds(dash, run, v):
    """The pairwise ego readings of a variant (list of dicts with frame_i, frame_j, t_s, kmh)."""
    if v.get("ego") == "c":
        sp = [dict(frame_i=s["frame_i"], frame_j=s["frame_j"], t_s=s["t_s"], kmh=s["kmh"]) for s in run["speeds"]]
    else:
        sp = [dict(s) for s in dash["speeds"]]
        if v.get("ego_w"):          # the same function `measure --ego-ruler local` runs
            sp = DM.local_ruler_speeds(dash, w=v["ego_w"], n_min=EGO_NMIN)
    if v.get("jump"):
        sp = jump_filter(sp)
    return sp


def rebuild(cars, dash, run, v):
    """A cars dict as cmd_measure would have written it under variant v (distances, speeds and ego bins)."""
    samples = [dict(fi=s["fi"], t=s["t"], objects=[{a: b for a, b in o.items() if a not in COMPUTED}
                                                     for o in s["objects"]]) for s in cars["samples"]]
    if v.get("dist_w"):
        k, L, step, fps = dash["k"], dash["cycle_m"], dash["step"], dash["fps"]
        cyc = [c for c in dash["periods"] if c["side"] in dash["lines_used"]]
        k_car = cars.get("k_car", k)
        # far cycles exactly when the clip's car ruler used them (cycles are quantised to dz, so k_car == k can
        # also happen with enough far cycles)
        if cars.get("car_ruler") == "far" and (cars.get("car_ruler_far_cycles") or 0) >= DM.PARAMS["car_ruler_min"]:
            cyc = [c for c in cyc if c["z_near"] * k >= DM.PARAMS["car_ruler_from"] * L]
        f = local_k([(c["n"] * step / fps, c["period"]) for c in cyc], L, v["dist_w"], DIST_NMIN)
        for s in samples:
            kk = f(s["t"])
            r = (kk if kk is not None else k_car) / k_car
            for o in s["objects"]:
                o["dist"] = o["dist"] * r
                # o["far"] stays: cmd_measure compares d and the reach scaled by the same k_car, i.e. model depth
                # against the far ends of the cycles; a local scale must not move a car across it
    sp = ego_speeds(dash, run, v)
    pe = dict(DM.PARAMS, **DM.PARAMS_C) if v.get("ego") == "c" else DM.PARAMS
    scale_rel = cars["ruler_scale_sigma"]
    DM.track_speeds(samples, sp, scale_rel, cars["params"]["half_win"], DM.PARAMS["rel_sigma_mult"], pe["ego_sigma_mult"])
    scale_ego = DM.ruler_scale_sigma(run) if v.get("ego") == "c" else scale_rel   # method C's ego ranges: its own ruler
    return dict(cars, samples=samples, ego=DM.ego_bins(dict(speeds=sp), cars["n_frames"], cars["fps"], scale_ego, p=pe))


def load_set(name):
    S = SETS[name]
    tags = [l.strip() for l in open(S["tags"]) if l.strip() and not l.startswith("#")]
    segs, refused = {}, {}
    for t in tags:
        pc = Path(str(S["cars"]).format(t=t))
        if not pc.exists():
            refused[t] = "no cars file (ruler refused)"
            continue
        cars = json.loads(pc.read_text())
        if cars.get("status") == "refused":
            refused[t] = cars.get("reason", "")[:80]
            continue
        dash = json.loads(Path(str(S["dash"]).format(t=t)).read_text())
        g = S["k_guard"]
        if g and not (1 / g <= dash["k"] <= g):
            refused[t] = f"k guard (k {dash['k']:.3f})"
            continue
        pr = Path(str(S["run"]).format(t=t))
        run = json.loads(pr.read_text()) if pr.exists() else None
        segs[t] = dict(cars=cars, dash=dash, run=run, paths=[pc, Path(str(S["dash"]).format(t=t))] + ([pr] if run else []))
    if S["seal"]:
        # every prediction file rebuild() reads: cars, dash (ego readings, cycles) and the method C run
        DM.check_sealed(S["seal"], [str(p) for s in segs.values() for p in s["paths"]])
    else:
        out = [t for t in tags if t not in DM.C2K19_DEV]
        if out:
            raise SystemExit(f"{len(out)} segments of {name} are not development segments -- refusing")
    return tags, segs, refused


def pool(per):
    pairs = [dict(p, seg=t) for t, (r, pr, eg) in per.items() for p in pr]
    ego = [dict(e, seg=t) for t, (r, pr, eg) in per.items() for e in eg]
    found = sum(r["found"]["matched"] for r, _, _ in per.values())
    total = sum(r["found"]["truth_inside_reach"] for r, _, _ in per.values())
    return DM.assemble(pairs, ego, found, total, None, "per segment", DM.PARAMS["radar_offset_m"],
                       DM.PARAMS["ego_bin_s"]), pairs, ego


def summary(P):
    d, g = P["distance"], lambda k, f: P[k].get(f, float("nan"))
    return dict(own=d["own lane, inside reach"].get("median_abs_pct"), own_bias=d["own lane, inside reach"].get("bias_pct"),
                next=d["next lanes, inside reach"].get("median_abs_pct"),
                next_bias=d["next lanes, inside reach"].get("bias_pct"),
                rel=g("relative_speed", "mae_kmh"), rel_n=g("relative_speed", "n"), rel_cov=g("relative_speed", "coverage_95"),
                abs=g("absolute_speed", "mae_kmh"), abs_n=g("absolute_speed", "n"), abs_cov=g("absolute_speed", "coverage_95"),
                ego=g("ego_per_bin", "mae_kmh"), ego_n=g("ego_per_bin", "n"), ego_bias=g("ego_per_bin", "bias_kmh"),
                ego_cov=g("ego_per_bin", "coverage_95"))


def paired_ego(base, other):
    """MAE of both on the ego bins both have (same segment, same second)."""
    b = {(e["seg"], e["t0"]): e["kmh"] - e["truth"] for e in base if e["kmh"] is not None and e.get("truth") is not None}
    o = {(e["seg"], e["t0"]): e["kmh"] - e["truth"] for e in other if e["kmh"] is not None and e.get("truth") is not None}
    k = sorted(set(b) & set(o))
    if not k:
        return dict(n=0)
    eb, eo = np.array([b[x] for x in k]), np.array([o[x] for x in k])
    segs = sorted({x[0] for x in k})
    wins = sum(np.mean(np.abs([o[x] for x in k if x[0] == s])) < np.mean(np.abs([b[x] for x in k if x[0] == s]))
               for s in segs)
    return dict(n=len(k), base=float(np.mean(np.abs(eb))), variant=float(np.mean(np.abs(eo))), segs=len(segs),
                variant_better_segs=int(wins))


def cmd_eval(a):
    if SETS[a.set]["seal"] and not a.registered_only:
        raise SystemExit(f"{a.set} is a scored blind group: --registered-only (variants are never scored on it)")
    tags, segs, refused = load_set(a.set)
    print(f"{a.set}: {len(tags)} segments, {len(segs)} with cars, {len(refused)} refused")
    rows, base_ego = [], None
    for v in (VARIANTS[:1] if a.registered_only else VARIANTS):
        per = {}
        for t, s in segs.items():
            if v.get("ego") == "c" and (s["run"] is None or s["run"].get("status") != "ok"):
                continue
            cars = copy.deepcopy(s["cars"])
            per[t] = DM.c2k19_score(rebuild(cars, s["dash"], s["run"], v), TRUTH / t, return_rows=True)
        P, pairs, ego = pool(per)
        sm = summary(P)
        if v["name"] == "registered":
            base_ego = ego
            st = json.loads(Path(SETS[a.set]["stored"]).read_text())["pooled"] if SETS[a.set]["stored"].exists() else None
            if st is not None:
                ref = summary(st)
                keys = ["own", "next", "rel", "abs", "ego", "rel_n", "abs_n", "ego_n"]
                same_mult = all(s["cars"]["params"].get(m) == DM.PARAMS[m] for s in segs.values()
                                for m in ("rel_sigma_mult", "ego_sigma_mult"))
                if same_mult:
                    keys += ["rel_cov", "abs_cov", "ego_cov"]
                bad = {k: (sm[k], ref[k]) for k in keys if (sm[k] is None) != (ref[k] is None) or
                       (sm[k] is not None and abs(sm[k] - ref[k]) > 1e-9)}
                if bad:
                    raise SystemExit(f"  DIFFERS from the stored scores: {bad}")
                print("  reproduces the stored scores" + ("" if same_mult else
                      " (errors and counts; the stored ranges used other multipliers, so coverage is not compared)"))
                sm["reproduces_stored"] = True
        sm["paired_ego_vs_registered"] = paired_ego(base_ego, ego)
        sm["segments"] = len(per)
        rows.append(dict(variant=v, **sm))
    print(f"\n{'variant':40s} {'segs':>4s} {'own%':>5s} {'next%':>5s} {'rel':>5s} {'abs':>5s} {'ego':>5s} "
          f"{'egoN':>5s} {'ego paired (reg -> var, segs better)':>38s} {'cov rel/abs/ego':>16s}")
    for r in rows:
        pe = r["paired_ego_vs_registered"]
        pes = f"{pe['base']:.2f} -> {pe['variant']:.2f} ({pe['variant_better_segs']}/{pe['segs']})" if pe.get("n") else "-"
        f = lambda x: f"{x:5.2f}" if isinstance(x, (int, float)) and x == x else "    -"
        print(f"{r['variant']['name']:40s} {r['segments']:4d} {f(r['own'])} {f(r['next'])} {f(r['rel'])} {f(r['abs'])} "
              f"{f(r['ego'])} {r['ego_n']:5d} {pes:>38s} "
              f"{100 * r['rel_cov']:4.0f}/{100 * r['abs_cov']:3.0f}/{100 * r['ego_cov']:3.0f}%")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(dict(set=a.set, refused=refused, ego_nmin=EGO_NMIN, dist_nmin=DIST_NMIN,
                                               jump=JUMP, rows=rows), indent=1))
        print(f"wrote {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--set", choices=sorted(SETS), required=True)
    e.add_argument("--out")
    e.add_argument("--registered-only", action="store_true",
                   help="only check that the unchanged run reproduces the stored scores (for the scored blind groups)")
    a = ap.parse_args()
    {"eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Twelfth version (development, 2026-11-10; not registered): the eleventh version's post step (v11f_post.py post) with
two changes, everything else unchanged (same inputs, parameters, order of steps):

  S1 per-side camera A in the main-group fill MH (tools/v12/car_a_sides.py): the fill's marking pair speeds are scaled
     line by line with that SIDE's camera constant instead of one camera A for both lines --
       MS (run passed):  kmh_side x A_cam_side / A_local_side   (v11: x A_cam / A_local_side)
       MC (run refused): kmh_side x A_cam_side / A_cam          (v11: kmh_side as measured with A_cam)
     the pair value is the mean over its lines; pairs the run's own speed filter dropped stay dropped. A side without a
     constant from another drive falls back to the single camera A.
  S3 (with --hz-tol) the MC fill is refused for a segment whose own horizon row (its marking run's median) lies more than
     hz_tol x image height from the camera's usual horizon (median of the other drives' medians,
     tools/v12/car_a_sides.reference_horizon): a wrong horizon
     scales every marking distance and speed.
  S2 standing still (tools/v12/stationary.py): a second whose image motion (--stand-key, default the 90th percentile of
     the road-band flow, motion_p90) is below --stand-thr, in a run of >= --stand-run such seconds, gets ego 0 km/h with a
     95 % half-width of --stand-ci km/h (replacing any value; source "stationary (image motion)"), before the absolute
     speeds are recomputed (abs = ego + relative, as in v11).

Both switches can be turned off (--no-sides, --no-stand); with both off the output equals the eleventh version's except
for the version key and file name. No truth is read; no GPU.
usage: v12_post.py --v11f PATH/v11f_post.py [--sides FILE] [--motion DIR --stand-thr X --stand-ci Y] [--no-sides] [--no-stand]
                   -- post --batch-def B.json --aux AUX --out DIR --params P.json   (the v11f_post arguments after --)
"""
import argparse, importlib.util, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import car_a_sides as CS  # noqa: E402

G = dict(sides=None, motion=None, thr=None, ci=None, key="motion_p90", run=1, hz_tol=None, tag=None, used={})


def load_v11f(path):
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("v11f_post", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["v11f_post"] = m
    spec.loader.exec_module(m)
    return m


def patch(F):
    orig_mh, orig_abs, orig_process = F.mh_seconds, F.step_abs_all, F.process

    def mh_seconds(run, mc, A_cam, A_sd, pm):
        if G["sides"] is None or A_cam is None:
            return orig_mh(run, mc, A_cam, A_sd, pm)
        As = {sd: CS.camera_A_side(G["sides"], G["tag"], sd) for sd in ("left", "right")}
        fb = sorted(sd for sd, v in As.items() if v is None)
        As = {sd: (A_cam if v is None else v) for sd, v in As.items()}
        G["used"][G["tag"]] = dict(A_cam=A_cam, A_side=As, fallback=fb)
        fps = run.get("fps") or 20.0
        nb = int(run["n_frames"] / fps // 1.0)
        if run.get("status") == "ok" and run.get("speeds"):
            rul = {int(x["frame"]): x for x in run.get("rulers") or []}
            side = []
            for x in run["speeds"]:
                if x.get("kmh") is None:
                    continue
                r = rul.get(int(x["frame_i"]))
                if r is None:
                    continue
                vs = [ps["kmh"] * As[sd] / r["A_side"][sd] for sd, ps in (x.get("per_side") or {}).items()
                      if ps.get("kmh") is not None and (r.get("A_side") or {}).get(sd)]
                if vs:
                    side.append((x["t_s"], float(np.mean(vs))))
            st, lab = F.bins_of(side, nb, pm["min_pairs"]), "MS"
        elif mc is not None:
            if G["hz_tol"] is not None:
                ref, how = CS.reference_horizon(G["sides"], G["tag"])
                hz = (run.get("horizon") or {}).get("median")
                tol = G["hz_tol"] * run["height"]
                G["used"][G["tag"]]["horizon"] = dict(segment=hz, reference=ref, how=how, tol_px=tol)
                if ref is not None and hz is not None and abs(hz - ref) > tol:
                    G["used"][G["tag"]]["horizon"]["refused"] = True
                    return "MC", {}
            pairs = []
            for x in mc["speeds"]:
                if x.get("kmh") is None:
                    continue
                vs = [ps["kmh"] * As[sd] / mc["A_cam"] for sd, ps in (x.get("per_side") or {}).items()
                      if ps and ps.get("kmh") is not None]
                if vs:
                    pairs.append((x["t_s"], float(np.mean(vs))))
            st, lab = F.bins_of(pairs, nb, pm["min_pairs"]), "MC"
        else:
            return None, {}
        out = {}
        for b, (med, n, sd) in st.items():
            sig = float(math.hypot(pm["median_se_factor"] * sd / math.sqrt(n), med * (A_sd or 0.0)))
            out[b] = (med, n, pm["z"] * pm["sigma_mult"] * sig, sig)
        return lab, out

    def stand(out):
        p = Path(G["motion"]) / f"{G['tag']}_motion.json"
        if not p.exists():
            out["v12_stand"] = dict(applied=False, why="no motion file")
            return
        m = json.loads(p.read_text())
        secs = sorted(m["seconds"], key=lambda s: s["second"])
        low = [s.get(G["key"]) is not None and s[G["key"]] < G["thr"] for s in secs]
        still, i = [], 0
        while i < len(secs):                       # only runs of >= G["run"] consecutive low-motion seconds count as standing
            if low[i]:
                j = i
                while j + 1 < len(secs) and low[j + 1] and secs[j + 1]["second"] == secs[j]["second"] + 1:
                    j += 1
                if j - i + 1 >= G["run"]:
                    still += [secs[q]["second"] for q in range(i, j + 1)]
                i = j + 1
            else:
                i += 1
        ego = out.setdefault("ego", [])
        idx = {int(round(e["t0"])): i for i, e in enumerate(ego)}
        rep = add = 0
        for b in still:
            new = dict(kmh=0.0, sigma_kmh=G["ci"] / 1.96, ci95_kmh=G["ci"], v11_source="stationary (image motion)",
                       stationary=True)
            if b in idx:
                e = ego[idx[b]]
                if e.get("kmh") is not None:
                    e["kmh_before_stationary"] = e["kmh"]
                    rep += 1
                else:
                    add += 1
                for k in ("kmh_v9", "kmh_v11_raw"):
                    if k in e:
                        e[k + "_before_stationary"] = e[k]
                        e[k] = 0.0
                e.update(new)
                e["v11_steps"] = list(e.get("v11_steps") or []) + ["V12-stand"]
            else:
                ego.append(dict(t0=float(b), n=0, tier=1, v11_steps=["V12-stand"], **new))
                add += 1
        ego.sort(key=lambda e: float(e["t0"]))
        out["v12_stand"] = dict(applied=True, key=G["key"], threshold=G["thr"], min_run_s=G["run"], ci95_kmh=G["ci"], still_seconds=len(still),
                                replaced=rep, added=add, motion_file=str(p))

    def step_abs_all(out, info):
        if G["motion"] is not None:
            stand(out)
        return orig_abs(out, info)

    def process(tag, group, ctx, a):
        G["tag"] = tag
        res, row = orig_process(tag, group, ctx, a)
        if res is not None:
            res["v12_changes"] = dict(version="v12", base="v11f_post.py post (unchanged steps)", sides=G["used"].get(tag),
                              stand=res.get("v12_stand"), switches=dict(sides=G["sides"] is not None, stand=G["motion"] is not None))
        return res, row

    F.mh_seconds, F.step_abs_all, F.process = mh_seconds, step_abs_all, process
    F.VERSION = "v12"


def main():
    if "--" not in sys.argv:
        sys.exit(__doc__)
    i = sys.argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v11f", required=True)
    ap.add_argument("--sides", default=None)
    ap.add_argument("--motion", default=None)
    ap.add_argument("--stand-thr", type=float, default=None)
    ap.add_argument("--stand-ci", type=float, default=None)
    ap.add_argument("--stand-key", default="motion_p90", help="motion value compared with --stand-thr (default motion_p90)")
    ap.add_argument("--stand-run", type=int, default=1, help="minimum number of consecutive low-motion seconds (default 1)")
    ap.add_argument("--hz-tol", type=float, default=None, help="S3: refuse the MC fill when the segment's horizon is more than this share of the image height from the camera's (e.g. 0.007)")
    ap.add_argument("--no-sides", action="store_true")
    ap.add_argument("--no-stand", action="store_true")
    a = ap.parse_args(sys.argv[1:i])
    if "c2k19_truth" in " ".join(sys.argv):
        sys.exit("refusing a truth path")
    if not a.no_sides:
        if not a.sides:
            sys.exit("--sides required (or --no-sides)")
        G["sides"] = json.loads(Path(a.sides).read_text())
        G["hz_tol"] = a.hz_tol
    if not a.no_stand:
        if not (a.motion and a.stand_thr is not None and a.stand_ci is not None):
            sys.exit("--motion, --stand-thr and --stand-ci required (or --no-stand)")
        G.update(motion=a.motion, thr=a.stand_thr, ci=a.stand_ci, key=a.stand_key, run=a.stand_run)
    F = load_v11f(a.v11f)
    patch(F)
    sys.argv = [a.v11f] + sys.argv[i + 1:]
    F.main()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The camera's k: the depth-to-metres factor of the dash ruler is a property of the camera and the depth model, so on
the same camera it can be learnt where the dash ruler works and reused where it does not (development, 2026-10-13).

  learn  (predictions only, no truth): for every segment of a batch, the median k of the SAME camera's daytime segments
         from OTHER drives whose dash ruler passed and whose k is inside the k guard [1/1.5, 1.5] (a k outside it is the
         half / double cycle); its relative spread (sigma = std / mean of those k). Camera = the comma2k19 dongle id.
  score  (OPENED data only; reads truth): distances inside the reach of a *_cars_kc.json directory (measure --k-camera)
         against the current method's *_cars_v5.json, split by whether the current method gave an output, by road
         class and day / night, with the scorer's 50 % gate and without it; and the ratio of each segment's own k and
         marking-geometry a to the camera's k.

  learn-night (predictions only): the depth model reads night frames at another scale, so the day k does not carry
         over. Per night segment of a batch: the median effective k at 20 m, (a x 20 + b) / 20 of the accepted ground fits
         (fourth version), over the SAME camera's night segments from OTHER drives that the fifth version accepted and
         whose effective k is within 1.3x of the day camera k (two rulers locked to the half period together pass the
         fifth version's own check; this catches them). --also BATCH:CURRENT:DAY adds another batch's night segments
         (its predictions only; e.g. the same camera's earlier sealed batch), for a batch whose night is one drive.

  python3 tools/camera_k.py learn --batch DIR --out k_camera.json
  python3 tools/camera_k.py learn-night --batch DIR --current DIR --day k_camera.json --out k_night.json
  python3 tools/camera_k.py score --batch DIR --kc DIR --current DIR --k-camera k_camera.json --out X.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
K_GUARD = 1.5


def camera(t):
    return t.split("_")[0]


def drive(t):
    return "_".join(t.split("_")[:2])


def learn(a):
    B = Path(a.batch)
    night = {l.strip() for l in open(B / "tags_night.txt") if l.strip()} if (B / "tags_night.txt").exists() else set()
    tags = ["_".join(l.split("|")[:3]) for l in (B / "run_list.txt").read_text().splitlines() if l.strip()]
    good = defaultdict(list)                     # camera -> [(drive, k)]
    for t in tags:
        p = B / f"{t}_dash.json"
        if t in night or not p.exists():
            continue
        d = json.loads(p.read_text())
        k = d.get("k")
        if d.get("status") == "ok" and k and 1 / K_GUARD <= k <= K_GUARD:
            good[camera(t)].append((drive(t), float(k)))
    out = {}
    for t in tags:
        ks = np.array([k for dr, k in good[camera(t)] if dr != drive(t)])
        if len(ks) < a.min_n:
            continue
        out[t] = dict(k_camera=float(np.median(ks)), sigma=float(ks.std() / ks.mean()), n=int(len(ks)),
                      drives=len({dr for dr, _ in good[camera(t)] if dr != drive(t)}))
    Path(a.out).write_text(json.dumps(dict(rule=__doc__.split("\n\n")[1].split("score")[0].strip(), segments=out), indent=1))
    ks = [v["k_camera"] for v in out.values()]
    print(f"{len(out)}/{len(tags)} segments got a camera k: median {np.median(ks):.3f} (range {min(ks):.3f}-{max(ks):.3f}), "
          f"sigma median {np.median([v['sigma'] for v in out.values()]):.3f}")


def effective_k(cars4, at_m=20.0, min_fits=20):
    ke = [(s["ground_fit"]["a"] * at_m + s["ground_fit"]["b"]) / at_m for s in cars4.get("samples", [])
          if (s.get("ground_fit") or {}).get("ok")]
    return float(np.median(ke)) if len(ke) >= min_fits else None


def accepted_night(B, CUR, day):
    good = []
    for t in [l.strip() for l in open(B / "tags_night.txt") if l.strip()]:
        p5, p4 = CUR / f"{t}_cars_v5.json", CUR / f"{t}_cars_v4.json"
        if not (p5.exists() and p4.exists() and t in day) or json.loads(p5.read_text()).get("status") == "refused":
            continue
        k = effective_k(json.loads(p4.read_text()))
        if k and 1 / 1.3 <= k / day[t]["k_camera"] <= 1.3:
            good.append((camera(t), drive(t), k))
    return good


def learn_night(a):
    B, CUR = Path(a.batch), Path(a.current)
    night = [l.strip() for l in open(B / "tags_night.txt") if l.strip()]
    good = accepted_night(B, CUR, json.loads(Path(a.day).read_text())["segments"])
    for x in a.also or []:
        b, c, d = x.split(":")
        good += accepted_night(Path(b), Path(c), json.loads(Path(d).read_text())["segments"])
    out = {}
    for t in night:
        ks = np.array([k for c, dr, k in good if c == camera(t) and dr != drive(t)])
        if len(ks) >= a.min_n:
            out[t] = dict(k_camera=float(np.median(ks)), sigma=float(ks.std() / ks.mean()), n=int(len(ks)))
    Path(a.out).write_text(json.dumps(dict(rule=__doc__.split("learn-night")[1].split("python3")[0].strip(), segments=out), indent=1))
    ks = [v["k_camera"] for v in out.values()]
    print(f"{len(out)}/{len(night)} night segments got a night k from {len(good)} accepted night segments"
          + (f": median {np.median(ks):.3f} ({min(ks):.3f}-{max(ks):.3f})" if ks else ""))


def score(a):
    sys.path.insert(0, str(ROOT / "tools"))
    import depth_dash_multicar as M
    B, KC, CUR = Path(a.batch), Path(a.kc), Path(a.current)
    truth = ROOT / "data/output/c2k19_truth"
    seal = json.loads((B / "seal.json").read_text())["files"]
    M.PARAMS["radar_processing_m"] = 2.70
    kcam = json.loads(Path(a.k_camera).read_text())["segments"]
    road = {"_".join(l.split("|")[:3]): l.split("|")[4].strip() for l in (B / "run_list.txt").read_text().splitlines() if l.strip()}
    night = {l.strip() for l in open(B / "tags_night.txt") if l.strip()}
    cells = defaultdict(lambda: defaultdict(list))   # (current output?, class) -> {"kc_gate": errs, ...}
    segs = defaultdict(set)
    ratios = []
    for t in sorted(road):
        if t not in kcam or not (KC / f"{t}_cars_kc.json").exists():
            continue
        if f"{t}_dash.json" not in seal or not (truth / t / "radar.csv").exists():
            raise SystemExit(f"{t}: not a sealed-and-opened segment -- refusing to score")
        cls = ("night" if t in night else "day") + "/" + road[t]
        cur_p = CUR / f"{t}_cars_v5.json"
        cur = json.loads(cur_p.read_text()) if cur_p.exists() else {"status": "refused"}
        has = cur.get("status") != "refused"
        kc = json.loads((KC / f"{t}_cars_kc.json").read_text())
        for gate in ("gate", "nogate"):
            M.PARAMS["sanity_gate"] = 0.5 if gate == "gate" else 1e9
            for lab, c in (("kc", kc), ("current", cur)) if has else (("kc", kc),):
                _, pairs, _ = M.c2k19_score(c, truth / t, return_rows=True)
                cells[(has, cls)][f"{lab}_{gate}"] += [p["err"] for p in pairs if not p["far"] and p["lane"] in (-1, 0, 1)]
        segs[(has, cls)].add(t)
        d = json.loads((B / f"{t}_dash.json").read_text())
        ka = (cur.get("v5") or {}).get("median_a")
        ratios.append(dict(seg=t, cls=cls, current_output=has, k_over_kcam=(d["k"] / kcam[t]["k_camera"]) if d.get("k") else None,
                           a_over_kcam=(ka / kcam[t]["k_camera"]) if ka else None))
    M.PARAMS["sanity_gate"] = 0.5

    def s(e):
        e = np.asarray(e, float)
        return dict(n=int(len(e))) if not len(e) else dict(n=int(len(e)), median_abs_pct=round(float(np.median(np.abs(e))), 2),
                                                            bias_pct=round(float(np.median(e)), 2),
                                                            gt25_pct=round(float(100 * np.mean(np.abs(e) > 25)), 1))
    res = {f"{'output' if has else 'refused'} | {cls}": dict(segments=len(segs[(has, cls)]), **{k: s(v) for k, v in d.items()})
           for (has, cls), d in sorted(cells.items(), key=lambda x: (not x[0][0], x[0][1]))}
    Path(a.out).write_text(json.dumps(dict(cells=res, ratios=ratios), indent=1))
    for k, v in res.items():
        line = f"{k:32s} segs {v['segments']:3d}"
        for lab in ("current_gate", "kc_gate", "current_nogate", "kc_nogate"):
            x = v.get(lab)
            if x and x.get("n"):
                line += f" | {lab} {x['median_abs_pct']:5.1f}% b{x['bias_pct']:+5.1f} >25 {x['gt25_pct']:4.1f} n{x['n']}"
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    le = sub.add_parser("learn")
    le.add_argument("--batch", required=True)
    le.add_argument("--min-n", type=int, default=5)
    le.add_argument("--out", required=True)
    ln = sub.add_parser("learn-night")
    for x in ("--batch", "--current", "--day", "--out"):
        ln.add_argument(x, required=True)
    ln.add_argument("--min-n", type=int, default=5)
    ln.add_argument("--also", action="append", help="BATCH:CURRENT:DAY_K_JSON of another batch (predictions only)")
    sc = sub.add_parser("score")
    for x in ("--batch", "--kc", "--current", "--k-camera", "--out"):
        sc.add_argument(x, required=True)
    a = ap.parse_args()
    dict(learn=learn, score=score, **{"learn-night": learn_night})[a.cmd](a)


if __name__ == "__main__":
    main()

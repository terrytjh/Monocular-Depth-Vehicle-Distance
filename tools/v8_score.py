#!/usr/bin/env python3
"""Eighth version scoring (docs/DEPTH_DASH_V8_PREREG.md section 3), comma2k19, corrected truth (radar - 0.33 m).

Run only after the seal: every prediction file it reads is first checked against the seal (sha256), before any truth
is read. The registered second to seventh versions and marking geometry are scored by score-batch as before; this adds
the eighth version, on the same segments:

  v6  the registered sixth version (*_cars_v6.json)
  v7  the registered seventh version (*_cars_v7.json)
  v8a + S5: distances of the two methods averaged inside the reach (not on daytime local roads) (*_cars_v8a.json)
  v8  + S6: in those averaged readings, the depth half from MoGe-2 instead of DA3 (*_cars_v8.json; the eighth version)

Near range first: the main numbers are the readings whose TRUE distance is within 30 m (own lane and the next lanes, the
scorer's 50 % gate), in % and in metres, and by 0-10 / 10-20 / 20-30 m. Readings we put beyond 60 m carry no number in the
eighth version's output and are not scored (their count is reported). --radar-delay S adds a reference scoring with the
radar read S seconds later (openpilot's radar delay; for the second car only, dongle 99c94dc769b5d96e).

Per road group (tags_main / tags_local_day / tags_night of the batch) and version: segments with an output; true distance
<= 30 m with and without the gate, own lane / next lanes, by bin; inside the reach; beyond the reach (with a number);
relative / absolute / ego speed (score-batch's pooled numbers). Paired, on the same readings (same sample, car and radar
target, true distance <= 30 m): v8 vs v7 (the whole eighth version), v8a vs v7 (S5 alone), v8 vs v8a (S6 alone); one-sided
sign test over segments with >= 10 paired readings.

  python3 tools/v8_score.py --batch DIR --seal DIR/seal.json --radar-delay 0.1 --out DIR/score_v8.json
  python3 tools/v8_score.py --batch DIR --dev-opened data/output/dash_scale/blind_v7 --out X.json    # opened data only
"""
import argparse
import contextlib
import io
import json
import sys
import tempfile
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import depth_dash_multicar as M  # noqa: E402

TRUTH = ROOT / "data/output/c2k19_truth"
STEPS = [("v6", "_cars_v6.json"), ("v7", "_cars_v7.json"), ("v8a", "_cars_v8a.json"), ("v8", "_cars_v8.json")]
PAIRS = [("v8", "v7"), ("v8a", "v7"), ("v8", "v8a")]
NO_NUMBER_M = 60.0
DELAY_CARS = {"99c94dc769b5d96e"}
GROUPS = ("main", "local_day", "night")
BINS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 60), (60, 80)]
NEAR = [(0, 10), (10, 20), (20, 30)]
MIN_PAIRED = 10


def tags_of(batch, g):
    p = batch / f"tags_{g}.txt"
    return [l.strip() for l in open(p) if l.strip() and not l.startswith("#")] if p.exists() else []


def summ(rows):
    if not rows:
        return dict(n=0)
    e = np.array([r["err"] for r in rows])
    m = np.array([abs(r["dist"] - r["truth"]) for r in rows])
    return dict(n=int(len(e)), median_abs_pct=round(float(np.median(np.abs(e))), 2), bias_pct=round(float(np.median(e)), 2),
                gt25_pct=round(float(100 * np.mean(np.abs(e) > 25)), 1), median_abs_m=round(float(np.median(m)), 2))


def load(batch, t, sfx):
    p = batch / f"{t}{sfx}"
    if not p.exists():
        return None
    c = json.loads(p.read_text())
    return None if c.get("status") == "refused" else c


def pairs_of(c, t, gate, troot=None):
    M.PARAMS["sanity_gate"] = 0.5 if gate else 1e9
    _, pr, _ = M.c2k19_score(c, (troot or TRUTH) / t, return_rows=True)
    M.PARAMS["sanity_gate"] = 0.5
    return [dict(p, seg=t) for p in pr if p["lane"] in (-1, 0, 1)]


def split_number(rows):
    """(rows with a number, count of rows whose distance is beyond NO_NUMBER_M: no number in the eighth version's output)."""
    return [p for p in rows if p["dist"] <= NO_NUMBER_M], sum(p["dist"] > NO_NUMBER_M for p in rows)


def delayed_truth(batch, tags, delay_s, fps=20.0):
    """Reference truth: the radar rows of frame f + round(delay x fps) used for frame f (second car only); the first car's
    truth is copied unchanged (openpilot: Toyota radar delay 0). Written next to the batch, never over the truth."""
    import csv
    import shutil
    out = batch / f"truth_radar_delay_{int(round(delay_s * 1000))}ms"
    k = int(round(delay_s * fps))
    for t in tags:
        d = out / t
        if (d / "radar.csv").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(TRUTH / t / "ego.csv", d / "ego.csv")
        rows = list(csv.DictReader(open(TRUTH / t / "radar.csv")))
        with open(d / "radar.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            for r in rows:
                if t.split("_")[0] in DELAY_CARS:
                    fi = int(r["frame_idx"]) - k
                    if fi < 0:
                        continue
                    r["frame_idx"] = str(fi)
                w.writerow(r)
    return out


def pooled_speeds(batch, tags, sfx, seal):
    lst = Path(tempfile.mkstemp(suffix=".txt")[1])
    lst.write_text("\n".join(tags) + "\n")
    tmp = Path(tempfile.mkstemp(suffix=".json")[1])
    with contextlib.redirect_stdout(io.StringIO()):
        M.cmd_score_batch(Namespace(list=str(lst), cars_dir=str(batch), suffix=sfx, refused_from="_run.json",
                                    truth_root=str(TRUTH), verify_sealed=seal or "", out=str(tmp), truth_offset="corrected"))
    p = json.loads(tmp.read_text())["pooled"]
    lst.unlink()
    tmp.unlink()
    return {k: {x: p[k].get(x) for x in ("n", "mae_kmh", "bias_kmh", "coverage_95", "guess0_mae_kmh") if x in p[k]}
            for k in ("relative_speed", "absolute_speed", "ego_per_bin")}


def sign_test(per_seg):
    """per_seg: [(error of the new, error of the old)] per segment; smaller is better."""
    better = sum(1 for a, b in per_seg if a < b)
    worse = sum(1 for a, b in per_seg if a > b)
    n = better + worse
    return dict(segments=len(per_seg), better=better, worse=worse, ties=len(per_seg) - n,
                p_one_sided=float(binomtest(better, n, 0.5, alternative="greater").pvalue) if n else None)


def key(p):
    return (p["seg"], round(p["t"], 3), p["id"], p["truth_id"])


def near(rows):
    """True distance within 30 m (the eighth version's main numbers)."""
    return [p for p in rows if p["truth"] <= 30]


def score_group(B, tags, seal, troot=None, with_speeds=True):
    r = dict(segments=len(tags), steps={}, paired={})
    rows, have = {}, {}
    for lab, sfx in STEPS:
        rows[lab] = dict(gate=[], nogate=[])
        have[lab] = set()
        for t in tags:
            c = load(B, t, sfx)
            if c is None:
                continue
            have[lab].add(t)
            rows[lab]["gate"] += pairs_of(c, t, True, troot)
            rows[lab]["nogate"] += pairs_of(c, t, False, troot)
        gp, n_gp = split_number(rows[lab]["gate"])
        ng, _ = split_number(rows[lab]["nogate"])
        r["steps"][lab] = dict(
            output=len(have[lab]),
            near30=dict(gate=summ(near(gp)), nogate=summ(near(ng)), own_lane=summ([p for p in near(gp) if p["lane"] == 0]),
                        next_lanes=summ([p for p in near(gp) if p["lane"] != 0]),
                        by_bin={f"{lo}-{hi} m": summ([p for p in gp if lo <= p["truth"] < hi]) for lo, hi in NEAR}),
            inside_reach=dict(own_lane=summ([p for p in gp if not p["far"] and p["lane"] == 0]),
                              next_lanes=summ([p for p in gp if not p["far"] and p["lane"] != 0])),
            beyond_reach_with_number=summ([p for p in gp if p["far"]]), readings_without_number=n_gp,
            by_true_distance_gate={f"{lo}-{hi} m": summ([p for p in gp if lo <= p["truth"] < hi]) for lo, hi in BINS},
            speeds=pooled_speeds(B, tags, sfx, seal) if with_speeds and troot is None else None)
    # new vs old on the same readings (true distance <= 30 m): per segment median |error|
    for a_, b_ in PAIRS:
        old = {key(p): p for p in near(rows[b_]["gate"])}
        both = [(p, old[key(p)]) for p in near(rows[a_]["gate"]) if key(p) in old]
        seg = defaultdict(list)
        for pn, po in both:
            seg[pn["seg"]].append((abs(pn["err"]), abs(po["err"])))
        per = [(float(np.median([x for x, _ in v])), float(np.median([y for _, y in v]))) for v in seg.values() if len(v) >= MIN_PAIRED]
        r["paired"][f"{a_}_vs_{b_}_near30"] = dict(
            readings=len(both), new=round(float(np.median([abs(a["err"]) for a, _ in both])), 2) if both else None,
            old=round(float(np.median([abs(b["err"]) for _, b in both])), 2) if both else None,
            new_m=round(float(np.median([abs(a["dist"] - a["truth"]) for a, _ in both])), 2) if both else None,
            old_m=round(float(np.median([abs(b["dist"] - b["truth"]) for _, b in both])), 2) if both else None, **sign_test(per))
    return r


def speed_dependence(B, tags, troot=None):
    """Spearman correlation of the distance error (true distance <= 30 m) with the radar's relative speed, for marking
    geometry (*_cars_c.json, which uses no depth model) and the eighth version: a time offset between the radar and the
    video shows up as this correlation (registration section 3.1, the radar-delay test)."""
    from scipy.stats import spearmanr
    out = {}
    for lab, sfx in (("marking", "_cars_c.json"), ("v8", "_cars_v8.json")):
        rows = []
        for t in tags:
            c = load(B, t, sfx)
            if c is not None:
                rows += [p for p in near(pairs_of(c, t, True, troot)) if p.get("rel_truth") is not None]
        if len(rows) >= 30:
            out[lab] = dict(n=len(rows), spearman=round(float(spearmanr([p["err"] for p in rows], [p["rel_truth"] for p in rows]).correlation), 3))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--seal", help="the batch's seal.json (blind segments)")
    ap.add_argument("--dev-opened", help="an opened batch (its seal.json lists these segments and their truth exists): no seal check")
    ap.add_argument("--radar-delay", type=float, default=None, help="also a reference scoring with the radar read this much later (s)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    B = Path(a.batch)
    M.PARAMS["radar_processing_m"] = 2.70
    alltags = sorted({t for g in GROUPS for t in tags_of(B, g)})
    if a.dev_opened:
        sealed = json.loads((Path(a.dev_opened) / "seal.json").read_text())["files"]
        for t in alltags:
            if f"{t}_dash.json" not in sealed or not (TRUTH / t / "radar.csv").exists():
                raise SystemExit(f"{t}: not a sealed-and-opened segment of {a.dev_opened} -- refusing to score")
        M.C2K19_DEV.update(alltags)
        seal = None
    else:
        if not a.seal:
            raise SystemExit("--seal is required (or --dev-opened for opened data)")
        files = [str(B / f"{t}{sfx}") for t in alltags for sfx in [x for _, x in STEPS] + ["_cars_c.json"] if (B / f"{t}{sfx}").exists()]
        M.check_sealed(a.seal, files)                     # before any truth is read
        seal = a.seal
    troot = delayed_truth(B, alltags, a.radar_delay) if a.radar_delay else None
    res = {}
    for g in GROUPS:
        tags = tags_of(B, g)
        if not tags:
            continue
        res[g] = score_group(B, tags, seal)
        res[g]["speed_dependence"] = speed_dependence(B, tags)
        if troot is not None:
            res[g]["reference_radar_delay"] = dict(delay_s=a.radar_delay, **score_group(B, tags, seal, troot, with_speeds=False),
                                                   speed_dependence=speed_dependence(B, tags, troot))
    Path(a.out).write_text(json.dumps(res, indent=1))
    for g, r in res.items():
        print(f"===== {g} ({r['segments']} segments)")
        for part in [r] + ([r["reference_radar_delay"]] if "reference_radar_delay" in r else []):
            tag = "radar delay" if part is not r else "registered truth"
            for lab, x in part["steps"].items():
                f = lambda d: f"{d.get('median_abs_pct', '-'):>5}% {d.get('median_abs_m', '-'):>5} m (bias {d.get('bias_pct', '-')}, n {d.get('n', 0)})"
                sp = x["speeds"] or {}
                print(f"  [{tag}] {lab} output {x['output']:3d} | true <= 30 m {f(x['near30']['gate'])} | "
                      + " ".join(f"{k} {v.get('median_abs_m', '-')} m" for k, v in x["near30"]["by_bin"].items())
                      + (f" | rel {sp.get('relative_speed', {}).get('mae_kmh')} ego {sp.get('ego_per_bin', {}).get('mae_kmh')}" if sp else ""))
            for k_, v_ in part["paired"].items():
                print(f"  [{tag}] {k_}: {v_}")
            print(f"  [{tag}] error vs relative speed: {part.get('speed_dependence')}")


if __name__ == "__main__":
    main()

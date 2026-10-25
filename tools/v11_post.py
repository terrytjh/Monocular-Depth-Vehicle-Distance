#!/usr/bin/env python3
"""Eleventh version (v11) = the tenth version + K4 + K2-ego + E1, as a post-processing step on the tenth version's own
per-segment output files (rules written 2026-10-21; changes found and chosen on opened development data only: Chunk_6,
Chunk_7 first and second half; Chunk_8 was only a reference, the changes were thought of after its answers).

No truth is read: only the tenth version's <tag>_cars_v10.json, the batch folder's prediction files of each segment
(<tag><suffix> = the ninth-version file: next_candidate.source and ego; <tag>_dash.json: status; <tag>_run.json: the
marking method's run), the batch's group lists (tags_main.txt, tags_local_day.txt, tags_night.txt), the camera's A
(car_a.json, as the tenth version) and the parameters file (tools/v11_params.json). No depth model or detector is run.
Refused segments stay refused.

  K4 (distance, every group): for every sample and object of the tenth-version file, on its boxes before any blanking,
     cov_front = the largest share of the box covered by another box of the same sample whose bottom row is lower by
     more than front_px (a nearer car): area(box_i & box_j) / max(area(box_i), 1 px^2), 0 when there is none;
     edge = x0 <= edge_px or x1 >= W - edge_px (W = image width; the bottom edge is not used).
     cov_front >= cov_front_min or edge, for a reading with a distance: the distance is kept as dist_v10 and dist
     becomes null ("no number": a scorer drops it after the matching, as a blank dist_m). Every other field (far, lane,
     relative and absolute speed, window, ...) is unchanged.
  New ego seconds, only for segments whose tenth-version file has no ego value at all, whose <tag>_run.json passed
  (status "ok") and whose camera has an A (tools/v10_post.camera_A: median of the camera's drive medians, leaving out
  the segment's own drive):
     K2-ego: night group, camera-k night segments (next_candidate.source "camera k night");
     E1:     local_day and night groups, own-ruler segments (source "current") whose depth dash ruler refused
             (<tag>_dash.json status not "ok") and whose ninth-version file has no ego value.
     The main group is never changed.
     Per 1 s bin b (0 <= b < floor(n_frames / fps)): the marking run's speed pairs with a value and b <= t_s < b + 1
     (pairs flagged "jump" carry no value and drop out; "unchecked" ones stay, as in the marking method's own ego bins);
     each pair's speed with the camera's A, v = kmh x A_camera / A_local (A_local of the run's ruler at frame_i; a pair
     without one drops the bin); >= min_pairs pairs: raw = median(v), sigma = hypot(1.253 sd(v) / sqrt(n), raw x s)
     (s = the run's ruler scale sigma, = tools/depth_dash_multicar.ruler_scale_sigma), hw9 = 1.96 x 1.9 x sigma.
     Then the tenth version's C2 (tools/v10_post.step_c2) with the group's C2 values on these seconds only: a straight
     line over the seconds within +-h s (local_day weighted by 1 / max(hw9, 0.5 km/h)^2, night unweighted); it uses up
     to h s after the second (after-the-fact measurement, as C2). 95 % half-width = factor (group, path) x hw9, sigma
     likewise. The segment's ego_refused text, its distances (other than K4), relative and absolute speeds are not
     changed; absolute speed is not filled for the new seconds.
  Every changed value keeps the tenth version's beside it (dist_v10) or its unsmoothed value (kmh_v11_raw) and lists
  its steps in v11_steps; the file carries a "v11" provenance block (parameters file, camera A file and group lists
  sha256, input files sha256, counts per step). The tenth-version files are checked to be the ones made from this
  batch (their own provenance: group, parameters sha256, ninth-version file sha256; v10_run.json when present).
  Deterministic.

  python3 tools/v11_post.py --v10 DIR --batch DIR --out DIR [--suffix _cars_v9.json] [--tags-dir DIR]
                            [--car-a car_a.json] [--params tools/v11_params.json] [--seal seal.json] [--v10-seal seal.json]
  (development check on Chunk_6: --suffix _cars_v7.json, as the tenth version)
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
import v10_post as V10  # noqa: E402

GROUPS = ("main", "local_day", "night")
EGO_RULES = ("K2-ego", "E1")
sha256 = V10.sha256


def read_plain(p: Path) -> bytes:
    """A file read outside the batch / tenth-version folders (lists, camera A, parameters): never a truth path."""
    if "c2k19_truth" in str(Path(p).resolve()):
        raise SystemExit(f"{p}: truth path -- refusing")
    return Path(p).read_bytes()


# ------------------------------------------------------------------ K4: covered by a nearer box, or cut by the left / right edge

def k4_sample(objects, p):
    """(cov_front, edge) for the objects of one sample, on their boxes as they are."""
    bx = np.array([o["box"][:4] for o in objects], float)
    x0, y0, x1, y1 = bx.T
    area = np.maximum((x1 - x0) * (y1 - y0), p["min_box_area_px2"])
    iw = np.clip(np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :]), 0, None)
    ih = np.clip(np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :]), 0, None)
    inter = iw * ih
    np.fill_diagonal(inter, 0)
    front = y1[None, :] > y1[:, None] + p["front_px"]          # [i, j]: box j's bottom is lower, a nearer car
    cov = np.where(front, inter / area[:, None], 0).max(axis=1) if len(objects) > 1 else np.zeros(1)
    edge = (x0 <= p["edge_px"]) | (x1 >= p["image_width_px"] - p["edge_px"])
    return cov, edge


def step_k4(out, p, info):
    """Blanks out's distances in place."""
    if p.get("bottom_edge"):
        raise SystemExit("K4: the bottom edge is not part of the rule")
    n_obj = n_blank = n_cov = n_edge = n_far = n_none = 0
    max_x1 = 0.0
    for s in out["samples"]:
        ob = s["objects"]
        if not ob:
            continue
        cov, edge = k4_sample(ob, p)
        max_x1 = max(max_x1, max(float(o["box"][2]) for o in ob))
        for i, o in enumerate(ob):
            n_obj += 1
            hit_c, hit_e = bool(cov[i] >= p["cov_front_min"]), bool(edge[i])
            if not (hit_c or hit_e):
                continue
            if o.get("dist") is None:
                n_none += 1
                continue
            o["dist_v10"] = o["dist"]
            o["dist"] = None
            o["k4"] = dict(cov_front=float(cov[i]), edge=hit_e)
            o["v11_steps"] = o.get("v11_steps", []) + ["K4"]
            n_blank += 1
            n_cov += hit_c
            n_edge += hit_e
            n_far += bool(o.get("far"))
    if max_x1 > p["image_width_px"] + 1.0:
        raise SystemExit(f"K4: a box reaches x = {max_x1:.1f} px, beyond the image width {p['image_width_px']} px "
                         "of the parameters")
    info.update(objects=n_obj, blanked=n_blank, blanked_covered=n_cov, blanked_edge=n_edge, blanked_far=n_far,
                flagged_without_distance=n_none)


# ------------------------------------------------------------------ K2-ego / E1: per-second marking speed with the camera A

def ruler_scale_sigma(run):
    """= tools/depth_dash_multicar.ruler_scale_sigma: the ruler's 95 % CI half-width / 1.96, mean over the lines used."""
    cis = [e["ci_half"] for sd, e in run.get("sides", {}).items() if sd in run.get("lines_used", []) and "ci_half" in e]
    return float(np.mean(cis)) / 1.96 if cis else 0.0


def bin_stats(v, sc, p):
    """Median, its sigma (median standard error with the ruler's scale uncertainty) and the marking method's own
    95 % half-width (hw9) for one bin's pair speeds v."""
    raw = float(np.median(v))
    sig = float(np.hypot(p["median_se_factor"] * np.std(v, ddof=1) / np.sqrt(len(v)), raw * sc))
    return dict(n=int(len(v)), raw=raw, sigma=sig, hw9=p["z"] * p["sigma_mult"] * sig)


def marking_seconds(run, A_cam, p):
    """{second: bin_stats} of the run's speed pairs read with the camera's A, the ruler scale sigma and bin counts."""
    rul = {int(x["frame"]): x.get("A_local") for x in run.get("rulers") or []}
    sc = ruler_scale_sigma(run)
    nb = int(run["n_frames"] / run["fps"] // 1.0)
    by = {}
    for x in run.get("speeds") or []:
        if x.get("kmh") is None:
            continue
        b = int(math.floor(x["t_s"]))
        if b < 0 or b >= nb:
            continue
        al = rul.get(int(x["frame_i"]))
        by.setdefault(b, []).append(x["kmh"] * A_cam / al if al else None)
    out, few, no_ruler = {}, 0, 0
    for b in sorted(by):
        v = by[b]
        if len(v) < p["min_pairs"]:
            few += 1
            continue
        if any(c is None for c in v):
            no_ruler += 1
            continue
        out[b] = bin_stats(np.array(v, float), sc, p)
    return out, sc, dict(bins_in_clip=nb, bins_with_pairs=len(by), bins_few_pairs=few, bins_without_local_ruler=no_ruler)


def c2_values(secs, raw, hw9, group, p):
    """The tenth version's step_c2 (group's C2 values) on these seconds only; returns the smoothed values in order."""
    P10 = {"C2": {group: dict(p["C2"][group], range_factor={"both": 1.0, "depth": 1.0})}}
    tmp = dict(ego=[dict(t0=float(b), kmh=r, sigma_kmh=None, ci95_kmh=h) for b, r, h in zip(secs, raw, hw9)])
    V10.step_c2(tmp, group, P10, {})
    return [float(e["kmh"]) for e in tmp["ego"]]


def step_ego(out, group, rule, run, A, pe, info):
    """Adds the new ego seconds to out in place."""
    secs, sc, cnt = marking_seconds(run, A, pe)
    info.update(camera_A=A, ruler_scale_sigma=sc, **cnt)
    if not secs:
        info.update(applied=False, why="no second with enough marking speed pairs", seconds_added=0)
        return 0
    ks = sorted(secs)
    vals = c2_values(ks, [secs[k]["raw"] for k in ks], [secs[k]["hw9"] for k in ks], group, pe)
    r = pe[rule]
    f = pe["range_factor"][group][r["path"]]
    new = {}
    for k, v in zip(ks, vals):
        q = secs[k]
        new[k] = dict(t0=float(k), n=q["n"], kmh=v, sigma_kmh=f * q["sigma"], ci95_kmh=f * q["hw9"],
                      kmh_v11_raw=q["raw"], sigma_kmh_v11_raw=q["sigma"], ci95_kmh_v11_raw=q["hw9"], camera_A=A,
                      v11_source=r["label"], v11_steps=[rule, "C2", "range"])
    kept = [e for e in out.get("ego") or [] if int(round(e["t0"])) not in new]
    if any(e.get("kmh") is not None for e in kept):
        raise SystemExit("new ego seconds on a segment that has ego values")
    out["ego"] = sorted(kept + list(new.values()), key=lambda e: float(e["t0"]))
    info.update(applied=True, seconds_added=len(new), range_factor=f, path_label=r["label"],
                negative_values=sum(1 for e in new.values() if e["kmh"] < 0))
    return len(new)


def choose_rule(group, src, pe):
    for rule in EGO_RULES:
        if src == pe[rule]["source"] and group in pe[rule]["groups"]:
            return rule
    return None


# ------------------------------------------------------------------ per segment

def process(tag, group, V, B, args, P, ctx):
    name10 = f"{tag}_cars_v10.json"
    c10 = V.load(name10)
    if c10 is None:
        return None, dict(status="absent")
    blk = c10.get("v10") or {}
    if blk.get("group") != group:
        raise SystemExit(f"{tag}: the tenth-version file was made for group {blk.get('group')!r}, the lists say {group!r}")
    if blk.get("params_sha256") != P["v10_params_sha256"]:
        raise SystemExit(f"{tag}: the tenth-version file was made with parameters {blk.get('params_sha256')}, "
                         f"not {P['v10_params_sha256']}")
    if ctx["v10_segments"] is not None and ctx["v10_segments"].get(tag) != V.read[name10]:
        raise SystemExit(f"{tag}: {name10} is not the file v10_run.json lists")
    out = copy.deepcopy(c10)
    prov = dict(version="v11", method="tenth version + K4 + K2-ego + E1 (tools/v11_post.py)", params_file=args.params,
                params_sha256=ctx["params_sha256"], car_a_sha256=ctx["car_a_sha256"], group_lists=ctx["group_lists"],
                group=group, base_file=name10)
    if c10.get("status") == "refused":
        prov.update(status="refused (kept)", inputs={name10: V.read[name10]})
        out["v11"] = prov
        return out, dict(status="refused")
    name9 = f"{tag}{args.suffix}"
    v9 = B.load(name9)
    if v9 is None or (blk.get("inputs") or {}).get(name9) != B.read.get(name9):
        raise SystemExit(f"{tag}: the tenth-version file was not made from this batch's {name9}")
    src = (v9.get("next_candidate") or {}).get("source", "current")
    if src != blk.get("path"):
        raise SystemExit(f"{tag}: source {src!r} of {name9} differs from the tenth version's path {blk.get('path')!r}")
    names = [name10, name9]
    steps = {"K4": {}}
    step_k4(out, P["K4"], steps["K4"])
    # ---- new ego seconds
    pe = P["ego"]
    eg = steps["ego"] = {}
    rule = choose_rule(group, src, pe)
    why = None
    if rule is None:
        why = "main group: ego speed never changed" if group == "main" else f"no new-ego rule for {src!r} in {group}"
    elif any(e.get("kmh") is not None for e in out.get("ego") or []):
        why = "the tenth version has ego values"
    elif rule == "E1":
        dname = f"{tag}_dash.json"
        dash = B.load(dname)
        names.append(dname)
        eg["dash_status"] = "absent" if dash is None else dash.get("status")
        if dash is not None and dash.get("status") == "ok":
            why = "the depth dash ruler passed"
        elif any(e.get("kmh") is not None for e in v9.get("ego") or []):
            why = "the ninth-version file has ego values"
    added = 0
    if rule is not None and why is None:
        rname = f"{tag}_run.json"
        run = B.load(rname)
        names.append(rname)
        A = V10.camera_A(ctx["car_a"], tag)
        eg["run_status"] = "absent" if run is None else run.get("status")
        if run is None or run.get("status") != pe["run_status"]:
            why = "the marking run did not pass"
        elif A is None:
            why = "no camera A for this camera"
        else:
            added = step_ego(out, group, rule, run, A, pe, eg)
            if not added:
                why = eg.get("why")
    eg.update(rule=rule, applied=bool(added), why=why)
    if not added:
        eg["seconds_added"] = 0
    inputs = {name10: V.read[name10]}
    inputs.update({k: B.read[k] for k in names[1:] if k in B.read})
    prov.update(path=src, steps=steps, inputs=inputs)
    out["v11"] = prov
    return out, dict(status="ok", path=src, k4_blanked=steps["K4"]["blanked"], ego_rule=rule if added else None,
                     ego_seconds_added=added)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v10", required=True, help="folder with the tenth version's <tag>_cars_v10.json (and v10_run.json)")
    ap.add_argument("--batch", required=True, help="the batch folder the tenth version was made from")
    ap.add_argument("--out", required=True, help="folder for <tag>_cars_v11.json and v11_run.json")
    ap.add_argument("--suffix", default="_cars_v9.json", help="ninth-version file (development Chunk_6: _cars_v7.json)")
    ap.add_argument("--tags-dir", default=None, help="folder of tags_{main,local_day,night}.txt (default: --batch)")
    ap.add_argument("--car-a", default=None, help="the camera A (default: <batch>/car_a.json)")
    ap.add_argument("--params", default=str(Path(__file__).resolve().parent / "v11_params.json"))
    ap.add_argument("--seal", default=None, help="check every batch file read against this seal (files by name)")
    ap.add_argument("--v10-seal", default=None, help="check every tenth-version file read against this seal")
    args = ap.parse_args()
    bdir, vdir, odir = Path(args.batch), Path(args.v10), Path(args.out)
    if odir.resolve() in (bdir.resolve(), vdir.resolve()):
        raise SystemExit("--out must be a folder of its own, not an input folder")
    tdir = Path(args.tags_dir) if args.tags_dir else bdir
    craw = read_plain(Path(args.car_a or bdir / "car_a.json"))
    praw = read_plain(Path(args.params))
    P = json.loads(praw)["values"]
    if P is None:
        raise SystemExit("the parameters file has no values")
    for rule in EGO_RULES:
        if "main" in P["ego"][rule]["groups"]:
            raise SystemExit("the main group is never changed")
    psha = sha256(praw)
    car_a = json.loads(craw)
    groups, gfiles = {}, {}
    for g in GROUPS:
        p = tdir / f"tags_{g}.txt"
        raw = read_plain(p)
        gfiles[p.name] = sha256(raw)
        for line in raw.decode().splitlines():
            t = line.strip()
            if t and not t.startswith("#"):
                if t in groups:
                    raise SystemExit(f"{t} in two groups")
                groups[t] = g
    rl = bdir / "run_list.txt"
    if rl.exists():
        listed = {"_".join(l.split("|")[:3]) for l in read_plain(rl).decode().splitlines() if l.strip()}
        if listed != set(groups):
            raise SystemExit(f"run_list.txt and the group lists differ: {sorted(listed ^ set(groups))[:5]}")
    r10p = vdir / "v10_run.json"
    v10_run, v10_segments = None, None
    if r10p.exists():
        r10raw = read_plain(r10p)
        r10 = json.loads(r10raw)
        for k, mine in (("group_lists", gfiles), ("params_sha256", P["v10_params_sha256"]), ("suffix", args.suffix),
                        ("car_a_sha256", sha256(craw))):
            if r10.get(k) != mine:
                raise SystemExit(f"{r10p}: {k} {r10.get(k)!r} differs from this run's {mine!r}")
        v10_segments = {t: r.get("sha256") for t, r in r10.get("segments", {}).items()}
        v10_run = dict(file=str(r10p), sha256=sha256(r10raw), checked=["group_lists", "params_sha256", "suffix",
                                                                       "car_a_sha256", "segment files"])
    B = V10.Batch(bdir, Path(args.seal) if args.seal else None)
    V = V10.Batch(vdir, Path(args.v10_seal) if args.v10_seal else None)
    ctx = dict(car_a=car_a, car_a_sha256=sha256(craw), group_lists=gfiles, params_sha256=psha, v10_segments=v10_segments)
    odir.mkdir(parents=True, exist_ok=True)
    summary = dict(version="v11", v10=str(vdir), batch=str(bdir), suffix=args.suffix, params_file=args.params,
                   params_sha256=psha, car_a=str(args.car_a or bdir / "car_a.json"), car_a_sha256=sha256(craw),
                   group_lists=gfiles, seal=args.seal, v10_seal=args.v10_seal, v10_run=v10_run, segments={})
    for tag in sorted(groups):
        res, row = process(tag, groups[tag], V, B, args, P, ctx)
        if res is not None:
            body = json.dumps(res, indent=1)
            (odir / f"{tag}_cars_v11.json").write_text(body)
            row["sha256"] = sha256(body.encode())
        summary["segments"][tag] = dict(group=groups[tag], **row)
    summary["inputs_read"] = dict(sorted(B.read.items()))
    summary["v10_inputs_read"] = dict(sorted(V.read.items()))
    cnt, k4, ego = defaultdict(int), defaultdict(int), defaultdict(int)
    for r in summary["segments"].values():
        cnt[(r["status"], r.get("path"), r.get("ego_rule"))] += 1
        k4[r["group"]] += r.get("k4_blanked", 0)
        if r.get("ego_rule"):
            ego[f"{r['group']}|{r['ego_rule']}"] += r["ego_seconds_added"]
    summary["counts"] = {f"{k[0]}|{k[1]}|ego {k[2]}": v for k, v in sorted(cnt.items(), key=str)}
    summary["k4_blanked_by_group"] = dict(sorted(k4.items()))
    summary["ego_seconds_added"] = dict(sorted(ego.items()))
    (odir / "v11_run.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {sum(1 for r in summary['segments'].values() if 'sha256' in r)} files to {odir}; {summary['counts']}; "
          f"K4 blanked {summary['k4_blanked_by_group']}; new ego seconds {summary['ego_seconds_added']}")


if __name__ == "__main__":
    main()

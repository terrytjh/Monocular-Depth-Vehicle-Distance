#!/usr/bin/env python3
"""Tenth version (docs/DEPTH_DASH_V10_PREREG.md): the parameter choice for tools/v10_post.py on OPENED development data
only (C6 = Chunk_6, the seventh version's sealed batch blind_v7; C7a = Chunk_7 first half, the ninth version's dry run
/mnt/d/depth_work/v9dry/run; C7b = Chunk_7 second half, the ninth version's sealed batch blind_v9). It reads those
batches' truth (per-second pose speed, per-frame radar relative speed), which is allowed for development data whose
answers were already opened; it never reads a file of a blind batch. No prediction program imports it.

Applies the rules written at 2026-10-19 04:34:42 (params_rules_only.json, sha256 3b7e31d0..., checked below) BEFORE any
three-batch objective of C2 / C4 / C5 was computed. The objectives are the candidate steps' own: the same rows
(candidates-speed/ego_rows.csv, rel_rows.csv: per-second / per-scored-reading prediction columns with the truth beside
them) and the same grids and series code as lobo_ego.py / lobo_rel.py (copied below, unchanged in substance), and for
D1 the per-batch grid table of candidates-distance/out/c1_camk_avg.json. The objective is taken over all three batches
instead of two. Writes only --out (choice.json); tools/v10_params.json was filled from it by build-post/fill_params.py.
The inputs are in the development folders below (not in git), so this copy documents the choice; it reproduces
build-post/choice.json except its computed_local time.

  python3 tools/v10_choose_params.py --out DIR/choice.json
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

DEV = Path("/mnt/d/depth_work/v10dev_20261019")
BUILD = Path("/mnt/d/depth_work/v10build_20261019/build-post")
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", required=True, help="where to write choice.json")
ap.add_argument("--rules", default=str(BUILD / "params_rules_only.json"))
ap.add_argument("--candidates-speed", default=str(DEV / "candidates-speed"))
ap.add_argument("--candidates-distance", default=str(DEV / "candidates-distance"))
ARGS = ap.parse_args()
CS, CD, RULES = Path(ARGS.candidates_speed), Path(ARGS.candidates_distance), Path(ARGS.rules)
assert hashlib.sha256(RULES.read_bytes()).hexdigest().startswith("3b7e31d0"), "rules file changed"
GROUPS = ("main", "local_day", "night")
F = lambda x: np.nan if x in ("", "None") else float(x)

# ------------------------------------------------------------------ D1 (table already computed by the candidate step)
d1 = json.loads((CD / "out/c1_camk_avg.json").read_text())
d1_rows = []
for g in d1["grid"]:
    p = g["params"]
    vals = g["main_after_pct"]
    obj = float(np.mean([vals["C6"], vals["C7a"], vals["C7b"]]))
    d1_rows.append(dict(params=p, per_batch={k: round(v, 4) for k, v in vals.items()}, objective=obj, rounded=round(obj, 2)))


def d1_simpler_key(r):     # no gate, then no smoothing, then w closer to 0.5
    p = r["params"]
    return (p["gate"] is not None, bool(p["smooth"]), abs(p["w"] - 0.5), p["w"])


best_r = min(r["rounded"] for r in d1_rows)
tied = [r for r in d1_rows if r["rounded"] == best_r]
d1_choice = min(tied, key=d1_simpler_key)

# ------------------------------------------------------------------ ego rows (lobo_ego.py arrays)
R = list(csv.DictReader(open(CS / "ego_rows.csv")))
N = len(R)
col = lambda k: np.array([F(r[k]) for r in R])
batch = np.array([r["batch"] for r in R]); group = np.array([r["group"] for r in R]); tag = np.array([r["tag"] for r in R])
t0 = np.array([int(r["t0"]) for r in R])
truth = col("truth")
v9, v9_ci = col("v9"), col("v9_ci")
mrk = col("mrk")
seg_slices = []
s0 = 0
for i in range(1, N + 1):
    if i == N or tag[i] != tag[s0] or batch[i] != batch[s0]:
        seg_slices.append((s0, i)); s0 = i
assert all(np.all(np.diff(t0[a:b]) == 1) for a, b in seg_slices)
HS = (1, 2, 3, 4, 5, 6, 8, 10)


def c2(base_v, base_ci, h, weighted):        # = lobo_ego.c2
    out = np.full(N, np.nan)
    for a, b in seg_slices:
        vv, cc = base_v[a:b], base_ci[a:b]
        ok = ~np.isnan(vv)
        if not ok.any():
            continue
        idx = np.where(ok)[0]
        for i in idx:
            nb = idx[np.abs(idx - i) <= h]
            u = (nb - i).astype(float)
            y = vv[nb]
            wt = 1.0 / np.maximum(cc[nb], 0.5) ** 2 if weighted else np.ones(len(nb))
            if len(nb) >= 3:
                W = np.sqrt(wt)
                X = np.vstack([np.ones_like(u), u]).T * W[:, None]
                out[a + i] = float(np.linalg.lstsq(X, y * W, rcond=None)[0][0])
            else:
                out[a + i] = float(np.sum(wt * y) / np.sum(wt))
    return out


def mae_e(v, sel):
    m = sel & ~np.isnan(v)
    return float(np.mean(np.abs(v[m] - truth[m]))) if m.any() else np.inf


ALL3 = np.isin(batch, ["Chunk_6", "Chunk_7a", "Chunk_7b"])
assert ALL3.all()
src = np.full(N, "", dtype=object)
has9 = ~np.isnan(v9)
src[has9 & ~np.isnan(mrk)] = "both"
src[has9 & np.isnan(mrk)] = "depth"
C2cache = {(h, w): c2(v9, v9_ci, h, w) for h in HS for w in (False, True)}
c2_choice, c2_grid = {}, {}
for g in GROUPS:
    sel = group == g
    tab = {(h, w): mae_e(C2cache[(h, w)], sel) for h in HS for w in (False, True)}
    c2_grid[g] = {f"h{h}{'w' if w else ''}": round(v, 4) for (h, w), v in tab.items()}
    best = min(round(v, 3) for v in tab.values())
    tied = [k for k, v in tab.items() if round(v, 3) == best]
    h, w = min(tied, key=lambda k: (k[1], k[0]))          # unweighted first, then smaller h
    sv = C2cache[(h, w)]
    m = sel & ~np.isnan(sv) & (v9_ci > 0)
    pooled = float(np.percentile(np.abs(sv[m] - truth[m]) / v9_ci[m], 95))
    fac, nsec = {}, {}
    for s in ("both", "depth"):
        ms = m & (src == s)
        nsec[s] = int(ms.sum())
        fac[s] = float(np.percentile(np.abs(sv[ms] - truth[ms]) / v9_ci[ms], 95)) if ms.sum() >= 30 else pooled
    per_batch = {b: dict(v9=round(mae_e(v9, sel & (batch == b) & has9), 4), v10=round(mae_e(sv, sel & (batch == b)), 4),
                         seconds=int((sel & (batch == b) & has9).sum())) for b in ("Chunk_6", "Chunk_7a", "Chunk_7b")}
    c2_choice[g] = dict(h=h, weighted=w, objective=round(tab[(h, w)], 4), v9_objective=round(mae_e(v9, sel & has9), 4),
                        range_factor={k: round(v, 4) for k, v in fac.items()}, range_factor_pooled=round(pooled, 4),
                        range_seconds=nsec, per_batch_in_sample=per_batch)

# ------------------------------------------------------------------ relative-speed rows (lobo_rel.py arrays)
RR = list(csv.DictReader(open(CS / "rel_rows.csv")))
NR = len(RR)
colr = lambda k: np.array([F(r[k]) for r in RR])
rbatch = np.array([r["batch"] for r in RR]); rgroup = np.array([r["group"] for r in RR])
path = np.array([r["path"] for r in RR]); averaged = np.array([r["averaged"] == "True" for r in RR])
truth_v = colr("rel_truth_inst"); rel, rel_ci = colr("rel"), colr("rel_ci")
sc_d, sc_m = colr("sc_d"), colr("sc_m")
RHWS = (2, 3, 4, 5, 6, 8, 10, 12, 16)
WSETS = {"4": (4,), "6-4": (6, 4), "8-4": (8, 6, 4), "10-4": (10, 8, 6, 4), "12-4": (12, 10, 8, 6, 4), "16-4": (16, 12, 10, 8, 6, 4)}
DS = (None, 28, 30, 32, 35, 40)
M_REL = 3.4
D_ = {hw: (colr(f"d{hw}"), colr(f"d{hw}_se")) for hw in RHWS}
M_ = {hw: (colr(f"m{hw}"), colr(f"m{hw}_se")) for hw in RHWS}
MX = {(hw, d): (colr(f"m{hw}_x{d}"), colr(f"m{hw}_x{d}_se")) for hw in RHWS for d in DS if d is not None}
has = ~np.isnan(rel)


def longest(table, W, pick=None):           # = lobo_rel.longest
    v, se = np.full(NR, np.nan), np.full(NR, np.nan)
    for hw in sorted(W):
        tv, ts = table[hw] if pick is None else table[(hw, pick)]
        ok = ~np.isnan(tv)
        v[ok], se[ok] = tv[ok], ts[ok]
    return v, se


def combine(vd, sd_se, vm, sm_se, use_m):   # = lobo_rel.combine with a = 0
    sd = np.hypot(sd_se, np.abs(vd) * sc_d)
    sm = np.hypot(sm_se, np.abs(vm) * sc_m)
    both = use_m & ~np.isnan(vm) & ~np.isnan(vd)
    v = np.where(both, 0.5 * vd + 0.5 * vm, vd)
    sig = np.where(both, np.sqrt(0.25 * sd ** 2 + 0.25 * sm ** 2), sd)
    v[~has] = np.nan
    return v, 1.96 * M_REL * sig


def cand(W, D=None, ext_sel=None):
    """C4 window set W; C5 extension to D m on the readings of ext_sel (lobo_rel.cand restricted to ext_sel)."""
    Wt = WSETS[W]
    vd, sd = longest(D_, Wt)
    vm, sm = longest(M_, Wt)
    use_m = averaged.copy()
    if D is not None:
        vx, sx = longest(MX, Wt, pick=D)
        ext = has & ~averaged & (path == "current") & ~np.isnan(vx) & ext_sel
        vm = np.where(ext, vx, vm); sm = np.where(ext, sx, sm)
        use_m = use_m | ext
    return combine(vd, sd, vm, sm, use_m)


def mae_r(v, sel):
    m = sel & has & ~np.isnan(v)
    return float(np.mean(np.abs(v[m] - truth_v[m])))


allr = np.ones(NR, bool)
mainr = rgroup == "main"
c4_tab = {W: mae_r(cand(W)[0], allr) for W in WSETS}
best = min(round(v, 3) for v in c4_tab.values())
order = list(WSETS)                                   # shorter set first
W_ch = [W for W in order if round(c4_tab[W], 3) == best][0]
c5_tab = {str(D): mae_r(cand(W_ch, D, mainr)[0], mainr) for D in DS}
best5 = min(round(v, 3) for v in c5_tab.values())
D_ch = [D for D in DS if round(c5_tab[str(D)], 3) == best5][0]   # none first, then smaller D
v10r, ci10r = cand(W_ch, D_ch, mainr)
m = has & ~np.isnan(v10r) & (ci10r > 0)
rel_factor = float(np.percentile(np.abs(v10r[m] - truth_v[m]) / ci10r[m], 95))
rel_in_sample = {}
for b in ("Chunk_6", "Chunk_7a", "Chunk_7b"):
    for g in GROUPS:
        sel = (rbatch == b) & (rgroup == g)
        rel_in_sample[f"{b}|{g}"] = dict(n=int((sel & has).sum()), v9=round(mae_r(rel, sel), 4), v10=round(mae_r(v10r, sel), 4),
                                         v10_cover95=round(100 * float(np.mean((np.abs(v10r - truth_v) <= rel_factor * ci10r)[sel & m])), 1))
out = dict(
    computed_local=__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
    D1=dict(grid=[dict(params=r["params"], per_batch_main_after_pct=r["per_batch"], objective_mean_pct=round(r["objective"], 4))
                  for r in d1_rows], chosen=d1_choice["params"], chosen_objective=round(d1_choice["objective"], 4),
            tied_at_rounding=[r["params"] for r in d1_rows if r["rounded"] == best_r]),
    C2=dict(grid_pooled_mae=c2_grid, chosen=c2_choice),
    C4=dict(grid_pooled_mae={k: round(v, 4) for k, v in c4_tab.items()}, chosen=W_ch, chosen_halfwidths_samples=list(WSETS[W_ch])),
    C5=dict(grid_pooled_mae_main={k: round(v, 4) for k, v in c5_tab.items()}, chosen_D=D_ch),
    rel_range=dict(factor=round(rel_factor, 4), readings=int(m.sum())),
    rel_in_sample_check=rel_in_sample)
Path(ARGS.out).parent.mkdir(parents=True, exist_ok=True)
Path(ARGS.out).write_text(json.dumps(out, indent=1))
print(json.dumps({k: (v if k not in ("D1", "C2") else {kk: vv for kk, vv in v.items() if kk not in ("grid", "grid_pooled_mae")})
                  for k, v in out.items()}, indent=1))

#!/usr/bin/env python3
# Copy of /mnt/d/depth_work/v11extra_20261025/score/f03_report.py for scoring comma2k19 Chunk_9 (docs/DEPTH_DASH_V11_PREREG.md); changed: the batch C9 in the list.
"""Step 3 of scoring (READS TRUTH via the step-1 tables; development batches C6 / C7a / C7b, C8 reference): the tables of
the final eleventh version against the tenth version, in Terry's format (decision 8):
  per truth-distance bin (<= 30, 30-60, 60-100, > 100 m): radar targets in scope, matched boxes, readings with a distance;
  |error| mean +- SD (m and %), median, p90, Tukey-outlier share (|e| > Q3 + 1.5 IQR of |e| %); coverage as matched box /
  targets and with a distance / targets; signed bias only in an internal column.
Also: per tier class, added / removed readings, segment and trip sign tests, ego / relative / absolute speed tables with
same-reading comparisons, and every 95 % range's coverage (leave-one-batch for the values learnt here).

  python3 -B f03_report.py --scores DIR --ranges ranges.json --out DIR [--versions v10 v11f ...]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import f_lib as F  # noqa: E402

S = F.S
GROUPS = ("main", "local_day", "night")
BINS = S.BIN_NAMES
BATCHES = ("C6", "C7a", "C7b", "C8", "C9")   # a batch without step-1 tables in --scores is skipped


def binom2(good, bad):
    n = good + bad
    if n == 0:
        return None
    k = min(good, bad)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return float(min(1.0, 2 * p))


def dstats(m, pct):
    m, pct = np.asarray(m, float), np.asarray(pct, float)
    if not len(m):
        return dict(n=0)
    am, ap_ = np.abs(m), np.abs(pct)
    q1, q3 = np.percentile(ap_, [25, 75])
    return dict(n=int(len(m)), mean_m=float(am.mean()), sd_m=float(am.std(ddof=1)) if len(m) > 1 else 0.0,
                mean_pct=float(ap_.mean()), sd_pct=float(ap_.std(ddof=1)) if len(m) > 1 else 0.0,
                med_m=float(np.median(am)), med_pct=float(np.median(ap_)), p90_m=float(np.percentile(am, 90)),
                p90_pct=float(np.percentile(ap_, 90)), tukey_pct=float(100 * np.mean(ap_ > q3 + 1.5 * (q3 - q1))),
                bias_pct=float(pct.mean()), gt25_pct=float(100 * np.mean(ap_ > 25)))


def sstats(e):
    e = np.asarray(e, float)
    if not len(e):
        return dict(n=0)
    a = np.abs(e)
    return dict(n=int(len(e)), mae=float(a.mean()), sd=float(a.std(ddof=1)) if len(e) > 1 else 0.0, med=float(np.median(a)),
                p90=float(np.percentile(a, 90)), bias=float(e.mean()))


def load(scores, b):
    d = Path(scores) / b
    return (pd.read_csv(d / "pairs.csv.gz", low_memory=False), pd.read_csv(d / "ego.csv.gz", low_memory=False),
            pd.read_csv(d / "truth_counts.csv.gz"), pd.read_csv(d / "segs.csv.gz"))


def fnum(x, f="{:.2f}"):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f.format(x)


def sign_test(P, va, vb, g, unit):
    """segments (unit seg) or trips (unit drive) where vb's median |error| % (truth <= 30 m, each version's own readings,
    >= 5 each) is below / above va's."""
    out = dict(good=0, bad=0)
    A = P[(P.version == va) & (P.group == g) & P.dist.notna() & (P.truth <= 30)]
    B = P[(P.version == vb) & (P.group == g) & P.dist.notna() & (P.truth <= 30)]
    ma = A.assign(e=A.err_pct.abs()).groupby(unit).e.agg(["median", "size"])
    mb = B.assign(e=B.err_pct.abs()).groupby(unit).e.agg(["median", "size"])
    j = ma.join(mb, lsuffix="_a", rsuffix="_b", how="inner")
    j = j[(j.size_a >= 5) & (j.size_b >= 5)]
    out["good"] = int((j.median_b < j.median_a - 1e-12).sum())
    out["bad"] = int((j.median_b > j.median_a + 1e-12).sum())
    out["p2"] = binom2(out["good"], out["bad"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--ranges", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--versions", nargs="+", default=["v10", "v11f"])
    a = ap.parse_args()
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    RJ = json.loads(Path(a.ranges).read_text())
    va, vb = a.versions[0], a.versions[1]
    drows, srows, erows, rrows, arows, crows = [], [], [], [], [], []
    txt = []
    for b in BATCHES:
        if not (Path(a.scores) / b / "pairs.csv.gz").exists():
            continue
        P, E, T, G = load(a.scores, b)
        for g in GROUPS:
            Tg = T[T.group == g]
            tg = {k: int(Tg[k].sum()) for k in BINS}
            for v in a.versions:
                Pv = P[(P.version == v) & (P.group == g)]
                classes = [("all", Pv)]
                if v != "v10":
                    for c in ("t1", "t1n", "t2k", "t2f", "t4d"):
                        classes.append((c, Pv[Pv.tier_class == c]))
                    classes.append(("t1_every_frame_new", Pv[Pv.t2new == 1]))
                    classes.append(("far60_only", Pv[Pv.far60 == 1]))
                for cname, X in classes:
                    for tb in BINS:
                        Y = X[X.tbin == tb]
                        W = Y[Y.dist.notna()]
                        st = dstats(W.err_m, W.err_pct)
                        drows.append(dict(batch=b, group=g, version=v, cls=cname, tbin=tb, targets=tg[tb], matched=int(len(Y)),
                                          with_dist=int(len(W)), cov_matched=100 * len(Y) / tg[tb] if tg[tb] else None,
                                          cov_dist=100 * len(W) / tg[tb] if tg[tb] else None, **st))
            # added / removed readings (v11f vs v10, same match keys)
            key = ["seg", "fi", "id", "truth_id"]
            A_ = P[(P.version == va) & (P.group == g)].set_index(key)
            B_ = P[(P.version == vb) & (P.group == g)].set_index(key)
            ja = A_[A_.dist.notna()]
            jb = B_[B_.dist.notna()]
            both = ja.index.intersection(jb.index)
            same_flag = pd.Series(np.abs(ja.loc[both, "dist"].values - jb.loc[both, "dist"].values) <= 1e-9, index=both)
            add = jb.loc[jb.index.difference(ja.index)]
            rem = ja.loc[ja.index.difference(jb.index)]
            for nm, X in (("same_readings", jb.loc[both]), ("added_in_v11f", add), ("removed_in_v11f", rem)):
                for tb in BINS + ["all"]:
                    Y = X if tb == "all" else X[X.tbin == tb]
                    st = dstats(Y.err_m, Y.err_pct)
                    sv = int(same_flag.loc[Y.index].sum()) if nm == "same_readings" else None
                    srows.append(dict(batch=b, group=g, kind=nm, tbin=tb, same_value=sv, **st))
            for unit in ("seg", "drive"):
                stt = sign_test(P, va, vb, g, unit)
                srows.append(dict(batch=b, group=g, kind=f"sign_{unit}", tbin="le30", n=stt["good"] + stt["bad"],
                                  good=stt["good"], bad=stt["bad"], p2=stt["p2"]))
            # ---- ego
            secs_truth = int(G[(G.version == va) & (G.group == g)].truth_secs.sum())
            Ea = E[(E.version == va) & (E.group == g) & E.truth.notna()]
            Eb = E[(E.version == vb) & (E.group == g) & E.truth.notna()]
            ka, kb = Ea.set_index(["seg", "sec"]), Eb.set_index(["seg", "sec"])
            new = Eb[~Eb.set_index(["seg", "sec"]).index.isin(ka.index)]
            same = kb.index.intersection(ka.index)
            ea = Ea.kmh.clip(lower=0) - Ea.truth
            eb = Eb.kmh.clip(lower=0) - Eb.truth
            en = new.kmh.clip(lower=0) - new.truth
            row = dict(batch=b, group=g, truth_secs=secs_truth, v10_secs=int(len(Ea)), v11_secs=int(len(Eb)),
                       v10_cov=100 * len(Ea) / secs_truth if secs_truth else None, v11_cov=100 * len(Eb) / secs_truth if secs_truth else None,
                       v10_mae=sstats(ea).get("mae"), v11_mae=sstats(eb).get("mae"),
                       same_secs=int(len(same)),
                       same_identical=int((np.abs(ka.loc[same, "kmh"].values - kb.loc[same, "kmh"].values) <= 1e-9).sum()) if len(same) else 0,
                       new_secs=int(len(new)), new_mae=sstats(en).get("mae"), new_p90=sstats(en).get("p90"), new_bias=sstats(en).get("bias"),
                       v10_cover95=100 * float((ea.abs() <= Ea.ci95).mean()) if len(Ea) else None,
                       v11_cover95=100 * float((eb.abs() <= Eb.ci95).mean()) if len(Eb) else None,
                       new_cover95=100 * float((en.abs() <= new.ci95).mean()) if len(new) else None)
            hard = new.acc.abs() >= S.HARD
            row.update(new_hard_n=int(hard.sum()), new_hard_mae=sstats(en[hard]).get("mae"))
            erows.append(row)
            for src, X in new.assign(src=np.where(new.rule.fillna("").astype(str) != "", new.rule.fillna("").astype(str),
                                                  "t2f-fill (camera-k C2)")).groupby("src"):
                e_ = X.kmh.clip(lower=0) - X.truth
                erows.append(dict(batch=b, group=g, source=src, new_secs=int(len(X)), segs=int(X.seg.nunique()),
                                  new_mae=sstats(e_).get("mae"), new_p90=sstats(e_).get("p90"), new_bias=sstats(e_).get("bias"),
                                  new_cover95=100 * float((e_.abs() <= X.ci95).mean()), hw_median=float(X.ci95.median())))
            # ---- relative / absolute speed (truth <= 30 m, readings matched to a radar target)
            for kind, col, tcol, ccol in (("rel", "rel", "rel_truth_inst", "rel_ci95"), ("abs", "abs", "abs_truth", "abs_ci95")):
                Ra = P[(P.version == va) & (P.group == g) & (P.truth <= 30) & P[col].notna()]
                Rb = P[(P.version == vb) & (P.group == g) & (P.truth <= 30) & P[col].notna()]
                ra, rb = Ra.set_index(key), Rb.set_index(key)
                bo = rb.index.intersection(ra.index)
                da = ra.loc[bo, col] - ra.loc[bo, tcol]
                db = rb.loc[bo, col] - rb.loc[bo, tcol]
                chg = np.abs(ra.loc[bo, col].values - rb.loc[bo, col].values) > 1e-9
                nb_ = rb.loc[rb.index.difference(ra.index)]
                dn = nb_[col] - nb_[tcol]
                seg_gb = pd.DataFrame(dict(seg=[k[0] for k in bo], ea=np.abs(da.values), eb=np.abs(db.values),
                                           drive=[S.L.drive_of(k[0]) for k in bo]))
                sg = {}
                for unit in ("seg", "drive"):
                    m = seg_gb.groupby(unit)[["ea", "eb"]].mean()
                    good, bad = int((m.eb < m.ea - 1e-12).sum()), int((m.eb > m.ea + 1e-12).sum())
                    sg[unit] = (good, bad, binom2(good, bad))
                r = dict(batch=b, group=g, kind=kind, v10_n=int(len(Ra)), v10_mae=sstats(Ra[col] - Ra[tcol]).get("mae"),
                         v11_n=int(len(Rb)), v11_mae=sstats(Rb[col] - Rb[tcol]).get("mae"),
                         same_n=int(len(bo)), same_changed=int(chg.sum()), same_v10_mae=sstats(da).get("mae"),
                         same_v11_mae=sstats(db).get("mae"),
                         changed_v10_mae=sstats(da[chg]).get("mae"), changed_v11_mae=sstats(db[chg]).get("mae"),
                         seg_good=sg["seg"][0], seg_bad=sg["seg"][1], seg_p2=sg["seg"][2], trip_good=sg["drive"][0],
                         trip_bad=sg["drive"][1], trip_p2=sg["drive"][2], added_n=int(len(nb_)), added_mae=sstats(dn).get("mae"),
                         v10_cover95=100 * float(((Ra[col] - Ra[tcol]).abs() <= Ra[ccol]).mean()) if len(Ra) else None,
                         v11_cover95=100 * float(((Rb[col] - Rb[tcol]).abs() <= Rb[ccol]).mean()) if len(Rb) else None,
                         added_cover95=100 * float((dn.abs() <= nb_[ccol]).mean()) if len(nb_) else None)
                if kind == "abs":
                    ne = Rb[Rb.abs_new_ego == 1]
                    r.update(new_ego_n=int(len(ne)), new_ego_mae=sstats(ne[col] - ne[tcol]).get("mae"),
                             new_ego_cover95=100 * float(((ne[col] - ne[tcol]).abs() <= ne[ccol]).mean()) if len(ne) else None)
                else:
                    t2 = Rb[Rb.t2rel == 1]
                    r.update(t2_n=int(len(t2)), t2_mae=sstats(t2[col] - t2[tcol]).get("mae"))
                (rrows if kind == "rel" else arows).append(r)
        S.L._TRUTH.clear()
    D = pd.DataFrame(drows)
    D.to_csv(od / "dist_table.csv", index=False, float_format="%.4f")
    pd.DataFrame(srows).to_csv(od / "dist_compare.csv", index=False, float_format="%.4f")
    pd.DataFrame(erows).to_csv(od / "ego_table.csv", index=False, float_format="%.4f")
    pd.DataFrame(rrows).to_csv(od / "rel_table.csv", index=False, float_format="%.4f")
    pd.DataFrame(arows).to_csv(od / "abs_table.csv", index=False, float_format="%.4f")
    # ---- coverage of the learnt ranges (leave-one-batch) and the fixed ones
    for hold, cells in RJ["lobo"].items():
        for k, v in cells.items():
            crows.append(dict(kind="distance", held_out=hold, cell=k, **v))
    for k, v in RJ.get("c8_reference", {}).items():
        crows.append(dict(kind="distance", held_out="C8 (three-batch values)", cell=k, **v))
    for hold, c in RJ["mh_detail"]["lobo"].items():
        crows.append(dict(kind="ego MH", held_out=hold, cell=f"q {RJ['mh_detail']['q']}", cover=c, hw=RJ["mh_hw_fixed_kmh"]))
    crows.append(dict(kind="ego MH", held_out="C8 (three-batch value)", cell="main", n=RJ["mh_c8"]["n"], cover=RJ["mh_c8"]["cover"],
                      hw=RJ["mh_hw_fixed_kmh"]))
    v = RJ["local_conservative_detail"]
    for hold, c in v["lobo"].items():
        crows.append(dict(kind="ego local_day E1 conservative", held_out=hold, cell=f"q {v['q']}", cover=c, factor=v["factor"]))
    crows.append(dict(kind="ego local_day E1 conservative", held_out="C8 (three-batch value)", cell=f"q {v['q']}", cover=v["c8"],
                      factor=v["factor"]))
    for path, v in RJ["night_conservative_detail"].items():
        for hold, c in v["lobo"].items():
            crows.append(dict(kind=f"ego night {path} conservative", held_out=hold, cell=f"q {v['q']}", cover=c, factor=v["factor"]))
        crows.append(dict(kind=f"ego night {path} conservative", held_out="C8 (three-batch value)", cell=f"q {v['q']}",
                          cover=v["c8"], factor=v["factor"]))
    pd.DataFrame(crows).to_csv(od / "coverage.csv", index=False)
    # ---- readable text
    for b in BATCHES:
        Db = D[D.batch == b]
        if not len(Db):
            continue
        for g in GROUPS:
            txt.append(f"\n== {b} {g}: distance per truth bin (v10 -> v11f all classes) ==")
            txt.append("tbin | targets | matched v10 -> v11f | with dist v10 -> v11f | cov dist % | mean±SD m | mean±SD % | med m / % | p90 m / % | Tukey % | bias % (internal)")
            for tb in BINS:
                x = Db[(Db.group == g) & (Db.cls == "all") & (Db.tbin == tb)].set_index("version")
                if va not in x.index or vb not in x.index:
                    continue
                p, q = x.loc[va], x.loc[vb]

                def ms(r, k1, k2):
                    return "–" if not r["with_dist"] else f"{r[k1]:.2f}±{r[k2]:.2f}"

                txt.append(f"{tb} | {int(q.targets)} | {int(p.matched)} -> {int(q.matched)} | {int(p.with_dist)} -> {int(q.with_dist)} | "
                           f"{fnum(p.cov_dist, '{:.1f}')} -> {fnum(q.cov_dist, '{:.1f}')} | {ms(p, 'mean_m', 'sd_m')} -> {ms(q, 'mean_m', 'sd_m')} | "
                           f"{ms(p, 'mean_pct', 'sd_pct')} -> {ms(q, 'mean_pct', 'sd_pct')} | "
                           f"{fnum(p.med_m)}/{fnum(p.med_pct)} -> {fnum(q.med_m)}/{fnum(q.med_pct)} | "
                           f"{fnum(p.p90_m)}/{fnum(p.p90_pct)} -> {fnum(q.p90_m)}/{fnum(q.p90_pct)} | "
                           f"{fnum(p.tukey_pct, '{:.1f}')} -> {fnum(q.tukey_pct, '{:.1f}')} | {fnum(p.bias_pct)} -> {fnum(q.bias_pct)}")
    (od / "tables.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt[:60]))


if __name__ == "__main__":
    main()

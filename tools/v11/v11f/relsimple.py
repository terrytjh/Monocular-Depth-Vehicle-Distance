#!/usr/bin/env python3
"""Eleventh version, relative speed "simple" (decision 5 of 2026-10-24): the car depth re-read 25 % of the box height above
its bottom (instead of the registered 6 %), registered calibration (the sample's stored ground fit D = a z + b), MoGe-2's
series where the segment has a usable MoGe-2 measurement (else DA3), then the tenth version's relative-speed rules
(C4 longest complete window, C5 for the main group; readings that carry a ninth-version relative speed only) and the mean
with marking geometry's slope at weight 0.5 where marking has a window. Distances are NOT changed by this.

A port, for new batches, of track DA's 07_carread.py (read points reg / up25 only) + 21_relvar.py (columns
r_reg_reg_da3_0.5 = identity and r_up25_reg_mg_0.5 = the variant), with pipe.Meas / RawSeries / rel_speeds /
marking_maps copied unchanged in substance (/mnt/d/depth_work/daytime_20261021/DA/scripts2/pipe.py, scripts3/13_variants.py,
scripts4/21_relvar.py, 2026-10-22). Reads no truth (any path with c2k19_truth is refused), runs no model, uses no GPU:
only the batch's prediction files and the cached depth maps.

Per segment (main group, ninth-version source "current", tenth-version file not refused) it writes <out>/<tag>.csv.gz with
fi, id, r_reg_reg_da3_0.5 (must reproduce the tenth version's relative speed: identity), r_up25_reg_mg_0.5 (the variant),
mg_used (1 if MoGe-2's series gave the depth slope), and <out>/relsimple_run.json (identity counts, inputs sha256).

  python -B relsimple.py --batch-def batches/C7b.json --out DIR [--tags ...]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

for _m in ("torch", "ultralytics", "depth_backends", "moge_backend"):
    sys.modules[_m] = None
sys.dont_write_bytecode = True
REPO = Path("/home/terry/Monocular-Depth-Vehicle-Distance")
sys.path.insert(0, str(REPO / "tools"))

import os as _os_nv11
if _os_nv11.environ.get("NV11_TOOLS"):
    sys.path.insert(0, _os_nv11.environ["NV11_TOOLS"])   # night_v11: arm override (patched depth_dash_multicar.py only)
import cv2  # noqa: E402
import depth_dash_multicar as M  # noqa: E402
import depth_dash_scale as D  # noqa: E402
import v10_post as V10  # noqa: E402

REACH = 30.0
HWS = (12, 10, 8, 6, 4)
MAX_JUMP = 0.15
C5_D = 40.0
GROUPS = ("main", "local_day", "night")


def _rb_retry(p):                                   # /mnt/d returns ENOMEM / short reads under load; a re-read returns the same bytes
    for _k in range(5):
        try:
            return Path(p).read_bytes()
        except OSError:
            if _k == 4:
                raise
            __import__("time").sleep(2.0)


def refuse(p):
    if "c2k19_truth" in str(Path(p).resolve()):
        raise SystemExit(f"{p}: truth path -- refusing")


def sha256b(b):
    return hashlib.sha256(b).hexdigest()


class Files:
    """The batch folder's files, read once, sha256 recorded; checked against the seal when one is given."""

    def __init__(self, folder, seal):
        self.folder, self.read = Path(folder), {}
        self.seal = (json.loads(Path(seal).read_text()).get("files") or {}) if seal else None

    def load(self, name):
        p = self.folder / name
        refuse(p)
        if not p.exists():
            return None
        raw = _rb_retry(p)
        h = sha256b(raw)
        if self.seal is not None and self.seal.get(name) not in (None, h):
            raise SystemExit(f"{p}: changed since the seal -- refusing")
        self.read[name] = h
        return json.loads(raw)


# ------------------------------------------------------------------ = pipe.py (track DA), unchanged in substance

class Meas:
    def __init__(self, cars, new_raw=None, reach=REACH):
        self.times = [s["t"] for s in cars["samples"]]
        self.fis = [s["fi"] for s in cars["samples"]]
        self.raw = {}
        for j, s in enumerate(cars["samples"]):
            for o in s["objects"]:
                if o.get("dist_raw") is None and o.get("dist") is None:
                    continue
                r = o.get("dist_raw", o.get("dist"))
                if new_raw is not None:
                    r = new_raw.get((s["fi"], o["id"]), None)
                    if r is None or not np.isfinite(r) or r <= 0:
                        continue
                self.raw[(j, o["id"])] = float(r)
        self.far = {k: bool(v > reach) for k, v in self.raw.items()}


class RawSeries(V10.Series):
    def __init__(self, m: Meas, max_jump=MAX_JUMP):
        self.t, self.fi, self.max_jump = m.times, m.fis, max_jump
        self.idx, self.by_id = {}, {}
        for (j, i), d in m.raw.items():
            self.by_id.setdefault(i, {})[j] = (d, m.far[(j, i)])
            self.idx[(m.fis[j], i)] = j


def marking_maps(mrk):
    md, mr = {}, {}
    if mrk is None or mrk.get("status") == "refused":
        return md, mr
    for s in mrk["samples"]:
        for o in s["objects"]:
            if o.get("dist") is not None:
                md[(s["fi"], o["id"])] = o["dist"]
            if o.get("rel_kmh") is not None:
                mr[(s["fi"], o["id"])] = o["rel_kmh"]
    return md, mr


def rel_speeds(Sd, Sm, mk_rel, keys, group, v9_has):
    """= pipe.rel_speeds: {(fi, id): (depth slope, marking slope or None)} with the tenth version's C4 / C5 rules."""
    out = {}
    for fi, tid in keys:
        f4 = Sd.longest(fi, tid, (4,))
        if (fi, tid) not in v9_has or f4 is None:
            continue
        averaged = (fi, tid) in mk_rel and Sm is not None
        fd = Sd.longest(fi, tid, HWS)
        fm = Sm.longest(fi, tid, HWS) if averaged else None
        if not averaged and group == "main" and Sm is not None:
            fx = Sm.longest(fi, tid, HWS, dmax=C5_D)
            if fx is not None:
                fm = fx
        out[(fi, tid)] = (fd[0], None if fm is None else fm[0])
    return out


# ------------------------------------------------------------------ car depth re-read (= 07_carread.py, reg and up25)

def load_retry(cache_dir, frame, tries=4):
    for k in range(tries):
        try:
            return D.load_depth(cache_dir, frame)
        except (ValueError, OSError):
            if k == tries - 1:
                raise
            time.sleep(2.0)


def patch(depth, yy, xx, r=2):
    v = depth[max(0, yy - r): yy + r + 1, max(0, xx - r): xx + r + 1]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else np.nan


def reread(cars, frames, cache_dir, h, w):
    """{(fi, id): (z_reg, z_up25, a, b, dist_raw)} for every reading with dist_raw in a sample with an accepted fit;
    worst |a z_reg + b - dist_raw|."""
    out, worst = {}, 0.0
    r_ = M.PARAMS["patch"] // 2
    for s in cars["samples"]:
        f = s.get("ground_fit")
        ob = [o for o in s["objects"] if o.get("dist_raw") is not None]
        if not ob or not (f and f.get("ok")):
            continue
        refuse(cache_dir)
        depth = load_retry(cache_dir, frames[s["fi"]])
        dh, dw = depth.shape
        for o in ob:
            x0, y0, x1, y1 = o["box"][:4]
            bh = y1 - y0
            xx = int((x0 + x1) / 2 * dw / w)
            yy = int((y1 - M.PARAMS["read_up"] * bh) * dh / h)
            z_reg = float(np.median(depth[max(0, yy - r_): yy + r_ + 1, max(0, xx - r_): xx + r_ + 1]))
            z_up = patch(depth, int((y1 - 0.25 * bh) * dh / h), xx, r_)
            fo = f                                 # v11extra (a) GL: the fit this reading was made with
            if o.get("gfit") in ("left", "right") and (s.get(f"ground_fit_{o['gfit']}") or {}).get("ok"):
                fo = s[f"ground_fit_{o['gfit']}"]
            worst = max(worst, abs(fo["a"] * z_reg + fo["b"] - o["dist_raw"]))
            out[(s["fi"], o["id"])] = (z_reg, z_up, fo["a"], fo["b"], float(o["dist_raw"]))
    return out, worst


def all_raw(cars):
    """{(fi, id): stored dist_raw} of every reading (21_relvar: a reading without a re-read keeps its registered value)."""
    return {(s["fi"], o["id"]): float(o["dist_raw"]) for s in cars["samples"] for o in s["objects"] if o.get("dist_raw") is not None}


def up25_raws(rr, cars):
    raws, fb = all_raw(cars), 0
    for k, (zr, zu, a, b, r0) in rr.items():
        v = a * zu + b if np.isfinite(zu) else np.nan
        if not np.isfinite(v) or v <= 0:
            v, fb = r0, fb + 1
        raws[k] = v
    return raws, fb


def mean_w(x, wm=0.5):
    return np.nan if x is None else (x[0] if x[1] is None else (1 - wm) * x[0] + wm * x[1])


def groups_of(tdir):
    g = {}
    for name in GROUPS:
        p = Path(tdir) / f"tags_{name}.txt"
        refuse(p)
        for line in p.read_text().splitlines():
            t = line.strip()
            if t and not t.startswith("#"):
                g[t] = name
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-def", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tags", nargs="*", default=None)
    a = ap.parse_args()
    B = json.loads(Path(a.batch_def).read_text())
    os.nice(10)
    F = Files(B["folder"], B.get("seal"))
    V = Files(B["v10"], B.get("v10_seal"))
    G = groups_of(B["tags"])
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    summ = dict(tool="relsimple.py", script_sha256=sha256b(Path(__file__).read_bytes()), batch_def=B, started=time.strftime("%F %T"),
                segments={}, identity=dict(compared=0, failed=0, max_abs=0.0))
    tags = sorted(t for t, g in G.items() if g == "main")
    if a.tags:
        tags = [t for t in tags if t in set(a.tags)]
    for n, t in enumerate(tags):
        fo = od / f"{t}.csv.gz"
        if fo.exists():
            summ["segments"][t] = dict(status="exists", sha256=sha256b(fo.read_bytes()))
            continue
        base = F.load(f"{t}{B['sfx']}")
        v10c = V.load(f"{t}_cars_v10.json")
        src = "refused" if (base is None or base.get("status") == "refused") else (base.get("next_candidate") or {}).get("source", "current")
        if src != "current" or v10c is None or v10c.get("status") == "refused":
            summ["segments"][t] = dict(status="skipped", source=src)
            continue
        dep = F.load(f"{t}_cars_v4h.json")
        mgc = F.load(f"{t}_cars_v4m.json")
        mg_on = mgc is not None and mgc.get("status") != "refused" and B.get("mg")
        mrk = F.load(f"{t}{B['msfx']}")
        md, mr = marking_maps(mrk)
        Sm = V10.Series(mrk, "dist", 0.15) if mrk is not None and mrk.get("status") != "refused" else None
        v9_has = {(s["fi"], o["id"]) for s in base["samples"] for o in s["objects"] if o.get("rel_kmh") is not None}
        keys0 = sorted({(s["fi"], o["id"]) for s in dep["samples"] for o in s["objects"] if o.get("dist_raw") is not None})
        fd = Path(B["frames_root"]) / t / "frames"
        refuse(fd)
        frames = D.list_frames(fd)
        h, w = cv2.imread(str(frames[0])).shape[:2]
        rr_d, worst_d = reread(dep, frames, Path(B["da3"]) / t, h, w)
        raw_d_up, fb_d = up25_raws(rr_d, dep)
        raw_d_reg = all_raw(dep)
        Rd_reg = rel_speeds(RawSeries(Meas(dep, raw_d_reg)), Sm, mr, keys0, "main", v9_has)
        Rd_up = rel_speeds(RawSeries(Meas(dep, raw_d_up)), Sm, mr, keys0, "main", v9_has)
        Rg_up, worst_m, fb_m = {}, None, None
        if mg_on:
            rr_m, worst_m = reread(mgc, frames, Path(B["mg"]) / t, h, w)
            raw_m_up, fb_m = up25_raws(rr_m, mgc)
            Rg_up = rel_speeds(RawSeries(Meas(mgc, raw_m_up)), Sm, mr, keys0, "main", v9_has)
        rows = []
        rv = {(s["fi"], o["id"]): o for s in v10c["samples"] for o in s["objects"]}
        n_c = n_bad = 0
        mx = 0.0
        for k in keys0:
            ident = mean_w(Rd_reg.get(k))
            xg = Rg_up.get(k)
            used_mg = xg is not None
            x = xg if xg is not None else Rd_up.get(k)
            simple = mean_w(x)
            rows.append((k[0], k[1], ident, simple, int(used_mg)))
            o = rv.get(k)
            if o is None or o.get("rel_kmh") is None or "rel-identity-failed-kept-v9" in (o.get("v10_steps") or []):
                continue
            dd = abs(ident - o["rel_kmh"]) if np.isfinite(ident) else np.inf
            n_c += 1
            n_bad += dd > 1e-6
            mx = max(mx, float(dd))
        with gzip.open(fo, "wt") as fh:
            fh.write("fi,id,r_reg_reg_da3_0.5,r_up25_reg_mg_0.5,mg_used\n")
            for fi, i, a1, a2, u in rows:
                fh.write(f"{fi},{i},{'' if not np.isfinite(a1) else repr(float(a1))},{'' if not np.isfinite(a2) else repr(float(a2))},{u}\n")
        summ["identity"]["compared"] += n_c
        summ["identity"]["failed"] += n_bad
        summ["identity"]["max_abs"] = max(summ["identity"]["max_abs"], mx)
        summ["segments"][t] = dict(status="written", readings=len(keys0), identity_compared=n_c, identity_failed=n_bad,
                                   identity_max_abs=mx, worst_reg_da3=worst_d, worst_reg_mg=worst_m, up25_fallback_da3=fb_d,
                                   up25_fallback_mg=fb_m, mg=bool(mg_on), sha256=sha256b(fo.read_bytes()))
        print(f"{B['name']} {n + 1}/{len(tags)} {t}: readings {len(keys0)}, identity {n_c - n_bad}/{n_c} (max {mx:.2e}), "
              f"reg worst {worst_d:.2e}, mg {bool(mg_on)} {time.strftime('%T')}", flush=True)
    summ["inputs_read"] = dict(sorted(F.read.items()))
    summ["v10_inputs_read"] = dict(sorted(V.read.items()))
    summ["ended"] = time.strftime("%F %T")
    (od / "relsimple_run.json").write_text(json.dumps(summ, indent=1))
    print("identity", summ["identity"])


if __name__ == "__main__":
    main()

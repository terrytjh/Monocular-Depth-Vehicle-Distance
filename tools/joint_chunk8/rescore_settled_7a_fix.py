"""Development numbers under the SETTLED shared scoring (2026-10-18): second car truth = processed radar - 0.70 m,
matching centre column 615 (582 as the sensitivity), duplicates merged 1 m / 0.5 m, width gate 0.8-4.5 m, no distance
gate, moving >= 7.2 km/h by pose speed, |lat| <= 5.49 m; readings beyond 60 m carry no number. Own re-implementation
(decisions_20261018/g1/rescore_lib.py, S rules), not the shared repo's code. Opened data only (Chunk_7 both halves)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/mnt/d/depth_work/decisions_20261018/g1")
import rescore_lib as L

DS = Path("/home/terry/Monocular-Depth-Vehicle-Distance/data/output/dash_scale")
RUNS = [("Chunk_7a", "v9 (dry run, S7+S8)", Path("/mnt/d/depth_work/v9dry/run"), DS / "blind_v8", False)]

def load_any(folder, tag, sealed):
    p = folder / f"{tag}_cars_v9.json"
    if not p.exists():
        return None
    c = json.loads(p.read_text())
    return None if c.get("status") == "refused" else c

def boot(rows, seed=20261018, n=2000):
    by = {}
    for r in rows: by.setdefault(r["seg"], []).append(abs(r["err"]))
    g = list(by.values()); rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        idx = rng.integers(0, len(g), len(g)); out.append(np.median(np.concatenate([g[i] for i in idx])))
    return np.percentile(out, [2.5, 97.5])

res = {}
for chunk, ver, folder, tagdir, sealed in RUNS:
    for cx in (615.0, 582.0):
        L.CAM_CX = cx
        o = dict(L.S_DEFAULT, off=-0.70)
        for grp in ("main", "local_day", "night"):
            tags = L.tags_of(tagdir, grp)
            rows, nt, nf, nout = [], 0, 0, 0
            for t in tags:
                c = load_any(folder, t, sealed)
                if c is None: continue
                nout += 1
                pr, a, b = L.score_seg(c, t, o, 30.0)
                rows += pr; nt += a; nf += b
            rows = [p for p in rows if p["dist"] <= 60.0]
            near = [p for p in rows if p["truth"] <= 30.0]
            if not near: continue
            e = np.array([p["err"] for p in near]); m = np.array([abs(p["dist"] - p["truth"]) for p in near])
            rel = [abs(p["rel"] - p["rel_truth_inst"]) for p in near if p.get("rel") is not None]
            ci = boot(near)
            key = f"{chunk} | {ver} | cx {cx:.0f} | {grp}"
            res[key] = dict(segments=len(tags), with_output=nout, n=len(near), med_pct=round(float(np.median(np.abs(e))), 2),
                            ci95=[round(float(x), 2) for x in ci], med_m=round(float(np.median(m)), 2),
                            bias_pct=round(float(np.median(e)), 2), gt25_pct=round(100 * float(np.mean(np.abs(e) > 25)), 1),
                            found_pct_le30=round(100 * nf / nt, 1) if nt else None, rel_mae=round(float(np.mean(rel)), 2) if rel else None,
                            bins_m={f"{lo}-{hi}": round(float(np.median([abs(p['dist'] - p['truth']) for p in near if lo < p['truth'] <= hi])), 2)
                                    for lo, hi in ((0, 10), (10, 20), (20, 30)) if any(lo < p['truth'] <= hi for p in near)})
            print(key, res[key], flush=True)
json.dump(res, open(Path(__file__).with_name("rescore_settled_7a_fix.json"), "w"), indent=1)

"""Same independent re-score (rescore_own.py) on the registration's two other basis runs: sealed v9 and v9d on the
second half of Chunk_7 (opened 2026-10-16). Main settings only (cx 615, readings as run.py writes them)."""
import json, sys, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import rescore_own as R
DS = Path("/home/terry/Monocular-Depth-Vehicle-Distance/data/output/dash_scale")
V9 = DS / "blind_v9"
RUNS = {"v9 sealed (7b)": V9, "v9d runpy_check (7b)": Path("/mnt/d/depth_work/runpy_check/work_v9d/repo/data/output/dash_scale/c7b_v9d")}
groups = {g: (V9 / f"tags_{g}.txt").read_text().split() for g in ("main", "local_day", "night")}
z = zipfile.ZipFile(R.ZIP)
truth = {"zip": {}, "terry": {}}
for tags in groups.values():
    for t in tags:
        truth["zip"][t] = R.truth_zip(z, t); truth["terry"][t] = R.truth_terry(t)
res = {}
for name, folder in RUNS.items():
    cars = {}
    for tags in groups.values():
        for t in tags:
            c = json.loads((folder / f"{t}_cars_v9.json").read_text()); cars[t] = None if c.get("status") == "refused" else c
    for tv in ("zip", "terry"):
        for cx in (615.0, 582.0):
            for g, tags in groups.items():
                P, ns, ns30, nout = [], 0, 0, 0
                for t in tags:
                    if cars[t] is None: continue
                    nout += 1
                    p, a, b = R.score_seg(t, cars[t], truth[tv][t], cx, False); P += p; ns += a; ns30 += b
                key = f"{name} | truth {tv} | cx {cx:.0f} | {g}"
                res[key] = dict(segments=len(tags), with_output=nout, **R.summarize(P, ns, ns30)); print(key, res[key], flush=True)
Path("/mnt/d/depth_work/audit_chunk8_20261019/dryrun/rescore_own_7b.json").write_text(json.dumps(res, indent=1))

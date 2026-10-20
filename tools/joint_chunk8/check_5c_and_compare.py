"""Before the seal (predictions only, no truth): (1) in each work folder, every step-5c / S8 segment's MoGe-2 measurement is newer
than its cache and every *_cars_v8.json is newer than the work/mg/*_cars_v5.json it was made from; (2) the files that do not use the
camera k are identical between work_v9d and work_v9 (addendum 1, section C)."""
import hashlib, glob, os, sys
from pathlib import Path
J = Path("/mnt/d/depth_work/joint/chunk8"); CM = J / "cache_moge2"
def O(v): return J / f"work_{v}/repo/data/output/dash_scale/comma2k19_chunk8"
bad = 0
for v in ("v9d", "v9"):
    o = O(v); n5c = n8 = nv8 = 0
    for f in sorted((o / "work/mg").glob("*_cars_v4.json")):
        t = f.name[:-13]; n5c += 1
        if not (CM / t).exists() or f.stat().st_mtime <= (CM / t).stat().st_mtime:
            print(v, "5c measurement not newer than its cache:", t); bad += 1
        v5 = o / "work/mg" / f"{t}_cars_v5.json"; c8 = o / f"{t}_cars_v8.json"
        if v5.exists() and c8.exists():
            nv8 += 1
            if c8.stat().st_mtime < v5.stat().st_mtime:
                print(v, "cars_v8 older than its MoGe-2 cars_v5:", t); bad += 1
    for f in sorted((o / "work/mg8").glob("*_cars_v4.json")):
        t = f.name[:-13]; n8 += 1
        if not (CM / t).exists() or f.stat().st_mtime <= (CM / t).stat().st_mtime:
            print(v, "S8 measurement not newer than its cache:", t); bad += 1
    print(f"{v}: step-5c measurements {n5c}, with cars_v8 checked {nv8}, S8 measurements {n8}")
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
same = diff = only = 0; diffs = {}
for suf in ("_dash.json", "_tracks.json", "_tracks_bt.json", "_run.json", "_cars_c.json", "_cars.json", "_cars_v3.json", "_cars_v4.json"):
    for f in sorted(O("v9d").glob(f"*{suf}")):
        g = O("v9") / f.name
        if not g.exists():
            only += 1; continue
        if sha(f) == sha(g): same += 1
        else:
            diff += 1; diffs.setdefault(suf, []).append(f.name)
print(f"k-independent files: identical {same}, different {diff}, only in v9d {only}")
for k, v in diffs.items():
    print(f"  {k}: {len(v)} differ, e.g. {v[:3]}")
print("problems in the 5c / S8 checks:", bad)

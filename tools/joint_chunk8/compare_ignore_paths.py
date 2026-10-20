"""Second pass of the v9d / v9 comparison: the same files, with the work-folder name in path strings made equal."""
import json, glob
from pathlib import Path
J = Path("/mnt/d/depth_work/joint/chunk8")
def O(v): return J / f"work_{v}/repo/data/output/dash_scale/comma2k19_chunk8"
def norm(x):
    if isinstance(x, dict): return {k: norm(v) for k, v in x.items()}
    if isinstance(x, list): return [norm(v) for v in x]
    if isinstance(x, str): return x.replace("/work_v9d/", "/work_X/").replace("/work_v9/", "/work_X/")
    return x
same = diff = 0; ex = []
for suf in ("_cars_c.json", "_cars.json", "_cars_v3.json", "_cars_v4.json"):
    for f in sorted(O("v9d").glob(f"*{suf}")):
        a = norm(json.loads(f.read_text())); b = norm(json.loads((O("v9") / f.name).read_text()))
        if a == b: same += 1
        else: diff += 1; ex.append(f.name)
print(f"after making the work-folder name equal: identical {same}, different {diff}", ex[:5])

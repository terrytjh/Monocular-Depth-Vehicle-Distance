"""Fetch front-center frames, calibration, poses and the ground-height map for a list of logs.
Deliberately does NOT fetch annotations.feather (the ground truth)."""
import json, os, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
B = "https://s3.amazonaws.com/argoverse"
PRE = "datasets/av2/sensor/val"

def keys(log, sub):
    out, tok = [], ""
    while True:
        u = f"{B}/?list-type=2&prefix={PRE}/{log}/{sub}&max-keys=1000" + (f"&continuation-token={urllib.parse.quote(tok)}" if tok else "")
        x = urllib.request.urlopen(u, timeout=60).read().decode()
        out += re.findall(r"<Key>(.*?)</Key>", x)
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", x)
        if not m: return out
        tok = m.group(1)

def get(a):
    k, dst = a
    if os.path.exists(dst) and os.path.getsize(dst) > 1000: return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    urllib.request.urlretrieve(B + "/" + k, dst); return 1

for log in [l.strip() for l in open(sys.argv[1]) if l.strip()]:
    root = f"data/input/av2/{log}"
    jobs = [(k, f"{root}/frames/" + k.split("/")[-1]) for k in keys(log, "sensors/cameras/ring_front_center/")]
    jobs += [(k, f"{root}/map_" + k.split("/")[-1]) for k in keys(log, "map/")]
    jobs += [(f"{PRE}/{log}/{f}", f"{root}/{f}") for f in
             ("calibration/intrinsics.feather", "calibration/egovehicle_SE3_sensor.feather", "city_SE3_egovehicle.feather")]
    with ThreadPoolExecutor(12) as ex: n = sum(ex.map(get, jobs))
    print(f"{log} {len(jobs)} files ({n} new)", flush=True)
print("FETCH_DONE")

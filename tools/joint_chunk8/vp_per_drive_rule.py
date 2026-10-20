"""Per-drive vanishing-point column by the rule written in addendum 1 of docs/JOINT_CHUNK8_PREREG.md (predictions only).
For every segment's *_run.json: windows with both own-lane lines; crossing y = (b_R - b_L) / (a_L - a_R), x = a_L * y + b_L
(lines are x = a * y + b); keep windows with 0 < x < 1164 and 200 < y < 700; a segment with at least 10 such windows gives the
median x; a drive's value is the median of its segments' medians (all segments, refused and non-main included).
  python vp_per_drive_rule.py <batch folder> [...]   -> one line per drive"""
import glob, json, os, re, sys, collections
import numpy as np

def seg_value(path):
    try:
        d = json.load(open(path))
    except Exception:
        return None
    xs = []
    for w in d.get("windows", []):
        L, R = w.get("left"), w.get("right")
        if not L or not R or abs(L["a"] - R["a"]) < 1e-9:
            continue
        y = (R["b"] - L["b"]) / (L["a"] - R["a"]); x = L["a"] * y + L["b"]
        if 0 < x < 1164 and 200 < y < 700:
            xs.append(x)
    return float(np.median(xs)) if len(xs) >= 10 else None

by = collections.defaultdict(list)
for b in sys.argv[1:]:
    for f in glob.glob(os.path.join(b, "*_run.json")):
        tag = os.path.basename(f)[:-9]
        m = re.match(r"^([0-9a-f]+)_(.+)_(\d+)$", tag)
        if not m:
            continue
        v = seg_value(f)
        if v is not None:
            by[(m.group(1), m.group(2))].append(v)
for (car, drive), v in sorted(by.items()):
    print(f"{car} {drive} segments {len(v):3d} vp {np.median(v):6.1f}")

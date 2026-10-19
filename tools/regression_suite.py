#!/usr/bin/env python3
"""Regression suite of method B (docs/REGRESSION_SUITE.txt): one fixed, small set of opened development segments that
every change of the method is run on, scored per group, so that a change that helps one batch or condition and hurts
another is seen at once. Opened development data only.

  select   apply the selection rule (its text from --rule, written before any selection) and write the suite file.
           Reads only the batches' group lists, the seal file names, whether depth-cache folders and truth files exist
           (not their content) and the segment names of chunk13_status.csv. No score, truth or prediction is read.
  score    score one or more versions on the suite's segments only, with the settled shared rules (the scoring core of
           tools/v10_score.py: second car truth = processed radar - 0.70 m, matching column 615 from the
           2018-06-25--22-19-02 remount on and 589 before it, first car - 0.33 m and 598; true distance <= 30 m;
           readings the method puts beyond 60 m carry no number); per car x group, per batch x group, per batch, per
           group and overall; paired against a baseline version on the same readings and the same 1 s bins.
           Ego-only groups: Haisheng hs_rerun (span mean of the per-second ego speed over each manual frame-method span,
           a "gap", not an error) and the two Taiwan web clips with a speed overlay (per second vs the overlay speed read
           after the seal, osd_speed.json).

A version is given as  --version LABEL SPEC [SPEC ...]  where each SPEC is
  @sealed | @v10base | @v10           a preset (below)
  BATCH=DIR:SUFFIX                    the version's files of that batch are DIR/<segment>SUFFIX
  *=DIR:SUFFIX                        the same folder for every batch (e.g. a folder of tools/v10_post.py outputs)
Later SPECs override earlier ones per batch. A batch with no SPEC is "not run" for that version (left out of its
cells); a mapped batch whose file is missing or refused counts as "no output" (in the coverage denominators).
Files in an opened batch folder whose name is in that batch's seal must still match the seal's sha256 (else refused);
other files are read unsealed and their sha256 is recorded.

  python3 tools/regression_suite.py select --rule RULE.txt --out docs/REGRESSION_SUITE.txt
  python3 tools/regression_suite.py score --version sealed @sealed --version v10 @v10 --version v10base @v10base \\
      --pair v10 v10base --out X.json --md X.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))
import v10_score as VS  # noqa: E402  (the settled shared rules)

DS = ROOT / "data/output/dash_scale"
TRUTH = ROOT / "data/output/c2k19_truth"
SUITE = ROOT / "docs/REGRESSION_SUITE.txt"
REG = Path("/mnt/d/depth_work/regression_20261019")
V9DRY = Path("/mnt/d/depth_work/v9dry/run")
CHUNK13 = Path("/mnt/d/depth_work/chunk13_audit_20261019/chunk13_status.csv")
HS_MANUAL = ROOT / "data/input/海盛_20251230/汽車行車紀錄器 - 人工標註"
TW_OSD = DS / "blind_v4_tw/osd_speed.json"
FORBIDDEN_PARTS = ("/mnt/d/depth_work/joint", "c2k19_own/Chunk_8", "radial", "RADIal")

GROUPS = ("main", "local_day", "night")
EGO_GROUPS = ("haisheng_ego", "taiwan_ego")
CAR_NAME = {VS.CAR1: "car1", VS.CAR2: "car2"}
POOL = (("blind_v3", "Chunk_2"), ("blind_v4", "Chunk_4"), ("blind_v5", "Chunk_5"), ("blind_v7", "Chunk_6"),
        ("blind_v8", "Chunk_7a"), ("blind_v9", "Chunk_7b"))
CHUNK_OF = dict(POOL)
CACHE = {b: DS / "cache" / b for b, _ in POOL}
CACHE["blind_v3"] = Path("/mnt/d/depth_work/improve/chunk2/cache")      # cache/blind_v3 was deleted on 10/2
CACHE["hs_rerun"] = DS / "cache/hs_rerun"
CACHE["blind_v4_tw"] = DS / "cache/blind_v4_tw"
MOGE2 = (Path("/mnt/d/depth_work/cache_v9_moge2"), Path("/mnt/d/depth_work/cache_v8_moge2"),
         V9DRY.parent / "moge_cache", Path("/mnt/d/depth_work/fusion/cache_moge2"))
SEALS = {"blind_v3": ("seal.json",), "blind_v4": ("seal.json",), "blind_v5": ("seal.json", "seal_v6.json"),
         "blind_v7": ("seal.json",), "blind_v8": ("seal.json",), "blind_v9": ("seal.json",),
         "hs_rerun": ("seal.json", "seal_005.json"), "blind_v4_tw": ("seal.json",)}
EGO_UNITS = (("hs1230car_002", "hs_rerun", "haisheng_ego"), ("hs1230car_005", "hs_rerun", "haisheng_ego"),
             ("hs1230car_006", "hs_rerun", "haisheng_ego"), ("yt_Pa1I2nT9FuQ", "blind_v4_tw", "taiwan_ego"),
             ("yt_yh0klTheNfU_s600", "blind_v4_tw", "taiwan_ego"))
QUOTA = {"car1": dict(main=6, local_day=4, night=4), "car2": dict(main=4, local_day=2, night=2)}
MAX_PER_DRIVE = 3
SEED = 20261019
# a cell is judged only with at least this much (else "too few to judge")
MIN_SEGMENTS, MIN_READINGS, MIN_SECONDS, MIN_SPANS = 3, 100, 60, 10

PRESETS = {
    "sealed": {   # the version each batch was sealed with (latest sealed one)
        "blind_v3": (DS / "blind_v3", "_cars_v3.json", "v3(封存)"),
        "blind_v4": (DS / "blind_v4", "_cars_v4.json", "v4(封存)"),
        "blind_v5": (DS / "blind_v5", "_cars_v6.json", "v6(封存 seal_v6)"),
        "blind_v7": (DS / "blind_v7", "_cars_v7.json", "v7(封存)"),
        "blind_v8": (DS / "blind_v8", "_cars_v8.json", "v8(封存)"),
        "blind_v9": (DS / "blind_v9", "_cars_v9.json", "v9(封存)"),
        "hs_rerun": (DS / "hs_rerun", "_cars_v6.json", "v6(封存,開發片)"),
        "blind_v4_tw": (DS / "blind_v4_tw", "_cars_v4.json", "v4(封存)"),
    },
    "v10base": {  # the input tools/v10_post.py was run on in each development batch
        "blind_v7": (DS / "blind_v7", "_cars_v7.json", "v7(封存)"),
        "blind_v8": (V9DRY, "_cars_v9.json", "v9 試跑(未封存)"),
        "blind_v9": (DS / "blind_v9", "_cars_v9.json", "v9(封存)"),
    },
    "v10": {
        "blind_v7": (REG / "v10/blind_v7", "_cars_v10.json", "v10(底 = v7 封存)"),
        "blind_v8": (REG / "v10/blind_v8", "_cars_v10.json", "v10(底 = v9 試跑)"),
        "blind_v9": (REG / "v10/blind_v9", "_cars_v10.json", "v10(底 = v9 封存)"),
    },
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256(p) -> str:
    return sha256_bytes(Path(p).read_bytes())


def drive_of(tag):
    return tag.rsplit("_", 1)[0]


def tags_of(folder: Path, g: str):
    p = folder / f"tags_{g}.txt"
    return [l.strip() for l in open(p) if l.strip() and not l.startswith("#")] if p.exists() else []


def chunk1_segments():
    out = set()
    with open(CHUNK13, newline="") as f:
        for r in csv.DictReader(f):
            if r["chunk"] == "Chunk_1":
                out.add(r["segment_id"])
    return out


def sealed_dash_names(batch):
    s = json.loads((DS / batch / "seal.json").read_text())
    return {k[: -len("_dash.json")] for k in s.get("files", {}) if k.endswith("_dash.json")}


def nonempty_dir(p: Path):
    return p.is_dir() and any(p.iterdir())


def moge2_cache(tag):
    for m in MOGE2:
        if nonempty_dir(m / tag):
            return str(m / tag)
    return "-"


# ------------------------------------------------------------------ select

def mount_of(tag):
    if not tag.startswith(VS.CAR2):
        return "-"
    return "before" if drive_of(tag).split("_", 1)[1] < VS.REMOUNT else "after"


def cmd_select(a):
    rule_p = Path(a.rule)
    rule = rule_p.read_text()
    rule_time = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(rule_p.stat().st_mtime))
    rng = random.Random(a.seed)
    c1 = chunk1_segments()
    pool, excluded = {}, Counter()
    for b, chunk in POOL:
        sealed = sealed_dash_names(b)
        rl = {"_".join(l.split("|")[:3]): l.split("|")[3] for l in (DS / b / "run_list.txt").read_text().splitlines() if l.strip()}
        for g in GROUPS:
            for t in sorted(tags_of(DS / b, g)):
                if t in c1:
                    raise SystemExit(f"{t}: a Chunk_1 segment in {b} -- refusing")
                if rl.get(t) != chunk[:7]:
                    raise SystemExit(f"{t}: run_list chunk {rl.get(t)} != {chunk}")
                if t not in sealed:
                    excluded[(b, g, "not in seal")] += 1
                    continue
                if not ((TRUTH / t / "radar.csv").exists() and (TRUTH / t / "ego.csv").exists()):
                    excluded[(b, g, "no truth file")] += 1
                    continue
                if not nonempty_dir(CACHE[b] / t):
                    excluded[(b, g, "no depth cache")] += 1
                    continue
                if t in pool:
                    raise SystemExit(f"{t} in two groups")
                pool[t] = dict(batch=b, car=CAR_NAME[t.split("_")[0]], group=g, chunk=chunk, mount=mount_of(t),
                               cache=str(CACHE[b] / t))
    strata = []      # (car, batch, group, mount, quota) in the fixed order of the rule
    for g in GROUPS:
        strata.append(("car1", "blind_v3", g, "-", QUOTA["car1"][g]))
    for b, _ in POOL[1:]:
        for g in GROUPS:
            q = QUOTA["car2"][g]
            if b == "blind_v4":
                strata += [("car2", b, g, "before", q // 2), ("car2", b, g, "after", q - q // 2)]
            else:
                strata.append(("car2", b, g, "any", q))
    per_drive = Counter()
    chosen, report = [], []
    for car, b, g, mount, q in strata:
        cand = sorted(t for t, r in pool.items() if r["car"] == car and r["batch"] == b and r["group"] == g
                      and (mount in ("-", "any") or r["mount"] == mount))
        by_drive = defaultdict(list)
        for t in cand:
            by_drive[drive_of(t)].append(t)
        drives = sorted(by_drive)
        rng.shuffle(drives)
        got = []
        while len(got) < q:
            moved = False
            for d in drives:
                if len(got) >= q:
                    break
                if per_drive[d] >= MAX_PER_DRIVE or not by_drive[d]:
                    continue
                t = rng.choice(by_drive[d])
                by_drive[d].remove(t)
                got.append(t)
                per_drive[d] += 1
                moved = True
            if not moved:
                break
        chosen += got
        report.append((car, b, g, mount, q, len(cand), len({drive_of(t) for t in cand}), len(got)))
    lines = []
    for t in chosen:
        r = pool[t]
        lines.append(f"{t}|{r['batch']}|{r['car']}|{r['group']}|{r['chunk']}|{r['mount']}|{r['cache']}|{moge2_cache(t)}")
    ego_lines = []
    for t, b, g in EGO_UNITS:
        if not nonempty_dir(CACHE[b] / t):
            raise SystemExit(f"{t}: no depth cache")
        ego_lines.append(f"{t}|{b}|{'hs' if b == 'hs_rerun' else 'tw'}|{g}|-|-|{CACHE[b] / t}|-")
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    H = [rule.rstrip("\n"), "#",
         f"# 規則檔 {rule_p}(sha256 {sha256(rule_p)},寫定 {rule_time});選段執行 {now},"
         f"tools/regression_suite.py select --seed {a.seed}。",
         "# 選段時沒有看任何誤差數字:選段程式不讀分數檔、真值內容或預測內容。",
         "#", "# 各層:車 | 批 | 組 | 重裝 | 配額 | 池裡段數 | 池裡趟數 | 選到"]
    H += [f"#   {c} | {b} | {g} | {m} | {q} | {n} | {nd} | {k}" for c, b, g, m, q, n, nd, k in report]
    H += ["# 池外(不收)的段:" + ("; ".join(f"{b} {g} {why} {n}" for (b, g, why), n in sorted(excluded.items())) or "無")]
    cnt = Counter((pool[t]["car"], pool[t]["group"]) for t in chosen)
    H += ["# 各組段數:" + "、".join(f"{c} {g} {cnt[(c, g)]}" for c in ("car1", "car2") for g in GROUPS)
          + f";comma2k19 合計 {len(chosen)} 段(趟數 {len({drive_of(t) for t in chosen})},每趟最多 "
          f"{max(Counter(drive_of(t) for t in chosen).values())} 段);海盛 3 支、台灣 2 支(只評自車速)。",
          "#", "# 欄位:segment|batch|car|group|chunk|mount|depth_cache|moge2_cache"
          "(chunk:Chunk_7a = Chunk_7 前半 blind_v8、Chunk_7b = 後半 blind_v9;mount = 第二台車相機重裝前 / 後;"
          "moge2_cache 只供參考,第八版起的流程在白天要用,'-' = 沒有,重跑完整流程時要先用 GPU 補)"]
    body = "\n".join(H) + "\n" + "\n".join(lines + ego_lines) + "\n"
    Path(a.out).write_text(body)
    print(f"wrote {a.out}: {len(chosen)} comma2k19 segments + {len(ego_lines)} ego-only clips")
    for x in report:
        print("  ", x)


# ------------------------------------------------------------------ suite and guards

class Unit:
    def __init__(self, line):
        f = line.split("|")
        self.tag, self.batch, self.car, self.group, self.chunk, self.mount, self.cache = f[:7]

    @property
    def ego_only(self):
        return self.group in EGO_GROUPS


def read_suite(path):
    raw = Path(path).read_bytes()
    units = [Unit(l) for l in raw.decode().splitlines() if l.strip() and not l.startswith("#")]
    if len({u.tag for u in units}) != len(units):
        raise SystemExit("duplicate segment in the suite")
    return units, sha256_bytes(raw)


def guard(units):
    """Before any truth is read: every unit must be opened development data with its truth on disk."""
    c1 = chunk1_segments()
    opened = {b: sealed_dash_names(b) for b, _ in POOL}
    for u in units:
        if any(x in str(Path(u.cache).resolve()) for x in FORBIDDEN_PARTS):
            raise SystemExit(f"{u.tag}: forbidden path")
        if u.ego_only:
            if (u.tag, u.batch, u.group) not in EGO_UNITS:
                raise SystemExit(f"{u.tag}: not one of the opened ego-only clips -- refusing")
            continue
        if u.batch not in opened or u.group not in GROUPS:
            raise SystemExit(f"{u.tag}: batch {u.batch} / group {u.group} is not an opened development batch -- refusing")
        if u.tag in c1:
            raise SystemExit(f"{u.tag}: Chunk_1 -- refusing")
        if u.tag not in opened[u.batch]:
            raise SystemExit(f"{u.tag}: not in the seal of {u.batch} -- refusing")
        if u.tag not in tags_of(DS / u.batch, u.group):
            raise SystemExit(f"{u.tag}: not in {u.batch}/tags_{u.group}.txt -- refusing")
        tp = (TRUTH / u.tag).resolve()
        if any(x in str(tp) for x in FORBIDDEN_PARTS) or not ((TRUTH / u.tag / "radar.csv").exists() and (TRUTH / u.tag / "ego.csv").exists()):
            raise SystemExit(f"{u.tag}: no opened truth -- refusing")


# ------------------------------------------------------------------ versions

def parse_version(label, specs):
    m = {}
    for s in specs:
        if s.startswith("@"):
            name = s[1:]
            if name not in PRESETS:
                raise SystemExit(f"unknown preset {s}")
            m.update({b: v for b, v in PRESETS[name].items()})
            continue
        if "=" not in s or ":" not in s:
            raise SystemExit(f"bad version spec {s!r} (BATCH=DIR:SUFFIX, *=DIR:SUFFIX or @preset)")
        b, rest = s.split("=", 1)
        d, suf = rest.rsplit(":", 1)
        bs = [x for x, _ in POOL] + ["hs_rerun", "blind_v4_tw"] if b == "*" else [b]
        for x in bs:
            m[x] = (Path(d), suf, f"{label}({d})")
    return m


class Version:
    def __init__(self, label, mapping):
        self.label, self.map = label, mapping
        self.files = {}

    def load(self, u):
        """('not_run' | 'absent' | 'refused' | 'ok', content)."""
        if u.batch not in self.map:
            return "not_run", None
        d, suf, _ = self.map[u.batch]
        p = Path(d) / f"{u.tag}{suf}"
        if any(x in str(p.resolve()) for x in FORBIDDEN_PARTS):
            raise SystemExit(f"{p}: forbidden path")
        if not p.exists():
            self.files[p.name] = dict(path=str(p), status="absent")
            return "absent", None
        raw = p.read_bytes()
        h = sha256_bytes(raw)
        rec = dict(path=str(p), sha256=h, sealed=None)
        if u.batch in SEALS and Path(d).resolve() == (DS / u.batch).resolve():
            for sn in SEALS[u.batch]:
                ent = json.loads((DS / u.batch / sn).read_text()).get("files", {})
                if p.name in ent:
                    if ent[p.name] != h:
                        raise SystemExit(f"{p}: changed since the seal {sn} -- refusing")
                    rec["sealed"] = f"{u.batch}/{sn}"
        c = json.loads(raw)
        st = "refused" if c.get("status") == "refused" else "ok"
        rec["status"] = st
        self.files[p.name] = rec
        return st, (c if st == "ok" else None)


# ------------------------------------------------------------------ scoring: comma2k19

def score_c2k(v, units):
    out = {}
    for u in units:
        st, c = v.load(u)
        if st == "not_run":
            continue
        n_sec = int(len(VS.load_truth(TRUTH, u.tag)[1]) // 20)
        rec = dict(unit=u, status=st, seconds_all=n_sec, rows=[], bins=[], n_truth=0, n_found=0)
        if st == "ok":
            pr, nt, nf, bins, _ = VS.settled_segment(c, u.tag, TRUTH, VS.column_of(drive_of(u.tag), {}))
            rec.update(rows=[p for p in pr if p["truth"] <= VS.SETTLED["near_max_m"]], bins=bins, n_truth=nt, n_found=nf)
        out[u.tag] = rec
    return out


def enough(rows, key_seg="seg", minimum=MIN_READINGS):
    return len(rows) >= minimum and len({r[key_seg] for r in rows}) >= MIN_SEGMENTS


def cell_c2k(recs):
    rows = [p for r in recs for p in r["rows"]]
    rel = [p for p in rows if p.get("rel") is not None]
    bins = [b for r in recs for b in r["bins"]]
    nt, nf = sum(r["n_truth"] for r in recs), sum(r["n_found"] for r in recs)
    sec_all = sum(r["seconds_all"] for r in recs)
    d = VS.dist_summ(rows, ci=True)
    d.update(found_pct=round(100 * nf / nt, 1) if nt else None, truth_in_scope=nt)
    e = VS.speed_summ([b["kmh"] - b["truth"] for b in bins])
    e.update(seconds_with_value=len(bins), seconds_all=sec_all,
             coverage_pct=round(100 * len(bins) / sec_all, 1) if sec_all else None,
             segments_with_seconds=len({b["seg"] for b in bins}))
    return dict(segments=len(recs), with_output=sum(r["status"] == "ok" for r in recs),
                segments_with_readings=len({p["seg"] for p in rows}),
                distance=d, relative_speed=VS.rel_summ(rows), ego=e,
                judge=dict(distance=enough(rows), relative_speed=enough(rel),
                           ego=enough(bins, minimum=MIN_SECONDS)))


def cells_of(units):
    """cell name -> predicate on a unit, in report order."""
    c = {}
    for car in ("car1", "car2"):
        for g in GROUPS:
            c[f"{car}|{g}"] = (lambda car, g: lambda u: u.car == car and u.group == g)(car, g)
    for b, _ in POOL:
        for g in GROUPS:
            c[f"{b}|{g}"] = (lambda b, g: lambda u: u.batch == b and u.group == g)(b, g)
        c[f"{b}|all"] = (lambda b: lambda u: u.batch == b)(b)
    for car in ("car1", "car2"):
        c[f"{car}|all"] = (lambda car: lambda u: u.car == car)(car)
    for g in GROUPS:
        c[f"all|{g}"] = (lambda g: lambda u: u.group == g)(g)
    c["all|all"] = lambda u: True
    return c


def paired_c2k(new, old, pred):
    tags = sorted(t for t in new if t in old and pred(new[t]["unit"]))
    if not tags:
        return None
    n = dict(rows=[p for t in tags for p in new[t]["rows"]], bins=[b for t in tags for b in new[t]["bins"]])
    o = dict(rows=[p for t in tags for p in old[t]["rows"]], bins=[b for t in tags for b in old[t]["bins"]])
    r = VS.paired_settled(n, o)
    r["segments_both_run"] = len(tags)
    for k, minimum in (("distance_pct", MIN_READINGS), ("relative_speed_kmh", MIN_READINGS), ("ego_per_second_kmh", MIN_SECONDS)):
        q = r[k]
        q["judged"] = bool(q.get("n", 0) >= minimum and q.get("sign_test_by_segment", {}).get("units", 0) >= MIN_SEGMENTS)
    return r


# ------------------------------------------------------------------ scoring: ego-only clips

def hs_spans(tag):
    from haisheng_manual_truth import parse
    case = tag.split("_")[1]
    xl = glob.glob(str(HS_MANUAL / f"{case}_*" / "*_manual.xlsx"))
    if len(xl) != 1:
        raise SystemExit(f"{tag}: {len(xl)} manual sheets")
    _, segs, _ = parse(xl[0])
    return segs


def ego_seconds(c):
    return {int(round(e["t0"])): max(0.0, float(e["kmh"])) for e in (c.get("ego") or []) if e.get("kmh") is not None}


def span_mean(c, a1, b1):
    """Overlap-weighted mean of the per-second values over original frames a1..b1 (1-based) of the manual span;
    None unless seconds with a value cover at least half of the span."""
    fps = float(c["fps"])
    ego = ego_seconds(c)
    s0, s1 = (a1 - 1) / fps, (b1 - 1) / fps
    num = w = 0.0
    for b in range(int(math.floor(s0)), int(math.ceil(s1))):
        ov = max(0.0, min(s1, b + 1) - max(s0, b))
        if ov > 0 and b in ego:
            num += ov * ego[b]
            w += ov
    return num / w if w >= 0.5 * (s1 - s0) else None


def agreement(d):
    d = np.asarray(d, float)
    if not len(d):
        return dict(n=0)
    out = dict(n=int(len(d)), mean_abs=round(float(np.mean(np.abs(d))), 3), bias=round(float(np.mean(d)), 3),
               max_abs=round(float(np.max(np.abs(d))), 2))
    if len(d) >= 2:
        s = float(np.std(d, ddof=1))
        out.update(sd=round(s, 3), loa95=[round(out["bias"] - 1.96 * s, 2), round(out["bias"] + 1.96 * s, 2)])
    return out


def score_ego_only(v, units, osd):
    out = {}
    for u in units:
        st, c = v.load(u)
        if st == "not_run":
            continue
        pts = {}
        if u.group == "haisheng_ego":
            spans = hs_spans(u.tag)
            total = len(spans)
            if c is not None:
                for sg in spans:
                    m = span_mean(c, sg["f_start"], sg["f_end"])
                    if m is not None:
                        pts[sg["seg"]] = (m, sg["speed_kmh"])
        else:
            ref = osd[u.tag]["speed_kmh"]
            total = len(ref)
            if c is not None:
                for b, val in ego_seconds(c).items():
                    if 0 <= b < len(ref) and ref[b] is not None:
                        pts[b] = (val, float(ref[b]))
        out[u.tag] = dict(unit=u, status=st, total=total, pts=pts)
    return out


def cell_ego(recs):
    d = [m - r for x in recs for m, r in x["pts"].values()]
    n_tot = sum(x["total"] for x in recs)
    a = agreement(d)
    with_pts = sum(1 for x in recs if x["pts"])
    a.update(clips=len(recs), with_output=sum(x["status"] == "ok" for x in recs), of=n_tot, clips_with_points=with_pts,
             coverage_pct=round(100 * len(d) / n_tot, 1) if n_tot else None,
             # one clip: >= MIN_SPANS points; a group of clips: also >= 2 clips with points
             judged=bool(len(d) >= MIN_SPANS and with_pts >= min(2, len(recs))))
    return a


def paired_ego(new, old, pred):
    tags = sorted(t for t in new if t in old and pred(new[t]["unit"]))
    pn, po = [], []
    for t in tags:
        for k in sorted(set(new[t]["pts"]) & set(old[t]["pts"])):
            m1, r = new[t]["pts"][k]
            m0, _ = old[t]["pts"][k]
            pn.append(m1 - r)
            po.append(m0 - r)
    if not pn:
        return dict(n=0, clips_both_run=len(tags))
    return dict(n=len(pn), clips_both_run=len(tags), new_mean_abs=round(float(np.mean(np.abs(pn))), 3),
                old_mean_abs=round(float(np.mean(np.abs(po))), 3), new_bias=round(float(np.mean(pn)), 3),
                old_bias=round(float(np.mean(po)), 3),
                better=int(sum(abs(a) < abs(b) for a, b in zip(pn, po))), worse=int(sum(abs(a) > abs(b) for a, b in zip(pn, po))))


# ------------------------------------------------------------------ report

def fmt(x, p=2):
    return "—" if x is None else f"{x:.{p}f}"


def md_tables(res):
    L = []
    for lab, vr in res["versions"].items():
        L += [f"### {lab}", "", "批次對應:" + ";".join(f"{b} = {n}" for b, n in vr["mapping"].items()), "",
              "| 格 | 段(有輸出/跑) | 距離讀值 | 中位 |誤差| % | m | 偏差 % | >25 % | 找到率 % | 相對速度 MAE(讀值) | 自車速逐秒 MAE(秒) | 自車速覆蓋 % | 判斷 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, c in vr["cells"].items():
            if not c["segments"]:
                continue
            d, r, e, j = c["distance"], c["relative_speed"], c["ego"], c["judge"]
            flag = "".join(x for x, ok in (("距", j["distance"]), ("相", j["relative_speed"]), ("自", j["ego"])) if not ok)
            L.append(f"| {k} | {c['with_output']}/{c['segments']} | {d.get('n', 0)} | {fmt(d.get('median_abs_pct'))} | "
                     f"{fmt(d.get('median_abs_m'))} | {fmt(d.get('bias_pct'))} | {fmt(d.get('gt25_pct'), 1)} | "
                     f"{fmt(d.get('found_pct'), 1)} | {fmt(r.get('mae_kmh'))}({r.get('n', 0)}) | "
                     f"{fmt(e.get('mae_kmh'))}({e.get('seconds_with_value', 0)}) | {fmt(e.get('coverage_pct'), 1)} | "
                     f"{('太少:' + flag) if flag else '可'} |")
        L += ["", "| 只評自車速(海盛 = 對人工畫格法的差距;台灣 = 對疊字車速) | 片(有輸出/跑) | 有值 / 全部 | 平均差距 km/h | 偏差 | 95% 一致界限 | 最大 | 判斷 |",
              "|---|---|---|---|---|---|---|---|"]
        for k, c in vr["ego_cells"].items():
            if not c["clips"]:
                continue
            L.append(f"| {k} | {c['with_output']}/{c['clips']} | {c['n']} / {c['of']} | {fmt(c.get('mean_abs'))} | "
                     f"{fmt(c.get('bias'))} | {c.get('loa95', '—')} | {fmt(c.get('max_abs'), 1)} | {'可' if c['judged'] else '太少'} |")
        L.append("")
    for k, pr in res["paired"].items():
        L += [f"### 配對 {k}(同一批讀值 / 同一秒)", "",
              "| 格 | 兩版都跑的段 | 距離 n | 舊 → 新 中位 |誤差| % | 差的 95% 區間 | 段 好/壞 | 相對速度 n | 舊 → 新 MAE | 段 好/壞 | 自車速 n(秒) | 舊 → 新 MAE | 段 好/壞 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c, r in pr["cells"].items():
            if r is None:
                continue
            q, s, e = r["distance_pct"], r["relative_speed_kmh"], r["ego_per_second_kmh"]

            def one(x):
                if not x.get("n"):
                    return "0", "—", "—", "—"
                st = x["sign_test_by_segment"]
                tag = "" if x.get("judged") else "(太少)"
                return (str(x["n"]), f"{x['old']:.2f} → {x['new']:.2f}{tag}", str(x.get("ci95_difference_by_segment", "—")),
                        f"{st['better']}/{st['worse']}")
            a, b, cc = one(q), one(s), one(e)
            L.append(f"| {c} | {r['segments_both_run']} | {a[0]} | {a[1]} | {a[2]} | {a[3]} | {b[0]} | {b[1]} | {b[3]} | "
                     f"{cc[0]} | {cc[1]} | {cc[3]} |")
        for c, r in pr["ego_cells"].items():
            if r.get("n"):
                L.append(f"| {c}(只評自車速) | {r['clips_both_run']} 片 | | | | | | | | {r['n']} | "
                         f"{r['old_mean_abs']:.2f} → {r['new_mean_abs']:.2f} | 單位 {r['better']}/{r['worse']} |")
        L.append("")
    return "\n".join(L)


def cmd_score(a):
    units, suite_sha = read_suite(a.suite)
    if a.only:
        bad = [t for t in a.only if t not in {u.tag for u in units}]
        if bad:
            raise SystemExit(f"not in the suite file -- refusing: {bad}")
        units = [u for u in units if u.tag in set(a.only)]
    guard(units)                                   # before any truth is read
    c2k = [u for u in units if not u.ego_only]
    ego = [u for u in units if u.ego_only]
    osd = None
    if any(u.group == "taiwan_ego" for u in ego):
        o = json.loads(TW_OSD.read_text())
        osd = {u.tag: o[u.tag] for u in ego if u.group == "taiwan_ego"}
    versions = [Version(lab, parse_version(lab, specs)) for lab, *specs in a.version]
    labels = [v.label for v in versions]
    if len(set(labels)) != len(labels):
        raise SystemExit("duplicate version label")
    pairs = [tuple(p) for p in a.pair]
    for n_, o_ in pairs:
        if n_ not in labels or o_ not in labels:
            raise SystemExit(f"--pair {n_} {o_}: not both among {labels}")
    cells = cells_of(c2k)
    ego_cells = {"haisheng_ego|all": lambda u: u.group == "haisheng_ego", "taiwan_ego|all": lambda u: u.group == "taiwan_ego"}
    for t, _, g in EGO_UNITS:
        ego_cells[f"{g}|{t}"] = (lambda t: lambda u: u.tag == t)(t)
    res = dict(meta=dict(tool="tools/regression_suite.py", run_local=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                         suite=str(a.suite), suite_sha256=suite_sha, units=len(units), comma2k19=len(c2k), ego_only=len(ego),
                         settled_rules=VS.SETTLED, min_to_judge=dict(segments=MIN_SEGMENTS, readings=MIN_READINGS,
                                                                     seconds=MIN_SECONDS, ego_only_points=MIN_SPANS),
                         ego_only_rules=dict(
                             haisheng="span mean: overlap-weighted mean of the per-second ego values (negative as 0) over "
                                      "original frames f_start..f_end (1-based) of each manual frame-method span, at the "
                                      "video's frame rate; a span needs seconds with a value over >= half of it; gap = ours - manual",
                             taiwan="per second: the ego value of bin t0 vs osd_speed.json speed_kmh[t0] (read after the seal "
                                    "at the middle of each 1 s bin); difference = ours - overlay")),
               versions={}, paired={})
    raw = {}
    for v in versions:
        rc, re_ = score_c2k(v, c2k), score_ego_only(v, ego, osd)
        raw[v.label] = (rc, re_)
        res["versions"][v.label] = dict(
            mapping={b: m[2] for b, m in v.map.items()},
            segments={t: r["status"] for t, r in {**rc, **re_}.items()},
            cells={k: cell_c2k([r for r in rc.values() if f(r["unit"])]) for k, f in cells.items()},
            ego_cells={k: cell_ego([r for r in re_.values() if f(r["unit"])]) for k, f in ego_cells.items()},
            files=v.files)
    for n_, o_ in pairs:
        (nc, ne), (oc, oe) = raw[n_], raw[o_]
        res["paired"][f"{n_}_vs_{o_}"] = dict(
            cells={k: paired_c2k(nc, oc, f) for k, f in cells.items()},
            ego_cells={k: paired_ego(ne, oe, f) for k, f in ego_cells.items()})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    md = md_tables(res)
    if a.md:
        Path(a.md).write_text(md + "\n")
    print(md)
    print(f"wrote {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--rule", required=True, help="the rule text, written before the selection (the suite file's header)")
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--out", default=str(SUITE))
    s.set_defaults(fn=cmd_select)
    s = sub.add_parser("score")
    s.add_argument("--suite", default=str(SUITE))
    s.add_argument("--version", nargs="+", action="append", required=True, metavar="LABEL SPEC",
                   help="LABEL then one or more of @preset, BATCH=DIR:SUFFIX, *=DIR:SUFFIX")
    s.add_argument("--pair", nargs=2, action="append", default=[], metavar=("NEW", "OLD"))
    s.add_argument("--only", nargs="*", default=None, help="score only these suite segments (others are refused)")
    s.add_argument("--out", required=True)
    s.add_argument("--md", default=None)
    s.set_defaults(fn=cmd_score)
    a = ap.parse_args()
    if a.cmd == "score":
        for v in a.version:
            if len(v) < 2:
                raise SystemExit("--version LABEL SPEC [SPEC ...]")
    a.fn(a)


if __name__ == "__main__":
    main()

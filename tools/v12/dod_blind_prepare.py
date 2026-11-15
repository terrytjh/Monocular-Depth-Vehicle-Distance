#!/usr/bin/env python3
"""DOD blind tests: the choice of clips, the copy from the external drive and the tag table.
Private outputs only (/mnt/d/depth_work/tw_own/lists, raw_<set>); nothing here reads a picture or the overlay.
--set blind1 (default): docs/DOD_V12_BLIND_PREREG.md;  --set blind2: docs/DOD_V12_BLIND2_PREREG.md (highway trips).

  draw   (blind1) names only (the front-camera name list made 2026-10-25, name + bytes; no file is opened):
         pool   = names recorded 2026-09-26 .. 2026-10-03, sorted;
         trips  = in start-time order (from the name), a new trip when a file starts more than 150 s after the previous
                  file's start (60-s files: more than 90 s after it ended);
         draw   = random.Random(20261113).sample(pool, len(pool)) is the order; walking it, a file is taken while its trip
                  has fewer than 4 taken files, until 60 are taken.
         writes lists/blind1_files.txt (sorted names) and lists/blind1_draw.json (rule, seed, pool sha256, per-trip counts).
  select (blind2) names only, no random draw: the pool is 2026-10-04 .. 2026-10-15 with the same trip rule; taken are the
         two files Terry named as the start of highway driving (2026_10_07_053512, 2026_10_08_131102) and every later file
         of the same trip. Writes lists/blind2_files.txt and lists/blind2_draw.json.
  copy   refuses unless the set's registration is committed and that commit is on origin/main (git fetch first), and the
         private files of the set's manifest equal their registered sha256. Copies the chosen files from the external
         drive folder into raw_<set> and writes lists/<set>_sha256.txt (name, sha256, "same" when the copy equals the source).
  tags   container timing of the copies without decoding (ffmpeg -c copy -f framecrc: sample pts and sizes only) ->
         lists/tags_<set>.tsv (file, tag, trip_label, trip_id, start, duration_s, pictures); tag = dod_<trip_id>_<HHMMSS>-<NN>,
         trip_id = start of the trip's first file in the POOL (so every clip of a trip shares the drive part).
  manifest  the private files of the set's manifest against their registered sha256 (also run by copy).
usage: dod_blind_prepare.py [--set blind1|blind2] draw | select | copy --src /mnt/e/front_norm | tags | manifest
"""
import argparse, csv, datetime as dt, hashlib, json, random, re, shutil, subprocess, sys
from pathlib import Path

TW = Path("/mnt/d/depth_work/tw_own")
LISTS = TW / "lists"
NAME_LIST = LISTS / "dod_front_list_20261025.txt"
REPO = Path(__file__).resolve().parents[2]
FFMPEG = Path.home() / ".local/bin/ffmpeg"
NAME_RE = re.compile(r"^(\d{4})_(\d{2})_(\d{2})_(\d{2})(\d{2})(\d{2})_(\d{2})\.MP4$")
SEED, N, CAP, GAP_S = 20261113, 60, 4, 150
SETS = {
    "blind1": dict(days=tuple(f"2026_09_{d:02d}" for d in range(26, 31)) + tuple(f"2026_10_{d:02d}" for d in range(1, 4)),
                   prereg="docs/DOD_V12_BLIND_PREREG.md", manifest="tools/v12/dod_blind_manifest.json", label="B"),
    "blind2": dict(days=tuple(f"2026_10_{d:02d}" for d in range(4, 16)),
                   prereg="docs/DOD_V12_BLIND2_PREREG.md", manifest="tools/v12/dod_blind2_manifest.json", label="H",
                   starts=("2026_10_07_053512", "2026_10_08_131102")),
}
SET, CFG, RAW = "blind1", SETS["blind1"], TW / "raw_blind1"


def use_set(name):
    global SET, CFG, RAW
    SET, CFG, RAW = name, SETS[name], TW / f"raw_{name}"


def start(n):
    return dt.datetime(*map(int, NAME_RE.match(n).groups()[:6]))


def pool_and_trips():
    names = sorted(l.split()[0] for l in NAME_LIST.read_text().splitlines() if l.strip())
    pool = [n for n in names if NAME_RE.match(n) and n[:10] in CFG["days"]]
    trips, prev = [], None
    for n in sorted(pool, key=lambda x: (start(x), x)):
        if prev is None or (start(n) - prev).total_seconds() > GAP_S:
            trips.append([])
        trips[-1].append(n)
        prev = start(n)
    trip_of = {n: i for i, t in enumerate(trips) for n in t}
    return pool, trips, trip_of


def sha256_file(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def write_choice(taken, pool, trips, per, rule, force, extra=None):
    out = LISTS / f"{SET}_files.txt"
    if out.exists() and not force:
        sys.exit(f"{out} exists (use --force only before the registration is committed)")
    out.write_text("\n".join(sorted(taken)) + "\n")
    rec = dict(rule=rule, n=len(taken), gap_s=GAP_S, name_list=str(NAME_LIST), name_list_sha256=sha256_file(NAME_LIST),
               pool_files=len(pool), pool_sha256=hashlib.sha256(("\n".join(pool) + "\n").encode()).hexdigest(),
               trips_in_pool=len(trips), trip_sizes=[len(t) for t in trips],
               taken_per_trip=[per.get(i, 0) for i in range(len(trips))], files_sha256=sha256_file(out),
               made=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **(extra or {}))
    (LISTS / f"{SET}_draw.json").write_text(json.dumps(rec, indent=1))
    print(f"pool {len(pool)} files in {len(trips)} trips; took {len(taken)} from {sum(1 for v in per.values() if v)} trips")
    print(f"{SET}_files.txt sha256 {rec['files_sha256']}")


def cmd_draw(a):
    if SET != "blind1":
        sys.exit("draw is the first blind test's rule; use select for blind2")
    pool, trips, trip_of = pool_and_trips()
    order = random.Random(SEED).sample(pool, len(pool))
    taken, per = [], {}
    for n in order:
        t = trip_of[n]
        if per.get(t, 0) < CAP:
            taken.append(n)
            per[t] = per.get(t, 0) + 1
            if len(taken) == N:
                break
    if len(taken) != N:
        sys.exit(f"only {len(taken)} files could be drawn")
    write_choice(taken, pool, trips, per, __doc__.split("  draw ")[1].split("  select ")[0].strip(), a.force,
                 dict(seed=SEED, cap_per_trip=CAP))


def cmd_select(a):
    if "starts" not in CFG:
        sys.exit("select is the highway test's rule (--set blind2)")
    pool, trips, trip_of = pool_and_trips()
    taken, per = [], {}
    for key in CFG["starts"]:
        hit = [n for n in pool if n.startswith(key + "_")]
        if len(hit) != 1:
            sys.exit(f"{key}: {len(hit)} files with this start in the pool")
        t = trips[trip_of[hit[0]]]
        part = t[t.index(hit[0]):]
        taken += part
        per[trip_of[hit[0]]] = len(part)
    write_choice(taken, pool, trips, per, __doc__.split("  select ")[1].split("  copy ")[0].strip(), a.force,
                 dict(starts=list(CFG["starts"])))


def prereg_pushed():
    PREREG = CFG["prereg"]

    def git(*x):
        return subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True)
    if git("ls-files", "--error-unmatch", PREREG).returncode != 0 or git("diff", "--quiet", "HEAD", "--", PREREG).returncode != 0:
        sys.exit(f"refused: {PREREG} is not committed as it is")
    c = git("log", "--diff-filter=A", "--format=%H", "--", PREREG).stdout.split()
    if git("fetch", "-q", "origin", "main").returncode != 0:
        sys.exit("refused: git fetch origin main failed (cannot confirm the registration is on GitHub)")
    head = git("log", "-1", "--format=%H", "--", PREREG).stdout.strip()
    for x in {c[-1], head}:
        if git("merge-base", "--is-ancestor", x, "origin/main").returncode != 0:
            sys.exit(f"refused: the registration commit {x[:8]} is not on origin/main (push first)")
    return head


def check_manifest():
    man = json.loads((REPO / CFG["manifest"]).read_text())["files"]
    bad = [k for k, h in man.items() if not (TW / k).is_file() or sha256_file(TW / k) != h]
    if bad:
        sys.exit(f"refused: private files differ from the registered sha256: {bad}")
    return len(man)


def cmd_copy(a):
    head = prereg_pushed()
    check_manifest()
    names = [l.strip() for l in (LISTS / f"{SET}_files.txt").read_text().splitlines() if l.strip()]
    draw = json.loads((LISTS / f"{SET}_draw.json").read_text())
    if sha256_file(LISTS / f"{SET}_files.txt") != draw["files_sha256"] or len(names) != draw["n"]:
        sys.exit(f"refused: {SET}_files.txt is not the recorded choice")
    src = Path(a.src)
    RAW.mkdir(exist_ok=True)
    rows = []
    for n in names:
        if not NAME_RE.match(n) or n[:10] not in CFG["days"]:
            sys.exit(f"refused: {n}")
        s, d = src / n, RAW / n
        if not s.is_file():
            sys.exit(f"{s}: not found (is the external drive attached?)")
        hs = sha256_file(s)
        if not d.exists():
            shutil.copy2(s, d)
        hd = sha256_file(d)
        rows.append((n, hd, "same" if hd == hs else "DIFFERENT"))
        print(n[:10], len(rows), rows[-1][2], flush=True)
    (LISTS / f"{SET}_sha256.txt").write_text("".join(f"{n}\t{h}\t{w}\n" for n, h, w in rows))
    bad = [r for r in rows if r[2] != "same"]
    print(f"copied {len(rows)} files to {RAW} (registration commit {head[:8]} on origin/main); different: {len(bad)}")
    sys.exit(1 if bad else 0)


def timing(p):
    r = subprocess.run([str(FFMPEG), "-nostdin", "-v", "error", "-i", str(p), "-map", "0:v:0", "-c", "copy", "-f", "framecrc", "-"],
                       capture_output=True, text=True)
    tb, pts = None, []
    for line in r.stdout.splitlines():
        if line.startswith("#tb 0:"):
            num, den = line.split(":", 1)[1].strip().split("/")
            tb = (int(num), int(den))
        elif line.strip() and not line.startswith("#"):
            f = [x.strip() for x in line.split(",")]
            if int(f[4]) > 0:                      # a picture = a sample of more than 0 bytes
                pts.append(int(f[2]))
    pts.sort()
    if tb is None or len(pts) < 2:
        return None
    step = sorted(b - c for b, c in zip(pts[1:], pts[:-1]))[len(pts) // 2 - 1]
    return dict(pictures=len(pts), duration=(pts[-1] - pts[0] + step) * tb[0] / tb[1])


def cmd_tags(a):
    pool, trips, trip_of = pool_and_trips()
    n_want = json.loads((LISTS / f"{SET}_draw.json").read_text())["n"]
    names = [l.split("\t")[0] for l in (LISTS / f"{SET}_sha256.txt").read_text().splitlines() if l.strip().endswith("same")]
    if len(names) != n_want:
        sys.exit(f"{len(names)} verified copies, not {n_want}")
    rows = []
    for n in sorted(names, key=lambda x: (start(x), x)):
        m = NAME_RE.match(n)
        t = trip_of[n]
        tid = start(trips[t][0]).strftime("%Y-%m-%d--%H-%M-%S")
        tm = timing(RAW / n)
        if tm is None:
            sys.exit(f"{n}: no picture timing")
        rows.append([n, f"dod_{tid}_{m.group(4)}{m.group(5)}{m.group(6)}-{m.group(7)}", f"{CFG['label']}{t + 1:02d}", tid,
                     start(n).strftime("%Y-%m-%d %H:%M:%S"), f"{tm['duration']:.2f}", tm["pictures"]])
    if len({r[1] for r in rows}) != len(rows):
        sys.exit("duplicate tags")
    out = LISTS / f"tags_{SET}.tsv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["file", "tag", "trip_label", "trip_id", "start", "duration_s", "pictures"])
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} clips, {len({r[2] for r in rows})} trips, pictures {min(r[6] for r in rows)}-{max(r[6] for r in rows)}")


def main():
    argv = sys.argv[1:]
    if "--set" in argv:
        i = argv.index("--set")
        if i + 1 >= len(argv) or argv[i + 1] not in SETS:
            sys.exit(f"--set must be one of {sorted(SETS)}")
        use_set(argv[i + 1])
        del argv[i:i + 2]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("draw", "select"):
        d = sp.add_parser(name)
        d.add_argument("--force", action="store_true")
    c = sp.add_parser("copy")
    c.add_argument("--src", required=True)
    sp.add_parser("tags")
    sp.add_parser("manifest")
    a = ap.parse_args(argv)
    if a.cmd == "manifest":
        print(f"{check_manifest()} private files equal the registered sha256")
        return
    {"draw": cmd_draw, "select": cmd_select, "copy": cmd_copy, "tags": cmd_tags}[a.cmd](a)


if __name__ == "__main__":
    main()

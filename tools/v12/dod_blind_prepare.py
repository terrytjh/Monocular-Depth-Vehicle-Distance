#!/usr/bin/env python3
"""First DOD blind test (docs/DOD_V12_BLIND_PREREG.md): the draw, the copy from the external drive and the tag table.
Private outputs only (/mnt/d/depth_work/tw_own/lists, raw_blind1); nothing here reads a picture or the overlay.

  draw   names only (the front-camera name list made 2026-10-25, name + bytes; no file is opened):
         pool   = names recorded 2026-09-26 .. 2026-10-03, sorted;
         trips  = in start-time order (from the name), a new trip when a file starts more than 150 s after the previous
                  file's start (60-s files: more than 90 s after it ended);
         draw   = random.Random(20261113).sample(pool, len(pool)) is the order; walking it, a file is taken while its trip
                  has fewer than 4 taken files, until 60 are taken.
         writes lists/blind1_files.txt (sorted names) and lists/blind1_draw.json (rule, seed, pool sha256, per-trip counts).
  copy   refuses unless docs/DOD_V12_BLIND_PREREG.md is committed and that commit is on origin/main (git fetch first).
         Copies the drawn files from the external drive folder into raw_blind1 and writes lists/blind1_sha256.txt
         (name, sha256, "same" when the copy's sha256 equals the source's).
  tags   container timing of the copies without decoding (ffmpeg -c copy -f framecrc: sample pts and sizes only) ->
         lists/tags_blind1.tsv (file, tag, trip_label, trip_id, start, duration_s, pictures); tag = dod_<trip_id>_<HHMMSS>-<NN>,
         trip_id = start of the trip's first file in the POOL (so every clip of a trip shares the drive part), labels B01...
  manifest  the private files of tools/v12/dod_blind_manifest.json against their registered sha256 (also run by copy).
usage: dod_blind_prepare.py draw | copy --src /mnt/e/front_norm | tags | manifest
"""
import argparse, csv, datetime as dt, hashlib, json, random, re, shutil, subprocess, sys
from pathlib import Path

TW = Path("/mnt/d/depth_work/tw_own")
LISTS = TW / "lists"
NAME_LIST = LISTS / "dod_front_list_20261025.txt"
RAW = TW / "raw_blind1"
REPO = Path(__file__).resolve().parents[2]
PREREG = "docs/DOD_V12_BLIND_PREREG.md"
FFMPEG = Path.home() / ".local/bin/ffmpeg"
NAME_RE = re.compile(r"^(\d{4})_(\d{2})_(\d{2})_(\d{2})(\d{2})(\d{2})_(\d{2})\.MP4$")
DAYS = ("2026_09_26", "2026_09_27", "2026_09_28", "2026_09_29", "2026_09_30", "2026_10_01", "2026_10_02", "2026_10_03")
SEED, N, CAP, GAP_S = 20261113, 60, 4, 150


def start(n):
    return dt.datetime(*map(int, NAME_RE.match(n).groups()[:6]))


def pool_and_trips():
    names = sorted(l.split()[0] for l in NAME_LIST.read_text().splitlines() if l.strip())
    pool = [n for n in names if NAME_RE.match(n) and n[:10] in DAYS]
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


def cmd_draw(a):
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
    out = LISTS / "blind1_files.txt"
    if out.exists() and not a.force:
        sys.exit(f"{out} exists (use --force only before the registration is committed)")
    out.write_text("\n".join(sorted(taken)) + "\n")
    rec = dict(rule=__doc__.split("  draw ")[1].split("  copy ")[0].strip(), seed=SEED, n=N, cap_per_trip=CAP, gap_s=GAP_S,
               name_list=str(NAME_LIST), name_list_sha256=sha256_file(NAME_LIST), pool_files=len(pool),
               pool_sha256=hashlib.sha256(("\n".join(pool) + "\n").encode()).hexdigest(), trips_in_pool=len(trips),
               trip_sizes=[len(t) for t in trips], taken_per_trip=[per.get(i, 0) for i in range(len(trips))],
               files_sha256=sha256_file(out), made=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    (LISTS / "blind1_draw.json").write_text(json.dumps(rec, indent=1))
    days = {}
    for n in taken:
        days[n[5:10]] = days.get(n[5:10], 0) + 1
    print(f"pool {len(pool)} files in {len(trips)} trips; drew {N} from {sum(1 for v in per.values() if v)} trips; per day {dict(sorted(days.items()))}")
    print(f"blind1_files.txt sha256 {rec['files_sha256']}")


def prereg_pushed():
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
    man = json.loads((REPO / "tools/v12/dod_blind_manifest.json").read_text())["files"]
    bad = [k for k, h in man.items() if not (TW / k).is_file() or sha256_file(TW / k) != h]
    if bad:
        sys.exit(f"refused: private files differ from the registered sha256: {bad}")
    return len(man)


def cmd_copy(a):
    head = prereg_pushed()
    check_manifest()
    names = [l.strip() for l in (LISTS / "blind1_files.txt").read_text().splitlines() if l.strip()]
    draw = json.loads((LISTS / "blind1_draw.json").read_text())
    if sha256_file(LISTS / "blind1_files.txt") != draw["files_sha256"] or len(names) != N:
        sys.exit("refused: blind1_files.txt is not the recorded draw")
    src = Path(a.src)
    RAW.mkdir(exist_ok=True)
    rows = []
    for n in names:
        if not NAME_RE.match(n) or n[:10] not in DAYS:
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
    (LISTS / "blind1_sha256.txt").write_text("".join(f"{n}\t{h}\t{w}\n" for n, h, w in rows))
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
    names = [l.split("\t")[0] for l in (LISTS / "blind1_sha256.txt").read_text().splitlines() if l.strip().endswith("same")]
    if len(names) != N:
        sys.exit(f"{len(names)} verified copies, not {N}")
    rows = []
    for n in sorted(names, key=lambda x: (start(x), x)):
        m = NAME_RE.match(n)
        t = trip_of[n]
        tid = start(trips[t][0]).strftime("%Y-%m-%d--%H-%M-%S")
        tm = timing(RAW / n)
        if tm is None:
            sys.exit(f"{n}: no picture timing")
        rows.append([n, f"dod_{tid}_{m.group(4)}{m.group(5)}{m.group(6)}-{m.group(7)}", f"B{t + 1:02d}", tid,
                     start(n).strftime("%Y-%m-%d %H:%M:%S"), f"{tm['duration']:.2f}", tm["pictures"]])
    if len({r[1] for r in rows}) != len(rows):
        sys.exit("duplicate tags")
    out = LISTS / "tags_blind1.tsv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["file", "tag", "trip_label", "trip_id", "start", "duration_s", "pictures"])
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} clips, {len({r[2] for r in rows})} trips, pictures {min(r[6] for r in rows)}-{max(r[6] for r in rows)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("draw")
    d.add_argument("--force", action="store_true")
    c = sp.add_parser("copy")
    c.add_argument("--src", required=True)
    sp.add_parser("tags")
    sp.add_parser("manifest")
    a = ap.parse_args()
    if a.cmd == "manifest":
        print(f"{check_manifest()} private files equal the registered sha256")
        return
    {"draw": cmd_draw, "copy": cmd_copy, "tags": cmd_tags}[a.cmd](a)


if __name__ == "__main__":
    main()

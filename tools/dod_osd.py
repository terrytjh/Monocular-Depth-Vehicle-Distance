#!/usr/bin/env python3
"""Read the GPS speed and the clock that the DOD RX800 front camera burns into its pictures.

A SCORING tool. What it writes is the reference (the answer) for ego speed on the DOD
front-camera files: no prediction or measurement program may import it, run it or read its
output folder (/mnt/d/depth_work/tw_own/osd_dev). Importing it raises ImportError on purpose.
Its outputs, its templates and the pictures it reads are private and stay on D:, never in a
repository; --out and --templates inside a repository are refused.

Development files only. A file is opened only if it sits in /mnt/d/depth_work/tw_own/raw_dev
(no link), its name is in lists/dev_front_files.txt and lists/dev_front_sha256.txt, it was
recorded on 2026-09-21, 09-23 or 09-24, and its recomputed sha256 equals the listed one.
Anything else (another folder, the external drive, another date) is refused. Reading the
first blind test is to be added only at its registration, behind a check that its predictions
are sealed.

The burned-in text (bottom left, yellow at about 32 % opacity, monospaced, 24-px cells, cell i
at x = 24 + 24 i, full resolution 1920 x 1080):
  line 1  y  920-949   model name                                  not read
  line 2  y  974-1003  YYYY-MM-DD HH:MM:SS (cells 0-18)            read: the clock
  line 3  y 1028-1057  latitude/longitude text (cells 0-21)        NEVER read
                       cell 22 empty, then 1-3 digits from cell 23 read: the speed
                       and 'km/h' right after them (cells 23-29)
All of it lies below y = 870, in the rows blacked out in every frame the method sees.

Privacy. ffmpeg crops x 24-743, y 974-1057 and paints line-3 cells 0-21 black (drawbox)
before any pixel leaves ffmpeg, so the latitude/longitude text never reaches Python. Every
frame is checked: that box must be one flat black (else the file stops and nothing is
written), and Python sets it to 0 again before anything else. Only the per-cell yellowness of
line 2 and of cells 22-29 of line 3 is used. No image is written except the validation crops
of `sample` (speed field only). ffmpeg's messages are counted, never stored.

Timing. The files hold 25 pictures per second (the container's ~30 fps counts an empty sample
after every 5 pictures). fps = round(1 / median picture interval, 4) from the packet table
(`ffmpeg -c copy -f framecrc`, no decoding); a file with any picture interval more than 10 %
off the median is `timing_irregular` (the 4 files closed at the next power-on; they lose their
last 1-2 pictures in decoding). Frame index n (0-based, decode order, `-fps_mode passthrough
-enc_time_base demux`) is picture n and t = n / fps; container timestamps are not used.

Per frame (the rules of the 2026-10-28 prototype, unchanged). Yellowness Y = min(R, G) - B
clipped at 0, one 30 x 24 patch per cell, compared with the templates by normalised
correlation (digits with +-1 px shifts). Ink fraction of a cell = share of Y > 40 in rows 2-27,
columns 2-21. Reasons a frame has no speed:
  blank            the largest ink fraction of the 7 speed cells is under 0.05
  no_unit          digit count n (1-3) = position of the best mean 'km/h' match; that mean < 0.55
  text_after_unit  a cell after the unit has ink fraction > 0.05
  low_digit        lowest digit score < 0.60 or lowest margin to the second best < 0.04
  leading_zero     n > 1 and the first digit is 0
  layout           cell 22 (must be empty; rows 1-28) has ink fraction > 0.06
The clock: the 14 digit cells of line 2 with the same templates and thresholds; an
impossible date or time is rejected.

Per file.
  Clock model: display(n) = D0 + number of ticks at frames p + 25 j <= n, voted over the tick
    phase p and D0 (the prototype's fit_clock). The clock time at frame 0 is taken as
    D0 + 1 - p / fps. Judged tick by tick: reads 0-2 frames after a tick are not judged (the
    visible change lags, and at a tens-of-seconds change the two digits update a frame or two
    apart); reads 3-12 frames after a tick that disagree before the display first agrees with
    the model are a late display update (counted per tick, not an error of the clock).
    `clock_irregular` when more than 1 % of the reads disagree otherwise, or 3 or more
    consecutive frames do.
  Line 3 absent (no GPS fix): blank stretches of 25 frames or more, or touching either end.
  Value changes: accepted frames whose value differs from the previous accepted one, at most
    3 frames apart, with no absent frame between.
  Update lattice s_k = q + P k, one per region: the update timer restarts when line 3 comes
    back after an outage (seen 2026-10-28: 13 changes before a mid-file outage fit a 25.5-frame
    lattice, the 2 after it were 16.5 frames off it), so a file is cut into regions at the start
    of each absent stretch that follows shown frames; a region = the absent stretch before a
    stretch of line 3 and that stretch (the last region also takes the absent end); only
    stretches with 5 or more accepted frames count (a yellow scene behind the empty field can
    make a few frames look non-blank). Per region:
    P searched 24.000-27.000 in 0.005 steps by the circular mean of its change frames, then
    least squares (the prototype's fit_lattice). Fewer than 5 changes (fallback): P = 25.50
    and q continued from the previous file of the trip when line 3 was shown across the file
    boundary, the two are contiguous (start gap <= 1 s) and that lattice was anchored, else the
    first change frame, else the frame where line 3 reappeared, else 0. A file without an
    outage has one region, as in the prototype.
  Slots: the lattice cells clipped to their region, numbered 0, 1, 2 ... in frame order; value =
    the most common accepted value. Any flag makes a slot invalid: line3_blank (no accepted
    frame, more than half absent), no_read, few_frames (< 5 accepted), low_agreement (< 80 % of
    the accepted frames agree), acquisition_zero (the leading slots with value 0 among the first
    5 slots after line 3 reappears, whatever follows; a file that starts with line 3 shown counts
    as a reappearance unless its contiguous predecessor ended with line 3 shown outside such a
    window), jump (> 15 km/h from the previous valid slot of the file). Slots cut by a file or
    region edge are marked partial and stay valid with 5 or more accepted frames.
  State handed to the next file of the trip: line 3 absent at the start and at the end, the
    acquisition slots still to check, the first and last lattice, the clock at frame 0. In
    --all / --trip runs every predecessor is read first; in --file runs a predecessor's state
    comes from its summary in --out when it was made by the same tool and templates, else it is
    unknown (and the file start is treated as a reappearance).

Outputs (in --out, keyed by tag):
  <tag>_osd_updates.csv   one row per update slot: slot (0, 1, 2 ...), first_frame, last_frame, frames,
      frames_accepted, frames_agree, frames_blank (line 3 absent), value_kmh (also for invalid
      slots), change_frame (first accepted frame with the slot's value when it differs from the
      previous slot's), changed, partial, valid, flags, clock_first
  <tag>_osd_seconds.csv   one row per clip second b = 0 .. floor(N / fps) - 1, the method's
      bins [round(b fps), round((b + 1) fps)): second, frame_center = round(b fps) + 12, kmh
      (the valid slot covering frame_center, as displayed, no lag), valid, reject (the slot's
      flags, or no_slot), slot, clock (the model clock at frame_center)
  <tag>_osd_summary.json  provenance (file, sha256, tool, ffmpeg, command, templates, tags
      table), counts, fps, timing, frame reasons, lattice, clock model, slot flags, valid slots
      and seconds, wall time, the state handed to the next file
  <tag>_osd_frames.csv    (--frames-csv) frame, kmh, reason, ndig, unit_score, digit_score,
      digit_margin, clock_read, clock_model, cell22_occ
  trip_continuity.json    pairs of consecutive files of a trip that are both read: clock at
      frame 0 of the next minus that of the previous minus N_previous / fps (to about one
      frame), flagged beyond 1 s; the same from the file names; where line 3 is shown across
      the boundary, the previous file's lattice carried over against the next file's fit
  Each run appends lines to /mnt/d/depth_work/tw_own/logs/osd_reads.log.

Templates (private; `read` requires --templates): digits.npy (10 x 30 x 24, '0'-'9') and
unit.npy (4 x 30 x 24, 'k' 'm' '/' 'h'), median yellowness patches. Version 1 is a byte copy
of the prototype's (digits from the clock line of 15 development files, labelled from the file
names, no speed answer used; units from the speed field). `templates` rebuilds a candidate set
from a list of files: digits from the clock (label = file-name time + o + ticks, tick phase
from the change energy of the seconds cell minus that of the static year cell, o in -3..+1 per
file by agreement with a first pass at o = -1), units = median of the 4 cells after the n
leading cells that the digit templates read as digits (n = position of the last inked cell of
the speed field + 1 - 4 must be 1-3, and every one of the n leading cells must score >= 0.60 as a
digit); it reports leave-one-day-out errors on the clock digits. A candidate replaces version 1
only after 0 differing valid slot values on the 30 prototype files and 0 clock errors
leave-one-day-out.

Validation (by someone who has not seen the reader output): `sample` draws valid slots with a
seed, stratified by day, digit count and background (textured = top 20 % by the spread of the
non-ink grey of the crop; otherwise bright = top 20 % by its median grey; else dark), one frame
per slot (the middle one), and writes speed-field crops (x 568-759, y 1022-1061; only when
cell 22 is empty) on sheets labelled S001...; the key with the reader's values goes to a
separate folder. `compare` checks a transcription (id<TAB>value, or x for unreadable) against
the key; pass = no wrong value among the samples both call readable.

  nice -n 10 python3 -B tools/dod_osd.py read --file NAME.MP4 [...] \\
      --tags /mnt/d/depth_work/tw_own/lists/tags_dev.tsv \\
      --templates /mnt/d/depth_work/tw_own/osd_templates/v1 --out /mnt/d/depth_work/tw_own/osd_dev
  nice -n 10 python3 -B tools/dod_osd.py read --all --tags ... --templates ... --out ...
  nice -n 10 python3 -B tools/dod_osd.py templates --files-list FILES.txt --out NEW_EMPTY_DIR
  nice -n 10 python3 -B tools/dod_osd.py sample --osd /mnt/d/depth_work/tw_own/osd_dev \\
      --n 200 --seed 20261029 --out NEW_EMPTY_DIR
  python3 -B tools/dod_osd.py compare --key DIR/key/key.csv --transcription FILE.tsv
"""
from __future__ import annotations

if __name__ != "__main__":
    raise ImportError("dod_osd.py reads the answer (the burned-in GPS speed); run it as a command, never import it")

import argparse
import bisect
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

TOOL_VERSION = "1 (2026-10-28)"
TOOL = Path(__file__).resolve()
REPO = TOOL.parents[1]
FORBIDDEN_TREES = (REPO, Path("/mnt/d/team_repo"))

RAW_DEV = Path("/mnt/d/depth_work/tw_own/raw_dev")
LIST_FILES = Path("/mnt/d/depth_work/tw_own/lists/dev_front_files.txt")
LIST_SHA = Path("/mnt/d/depth_work/tw_own/lists/dev_front_sha256.txt")
DEV_DAYS = ("2026_09_21", "2026_09_23", "2026_09_24")
FRAMES_ROOTS = (Path("/mnt/d/depth_work/tw_own/frames_dev"), Path("/mnt/d/depth_work/tw_own/frames_ab1080"))
READ_LOG = Path("/mnt/d/depth_work/tw_own/logs/osd_reads.log")
FFMPEG = Path.home() / ".local/bin/ffmpeg"
FFMPEG_SHA256 = "e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99"

NAME_RE = re.compile(r"^(\d{4})_(\d{2})_(\d{2})_(\d{2})(\d{2})(\d{2})_(\d{2})\.MP4$")
TAG_RE = re.compile(r"^dod_\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}_\d{6}-\d{2}$")
TAG_COLS = ("file", "tag", "trip_label", "trip_id", "start", "duration_s", "pictures")

# decoded band (full-resolution pixels): x 24-743, y 974-1057 -> line 2 = rows 0-29, line 3 = rows 54-83
BX, BY, BW, BH = 24, 974, 720, 84
CELL = 24
L2_ROW, L3_ROW = 0, 54
HIDE_W = 22 * CELL                       # line-3 cells 0-21 (latitude/longitude): crop columns 0-527
VF_READ = f"crop={BW}:{BH}:{BX}:{BY},drawbox=x=0:y={L3_ROW}:w={HIDE_W}:h=30:color=black:t=fill"
HIDDEN_MAX, HIDDEN_PTP = 20, 2           # the drawn box must be one flat dark value
# validation crop: x 552-759, y 1022-1061 decoded (cell 22 checked), x 568-759 written
SX, SY, SW, SH = 552, 1022, 208, 40
SAVE_X0 = 568 - SX
CLOCK_CELLS = (0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18)

INK = 40
UNIT_MIN, DIG_MIN, DIG_MARGIN, BLANK_MAX, LAYOUT_MAX = 0.55, 0.60, 0.04, 0.05, 0.06
BLANK_GAP = 25           # blank frames in a row = line 3 absent (no GPS fix)
CHANGE_MAX_GAP = 3
MIN_CHANGES = 5
FALLBACK_P = 25.50
SLOT_MIN_FRAMES, SLOT_MIN_AGREE = 5, 0.80
JUMP_KMH = 15
ACQ_SLOTS = 5
TICK = 25
CLOCK_LAG = 2            # frames after a tick in which any clock read is accepted
LATE_MAX = 12            # disagreeing reads up to this many frames after a tick, before the display agrees = late display
FPS_RANGE = (24.95, 25.05)
EPOCH = dt.datetime(2000, 1, 1)
FLAGS = ("line3_blank", "no_read", "few_frames", "low_agreement", "acquisition_zero", "jump")
REASONS = ("ok", "blank", "no_unit", "text_after_unit", "low_digit", "leading_zero", "layout")


# ----------------------------------------------------------------------------- small helpers
def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def refuse(msg: str):
    raise SystemExit("refused: " + msg)


def inside(p: Path, root: Path) -> bool:
    try:
        Path(os.path.realpath(p)).relative_to(os.path.realpath(root))
        return True
    except ValueError:
        return False


def guard_private_dir(p: str, what: str) -> Path:
    path = Path(p)
    s = str(path).lower()
    if s.startswith("/mnt/e") or s[:2] == "e:":
        refuse(f"{what} on the external drive")
    for root in FORBIDDEN_TREES:
        if inside(path, root):
            refuse(f"{what} inside a repository ({root}); answers and templates stay private")
    for root in FRAMES_ROOTS + (RAW_DEV,):
        if inside(path, root):
            refuse(f"{what} inside {root}, which the method or the copy step reads")
    return path


def check_ffmpeg() -> str:
    h = sha256_file(FFMPEG)
    if h != FFMPEG_SHA256:
        raise SystemExit(f"stop: {FFMPEG} sha256 {h} is not the agreed static build {FFMPEG_SHA256}")
    return h


def git_info() -> dict:
    def run(*a):
        r = subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)
        return r.returncode, r.stdout.strip()
    _, head = run("rev-parse", "HEAD")
    rc, _ = run("ls-files", "--error-unmatch", str(TOOL.relative_to(REPO)))
    _, st = run("status", "--porcelain", "--", str(TOOL.relative_to(REPO)))
    return {"repo_head": head, "tool_committed": rc == 0 and st == ""}


def write_csv(path: Path, cols, rows):
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r.get(c) is None else r[c] for c in cols])
    os.replace(tmp, path)


def write_json(path: Path, obj):
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def log_line(text: str):
    READ_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(READ_LOG, "a") as f:
        f.write(f"{now()}\t{text}\n")


def name_time(name: str) -> dt.datetime:
    m = NAME_RE.match(name)
    return dt.datetime(*(int(m.group(i)) for i in range(1, 7)))


def epoch_s(t: dt.datetime) -> int:
    return int((t - EPOCH).total_seconds())


def fmt_clock(s, with_date=True):
    if s is None:
        return ""
    t = EPOCH + dt.timedelta(seconds=int(s))
    return t.strftime("%Y-%m-%d %H:%M:%S" if with_date else "%H:%M:%S")


# ----------------------------------------------------------------------------- input checks
def load_dev_lists():
    names = {l.strip() for l in LIST_FILES.read_text().splitlines() if l.strip() and not l.startswith("#")}
    sha = {}
    for line in LIST_SHA.read_text().splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 2 and NAME_RE.match(parts[0]):
            sha[parts[0]] = parts[1].lower()
    return names, sha


def source_path(name: str, dev_names, dev_sha):
    """Path of a development file, after every check that it is one; refuses anything else."""
    s = str(name)
    if s.lower().startswith("/mnt/e") or s[:2].lower() == "e:" or "\\" in s:
        refuse(f"{s}: only /mnt/d/depth_work/tw_own/raw_dev may be read")
    p = Path(s)
    if str(p.parent) not in (".", "") and os.path.realpath(p.parent) != str(RAW_DEV):
        refuse(f"{s}: not in {RAW_DEV}")
    base = p.name
    if not NAME_RE.match(base):
        refuse(f"{s}: not a front-camera file name")
    if base[:10] not in DEV_DAYS:
        refuse(f"{s}: not recorded on a development day ({', '.join(DEV_DAYS)})")
    if base not in dev_names or base not in dev_sha:
        refuse(f"{s}: not in {LIST_FILES.name} and {LIST_SHA.name}")
    if os.path.realpath(RAW_DEV) != str(RAW_DEV):
        refuse(f"{RAW_DEV} is a link")
    path = RAW_DEV / base
    if path.is_symlink() or not path.is_file() or Path(os.path.realpath(path)).parent != RAW_DEV:
        refuse(f"{path}: not a regular file in {RAW_DEV}")
    return path, dev_sha[base]


def load_tags(path: str):
    p = Path(path)
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows or any(c not in rows[0] for c in TAG_COLS):
        raise SystemExit(f"stop: {p} lacks the columns {TAG_COLS}")
    seen_t, seen_f = set(), set()
    for r in rows:
        if not TAG_RE.match(r["tag"]) or r["tag"].count("_") != 2:
            raise SystemExit(f"stop: bad tag {r['tag']!r} in {p}")
        if r["tag"] in seen_t or r["file"] in seen_f:
            raise SystemExit(f"stop: duplicate tag or file {r['tag']} / {r['file']} in {p}")
        seen_t.add(r["tag"])
        seen_f.add(r["file"])
    return rows, sha256_file(p)


def load_templates(d: str):
    tdir = guard_private_dir(d, "--templates")
    TD = np.load(tdir / "digits.npy")
    TU = np.load(tdir / "unit.npy")
    if TD.shape != (10, 30, 24) or TU.shape != (4, 30, 24):
        raise SystemExit(f"stop: template shapes {TD.shape}, {TU.shape} (want 10x30x24 and 4x30x24)")
    sha = {"dir": str(tdir), "digits_sha256": sha256_file(tdir / "digits.npy"),
           "unit_sha256": sha256_file(tdir / "unit.npy")}
    return TD.astype(np.float32), TU.astype(np.float32), sha


# ----------------------------------------------------------------------------- container timing
def packet_table(path: Path) -> dict:
    """Picture timing from the packet list only (no decoding): the A.4 rule of the frames tool."""
    r = subprocess.run([str(FFMPEG), "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0",
                        "-c", "copy", "-f", "framecrc", "-"], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"stop: packet listing of {path.name} failed (exit {r.returncode})")
    tb = None
    pk = []
    for line in r.stdout.decode("ascii", "replace").splitlines():
        if line.startswith("#tb 0:"):
            num, den = line.split(":", 1)[1].strip().split("/")
            tb = int(num) / int(den)
        if not line.strip() or line.startswith("#"):
            continue
        f = [x.strip() for x in line.split(",")]
        pk.append((int(f[2]), int(f[4])))
    if tb is None or len(pk) < 2:
        raise SystemExit(f"stop: no packet table for {path.name}")
    pk.sort(key=lambda t: t[0])
    pts = np.array([t for t, size in pk if size > 0], float) * tb
    iv = np.diff(pts)
    med = float(np.median(iv))
    fps = round(1.0 / med, 4)
    if not FPS_RANGE[0] <= fps <= FPS_RANGE[1]:
        raise SystemExit(f"stop: {path.name} fps {fps} outside {FPS_RANGE}")
    off = int(np.sum(np.abs(iv - med) > 0.10 * med))
    span = float(pts[-1] - pts[0])
    return {"samples": len(pk), "empty_samples": sum(1 for _, s in pk if s == 0), "pictures": int(len(pts)),
            "interval_ms_median": round(med * 1000, 3), "interval_ms_min": round(float(iv.min()) * 1000, 3),
            "interval_ms_max": round(float(iv.max()) * 1000, 3), "intervals_off_median_gt_10pct": off,
            "fps": fps, "fps_span": round((len(pts) - 1) / span, 4) if span > 0 else None,
            "sample_rate_not_used": round((len(pk) - 1) / float((pk[-1][0] - pk[0][0]) * tb), 4)
            if pk[-1][0] > pk[0][0] else None,
            "timing_irregular": off > 0}


# ----------------------------------------------------------------------------- decoding
def ffmpeg_cmd(path: Path, vf: str):
    return [str(FFMPEG), "-nostdin", "-hide_banner", "-v", "error", "-threads", "4", "-i", str(path),
            "-map", "0:v:0", "-an", "-sn", "-dn", "-fps_mode", "passthrough", "-enc_time_base", "demux",
            "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]


def decode(path: Path, vf: str, w: int, h: int, on_frame):
    """Run ffmpeg, hand every frame (h x w x 3 BGR, writable, reused buffer) to on_frame(n, frame).
    Returns (frames, exit code, number of message lines, command). Messages stay in memory."""
    cmd = ffmpeg_cmd(path, vf)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err = []
    th = threading.Thread(target=lambda: err.append(p.stderr.read()), daemon=True)
    th.start()
    size = w * h * 3
    buf = bytearray(size)
    mv = memoryview(buf)
    frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    n = 0
    partial = False
    try:
        while True:
            got = 0
            while got < size:
                k = p.stdout.readinto(mv[got:])
                if not k:
                    break
                got += k
            if got == 0:
                break
            if got < size:
                partial = True
                break
            on_frame(n, frame)
            n += 1
    except BaseException:
        p.kill()
        raise
    finally:
        p.stdout.close()
        p.wait()
        th.join()
    lines = len(b"".join(err).splitlines())
    err.clear()
    if partial:
        raise SystemExit(f"stop: {path.name}: ffmpeg ended inside a frame")
    return n, p.returncode, lines, cmd


def yellowness(bgr: np.ndarray) -> np.ndarray:
    """min(R, G) - B clipped to 0..255, as float32 (grey backgrounds give about 0)."""
    b = bgr[..., 0].astype(np.int16)
    g = bgr[..., 1].astype(np.int16)
    r = bgr[..., 2].astype(np.int16)
    return np.clip(np.minimum(r, g) - b, 0, 255).astype(np.uint8).astype(np.float32)


class HiddenBox:
    """Checks that ffmpeg blacked the latitude/longitude cells, then zeroes them again."""

    def __init__(self):
        self.frames, self.vmin, self.vmax, self.ptp = 0, 255, 0, 0

    def __call__(self, frame: np.ndarray, name: str):
        box = frame[L3_ROW:L3_ROW + 30, :HIDE_W]
        lo, hi = int(box.min()), int(box.max())
        if hi > HIDDEN_MAX or hi - lo > HIDDEN_PTP:
            raise SystemExit(f"stop: {name}: the latitude/longitude cells were not blacked by ffmpeg; nothing written")
        box[...] = 0
        self.frames += 1
        self.vmin, self.vmax, self.ptp = min(self.vmin, lo), max(self.vmax, hi), max(self.ptp, hi - lo)

    def record(self):
        return {"crop_rows": [L3_ROW, L3_ROW + 29], "crop_cols": [0, HIDE_W - 1], "absolute": "x 24-551, y 1028-1057",
                "frames_checked": self.frames, "value_min": self.vmin, "value_max": self.vmax, "max_ptp": self.ptp,
                "zeroed_in_python": True, "ok": True}


def cell_patches(frame: np.ndarray):
    """Line-2 cells 0-18 and speed cells 23-29 as (19, 30, 24) and (7, 30, 24) yellowness, and cell 22's ink."""
    y2 = yellowness(frame[L2_ROW:L2_ROW + 30, :CELL * 19])
    y3 = yellowness(frame[L3_ROW:L3_ROW + 30, CELL * 22:CELL * 30])
    l2 = y2.reshape(30, 19, CELL).transpose(1, 0, 2)
    sp = y3[:, CELL:].reshape(30, 7, CELL).transpose(1, 0, 2)
    occ22 = float((y3[1:29, 2:22] > INK).mean())
    return l2, sp, occ22


# ----------------------------------------------------------------------------- classification (prototype)
def norm(a):
    a = np.asarray(a, np.float32)
    a = a - a.mean(axis=(-2, -1), keepdims=True)
    return a / (np.linalg.norm(a, axis=(-2, -1), keepdims=True) + 1e-6)


def shifted_ncc(cells, templ_n, shifts=(-1, 0, 1)):
    """cells (K, 30, 24) raw; templ_n (D, 30, 24) normalised. Max NCC over small x/y shifts of the cell."""
    best = None
    for dy in shifts:
        for dx in shifts:
            c = np.roll(np.roll(cells, dy, axis=1), dx, axis=2)
            s = np.einsum("kij,dij->kd", norm(c), templ_n)
            best = s if best is None else np.maximum(best, s)
    return best


def read_speed_cells(cells, TDn, TUn):
    """cells: (7, 30, 24) yellowness of line-3 cells 23..29 -> value (int or None) and diagnostics."""
    cells = np.asarray(cells, np.float32)
    occ = (cells[:, 2:28, 2:22] > INK).mean((1, 2))
    res = {"value": None, "ndig": None, "unit": 0.0, "dscore": None, "dmargin": None, "reason": ""}
    if occ.max() < BLANK_MAX:
        res["reason"] = "blank"
        return res
    us = []
    for s in (1, 2, 3):          # number of digits -> unit cells s..s+3
        if s + 4 > 7:
            us.append(-1)
            continue
        sc = shifted_ncc(cells[s:s + 4], TUn, shifts=(0,))
        us.append(float(np.mean([sc[j, j] for j in range(4)])))
    nd = int(np.argmax(us)) + 1
    res["unit"], res["ndig"] = us[nd - 1], nd
    if us[nd - 1] < UNIT_MIN:
        res["reason"] = "no_unit"
        return res
    after = occ[nd + 4:] if nd + 4 < 7 else np.array([])
    if after.size and after.max() > BLANK_MAX:
        res["reason"] = "text_after_unit"
        return res
    sc = shifted_ncc(cells[:nd], TDn)
    srt = np.sort(sc, 1)
    dig = sc.argmax(1)
    res["dscore"] = float(srt[:, -1].min())
    res["dmargin"] = float((srt[:, -1] - srt[:, -2]).min())
    if res["dscore"] < DIG_MIN or res["dmargin"] < DIG_MARGIN:
        res["reason"] = "low_digit"
        return res
    if nd > 1 and dig[0] == 0:
        res["reason"] = "leading_zero"
        return res
    res["value"] = int("".join(map(str, dig)))
    res["reason"] = "ok"
    return res


def read_clock_cells(cells, TDn):
    """cells: (19, 30, 24) of line 2 -> clock as seconds since 2000-01-01 (or None), lowest score, margin."""
    sc = shifted_ncc(np.asarray(cells, np.float32)[list(CLOCK_CELLS)], TDn)
    srt = np.sort(sc, 1)
    d = sc.argmax(1)
    smin = float(srt[:, -1].min())
    mmin = float((srt[:, -1] - srt[:, -2]).min())
    if smin < DIG_MIN or mmin < DIG_MARGIN:
        return None, smin, mmin
    v = [int("".join(map(str, d[a:b]))) for a, b in ((0, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14))]
    Y, M, D, h, m, s = v
    if not (1 <= M <= 12 and 1 <= D <= 31 and h <= 23 and m <= 59 and s <= 59):
        return None, smin, mmin
    try:
        t = dt.datetime(Y, M, D, h, m, s)
    except ValueError:
        return None, smin, mmin
    return epoch_s(t), smin, mmin


# ----------------------------------------------------------------------------- clock model
def fit_clock(reads, n):
    """display(i) = D0 + number of ticks at frames p + 25 j (j >= 0) not after i; vote over (p, D0)."""
    idx = [i for i, c in enumerate(reads) if c is not None]
    if len(idx) < 10:
        return None
    best = None
    for p in range(TICK):
        votes = {}
        for i in idx:
            k = (i - p) // TICK + 1 if i >= p else 0
            t0 = reads[i] - k
            votes[t0] = votes.get(t0, 0) + 1
        t0, v = max(votes.items(), key=lambda kv: kv[1])
        if best is None or v > best[2]:
            best = (p, t0, v)
    p, d0, v = best
    model = [d0 + ((i - p) // TICK + 1 if i >= p else 0) for i in range(n)]
    return {"p": p, "d0": d0, "agree_vote": v, "reads": len(idx), "model": model}


def clock_check(reads, cl):
    """Reads against the tick model, tick by tick (block = the 25 frames from a model tick). Not counted as
    disagreeing: any read 0-2 frames after a tick (the visible change lags; at a tens-of-seconds change the two digits
    update a frame or two apart), and reads 3-12 frames after a tick that disagree before the first agreeing read of
    the block (a late display update, counted per tick). Irregular: more than 1 % of the reads disagree otherwise, or
    3 or more consecutive frames do."""
    p, model, n = cl["p"], cl["model"], len(reads)
    agree = lagw = late = off = 0
    late_ticks, late_max = 0, 0
    offm = [False] * n
    starts = ([0] if p > 0 else []) + list(range(p, n, TICK))
    for s0 in starts:
        is_tick = s0 >= p
        e0 = min(n, s0 + TICK) if is_tick else p
        first_agree = next((i - s0 for i in range(s0, e0) if reads[i] is not None and reads[i] == model[i]), None)
        block_late = 0
        for i in range(s0, e0):
            c = reads[i]
            if c is None:
                continue
            ph = i - s0
            if c == model[i]:
                agree += 1
            elif is_tick and ph <= CLOCK_LAG:
                lagw += 1
            elif is_tick and ph <= LATE_MAX and (first_agree is None or ph < first_agree):
                late += 1
                block_late = ph + 1
            else:
                off += 1
                offm[i] = True
        if block_late:
            late_ticks += 1
            late_max = max(late_max, block_late)
    runs3, run = 0, 0
    for x in offm + [False]:
        if x:
            run += 1
        else:
            runs3 += run >= 3
            run = 0
    nreads = agree + lagw + late + off
    return {"reads": nreads, "agree": agree, "lag_window_reads": lagw, "late_display_reads": late,
            "late_display_ticks": late_ticks, "late_display_max_frames": late_max,
            "off_model_reads": off, "off_model_runs_3": int(runs3),
            "clock_irregular": bool(runs3 > 0 or off > 0.01 * nreads)}


# ----------------------------------------------------------------------------- speed: lattice and slots
def absent_mask(isblank):
    """Blank stretches of >= BLANK_GAP frames, or touching the file start or end = line 3 absent."""
    n = len(isblank)
    out = [False] * n
    i = 0
    while i < n:
        if isblank[i]:
            j = i
            while j < n and isblank[j]:
                j += 1
            if j - i >= BLANK_GAP or i == 0 or j == n:
                for t in range(i, j):
                    out[t] = True
            i = j
        else:
            i += 1
    return out


def change_frames(vals, absent):
    acc = [i for i, v in enumerate(vals) if v is not None]
    return [acc[t] for t in range(1, len(acc))
            if vals[acc[t]] != vals[acc[t - 1]] and acc[t] - acc[t - 1] <= CHANGE_MAX_GAP
            and not any(absent[acc[t - 1]:acc[t]])]


def fit_lattice(change):
    """s_k = q + P k fitted to the change frames (circular-mean start, then least squares): P, q, residual SD."""
    ch = np.asarray(change, float)
    best = max(((P, abs(np.mean(np.exp(2j * np.pi * ch / P)))) for P in np.arange(24.0, 27.0, 0.005)),
               key=lambda t: t[1])
    mu = np.angle(np.mean(np.exp(2j * np.pi * ch / best[0]))) / (2 * np.pi)
    k = np.round((ch - mu * best[0]) / best[0])
    A = np.vstack([k, np.ones_like(k)]).T
    sol = np.linalg.lstsq(A, ch, rcond=None)[0]
    P, q = float(sol[0]), float(sol[1])
    return P, q % P, float(np.std(ch - A @ sol))


def shown_segments(absent):
    """Stretches of frames with line 3 shown: [(first, last), ...]."""
    segs, i, n = [], 0, len(absent)
    while i < n:
        if absent[i]:
            i += 1
            continue
        j = i
        while j < n and not absent[j]:
            j += 1
        segs.append((i, j - 1))
        i = j
    return segs


def make_slots(vals, absent, P, q, lo, hi):
    """Slots of the lattice q + P k (file frames) over the frames [lo, hi), clipped to it."""
    out = []
    k = int(np.floor((lo - q) / P))
    while q + P * k < hi:
        a = max(lo, int(np.ceil(q + P * k)))
        b = min(hi, int(np.ceil(q + P * (k + 1))))
        if b <= a:
            k += 1
            continue
        acc = [vals[i] for i in range(a, b) if vals[i] is not None]
        nb = sum(1 for i in range(a, b) if absent[i])
        row = {"first_frame": a, "last_frame": b - 1, "frames": b - a, "frames_accepted": len(acc),
               "frames_agree": 0, "frames_blank": nb, "value": None,
               "partial": (q + P * k < lo) or (q + P * (k + 1) > hi), "flags": []}
        if acc:
            u, c = np.unique(acc, return_counts=True)
            row["value"] = int(u[np.argmax(c)])
            row["frames_agree"] = int(c.max())
            if row["frames_agree"] < SLOT_MIN_AGREE * len(acc):
                row["flags"].append("low_agreement")
            if len(acc) < SLOT_MIN_FRAMES:
                row["flags"].append("few_frames")
        elif nb > (b - a) / 2:
            row["flags"].append("line3_blank")
        else:
            row["flags"].append("no_read")
        out.append(row)
        k += 1
    return out


def flag_acquisition(slots, absent, start_window, ends_shown):
    """acquisition_zero on the leading slots with value 0 among the first ACQ_SLOTS slots after line 3 reappears
    (slots without a value are skipped; the first slot with another value ends the run), whatever follows.
    start_window = slots to check from the file start (a reappearance at frame 0, or a window carried over).
    Returns the windows and the number of slots still to check in the next file."""
    n = len(absent)
    starts = [s["first_frame"] for s in slots]
    windows = []
    if n and slots and not absent[0] and start_window > 0:
        windows.append((0, start_window))
    for r in range(1, n):
        if absent[r - 1] and not absent[r]:
            windows.append((bisect.bisect_right(starts, r) - 1, ACQ_SLOTS))
    for j0, cnt in windows:
        for j in range(j0, min(len(slots), j0 + cnt)):
            v = slots[j]["value"]
            if v is None:
                continue
            if v != 0:
                break
            if "acquisition_zero" not in slots[j]["flags"]:
                slots[j]["flags"].append("acquisition_zero")
    left = 0
    if windows and ends_shown:
        j0, cnt = windows[-1]
        tail = slots[j0:]
        if all(t["value"] in (None, 0) for t in tail):         # the run of leading zeros may go on in the next file
            used = len(tail) - (1 if slots[-1]["partial"] else 0)
            left = max(0, cnt - used)
    return windows, left


def flag_jumps(slots):
    prev = None
    for r in slots:
        if r["value"] is not None and not r["flags"]:
            if prev is not None and abs(r["value"] - prev) > JUMP_KMH:
                r["flags"].append("jump")
            prev = r["value"]


# ----------------------------------------------------------------------------- one file
def read_one(row, path, sha_listed, TDn, TUn, tsha, tags_info, out, frames_root, frames_csv, pstate, pinfo, run):
    t_all = time.time()
    name, tag = row["file"], row["tag"]
    t0 = time.time()
    sha = sha256_file(path)
    if sha != sha_listed:
        raise SystemExit(f"stop: {name} sha256 differs from {LIST_SHA}")
    t_sha = time.time() - t0
    t0 = time.time()
    pk = packet_table(path)
    t_pk = time.time() - t0
    if str(pk["pictures"]) != str(row["pictures"]).strip():
        raise SystemExit(f"stop: {name}: {pk['pictures']} pictures, the tags table says {row['pictures']}")
    fps = pk["fps"]

    hidden = HiddenBox()
    res = []

    def on_frame(i, frame):
        hidden(frame, name)
        l2, sp, occ22 = cell_patches(frame)
        r = read_speed_cells(sp, TDn, TUn)
        c, cs, cm = read_clock_cells(l2, TDn)
        if r["value"] is not None and occ22 > LAYOUT_MAX:
            r["value"], r["reason"] = None, "layout"
        res.append((r["value"], r["reason"], r["ndig"], r["unit"], r["dscore"], r["dmargin"], c, occ22))

    t0 = time.time()
    n, rc, nmsg, cmd = decode(path, VF_READ, BW, BH, on_frame)
    t_dec = time.time() - t0
    if rc != 0:
        raise SystemExit(f"stop: {name}: ffmpeg exit {rc}")
    tail = pk["pictures"] - n
    if not (tail == 0 or (pk["timing_irregular"] and 1 <= tail <= 2)):
        raise SystemExit(f"stop: {name}: {n} frames decoded, {pk['pictures']} pictures"
                         f" (timing_irregular={pk['timing_irregular']})")
    mask = {"path": str(Path(frames_root) / tag / "mask.json"), "present": False}
    mp = Path(mask["path"])
    if mp.is_file():
        try:
            mj = json.loads(mp.read_text())
        except ValueError:
            mj = None
        if mj is None:
            mask.update(present=True, readable=False, match=None)
        else:
            mcount = (mj.get("frames") or {}).get("count")
            mfps = (mj.get("container") or {}).get("fps", (mj.get("for_run_list") or {}).get("fps"))
            match = mcount == n and (mfps is None or abs(float(mfps) - fps) < 1e-9)
            mask.update(present=True, readable=True, frames_count=mcount, fps=mfps, checks_ok=mj.get("checks_ok"),
                        match=match)
            if not match and mj.get("checks_ok") is True:     # a record the method will use must agree
                raise SystemExit(f"stop: {name}: {n} frames at {fps} fps here, mask.json says {mcount} at {mfps}")

    vals = [r[0] for r in res]
    reasons = [r[1] for r in res]
    reads = [r[6] for r in res]

    # clock
    cl = fit_clock(reads, n)
    tname = epoch_s(name_time(name))
    if cl:
        cc = clock_check(reads, cl)
        t0_clock = cl["d0"] + 1 - cl["p"] / fps
        clock = {"tick_phase_frame": cl["p"], "display_before_first_tick": fmt_clock(cl["d0"]),
                 "display_at_frame0": fmt_clock(cl["model"][0]),
                 "frame0_display_minus_filename_s": int(cl["model"][0] - tname),
                 "frame0_estimate_minus_filename_s": round(t0_clock - tname, 3), **cc}
        model = cl["model"]
    else:
        t0_clock, model = None, None
        clock = {"tick_phase_frame": None, "reads": sum(c is not None for c in reads), "clock_irregular": True,
                 "note": "fewer than 10 clock reads"}
    clock_ok = cl is not None and not clock["clock_irregular"]

    # predecessor: contiguity, carried lattice and acquisition window
    pred = dict(pinfo)
    contiguous, gap_frames = False, 0
    if pstate is not None:
        name_delta = tname - (pstate["start_name_s"] + pstate["frames"] / pstate["fps"])
        if clock_ok and pstate.get("t0_clock") is not None and pstate.get("clock_ok"):
            basis, delta = "clock", t0_clock - (pstate["t0_clock"] + pstate["frames"] / pstate["fps"])
        else:
            basis, delta = "file names", name_delta
        contiguous = abs(delta) <= 1.0
        gap_frames = int(round(delta * fps)) if basis == "clock" and contiguous else 0
        pred.update(contiguous=contiguous, basis=basis, gap_s=round(delta, 3), name_gap_s=round(name_delta, 3),
                    ended_absent=pstate["ended_absent"], acq_slots_left=pstate["acq_slots_left"])
    if pstate is not None and contiguous and not pstate["ended_absent"]:
        start_window, start_kind = pstate["acq_slots_left"], "carried window" if pstate["acq_slots_left"] else "none"
    else:
        start_window = ACQ_SLOTS
        start_kind = ("predecessor ended with line 3 absent" if pstate is not None and contiguous
                      else "predecessor not contiguous" if pstate is not None
                      else "predecessor " + pinfo.get("state", "unknown"))
    pred["file_start_acquisition_window"] = {"slots": start_window, "why": start_kind}

    # speed: one update lattice per stretch of line 3 (the update timer restarts when the GPS comes back)
    absent = absent_mask([x == "blank" for x in reasons])
    ch = change_frames(vals, absent)
    segs = [g for g in shown_segments(absent)            # stretches with real text (yellow scene alone gives none)
            if sum(vals[i] is not None for i in range(g[0], g[1] + 1)) >= SLOT_MIN_FRAMES]
    bounds = [0] + [segs[j - 1][1] + 1 for j in range(1, len(segs))] + [n]
    starts_shown = bool(segs) and segs[0][0] == 0
    ends_shown = bool(segs) and segs[-1][1] == n - 1
    continuous = bool(pstate is not None and contiguous and not pstate["ended_absent"] and starts_shown)
    regions, slots = [], []
    for j in range(max(1, len(segs))):
        lo, hi = bounds[j], bounds[j + 1]
        seg = segs[j] if segs else None
        rch = [c for c in ch if seg and seg[0] <= c <= seg[1]]
        carried_from = None
        if len(rch) >= MIN_CHANGES:
            P, q, sd = fit_lattice(rch)
            anchor = "fit"
        else:
            P, sd = FALLBACK_P, None
            lat = (pstate or {}).get("lattice") or {}
            if j == 0 and continuous and lat.get("anchor") not in (None, "none"):
                q = (lat["q"] - pstate["frames"] - gap_frames) % lat["P"]
                anchor, carried_from = "carried", pinfo.get("tag")
            elif rch:
                q, anchor = float(rch[0]) % P, "first_change"
            elif seg and seg[0] > 0:
                q, anchor = float(seg[0]) % P, "reappearance"
            else:
                q, anchor = 0.0, "none"
        regions.append({"first_frame": lo, "last_frame": hi - 1, "shown": list(seg) if seg else None,
                        "period_frames": round(P, 4), "phase_frame": round(q, 3), "changes": len(rch),
                        "residual_sd_frames": None if sd is None else round(sd, 3), "fallback": anchor != "fit",
                        "anchor": anchor, "carried_from": carried_from, "_P": P, "_q": q})
        slots += make_slots(vals, absent, P, q, lo, hi)
    for i, s_ in enumerate(slots):
        s_["slot"] = i
    windows, acq_left = flag_acquisition(slots, absent, start_window, ends_shown)
    pred["file_start_acquisition_window"]["applied"] = bool(n and slots and not absent[0] and start_window > 0)
    flag_jumps(slots)
    prev_v = None
    for j, s in enumerate(slots):
        s["valid"] = s["value"] is not None and not s["flags"]
        s["changed"] = int(s["value"] is not None and prev_v is not None and s["value"] != prev_v)
        s["change_frame"] = (next(i for i in range(s["first_frame"], s["last_frame"] + 1) if vals[i] == s["value"])
                             if s["changed"] else None)
        prev_v = s["value"]

    up_rows = [{"slot": s["slot"], "first_frame": s["first_frame"], "last_frame": s["last_frame"],
                "frames": s["frames"], "frames_accepted": s["frames_accepted"], "frames_agree": s["frames_agree"],
                "frames_blank": s["frames_blank"], "value_kmh": s["value"], "change_frame": s["change_frame"],
                "changed": s["changed"], "partial": int(s["partial"]), "valid": int(s["valid"]),
                "flags": "|".join(s["flags"]), "clock_first": fmt_clock(model[s["first_frame"]], False) if model else ""}
               for s in slots]
    starts = [s["first_frame"] for s in slots]
    sec_rows = []
    for b in range(int(math.floor(n / fps))):
        c = int(round(b * fps)) + 12
        j = bisect.bisect_right(starts, c) - 1
        s = slots[j] if 0 <= j < len(slots) and slots[j]["first_frame"] <= c <= slots[j]["last_frame"] else None
        sec_rows.append({"second": b, "frame_center": c,
                         "kmh": s["value"] if s and s["valid"] else None,
                         "valid": int(bool(s and s["valid"])),
                         "reject": "" if s and s["valid"] else ("|".join(s["flags"]) if s else "no_slot"),
                         "slot": s["slot"] if s else None,
                         "clock": fmt_clock(model[c], False) if model and c < n else ""})

    files = {"updates": f"{tag}_osd_updates.csv", "seconds": f"{tag}_osd_seconds.csv"}
    write_csv(out / files["updates"], ["slot", "first_frame", "last_frame", "frames", "frames_accepted", "frames_agree",
                                       "frames_blank", "value_kmh", "change_frame", "changed", "partial", "valid",
                                       "flags", "clock_first"], up_rows)
    write_csv(out / files["seconds"], ["second", "frame_center", "kmh", "valid", "reject", "slot", "clock"], sec_rows)
    if frames_csv:
        files["frames"] = f"{tag}_osd_frames.csv"
        fr_rows = [{"frame": i, "kmh": r[0], "reason": r[1], "ndig": r[2], "unit_score": round(r[3], 3),
                    "digit_score": None if r[4] is None else round(r[4], 3),
                    "digit_margin": None if r[5] is None else round(r[5], 3),
                    "clock_read": fmt_clock(r[6]), "clock_model": fmt_clock(model[i]) if model else "",
                    "cell22_occ": round(r[7], 3)} for i, r in enumerate(res)]
        write_csv(out / files["frames"], ["frame", "kmh", "reason", "ndig", "unit_score", "digit_score", "digit_margin",
                                          "clock_read", "clock_model", "cell22_occ"], fr_rows)

    rcount = {k: reasons.count(k) for k in REASONS if reasons.count(k)}
    accepted = sum(v is not None for v in vals)
    by_ndig = {str(d): sum(1 for r in res if r[0] is not None and r[2] == d) for d in (1, 2, 3)}
    carry = {"tag": tag, "frames": n, "fps": fps, "start_name_s": tname, "t0_clock": t0_clock, "clock_ok": clock_ok,
             "ended_absent": not ends_shown, "acq_slots_left": int(acq_left), "starts_absent": not starts_shown,
             "lattice": {"P": regions[-1]["_P"], "q": regions[-1]["_q"], "anchor": regions[-1]["anchor"]},
             "lattice_first": {"P": regions[0]["_P"], "q": regions[0]["_q"], "anchor": regions[0]["anchor"]}}
    summ = {"tool": "tools/dod_osd.py", "tool_version": TOOL_VERSION, "tool_sha256": run["tool_sha256"],
            **run["git"], "made": now(),
            "file": name, "tag": tag, "trip_label": row["trip_label"], "trip_id": row["trip_id"],
            "source": {"path": str(path), "bytes": path.stat().st_size, "sha256": sha_listed,
                       "sha256_list": str(LIST_SHA), "sha256_rechecked": True},
            "tags_table": tags_info, "templates": tsha,
            "ffmpeg": {"path": str(FFMPEG), "sha256": run["ffmpeg_sha256"], "command": cmd, "exit": rc,
                       "message_lines": nmsg},
            "container": {k: v for k, v in pk.items() if k != "timing_irregular"},
            "timing_irregular": pk["timing_irregular"], "frames": n, "pictures": pk["pictures"], "tail_missing": tail,
            "fps": fps, "time_rule": "t = frame index / fps; container timestamps not used",
            "mask_json": mask, "hidden_box": hidden.record(),
            "frame_reasons": rcount, "frames_line3_absent": int(sum(absent)), "frames_accepted": accepted,
            "frames_accepted_by_digits": by_ndig,
            "lattice": {"regions": [{k: v for k, v in r.items() if not k.startswith("_")} for r in regions],
                        "changes": len(ch), "fallback": any(r["fallback"] for r in regions),
                        "anchors": "+".join(r["anchor"] for r in regions)},
            "clock": clock, "predecessor": pred,
            "acquisition_windows": [{"first_slot": slots[j0]["slot"], "slots": cnt} for j0, cnt in windows if slots],
            "update_slot_numbering": "0, 1, 2 ... in frame order (per file)",
            "slots": len(slots), "slots_valid": sum(s["valid"] for s in slots),
            "slot_flags": {f: sum(1 for s in slots if f in s["flags"]) for f in FLAGS},
            "seconds": len(sec_rows), "seconds_valid": sum(r["valid"] for r in sec_rows),
            "seconds_reject": {k: v for k, v in sorted(_count(r["reject"] for r in sec_rows if not r["valid"]).items())},
            "outputs": files, "carry": carry,
            "wall_s": {"sha256": round(t_sha, 2), "packets": round(t_pk, 2), "decode_and_read": round(t_dec, 2),
                       "total": round(time.time() - t_all, 2)},
            "complete": True}
    write_json(out / f"{tag}_osd_summary.json", summ)
    return summ


def _count(it):
    d = {}
    for x in it:
        d[x] = d.get(x, 0) + 1
    return d


# ----------------------------------------------------------------------------- read
def cmd_read(a):
    ffsha = check_ffmpeg()
    out = guard_private_dir(a.out, "--out")
    TD, TU, tsha = load_templates(a.templates)
    TDn, TUn = norm(TD), norm(TU)
    rows, tags_sha = load_tags(a.tags)
    tags_info = {"path": str(Path(a.tags).resolve()), "sha256": tags_sha}
    dev_names, dev_sha = load_dev_lists()
    pred, last = {}, {}
    for r in rows:
        pred[r["tag"]] = last.get(r["trip_label"])
        last[r["trip_label"]] = r
    if a.all:
        sel = list(rows)
    elif a.trip:
        labels = set(a.trip)
        missing = labels - {r["trip_label"] for r in rows}
        if missing:
            raise SystemExit(f"stop: trips not in the tags table: {sorted(missing)}")
        sel = [r for r in rows if r["trip_label"] in labels]
    else:
        by_file = {r["file"]: r for r in rows}
        want = []
        for f in a.file:
            base = source_path(f, dev_names, dev_sha)[0].name      # refuses other folders, other days, links
            if base not in by_file:
                refuse(f"{f}: not in the tags table")
            want.append(base)
        sel = [r for r in rows if r["file"] in set(want)]
    checked = [(r, *source_path(r["file"], dev_names, dev_sha)) for r in sel]   # every check before any reading
    out.mkdir(parents=True, exist_ok=True)
    run = {"tool_sha256": sha256_file(TOOL), "git": git_info(), "ffmpeg_sha256": ffsha}
    log_line(f"read start\tfiles={len(checked)}\tout={out}\ttemplates={tsha['dir']}\ttool_sha256={run['tool_sha256'][:16]}"
             f"\t{','.join(r['file'] for r, _, _ in checked)}")
    states, done, reread_trips = {}, [], set()

    def same_build(S):
        return bool(S and S.get("complete") and S.get("tool_sha256") == run["tool_sha256"] and S.get("templates") == tsha)

    def summary_of(tag):
        sp = out / f"{tag}_osd_summary.json"
        return json.loads(sp.read_text()) if sp.is_file() else None

    try:
        for r, path, sha_listed in checked:
            tag = r["tag"]
            p = pred[tag]
            if not a.redo and r["trip_label"] not in reread_trips:
                S = summary_of(tag)
                if same_build(S) and S.get("source", {}).get("sha256") == sha_listed:
                    was_unknown = str(S.get("predecessor", {}).get("state", "")).startswith("unknown")
                    known_now = p is not None and (p["tag"] in states or same_build(summary_of(p["tag"])))
                    if not (was_unknown and known_now):
                        states[tag] = S["carry"]
                        print(f"{tag}: up to date, skipped")
                        continue
            pstate, pinfo = None, {"tag": p["tag"] if p else None}
            if p is None:
                pinfo["state"] = "none (first file of the trip)"
            elif p["tag"] in states:
                pstate, pinfo["state"] = states[p["tag"]], "read in this run"
            else:
                PS = summary_of(p["tag"])
                if same_build(PS):
                    pstate, pinfo["state"] = PS["carry"], "from its summary"
                else:
                    pinfo["state"] = "unknown (not read)"
            S = read_one(r, path, sha_listed, TDn, TUn, tsha, tags_info, out, a.frames_root, a.frames_csv, pstate,
                         pinfo, run)
            states[tag] = S["carry"]
            done.append(tag)
            reread_trips.add(r["trip_label"])
            print(f"{tag}: frames {S['frames']} fps {S['fps']} absent {S['frames_line3_absent']} accepted "
                  f"{S['frames_accepted']} slots {S['slots']} valid {S['slots_valid']} seconds valid "
                  f"{S['seconds_valid']}/{S['seconds']} lattice {S['lattice']['anchors']} "
                  f"clock_irregular {S['clock']['clock_irregular']} {S['wall_s']['total']} s", flush=True)
    except BaseException as e:
        log_line(f"read stopped\tdone={len(done)}\t{type(e).__name__}: {str(e)[:200]}")
        raise
    write_continuity(out, rows, pred, states, run)
    log_line(f"read done\tfiles={len(done)} read, {len(checked) - len(done)} up to date\tout={out}")


def write_continuity(out, rows, pred, states, run):
    path = out / "trip_continuity.json"
    old = json.loads(path.read_text()) if path.is_file() else {}
    pairs = {(x["prev"], x["next"]): x for x in old.get("pairs", [])}
    for r in rows:
        p = pred[r["tag"]]
        if p is None or r["tag"] not in states or p["tag"] not in states:
            continue
        a, b = states[p["tag"]], states[r["tag"]]
        step = a["frames"] / a["fps"]
        rec = {"trip": r["trip_label"], "prev": p["tag"], "next": r["tag"], "prev_frames": a["frames"],
               "prev_fps": a["fps"], "name_delta_s": round(b["start_name_s"] - a["start_name_s"] - step, 3)}
        if a.get("t0_clock") is not None and b.get("t0_clock") is not None:
            d = b["t0_clock"] - a["t0_clock"] - step
            rec.update(clock_delta_s=round(d, 3), clock_ok_both=bool(a.get("clock_ok") and b.get("clock_ok")),
                       flag_gt_1s=bool(abs(d) > 1.0))
        else:
            rec.update(clock_delta_s=None, clock_ok_both=False, flag_gt_1s=None)
        la, lb = a["lattice"], b.get("lattice_first", b["lattice"])
        rec["lattice"] = {"prev_anchor": la["anchor"], "next_anchor": lb["anchor"],
                          "line3_continuous": bool(not a.get("ended_absent") and not b.get("starts_absent", True))}
        if la["anchor"] == "fit" and lb["anchor"] == "fit" and rec.get("clock_delta_s") is not None \
                and abs(rec["clock_delta_s"]) <= 1.0 and rec["lattice"]["line3_continuous"]:
            proj = (la["q"] - a["frames"] - round(rec["clock_delta_s"] * b["fps"])) % la["P"]
            e = (proj - lb["q"] + lb["P"] / 2) % lb["P"] - lb["P"] / 2
            rec["lattice"]["projection_error_frames"] = round(float(e), 2)
        pairs[(p["tag"], r["tag"])] = rec
    if not pairs:
        return
    write_json(path, {"tool_sha256": run["tool_sha256"], "updated": now(),
                      "rule": "clock_delta_s = clock at frame 0 of next - clock at frame 0 of prev - N_prev / fps, with "
                              "clock at frame 0 = D0 + 1 - p / fps; flagged when |value| > 1 s",
                      "pairs": sorted(pairs.values(), key=lambda x: x["next"])})


# ----------------------------------------------------------------------------- templates (rebuild)
def cmd_templates(a):
    ffsha = check_ffmpeg()
    out = guard_private_dir(a.out, "--out")
    if out.exists() and any(out.iterdir()):
        refuse(f"{out} exists and is not empty (version 1 is never overwritten)")
    names = [l.strip() for l in open(a.files_list) if l.strip() and not l.startswith("#")]
    dev_names, dev_sha = load_dev_lists()
    paths = [(Path(x).name, *source_path(x, dev_names, dev_sha)) for x in names]
    log_line(f"templates start\tfiles={len(paths)}\tout={out}")
    data = {}
    for name, path, sha_listed in paths:
        if sha256_file(path) != sha_listed:
            raise SystemExit(f"stop: {name} sha256 differs from the list")
        pk = packet_table(path)
        L2, SP = [], []
        hidden = HiddenBox()

        def on_frame(i, frame):
            hidden(frame, name)
            L2.append(yellowness(frame[L2_ROW:L2_ROW + 30, :CELL * 19]).astype(np.uint8))
            SP.append(yellowness(frame[L3_ROW:L3_ROW + 30, CELL * 23:CELL * 30]).astype(np.uint8))

        n, rc, _, _ = decode(path, VF_READ, BW, BH, on_frame)
        tail = pk["pictures"] - n
        if rc != 0 or not (tail == 0 or (pk["timing_irregular"] and 1 <= tail <= 2)):
            raise SystemExit(f"stop: {name}: exit {rc}, {n} frames of {pk['pictures']}")
        data[name[:20]] = (np.stack(L2), np.stack(SP))
        print(f"{name}: {n} frames decoded", flush=True)

    def phase_of(L2):
        def dcell(i):
            c = L2[:, :, i * CELL:(i + 1) * CELL].astype(np.float32)
            return np.r_[0, np.abs(np.diff(c, axis=0)).mean((1, 2))]
        g = dcell(18) - dcell(0)
        return int(np.argmax([np.mean(g[p::TICK]) for p in range(TICK)]))

    phases = {k: phase_of(v[0]) for k, v in data.items()}

    def examples(key, o):
        L2 = data[key][0]
        n, p = len(L2), phases[key]
        T0 = name_time(key + ".MP4")
        X, y = [], []
        for m in range(p - 13, n, TICK):
            if m < 0:
                continue
            k = (m - p) // TICK + 1 if m >= p else 0
            s = (T0 + dt.timedelta(seconds=o + k)).strftime("%Y-%m-%d %H:%M:%S")
            for i in CLOCK_CELLS:
                X.append(L2[m, :, i * CELL:(i + 1) * CELL])
                y.append(int(s[i]))
        return np.array(X, np.float32), np.array(y)

    ex = {k: examples(k, -1) for k in data}
    X = np.concatenate([e[0] for e in ex.values()])
    y = np.concatenate([e[1] for e in ex.values()])
    Tn = norm(np.array([np.median(X[y == d], 0) for d in range(10)]))
    offs = {}
    for k in data:
        best = None
        for o in (-3, -2, -1, 0, 1):
            Xo, yo = examples(k, o)
            acc = float((np.einsum("nij,dij->nd", norm(Xo), Tn).argmax(1) == yo).mean())
            if best is None or acc > best[1]:
                best = (o, acc)
        offs[k] = best
        ex[k] = examples(k, best[0])
    X = np.concatenate([e[0] for e in ex.values()])
    y = np.concatenate([e[1] for e in ex.values()])
    days = np.concatenate([[k[:10]] * len(e[1]) for k, e in ex.items()])
    lodo = {}
    for d in sorted(set(days)):
        tr = days != d
        Td = norm(np.array([np.median(X[tr & (y == q)], 0) for q in range(10)]))
        s = shifted_ncc(X[~tr], Td)
        srt = np.sort(s, 1)
        lodo[d] = {"examples": int((~tr).sum()), "errors": int((s.argmax(1) != y[~tr]).sum()),
                   "top_score_pct_0.1_1_50": np.percentile(srt[:, -1], [0.1, 1, 50]).round(3).tolist(),
                   "margin_pct_0.1_1_50": np.percentile(srt[:, -1] - srt[:, -2], [0.1, 1, 50]).round(3).tolist()}
    TD = np.array([np.median(X[y == q], 0) for q in range(10)]).astype(np.float32)
    TDn = norm(TD)
    U, per_file_units = [], {}
    for k, (L2, SP) in data.items():
        cells = SP.reshape(len(SP), 30, 7, CELL).transpose(0, 2, 1, 3).astype(np.float32)
        occ = (cells[:, :, 2:28, 2:22] > INK).mean((2, 3))
        best = shifted_ncc(cells[:, :4].reshape(-1, 30, CELL), TDn).max(1).reshape(-1, 4)
        cnt = 0
        for i in range(len(cells)):
            inked = occ[i] > BLANK_MAX
            if not inked.any():
                continue
            length = int(np.nonzero(inked)[0].max()) + 1          # text = n digits + 'km/h'; 'h' is always inked
            nd = length - 4                                         # ('/' and a narrow digit often are not)
            if not 1 <= nd <= 3 or not (best[i, :nd] >= DIG_MIN).all():
                continue
            U.append(cells[i, nd:nd + 4])
            cnt += 1
        per_file_units[k] = cnt
    if not U:
        raise SystemExit("stop: no unit examples found")
    TU = np.median(np.stack(U), 0).astype(np.float32)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "digits.npy", TD)
    np.save(out / "unit.npy", TU)
    (out / "template_files.txt").write_text("".join(n + "\n" for n, _, _ in paths))
    info = {"tool": "tools/dod_osd.py templates", "tool_sha256": sha256_file(TOOL), "ffmpeg_sha256": ffsha,
            "made": now(), "files": [n for n, _, _ in paths],
            "offset_o": {k: offs[k][0] for k in offs}, "offset_agreement": {k: round(offs[k][1], 4) for k in offs},
            "phase_p": phases, "digit_examples": np.bincount(y, minlength=10).tolist(),
            "leave_one_day_out": lodo, "unit_examples": int(len(U)), "unit_examples_per_file": per_file_units,
            "labels": "clock digits: file-name time + o + ticks; units: digit-template segmentation; no speed answer used "
                      "for the digits"}
    write_json(out / "build_info.json", info)
    sums = "".join(f"{sha256_file(out / f)}  {f}\n" for f in ("digits.npy", "unit.npy", "build_info.json",
                                                             "template_files.txt"))
    (out / "SHA256SUMS").write_text(sums)
    print(json.dumps({"leave_one_day_out": lodo, "unit_examples": len(U), "offsets": info["offset_o"]}, indent=1))
    log_line(f"templates done\tfiles={len(paths)}\tout={out}")


# ----------------------------------------------------------------------------- validation sample
def cmd_sample(a):
    import cv2
    ffsha = check_ffmpeg()
    osd = Path(a.osd)
    out = guard_private_dir(a.out, "--out")
    if out.exists() and any(out.iterdir()):
        refuse(f"{out} exists and is not empty")
    if inside(out, osd) or inside(osd, out):
        refuse("--out must be apart from --osd")
    dev_names, dev_sha = load_dev_lists()
    cand = []
    for spath in sorted(osd.glob("*_osd_summary.json")):
        S = json.loads(spath.read_text())
        with open(osd / S["outputs"]["updates"], newline="") as f:
            for r in csv.DictReader(f):
                if r["valid"] == "1":
                    a0, b0 = int(r["first_frame"]), int(r["last_frame"])
                    cand.append({"tag": S["tag"], "file": S["file"], "slot": int(r["slot"]), "frame": (a0 + b0) // 2,
                                 "value_kmh": int(r["value_kmh"]), "ndig": len(r["value_kmh"]), "day": S["file"][:10]})
    if not cand:
        raise SystemExit(f"stop: no valid slots in {osd}")
    need = {}
    for c in cand:
        need.setdefault(c["file"], set()).add(c["frame"])
    crops = {}
    log_line(f"sample start\tfiles={len(need)}\tcandidates={len(cand)}\tout={out}")
    for name in sorted(need):
        path, sha_listed = source_path(name, dev_names, dev_sha)
        frames = need[name]

        def on_frame(i, frame):
            if i in frames:
                yl = yellowness(frame)
                ok22 = float((yl[7:35, 2:22] > INK).mean()) <= LAYOUT_MAX
                img = frame[:, SAVE_X0:].copy()
                grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                bg = grey[yl[:, SAVE_X0:] <= INK]
                crops[(name, i)] = (img if ok22 else None, float(np.median(bg)) if bg.size else 0.0,
                                    float(bg.std()) if bg.size else 0.0)

        n, rc, _, _ = decode(path, f"crop={SW}:{SH}:{SX}:{SY}", SW, SH, on_frame)
        if rc != 0:
            raise SystemExit(f"stop: {name}: ffmpeg exit {rc}")
    usable = [c for c in cand if crops.get((c["file"], c["frame"]), (None,))[0] is not None]
    sds = np.array([crops[(c["file"], c["frame"])][2] for c in usable])
    lums = np.array([crops[(c["file"], c["frame"])][1] for c in usable])
    tex_thr = float(np.percentile(sds, 80)) if len(sds) else 0.0
    lum_thr = float(np.percentile(lums, 80)) if len(lums) else 0.0
    for c in usable:
        _, lum, sd = crops[(c["file"], c["frame"])]
        c["bg_luma"], c["bg_sd"] = round(lum, 1), round(sd, 1)
        c["background"] = "textured" if sd >= tex_thr else ("bright" if lum >= lum_thr else "dark")
    strata = {}
    for c in usable:
        strata.setdefault((c["day"], c["ndig"], c["background"]), []).append(c)
    keys = sorted(strata)
    quota = {k: 0 for k in keys}
    left = min(a.n, len(usable))
    while left > 0:
        moved = False
        for k in keys:
            if left and quota[k] < len(strata[k]):
                quota[k] += 1
                left -= 1
                moved = True
        if not moved:
            break
    rnd = random.Random(a.seed)
    pick = []
    for k in keys:
        pick += rnd.sample(strata[k], quota[k])
    rnd.shuffle(pick)
    (out / "sheets").mkdir(parents=True)
    (out / "key").mkdir()
    keyrows, tiles = [], []
    for i, c in enumerate(pick):
        c["id"] = "S%03d" % (i + 1)
        keyrows.append(c)
        if not a.dry_run:
            im = cv2.resize(crops[(c["file"], c["frame"])][0], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
            lab = np.zeros((im.shape[0], 110, 3), np.uint8)
            cv2.putText(lab, c["id"], (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tiles.append(np.hstack([lab, im]))
    sheets = []
    for s in range(0, len(tiles), 20):
        p = out / "sheets" / ("validation_sheet_%02d.png" % (s // 20 + 1))
        cv2.imwrite(str(p), np.vstack(tiles[s:s + 20]))
        sheets.append(p.name)
    (out / "sheets" / "README.txt").write_text(
        "Each tile shows the speed field of one frame (number and km/h), labelled S001, S002, ...\n"
        "Write one line per id: the id, a TAB, the number you read (or x if it cannot be read).\n"
        "Do not open the key folder and do not look at any reader output before you finish.\n")
    cols = ["id", "tag", "file", "slot", "frame", "value_kmh", "ndig", "day", "background", "bg_luma", "bg_sd"]
    write_csv(out / "key" / "key.csv", cols, keyrows)
    write_json(out / "key" / "strata.json", {
        "made": now(), "tool_sha256": sha256_file(TOOL), "ffmpeg_sha256": ffsha, "osd": str(osd), "n": a.n,
        "seed": a.seed, "candidates": len(cand), "usable": len(usable),
        "textured_sd_threshold": round(tex_thr, 2), "bright_luma_threshold": round(lum_thr, 2),
        "background_rule": "textured = grey SD >= its 80th percentile; else bright = median grey >= its 80th "
                           "percentile; else dark (non-ink pixels of the crop, over all usable slots)",
        "dry_run": bool(a.dry_run), "sheets": sheets,
        "strata": [{"day": k[0], "ndig": k[1], "background": k[2], "available": len(strata[k]), "chosen": quota[k]}
                   for k in keys]})
    log_line(f"sample done\tchosen={len(pick)}\tsheets={len(sheets)}\tout={out}")
    print(f"{len(pick)} samples from {len(usable)} usable slots in {len(keys)} strata; sheets {sheets}; "
          f"key {out / 'key' / 'key.csv'}")


def upper_bound(k, n, conf=0.95):
    """One-sided Clopper-Pearson upper bound for k errors in n."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0

    def cdf(p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if cdf(mid) > 1 - conf:
            lo = mid
        else:
            hi = mid
    return hi


def cmd_compare(a):
    with open(a.key, newline="") as f:
        key = {r["id"]: r for r in csv.DictReader(f)}
    tr = {}
    for line in open(a.transcription, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 2:
            raise SystemExit(f"stop: bad line in transcription: {line!r}")
        v = parts[1]
        tr[parts[0].upper()] = None if v.lower() == "x" else int(v)
    res = {"key": a.key, "transcription": a.transcription, "made": now(), "n_key": len(key), "n_transcribed": len(tr),
           "missing": sorted(set(key) - set(tr)), "unknown_ids": sorted(set(tr) - set(key)), "unreadable": 0,
           "both_readable": 0, "agree": 0, "wrong": [], "by_stratum": {}}
    for sid in sorted(set(key) & set(tr)):
        k, v = key[sid], tr[sid]
        st = res["by_stratum"].setdefault(f"{k['day']}|{k['ndig']}|{k['background']}", {"n": 0, "wrong": 0, "x": 0})
        st["n"] += 1
        if v is None:
            res["unreadable"] += 1
            st["x"] += 1
            continue
        res["both_readable"] += 1
        if v == int(k["value_kmh"]):
            res["agree"] += 1
        else:
            res["wrong"].append({"id": sid, "reader": int(k["value_kmh"]), "transcribed": v, "file": k["file"],
                                 "frame": int(k["frame"])})
            st["wrong"] += 1
    nw = len(res["wrong"])
    res["wrong_count"] = nw
    res["upper95_wrong_rate"] = round(upper_bound(nw, res["both_readable"]), 4)
    res["pass"] = nw == 0 and res["both_readable"] > 0
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read", help="read the clock and the speed of development files")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", nargs="+", help="file names (NAME.MP4) in raw_dev")
    g.add_argument("--trip", nargs="+", help="trip labels of the tags table (T01 ...)")
    g.add_argument("--all", action="store_true", help="every file of the tags table")
    r.add_argument("--tags", required=True, help="tags table (file, tag, trip_label, trip_id, start, duration_s, pictures)")
    r.add_argument("--templates", required=True, help="folder with digits.npy and unit.npy (private)")
    r.add_argument("--out", required=True, help="output folder (private; no prediction program reads it)")
    r.add_argument("--frames-csv", action="store_true", help="also write <tag>_osd_frames.csv")
    r.add_argument("--frames-root", default=str(FRAMES_ROOTS[0]), help="masked frames root, to cross-check mask.json")
    r.add_argument("--redo", action="store_true", help="read again even when an up-to-date summary exists")
    t = sub.add_parser("templates", help="rebuild candidate templates from a list of development files")
    t.add_argument("--files-list", required=True)
    t.add_argument("--out", required=True, help="new, empty folder")
    s = sub.add_parser("sample", help="validation crops of valid slots (speed field only) and a separate key")
    s.add_argument("--osd", required=True, help="a read output folder")
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--seed", type=int, default=20261029)
    s.add_argument("--out", required=True, help="new, empty folder; sheets/ for the transcriber, key/ kept apart")
    s.add_argument("--dry-run", action="store_true", help="choose and write the key, but no image")
    c = sub.add_parser("compare", help="compare a transcription with the key")
    c.add_argument("--key", required=True)
    c.add_argument("--transcription", required=True)
    c.add_argument("--out", help="also write the result as JSON")
    a = ap.parse_args()
    {"read": cmd_read, "templates": cmd_templates, "sample": cmd_sample, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()

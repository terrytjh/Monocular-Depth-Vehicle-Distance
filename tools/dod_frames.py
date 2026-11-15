#!/usr/bin/env python3
"""Masked frames of the DOD development clips for the Taiwan ego-speed check (2026-10-28; CPU only, no model, no answer).

The recorder burns its model name, date and time, position and speed into the bottom-left of every picture, and the
dashboard below the road carries another speed display. The agreed blinding rule: in every frame the method sees, all
rows y >= 870 of the 1080, full width, are black (the burned-in lines, the dashboard display and the bonnet). This tool
blacks from row 864 (Terry, 2026-10-28: 6 rows more, so that the mask edge falls on the 16-row JPEG block grid at both
sizes, 864 = 54 x 16 and at 720p 576 = 36 x 16, and every pixel of rows >= 870 is exactly 0 in the written frames). The
mask is applied INSIDE ffmpeg (the rows are cropped away and padded back black) before any frame is encoded, so no
unmasked frame is ever written. It never reads the burned-in text, the audio or a metadata value, and it never writes
ffmpeg's messages anywhere (they carry container metadata); only frame numbers, pts and error counts are kept from them.

Input: only /mnt/d/depth_work/tw_own/raw_dev/<name>.MP4 with <name> in lists/dev_front_files.txt and in
lists/dev_front_sha256.txt (the development days); with --set blind1 (first blind test) raw_blind1, blind1_files.txt,
blind1_sha256.txt, tags_blind1.tsv and frames_blind1 instead, and no sheet / selftest. Refused before anything is opened: /mnt/e/ and Windows paths, other
folders, symlinks, names that are not listed. The source's sha256 is recomputed and must equal the list's. One tag per
60-s file, from the tag table (lists/tags_dev.tsv: file, tag, trip_label, trip_id, start, duration_s, pictures):
dod_<trip_id>_<HHMMSS>-<NN>, exactly two '_' (tools/car_a.py splits the tag in three; the drive part is the trip).

  extract   python -B tools/dod_frames.py extract (--file NAME.MP4 ... | --trip T05 ... | --all) [--tags TSV]
                [--res 720|1080] [--frames-root DIR] [--remake]
            1. timing  ffmpeg -c copy -f framecrc (no decoding): the video samples sorted by pts; a picture is a sample
                       of more than 0 bytes (the recorder adds an empty sample after every 5 pictures, which is why the
                       container says ~30 fps). fps = round(1 / median picture interval, 4), the rule of
                       tools/c2k19_extract.py; outside 24.95-25.05 the clip is not decoded. timing_irregular = some
                       picture interval more than 10 % off the median (files closed late, at the next power-on).
            2. decode  one JPEG per picture with -fps_mode passthrough -enc_time_base demux (without the second flag
                       ffmpeg stops early on the irregular files), the mask first, at full resolution:
                         1080  crop=1920:864:0:0,pad=1920:1080:0:0:black
                         720   the same, then scale=1280:720:flags=area,crop=1280:576:0:0,pad=1280:720:0:0:black
                               (864 x 2/3 = 576: rows >= 576 come from black rows only, and are blacked again)
                       mjpeg -qscale:v 2 with one encoder thread (a re-made frame is byte-identical). f00001.jpg is
                       picture 0: frame index i (0-based) = f{i+1:05d}.jpg, t = i / fps. The frames go to a
                       temporary folder that is renamed to <tag>/frames/ when the clip is done.
            3. checks  any failure: checks_ok false, the clip stays out of the run list, exit code 3.
                       ffmpeg exit 0; JPEG count N == showinfo's frame count, showinfo's pts == the first N picture pts
                       in order; tail_missing = pictures - N is 0, or 1-2 on a timing_irregular file; every frame read
                       back (cv2, BGR) has the chosen size and is exactly 0 in all channels in the rows of the agreed
                       rule, y >= 870 (580 at 720p) (if not, or if a frame cannot be read, the new frames are removed
                       at once); source sha256 == the list's; tools/band_bottom.py and tools/day_night.brightness (both
                       unchanged) run on the masked frames. The masked rows above the rule rows (864-869, at 720p
                       576-579) were black before encoding; only the decoder's chroma interpolation from the row above
                       the edge can reach the first of them: their per-row maxima and the first row from which every
                       frame is exactly 0 are recorded, not bounded.
            Writes <frames-root>/<tag>/frames/, frames.csv (index,name,pts_s,sha256; pts_s from the container, for
            diagnostics only) and mask.json (numbers and names only), and one line per clip in logs/dod_frames.log.
            A clip whose mask.json says checks_ok at the same --res is skipped, so a batch can be restarted; a record
            made with another mask edge is refused (frames are never re-used across masks). A non-empty frames folder
            without such a record is refused. --remake decodes again and must reproduce the stored
            frames.csv exactly (the frames are then replaced by the identical new ones), or it fails without touching
            the old record. Stops before a clip when the disk has less than 15.6 GB free (exit code 4).
  runlist   python -B tools/dod_frames.py runlist [--tags TSV] [--frames-root DIR] --out-dir DIR
                [--band camera|clip|<x>] [--roadclass TSV] [--only TAGFILE]
            One line per clip that passed, in tag-table order, for tools/run_v4.sh (fields d|r|s|chunk|road|fps|dir|band):
              dod|<trip_id>|<HHMMSS-NN>|dod|taiwan|<fps, 4 decimals>|<frames dir>|<band bottom, 3 decimals>
            (road "taiwan" = article 182: 10 m, painted 0.4, line-width gate tw, marking step 2). Band: camera (default)
            = the median band bottom of the daytime (grey >= 50), non-suspect clips; clip = each clip's own value (a
            suspect clip takes the camera value); <x> = a frozen value; never above 0.780 (864 / 1080 - 0.02, the
            2 % margin of band_bottom.py above the black rows). Groups: tags_night.txt = grey < 50 (the registered
            day / night rule); tags_local_day.txt = the daytime clips that --roadclass (lines: tag or trip_id <tab>
            main|local, judged on masked frames) calls local; all other clips tags_main.txt. Also run_list.txt and
            runlist_summary.json (per-clip band, grey, group and flags, the camera band, excluded clips and why).
  verify    python -B tools/dod_frames.py verify [--frames-root DIR] [--tag TAG ...]
            Re-hashes every frame against frames.csv, and frames.csv against mask.json.
  sheet     python -B tools/dod_frames.py sheet [--frames-root DIR] --out DIR [--tag TAG ... | --trip T05 ...]
            One JPEG per clip from the masked frames only: a 3 x 3 contact sheet and below it the rows just above the
            mask from 3 frames, at 1920 px wide (road class; a look for leftover text or reflections).
  selftest  python -B tools/dod_frames.py selftest [--work DIR] [--keep]
            Synthetic clip only (ffmpeg testsrc2, HEVC, a white box across rows 860-1079), no recorder file opened:
            both sizes, masked-row maxima, counts, fps, determinism, --remake, refusals, run-list parsing, verify, sheet.

Run as  nice -n 10 env LD_PRELOAD=$HOME/miniconda3/lib/libstdc++.so.6 ~/venvs/depthbench/bin/python -B tools/dod_frames.py ...
Exit codes: 0 done; 2 an input refused (nothing processed); 3 a clip failed a check or was refused; 4 disk guard.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import band_bottom as BB  # noqa: E402
import day_night as DN  # noqa: E402

TOOL = "tools/dod_frames.py"
TOOL_VERSION = "2 (2026-10-28, mask edge 864)"
TW = Path("/mnt/d/depth_work/tw_own")
RAW_DEV = TW / "raw_dev"
FILES_LIST = TW / "lists" / "dev_front_files.txt"
SHA_LIST = TW / "lists" / "dev_front_sha256.txt"
TAGS_DEFAULT = TW / "lists" / "tags_dev.tsv"
FRAMES_ROOT_DEFAULT = TW / "frames_dev"
LOG_DEFAULT = TW / "logs" / "dod_frames.log"
# Input sets (--set, default dev). blind1 = the first DOD blind test (docs/DOD_V12_BLIND_PREREG.md): the 60 drawn clips copied
# from the external drive into raw_blind1 with their sha256 at copy time; frames are written but never shown (no sheet).
# blind2 = the highway blind test (docs/DOD_V12_BLIND2_PREREG.md), same rules.
SETS = {
    "dev": dict(raw=TW / "raw_dev", files=TW / "lists" / "dev_front_files.txt", sha=TW / "lists" / "dev_front_sha256.txt",
                tags=TW / "lists" / "tags_dev.tsv", frames=TW / "frames_dev", log=TW / "logs" / "dod_frames.log"),
    "blind1": dict(raw=TW / "raw_blind1", files=TW / "lists" / "blind1_files.txt", sha=TW / "lists" / "blind1_sha256.txt",
                   tags=TW / "lists" / "tags_blind1.tsv", frames=TW / "frames_blind1", log=TW / "logs" / "dod_frames_blind1.log"),
    "blind2": dict(raw=TW / "raw_blind2", files=TW / "lists" / "blind2_files.txt", sha=TW / "lists" / "blind2_sha256.txt",
                   tags=TW / "lists" / "tags_blind2.tsv", frames=TW / "frames_blind2", log=TW / "logs" / "dod_frames_blind2.log"),
}
INPUT_SET = "dev"
FFMPEG = Path.home() / ".local/bin/ffmpeg"
FFMPEG_SHA256 = "e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99"   # 7.0.2 static, as the joint runs

SRC_W, SRC_H = 1920, 1080
MASK_Y = 864              # mask edge: the agreed rows >= 870 plus 6, on the 16-row JPEG block grid (Terry, 2026-10-28)
RULE_Y = 870              # the agreed rule: every pixel of rows >= 870 is exactly 0 in every written frame
_MASK_VF = f"crop={SRC_W}:{MASK_Y}:0:0,pad={SRC_W}:{SRC_H}:0:0:black"
RES = {
    "1080": dict(size=(1920, 1080), mask_row=864, rule_row=870, vf=_MASK_VF),
    "720": dict(size=(1280, 720), mask_row=576, rule_row=580,
                vf=_MASK_VF + ",scale=1280:720:flags=area,crop=1280:576:0:0,pad=1280:720:0:0:black"),
}
MASK_RULE = ("rows y >= 864 of 1080 black, full width, inside ffmpeg before encoding (the agreed rows >= 870 plus 6, on "
             "the 16-row JPEG block grid); at 720p rows >= 576 black again after scaling; every pixel of rows >= 870 "
             "(720p: 580) exactly 0 in every written frame")
EDGE_RULE = ("rows mask_row .. rule_row - 1 were black before encoding; only the decoder's chroma interpolation from the "
             "row above the edge can reach the first of them: per-row maxima and the first row from which every frame "
             "is exactly 0 recorded, not bounded; rows from rule_row on must be exactly 0")
FPS_RANGE = (24.95, 25.05)
IRREGULAR_TOL = 0.10      # a picture interval this far off the median makes the file timing_irregular
TAIL_MAX = 2              # missing last pictures allowed on a timing_irregular file
DISK_MIN_BYTES = int(15.6e9)
BAND_MAX = 0.780          # 864 / 1080 - 0.02
NIGHT_GREY = 50.0         # registered day / night rule (tools/day_night.py, grey < 50 = night)
JPEG_ARGS = ["-c:v", "mjpeg", "-qscale:v", "2", "-threads", "1"]
TAG_RE = re.compile(r"^dod_\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}_\d{6}-\d{2}$")
NAME_RE = re.compile(r"^(\d{4})_(\d{2})_(\d{2})_(\d{6})_(\d{2})\.MP4$")
TAG_COLS = ["file", "tag", "trip_label", "trip_id", "start", "duration_s", "pictures"]
SHOWINFO = re.compile(r"\[Parsed_showinfo_\d+ @ [^\]]*\] (?:\[\w+\] )?n:\s*(\d+) pts:\s*(-?\d+) pts_time:")
SHOWINFO_TB = re.compile(r"\[Parsed_showinfo_\d+ @ [^\]]*\] (?:\[\w+\] )?config in time_base: (\d+)/(\d+)")
LEVEL = re.compile(r"^(?:\[[^\]]* @ [^\]]*\] )*\[(trace|debug|verbose|info|warning|error|fatal|panic)\] ")
REPEAT = re.compile(r"Last message repeated (\d+) times")


class Refused(Exception):
    """An input or an output this tool must not touch."""


class DiskLow(Exception):
    """Less free space than DISK_MIN_BYTES."""


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256_file(p, buf=1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def git(repo: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


# ----------------------------------------------------------------------------------------------- inputs and lists

def read_lists(files_list=None, sha_list=None):
    """Names of the input set's files, and their sha256 (copy == source, third column 'same')."""
    files_list = files_list or FILES_LIST
    sha_list = sha_list or SHA_LIST
    names = {l.strip() for l in Path(files_list).read_text().splitlines() if l.strip()}
    sha = {}
    for l in Path(sha_list).read_text().splitlines():
        f = [x.strip() for x in l.split("\t")]
        if len(f) >= 2 and f[0]:
            if len(f) >= 3 and f[2] != "same":
                continue                     # copy and source differ: not usable
            sha[f[0]] = f[1].lower()
    return names, sha


def load_tags(path, names, sha):
    """The tag table, checked: every file listed for development, tags well formed, unique, consistent."""
    path = Path(path)
    raw = path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8")), delimiter="\t"))
    if not rows or list(rows[0].keys()) != TAG_COLS:
        raise Refused(f"{path}: columns must be {' '.join(TAG_COLS)}")
    seen_t, seen_f = set(), set()
    for r in rows:
        f, t = r["file"], r["tag"]
        m = NAME_RE.match(f)
        if not m:
            raise Refused(f"{path}: unexpected file name {f!r}")
        if f not in names or f not in sha:
            raise Refused(f"{path}: {f} is not in {FILES_LIST.name} / {SHA_LIST.name}")
        if not TAG_RE.match(t) or t.count("_") != 2:
            raise Refused(f"{path}: bad tag {t!r}")
        if t != f"dod_{r['trip_id']}_{m.group(4)}-{m.group(5)}":
            raise Refused(f"{path}: tag {t} does not follow from {f} and trip {r['trip_id']}")
        if t in seen_t or f in seen_f:
            raise Refused(f"{path}: duplicate {t} / {f}")
        seen_t.add(t)
        seen_f.add(f)
    return rows, hashlib.sha256(raw).hexdigest()


def resolve_source(arg, names, sha) -> Path:
    """RAW_DEV/<listed name>, or Refused. Strings are checked before the file system is touched."""
    s = str(arg).strip()
    if "\\" in s or re.match(r"^[A-Za-z]:", s):
        raise Refused(f"{s}: Windows paths are refused; only {RAW_DEV}/<name>.MP4 is read")
    low = s.lower()
    norm = os.path.normpath(os.path.abspath(s)) if "/" in s else str(RAW_DEV / s)
    if any(x == "/mnt/e" or x.startswith("/mnt/e/") for x in (low, norm.lower())):
        raise Refused(f"{s}: the external drive is never read")
    p = Path(norm)
    name = p.name
    if not NAME_RE.match(name):
        raise Refused(f"{s}: not a recorder file name")
    if name not in names or name not in sha:
        raise Refused(f"{name}: not in {FILES_LIST.name} and {SHA_LIST.name} (input set {INPUT_SET} only)")
    if p.parent != RAW_DEV:
        raise Refused(f"{s}: only files directly in {RAW_DEV} are read")
    if Path(os.path.realpath(RAW_DEV)) != RAW_DEV:
        raise Refused(f"{RAW_DEV} is not a plain folder")
    if p.is_symlink():
        raise Refused(f"{p}: a symlink")
    rp = p.resolve(strict=True)
    if rp.parent != RAW_DEV or rp.name != name or not rp.is_file():
        raise Refused(f"{p}: does not resolve to a file in {RAW_DEV}")
    return rp


def check_out_root(root) -> Path:
    """Frames never go into a repository or onto the external drive."""
    r = Path(os.path.abspath(root))
    if str(r).lower().startswith("/mnt/e"):
        raise Refused(f"{r}: never write to the external drive")
    for d in (r, *r.parents):
        if (d / ".git").exists():
            raise Refused(f"{r}: inside a git work tree ({d}); recorder frames are private")
    return r


def make_ctx(tags_path, tags_sha, log_path):
    ff_sha = sha256_file(FFMPEG.resolve())
    if ff_sha != FFMPEG_SHA256:
        raise Refused(f"{FFMPEG}: sha256 {ff_sha[:16]}... is not the agreed ffmpeg 7.0.2 static binary")
    tool = Path(__file__).resolve()
    repo = tool.parent.parent
    head = git(repo, "rev-parse", "HEAD")
    rel = str(tool.relative_to(repo))
    tracked = git(repo, "ls-files", "--error-unmatch", rel).returncode == 0
    clean = git(repo, "diff", "--quiet", "HEAD", "--", rel).returncode == 0
    return dict(tool_sha256=sha256_file(tool), repo_head=head.stdout.strip() if head.returncode == 0 else None,
                tool_committed=bool(tracked and clean), ffmpeg=str(FFMPEG), ffmpeg_sha256=ff_sha,
                tags_table=None if tags_path is None else str(tags_path), tags_table_sha256=tags_sha,
                log=Path(log_path))


# ----------------------------------------------------------------------------------------------- timing and decode

def timing(src):
    """Packet table without decoding -> (numbers, irregular, picture pts as ints, time base (num, den)) or a reason."""
    cmd = [str(FFMPEG), "-nostdin", "-v", "error", "-i", str(src), "-map", "0:v:0", "-c", "copy", "-f", "framecrc", "-"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    err = sum(1 for l in p.stderr.splitlines() if l.strip())          # counted, never kept
    tb, pts, size = None, [], []
    try:
        for line in p.stdout.splitlines():
            if line.startswith("#tb 0:"):
                num, den = line.split(":", 1)[1].strip().split("/")
                tb = (int(num), int(den))
                continue
            if not line.strip() or line.startswith("#"):
                continue
            f = [x.strip() for x in line.split(",")]
            if f[0] == "0":
                pts.append(int(f[2]))
                size.append(int(f[4]))
    except ValueError:
        return None, f"packet table not readable (exit {p.returncode})"
    if p.returncode or tb is None or len(pts) < 2:
        return None, f"packet table failed (exit {p.returncode}, {err} error lines, {len(pts)} samples)"
    pts, size = np.array(pts, np.int64), np.array(size, np.int64)
    o = np.argsort(pts, kind="stable")
    pts, size = pts[o], size[o]
    pic = pts[size > 0]
    if len(pic) < 2:
        return None, f"only {len(pic)} pictures"
    sec = tb[0] / tb[1]
    iv = np.diff(pic) * sec
    med = float(np.median(iv))
    off = np.abs(iv - med) > IRREGULAR_TOL * med
    info = dict(samples=int(len(pts)), empty_samples=int((size == 0).sum()), pictures=int(len(pic)),
                time_base=f"{tb[0]}/{tb[1]}", interval_ms_median=round(med * 1000, 3),
                interval_ms_min=round(float(iv.min()) * 1000, 3), interval_ms_max=round(float(iv.max()) * 1000, 3),
                intervals_off_median_gt_10pct=int(off.sum()), fps=round(1.0 / med, 4),
                fps_span=round(float((len(pic) - 1) / ((pic[-1] - pic[0]) * sec)), 4),
                sample_rate_not_used=round(float((len(pts) - 1) / ((pts[-1] - pts[0]) * sec)), 4),
                packet_table_exit=p.returncode, packet_table_error_lines=err)
    return (info, bool(off.any()), [int(x) for x in pic], tb), None


def decode(src, tmp: Path, res: str):
    """ffmpeg -> masked JPEGs in tmp. Of the messages only showinfo's frame numbers / pts and level counts are kept."""
    cmd = [str(FFMPEG), "-nostdin", "-hide_banner", "-loglevel", "level+info", "-threads", "4", "-i", str(src),
           "-map", "0:v:0", "-an", "-sn", "-dn", "-fps_mode", "passthrough", "-enc_time_base", "demux",
           "-vf", RES[res]["vf"] + ",showinfo", *JPEG_ARGS, "-start_number", "1", str(tmp / "f%05d.jpg")]
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    shown, tb, err, warn = [], None, 0, 0
    for line in p.stderr.decode("utf-8", "replace").splitlines():
        m = SHOWINFO.search(line)
        if m:
            shown.append((int(m.group(1)), int(m.group(2))))
            continue
        m = SHOWINFO_TB.search(line)
        if m:
            tb = (int(m.group(1)), int(m.group(2)))
            continue
        m = LEVEL.match(line)
        if m:
            r = REPEAT.search(line)
            k = int(r.group(1)) if r else 1
            if m.group(1) in ("error", "fatal", "panic"):
                err += k
            elif m.group(1) == "warning":
                warn += k
    return dict(command=cmd, exit=p.returncode, error_lines=err, warning_lines=warn), shown, tb


def tail_ok(pictures: int, n: int, irregular: bool) -> bool:
    missing = pictures - n
    return missing == 0 or (irregular and 1 <= missing <= TAIL_MAX)


def _read_one(f: Path, res: str):
    W, H = RES[res]["size"]
    y0, yr = RES[res]["mask_row"], RES[res]["rule_row"]
    data = f.read_bytes()
    h = hashlib.sha256(data).hexdigest()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)       # BGR, the decoder of cv2.imread
    if img is None:
        return h, len(data), "unreadable", 0, 0, None
    if img.shape != (H, W, 3):
        return h, len(data), "size", 0, 0, None
    return h, len(data), "ok", int(img[y0:].max()), int(img[yr:].max()), img[y0:yr].max(axis=(1, 2))


def read_back(files, res, workers=4):
    """Hash and decode every frame from the same bytes (threads; order kept); masked-row maxima."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        got = list(ex.map(lambda f: _read_one(f, res), files))
    ok = [g for g in got if g[2] == "ok"]
    y0 = RES[res]["mask_row"]
    max_rule = max((g[4] for g in ok), default=0)
    edge = np.max([g[5] for g in ok], axis=0) if ok else None      # per row of mask_row .. rule_row - 1, over frames
    first_zero = None
    if edge is not None and max_rule == 0:
        nz = np.flatnonzero(edge)
        first_zero = y0 + (int(nz[-1]) + 1 if len(nz) else 0)
    return dict(hashes=[g[0] for g in got], bytes=sum(g[1] for g in got),
                max_mask=max((g[3] for g in ok), default=0), max_rule=max_rule,
                edge_row_max=None if edge is None else [int(v) for v in edge], first_zero_row=first_zero,
                bad_size=sum(g[2] == "size" for g in got), unreadable=sum(g[2] == "unreadable" for g in got))


def remove_frames_dir(d: Path):
    """Remove a folder this tool made: only f*.jpg files inside (never anything else)."""
    if not d.exists():
        return
    others = [p for p in d.iterdir() if not (p.is_file() and re.fullmatch(r"f\d{5}\.jpg", p.name))]
    if others:
        raise Refused(f"{d}: holds {len(others)} entries this tool did not make; not removed")
    for p in d.iterdir():
        p.unlink()
    d.rmdir()


def write_atomic(path: Path, text: str):
    t = path.with_name(path.name + ".tmp")
    t.write_text(text)
    t.replace(path)


class TagLock:
    """One process per tag folder at a time (a lock file holding the pid; a dead pid's lock is taken over)."""

    def __init__(self, out: Path):
        self.p = out / ".dod_frames.lock"

    def __enter__(self):
        for _ in range(2):
            try:
                fd = os.open(self.p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    pid = int(self.p.read_text().strip() or 0)
                    if pid <= 0:
                        raise ValueError
                    os.kill(pid, 0)
                    raise Refused(f"{self.p.parent.name}: being made by process {pid} (lock {self.p})")
                except PermissionError:
                    raise Refused(f"{self.p.parent.name}: locked by a live process (lock {self.p})")
                except (ValueError, ProcessLookupError):
                    self.p.unlink(missing_ok=True)          # stale lock of a process that is gone
        raise Refused(f"{self.p}: lock not obtained")

    def __exit__(self, *exc):
        self.p.unlink(missing_ok=True)


# ----------------------------------------------------------------------------------------------- one clip

def make_clip(src: Path, row: dict, sha_expected: str, res: str, out: Path, ctx: dict):
    """Decode one clip into out/frames.tmp and run every check. Returns (record, tmp folder or None, frames.csv text
    or None). Nothing outside out/frames.tmp is written here."""
    t0 = time.time()
    sec = {}
    rec = dict(tool=TOOL, tool_version=TOOL_VERSION, tool_sha256=ctx["tool_sha256"], repo_head=ctx["repo_head"],
               tool_committed=ctx["tool_committed"], made=None,
               source=dict(file=src.name, path=str(src), bytes=src.stat().st_size, sha256=sha_expected,
                           sha256_list=str(SHA_LIST), sha256_rechecked=None),
               tag=row["tag"], trip_label=row["trip_label"], trip_id=row["trip_id"], frames_root=str(out.parent),
               tags_table=ctx["tags_table"],
               tags_table_sha256=ctx["tags_table_sha256"], container=None, timing_irregular=None, decode=None,
               frames=None, mask=None, for_run_list=None, checks={}, reasons=[], checks_ok=False)
    checks, reasons = rec["checks"], rec["reasons"]

    def done(tmp=None, csv_text=None):
        sec["total"] = round(time.time() - t0, 2)
        rec["seconds"] = sec
        rec["checks_ok"] = bool(checks) and all(checks.values()) and not reasons
        return rec, tmp, csv_text

    sha = sha256_file(src)
    sec["sha256"] = round(time.time() - t0, 2)
    rec["source"]["sha256_rechecked"] = checks["source_sha256"] = sha == sha_expected
    if sha != sha_expected:
        reasons.append("source sha256 differs from the list (not decoded)")
        return done()
    t1 = time.time()
    tm, why = timing(src)
    sec["timing"] = round(time.time() - t1, 2)
    if tm is None:
        checks["packet_table"] = False
        reasons.append(why)
        return done()
    info, irregular, pic, ctb = tm
    rec["container"], rec["timing_irregular"] = info, irregular
    checks["packet_table"] = True
    checks["fps_range"] = FPS_RANGE[0] <= info["fps"] <= FPS_RANGE[1]
    if not checks["fps_range"]:
        reasons.append(f"fps {info['fps']} outside {FPS_RANGE[0]}-{FPS_RANGE[1]} (not decoded)")
        return done()

    tmp = out / "frames.tmp"
    if tmp.exists():
        remove_frames_dir(tmp)                       # left by an interrupted run of this tool
    tmp.mkdir(parents=True)
    t2 = time.time()
    dec, shown, stb = decode(src, tmp, res)
    sec["decode"] = round(time.time() - t2, 2)
    files = sorted(tmp.iterdir())
    n = len(files)
    names_ok = [p.name for p in files] == [f"f{i + 1:05d}.jpg" for i in range(n)]
    pts_match = (stb is not None and len(shown) == n and n <= len(pic)
                 and all(k == i for i, (k, _) in enumerate(shown))
                 and all(sp * stb[0] * ctb[1] == cp * ctb[0] * stb[1] for (_, sp), cp in zip(shown, pic)))
    missing = info["pictures"] - n
    dec.update(ffmpeg_sha256=ctx["ffmpeg_sha256"], frames_written=n, showinfo_frames=len(shown),
               showinfo_time_base=None if stb is None else f"{stb[0]}/{stb[1]}", tail_missing=missing,
               pts_prefix_match=bool(pts_match), output="frames.tmp/f%05d.jpg, renamed to frames/ when done")
    rec["decode"] = dec
    checks["ffmpeg_exit"] = dec["exit"] == 0
    checks["frame_names"] = names_ok and n > 0
    checks["counts_pts"] = bool(pts_match)
    checks["tail"] = tail_ok(info["pictures"], n, irregular)
    if not checks["ffmpeg_exit"]:
        reasons.append(f"ffmpeg exit {dec['exit']} ({dec['error_lines']} error lines)")
    if not checks["frame_names"]:
        reasons.append("frame files are not exactly f00001.jpg ... in order")
    if not checks["counts_pts"]:
        reasons.append(f"{n} JPEGs, {len(shown)} showinfo frames: not the first {n} pictures in order")
    if not checks["tail"]:
        reasons.append(f"{missing} pictures not decoded ({'irregular' if irregular else 'regular'} file)")

    t3 = time.time()
    rb = read_back(files, res) if checks["frame_names"] else None
    sec["checks"] = round(time.time() - t3, 2)
    W, H = RES[res]["size"]
    y0, yr = RES[res]["mask_row"], RES[res]["rule_row"]
    leak = rb is None or rb["max_rule"] > 0 or rb["bad_size"] > 0 or rb["unreadable"] > 0
    rec["mask"] = dict(rule=MASK_RULE, source_rows=[MASK_Y, SRC_H - 1], rule_source_rows=[RULE_Y, SRC_H - 1],
                       rows_at_res=[y0, H - 1], rule_rows_at_res=[yr, H - 1],
                       max_in_masked_rows=None if rb is None else rb["max_mask"],
                       max_in_rule_rows=None if rb is None else rb["max_rule"],
                       edge_rows=[y0, yr - 1], edge_row_max=None if rb is None else rb["edge_row_max"],
                       first_zero_row=None if rb is None else rb["first_zero_row"],
                       edge_rule=EDGE_RULE, frames_wrong_size=None if rb is None else rb["bad_size"],
                       frames_unreadable=None if rb is None else rb["unreadable"],
                       ok=bool(rb is not None and not leak), frames_removed=False)
    checks["mask"] = rec["mask"]["ok"]
    if leak:
        remove_frames_dir(tmp)                       # a frame the mask check cannot clear is not kept
        rec["mask"]["frames_removed"] = True
        reasons.append("mask check not possible: no frames or wrong file names; frames removed" if rb is None else
                       f"mask check: wrong size, unreadable frame or a non-zero pixel in rows >= {yr}; frames removed")
        return done()

    lines = ["index,name,pts_s,sha256"]
    for i, (f, h) in enumerate(zip(files, rb["hashes"])):
        pts_s = "" if i >= len(pic) else f"{pic[i] * ctb[0] / ctb[1]:.6f}"
        lines.append(f"{i},{f.name},{pts_s},{h}")
    csv_text = "\n".join(lines) + "\n"
    rec["frames"] = dict(dir=str((out / "frames").resolve()), pattern="f%05d.jpg", start_number=1, count=n,
                         size=[W, H], res=res, index_rule="frame index i (0-based) = f{i+1:05d}.jpg; t = i / fps",
                         jpeg="mjpeg -qscale:v 2 -threads 1", bytes=rb["bytes"],
                         frames_csv_sha256=hashlib.sha256(csv_text.encode()).hexdigest())

    t4 = time.time()
    try:
        band, binfo = BB.band_bottom(tmp)
        grey = DN.brightness(tmp)
        checks["band_grey"] = grey is not None
        rec["for_run_list"] = dict(fps=info["fps"], band_bottom=band, band_reason=binfo["reason"],
                                   band_suspect=bool(binfo["suspect"]), band_block_top=binfo.get("block_top"),
                                   text_boxes=len(binfo.get("text_boxes") or []),
                                   text_boxes_xyxy=[list(map(int, b)) for b in binfo.get("text_boxes") or []],
                                   grey=None if grey is None else round(grey, 2),
                                   night=None if grey is None else bool(grey < NIGHT_GREY))
    except (Exception, SystemExit) as e:                 # band_bottom exits when it finds no frame
        checks["band_grey"] = False
        reasons.append(f"band_bottom / brightness failed: {type(e).__name__}")
    sec["band_grey"] = round(time.time() - t4, 2)
    return done(tmp, csv_text)


def prev_summary(old: dict) -> dict:
    keys = ("made", "tool_sha256", "tool_version", "checks_ok")
    d = {k: old.get(k) for k in keys}
    d["frames_csv_sha256"] = (old.get("frames") or {}).get("frames_csv_sha256")
    return d


def process_clip(src: Path, row: dict, sha_expected: str, res: str, root, remake: bool, ctx: dict):
    """Skip, refuse or make one clip. Returns (status, message, record); status in made / failed / skipped / refused."""
    tag = row["tag"]
    out = Path(root) / tag
    fdir, mj, csvp = out / "frames", out / "mask.json", out / "frames.csv"
    old = None
    if mj.exists():
        try:
            old = json.loads(mj.read_text())
        except ValueError:
            return "refused", "mask.json is not readable; left as it is", None
    old_res = ((old or {}).get("frames") or {}).get("res")
    if old_res not in (None, res):
        return "refused", f"stored at {old_res}p; use another --frames-root for {res}p", old
    old_edge = (((old or {}).get("mask") or {}).get("source_rows") or [None])[0]
    if old_edge not in (None, MASK_Y):
        return "refused", f"stored with the mask edge at row {old_edge}, not {MASK_Y}; move it away (not re-used)", old
    if old and old.get("checks_ok") and fdir.is_dir() and not remake:
        same = old.get("tool_sha256") == ctx["tool_sha256"]
        return "skipped", "passing record" + ("" if same else " (made by another version of this tool)"), old
    nonempty = fdir.is_dir() and any(fdir.iterdir())
    if nonempty and not remake:
        return "refused", "frames folder not empty and no passing record (--remake re-makes it)", old
    if nonempty and not csvp.exists():
        return "refused", "--remake: no stored frames.csv to reproduce; left as it is", old
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free
    if free < DISK_MIN_BYTES:
        raise DiskLow(f"{free / 1e9:.1f} GB free < {DISK_MIN_BYTES / 1e9:.1f} GB")
    with TagLock(out):
        rec, tmp, csv_text = make_clip(src, row, sha_expected, res, out, ctx)
        rec["made"] = now()
        if nonempty:                                  # --remake of a stored clip: reproduce or leave it alone
            old_csv = csvp.read_text()
            if csv_text is None or csv_text != old_csv:
                if tmp is not None and tmp.exists():
                    remove_frames_dir(tmp)
                if csv_text is None:
                    why = "; ".join(rec["reasons"]) or "no frames"
                else:
                    a, b = old_csv.splitlines(), csv_text.splitlines()
                    k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                    why = f"frames.csv not reproduced (first difference at line {k + 1}; {len(a)} vs {len(b)} lines)"
                log(ctx, rec, res, "remake-failed", why)
                return "failed", "--remake: " + why + "; old record untouched", rec
            old_dir = out / "frames.old"
            if old_dir.exists():
                remove_frames_dir(old_dir)            # left by an interrupted --remake of this tool
            fdir.rename(old_dir)
            tmp.rename(fdir)
            remove_frames_dir(old_dir)                # replaced by the byte-identical new frames
            rec["previous"] = prev_summary(old)
        else:
            if tmp is not None:
                if fdir.exists():
                    fdir.rmdir()                      # empty (checked above)
                tmp.rename(fdir)
                write_atomic(csvp, csv_text)
        write_atomic(mj, json.dumps(rec, indent=1))
    status = "made" if rec["checks_ok"] else "failed"
    log(ctx, rec, res, status, "; ".join(rec["reasons"]))
    return status, "; ".join(rec["reasons"]) or "all checks passed", rec


def log(ctx, rec, res, status, why):
    p = Path(ctx["log"])
    p.parent.mkdir(parents=True, exist_ok=True)
    n = (rec.get("frames") or {}).get("count", (rec.get("decode") or {}).get("frames_written"))
    with open(p, "a") as f:
        f.write(f"{now()}\t{rec['source']['file']}\t{rec['tag']}\t{res}\t{status}\tframes={n}\t"
                f"checks_ok={str(rec['checks_ok']).lower()}\t{(rec.get('seconds') or {}).get('total')} s\t"
                f"root={rec.get('frames_root')}\t{why}\n")


def clip_line(tag, status, msg, rec) -> str:
    if not rec or status in ("refused", "skipped") and not rec.get("frames"):
        return f"{tag}: {status}: {msg}"
    c, m, fr = rec.get("container") or {}, rec.get("mask") or {}, rec.get("for_run_list") or {}
    n = (rec.get("frames") or {}).get("count", (rec.get("decode") or {}).get("frames_written"))
    return (f"{tag}: {status}: {n} frames / {c.get('pictures')} pictures, fps {c.get('fps')}, irregular "
            f"{rec.get('timing_irregular')}, masked max {m.get('max_in_masked_rows')} / rule rows "
            f"{m.get('max_in_rule_rows')}, first zero row {m.get('first_zero_row')}, "
            f"band {fr.get('band_bottom')}{' SUSPECT' if fr.get('band_suspect') else ''}, grey {fr.get('grey')}, "
            f"{(rec.get('seconds') or {}).get('total')} s; {msg}")


# ----------------------------------------------------------------------------------------------- commands

def select_rows(rows, a, names, sha):
    by_file = {r["file"]: r for r in rows}
    if a.all:
        return list(rows)
    if a.trip:
        known = {r["trip_label"] for r in rows} | {r["trip_id"] for r in rows}
        bad = [t for t in a.trip if t not in known]
        if bad:
            raise Refused(f"unknown trip(s) {bad}")
        return [r for r in rows if r["trip_label"] in a.trip or r["trip_id"] in a.trip]
    sel = []
    for x in a.file:
        p = resolve_source(x, names, sha)
        if p.name not in by_file:
            raise Refused(f"{p.name}: not in the tag table")
        sel.append(by_file[p.name])
    return sel


def cmd_extract(a):
    try:
        names, sha = read_lists()
        rows, tags_sha = load_tags(a.tags, names, sha)
        sel = select_rows(rows, a, names, sha)
        srcs = [resolve_source(r["file"], names, sha) for r in sel]
        root = check_out_root(a.frames_root)
        ctx = make_ctx(Path(a.tags).resolve(), tags_sha, LOG_DEFAULT)
    except (Refused, FileNotFoundError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    counts = {}
    for r, src in zip(sel, srcs):
        try:
            status, msg, rec = process_clip(src, r, sha[r["file"]], a.res, root, a.remake, ctx)
        except DiskLow as e:
            print(f"stopped before {r['tag']}: {e}", file=sys.stderr)
            return 4
        except Refused as e:
            status, msg, rec = "refused", str(e), None
        counts[status] = counts.get(status, 0) + 1
        print(clip_line(r["tag"], status, msg, rec), flush=True)
    print(f"extract: {len(sel)} clips: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), flush=True)
    return 3 if counts.get("failed") or counts.get("refused") else 0


def build_runlist(rows, tags_sha, root, out_dir, band_mode="camera", roadclass=None, tags_path=None):
    """Run list + group lists + summary from the clips' mask.json. Returns the summary."""
    root = Path(root)
    recs, excluded = {}, []
    for r in rows:
        t = r["tag"]
        mj = root / t / "mask.json"
        if not mj.exists():
            excluded.append(dict(tag=t, reason="no mask.json (not extracted)"))
            continue
        d = json.loads(mj.read_text())
        edge = ((d.get("mask") or {}).get("source_rows") or [None])[0]
        if edge != MASK_Y:
            raise Refused(f"{t}: made with the mask edge at row {edge}, not {MASK_Y}; masks are not mixed in a run list")
        if not d.get("checks_ok"):
            excluded.append(dict(tag=t, reason="checks failed: " + "; ".join(d.get("reasons") or [])))
            continue
        if not (root / t / "frames").is_dir():
            excluded.append(dict(tag=t, reason="frames folder missing"))
            continue
        recs[t] = d
    sizes = {d["frames"]["res"] for d in recs.values()}
    if len(sizes) > 1:
        raise Refused(f"mixed resolutions {sorted(sizes)} in {root}")
    fr = {t: d["for_run_list"] for t, d in recs.items()}
    day_ok = [v["band_bottom"] for v in fr.values() if not v["night"] and not v["band_suspect"]]
    cam = min(BAND_MAX, round(statistics.median(day_ok), 3)) if day_ok else BAND_MAX
    lines, groups, clips = [], {"main": [], "local_day": [], "night": []}, {}
    for r in rows:
        t = r["tag"]
        if t not in recs:
            continue
        v = fr[t]
        if band_mode == "camera":
            band = cam
        elif band_mode == "clip":
            band = cam if v["band_suspect"] else min(BAND_MAX, v["band_bottom"])
        else:
            band = min(BAND_MAX, float(band_mode))
        cam_, drive, seg = t.split("_")
        fdir = (root / t / "frames").resolve()
        lines.append(f"{cam_}|{drive}|{seg}|dod|taiwan|{v['fps']:.4f}|{fdir}|{band:.3f}")
        if v["night"]:
            g = "night"
        else:
            g = "local_day" if (roadclass or {}).get(t, (roadclass or {}).get(drive, "main")) == "local" else "main"
        groups[g].append(t)
        clips[t] = dict(band_bottom=v["band_bottom"], band_used=band, band_suspect=v["band_suspect"], grey=v["grey"],
                        night=v["night"], group=g, timing_irregular=recs[t]["timing_irregular"], fps=v["fps"],
                        tail_missing=recs[t]["decode"]["tail_missing"], tool_sha256=recs[t]["tool_sha256"])
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / "run_list.txt").write_text("".join(x + "\n" for x in lines))
    for g, tt in groups.items():
        (od / f"tags_{g}.txt").write_text("".join(x + "\n" for x in tt))
    vals = sorted(v["band_bottom"] for v in fr.values())
    tools_used = {}
    for d in recs.values():
        tools_used[d["tool_sha256"]] = tools_used.get(d["tool_sha256"], 0) + 1
    summ = dict(made=now(), tool_sha256=sha256_file(Path(__file__).resolve()), tags_table=tags_path,
                tags_table_sha256=tags_sha, frames_root=str(root.resolve()), res=next(iter(sizes), None),
                band_mode=band_mode, band_camera=cam, band_camera_fallback=not day_ok, band_max=BAND_MAX,
                daytime_non_suspect=len(day_ok),
                band_values=dict(n=len(vals), min=vals[0] if vals else None,
                                 median=statistics.median(vals) if vals else None, max=vals[-1] if vals else None),
                night_rule=f"grey < {NIGHT_GREY:g}",
                near_night_threshold=sorted(t for t, v in fr.items() if v["grey"] is not None
                                            and abs(v["grey"] - NIGHT_GREY) <= 5),
                groups={g: len(v) for g, v in groups.items()}, clips=clips, excluded=excluded,
                tool_versions=tools_used, lines=len(lines))
    (od / "runlist_summary.json").write_text(json.dumps(summ, indent=1))
    return summ


def read_roadclass(path, rows):
    if not path:
        return None
    known = {r["tag"] for r in rows} | {r["trip_id"] for r in rows}
    rc = {}
    for l in Path(path).read_text().splitlines():
        if not l.strip() or l.startswith("#"):
            continue
        k, v = [x.strip() for x in l.split("\t")[:2]]
        if v not in ("main", "local"):
            raise Refused(f"road class {v!r}: main or local")
        if k not in known:
            raise Refused(f"road class key {k!r}: not a tag or trip_id of the tag table")
        rc[k] = v
    return rc


def cmd_runlist(a):
    try:
        names, sha = read_lists()
        rows, tags_sha = load_tags(a.tags, names, sha)
        rc = read_roadclass(a.roadclass, rows)
        if a.only:
            keep = {l.strip() for l in Path(a.only).read_text().splitlines() if l.strip() and not l.startswith("#")}
            unknown = keep - {r["tag"] for r in rows}
            if unknown:
                raise Refused(f"--only: unknown tags {sorted(unknown)[:3]}")
            rows = [r for r in rows if r["tag"] in keep]
        if a.band not in ("camera", "clip"):
            float(a.band)
        summ = build_runlist(rows, tags_sha, a.frames_root, a.out_dir, a.band, rc, str(Path(a.tags).resolve()))
    except (Refused, ValueError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    print(json.dumps({k: summ[k] for k in ("lines", "band_mode", "band_camera", "band_camera_fallback",
                                           "daytime_non_suspect", "groups", "near_night_threshold")}))
    if summ["excluded"]:
        print(f"excluded {len(summ['excluded'])}: " + "; ".join(f"{x['tag']} ({x['reason']})" for x in summ["excluded"][:5]))
    return 0


def verify_tag(d: Path):
    problems = []
    rec = json.loads((d / "mask.json").read_text())
    csvp = d / "frames.csv"
    if not csvp.exists():
        return rec, ["no frames.csv"]
    if sha256_file(csvp) != (rec.get("frames") or {}).get("frames_csv_sha256"):
        problems.append("frames.csv differs from mask.json")
    listed = list(csv.DictReader(open(csvp, newline="")))
    fdir = d / "frames"
    present = sorted(p.name for p in fdir.iterdir()) if fdir.is_dir() else []
    if present != [x["name"] for x in listed]:
        problems.append(f"{len(present)} files in frames/, {len(listed)} in frames.csv")
    if len(listed) != (rec.get("frames") or {}).get("count"):
        problems.append(f"frames.csv has {len(listed)} rows, mask.json says {(rec.get('frames') or {}).get('count')}")
    bad = sum(1 for x in listed if not (fdir / x["name"]).is_file()
              or hashlib.sha256((fdir / x["name"]).read_bytes()).hexdigest() != x["sha256"])
    if bad:
        problems.append(f"{bad} frames differ from frames.csv")
    return rec, problems


def cmd_verify(a):
    root = Path(a.frames_root)
    tags = a.tag or sorted(p.name for p in root.iterdir() if (p / "mask.json").exists())
    n_bad = 0
    for t in tags:
        if not (root / t / "mask.json").exists():
            print(f"{t}: no mask.json")
            n_bad += 1
            continue
        rec, problems = verify_tag(root / t)
        n_bad += bool(problems)
        print(f"{t}: {'ok' if not problems else 'DIFFERS: ' + '; '.join(problems)} "
              f"({(rec.get('frames') or {}).get('count')} frames, checks_ok {rec.get('checks_ok')})")
    print(f"verify: {len(tags)} clips, {n_bad} with problems")
    return 3 if n_bad else 0


def make_sheet(tag_dir: Path, out_file: Path):
    """3 x 3 contact sheet + the rows just above the mask (3 frames), from the masked frames only."""
    files = sorted((tag_dir / "frames").glob("f*.jpg"))
    if not files:
        return None
    rec = json.loads((tag_dir / "mask.json").read_text())
    y0 = rec["mask"]["rows_at_res"][0]
    n = len(files)
    tiles = [cv2.resize(cv2.imread(str(files[int(i)])), (640, 360), interpolation=cv2.INTER_AREA)
             for i in np.linspace(0, n - 1, 9).round().astype(int)]
    grid = np.vstack([np.hstack(tiles[r * 3:(r + 1) * 3]) for r in range(3)])
    strips = []
    for q in (0.25, 0.5, 0.75):
        img = cv2.imread(str(files[int(round(q * (n - 1)))]))
        h, w = img.shape[:2]
        band = img[max(0, y0 - 100): min(h, y0 + 20)]
        s = 1920 / w
        strips.append(cv2.resize(band, (1920, int(round(band.shape[0] * s))), interpolation=cv2.INTER_AREA))
        strips.append(np.full((6, 1920, 3), 90, np.uint8))
    head = np.full((44, 1920, 3), 40, np.uint8)
    cv2.putText(head, f"{tag_dir.name}  {n} frames  masked rows >= {y0}  (below: rows {max(0, y0 - 100)}-{y0 + 19} "
                      f"of frames at 25 / 50 / 75 %)", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    sheet = np.vstack([head, grid, np.full((10, 1920, 3), 90, np.uint8), *strips])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return sheet.shape


def cmd_sheet(a):
    root, out = Path(a.frames_root), Path(a.out)
    try:
        check_out_root(out)
        if a.trip:
            names, sha = read_lists()
            rows, _ = load_tags(a.tags, names, sha)
            tags = [r["tag"] for r in rows if r["trip_label"] in a.trip or r["trip_id"] in a.trip]
        else:
            tags = a.tag or sorted(p.name for p in root.iterdir() if (p / "mask.json").exists())
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    for t in tags:
        out_file = out / f"{t}_sheet.jpg"
        shape = make_sheet(root / t, out_file) if (root / t / "mask.json").exists() else None
        print(f"{t}: no frames" if shape is None else f"{t}: {shape[1]} x {shape[0]} -> {out_file}")
    return 0


# ----------------------------------------------------------------------------------------------- selftest

def cmd_selftest(a):
    work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="dod_frames_selftest_"))
    work.mkdir(parents=True, exist_ok=True)
    res_ok = []

    def check(name, ok, detail=""):
        res_ok.append(bool(ok))
        print(f"  [{'pass' if ok else 'FAIL'}] {name}{'  (' + str(detail) + ')' if detail != '' else ''}", flush=True)

    print(f"selftest in {work}")
    syn = work / "2000_01_01_000000_00.MP4"
    r = subprocess.run([str(FFMPEG), "-nostdin", "-hide_banner", "-v", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=1920x1080:rate=25:duration=4", "-vf",
                        "drawbox=x=0:y=860:w=1920:h=220:color=white:t=fill,format=yuvj420p", "-c:v", "libx265",
                        "-x265-params", "log-level=error", "-tag:v", "hvc1", "-video_track_timescale", "120000",
                        "-y", str(syn)], capture_output=True)
    check("synthetic HEVC clip made (100 frames, white box over rows 860-1079)", r.returncode == 0 and syn.exists())
    row = dict(file=syn.name, tag="dod_2000-01-01--00-00-00_000000-00", trip_label="T00", trip_id="2000-01-01--00-00-00",
               start="2000-01-01 00:00:00", duration_s="4.00", pictures="100")
    sha = sha256_file(syn)
    ctx = make_ctx(None, "selftest", work / "selftest.log")
    recs = {}
    for res in ("720", "1080"):
        st, msg, rec = process_clip(syn, row, sha, res, work / f"root_{res}", False, ctx)
        recs[res] = rec
        print(f" {res}p: {st}: {msg}")
        c, m, d = rec["container"], rec["mask"], rec["decode"]
        check(f"{res}: 100 pictures, 100 JPEGs, showinfo pts = container pts", c["pictures"] == 100
              and rec["frames"]["count"] == 100 and d["pts_prefix_match"] and d["tail_missing"] == 0)
        check(f"{res}: fps 25.0000, not timing_irregular", c["fps"] == 25.0 and rec["timing_irregular"] is False,
              f"fps {c['fps']}")
        check(f"{res}: every pixel of the rule rows (>= {RES[res]['rule_row']}) exactly 0", m["max_in_rule_rows"] == 0,
              m["max_in_rule_rows"])
        check(f"{res}: rows between the mask edge and the rule rows recorded (white box touching the edge: worst case)",
              m["edge_row_max"] is not None and m["first_zero_row"] is not None,
              f"per-row max {m['edge_row_max']}, first zero row {m['first_zero_row']}")
        check(f"{res}: checks_ok, frames kept, band bottom and grey computed", rec["checks_ok"]
              and (work / f"root_{res}" / row["tag"] / "frames").is_dir() and rec["for_run_list"] is not None,
              (rec["for_run_list"] or {}).get("band_bottom"))
    # a chain without the mask must be caught and its frames removed (the safety path of the read-back check)
    keep_vf = RES["720"]["vf"]
    RES["720"]["vf"] = "scale=1280:720:flags=area"
    try:
        st, msg, rec_l = process_clip(syn, row, sha, "720", work / "root_leak", False, ctx)
    finally:
        RES["720"]["vf"] = keep_vf
    check("an unmasked chain fails the mask check and its frames are removed", st == "failed"
          and rec_l["mask"]["frames_removed"] and not (work / "root_leak" / row["tag"] / "frames").exists()
          and not (work / "root_leak" / row["tag"] / "frames.tmp").exists(), rec_l["mask"]["max_in_rule_rows"])
    check("mask edge on the 16-row JPEG block grid at both sizes, 2/3 of it at 720p, rule rows below it",
          MASK_Y % 16 == 0 and RES["720"]["mask_row"] % 16 == 0 and RES["720"]["mask_row"] * 3 == MASK_Y * 2
          and RES["720"]["rule_row"] * 3 == RULE_Y * 2 and RES["1080"]["mask_row"] == MASK_Y
          and RES["1080"]["rule_row"] == RULE_Y and MASK_Y <= RULE_Y, f"{MASK_Y} / {RES['720']['mask_row']}")
    root = work / "root_720"
    tdir = root / row["tag"]
    # determinism and --remake
    st, msg, rec2 = process_clip(syn, row, sha, "720", work / "root_720b", False, ctx)
    same = (tdir / "frames.csv").read_bytes() == (work / "root_720b" / row["tag"] / "frames.csv").read_bytes()
    check("two extractions give identical frames.csv", same)
    st, msg, _ = process_clip(syn, row, sha, "720", root, False, ctx)
    check("a passing clip is skipped", st == "skipped", st)
    st, msg, _ = process_clip(syn, row, sha, "720", root, True, ctx)
    check("--remake reproduces the stored frames.csv", st == "made" and "previous" in json.loads((tdir / "mask.json").read_text()), st)
    old_csv = (tdir / "frames.csv").read_text()
    bad = old_csv.replace(old_csv.splitlines()[5].split(",")[-1], "0" * 64, 1)
    (tdir / "frames.csv").write_text(bad)
    before = (tdir / "mask.json").read_bytes(), (tdir / "frames.csv").read_bytes(), sorted(p.name for p in (tdir / "frames").iterdir())
    st, msg, _ = process_clip(syn, row, sha, "720", root, True, ctx)
    after = (tdir / "mask.json").read_bytes(), (tdir / "frames.csv").read_bytes(), sorted(p.name for p in (tdir / "frames").iterdir())
    check("--remake that does not reproduce fails and leaves the record untouched", st == "failed" and before == after
          and not (tdir / "frames.tmp").exists(), msg[:60])
    (tdir / "frames.csv").write_text(old_csv)
    st, msg, _ = process_clip(syn, row, sha, "1080", root, False, ctx)
    check("another --res in the same frames root is refused", st == "refused", msg)
    # refusals of the input checks (the development lists are read; no recorder file is opened)
    names, shas = read_lists()
    listed = sorted(names)[0]
    lk = work / "links"
    lk.mkdir(exist_ok=True)
    if not (lk / listed).is_symlink():
        os.symlink(RAW_DEV / listed, lk / listed)
    for arg, what in (("/mnt/e/front_norm/" + listed, "a path on the external drive"),
                      ("/mnt/e/../d/depth_work/tw_own/raw_dev/" + listed, "a path through the external drive"),
                      ("E:\\front_norm\\" + listed, "a Windows path"),
                      ("2026_09_21_000000_00.MP4", "a name not in the lists"),
                      (str(syn), "an unlisted file outside raw_dev"),
                      (str(TW / listed), "a listed name in another folder"),
                      (str(lk / listed), "a symlink outside raw_dev to a listed file"),
                      ("/mnt/d/depth_work/tw_own/raw_dev/../raw_dev/notes.txt", "not a recorder file name")):
        try:
            resolve_source(arg, names, shas)
            check(f"refused: {what}", False)
        except Refused as e:
            check(f"refused: {what}", True, str(e)[:70])
    nonempty = work / "root_x" / row["tag"] / "frames"
    nonempty.mkdir(parents=True, exist_ok=True)
    shutil.copy(tdir / "frames" / "f00001.jpg", nonempty / "f00001.jpg")
    st, msg, _ = process_clip(syn, row, sha, "720", work / "root_x", False, ctx)
    check("a non-empty frames folder without a passing record is refused", st == "refused", msg)
    (work / "repo_like" / ".git").mkdir(parents=True, exist_ok=True)
    for pth, what in ((work / "repo_like" / "frames", "a frames root inside a git work tree"),
                      (Path("/mnt/e/frames"), "a frames root on the external drive")):
        try:
            check_out_root(pth)
            check(f"refused: {what}", False)
        except Refused:
            check(f"refused: {what}", True)
    check("tail rule: 0 missing ok; 1-2 only on an irregular file", tail_ok(1500, 1500, False) and tail_ok(901, 900, True)
          and tail_ok(1101, 1099, True) and not tail_ok(1500, 1499, False) and not tail_ok(1442, 1439, True)
          and not tail_ok(100, 101, True))
    # run list
    summ = build_runlist([row], "selftest", root, work / "lists", "camera")
    rl = (work / "lists" / "run_list.txt").read_text()
    r = subprocess.run(["bash", "-c", 'IFS="|" read -r d r s ch road fps fdir band < "$1"; '
                        'printf "%s\\n" "$d" "$r" "$s" "$ch" "$road" "$fps" "$fdir" "$band"', "_",
                        str(work / "lists" / "run_list.txt")], capture_output=True, text=True)
    f = r.stdout.splitlines()
    ok = (len(f) == 8 and f[:6] == ["dod", row["trip_id"], "000000-00", "dod", "taiwan", "25.0000"]
          and Path(f[6]).is_dir() and re.fullmatch(r"0\.\d{3}", f[7]) is not None
          and "_".join(f[:3]) == row["tag"])
    check("run-list line parses with IFS='|' read -r d r s ch road fps fdir band", ok, rl.strip()[-40:])
    check("camera band from the daytime clip, <= 0.780", summ["band_camera"] <= BAND_MAX and not summ["band_camera_fallback"],
          summ["band_camera"])
    oth = work / "root_other_mask"
    shutil.copytree(tdir, oth / row["tag"])
    mj = oth / row["tag"] / "mask.json"
    d = json.loads(mj.read_text())
    d["mask"]["source_rows"][0] = 870
    mj.write_text(json.dumps(d))
    st, msg, _ = process_clip(syn, row, sha, "720", oth, False, ctx)
    try:
        build_runlist([row], "selftest", oth, work / "lists_other", "camera")
        refused_rl = False
    except Refused:
        refused_rl = True
    check("a record made with another mask edge is neither re-used nor put in a run list", st == "refused" and refused_rl,
          msg[:60])
    # verify and sheet
    rec_v, problems = verify_tag(work / "root_720b" / row["tag"])
    check("verify passes on an untouched clip", not problems, problems)
    fb = work / "root_720b" / row["tag"] / "frames" / "f00050.jpg"
    fb.write_bytes(fb.read_bytes() + b"\0")
    _, problems = verify_tag(work / "root_720b" / row["tag"])
    check("verify finds a changed frame", any("differ" in p for p in problems), problems)
    shape = make_sheet(tdir, work / "sheets" / f"{row['tag']}_sheet.jpg")
    check("sheet made from the masked frames", shape is not None and shape[1] == 1920, shape)
    check("disk guard refuses below the limit", shutil.disk_usage(work).free < (1 << 62))
    n_pass, n = sum(res_ok), len(res_ok)
    print(f"selftest: {n_pass} / {n} passed" + ("" if n_pass == n else f", {n - n_pass} FAILED"))
    if not a.keep and not a.work:
        shutil.rmtree(work)
    return 0 if n_pass == n else 1


def use_set(name):
    """Switch the module's input / output defaults to an input set before anything is parsed or opened."""
    global RAW_DEV, FILES_LIST, SHA_LIST, TAGS_DEFAULT, FRAMES_ROOT_DEFAULT, LOG_DEFAULT, INPUT_SET
    s = SETS[name]
    RAW_DEV, FILES_LIST, SHA_LIST = s["raw"], s["files"], s["sha"]
    TAGS_DEFAULT, FRAMES_ROOT_DEFAULT, LOG_DEFAULT, INPUT_SET = s["tags"], s["frames"], s["log"], name


def main():
    argv = sys.argv[1:]
    name = "dev"
    if "--set" in argv:
        i = argv.index("--set")
        if i + 1 >= len(argv) or argv[i + 1] not in SETS:
            print(f"refused: --set must be one of {sorted(SETS)}", file=sys.stderr)
            return 2
        name = argv[i + 1]
        del argv[i:i + 2]
    use_set(name)
    if name != "dev" and argv and argv[0] in ("sheet", "selftest"):
        print(f"refused: {argv[0]} is for the development set only (blind frames are not looked at before the seal)",
              file=sys.stderr)
        return 2
    sys.argv = [sys.argv[0]] + argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    g = e.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", nargs="+")
    g.add_argument("--trip", nargs="+")
    g.add_argument("--all", action="store_true")
    e.add_argument("--tags", default=str(TAGS_DEFAULT))
    e.add_argument("--res", choices=("720", "1080"), default="720")
    e.add_argument("--frames-root", default=str(FRAMES_ROOT_DEFAULT))
    e.add_argument("--remake", action="store_true")
    r = sub.add_parser("runlist")
    r.add_argument("--tags", default=str(TAGS_DEFAULT))
    r.add_argument("--frames-root", default=str(FRAMES_ROOT_DEFAULT))
    r.add_argument("--out-dir", required=True)
    r.add_argument("--band", default="camera", help="camera | clip | a value")
    r.add_argument("--roadclass", default=None)
    r.add_argument("--only", default=None, help="file with the tags to keep")
    v = sub.add_parser("verify")
    v.add_argument("--frames-root", default=str(FRAMES_ROOT_DEFAULT))
    v.add_argument("--tag", nargs="+")
    s = sub.add_parser("sheet")
    s.add_argument("--frames-root", default=str(FRAMES_ROOT_DEFAULT))
    s.add_argument("--out", required=True)
    s.add_argument("--tags", default=str(TAGS_DEFAULT))
    gs = s.add_mutually_exclusive_group()
    gs.add_argument("--tag", nargs="+")
    gs.add_argument("--trip", nargs="+")
    t = sub.add_parser("selftest")
    t.add_argument("--work", default=None, help="folder for the synthetic clip (default: a new temporary folder)")
    t.add_argument("--keep", action="store_true", help="keep the temporary folder")
    a = ap.parse_args()
    return {"extract": cmd_extract, "runlist": cmd_runlist, "verify": cmd_verify, "sheet": cmd_sheet,
            "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

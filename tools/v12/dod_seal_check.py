#!/usr/bin/env python3
"""Seals of the DOD blind predictions: make the public digest of a private seal, and check seals before any answer is read.

The DOD file names carry the dates and times of private drives, so the full seals (tools/seal_predictions.py: file name ->
sha256) stay on D:. What is committed is a digest per seal: the sha256 of the private seal file, the number of files, the
seal time. The digest pins every prediction file through the private seal.

  digest  --seal PRIVATE.json --what TEXT --out docs/seals/NAME.json
  check   --tags TAGS.tsv --seal docs/seals/A.json [...] --seal-private PRIVATE_A.json [...] --pred-dir DIR_A [...]
          (same order). For every triple: the digest is committed as it is and its commit is on origin/main (git fetch
          first); the private seal's sha256 equals the digest's; every file the seal lists is in the folder with the same
          sha256; every tag of the tags table has at least one sealed file whose name starts with "<tag>_". Exit 1 otherwise.
Reads no answer.
"""
import argparse, csv, hashlib, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def sha256_file(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)


def cmd_digest(a):
    seal = json.loads(Path(a.seal).read_text())
    out = dict(what=a.what, private_seal_sha256=sha256_file(a.seal), files=len(seal.get("files") or {}),
               inputs=len(seal.get("inputs") or {}), sealed_utc=seal.get("sealed_utc"), git_commit=seal.get("git_commit"),
               absent_at_seal=seal.get("absent_at_seal"),
               note="the full seal (file names with recording dates and times) is kept privately; this digest pins it")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{a.out}: {out['files']} files, sealed {out['sealed_utc']}")


def fail(msg):
    print("refused: " + msg, file=sys.stderr)
    sys.exit(1)


def cmd_check(a):
    if not (len(a.seal) == len(a.seal_private) == len(a.pred_dir)) or not a.seal:
        fail("--seal, --seal-private and --pred-dir must come in equal numbers")
    tags = [r["tag"] for r in csv.DictReader(open(a.tags), delimiter="\t")]
    if not tags:
        fail("no tags")
    if not a.no_github and git("fetch", "-q", "origin", "main").returncode != 0:
        fail("git fetch origin main failed: cannot confirm the seals are on GitHub")
    for dg, pv, pd in zip(a.seal, a.seal_private, a.pred_dir):
        dgp = Path(dg).resolve()
        try:
            rel = str(dgp.relative_to(REPO))
        except ValueError:
            fail(f"{dgp}: a digest must be a file of this repository")
        if not a.no_github:
            if git("ls-files", "--error-unmatch", rel).returncode != 0 or git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
                fail(f"{rel}: not committed as it is")
            c = git("log", "-1", "--format=%H", "--", rel).stdout.strip()
            if not c or git("merge-base", "--is-ancestor", c, "origin/main").returncode != 0:
                fail(f"{rel}: its commit is not on origin/main (push first)")
        d = json.loads(dgp.read_text())
        if sha256_file(pv) != d.get("private_seal_sha256"):
            fail(f"{pv}: not the seal that {rel} pins")
        files = json.loads(Path(pv).read_text()).get("files") or {}
        if len(files) != d.get("files"):
            fail(f"{pv}: {len(files)} files, the digest says {d.get('files')}")
        miss = [t for t in tags if not any(n.startswith(t + "_") for n in files)]
        if miss:
            fail(f"{rel}: {len(miss)} tags have no sealed file")
        bad = [n for n, h in files.items() if not (Path(pd) / n).is_file() or sha256_file(Path(pd) / n) != h]
        if bad:
            fail(f"{pd}: {len(bad)} sealed files missing or changed")
        print(f"seal ok: {rel} ({len(files)} files, sealed {d.get('sealed_utc')}" + ("" if a.no_github else ", on origin/main") + ")")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("digest")
    d.add_argument("--seal", required=True)
    d.add_argument("--what", required=True)
    d.add_argument("--out", required=True)
    c = sp.add_parser("check")
    c.add_argument("--tags", required=True)
    c.add_argument("--seal", nargs="+", required=True)
    c.add_argument("--seal-private", nargs="+", required=True)
    c.add_argument("--pred-dir", nargs="+", required=True)
    c.add_argument("--no-github", action="store_true", help="tests only: skip the committed / pushed checks")
    a = ap.parse_args()
    cmd_digest(a) if a.cmd == "digest" else cmd_check(a)


if __name__ == "__main__":
    main()

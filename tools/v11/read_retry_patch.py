"""v11extra: re-read a depth map that came back short (/mnt/d returns partial reads under many concurrent readers:
'Failed to read all data ... could only read 1984 elements' = the first 4 KiB; the file is complete on disk and a second
read returns all of it). Applied to work copies only; a re-read returns the same bytes, so no result changes."""
from pathlib import Path

OLD = '''    if fp is not None and fp.exists():
        return np.load(fp).astype(np.float32)'''
NEW = '''    if fp is not None and fp.exists():
        for _k in range(5):                        # v11extra: /mnt/d short reads under load; a re-read returns the same data
            try:
                return np.load(fp).astype(np.float32)
            except (ValueError, OSError):
                if _k == 4:
                    raise
                __import__("time").sleep(2.0)'''


def apply(path):
    p = Path(path)
    t = p.read_text()
    if NEW in t:
        return False
    assert t.count(OLD) == 1, path
    p.write_text(t.replace(OLD, NEW))
    return True


if __name__ == "__main__":
    import sys
    for a in sys.argv[1:]:
        print(a, apply(a))

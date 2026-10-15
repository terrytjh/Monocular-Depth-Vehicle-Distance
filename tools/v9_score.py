#!/usr/bin/env python3
"""Ninth version scoring (docs/DEPTH_DASH_V9_PREREG.md section 3), comma2k19, corrected truth (radar - 0.33 m).

tools/v8_score.py unchanged (seal check before any truth is read, true distance within 30 m first, % and metres, the
0-10 / 10-20 / 20-30 m bins, speeds, one-sided sign test over segments with >= 10 paired readings), with these steps on
the same segments:

  c    marking geometry alone, each segment's own A (*_cars_c.json)
  cca  marking geometry alone, the camera's A (*_cars_cca.json)
  v7   the registered seventh version (*_cars_v7.json)
  v8a  the registered eighth version's S5 (*_cars_v8a.json)
  v8   the registered eighth version (*_cars_v8.json)
  v9a  + S7 at the S5 step (*_cars_v9a.json)
  v9b  + S7 with the eighth version's S6 (*_cars_v9b.json)
  v9   + S8 on daytime local roads (*_cars_v9.json; the ninth version)

and these pairs on the same readings: v9 vs v8 (the whole ninth version), v9a vs v8a (S7 alone at S5), v9b vs v8 (S7 with
S6), v9 vs v9b (S8 alone), cca vs c (S7 on marking geometry alone).

  python3 tools/v9_score.py --batch DIR --seal DIR/seal.json --out DIR/score_v9.json
  python3 tools/v9_score.py --batch DIR --dev-opened data/output/dash_scale/blind_v8 --out X.json    # opened data only
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v8_score as S  # noqa: E402

S.STEPS = [("c", "_cars_c.json"), ("cca", "_cars_cca.json"), ("v7", "_cars_v7.json"), ("v8a", "_cars_v8a.json"),
           ("v8", "_cars_v8.json"), ("v9a", "_cars_v9a.json"), ("v9b", "_cars_v9b.json"), ("v9", "_cars_v9.json")]
S.PAIRS = [("v9", "v8"), ("v9a", "v8a"), ("v9b", "v8"), ("v9", "v9b"), ("cca", "c")]
S.__doc__ = __doc__

if __name__ == "__main__":
    S.main()

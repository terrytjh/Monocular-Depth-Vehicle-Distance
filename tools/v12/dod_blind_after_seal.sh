#!/usr/bin/env bash
# First DOD blind test, after the seals (docs/DOD_V12_BLIND_PREREG.md section 4, step 5): the overlay speed of the blind
# clips is read only after tools/v12/dod_seal_check.py has confirmed the five digests on origin/main, the private seals
# they pin and every sealed file; then the comparison of the eleventh and twelfth version and the registered numbers.
set -u
REPO=/home/terry/Monocular-Depth-Vehicle-Distance
TW=/mnt/d/depth_work/tw_own
RUN=${RUN:-$TW/blind1_v12}
P=/home/terry/venvs/depthbench/bin/python
[ -f "$HOME/miniconda3/lib/libstdc++.so.6" ] && export LD_PRELOAD=$HOME/miniconda3/lib/libstdc++.so.6
cd $REPO
LOG=$RUN/logs/after_seal.log
log () { echo "=== $* $(date '+%F %T')" | tee -a "$LOG"; }
stop () { log "$*: stopping"; exit 1; }
[ -e "$RUN/BLIND1_SEALED" ] || stop "not sealed"
N="v10 v11 v12 ego_v11 ego_v12"
DG=$(for n in $N; do echo docs/seals/dod_blind1_${n}_seal.json; done)
PV=$(for n in $N; do echo $RUN/seals/private/$n.json; done)
PD="$RUN/v10 $RUN/v11 $RUN/v12 $RUN/ego $RUN/ego_v12"
TAGS=$RUN/lists/tags_blind1_run.tsv
$P -B tools/v12/dod_seal_check.py check --tags $TAGS --seal $DG --seal-private $PV --pred-dir $PD >> "$LOG" 2>&1 || stop "seal check"
log "seals confirmed; reading the overlay of the blind clips"
if [ ! -e "$TW/osd_blind1/.ALL_DONE_rc0" ]; then
  nice -n 10 $P -B tools/dod_osd.py --set blind1 read --all --tags $TAGS --templates $TW/osd_templates/v1 --out $TW/osd_blind1 \
     --frames-csv --seal $DG --seal-private $PV --pred-dir $PD >> "$RUN/logs/osd_read.log" 2>&1 || stop "dod_osd.py read"
  touch "$TW/osd_blind1/.ALL_DONE_rc0"
fi
mkdir -p "$RUN/view_v12"; [ -e "$RUN/view_v12/ego" ] || ln -s ../ego_v12 "$RUN/view_v12/ego"; [ -e "$RUN/view_v12/lists" ] || ln -s ../lists "$RUN/view_v12/lists"
$P -B tools/dod_compare.py --run "$RUN" --osd $TW/osd_blind1 --frames $TW/frames_blind1 --out "$RUN/cmp_v11" >> "$LOG" 2>&1 || stop "compare v11"
$P -B tools/dod_compare.py --run "$RUN/view_v12" --osd $TW/osd_blind1 --frames $TW/frames_blind1 --out "$RUN/cmp_v12" >> "$LOG" 2>&1 || stop "compare v12"
$P -B tools/v12/dod_blind_score.py --cmp11 "$RUN/cmp_v11" --cmp12 "$RUN/cmp_v12" --out "$RUN/score_blind1.json" \
   --label "first DOD blind test (60 clips drawn from 2026-09-26 .. 10-03)" | tee -a "$LOG"
log "scored: $RUN/score_blind1.json"

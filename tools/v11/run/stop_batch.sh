#!/usr/bin/env bash
# Stop a run_batch.sh run: kill its process group by the number in RUN_DIR/PID_run_batch (never by name).
RUN=$(realpath -m "$1")
p=$(cat "$RUN/PID_run_batch" 2>/dev/null) || { echo "no PID file in $RUN"; exit 1; }
if kill -0 "$p" 2>/dev/null; then kill -- -"$p" && echo "stopped process group $p ($(date '+%F %T'))" | tee -a "$RUN/logs/run_batch.log"; else echo "process $p is not running"; fi

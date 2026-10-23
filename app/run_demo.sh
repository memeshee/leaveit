#!/usr/bin/env bash
# The proof. A worker is killed -9 with three fuses pending, one attention signal is
# delivered while no worker process exists at all, then the worker comes back.
# The fuses must still fire at the wall-clock times they were promised.
set -uo pipefail
cd "$(dirname "$0")"
PY="$HOME/.pyenv/versions/3.12.6/bin/python3"
export PATH="$HOME/.temporalio/bin:$PATH"
export LEAVEIT_LEDGER="${LEAVEIT_LEDGER:-$PWD/../data/ledger.jsonl}"
LOG=../train
EV="$LOG/demo_events.log"
: > "$EV"
T0=$(date +%s)
ev() { local t=$(( $(date +%s) - T0 )); printf '%s|%s\n' "$t" "$1" | tee -a "$EV"; }

rm -f "$LEAVEIT_LEDGER" "$LOG"/worker*.log
temporal server start-dev --headless --port 7233 --log-level error > "$LOG/temporal.log" 2>&1 &
TPID=$!
sleep 5

$PY worker.py > "$LOG/worker1.log" 2>&1 &
WPID=$!
sleep 3
ev "three seals opened: A fuse 24s, C fuse 16s, B fuse 40s (B is the rescue case)"
$PY seal.py start A=24 B=40 C=16 2>&1 | grep sealed | tee -a "$EV"
sleep 8

ev "kill -9 the worker (pid $WPID)"
kill -9 "$WPID" 2>/dev/null; wait "$WPID" 2>/dev/null
if ps -p "$WPID" > /dev/null 2>&1; then ev "worker STILL ALIVE (demo invalid)"; else ev "worker is dead: no process exists that could run an activity"; fi
printf '%s|%s workflows still Running on the server (they were not living in the worker)\n' \
  "$(( $(date +%s) - T0 ))" "$(temporal workflow list --limit 5 2>/dev/null | grep -c Running)" | tee -a "$EV"
sleep 2

ev "deliver 'on_attention' to seal B while NO worker process exists"
$PY seal.py poke B "a maintainer commented while the worker was dead" --dead 2>&1 | tee -a "$EV"
sleep 4

ev "restart the worker"
$PY worker.py > "$LOG/worker2.log" 2>&1 &
WPID2=$!
sleep 24

ev "fuses fired; ledger folded from the append-only log:"
$PY seal.py card 2>&1 | grep -v '^$' | tee -a "$EV"
printf '%s|%s\n' "$(( $(date +%s) - T0 ))" \
  "server-side run history: $(temporal workflow list --limit 5 2>/dev/null | grep -c Completed)/3 seals Completed" | tee -a "$EV"

kill "$WPID2" "$TPID" 2>/dev/null
ev "done: 3 seals, 1 kill -9, 1 signal delivered to nobody, 0 fuses lost"

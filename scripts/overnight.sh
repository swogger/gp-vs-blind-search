#!/bin/sh
# Launch a phase in the background so it runs unattended overnight: no terminal and no
# Claude session needed. It keeps the Mac awake (caffeinate), survives closing the
# terminal (nohup), and resumes where it stopped if launched again.
#
#   scripts/overnight.sh PHASE [run options, e.g. --workers 6]
#
# Keep the laptop plugged in with the lid open (the screen may turn off). If the Mac sleeps
# anyway, runs pause and continue on wake; CPU-time limits are unaffected by sleep.
set -eu
cd "$(dirname "$0")/.."
PHASE="${1:?usage: scripts/overnight.sh PHASE [run options]}"
shift
DIR="data/runs/$PHASE"
mkdir -p "$DIR"
if [ -f "$DIR/queue.pid" ] && kill -0 "$(cat "$DIR/queue.pid")" 2>/dev/null; then
    echo "phase $PHASE is already running (pid $(cat "$DIR/queue.pid"))"
    exit 1
fi
if [ -f "$DIR/STOP" ]; then
    echo "removing old STOP file $DIR/STOP"
    rm -f "$DIR/STOP"
fi
nohup caffeinate -is .venv/bin/python -m reach.runner run "$PHASE" "$@" >> "$DIR/console.log" 2>&1 &
echo $! > "$DIR/queue.pid"
echo "started $PHASE in the background (pid $!). You can close this terminal."
echo "  progress: tail -f $DIR/console.log"
echo "  status:   .venv/bin/python -m reach.runner status $PHASE"
echo "  drain:    touch $DIR/STOP   (running tasks finish; launch again later to resume)"

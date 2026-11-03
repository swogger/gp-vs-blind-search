#!/bin/sh
# Unattended orchestration of Phase 1 end and the Phase 3 pilot (author authorised overnight launching, 2026-10-19).
#   nohup scripts/s2_overnight.sh >> data/runs/overnight_chain.log 2>&1 &
# Memory chain (never two memory-heavy phases at once): enum_bool -> targets_a -> enum_extra -> precision_extra ->
#   enum_ceiling -> targets_b.
# CPU chain (GP runs, 4 workers, in parallel): after targets_a: reach_a -> mech -> long -> rand_a; after targets_b:
#   reach_b -> rand_b.
# A phase that does not start within 2 minutes is logged as FAILED and skipped. Nothing confirmatory runs here.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
log() { echo "$(date '+%F %T') $*"; }
done_() { $PY -c "import sys; from reach.runner.queue import Phase; p = Phase('$1'); sys.exit(0 if p.manifest_path.exists() and len(p.results()) >= p.manifest()['n_tasks'] else 1)" 2>/dev/null; }
running() { [ -f "data/runs/$1/queue.pid" ] && kill -0 "$(cat "data/runs/$1/queue.pid")" 2>/dev/null; }
wait_for() { until done_ "$1"; do sleep 300; done; }
run() {   # run PHASE WORKERS: start, then wait until finished (or log a failure to start)
    log "start $1 (workers $2)"
    scripts/ensure_running.sh "$1" --workers "$2" > /dev/null 2>&1
    sleep 120
    if ! running "$1" && ! done_ "$1"; then log "FAILED to start $1 (see data/runs/$1/console.log); skipped"; return 1; fi
    until done_ "$1" || ! running "$1"; do sleep 300; done
    if done_ "$1"; then log "finished $1"; else log "stopped early $1 (queue exited)"; fi
}
memory_chain() {
    wait_for s2_enum_bool; log "s2_enum_bool finished"
    run s2_targets_pilot_a 2
    run s2_enum_extra 2
    run s2_precision_extra 2
    run s2_enum_ceiling 1
    run s2_targets_pilot_b 2
    log "memory chain done"
}
cpu_chain() {
    wait_for s2_targets_pilot_a
    run s2_pilot_reach_a 4
    run s2_pilot_mech 4
    run s2_pilot_long 4
    run s2_pilot_rand_a 4
    wait_for s2_targets_pilot_b
    run s2_pilot_reach_b 4
    run s2_pilot_rand_b 4
    log "cpu chain done"
}
log "orchestrator started (pid $$)"
memory_chain &
cpu_chain &
wait
log "all done"

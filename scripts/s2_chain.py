"""Stage 2 phases back to back, in the order the Stage 2 registration states (OSF dg7f3; D096, D097; REQ-RUN-37):

    nohup .venv/bin/python scripts/s2_chain.py >> data/runs/s2_chain.log 2>&1 &

Each phase: wait while its queue runs; skip it when every task has a row; otherwise run it through the queue in the
foreground (caffeinate, 6 workers), as scripts/overnight.sh does. Error rows are rerun once per phase (Stage 1 Missing
data rule; a marker file keeps it to once across restarts); rows still in error are reported and excluded. The chain
stops at a phase left incomplete (drained with STOP, or the queue ended early) or holding a STOP file. s2_ho_cgp is not
in the default order: it runs after the CGP addendum (D083), launched alone with `scripts/s2_chain.py s2_ho_cgp`.
Kept outside src/reach, so the runs' code hash is unchanged.

Every launch is guarded (REQ-RUN-38, D098): the queue snapshots src/reach when a phase starts and resumes a phase on its
own snapshot, so a phase without rows starts only when src/reach hashes to the registered code, and a phase with rows
only when its rows carry that hash.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data/runs"
ORDER = ("s2_ho_a0", "s2_ho_a16", "s2_ho_bench", "s2_ho_operon", "s2_cal_cgp")
LATER = ("s2_ho_cgp",)                    # launched by name after the CGP addendum
WORKERS = 6
RETRIED = ".errors_rerun"                 # marker: this phase's error rows have had their one rerun
REGISTERED_CODE = "8dff35554ec2"          # code hash of src/reach at the Stage 2 registered commit (D096; tested)
ADDENDUM = ROOT / "data/registration/stage2_addendum.txt"   # the CGP addendum's OSF link (s2_ho_cgp waits for it)


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def state(phase, runs=RUNS):
    """('unplanned' | 'incomplete' | 'complete', rows, error rows), counting the last row of each key."""
    d = Path(runs) / phase
    if not (d / "manifest.json").exists():
        return "unplanned", 0, 0
    n = json.loads((d / "manifest.json").read_text())["n_tasks"]
    rows = {}
    if (d / "results.jsonl").exists():
        for line in (d / "results.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["key"]] = r
    errors = sum(r.get("terminal_state") == "error" for r in rows.values())
    return ("complete" if len(rows) >= n else "incomplete"), len(rows), errors


def running(phase, runs=RUNS):
    """True while this phase's queue runs: the pid in its queue.pid is alive and is `reach.runner run PHASE` (a pid file
    left by a finished queue can name a pid the system has reused; guardrail row 144)."""
    pid = Path(runs) / phase / "queue.pid"
    if not pid.exists():
        return False
    ps = subprocess.run(["ps", "-ww", "-o", "command=", "-p", pid.read_text().strip()],     # -ww: whole line
                        capture_output=True, text=True)
    argv = ps.stdout.split()
    return ps.returncode == 0 and "reach.runner" in argv and any(
        a == "run" and b == phase for a, b in zip(argv, argv[1:]))


def current_code():
    """The code hash of src/reach now, as the queue computes it when it snapshots a phase."""
    from reach.runner.queue import code_hash
    return code_hash(ROOT / "src" / "reach")


def code_ok(phase, runs=RUNS):
    """(ok, hashes): whether the queue would run `phase` on the registered code. A phase with rows resumes on its own
    snapshot, so the code hash its rows carry decides; a phase without rows snapshots src/reach at launch."""
    hashes = set()
    res = Path(runs) / phase / "results.jsonl"
    if res.exists():
        hashes = {json.loads(l).get("code_hash") for l in res.read_text().splitlines() if l.strip()} - {None}
    if not hashes:
        hashes = {current_code()}
    return hashes == {REGISTERED_CODE}, sorted(hashes)


def guarded(phase, runs):
    ok, got = code_ok(phase, runs)
    if not ok:
        log(f"{phase}: refused: it would run on code {', '.join(got)}, not the registered {REGISTERED_CODE}; "
            "the chain stops")
    return ok


def run(phase, runs=RUNS, retry=False):
    d = Path(runs) / phase
    d.mkdir(parents=True, exist_ok=True)
    cmd = ["caffeinate", "-is", str(ROOT / ".venv/bin/python"), "-m", "reach.runner", "run", phase,
           "--workers", str(WORKERS)] + (["--retry-errors"] if retry else [])
    with open(d / "console.log", "a") as out:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        (d / "queue.pid").write_text(f"{p.pid}\n")
        return p.wait()


def chain(runs=RUNS, order=None):
    order = ORDER if order is None else tuple(order)
    for phase in order:
        d = Path(runs) / phase
        while running(phase, runs):
            time.sleep(300)
        st, n, err = state(phase, runs)
        if st != "complete":
            if (d / "STOP").exists():
                log(f"{phase}: STOP file present; the chain stops (remove it and relaunch to resume)")
                return 1
            if not guarded(phase, runs):
                return 1
            if phase in LATER and not (ADDENDUM.exists() and ADDENDUM.read_text().strip()):
                log(f"{phase}: refused: the CGP addendum is not recorded in {ADDENDUM}; the chain stops")
                return 1
            log(f"start {phase} ({st}, {n} rows)")
            run(phase, runs)
            st, n, err = state(phase, runs)
            if st != "complete":
                log(f"{phase}: {st} after its run ({n} rows); the chain stops")
                return 1
        if err and not (d / RETRIED).exists():
            if not guarded(phase, runs):
                return 1
            (d / RETRIED).write_text(f"{err} error rows rerun once, {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log(f"{phase}: {err} error rows; rerun once")
            run(phase, runs, retry=True)
            st, n, err = state(phase, runs)
        log(f"{phase} complete: {n} rows" + (f", {err} still in error (reported and excluded)" if err else ""))
    log("chain done" + ("; s2_ho_cgp waits for the CGP addendum" if order == ORDER else ""))
    return 0


def phases_from(argv):
    """The phases named on the command line (default: the registered order); only Stage 2 phases are accepted."""
    bad = [p for p in argv if p not in ORDER + LATER]
    if bad:
        sys.exit(f"not a Stage 2 phase: {', '.join(bad)}")
    return tuple(argv) or ORDER


if __name__ == "__main__":
    phases = phases_from(sys.argv[1:])
    log(f"chain started (pid {__import__('os').getpid()}): {', '.join(phases)}")
    sys.exit(chain(order=phases))

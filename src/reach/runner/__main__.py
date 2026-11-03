"""Command line for the run queue. Runs need no Claude session: launch them from any terminal.

    .venv/bin/python -m reach.runner list
    .venv/bin/python -m reach.runner plan PHASE            # freeze the task list (run does this too)
    .venv/bin/python -m reach.runner run PHASE --workers 6 # foreground, progress + ETA in the console
    scripts/overnight.sh PHASE --workers 6                 # background, survives closing the terminal
    .venv/bin/python -m reach.runner status PHASE          # from any terminal, while running or after
    .venv/bin/python -m reach.runner export PHASE          # results.jsonl -> results.csv
    .venv/bin/python -m reach.runner verify PHASE          # phase-specific check, if defined
"""
import argparse
import csv
import json
import os
import socket
import sys
import time

from reach.runner.phases import PHASES
from reach.runner.queue import RUNS, Phase, Queue, fmt_dur


def cmd_list(a):
    for name, p in PHASES.items():
        ph = Phase(name, a.runs)
        state = ""
        if ph.manifest_path.exists():
            state = f"  [{len(ph.results())}/{ph.manifest()['n_tasks']} done]"
        print(f"{name:16s} {p['description']}{state}")


def cmd_plan(a, quiet=False):
    if a.phase not in PHASES:
        sys.exit(f"unknown phase {a.phase!r}; see `python -m reach.runner list`")
    ph = Phase(a.phase, a.runs)
    tasks = PHASES[a.phase]["build"]()
    from reach import s2_lock              # confirmatory and held-out: exactly the registered task list (REQ-RUN-19);
    reg = s2_lock.registered_tasks_file(a.phase)   # Stage 1 or Stage 2 list by phase, not by prefix (row 89)
    if reg is not None:
        s2_lock.check_registered_tasks(a.phase, tasks, reg)
    for t in tasks:                      # the defaults write_manifest fills in
        t.setdefault("cwd", "."); t.setdefault("weight", 1.0); t.setdefault("meta", {})
    if ph.manifest_path.exists():
        frozen = ph.manifest()["tasks"]
        if frozen == tasks:
            if not quiet:
                print(f"[{a.phase}] manifest up to date ({len(tasks)} tasks)")
            return
        if ph.results() and not a.force:
            sys.exit(f"[{a.phase}] the phase definition changed but the frozen manifest already has "
                     "results. Start a new phase name, or pass --force to replace the manifest.")
        if not a.force:
            sys.exit(f"[{a.phase}] the phase definition changed; pass --force to replace the manifest.")
    ph.write_manifest(tasks, PHASES[a.phase]["description"])
    print(f"[{a.phase}] manifest frozen: {len(tasks)} tasks -> {ph.manifest_path}")


def cmd_run(a):
    ph = Phase(a.phase, a.runs)
    if not ph.manifest_path.exists():
        a.force = False
        cmd_plan(a)
    q = Queue(ph, workers=a.workers, retry_errors=a.retry_errors, order=a.order, limit=a.limit,
              allow_code_change=a.allow_code_change, heartbeat=a.heartbeat)
    sys.exit(q.run())


def _alive(pid, host):
    if host != socket.gethostname():
        return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def cmd_status(a):
    ph = Phase(a.phase, a.runs)
    if not ph.manifest_path.exists():
        sys.exit(f"[{a.phase}] not planned yet")
    man, res = ph.manifest(), ph.results()
    counts = {}
    for r in res.values():
        counts[r["terminal_state"]] = counts.get(r["terminal_state"], 0) + 1
    cpu = sum(r.get("cpu_s", 0) for r in res.values())
    print(f"[{a.phase}] {len(res)}/{man['n_tasks']} done  "
          f"{', '.join(f'{k} {v}' for k, v in sorted(counts.items())) or ''}  total CPU {fmt_dur(cpu)}")
    if ph.status_path.exists():
        st = json.loads(ph.status_path.read_text())
        alive = _alive(st["pid"], st["host"])
        label = {True: "RUNNING", False: "not running", None: f"on {st['host']}"}[alive]
        print(f"queue: {label} (pid {st['pid']}), last update {st['updated']}, "
              f"elapsed {fmt_dur(st['elapsed_s'])}, ETA {fmt_dur(st['eta_s'])}"
              + (f" (~{st['eta_clock']})" if st.get("eta_clock") and alive else ""))
        if alive:
            for r in st["running"]:
                print(f"  running {r['key']}  {fmt_dur(r['elapsed_s'])}")
            print(f"drain: touch {ph.stop_path}    hard stop: kill {st['pid']}")
    if ph.stop_path.exists():
        print(f"STOP file present: {ph.stop_path}")
    errs = [k for k, r in res.items() if r["terminal_state"] == "error"]
    if errs:
        print(f"{len(errs)} error rows (rerun with --retry-errors); first: {errs[0]}; logs in {ph.logs}")


def cmd_export(a):
    ph = Phase(a.phase, a.runs)
    rows = list(ph.results().values())
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    out = ph.dir / "results.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"[{a.phase}] {len(rows)} rows -> {out}")


def cmd_verify(a):
    v = PHASES.get(a.phase, {}).get("verify")
    if v is None:
        sys.exit(f"[{a.phase}] has no verify step")
    sys.exit(v(Phase(a.phase, a.runs)))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m reach.runner", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=lambda s: __import__("pathlib").Path(s), default=RUNS,
                    help="runs directory (default data/runs)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    p = sub.add_parser("plan"); p.add_argument("phase"); p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("run"); p.add_argument("phase")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--retry-errors", action="store_true", help="rerun tasks that ended in the error state")
    p.add_argument("--order", choices=["shuffle", "manifest"], default="shuffle",
                   help="shuffle (default, fixed seed) mixes cheap and costly tasks so the ETA is realistic")
    p.add_argument("--limit", type=int, default=None, help="run at most N tasks this session")
    p.add_argument("--allow-code-change", action="store_true")
    p.add_argument("--heartbeat", type=float, default=300.0, help="seconds between 'still working' lines")
    p.set_defaults(fn=cmd_run)
    for name, fn in (("status", cmd_status), ("export", cmd_export), ("verify", cmd_verify)):
        p = sub.add_parser(name); p.add_argument("phase"); p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

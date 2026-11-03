"""Resumable, CPU-time-limited run queue (a port of legacy/research/rerun/run_queue.py).

Every task is one subprocess under a hard CPU-time limit (`ulimit -t`). Its exit code,
signal, CPU seconds (from wait4 rusage) and wall seconds are recorded with every result,
so censoring is never silent. A phase lives in data/runs/<phase>/:

    manifest.json   frozen task list, written once by `plan`; `run` executes it
    results.jsonl   one JSON object per finished task, appended and fsynced
    status.json     progress snapshot, rewritten after every finished task
    logs/           output of tasks that ended in the error state (kept for debugging)
    code/<hash>/    snapshot of src/reach taken at launch; tasks import this copy, so
                    editing src/ during an overnight run cannot change later runs
    STOP            create this file to drain: running tasks finish, no new ones start

Running `run` again skips every key already in results.jsonl, so an interrupted phase
resumes where it stopped. A task that was interrupted leaves no row and simply reruns.

Task protocol: the task prints one line `RESULT {json}` on stdout. If the JSON has a
`terminal_state` (converged, budget_exhausted, early_stopped, timeout), it is kept; the
queue overrides it with `timeout` when the CPU limit killed the process and with `error`
when the task crashed or printed no RESULT line.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from reach.paths import ROOT  # noqa: E402

SRC = ROOT / "src"
RUNS = ROOT / "data" / "runs"
SHUFFLE_SEED = 20261012
TERMINAL_STATES = ("converged", "budget_exhausted", "early_stopped", "timeout", "error")


# ------------------------------------------------------------------ small helpers
def fmt_dur(s: float | None) -> str:
    if s is None:
        return "--"
    s = int(round(s))
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def git_version() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return commit + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def code_hash(src: Path = SRC / "reach") -> str:
    h = hashlib.sha256()
    for p in sorted(src.rglob("*.py")):
        h.update(str(p.relative_to(src)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def snapshot_code(phase_dir: Path) -> tuple[str, Path]:
    """Copy src/reach into phase_dir/code/<hash>/reach; return (hash, dir for PYTHONPATH)."""
    ch = code_hash()
    dest = phase_dir / "code" / ch
    if not (dest / "reach").exists():
        tmp = phase_dir / "code" / f".{ch}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(SRC / "reach", tmp / "reach",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        tmp.rename(dest)
    return ch, dest


def read_results(path: Path) -> dict[str, dict]:
    """Last row per key. A truncated final line (crash mid-write) is ignored."""
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[row["key"]] = row
    return rows


# ------------------------------------------------------------------ phase files
class Phase:
    def __init__(self, name: str, runs_dir: Path = RUNS):
        self.name = name
        self.dir = runs_dir / name
        self.manifest_path = self.dir / "manifest.json"
        self.results_path = self.dir / "results.jsonl"
        self.status_path = self.dir / "status.json"
        self.stop_path = self.dir / "STOP"
        self.logs = self.dir / "logs"

    def write_manifest(self, tasks: list[dict], description: str = "") -> None:
        keys = [t["key"] for t in tasks]
        if len(set(keys)) != len(keys):
            raise ValueError(f"phase {self.name}: duplicate task keys")
        for t in tasks:
            t.setdefault("cwd", ".")
            t.setdefault("weight", 1.0)
            t.setdefault("meta", {})
            if not isinstance(t.get("cpu_limit"), (int, float)) or t["cpu_limit"] <= 0:
                raise ValueError(f"task {t['key']}: cpu_limit (seconds) is required")
            if set(t.get("env") or {}) & {"PYTHONPATH", "REACH_ROOT"}:
                raise ValueError(f"task {t['key']}: env may not override PYTHONPATH or REACH_ROOT (code pinning)")
        self.dir.mkdir(parents=True, exist_ok=True)
        manifest = dict(phase=self.name, description=description,
                        created=time.strftime("%Y-%m-%d %H:%M:%S"), git=git_version(),
                        n_tasks=len(tasks), tasks=tasks)
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=1))
        tmp.rename(self.manifest_path)

    def manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text())

    def results(self) -> dict[str, dict]:
        return read_results(self.results_path)


# ------------------------------------------------------------------ orphans (guardrail row 138)
def _processes() -> list[tuple[int, str]]:
    """(pid, full command line) of every process, from ps (macOS and Linux)."""
    out = subprocess.run(["ps", "-A", "-ww", "-o", "pid=,command="], capture_output=True, text=True).stdout
    procs = []
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if pid.isdigit():
            procs.append((int(pid), cmd.strip()))
    return procs


def live_task_processes(tasks: list[dict]) -> list[tuple[int, str]]:
    """(pid, key) of live processes running one of `tasks`: the command line ends with the task's argv after the
    interpreter (so the shell wrapper and the interpreter's path do not matter). Tasks start in their own session and
    outlive a hard-killed queue; a relaunch must not run the same key again beside them."""
    tails = {" ".join(str(a) for a in t["argv"][1:]): t["key"] for t in tasks if len(t["argv"]) > 2}
    me, found = os.getpid(), []
    for pid, cmd in _processes():
        if pid == me:
            continue
        words = cmd.split(" ")
        for i in range(1, len(words)):
            key = tails.get(" ".join(words[i:]))
            if key is not None:
                found.append((pid, key))
                break
    return sorted(found)


# ------------------------------------------------------------------ the queue
class Queue:
    def __init__(self, phase: Phase, workers: int = 6, retry_errors: bool = False,
                 order: str = "shuffle", limit: int | None = None,
                 allow_code_change: bool = False, heartbeat: float = 300.0,
                 install_signals: bool = True, out=sys.stdout):
        self.phase, self.workers, self.out = phase, workers, out
        self.install_signals = install_signals
        self.heartbeat = heartbeat
        man = phase.manifest()
        self.tasks = man["tasks"]
        done = phase.results()
        if retry_errors:
            done = {k: r for k, r in done.items() if r.get("terminal_state") != "error"}
        hashes = {r.get("code_hash") for r in done.values()} - {None}
        pinned = phase.dir / "code" / next(iter(hashes)) if len(hashes) == 1 else None
        if pinned is not None and (pinned / "reach").exists():
            # resume on the phase's own snapshot: same code as its earlier results, whatever src/reach is now (row 54)
            self.code_hash, self.code_dir = next(iter(hashes)), pinned
            self.pinned_git = next((r.get("git") for r in done.values() if r.get("git")), None)   # provenance (row 69)
        else:
            self.pinned_git = None
            self.code_hash, self.code_dir = snapshot_code(phase.dir)
        if hashes and hashes != {self.code_hash} and not allow_code_change:
            raise SystemExit(
                f"[{phase.name}] src/reach changed since earlier results in this phase "
                f"(earlier code {sorted(hashes)}, now {self.code_hash}). Results from mixed code "
                "versions are not comparable. Revert the change, start a new phase, or pass "
                "--allow-code-change if the change cannot affect results.")
        self.git = self.pinned_git or git_version()
        self.todo = [t for t in self.tasks if t["key"] not in done]
        if order == "shuffle":
            random.Random(SHUFFLE_SEED).shuffle(self.todo)
        if limit is not None:
            self.todo = self.todo[:limit]
        self.n_total = len(self.tasks)
        self.n_done_before = len(done)
        self.counts = {s: 0 for s in TERMINAL_STATES}
        for r in done.values():
            s = r.get("terminal_state", "error")
            self.counts[s] = self.counts.get(s, 0) + 1
        self.remaining_weight = sum(t["weight"] for t in self.todo)
        self.session_done = 0
        self.session_weight = 0.0
        self.lock = threading.RLock()     # re-entrant: the signal handler may run while the main thread holds it
        self.next_i = 0
        self.running: dict[int, tuple[dict, float]] = {}   # pid -> (task, start time)
        self.user_killed: set[int] = set()
        self.draining = False
        self.aborting = False
        self.t0 = time.time()
        self.last_print = time.time()

    # ---------------------------------------------------------- progress
    def eta(self) -> float | None:
        if self.session_done < min(3, len(self.todo)) or self.session_weight <= 0:
            return None
        rate = self.session_weight / (time.time() - self.t0)
        return self.remaining_weight / rate if rate > 0 else None

    def say(self, msg: str) -> None:
        print(f"{time.strftime('%H:%M:%S')}  {msg}", file=self.out, flush=True)
        self.last_print = time.time()

    def progress_prefix(self) -> str:
        done = self.n_done_before + self.session_done
        eta = self.eta()
        clock = "--" if eta is None else time.strftime("%a %H:%M", time.localtime(time.time() + eta))
        return (f"{done:>{len(str(self.n_total))}}/{self.n_total} "
                f"({100 * done / max(self.n_total, 1):5.1f}%)  ETA {fmt_dur(eta)} (~{clock})")

    def write_status(self) -> None:
        eta = self.eta()
        st = dict(phase=self.phase.name, pid=os.getpid(), host=socket.gethostname(),
                  updated=time.strftime("%Y-%m-%d %H:%M:%S"), workers=self.workers,
                  total=self.n_total, done=self.n_done_before + self.session_done,
                  this_session=self.session_done, elapsed_s=round(time.time() - self.t0),
                  eta_s=None if eta is None else round(eta),
                  eta_clock=None if eta is None else time.strftime(
                      "%Y-%m-%d %H:%M", time.localtime(time.time() + eta)),
                  counts=self.counts, draining=self.draining, code_hash=self.code_hash,
                  running=[dict(key=t["key"], elapsed_s=round(time.time() - t0))
                           for t, t0 in self.running.values()])
        tmp = self.phase.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1))
        tmp.rename(self.phase.status_path)

    # ---------------------------------------------------------- one task
    def _next_task(self) -> dict | None:
        with self.lock:
            if self.draining or self.aborting or self.next_i >= len(self.todo):
                return None
            if self.phase.stop_path.exists():
                self.draining = True
                self.say(f"STOP file found: draining ({len(self.running)} running tasks will finish)")
                return None
            t = self.todo[self.next_i]
            self.next_i += 1
            return t

    def run_task(self, task: dict) -> None:
        self.phase.logs.mkdir(parents=True, exist_ok=True)
        log_path = self.phase.logs / (task["key"].replace("|", "_").replace("/", "_") + ".log")
        argv = [sys.executable if a == "{python}" else a for a in task["argv"]]
        # macOS strips DYLD_* from a protected binary's environment (/bin/sh), so task variables are exported inside
        # the shell, after it starts (guardrail row 52)
        exports = "".join(f"export {k}={shlex.quote(str(v))}; " for k, v in (task.get("env") or {}).items())
        sh = f"ulimit -t {int(task['cpu_limit'])}; {exports}exec {shlex.join(argv)}"
        env = dict(os.environ, **(task.get("env") or {}))     # extra variables (e.g. Operon's library path)
        env.update(PYTHONPATH=str(self.code_dir), PYTHONUNBUFFERED="1", REACH_ROOT=str(ROOT))
        w0 = time.time()
        with open(log_path, "w") as lf:
            proc = subprocess.Popen(["/bin/sh", "-c", sh], stdout=lf, stderr=subprocess.STDOUT,
                                    cwd=ROOT / task["cwd"], env=env, start_new_session=True)
            with self.lock:
                self.running[proc.pid] = (task, w0)
            _, status, ru = os.wait4(proc.pid, 0)
            proc.returncode = status          # reaped here; stop Popen from waiting again
        wall = time.time() - w0
        with self.lock:
            self.running.pop(proc.pid, None)
            if proc.pid in self.user_killed:  # stopped by the user: no row, reruns next launch
                log_path.unlink(missing_ok=True)
                return
        sig = os.WTERMSIG(status) if os.WIFSIGNALED(status) else 0
        code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
        text = log_path.read_text(errors="replace")
        res = [l for l in text.splitlines() if l.startswith("RESULT ")]
        data = {}
        if res:
            try:
                data = json.loads(res[-1][7:])
            except json.JSONDecodeError:
                data = {}
        cpu_used = ru.ru_utime + ru.ru_stime
        if sig == signal.SIGXCPU or (sig == signal.SIGKILL and cpu_used >= 0.95 * task["cpu_limit"]):
            state, qstatus = "timeout", "cpu_limit"
        elif sig:                     # killed from outside before its CPU limit: an error, never silent censoring
            state, qstatus = "error", f"killed_signal_{sig}"
        elif code != 0 or not data:
            state, qstatus = "error", "error"
        else:
            state, qstatus = data.get("terminal_state", "completed"), "ok"
        row = dict(task.get("meta", {}))
        row.update(data)
        row.update(key=task["key"], terminal_state=state, queue_status=qstatus, exit=code,
                   signal=sig, cpu_s=round(ru.ru_utime + ru.ru_stime, 3), wall_s=round(wall, 3),
                   cpu_limit=task["cpu_limit"], finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                   code_hash=self.code_hash, git=self.git)
        if state == "error":
            row["error_tail"] = "\n".join(text.splitlines()[-15:])
        else:
            log_path.unlink(missing_ok=True)
        line = json.dumps(row) + "\n"
        with self.lock:
            with open(self.phase.results_path, "a") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            self.counts[state] = self.counts.get(state, 0) + 1
            self.session_done += 1
            self.session_weight += task["weight"]
            self.remaining_weight -= task["weight"]
            self.say(f"{self.progress_prefix()}  {task['key']}  {state}  cpu {row['cpu_s']:.1f}s")
            self.write_status()

    def worker(self) -> None:
        while True:
            t = self._next_task()
            if t is None:
                return
            try:
                self.run_task(t)
            except Exception as e:  # keep the queue alive; the task has no row and reruns
                self.say(f"queue error on {t['key']}: {e!r}")

    # ---------------------------------------------------------- control
    def kill_running(self) -> None:
        with self.lock:
            for pid in list(self.running):
                self.user_killed.add(pid)
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def on_signal(self, signum, frame) -> None:
        if signum == signal.SIGINT and not self.draining:
            self.draining = True
            self.say("Ctrl-C: draining. Running tasks will finish; press Ctrl-C again to stop "
                     "them now (they rerun on the next launch).")
        else:
            self.aborting = True
            self.say("stopping now: killing running tasks (they rerun on the next launch)")
            self.kill_running()

    def summary(self) -> str:
        c = ", ".join(f"{k} {v}" for k, v in self.counts.items() if v)
        return (f"[{self.phase.name}] {self.n_done_before + self.session_done}/{self.n_total} done "
                f"({c or 'none'}); this session {self.session_done} in {fmt_dur(time.time() - self.t0)}")

    def run(self) -> int:
        self.say(f"[{self.phase.name}] {self.n_total} tasks, {self.n_done_before} already done, "
                 f"{len(self.todo)} to run on {self.workers} workers. Code {self.code_hash} "
                 f"(git {self.git}). Drain with Ctrl-C or: touch {self.phase.stop_path}")
        if not self.todo:
            self.say(self.summary())
            return 0
        orphans = live_task_processes(self.todo)      # a hard-killed queue leaves its tasks running (row 138)
        if orphans:
            self.say(f"[{self.phase.name}] {len(orphans)} task(s) of this phase are still running outside a queue "
                     f"(pid, key): {orphans[:5]}{' ...' if len(orphans) > 5 else ''}. Relaunching would run them again "
                     "beside them. Wait until they end (each has its CPU limit), or kill their process groups "
                     "(kill -9 -PID), then relaunch. Stop a queue by draining (touch STOP), not with kill -9.")
            return 3
        if self.install_signals and threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self.on_signal)
            signal.signal(signal.SIGTERM, self.on_signal)
        threads = [threading.Thread(target=self.worker, daemon=True)
                   for _ in range(min(self.workers, len(self.todo)))]
        for th in threads:
            th.start()
        while any(th.is_alive() for th in threads):
            for th in threads:
                th.join(timeout=1.0)
            if time.time() - self.last_print > self.heartbeat:
                with self.lock:
                    run = sorted(self.running.values(), key=lambda x: x[1])
                    longest = f"; longest {run[0][0]['key']} {fmt_dur(time.time() - run[0][1])}" if run else ""
                    self.say(f"{self.progress_prefix()}  (still working: {len(run)} running{longest})")
                    self.write_status()
        self.write_status()
        self.say(self.summary())
        if self.draining and self.phase.stop_path.exists():
            self.say(f"remove {self.phase.stop_path} before the next launch")
        return 130 if self.aborting else 0

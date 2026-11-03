"""Resumable, priority-ordered rerun queue for the 2026-09 revision.

Every run is a subprocess under a CPU-time limit (`ulimit -t`), unbuffered, with
its exit status, signal, CPU and wall seconds recorded explicitly, so censoring
is never silent. Results append to results_rerun/<phase>.csv; completed keys are
skipped on restart. Create results_rerun/STOP to drain gracefully.

    caffeinate -i python3 research/rerun/run_queue.py --workers 6
    python3 research/rerun/run_queue.py --dry_run
    python3 research/rerun/run_queue.py --smoke          # 1 short task per phase
"""
from __future__ import annotations

import argparse, csv, json, os, shlex, signal, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(RESEARCH, ".."))
SYN = os.path.join(ROOT, "results", "synthetic_dense_k")
sys.path.insert(0, RESEARCH)
from seeds import SEEDS                                  # noqa: E402
from test_suite import TEST_MATRIX, eml_plateau_mult     # noqa: E402

PY = sys.executable
TESTS = ["1.1", "1.2", "1.3", "1.4", "2.1", "3.1", "3.2", "3.3", "3.4", "3.5", "4.1", "4.2"]
SSC_TESTS = ["1.1", "1.2", "1.3", "2.1", "3.2", "3.3"]
ABL6 = "+,-,*,/,exp,log"
FIX8 = "+,-,*,/,exp,log,sin,cos"


def _targets(fname, kset=None):
    ts = json.load(open(os.path.join(SYN, fname)))
    return [t for t in ts if kset is None or t["K_construction"] in kset]


def synth_tasks(phase, fname, seeds, kset=None, gens=200, extra=(), plateau=None):
    path = os.path.join(SYN, fname)
    for t in _targets(fname, kset):
        for reg in ("A", "B"):
            for s in seeds:
                cmd = [PY, "-u", os.path.join(HERE, "synth_task.py"), "--targets", path,
                       "--target_id", t["target_id"], "--regime", reg, "--seed", str(s),
                       "--generations", str(gens), "--timeout", "600", "--no_size_penalty",
                       "--cpu_clock", *extra]
                if plateau:
                    cmd += ["--plateau", str(plateau)]
                yield dict(phase=phase, kind="synth", key=f"{t['target_id']}|{reg}|{s}",
                           cmd=cmd, cpu_limit=700)


def bench_tasks(phase, tests, algos, regimes, cells, seeds, ops=None):
    for G, P in cells:
        for tid in tests:
            for algo in algos:
                for reg in regimes:
                    for i, s in seeds:
                        eml = algo == "eml"
                        mult = eml_plateau_mult(TEST_MATRIX[tid]["std_ops"]) if eml else 1
                        cmd = [PY, "-u", os.path.join(RESEARCH, "deduce_formula.py"),
                               "--test_id", tid, "--generations", str(G), "--pop_size", str(P),
                               "--seed", str(s), "--plateau_mult", str(mult),
                               "--no_size_penalty", "--trace", "--no_simplify"]
                        if eml:
                            cmd.append("--use_eml")
                        if reg == "B":
                            cmd.append("--mutation_only")
                        if ops:
                            cmd += ["--std_ops", ops]
                        yield dict(phase=phase, kind="bench",
                                   key=f"{tid}|{algo}|{reg}|{G}|{P}|{i:02d}", cmd=cmd,
                                   cpu_limit=600, meta=dict(test_id=tid, algo=algo, regime=reg,
                                                            G=G, P=P, run=i, seed=s, ops=ops or "table2"))


def ssc_tasks(phase, seeds):
    for tid in SSC_TESTS:
        for algo in ("std", "eml"):
            for i, s in seeds:
                cmd = [PY, "-u", os.path.join(HERE, "ssc_task.py"), "--test_id", tid,
                       "--algo", algo, "--seed", str(s), "--mode", "uy", "--no_size_penalty"]
                yield dict(phase=phase, kind="ssc", key=f"{tid}|{algo}|{i:02d}", cmd=cmd,
                           cpu_limit=600, meta=dict(test_id=tid, algo=algo, run=i, seed=s))


def sr_tasks(phase, seeds):
    for bench in ("nguyen1", "nguyen7", "keijzer6", "vladislavleva4"):
        for algo in ("std", "eml"):
            for reg in ("A", "B"):
                for i, sd in seeds:
                    cmd = [PY, "-u", os.path.join(HERE, "sr_task.py"), "--bench", bench, "--algo", algo,
                           "--regime", reg, "--seed", str(sd), "--no_size_penalty"]
                    yield dict(phase=phase, kind="sr", key=f"{bench}|{algo}|{reg}|{i:02d}", cmd=cmd,
                               cpu_limit=600, meta=dict(bench=bench, algo=algo, regime=reg, run=i, seed=sd))


def build_phases():
    s10 = SEEDS[1:11]
    s3 = SEEDS[1:4]
    runs30 = [(i, SEEDS[i]) for i in range(1, 31)]
    ref = [(200, 100)]
    grid = [(g, p) for g in (100, 200, 500) for p in (100, 200, 500) if (g, p) != (200, 100)]
    k7_13 = {7, 9, 11, 13}
    # Full plan (deadline relaxed): essentials first, then every remaining phase.
    return [
        ("P2_bench_ref", list(bench_tasks("P2_bench_ref", ["1.1", "1.2", "1.3", "3.1"], ("std", "eml"),
                                          ("A", "B"), ref, runs30))),
        ("P1_synth_main", list(synth_tasks("P1_synth_main", "targets.json", s10))),
        ("P1b_synth_lowk", list(synth_tasks("P1b_synth_lowk", "targets_lowk.json", s10))),
        ("P3_control_abl6", list(bench_tasks("P3_control_abl6", TESTS, ("std",), ("B",), ref, runs30, ABL6))),
        ("P4_ssc_uy", list(ssc_tasks("P4_ssc_uy", runs30))),
        ("P6_native_grid", list(bench_tasks("P6_native_grid", ["1.2"], ("std", "eml"), ("A", "B"), grid, runs30))),
        ("P5_synth_pointmut", list(synth_tasks("P5_synth_pointmut", "targets.json", s3, k7_13,
                                               extra=("--point_mutation",)))),
        # --- Sunday-afternoon target: essentials first, optional tail after ---
        ("P2_bench_ref+", list(bench_tasks("P2_bench_ref", [t for t in TESTS if t not in ("1.1", "1.2", "1.3", "3.1")],
                                           ("std", "eml"), ("A", "B"), ref, runs30))),
        ("P3_control_abl6+A", list(bench_tasks("P3_control_abl6", TESTS, ("std",), ("A",), ref, runs30, ABL6))),
        ("P8_control_fix8", list(bench_tasks("P8_control_fix8", TESTS, ("std",), ("B",), ref, runs30, FIX8))),
        ("P9_synth_rigid", list(synth_tasks("P9_synth_rigid", "targets_rigid_oversample.json", s10))),
        ("P7_synth_budget800", list(synth_tasks("P7_synth_budget800", "targets.json", s3, k7_13, gens=800, plateau=600))
                                + list(synth_tasks("P7_synth_budget800", "targets_lowk.json", s3, gens=800, plateau=600))),
        # added 2026-09-27: Regime A of the 8-primitive control, so both regimes use the corrected fitness
        ("P8_control_fix8+A", list(bench_tasks("P8_control_fix8", TESTS, ("std",), ("A",), ref, runs30, FIX8))),
        # optional tail: runs only if time allows; nothing in the paper waits on it
        ("P10_sr_bench", list(sr_tasks("P10_sr_bench", runs30))),
        ("P7b_synth_budget3200", list(synth_tasks("P7b_synth_budget3200", "targets.json", s3, k7_13, gens=3200, plateau=600))
                                  + list(synth_tasks("P7b_synth_budget3200", "targets_lowk.json", s3, gens=3200, plateau=600))),
    ]


# ------------------------------------------------------------------ parsing
def parse(task, log_path, code, sig, cpu, wall):
    text = open(log_path, errors="replace").read() if os.path.exists(log_path) else ""
    killed = sig in (signal.SIGXCPU, signal.SIGKILL)
    row = dict(key=task["key"], exit=code, signal=sig, cpu_s=f"{cpu:.2f}", wall_s=f"{wall:.2f}")
    if task["kind"] in ("synth", "ssc", "sr"):
        res = [l for l in text.splitlines() if l.startswith("RESULT ")]
        data = json.loads(res[-1][7:]) if res else {}
        row.update(data)
        if task["kind"] == "synth":
            row["status"] = ("cpu_limit" if killed else data.get("terminal_state", "crash")) if res or killed else "crash"
            row.pop("best_expr", None)
        else:
            row.update(task["meta"])
            gens = [l.split() for l in text.splitlines() if l.startswith("GEN ")]
            row["status"] = "cpu_limit" if killed else (data.get("stop_reason") or "crash")
            if killed and gens:
                row["final_err"], row["last_gen"] = gens[-1][3], gens[-1][1]
    else:  # bench
        row.update(task["meta"])
        kv = {}
        for l in text.splitlines():
            if l[:6] in ("STOP_R", "LAST_G", "FINAL_") and " " in l:
                k, v = l.split(" ", 1)
                kv[k] = v
        gens = [l.split() for l in text.splitlines() if l.startswith("GEN ")]
        if killed:
            row["status"] = "cpu_limit"
        elif code == 0 and "STOP_REASON" in kv:
            row["status"] = kv["STOP_REASON"]
        else:
            row["status"] = "crash"
        row["final_err"] = kv.get("FINAL_ERR", gens[-1][3] if gens else "")
        row["final_nodes"] = kv.get("FINAL_NODES", gens[-1][7] if gens else "")
        row["last_gen"] = kv.get("LAST_GEN", gens[-1][1] if gens else "")
        row["log"] = os.path.relpath(log_path, ROOT)
    return row


# ------------------------------------------------------------------ runner
class Queue:
    def __init__(self, out, workers):
        self.out, self.workers = out, workers
        self.lock = threading.Lock()
        self.stats = {}
        self.t0 = time.time()
        self.n_done = 0

    def done_keys(self, phase):
        p = os.path.join(self.out, f"{phase}.csv")
        if not os.path.exists(p):
            return set()
        with open(p, newline="") as fh:
            return {r["key"] for r in csv.DictReader(fh)}

    FIELDS = {
        "synth": ["key", "target_id", "K_construction", "K_proven", "m_operators", "regime", "seed",
                  "converged", "exact_hash_match", "gens_to_converge", "final_nmse", "final_tree_size",
                  "terminal_state", "wall_seconds", "cpu_seconds", "gens_run", "status", "exit",
                  "signal", "cpu_s", "wall_s"],
        "bench": ["key", "test_id", "algo", "regime", "G", "P", "run", "seed", "ops", "status",
                  "final_err", "final_nodes", "last_gen", "exit", "signal", "cpu_s", "wall_s", "log"],
        "ssc": ["key", "test_id", "algo", "run", "seed", "status", "final_err", "last_gen",
                "converged", "stop_reason", "exit", "signal", "cpu_s", "wall_s"],
        "sr": ["key", "bench", "algo", "regime", "run", "seed", "status", "final_err", "last_gen",
               "converged", "stop_reason", "exit", "signal", "cpu_s", "wall_s"],
    }

    def write(self, phase, kind, row):
        p = os.path.join(self.out, f"{phase}.csv")
        with self.lock:
            new = not os.path.exists(p)
            with open(p, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=self.FIELDS[kind], extrasaction="ignore", restval="")
                if new:
                    w.writeheader()
                w.writerow(row)
            st = self.stats.setdefault(phase, {})
            st[row["status"]] = st.get(row["status"], 0) + 1
            self.n_done += 1
            if self.n_done % 20 == 0:
                self.progress()

    def progress(self):
        el = (time.time() - self.t0) / 3600
        lines = [f"updated {time.strftime('%Y-%m-%d %H:%M')}  elapsed {el:.1f} h  "
                 f"completed this session {self.n_done}", ""]
        for name, keys in self.totals:
            csvname = name.split("+")[0]
            d = len(self.done_keys(csvname) & keys)
            lines.append(f"{name:22s} {d:5d}/{len(keys):<5d}  {self.stats.get(csvname, {})}")
        with open(os.path.join(self.out, "PROGRESS.txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")

    def run(self, task):
        if os.path.exists(os.path.join(self.out, "STOP")):
            return
        logdir = os.path.join(self.out, "logs", task["phase"])
        os.makedirs(logdir, exist_ok=True)
        log_path = os.path.join(logdir, task["key"].replace("|", "_") + ".log")
        sh = f"ulimit -t {task['cpu_limit']}; exec {shlex.join(task['cmd'])}"
        w0 = time.time()
        with open(log_path, "w") as lf:
            proc = subprocess.Popen(["/bin/sh", "-c", sh], stdout=lf, stderr=subprocess.STDOUT,
                                    cwd=RESEARCH)
            _, status, ru = os.wait4(proc.pid, 0)
            proc.returncode = status
        sig = os.WTERMSIG(status) if os.WIFSIGNALED(status) else 0
        code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
        row = parse(task, log_path, code, sig, ru.ru_utime + ru.ru_stime, time.time() - w0)
        self.write(task["phase"], task["kind"], row)
        if task["kind"] == "synth" and row["status"] not in ("crash",):
            os.remove(log_path)          # synthetic logs carry nothing beyond the CSV row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--phases", default="all")
    ap.add_argument("--out", default=os.path.join(ROOT, "results_rerun"))
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    phases = build_phases()
    if a.phases != "all":
        want = set(a.phases.split(","))
        phases = [(n, t) for n, t in phases if n in want or n.split("_")[0] in want]
    if a.smoke:
        a.out = os.path.join(a.out, "_smoke")
        for _, ts in phases:
            del ts[1:]
            c = ts[0]["cmd"]
            if "--generations" in c:
                c[c.index("--generations") + 1] = "5"
            else:
                c += ["--generations", "5"]
    os.makedirs(a.out, exist_ok=True)
    q = Queue(a.out, a.workers)
    q.totals = [(n, {t['key'] for t in ts}) for n, ts in phases]
    todo = []
    for n, ts in phases:
        done = q.done_keys(n.split("+")[0])
        todo += [t for t in ts if t["key"] not in done]
        print(f"{n:22s} total {len(ts):5d}  remaining {sum(t['key'] not in done for t in ts):5d}")
    if a.dry_run:
        return
    print(f"running {len(todo)} tasks on {a.workers} workers -> {a.out}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for f in [ex.submit(q.run, t) for t in todo]:
            try:
                f.result()
            except Exception as e:  # keep the queue alive; the task stays un-done and reruns next launch
                print("worker error:", repr(e), flush=True)
    q.progress()
    print("queue finished", flush=True)


if __name__ == "__main__":
    main()

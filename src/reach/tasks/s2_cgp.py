"""One Cartesian GP run (D072; docs/design/cgp.md) -> one `RESULT {json}` line, like reach.tasks.s2_run.

Every scientific setting is required (C6): genome length, levels-back, mutation, strategy, penalty, stop rule, budget,
CPU limit. The task refuses held-out sets before the Stage 2 registration (s2_lock), a changed target file (REQ-MEA-10)
and a binary not built from the Rust source its pinned code expects (REQ-RUN-23). It runs the binary once, as a child
process, and re-scores in Python every exact claim and the final parent with the registered exact rule (REQ-ENG-19).
CPU time includes the child's (REQ-RUN-24). If the child is killed by a signal (its hard CPU limit or from outside),
the task prints what it knows (with the tail of the engine's stderr: a Rust panic aborts with SIGABRT, guardrail row 85)
and dies by the same signal, so the queue records the signal and the state; the row then has no evals_reached and is
censored at every budget (load.tidy).
"""
import argparse
import hashlib
import json
import os
import signal
import sys
import time

from reach import cgp
from reach import grammar as G
from reach.s2_lock import check_search_allowed

ap = argparse.ArgumentParser()
req = dict(required=True)
ap.add_argument("--set", **req)
ap.add_argument("--targets", **req)
ap.add_argument("--target_id", **req)
ap.add_argument("--targets_sha256", **req)
ap.add_argument("--seed", type=int, **req)
ap.add_argument("--n_nodes", type=int, help="genome length (function nodes in the row)", **req)
ap.add_argument("--levels_back", choices=["unrestricted"], **req)
ap.add_argument("--mutation", choices=["single"], **req)
ap.add_argument("--strategy", choices=["1+4"], **req)
ap.add_argument("--budget", type=float, help="multiples of E0 = 2x10^4 evaluations (one run gives every prefix)", **req)
ap.add_argument("--penalty", choices=["scaled", "fixed"], **req)
ap.add_argument("--stop_on", choices=["exact"], **req)
ap.add_argument("--timeout", type=float, help="CPU seconds (soft limit in the engine; the queue adds a hard one)", **req)
a = ap.parse_args()

try:
    check_search_allowed(a.set)
except PermissionError as err:
    sys.exit(f"refused: {err}")
raw = open(a.targets, "rb").read()
sha = hashlib.sha256(raw).hexdigest()
if sha != a.targets_sha256:
    sys.exit(f"refused: target file checksum {sha} differs from the registered {a.targets_sha256}")
try:
    exe = cgp.binary()
except cgp.BinaryError as err:
    sys.exit(f"refused: {err}")

t0, c0 = time.time(), time.process_time()
g = G.get(a.set)
tgt = next(t for t in json.loads(raw) if t["target_id"] == a.target_id)
tree = G.parse_expr(tgt["expression"], g)
ctx = cgp.context(g, tree, tgt.get("var_target"), tgt.get("target_scale"), a.penalty)
max_evals = int(round(a.budget * cgp.E0))
row = dict(set=g.name, domain=g.kind, algorithm="CGP", config=f"n{a.n_nodes}", seed=a.seed, target_id=tgt["target_id"],
           K_proven=tgt.get("K_proven"), K_construction=tgt.get("K_construction"), units=tgt.get("units"),
           target_dag_nodes=cgp.sizes(ctx["program"])["dag_size"],      # distinct subtrees: an upper bound
           budget=a.budget, budget_E=max_evals, n_nodes=a.n_nodes, levels_back=a.levels_back, mutation=a.mutation,
           strategy=a.strategy, pop_size=5, penalty=a.penalty, stop_on=a.stop_on, seed_leaves="off",
           targets_sha256=sha, binary_sha256=cgp.file_sha256(exe), rust_source_sha256=cgp.RUST_SOURCE_SHA256)
spec = cgp.run_spec(g, ctx, seed=a.seed, n_nodes=a.n_nodes, max_evals=max_evals, cpu_limit=a.timeout)
status, child_cpu, stdout, stderr = cgp.run_engine(exe, "run", spec)
row.update(child_cpu_seconds=child_cpu, cpu_seconds=time.process_time() - c0 + child_cpu,
           wall_seconds=time.time() - t0)
if os.WIFSIGNALED(status):
    sig = os.WTERMSIG(status)
    row.update(child_signal=sig, engine_stderr=stderr[-2000:])   # a Rust panic aborts (SIGABRT): keep its message
    print(f"cgp engine killed by signal {sig} after {child_cpu:.1f} CPU s; its stderr (tail):\n{stderr[-2000:]}",
          file=sys.stderr, flush=True)
    print("RESULT " + json.dumps(row), flush=True)
    if sig not in (signal.SIGKILL, signal.SIGSTOP):      # their handlers cannot be set (Linux kills at the hard limit)
        signal.signal(sig, signal.SIG_DFL)
    os.kill(os.getpid(), sig)                    # the queue records this signal: timeout if SIGXCPU, else error
    sys.exit(f"signal {sig} did not end the task")
if os.WEXITSTATUS(status) != 0:
    sys.exit(f"cgp engine failed (exit {os.WEXITSTATUS(status)}):\n{stderr[-2000:]}")
r = json.loads(stdout)
if r["source_sha256"] != cgp.RUST_SOURCE_SHA256:
    sys.exit(f"refused: the engine reports source {r['source_sha256']}, not {cgp.RUST_SOURCE_SHA256}")
row.update(cgp.outcome(g, ctx, r), child_signal=0)
print("RESULT " + json.dumps(row, default=float), flush=True)

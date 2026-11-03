"""Enumerate one grammar up to kmax (or until a budget) -> RESULT line.
Levels are checkpointed in <root>/<grammar>/, so a rerun resumes. Study 2 phases pass --root data/enumeration_s2/...,
a CPU budget below the queue's hard limit (clean stop, REQ-MEA-18) and a free-disk floor (REQ-RUN-16)."""
import argparse
import json

from reach import grammar as G
from reach.enumerate import Enumerator
from reach.runner.queue import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--grammar", required=True)
ap.add_argument("--kmax", type=int, default=31)
ap.add_argument("--max_gb", type=float, default=4.0)
ap.add_argument("--root", default="data/enumeration", help="output root, relative to the repository")
ap.add_argument("--cpu_budget", type=float, default=None, help="stop cleanly before exceeding this many CPU seconds")
ap.add_argument("--min_free_gb", type=float, default=0.0, help="stop cleanly if less free disk would remain")
ap.add_argument("--precision", type=int, default=G.HASH_PRECISION, help="decimals in the real-valued key")
a = ap.parse_args()
e = Enumerator(G.get(a.grammar), ROOT / a.root / a.grammar, log=lambda s: print(s, flush=True), precision=a.precision)
k, state = e.run(a.kmax, a.max_gb, cpu_budget=a.cpu_budget, min_free_gb=a.min_free_gb)
if e.stop_reason == "disk":       # the machine, not the grammar: fail so the queue records an error and it is retried
    import sys
    sys.exit(f"stopped for disk at K={k}: free disk below {a.min_free_gb} GB; this is not a ceiling (row 70)")
print("RESULT " + json.dumps(dict(grammar=a.grammar, terminal_state=state, k_reached=k, stop_reason=e.stop_reason,
                                  U_cum=e.stats[-1]["U_cum"], levels=len(e.stats), root=a.root)), flush=True)

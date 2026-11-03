"""One Operon run (tool T20; H9; D056) -> one `RESULT {json}` line. Runs in .venv-ext (Python 3.11, pyoperon 0.6.2)
with DYLD_LIBRARY_PATH=.venv-ext/operon-libs and PYTHONPATH=src.

Training data: 75 points drawn uniformly on (2, 3) with the run's seed (fixed for the run). Budget: max_evaluations =
budget x 2x10^4 (one run per budget; no prefix property, D056). Configurations: "symbolic" (symbolic_mode, integer
coefficients, objective mse) and "default" (Operon's practitioner setting: R^2 objective, 10 local-search iterations).
Exactness is scored in double precision (s2_operon_score). Held-out sets are refused before Stage 2 (s2_lock)."""
import argparse
import hashlib
import json
import sys
import time

import numpy as np

from reach import grammar as G
from reach.s2_lock import check_search_allowed
from reach.tasks.s2_operon_score import is_exact

SYMBOLS = {"R-ARITH": "add,sub,mul", "R-ARITH4": "add,sub,mul,div", "R-ELEM6": "add,sub,mul,div,exp,log",
           "R-ELEM4": "add,mul,exp,log", "R-TRIG": "add,mul,sin,cos",
           "R-SRB": "add,sub,mul,div,exp,log,sin,cos,sqrt,square"}

ap = argparse.ArgumentParser()
for name in ("--set", "--targets", "--target_id", "--targets_sha256"):
    ap.add_argument(name, required=True)
ap.add_argument("--config", choices=["symbolic", "default"], required=True)
ap.add_argument("--budget", type=float, required=True, help="multiples of E0 = 2x10^4 evaluations")
ap.add_argument("--seed", type=int, required=True)
a = ap.parse_args()
try:
    check_search_allowed(a.set)
except PermissionError as err:
    sys.exit(f"refused: {err}")
raw = open(a.targets, "rb").read()
if hashlib.sha256(raw).hexdigest() != a.targets_sha256:
    sys.exit("refused: target file checksum differs from the registered one")
from pyoperon.sklearn import SymbolicRegressor   # noqa: E402  (after the checks: fail fast without Operon)

g = G.get(a.set)
tgt = next(t for t in json.loads(raw) if t["target_id"] == a.target_id)
tree = G.parse_expr(tgt["expression"], g)
rng = np.random.default_rng(a.seed)
xtr = rng.uniform(2.0, 3.0, 75)
ytr = G.evaluate(g, tree, xtr.astype(complex)).real
grid = G.sample_grid(g)
gv = np.array([complex(*z) for z in tgt["grid_values"]]) if "grid_values" in tgt else G.evaluate(g, tree, grid)
kw = dict(allowed_symbols=SYMBOLS[a.set] + ",variable,constant", max_evaluations=int(a.budget * 20_000),
          generations=100_000, random_state=a.seed, n_threads=1)
kw.update(dict(symbolic_mode=True, objectives=["mse"]) if a.config == "symbolic" else dict(optimizer_iterations=10))
t0, c0 = time.time(), time.process_time()
m = SymbolicRegressor(**kw)
m.fit(xtr.reshape(-1, 1), ytr)
s = m.get_model_string(m.model_, 17)
exact = is_exact(s, gv, tgt["target_scale"], grid.real)
print("RESULT " + json.dumps(dict(
    set=g.name, domain="real", algorithm="operon", config=a.config, seed=a.seed, target_id=tgt["target_id"],
    K_proven=tgt.get("K_proven"), K_construction=tgt.get("K_construction"), units=tgt.get("units"),
    budget=a.budget, budget_E=int(a.budget * 20_000), evaluations=int(m.stats_.get("evaluation_count", -1)),
    residual_evaluations=int(m.stats_.get("residual_evaluations", -1)),          # with local search these, plus the
    jacobian_evaluations=int(m.stats_.get("jacobian_evaluations", -1)),          # Jacobian ones, use max_evaluations
    generations=int(m.stats_.get("generations", -1)), exact=bool(exact),
    terminal_state="converged" if exact else "budget_exhausted", model=s[:2000],
    cpu_seconds=time.process_time() - c0,
    wall_seconds=time.time() - t0)),        # recorded only, never compared (CPU-time rule)
    flush=True)

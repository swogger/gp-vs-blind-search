"""Exploratory modern-system check (D024): PySR on the calculator targets of one EML-family operator.
Runs under .venv-pysr (Python 3.11 + Julia; PySR 2.7). Constants are disabled (complexity 100 > maxsize); the
operator's distinguished constant is supplied as an input column `c`. Exact recovery = some hall-of-fame
equation matches the target on the definition grid (scale-aware 9-digit hash, as for GP). Julia uses IEEE
complex arithmetic, not our faithful semantics. Writes data/family/pysr_<op>.jsonl (one line per run)."""
import argparse
import json
import time

import numpy as np
import sympy
from pysr import PySRRegressor

from reach import grammar as G
from reach.paths import ROOT

OPS = {"F-EML": ("eml(x, y) = exp(x) - log(y)", lambda a, b: sympy.exp(a) - sympy.log(b), 1.0),
       "F-EDL": ("edl(x, y) = exp(x) / log(y)", lambda a, b: sympy.exp(a) / sympy.log(b), float(np.e)),
       "F-LDE0": ("lde(x, y) = log(x) / exp(y)", lambda a, b: sympy.log(a) / sympy.exp(b), 0.0),
       "F-LDE1": ("lde(x, y) = log(x) / exp(y)", lambda a, b: sympy.log(a) / sympy.exp(b), 1.0),
       "F-PLI": ("pli(x, y) = log(x)^(1/y)", lambda a, b: sympy.log(a) ** (1 / b), 1.0),
       "F-PLM": ("plm(x, y) = log(x)^(-y)", lambda a, b: sympy.log(a) ** (-b), 1.0)}

ap = argparse.ArgumentParser()
ap.add_argument("--grammar", required=True)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--budgets", default="20000,320000")
ap.add_argument("--fit_timeout", type=float, default=None)   # D027: completion runs cap each fit (seconds)
a = ap.parse_args()
op_def, op_sym, cval = OPS[a.grammar]
op_name = op_def.split("(")[0]
out_path = ROOT / f"data/family/pysr_{a.grammar}.jsonl"
done = set()
if out_path.exists():
    for line in open(out_path):
        try:
            r = json.loads(line); done.add((r["target_id"], r["budget"], r["seed"]))
        except json.JSONDecodeError:
            pass
sets = [(a.grammar, "calc"), ("F2-" + a.grammar[2:], "calc2")]
n_runs = n_exact = 0
with open(out_path, "a") as fh:
    for gname, tname in sets:
        p = ROOT / f"data/targets/{gname}/{tname}.json"
        if not p.exists():
            continue
        g = G.get(gname)
        grid = G.sample_grid(g)
        cols = [grid] if len(g.variables) == 1 else [grid[v] for v in g.variables]
        X = np.stack(cols + [np.full(cols[0].shape, complex(cval))], axis=1)
        names = list(g.variables) + ["c"]
        for t in json.loads(p.read_text()):
            y = G.evaluate(g, G.parse_expr(t["expression"], g), grid)
            for budget in [int(b) for b in a.budgets.split(",")]:
                for seed in range(a.seeds):
                    if (t["target_id"], budget, seed) in done:
                        continue
                    t0 = time.time()
                    m = PySRRegressor(binary_operators=[op_def], unary_operators=[],
                                      extra_sympy_mappings={op_name: op_sym}, maxsize=25,
                                      complexity_of_constants=100, should_optimize_constants=False, parsimony=0.0,
                                      max_evals=budget, niterations=10 ** 7, populations=4, population_size=25,
                                      deterministic=True, parallelism="serial", random_state=seed, precision=64,
                                      progress=False, verbosity=0, temp_equation_file=True,
                                      timeout_in_seconds=a.fit_timeout)
                    exact, best, err = False, "", ""
                    try:
                        m.fit(X, y, variable_names=names)
                        eqs = m.equations_
                        best = str(eqs.iloc[eqs["loss"].values.argmin()]["equation"])
                        h = G.scaled_func_hash(y, t["target_scale"])
                        for i in range(len(eqs)):
                            if G.scaled_func_hash(m.predict(X, index=i), t["target_scale"]) == h:
                                exact = True
                                break
                    except Exception as e:      # recorded, never silent
                        err = repr(e)[:200]
                    rec = dict(grammar=a.grammar, target_set=tname, target_id=t["target_id"], function=t["function"],
                               K=t["K_construction"], budget=budget, seed=seed, exact=exact, best=best, error=err,
                               wall_s=round(time.time() - t0, 2), fit_timeout=a.fit_timeout)
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                    n_runs += 1; n_exact += exact
                    print(f"{t['target_id']} E={budget} seed={seed} exact={exact} {rec['wall_s']}s", flush=True)
print("RESULT " + json.dumps(dict(grammar=a.grammar, terminal_state="completed", runs_this_session=n_runs,
                                  exact_this_session=n_exact)), flush=True)

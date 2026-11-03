"""One tree-GP run -> one `RESULT {json}` line (generalises legacy research/rerun/synth_task.py)."""
import argparse
import json

from reach import grammar as G
from reach.engines.treegp import run_gp

VAL_SEED = 20260721 + 1          # EML-GP's validation stream (run_synthetic.py)

ap = argparse.ArgumentParser()
ap.add_argument("--grammar", required=True)
ap.add_argument("--targets", required=True, help="JSON list of targets (expression, target_id, ...)")
ap.add_argument("--target_id", required=True)
ap.add_argument("--regime", choices=["A", "B"], required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--generations", type=int, default=200)
ap.add_argument("--pop_size", type=int, default=100)
ap.add_argument("--timeout", type=float, default=600.0, help="CPU seconds")
ap.add_argument("--legacy", action="store_true", help="reproduce EML-GP's engine exactly")
ap.add_argument("--seed_leaves", choices=["default", "on", "off"], default="default")
ap.add_argument("--stop_on", choices=["tolerance", "exact"], default="tolerance")
ap.add_argument("--algorithm", choices=["gp", "es"], default="gp", help="es: (1+4)-ES, budget = pop_size x generations")
ap.add_argument("--penalty", choices=["fixed", "scaled"], default="fixed", help="non-finite residual cost (D034)")
ap.add_argument("--crossover", choices=["subtree", "headless"], default="subtree", help="headless-chicken control (D034)")
a = ap.parse_args()

g = G.get(a.grammar)
tgt = next(t for t in json.load(open(a.targets)) if t["target_id"] == a.target_id)
tree = G.parse_expr(tgt["expression"], g)
kw = {}
if g.kind == "real":
    grid, val_xs = G.sample_grid(g), G.sample_val(g, VAL_SEED)
    kw = dict(val_xs=val_xs, val_target=G.evaluate(g, tree, val_xs), var_norm=tgt["var_target"],
              grid_xs=grid, grid_target=G.evaluate(g, tree, grid), grid_scale=tgt["target_scale"])
if a.algorithm == "es":
    from reach.engines.es import run_es
    r = run_es(g, tree, a.seed, evaluations=a.pop_size * a.generations, timeout_seconds=a.timeout,
               undefined_penalty=a.penalty, **kw)
else:
  r = run_gp(g, tree, a.regime, a.seed, generations=a.generations, pop_size=a.pop_size,
           timeout_seconds=a.timeout, legacy=a.legacy,
           seed_leaves=None if a.seed_leaves == "default" else a.seed_leaves == "on", stop_on=a.stop_on,
           undefined_penalty=a.penalty, crossover_mode=a.crossover, **kw)
print("RESULT " + json.dumps({
    "target_id": tgt["target_id"], "grammar": g.name, "K_construction": tgt.get("K_construction"),
    "K_proven": tgt.get("K_proven"), "regime": a.regime, "seed": a.seed,
    "budget_E": a.pop_size * a.generations, "legacy": a.legacy, "stop_on": a.stop_on, "algorithm": a.algorithm,
    "penalty": a.penalty, "crossover": a.crossover,
    "converged": "true" if r["converged"] else "false",
    "exact_hash_match": "true" if r["exact_hash_match"] else "false",
    "gens_to_converge": "" if r["gens_to_converge"] is None else r["gens_to_converge"],
    "final_nmse": repr(r["final_nmse"]), "final_tree_size": r["final_tree_size"],
    "terminal_state": r["terminal_state"], "gens_run": r["gens_run"], "evals": r["evals"],
    "wall_seconds": repr(r["wall_seconds"]), "cpu_seconds": repr(r["cpu_seconds"]),
    "best_expr": r["best_expr"]}), flush=True)

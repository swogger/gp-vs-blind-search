"""One synthetic dense-K run -> one JSON line on stdout (schema of synthetic_runs.csv + extras)."""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "synthetic"))
import eml_synth_core as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--targets", required=True)
ap.add_argument("--target_id", required=True)
ap.add_argument("--regime", choices=["A", "B"], required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--generations", type=int, default=C.GENERATIONS)
ap.add_argument("--plateau", type=int, default=None)
ap.add_argument("--timeout", type=float, default=C.TIMEOUT_SECONDS)
ap.add_argument("--no_size_penalty", action="store_true")
ap.add_argument("--point_mutation", action="store_true")
ap.add_argument("--cpu_clock", action="store_true")
a = ap.parse_args()

tgt = next(t for t in json.load(open(a.targets)) if t["target_id"] == a.target_id)
tree = C.parse_expr(tgt["expression"])
grid, val_xs = C.grid_sample(), C.validation_sample(20260721 + 1)   # as run_synthetic.py
r = C.run_gp(tree, a.regime, a.seed, val_xs, C.vec_eval(tree, val_xs), tgt["var_target"],
             grid_xs=grid, grid_target=C.vec_eval(tree, grid), grid_scale=tgt["target_scale"],
             generations=a.generations, timeout_seconds=a.timeout,
             size_penalty=not a.no_size_penalty, point_mutation=a.point_mutation,
             cpu_clock=a.cpu_clock, plateau_gens=a.plateau)
print("RESULT " + json.dumps({
    "target_id": tgt["target_id"], "K_construction": tgt["K_construction"],
    "K_proven": tgt["K_proven"], "m_operators": tgt["m_operators"], "regime": a.regime,
    "seed": a.seed, "converged": "true" if r["converged"] else "false",
    "exact_hash_match": "true" if r["exact_hash_match"] else "false",
    "gens_to_converge": "" if r["gens_to_converge"] is None else r["gens_to_converge"],
    "final_nmse": repr(r["final_nmse"]), "final_tree_size": r["final_tree_size"],
    "terminal_state": r["terminal_state"], "wall_seconds": repr(r["wall_seconds"]),
    "cpu_seconds": repr(r["cpu_seconds"]), "gens_run": r["gens_run"],
    "best_expr": r["best_expr"]}), flush=True)

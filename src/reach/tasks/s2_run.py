"""One Study 2 search run -> one `RESULT {json}` line (tool T6; C6; REQ-RUN-08).

Every scientific setting is required: there are no defaults for algorithm, configuration, penalty, stop rule, seeded
leaf, budget or CPU limit (the archived pilot ran on silent defaults; review F8, F12). The task refuses held-out sets
before the Stage 2 registration (s2_lock) and a target file whose SHA-256 differs from the one given (REQ-MEA-10).
With stop on exact, `gen_exact` is the generation of the first exact match, so every shorter budget is read from this
run (C2); a run stopped by its CPU limit records the generation it reached (REQ-RUN-15).
Stage 5 (D125; REQ-ENG-30): a configuration that names one domain is refused on a set of the other; `--keys_out PATH`
(with --count_distinct) writes the run's distinct in-lookup keys to a side file and adds only the field `keys`.
"""
import argparse
import hashlib
import json
import math
import sys

from reach import grammar as G
from reach.engines.treegp import run_gp
from reach.s2_lock import check_search_allowed
from reach.tasks.s2_configs import CONFIGS

VAL_SEED = 20260721 + 1          # EML-GP's validation stream (as all paper-1 and Study 2 tasks)
ALGORITHMS = {"A": dict(regime="A", crossover_mode="subtree"), "B": dict(regime="B", crossover_mode="subtree"),
              "HC": dict(regime="A", crossover_mode="headless")}

ap = argparse.ArgumentParser()
req = dict(required=True)
ap.add_argument("--set", **req)
ap.add_argument("--targets", **req)
ap.add_argument("--target_id", **req)
ap.add_argument("--targets_sha256", **req)
ap.add_argument("--algorithm", choices=sorted(ALGORITHMS), **req)
ap.add_argument("--config", choices=sorted(CONFIGS), **req)
ap.add_argument("--seed", type=int, **req)
ap.add_argument("--generations", type=int, **req)
ap.add_argument("--pop_size", type=int, **req)
ap.add_argument("--penalty", choices=["scaled", "fixed"], **req)
ap.add_argument("--stop_on", choices=["exact"], **req)
ap.add_argument("--seed_leaves", choices=["off", "on"], **req)
ap.add_argument("--timeout", type=float, help="CPU seconds (soft limit; the queue adds a hard limit)", **req)
ap.add_argument("--print_tree", action="store_true",
                help="also record best_expr_full, the whole best tree (reruns, D091; changes nothing else)")
ap.add_argument("--count_distinct", action="store_true",
                help="also record rv: distinct functions per budget prefix (H6d re-runs, D098; changes nothing else)")
ap.add_argument("--enum_dir", help="the set's enumeration, for --count_distinct")
ap.add_argument("--keys_out", help="side file for the run's distinct in-lookup keys (Stage 5, REQ-ENG-28; needs "
                                   "--count_distinct; changes nothing else)")
a = ap.parse_args()
if a.count_distinct and not a.enum_dir:
    sys.exit("refused: --count_distinct needs --enum_dir")
if a.keys_out and not a.count_distinct:
    sys.exit("refused: --keys_out needs --count_distinct")

try:
    check_search_allowed(a.set)
except PermissionError as err:
    sys.exit(f"refused: {err}")
raw = open(a.targets, "rb").read()
sha = hashlib.sha256(raw).hexdigest()
if sha != a.targets_sha256:
    sys.exit(f"refused: target file checksum {sha} differs from the registered {a.targets_sha256}")

cfg = CONFIGS[a.config]
if a.pop_size != cfg["pop_size"]:
    sys.exit(f"refused: configuration {a.config} has population {cfg['pop_size']}, not {a.pop_size}")
g = G.get(a.set)
if cfg.get("domain", g.kind) != g.kind:
    sys.exit(f"refused: configuration {a.config} is for {cfg['domain']} sets, {a.set} is {g.kind}")
tgt = next(t for t in json.loads(raw) if t["target_id"] == a.target_id)
tree = G.parse_expr(tgt["expression"], g)
kw = {}
if g.kind == "real":
    grid, val = G.sample_grid(g), G.sample_val(g, VAL_SEED)
    kw = dict(val_xs=val, val_target=G.evaluate(g, tree, val), var_norm=tgt["var_target"], grid_xs=grid,
              grid_target=G.evaluate(g, tree, grid), grid_scale=tgt["target_scale"])
    if cfg.get("batch_var") == "wide":
        from reach.tasks.s2_configs import wide_var_norm
        kw["batch_var_norm"] = wide_var_norm(g, tree, cfg["kwargs"]["batch_domain"])
        if not (math.isfinite(kw["batch_var_norm"]) and kw["batch_var_norm"] > 0):
            sys.exit(f"refused: {tgt['target_id']}'s normaliser on the wide interval is {kw['batch_var_norm']}, not "
                     "finite and positive; the target is excluded from the wider-interval arm (guardrail row 173)")
counter = None
if a.count_distinct:
    from reach import revisit
    counter = revisit.Counter(g, tree, revisit.Lookup(g, a.enum_dir), a.pop_size,
                              [p for p in revisit.PREFIXES if p <= a.pop_size * a.generations],
                              record_keys=bool(a.keys_out))
r = run_gp(g, tree, ALGORITHMS[a.algorithm]["regime"], a.seed, generations=a.generations, pop_size=a.pop_size,
           timeout_seconds=a.timeout, legacy=False, seed_leaves=a.seed_leaves == "on", stop_on=a.stop_on,
           undefined_penalty=a.penalty, crossover_mode=ALGORITHMS[a.algorithm]["crossover_mode"], full_expr=a.print_tree,
           counter=counter,
           **cfg["kwargs"], **kw)
exact = bool(r["exact_hash_match"]) and r["terminal_state"] == "converged"
keys = counter.write_keys(a.keys_out) if a.keys_out else None
print("RESULT " + json.dumps(dict(
    set=g.name, domain=g.kind, algorithm=a.algorithm, config=a.config, seed=a.seed, target_id=tgt["target_id"],
    K_proven=tgt.get("K_proven"), K_construction=tgt.get("K_construction"), units=tgt.get("units"),
    budget_E=a.pop_size * a.generations, generations=a.generations, pop_size=a.pop_size, evals_per_gen=a.pop_size,
    penalty=a.penalty, stop_on=a.stop_on, seed_leaves=a.seed_leaves, exact=exact, converged=bool(r["converged"]),
    gen_exact=r["gens_run"] if exact else None, gen_tol=r["gens_to_converge"], gens_reached=r["gens_run"],
    terminal_state=r["terminal_state"], final_error=r["final_nmse"], final_tree_size=r["final_tree_size"],
    cpu_seconds=r["cpu_seconds"], wall_seconds=r["wall_seconds"], best_expr=r["best_expr"], targets_sha256=sha,
    **({"best_expr_full": r["best_expr_full"]} if a.print_tree else {}),
    **({"rv": counter.record()} if counter is not None else {}),
    **({"keys": keys} if keys is not None else {})),
    default=float), flush=True)

"""Execute the Dense-K synthetic GP runs.

10 paired seeds x 2 regimes per target, at the main study's reference cell
(pop 100, gen 200, node cap 2000, 600 s timeout).  Resumable: completed
(target_id, regime, seed) triples are read back from the output CSV and
skipped, so the job can be interrupted and restarted.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import eml_synth_core as C  # noqa: E402
from seeds import SEEDS     # noqa: E402

N_SEEDS = 10
RUN_COLUMNS = ["target_id", "K_construction", "K_proven", "m_operators", "regime",
               "seed", "converged", "exact_hash_match", "gens_to_converge",
               "final_nmse", "final_tree_size", "terminal_state", "wall_seconds"]

_G = {}
_G_OVERRIDE = {}


def _init(override=None):
    if override:
        _G_OVERRIDE.update(override)
    _G["grid"] = C.grid_sample()
    _G["val_xs"] = C.validation_sample(20260721 + 1)


def _work(task):
    tgt, regime, seed = task
    tree = C.parse_expr(tgt["expression"])
    grid, val_xs = _G["grid"], _G["val_xs"]
    grid_target = C.vec_eval(tree, grid)
    val_target = C.vec_eval(tree, val_xs)
    r = C.run_gp(
        tree, regime, seed, val_xs, val_target, tgt["var_target"],
        grid_xs=grid, grid_target=grid_target, grid_scale=tgt["target_scale"],
        **_G_OVERRIDE,
    )
    return {
        "target_id": tgt["target_id"],
        "K_construction": tgt["K_construction"],
        "K_proven": tgt["K_proven"],
        "m_operators": tgt["m_operators"],
        "regime": regime,
        "seed": seed,
        "converged": "true" if r["converged"] else "false",
        "exact_hash_match": "true" if r["exact_hash_match"] else "false",
        "gens_to_converge": "" if r["gens_to_converge"] is None else r["gens_to_converge"],
        "final_nmse": repr(r["final_nmse"]),
        "final_tree_size": r["final_tree_size"],
        "terminal_state": r["terminal_state"],
        "wall_seconds": repr(r["wall_seconds"]),
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.abspath(
        os.path.join(here, "..", "..", "results", "synthetic_dense_k")))
    ap.add_argument("--targets", default="targets.json")
    ap.add_argument("--out", default="synthetic_runs.csv")
    ap.add_argument("--workers", type=int, default=6)
    # Budget sweep (Definition 2 is budget-relative; K50 is otherwise measured at one budget).
    ap.add_argument("--generations", type=int, default=None,
                    help="Override the generation budget. Default: the module constant (200).")
    ap.add_argument("--pop_size", type=int, default=None,
                    help="Override the population size. Default: the module constant (100).")
    ap.add_argument("--k_max", type=int, default=None,
                    help="Restrict to targets with K_construction <= this value.")
    ap.add_argument("--n_seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()

    if args.generations is not None:
        _G_OVERRIDE["generations"] = args.generations
    if args.pop_size is not None:
        _G_OVERRIDE["pop_size"] = args.pop_size

    with open(os.path.join(args.dir, args.targets), encoding="utf-8") as fh:
        targets = json.load(fh)
    if args.k_max is not None:
        targets = [t for t in targets if t["K_construction"] <= args.k_max]
    out_path = os.path.join(args.dir, args.out)

    done = set()
    if os.path.exists(out_path):
        with open(out_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                done.add((row["target_id"], row["regime"], int(row["seed"])))

    seeds = SEEDS[1:args.n_seeds + 1]
    tasks = [(t, regime, s)
             for t in targets for regime in ("A", "B") for s in seeds
             if (t["target_id"], regime, s) not in done]

    total = len(targets) * 2 * args.n_seeds
    print(f"targets={len(targets)}  seeds={args.n_seeds}  regimes=2  total runs={total}")
    print(f"budget: generations={_G_OVERRIDE.get('generations', C.GENERATIONS)} "
          f"pop={_G_OVERRIDE.get('pop_size', C.POP_SIZE)}")
    print(f"already done={len(done)}  to run={len(tasks)}  workers={args.workers}")
    if not tasks:
        return

    new_file = not os.path.exists(out_path)
    t0 = time.time()
    n = 0
    with open(out_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RUN_COLUMNS)
        if new_file:
            w.writeheader()
        with Pool(args.workers, initializer=_init, initargs=(dict(_G_OVERRIDE),)) as pool:
            for row in pool.imap_unordered(_work, tasks, chunksize=4):
                w.writerow(row)
                n += 1
                if n % 50 == 0 or n == len(tasks):
                    fh.flush()
                    el = time.time() - t0
                    rate = n / el
                    eta = (len(tasks) - n) / rate if rate > 0 else 0
                    print(f"  {n}/{len(tasks)} runs  {el/60:.1f} min elapsed  "
                          f"{rate:.2f} runs/s  ETA {eta/60:.1f} min", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()

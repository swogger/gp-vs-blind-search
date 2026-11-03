"""Validate the vectorised evaluator against the reference cmath evaluator.

The GP's search operators are ga-lib verbatim; only fitness *evaluation* is
vectorised.  This script quantifies the difference on trees drawn from the same
generator the GP uses, measured in the units the experiment actually cares
about: nMSE against real accepted targets.

The number that matters is `min_nmse_among_disagreements` -- the best fitness
any tree achieved (in either evaluator) among trees where the two evaluators
disagreed.  If that is many orders of magnitude above the 1e-6 convergence
threshold, no disagreement can flip a converged/not-converged outcome.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eml_synth_core as C  # noqa: E402
from generator import generate_random_tree  # noqa: E402


def agreement_for_target(expr, var, n_trees, rng, n_points=75):
    import random as _rm
    _rm.choice, _rm.random = rng.choice, rng.random
    _rm.sample, _rm.uniform = rng.sample, rng.uniform

    xs_list = [rng.uniform(C.DOMAIN_LO, C.DOMAIN_HI) for _ in range(n_points)]
    xs = np.array([complex(v) for v in xs_list], dtype=np.complex128)
    tgt_tree = C.parse_expr(expr)
    tgt = C.vec_eval(tgt_tree, xs)
    tgt_list = [complex(v) for v in tgt]

    def ref_nmse(tree):
        if C.count_nodes(tree) > C.MAX_NODES:
            return float("inf")
        error = 0.0
        try:
            for xv, tv in zip(xs_list, tgt_list):
                d = abs(tree.evaluate(x=xv) - tv)
                error += 1000.0 if (math.isnan(d) or math.isinf(d)) else d * d
            return error / len(xs_list) / var
        except Exception:
            return float("inf")

    n_cmp = n_dis = 0
    worst_rel = 0.0
    min_nmse_dis = float("inf")
    for _ in range(n_trees):
        depth = rng.randint(1, C.MAX_DEPTH)
        method = "full" if rng.random() < 0.5 else "grow"
        tree = generate_random_tree(depth, C.OPERATORS, C.TERMINALS, C.VARIABLES, method)
        if C.count_nodes(tree) > C.MAX_NODES:
            continue
        a = C.mse_against(C.vec_eval(tree, xs), tgt) / var
        b = ref_nmse(tree)
        if math.isinf(a) and math.isinf(b):
            continue
        n_cmp += 1
        rel = abs(a - b) / max(abs(a), abs(b), 1e-300)
        if rel > 1e-9:
            n_dis += 1
            min_nmse_dis = min(min_nmse_dis, min(a, b))
        worst_rel = max(worst_rel, rel if math.isfinite(rel) else 1.0)
    return n_cmp, n_dis, worst_rel, min_nmse_dis


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.abspath(os.path.join(here, "..", "..", "results", "synthetic_dense_k"))
    targets = json.load(open(os.path.join(res, "targets.json"), encoding="utf-8"))

    rng = random.Random(4242)
    by_k = {}
    for t in targets:
        by_k.setdefault(t["K_construction"], []).append(t)
    sample = [rng.choice(v) for k, v in sorted(by_k.items())]

    tot_cmp = tot_dis = 0
    worst = 0.0
    min_dis = float("inf")
    per_target = []
    for t in sample:
        n_cmp, n_dis, w, mn = agreement_for_target(
            t["expression"], t["var_target"], 1500, rng)
        tot_cmp += n_cmp
        tot_dis += n_dis
        worst = max(worst, w)
        min_dis = min(min_dis, mn)
        per_target.append({"target_id": t["target_id"], "K": t["K_construction"],
                           "n_compared": n_cmp, "n_disagreements": n_dis,
                           "min_nmse_among_disagreements": None if math.isinf(mn) else mn})
        print(f"  {t['target_id']:>10}  compared={n_cmp:5d}  disagree={n_dis:4d}  "
              f"min nMSE among disagreements={mn:.4g}")

    out = {
        "description": "vectorised numpy evaluator vs ga-lib Node.evaluate (cmath), "
                       "random trees from the GP's own generator, nMSE units",
        "n_trees_compared": tot_cmp,
        "n_disagreements": tot_dis,
        "disagreement_rate": tot_dis / tot_cmp if tot_cmp else None,
        "worst_relative_nmse_difference": worst,
        "min_nmse_among_disagreements": None if math.isinf(min_dis) else min_dis,
        "convergence_threshold": C.CONV_THRESHOLD,
        "orders_of_magnitude_above_threshold": (
            None if math.isinf(min_dis) else math.log10(min_dis / C.CONV_THRESHOLD)),
        "per_target": per_target,
    }
    path = os.path.join(res, "evaluator_agreement.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1)
    print(f"\ncompared={tot_cmp} disagreements={tot_dis} "
          f"({100*tot_dis/max(tot_cmp,1):.2f}%)  "
          f"min nMSE among disagreements={min_dis:.4g} "
          f"({out['orders_of_magnitude_above_threshold']:.1f} orders above 1e-6)")


if __name__ == "__main__":
    main()

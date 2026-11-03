"""Random-sampling baseline (D038): draw trees exactly as GP's initial population (ramped half-and-half, depth 2 to
MAX_DEPTH, no seeded leaves) and count, for every random target of a grammar, how many sampled trees compute it
(same exact-recovery hash as the GP runs). One task per (grammar, part) -> one `RESULT {json}` line."""
import argparse
import json
import random

import numpy as np

from reach import grammar as G
from reach.engines.treegp import MAX_DEPTH, _patch
from reach.galib.generator import generate_ramped_population

ap = argparse.ArgumentParser()
ap.add_argument("--grammar", required=True)
ap.add_argument("--targets", required=True)
ap.add_argument("--n", type=int, required=True, help="trees to sample")
ap.add_argument("--seed", type=int, required=True)
a = ap.parse_args()

g = G.get(a.grammar)
xs = G.sample_grid(g)
want = {}
for t in json.load(open(a.targets)):
    v = G.evaluate(g, G.parse_expr(t["expression"], g), xs)
    want.setdefault(G.scaled_func_hash(v, t["target_scale"]), []).append(t["target_id"])
_patch(random.Random(a.seed))
hits, n, finite = {}, 0, 0
while n < a.n:
    for tree in generate_ramped_population(100, MAX_DEPTH, g.operators, list(g.constants), list(g.variables),
                                           seed_variable_leaves=False):
        n += 1
        if G.count_nodes(tree) > G.MAX_NODES:
            continue
        v = G.evaluate(g, tree, xs)
        if not np.isfinite(v).all():
            continue
        finite += 1
        for tid in want.get(G.scaled_func_hash(v, max(1.0, float(np.abs(v).max()))), ()):
            hits[tid] = hits.get(tid, 0) + 1
print("RESULT " + json.dumps({"grammar": g.name, "n": n, "finite": finite, "hits": hits, "seed": a.seed}), flush=True)

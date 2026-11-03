"""Random-sampling baseline for Study 2 (tool T9; REQ-MEA-13): draw trees exactly as GP's initial population (ramped
half-and-half, depth 2 to MAX_DEPTH, no seeded leaves) and count, for every target of a set, how many draws compute it
exactly (real: the same scale-aware hash as exact recovery, i.e. with the TARGET's scale; Boolean: the truth table).
A hash with the tree's own scale is only a pre-filter: alone it would count c·f as a hit on f (found 2026-10-19 in
paper 1's random_sample; guardrail row 42). Also records the shares of
undefined and constant draws (grammar properties, T11). P(hit within E) = 1 - (1 - hits/n)^E.

    python -m reach.tasks.s2_random --set S --targets F --targets_sha256 H --n 250000 --seed 1   -> RESULT line
"""
import argparse
import hashlib
import json
import random
import sys

import numpy as np

from reach import grammar as G
from reach.engines.treegp import MAX_DEPTH, _patch
from reach.galib.generator import generate_ramped_population


def draw_trees(g, n, seed):
    """n trees drawn like GP's initial population, from one seeded stream (ga-lib's global RNG is patched)."""
    _patch(random.Random(seed))
    k = 0
    while k < n:
        for tree in generate_ramped_population(100, MAX_DEPTH, g.operators, list(g.constants), list(g.variables),
                                               seed_variable_leaves=False):
            if k == n:
                return
            k += 1
            yield tree


def main():
    ap = argparse.ArgumentParser()
    for name in ("--set", "--targets", "--targets_sha256"):
        ap.add_argument(name, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    raw = open(a.targets, "rb").read()
    if hashlib.sha256(raw).hexdigest() != a.targets_sha256:
        sys.exit("refused: target file checksum differs from the registered one")
    g = G.get(a.set)
    targets = json.loads(raw)
    xs = G.sample_grid(g) if g.kind == "real" else None
    want, confirm = {}, {}
    for t in targets:
        tree = G.parse_expr(t["expression"], g)
        if g.kind == "real":
            v = G.evaluate(g, tree, xs)
            pre = G.scaled_func_hash(v, max(1.0, float(np.abs(v).max())))         # pre-filter (own scale)
            confirm[t["target_id"]] = (t["target_scale"], G.scaled_func_hash(v, t["target_scale"]))
            want.setdefault(pre, []).append(t["target_id"])
        else:
            want.setdefault(G.evaluate(g, tree), []).append(t["target_id"])
    hits = {t["target_id"]: 0 for t in targets}
    n = undefined = constant = big = 0
    for tree in draw_trees(g, a.n, a.seed):
        n += 1
        if G.count_nodes(tree) > G.MAX_NODES:
            big += 1
            continue
        if g.kind == "real":
            v = G.evaluate(g, tree, xs)
            if not np.isfinite(v).all():
                undefined += 1
                continue
            if np.abs(v - v.mean()).max() <= 1e-12:
                constant += 1
            key = G.scaled_func_hash(v, max(1.0, float(np.abs(v).max())))
        else:
            key = G.evaluate(g, tree)
            constant += key in (0, g.mask)
        for tid in want.get(key, ()):
            if g.kind == "real":                                                 # confirm with the target's scale
                scale, h = confirm[tid]
                if G.scaled_func_hash(v, scale) != h:
                    continue
            hits[tid] += 1
    print("RESULT " + json.dumps(dict(set=g.name, terminal_state="completed", n=n, seed=a.seed, hits=hits,
                                      undefined_share=undefined / n, constant_share=constant / n,
                                      oversize_share=big / n)), flush=True)


if __name__ == "__main__":
    main()

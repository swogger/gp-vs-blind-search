"""Target generation (M5; decision D018).

Within a grammar's enumeration ceiling: draw uniformly among the distinct functions whose
minimal size is exactly K (K_proven = K). Above it: compose an operator with minimal
functions from the enumeration whose sizes sum to K - 1 (depth <= 10); keep viable, novel ones that are absent from the enumeration
(so their minimal size exceeds the ceiling) (K_proven = None,
K_construction = K, an upper bound). Every real target must also pass EML-GP's dead-subtree
filter (D020), so none is within tolerance of a smaller tree. Real viability follows EML-GP: finite on the grid
and on the validation sample, variance >= 1e-12, magnitude ratio <= 1e6.
"""
import json
import random
from pathlib import Path

import numpy as np

from reach import grammar as G
from reach.engines.treegp import CONV_THRESHOLD, MAX_DEPTH
from reach.enumerate import Enumerator, read_level
from reach.galib.node import Node

VAL_SEED = 20260721 + 1
MIN_VARIANCE, MAX_MAGNITUDE_RATIO = 1e-12, 1e6


def load_structure(g, enum_dir):
    """Keys and parent pointers of every saved level (no values: cheap)."""
    e = Enumerator(g, enum_dir, log=lambda *_: None)
    stats = json.loads((Path(enum_dir) / "counts.json").read_text())["levels"]
    for st in stats:
        keys, par, depth = read_level(Path(enum_dir) / f"level_{st['K']:03d}.npz")
        e.keys[st["K"]], e.parents[st["K"]], e.depth[st["K"]] = keys, par, depth
    e.stats = stats
    e.known = np.sort(np.concatenate([e.keys[k] for k in e.keys]))
    return e


def _real_info(g, tree):
    grid, val = G.sample_grid(g), G.sample_val(g, VAL_SEED)
    v, vv = G.evaluate(g, tree, grid), G.evaluate(g, tree, val)
    if not (np.isfinite(v).all() and np.isfinite(vv).all()):
        return None, "nonfinite"
    with np.errstate(all="ignore"):
        var = float(np.mean(np.abs(v - v.mean()) ** 2))
        mag = np.abs(v)
    if not np.isfinite(var) or var < MIN_VARIANCE:
        return None, "variance"
    if mag.min() <= 0 or mag.max() / mag.min() > MAX_MAGNITUDE_RATIO:
        return None, "magnitude_ratio"
    return dict(var_target=var, target_scale=float(max(1.0, mag.max())), vals=v), ""


def _subtrees(tree, path=()):
    yield path, tree
    if tree.is_operator:
        yield from _subtrees(tree.left, path + ("l",))
        if tree.right is not None:
            yield from _subtrees(tree.right, path + ("r",))


def _replace(tree, path, new):
    if not path:
        return new
    t = tree.copy()
    node = t
    for step in path[:-1]:
        node = node.left if step == "l" else node.right
    if path[-1] == "l":
        node.left = new
    else:
        node.right = new
    return t


def _same(a, b):
    """EML-GP's identity test: equal after rounding to 9 decimals (func_hash)."""
    with np.errstate(all="ignore"):
        ra = np.round(a, G.HASH_PRECISION) + 0.0
        rb = np.round(b, G.HASH_PRECISION) + 0.0
    return np.array_equal(ra, rb, equal_nan=True)


def dead_subtree(g, tree, vals, var):
    """EML-GP's test: some proper subtree can be replaced by a probe leaf (a grammar leaf or
    the constants 2.5, 0.5) without leaving the convergence tolerance."""
    grid = G.sample_grid(g)
    probes = [G.leaf(v) for v in list(g.variables) + list(g.constants)] + [G.leaf(2.5), G.leaf(0.5)]
    probe_vals = [G.evaluate(g, p, grid) for p in probes]
    for path, node in _subtrees(tree):
        if not path:
            continue
        own = G.evaluate(g, node, grid)
        for p, pv in zip(probes, probe_vals):
            if _same(pv, own):                              # the same function: not a substitution
                continue
            cand = _replace(tree, path, p.copy())
            if G.mse_against(G.evaluate(g, cand, grid), vals) / var <= CONV_THRESHOLD:
                return True
    return False


def generate(g, enum_dir, levels, n_per_level, seed, max_attempts=200_000):
    """Returns (targets, log). levels: sizes K; above the ceiling the sampler is used."""
    e = load_structure(g, enum_dir)
    ceiling = max(k for k in e.keys)
    rng = random.Random(seed)
    targets, log = [], {}
    for K in levels:
        rej = {}
        got = []
        if K <= ceiling:
            idx = list(range(len(e.keys.get(K, []))))
            rng.shuffle(idx)
            for i in idx:
                if len(got) == n_per_level:
                    break
                tree = G.parse_expr(e.expr(K, i), g)
                if G.tree_depth(tree) > MAX_DEPTH:
                    rej["depth"] = rej.get("depth", 0) + 1
                    continue
                rec = _record(g, tree, K, K, rej)
                if rec:
                    got.append(rec)
        else:                       # compose minimal parts from the enumeration (D018)
            seen = set()
            jobs = [(o, K - 1, None) for o in g.ops if o.arity == 1 and K - 1 in e.keys]
            jobs += [(o, i, K - 1 - i) for o in g.ops if o.arity == 2 for i in range(1, K - 1)
                     if i in e.keys and K - 1 - i in e.keys]
            w = [len(e.keys[i]) * (1 if j is None else len(e.keys[j])) for _, i, j in jobs]
            jobs = [jb for jb, wt in zip(jobs, w) if wt > 0]        # e.g. NAND has no even sizes
            w = [wt for wt in w if wt > 0]
            for _ in range(max_attempts if jobs else 0):
                if len(got) == n_per_level:
                    break
                o, i, j = rng.choices(jobs, weights=w)[0]
                left = G.parse_expr(e.expr(i, rng.randrange(len(e.keys[i]))), g)
                right = None if j is None else G.parse_expr(e.expr(j, rng.randrange(len(e.keys[j]))), g)
                tree = Node(o.name, left, right, True)
                if G.tree_depth(tree) > MAX_DEPTH:
                    rej["depth"] = rej.get("depth", 0) + 1
                    continue
                rec = _record(g, tree, K, None, rej, e=e, seen=seen)
                if rec:
                    got.append(rec)
        for j, r in enumerate(got):
            r["target_id"] = f"{g.name}_K{K:02d}_{j:02d}"
        targets += got
        log[K] = dict(found=len(got), rejected=rej, proven=K <= ceiling)
    return targets, dict(grammar=g.name, seed=seed, ceiling=ceiling, levels=log)


def _record(g, tree, K, k_proven, rej, e=None, seen=None):
    def no(reason):
        rej[reason] = rej.get(reason, 0) + 1
    rec = dict(expression=G.expr_str(tree), K_construction=K, K_proven=k_proven, depth=G.tree_depth(tree))
    if g.kind == "real":
        info, why = _real_info(g, tree)
        if info is None:
            return no(why)
        vals = info.pop("vals")
        rec.update(info)
    else:
        vals = G.evaluate(g, tree)
        if vals in (0, g.mask):
            return no("constant")
    if e is not None:                                    # above the ceiling
        key = e._key(vals[None, :] if g.kind == "real" else np.array([vals], dtype=np.uint64))[0]
        if key in seen:
            return no("duplicate")
        pos = np.searchsorted(e.known, key)
        if pos < len(e.known) and e.known[pos] == key:
            return no("smaller_in_enumeration")
    if g.kind == "real" and dead_subtree(g, tree, vals, rec["var_target"]):
        return no("dead_subtree")                      # every level (D020): no near-smaller targets
    if seen is not None:
        seen.add(key)
    return rec

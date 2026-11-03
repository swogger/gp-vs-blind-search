"""Uniform sampling of trees with exactly K nodes and depth <= h, for any grammar (M3).

T[k][h] = number of shapes-with-symbols: T[1][h] = a0 (h >= 1),
T[k][h] = a1 T[k-1][h-1] + a2 sum_{i+j=k-1} T[i][h-1] T[j][h-1]. Depth: a leaf has depth 1.
"""
from functools import lru_cache

from reach import grammar as G
from reach.galib.node import Node


def make_sampler(g):
    a0, a1, a2 = g.arity_counts()
    leaves = list(g.variables) + list(g.constants)
    un = [o.name for o in g.ops if o.arity == 1]
    bi = [o.name for o in g.ops if o.arity == 2]

    @lru_cache(maxsize=None)
    def T(k, h):
        if k < 1 or h < 1:
            return 0
        if k == 1:
            return a0
        n = a1 * T(k - 1, h - 1)
        return n + a2 * sum(T(i, h - 1) * T(k - 1 - i, h - 1) for i in range(1, k - 1))

    def sample(k, h, rng):
        if T(k, h) == 0:
            raise ValueError(f"no tree with {k} nodes and depth <= {h}")
        if k == 1:
            return G.leaf(rng.choice(leaves))
        r = rng.randrange(T(k, h))
        u = a1 * T(k - 1, h - 1)
        if r < u:
            return Node(rng.choice(un), sample(k - 1, h - 1, rng), None, True)
        r -= u
        for i in range(1, k - 1):
            w = a2 * T(i, h - 1) * T(k - 1 - i, h - 1)
            if r < w:
                return Node(rng.choice(bi), sample(i, h - 1, rng), sample(k - 1 - i, h - 1, rng), True)
            r -= w
        raise AssertionError("unreachable")

    sample.count = T
    return sample

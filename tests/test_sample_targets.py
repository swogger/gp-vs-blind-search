"""Sampler counts and uniformity; target generation labels sizes correctly."""
import collections
import random

import numpy as np

from reach import grammar as G
from reach.sample import make_sampler
from reach.targets import generate
from reach.paths import ROOT
import pytest


def _need_levels(name, k):
    if not (ROOT / f"data/enumeration/{name}/level_{k:03d}.npz").exists():
        pytest.skip(f"enumeration levels of {name} not present (untracked; rebuild with the enumeration phase)")


def test_counts_match_recurrence_without_depth_limit():
    for name in ("EML", "ELEM6", "BOOL-AON-n3"):
        g = G.get(name)
        s = make_sampler(g)
        assert [s.count(k, 50) for k in range(1, 12)] == G.syntactic_counts(g, 11)[1:]


def test_sampler_is_uniform_on_small_space():
    g = G.get("EML")
    s, rng = make_sampler(g), random.Random(1)
    c = collections.Counter(G.expr_str(s(5, 10, rng)) for _ in range(16000))
    assert len(c) == 16 and min(c.values()) > 850 and max(c.values()) < 1150   # expected 1000 each


def test_depth_limit_respected():
    g, rng = G.get("ELEM6"), random.Random(2)
    s = make_sampler(g)
    assert all(G.tree_depth(s(15, 6, rng)) <= 6 for _ in range(300))


def test_targets_proven_and_constructed():
    _need_levels("ELEM6", 12)
    g = G.get("ELEM6")
    ts, log = generate(g, ROOT / "data/enumeration/ELEM6", [7, 13], 3, seed=5)
    assert [t["K_proven"] for t in ts] == [7, 7, 7, None, None, None]
    assert log["levels"][13]["proven"] is False
    for t in ts:
        assert G.count_nodes(G.parse_expr(t["expression"], g)) == t["K_construction"]


def test_dead_subtree_controls():
    import json
    from reach.targets import dead_subtree
    g, grid = G.get("EML"), G.grid_sample()
    ts = [t for t in json.loads((ROOT / "legacy/results/synthetic_dense_k/targets.json").read_text())[:61]
          if t["target_id"] != "reviewer_ce"]          # force-included in EML-GP, bypassed filters
    flagged = sum(dead_subtree(g, G.parse_expr(t["expression"], g),
                               G.evaluate(g, G.parse_expr(t["expression"], g), grid), t["var_target"]) for t in ts)
    assert flagged == 0                               # EML-GP's targets all passed this filter
    g6 = G.get("ELEM6")
    tree = G.parse_expr("+(x, *(-(x, x), exp(x)))", g6)   # exp(x) is dead: (x - x) * anything = 0
    v = G.evaluate(g6, tree, grid)
    assert dead_subtree(g6, tree, v, float(np.var(v.real)))


def test_generated_real_targets_have_no_dead_subtree():
    from reach.targets import dead_subtree
    _need_levels("EML", 17)
    g, grid = G.get("EML"), G.grid_sample()
    ts, _ = generate(g, ROOT / "data/enumeration/EML", [17], 5, seed=9)
    for t in ts:
        tr = G.parse_expr(t["expression"], g)
        assert not dead_subtree(g, tr, G.evaluate(g, tr, grid), t["var_target"])

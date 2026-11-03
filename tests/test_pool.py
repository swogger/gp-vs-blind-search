"""T12 shared-function pool (written before the code). REQ-MEA-15."""
import json

import numpy as np
import pytest

from reach import grammar as G
from reach.enumerate import Enumerator
from reach.tasks import s2_pool as P

QUIET = dict(log=lambda *_: None)


@pytest.fixture(scope="module")
def bool3(tmp_path_factory):
    d = tmp_path_factory.mktemp("pool")
    dirs = {}
    for name in ("BOOL-NAND-n3", "BOOL-AON-n3", "BOOL-AONX-n3", "BOOL-RICH-n3"):
        Enumerator(G.get(name), d / name, **QUIET).run(40)
        dirs[name] = d / name
    return dirs


def test_join_finds_functions_proven_in_enough_sets(bool3):
    sizes = P.join_sizes(bool3)
    assert sizes, "no shared functions"
    for key, per in sizes.items():
        for s, k in per.items():
            g = G.get(s)
            e = Enumerator(g, bool3[s], **QUIET)
            e.load()
            assert e.min_size(np.uint64(key)) == k          # the joined size is the set's proven minimum size
    full = {k: v for k, v in sizes.items() if len(v) == 4}
    assert len(full) > 50


def test_pool_has_the_same_function_in_every_set(bool3):
    pool = P.build_pool(bool3, min_sets=3, n=20, seed=1)
    assert 0 < len({p["pool_id"] for t in pool.values() for p in t}) <= 20
    by_id = {}
    for s, ts in pool.items():
        g = G.get(s)
        for t in ts:
            tt = G.evaluate(g, G.parse_expr(t["expression"], g))
            by_id.setdefault(t["pool_id"], set()).add(int(tt))
            assert t["K_proven"] == G.count_nodes(G.parse_expr(t["expression"], g))
    assert all(len(v) == 1 for v in by_id.values())          # one function per pool id across sets
    for pid in by_id:
        assert sum(1 for ts in pool.values() for t in ts if t["pool_id"] == pid) >= 3


def test_stratified_by_size_spread(bool3):
    pool = P.build_pool(bool3, min_sets=3, n=40, seed=2)
    spreads = {t["pool_id"]: t["size_spread"] for ts in pool.values() for t in ts}
    assert len(set(spreads.values())) >= 3                    # several spread bins represented


def test_anchor_path_matches_the_full_join(bool3):
    """Boolean path (candidates from the anchor) gives the same sizes as the full join for those keys."""
    import random
    full = P.join_sizes(bool3, min_sets=1)
    cand = P._anchor_candidates(bool3["BOOL-NAND-n3"], 300, random.Random(3))
    sub = P.join_sizes(bool3, min_sets=1, candidates=cand)
    assert sub and all(sub[k] == full[k] for k in sub)
    pool = P.build_pool(bool3, min_sets=3, n=10, seed=4, anchor="BOOL-NAND-n3", n_cand=300)
    assert pool and all(len({t["pool_id"] for t in ts}) <= 10 for ts in pool.values())


def test_pool_uses_exactly_the_sets_that_get_pool_runs():
    """Guardrail row 49: the pool's minimum-set count must count only sets that s2_cal_pool runs (no dropped set,
    no control or size-check set, no held-out set)."""
    from reach.runner.phases import DROPPED, FULL_SETS, REDUCED_SETS
    from reach.s2_lock import held_out
    for dom in ("real", "bool"):
        sets = P.pool_sets(dom)
        assert sets and set(sets) <= set(FULL_SETS)
        assert not set(sets) & (set(DROPPED) | set(REDUCED_SETS) | set(held_out()))
        assert all(G.get(g).kind == dom for g in sets)


def test_quotas_are_filled_after_the_filters():
    """Guardrail row 51: spread-0 functions always pass the filters, functions with a spread rarely do; the pool must
    still hold equal shares per spread bin while enough of them exist."""
    import random
    rng = random.Random(5)
    spread_of = {}
    bins = {b: [] for b in range(5)}
    for i in range(2000):
        b = i % 5
        spread_of[i] = (0, 2, 4, 8, 12)[b]
        bins[b].append(i)
    passes = {i: (spread_of[i] == 0 or rng.random() < 0.12) for i in spread_of}
    out = P._select(bins, lambda k: ({"S": {}}, spread_of[k]) if passes[k] else None, 40)
    from collections import Counter
    c = Counter(P._spread_bin(x[2]) for x in out)
    assert len(out) == 40 and all(v == 8 for v in c.values()), c
    few = {b: (v if b == 0 else v[:20]) for b, v in bins.items()}            # too few with a spread: fill from spare
    out2 = P._select(few, lambda k: ({"S": {}}, spread_of[k]) if passes[k] else None, 40)
    assert len(out2) == 40 and Counter(P._spread_bin(x[2]) for x in out2)[0] >= 8

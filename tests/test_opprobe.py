"""T13 operator probe (lesson 20; REQ-RULE-08), written before the code."""
import random

from reach import grammar as G
from reach.tasks import s2_opprobe as P


def test_probe_flags_a_no_op_operator():
    g = G.get("BOOL-NAND-n3")
    res = P.probe(g, {"noop": lambda t, rng: t.copy()}, n=200, seed=1)
    assert res["noop"]["changed_share"] == 0.0 and res["noop"]["flag"]


def test_real_operators_change_semantics():
    for name in ("BOOL-NAND-n3", "R-ARITH4"):
        g = G.get(name)
        res = P.probe(g, P.operators(g), n=400, seed=2)
        for op in ("subtree_mutation", "subtree_crossover", "headless_crossover"):
            assert res[op]["changed_share"] > P.FLAG_BELOW and not res[op]["flag"], (name, op, res[op])
        assert set(res["subtree_mutation"]) >= {"changed_share", "size_change_mean", "rejected_share", "events"}

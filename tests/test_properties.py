"""T11 grammar properties (written before the code)."""
import numpy as np

from reach import grammar as G
from reach.analysis.s2 import properties as P
from reach.enumerate import Enumerator


def test_key_construction_sizes_on_a_small_enumeration(tmp_path):
    g = G.get("BOOL-AON-n3")
    Enumerator(g, tmp_path, log=lambda *_: None).run(12)
    sizes = P.key_sizes(g, tmp_path)
    assert sizes["and2"] == 3 and sizes["or2"] == 3          # and(x1, x2), or(x1, x2)
    assert sizes["xor2"] is not None and sizes["xor2"] > 3   # xor needs a construction in AON
    g2 = G.get("BOOL-AONX-n3")
    Enumerator(g2, tmp_path / "x", log=lambda *_: None).run(8)
    assert P.key_sizes(g2, tmp_path / "x")["xor2"] == 3


def test_real_key_sizes(tmp_path):
    g = G.get("R-ARITH4")
    Enumerator(g, tmp_path, log=lambda *_: None).run(7)
    s = P.key_sizes(g, tmp_path)
    assert s["x_plus_1"] == 3 and s["x_squared"] == 3 and s["x_cubed"] == 5 and s["exp"] is None


def test_redundancy_is_trees_per_function():
    levels = [dict(K=1, U=2, U_cum=2, N_syntactic=2), dict(K=3, U=3, U_cum=5, N_syntactic=12)]
    assert P.redundancy(levels, 3) == 12 / 3

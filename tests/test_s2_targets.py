"""T4 targets v3 (written before the code). REQ-MEA-06, -07, -08, -09, -10, -16, -22."""
import hashlib
import json

import numpy as np
import pytest

from reach import grammar as G
from reach import s2_targets as T
from reach.enumerate import Enumerator

QUIET = dict(log=lambda *_: None)


@pytest.fixture(scope="module")
def an3(tmp_path_factory):
    d = tmp_path_factory.mktemp("an3")
    Enumerator(G.get("BOOL-AN-n3"), d, **QUIET).run(11)
    return d


def test_depth_aware_draw_and_excluded_share(an3, monkeypatch):
    """REQ-MEA-16: only functions with a minimal tree within the depth limit are drawn; the excluded share is logged."""
    monkeypatch.setattr(T, "DEPTH_LIMIT", 5)
    g = G.get("BOOL-AN-n3")
    ts, log = T.generate(g, an3, [9, 11], 50, seed=1)
    assert ts and all(t["depth"] <= 5 for t in ts)
    for K in (9, 11):
        lv = log["levels"][str(K)]
        assert lv["n_level"] > lv["n_depth_ok"] > 0                      # AN at n = 3 has deep minimal trees here
        assert lv["excluded_share"] == pytest.approx(1 - lv["n_depth_ok"] / lv["n_level"])


def test_proven_targets_have_their_minimum_size_and_labels(an3):
    g = G.get("BOOL-AN-n3")
    e = Enumerator(g, an3, **QUIET)
    e.load()
    ts, _ = T.generate(g, an3, [7, 9], 10, seed=2)
    for t in ts:
        tree = G.parse_expr(t["expression"], g)
        assert G.count_nodes(tree) == t["K_construction"] == t["K_proven"]
        assert e.min_size(G.evaluate(g, tree)) == t["K_proven"]
        assert {"nodes", "bits_uniform", "bits_syn", "bits_sem"} <= set(t["units"])


def test_constructed_targets_above_the_ceiling(an3):
    g = G.get("BOOL-AN-n3")
    e = Enumerator(g, an3, **QUIET)
    e.load()
    ceiling = max(k for k in e.keys if len(e.keys[k]))
    ts, log = T.generate(g, an3, [ceiling + 2], 5, seed=3)
    for t in ts:
        assert t["K_proven"] is None and t["K_construction"] == ceiling + 2 and t["constructed"]
        assert e.min_size(G.evaluate(g, G.parse_expr(t["expression"], g))) is None


def test_boolean_properties_known_values():
    g = G.get("BOOL-AON-n3")
    vm = G.var_masks(3)
    x1 = T.boolean_properties(vm["x1"], 3)
    assert (x1["essential_inputs"], x1["degree"], x1["nonlinearity"]) == (1, 1, 0)
    a = T.boolean_properties(vm["x1"] & vm["x2"], 3)
    assert (a["essential_inputs"], a["degree"], a["nonlinearity"]) == (2, 2, 2)
    maj = (vm["x1"] & vm["x2"]) | (vm["x1"] & vm["x3"]) | (vm["x2"] & vm["x3"])
    m = T.boolean_properties(maj, 3)
    assert (m["essential_inputs"], m["degree"], m["nonlinearity"]) == (3, 2, 2)
    par = vm["x1"] ^ vm["x2"] ^ vm["x3"]
    assert T.boolean_properties(par, 3)["degree"] == 1


# ------------------------------------------------------------------ real filters: positive and negative controls
def _check(name, expr, vals=None):
    g = G.get(name)
    return T.real_filters(g, G.parse_expr(expr, g), K=G.count_nodes(G.parse_expr(expr, g)), vals=vals)


def test_complex_valued_target_is_rejected():
    """REQ-MEA-22: log(1 - x) is complex on (2, 3); a real function passes."""
    assert _check("R-ELEM6", "log(-(1, x))")[1] == "complex"
    assert _check("R-ELEM6", "log(+(x, 1))")[1] == ""


def test_floating_point_dependent_target_is_rejected():
    """(s + 1) - s with s = (exp(exp x))^4 ~ 1e35: double gives 0, 50 digits give 1."""
    big = "square(square(exp(exp(x))))"
    assert _check("R-SRB", f"-(+({big}, 1), {big})")[1] in ("fp_dependent", "variance", "magnitude_ratio")  # 0 in double
    assert _check("R-SRB", "+(x, 1)")[1] == ""
    # a target that only the 50-digit screen catches: finite, varying, but wrong in double
    expr = f"+(x, -(+({big}, 1), {big}))"            # x + 1 exactly; double gives x
    assert _check("R-SRB", expr)[1] == "fp_dependent"


def test_near_duplicate_is_rejected():
    g = G.get("R-ARITH")
    tree = G.parse_expr("*(x, +(x, 1))", g)
    f = G.evaluate(g, tree, G.grid_sample())
    near = {3: (f * (1 + 1e-6))[None, :]}                             # a smaller "function" within nMSE 1e-6
    far = {3: (f + 1.0)[None, :]}
    assert T.real_filters(g, tree, K=5, vals=near)[1] == "near_duplicate"
    assert T.real_filters(g, tree, K=5, vals=far)[1] == ""


def test_dead_subtree_is_rejected():
    assert _check("R-ELEM6", "+(x, *(-(x, x), exp(x)))")[1] == "dead_subtree"     # exp(x) is dead (paper 1 control)
    assert _check("R-ELEM6", "+(x, *(x, exp(x)))")[1] == ""


# ------------------------------------------------------------------ freezing
def test_files_frozen_with_checksum_and_reproducible(an3, tmp_path):
    g = G.get("BOOL-AN-n3")
    p1 = T.write_frozen(tmp_path / "a", "targets", *T.generate(g, an3, [7, 9], 5, seed=4))
    p2 = T.write_frozen(tmp_path / "b", "targets", *T.generate(g, an3, [7, 9], 5, seed=4))
    assert p1.read_bytes() == p2.read_bytes()
    sha = (p1.parent / "targets.sha256").read_text().split()[0]
    assert sha == hashlib.sha256(p1.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):                              # frozen: never overwritten
        T.write_frozen(tmp_path / "a", "targets", *T.generate(g, an3, [7], 5, seed=5))


def test_pilot_levels():
    """Up to 12 evenly spaced non-empty proven sizes from 3 to the ceiling, plus ceiling + 2, + 4, + 8."""
    levels = [dict(K=k, U=(1 if k % 2 else 0)) for k in range(1, 33)]          # NAND-like: odd sizes only
    proven, constructed = T.pilot_levels(levels)
    assert proven[0] == 3 and proven[-1] == 31 and len(proven) <= 12 and all(k % 2 for k in proven)
    assert constructed == [33, 35, 39]


def test_pilot_target_phases_cover_calibration_sets_only():
    from reach import s2_sets as S
    from reach.runner.phases import PHASES
    from reach.s2_lock import held_out
    names = set()
    for ph in ("s2_targets_pilot_a", "s2_targets_pilot_b"):
        for t in PHASES[ph]["build"]():
            assert "--seed" in t["argv"] and t["meta"]["out"].startswith("data/targets_s2/")
            names.add(t["meta"]["grammar"])
    from reach.runner.phases import DROPPED
    assert names == {g for g in S.SETS} - held_out() - set(DROPPED)

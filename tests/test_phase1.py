"""Phase 1 (docs/plan/phases/phase1_enumeration.md): guarded real sets, IMP0, enumerator v2.

Written before the code (D049, test first). Requirements: REQ-ENG-05, REQ-MEA-01, -05, -06, -17 to -20, REQ-RUN-16.
"""
import cmath
import json
import math
import random
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

from reach import grammar as G
from reach import enumerate as EN
from reach.enumerate import Enumerator, read_level
from reach.sample import make_sampler
from test_grammar_enumerate import _brute_min_sizes

ROOT = Path(__file__).resolve().parents[1]
S2_REAL = ["R-ARITH", "R-ARITH4", "R-ELEM6", "R-ELEM4", "R-HYB", "R-TRIG", "R-SRB", "F-EML-R"]
XS = G.grid_sample()
TINY = np.finfo(float).tiny
NAN = complex(math.nan, math.nan)
QUIET = dict(log=lambda *_: None)


# ------------------------------------------------------------------ an independent scalar reference (guarded policy)
class Undetermined(Exception):
    """An infinite intermediate (other than the exact -inf of ln 0): IEEE corner cases differ between numpy and
    cmath, so such points are not compared (as in tests/test_family_ops.py)."""


class BranchCut(Exception):
    """A log argument exactly on the negative real axis (sign-of-zero convention; excluded as in paper 1)."""


def _fin(z):
    return cmath.isfinite(z)


def _check_input(z):
    if _fin(z) or (z.real == -math.inf and z.imag == 0):
        return z
    if math.isinf(z.real) or math.isinf(z.imag):
        raise Undetermined
    return NAN                                       # NaN propagates as NaN in both evaluators


def _guard(r, *inputs):
    return NAN if all(_fin(i) for i in inputs) and not _fin(r) else r


def _exp(a):
    if math.isnan(a.real) or math.isnan(a.imag):
        return NAN
    if a.real == -math.inf and a.imag == 0:
        return 0j
    if a.real > G.EXP_HI or a.real < G.EXP_LO:
        return NAN                                   # lossy overflow or underflow
    return cmath.exp(a)


def _log(a):
    if math.isnan(a.real) or math.isnan(a.imag):
        return NAN
    if a == 0:
        return complex(-math.inf, 0)
    if a.imag == 0 and a.real < 0:
        raise BranchCut
    if not _fin(a):
        raise Undetermined
    return cmath.log(a)


def _mul(a, b):
    try:
        r = a * b
    except OverflowError:
        return NAN
    if all(map(_fin, (a, b))) and a != 0 and b != 0 and abs(r) < TINY:
        return NAN
    return _guard(r, a, b)


def _div(a, b):
    if b == 0:
        return complex(math.inf, math.nan)          # non-finite; its exact value is not compared
    try:
        r = a / b
    except (OverflowError, ZeroDivisionError):
        return NAN
    if all(map(_fin, (a, b))) and a != 0 and abs(r) < TINY:
        return NAN
    return _guard(r, a, b)


def _sqrt(a):
    if not _fin(a):
        return NAN
    if a.imag == 0 and a.real < 0:
        raise BranchCut
    return cmath.sqrt(a)


def _trig(f, a):
    if not _fin(a):
        return NAN
    try:
        return _guard(f(a), a)
    except OverflowError:
        return NAN


REF = {
    "+": lambda a, b: _guard(a + b, a, b),
    "-": lambda a, b: _guard(a - b, a, b),
    "*": _mul, "/": _div,
    "exp": _exp, "log": _log,
    "sin": lambda a: _trig(cmath.sin, a), "cos": lambda a: _trig(cmath.cos, a),
    "eml": lambda a, b: (lambda e, l: _guard(e - l, e, l))(_exp(a), _log(b)),
    "sqrt": _sqrt, "square": lambda a: _mul(a, a),
}
REF["reml"] = lambda a, b: REF["eml"](b, a)


def _scalar(tree, x):
    if not tree.is_operator:
        return complex(x) if isinstance(tree.value, str) else complex(tree.value)
    a = _check_input(_scalar(tree.left, x))
    if tree.right is None:
        return REF[tree.value](a)
    return REF[tree.value](a, _check_input(_scalar(tree.right, x)))


@pytest.mark.parametrize("name", S2_REAL)
def test_guarded_sets_match_scalar_reference(name):
    """Values agree where the reference is finite; the classification (finite or not) agrees everywhere the
    reference is determined, so a guard that should give NaN cannot silently give a number, and vice versa."""
    g, rng = G.get(name), random.Random(11)
    sample = make_sampler(g)
    xs = XS[::10]
    finite = nonfinite = 0
    for _ in range(400):
        tree = sample(rng.choice([3, 5, 7, 9, 11]), 10, rng)
        v = G.evaluate(g, tree, xs)
        for x, vi in zip(xs, v):
            try:
                r = _scalar(tree, x)
            except (Undetermined, BranchCut):
                continue
            assert bool(np.isfinite(vi)) == _fin(r), (G.expr_str(tree), x, vi, r)
            if _fin(r):
                assert abs(vi - r) <= 1e-9 * max(1.0, abs(r)), (G.expr_str(tree), x, vi, r)
                finite += 1
            else:
                nonfinite += 1
    assert finite > 500


@pytest.mark.parametrize("name,expr,f", [
    ("R-ARITH4", "/(x, x)", lambda x: np.ones_like(x)),
    ("R-TRIG", "+(*(sin(x), sin(x)), *(cos(x), cos(x)))", lambda x: np.ones_like(x)),
    ("R-ELEM4", "exp(log(x))", lambda x: x),
    ("R-HYB", "eml(x, 1)", np.exp),
    ("R-ELEM6", "-(exp(x), log(x))", lambda x: np.exp(x) - np.log(x)),
])
def test_guarded_identities(name, expr, f):
    g = G.get(name)
    assert np.allclose(G.evaluate(g, G.parse_expr(expr, g), XS), f(XS), rtol=1e-12)


@pytest.mark.parametrize("name,expr", [
    ("R-ELEM4", "exp(exp(exp(x)))"),                 # overflow: NaN, not +inf (D021 policy)
    ("R-ELEM6", "exp(-(1, exp(exp(x))))"),           # underflow: NaN, not 0
])
def test_guards_give_nan(name, expr):
    g = G.get(name)
    v = G.evaluate(g, G.parse_expr(expr, g), XS)
    assert np.isnan(v).all()
    ieee = {"R-ELEM4": "ELEM6", "R-ELEM6": "ELEM6"}[name]   # negative control: plain IEEE gives inf or 0
    gi = G.get(ieee)
    vi = G.evaluate(gi, G.parse_expr(expr, gi), XS)
    assert not np.isnan(vi).all()


def test_division_by_zero_is_not_finite():
    g = G.get("R-ARITH4")
    assert not np.isfinite(G.evaluate(g, G.parse_expr("/(x, -(x, x))", g), XS)).any()


# ------------------------------------------------------------------ minimum sizes and depths by brute force
def _all_trees(g, kmax):
    leaves = [G.leaf(v) for v in list(g.variables) + list(g.constants)]
    trees = {1: leaves}
    for k in range(2, kmax + 1):
        trees[k] = [G.Node(o.name, t, None, True) for o in g.ops if o.arity == 1 for t in trees[k - 1]]
        trees[k] += [G.Node(o.name, l, r, True) for o in g.ops if o.arity == 2
                     for i in range(1, k - 1) for l in trees[i] for r in trees[k - 1 - i]]
    return trees


def _wellcond(v):
    """Real functions compared by brute force: finite and at most 1e6 in magnitude on the grid. Above that, 9 absolute
    decimals are finer than double precision, and an ill-conditioned tree such as exp(exp(exp(log x))) differs from
    exp(exp x) by float noise amplified to about 4e-6 (REQ-MEA-21). Keys at 9 decimals are also not compositional
    near cancellations: complex x/x differs from 1 by one ulp at some points, so the literal tree
    1/(1 - x/x) gives values near 1e16 or inf, while the enumerator (correctly) treats x/x as the constant 1
    (diagnosed 2026-10-19; guardrail register row 35). Such artefacts are excluded from targets by the 50-digit
    screen (REQ-MEA-07, REQ-MEA-14)."""
    return bool(np.isfinite(v).all() and np.abs(v).max() <= 1e6)


def _on_branch_cut(g, tree, xs):
    """True if some log or sqrt in the tree gets a value exactly on the negative real axis: the result's sign then
    depends on the sign of a zero imaginary part, i.e. on the computation path (paper 1's convention, D021;
    guardrail row 40). Such trees are left out of the brute-force comparison, as from the scalar reference."""
    if not tree.is_operator:
        return False
    if tree.value in ("log", "sqrt"):
        v = G.evaluate(g, tree.left, xs)
        if np.any((v.imag == 0) & (v.real < 0)):
            return True
    return any(_on_branch_cut(g, c, xs) for c in (tree.left, tree.right) if c is not None)


def _brute(g, kmax, cut=None):
    """{key: (minimum size, smallest depth among the minimum-size trees)} by explicit tree enumeration. Real sets:
    trees on a log/sqrt branch cut are skipped and their keys collected in `cut`."""
    coeffs = EN._coeffs(2 * G.N_POINTS)
    best, skip = {}, set()
    for k, ts in _all_trees(g, kmax).items():
        for t in ts:
            if g.kind == "real":
                v = G.evaluate(g, t, G.sample_grid(g))
                key = int(EN.real_keys(v[None, :], coeffs)[0])
                if not _wellcond(v):
                    skip.add(key)
                    continue
                if _on_branch_cut(g, t, G.sample_grid(g)):
                    if cut is not None:
                        cut.add(key)
                    continue
            else:
                key = int(G.evaluate(g, t))
            d = G.tree_depth(t)
            if key not in best:
                best[key] = (k, d)
            elif best[key][0] == k and d < best[key][1]:
                best[key] = (k, d)
    return best


def _ours(e):
    out = {}
    for k in e.keys:
        for idx, (key, d) in enumerate(zip(e.keys[k], e.depth[k])):
            if e.real and not _wellcond(e.vals[k][idx]):
                continue
            out[int(key)] = (k, int(d))
    return out


@pytest.mark.parametrize("name,kmax", [("R-ARITH", 7), ("R-ARITH4", 7), ("R-ELEM6", 5), ("R-ELEM4", 6),
                                       ("R-HYB", 7), ("R-TRIG", 6), ("R-SRB", 5), ("F-EML-R", 7)])
def test_guarded_real_sets_match_brute_force(name, kmax, tmp_path):
    g = G.get(name)
    e = Enumerator(g, tmp_path, **QUIET)
    e.run(kmax)
    cut = set()
    ours, brute = _ours(e), _brute(g, kmax, cut)
    assert {k: v for k, v in ours.items() if k in brute} == brute          # every regular function, same size, depth
    assert set(ours) - set(brute) <= cut                                   # the rest only via branch-cut trees


@pytest.mark.parametrize("name", ["BOOL-IMP0-n3", "BOOL-IMP0R-n3", "BOOL-NAND-n3", "BOOL-AN-n3", "BOOL-NAND-n6"])
def test_new_boolean_sets_match_brute_force(name, tmp_path):
    g = G.get(name)
    kmax = 7
    e = Enumerator(g, tmp_path, **QUIET)
    e.run(kmax)
    assert _ours(e) == _brute(g, kmax)
    assert {int(k_): k for k_, (k, _) in _ours(e).items()} == _brute_min_sizes(g, kmax)


def test_min_depth_matches_brute_force(tmp_path):
    """REQ-MEA-06 with a positive control: AN at n = 3 has functions whose first-found minimal tree is deeper
    than their shallowest minimal tree, so a first-found representative would fail this test."""
    g = G.get("BOOL-AN-n3")
    e = Enumerator(g, tmp_path, **QUIET)
    e.run(9)
    brute = _brute(g, 9)
    assert _ours(e) == brute
    first_found = {}
    for k, ts in _all_trees(g, 9).items():
        for t in ts:
            first_found.setdefault(G.evaluate(g, t), (k, G.tree_depth(t)))
    assert any(first_found[key] != brute[key] for key in brute)


@pytest.mark.parametrize("base,control,kmax", [("F-EML", "F-EML-R", 11), ("BOOL-IMP0-n3", "BOOL-IMP0R-n3", 9)])
def test_redundancy_controls_have_identical_functions(base, control, kmax, tmp_path):
    """D052: a redundancy control adds an operator that is the mirror of an existing one (b op a), so every function
    has the same minimum size and U is identical, while the syntax (and GP's operator mix) differs."""
    e1 = Enumerator(G.get(base), tmp_path / "a", **QUIET)
    e2 = Enumerator(G.get(control), tmp_path / "b", **QUIET)
    e1.run(kmax)
    e2.run(kmax)
    for k in e1.keys:
        assert np.array_equal(np.sort(e1.keys[k]), np.sort(e2.keys[k])), k
    assert G.syntactic_counts(G.get(control), kmax)[kmax] > G.syntactic_counts(G.get(base), kmax)[kmax]


def test_galib_knows_every_unary_operator():
    """ga-lib decides arity by name (UNARY_OPS); an unknown unary operator would silently get two children."""
    from reach.galib.generator import UNARY_OPS
    for g in G.GRAMMARS.values():
        for o in g.ops:
            assert (o.arity == 1) == (o.name in UNARY_OPS), (g.name, o.name)


def test_constant0_leaf(tmp_path):
    """Review F14 (fixed): a Boolean constant 0 is the all-zeros table."""
    g = G.Grammar("BOOL-TEST0-n3", "bool", (G.BOOL_OPS["nand"],), ("x1", "x2", "x3"), (0,), 3)
    vals, _ = Enumerator(g, tmp_path, **QUIET)._leaves()
    assert int(vals[-1]) == 0 == G.evaluate(g, G.leaf(0))


def test_imp0_is_complete():
    g = G.get("BOOL-IMP0-n3")
    m = g.mask
    vm = G.var_masks(3)
    assert G.evaluate(g, G.parse_expr("imp(x1, 0)", g)) == m ^ vm["x1"]                    # not
    assert G.evaluate(g, G.parse_expr("imp(imp(x1, 0), x2)", g)) == vm["x1"] | vm["x2"]     # or


# ------------------------------------------------------------------ storage, counts, budgets
def test_level_files_compact_and_v1_readable(tmp_path):
    g = G.get("BOOL-AON-n3")
    e = Enumerator(g, tmp_path / "v2", **QUIET)
    e.run(6)
    for k in range(1, 7):
        keys, par, depth = read_level(tmp_path / "v2" / f"level_{k:03d}.npz")
        assert np.array_equal(keys, e.keys[k]) and np.array_equal(par, e.parents[k])
        assert np.array_equal(depth, e.depth[k])
        assert keys.dtype == np.uint64 and par.dtype == np.int32
    z = np.load(tmp_path / "v2" / "level_006.npz")
    assert "parents" not in z and z["keys"].dtype == np.uint32       # compact on disk (n = 3: 8-bit tables)
    v1 = tmp_path / "v1.npz"                                         # paper-1 format: int64 parents, no depth
    np.savez(v1, keys=e.keys[6], parents=e.parents[6].astype(np.int64))
    keys, par, depth = read_level(v1)
    assert np.array_equal(par, e.parents[6]) and depth is None


# Real-valued counts depend on the platform's maths library: on Linux (CI) IEEE ELEM6 has 2,516 functions at K = 7
# against 2,515 on the study machine, where the archived counts were made (guardrail row 38). Boolean counts are exact.
_REAL_V1 = [pytest.param(n, k, marks=pytest.mark.study_machine) for n, k in (("EML", 13), ("ARITH", 13), ("ELEM6", 7))]


@pytest.mark.parametrize("name,kmax", _REAL_V1 +
                         [(f"BOOL-{b}-n3", 40) for b in ("NAND", "NOR", "AON", "AN", "AONX", "AX1", "RICH")])
def test_counts_identical_to_v1(name, kmax, tmp_path):
    """REQ-MEA-19: the archived enumeration counts (v1) are reproduced level by level (real sets: study machine)."""
    v1 = {st["K"]: st for st in json.loads((ROOT / f"data/enumeration/{name}/counts.json").read_text())["levels"]}
    e = Enumerator(G.get(name), tmp_path, **QUIET)
    e.run(min(kmax, max(v1)))
    for st in e.stats:
        assert (st["U"], st["candidates"]) == (v1[st["K"]]["U"], v1[st["K"]]["candidates"]), st["K"]


def test_enumeration_stops_cleanly_on_cpu_budget(tmp_path):
    g = G.get("BOOL-AON-n4")
    e = Enumerator(g, tmp_path, **QUIET)
    import time
    k, state = e.run(40, cpu_budget=time.process_time() + 0.5)
    assert state == "budget_exhausted" and e.stop_reason == "cpu"
    assert k < 35                                              # n = 4 completes at 35 without the budget
    e2 = Enumerator(g, tmp_path, **QUIET)                      # the checkpoint is intact and resumes
    assert e2.load() == k


def test_enumeration_stops_cleanly_on_low_disk(tmp_path, monkeypatch):
    """REQ-RUN-16: with too little free disk the enumeration stops before writing a level."""
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(EN.shutil, "disk_usage", lambda p: Usage(100 * 2 ** 30, 90 * 2 ** 30, 10 * 2 ** 30))
    e = Enumerator(G.get("BOOL-AON-n3"), tmp_path, **QUIET)
    k, state = e.run(10, min_free_gb=15)
    assert state == "budget_exhausted" and e.stop_reason == "disk" and k == 1


def test_enumeration_stops_cleanly_on_memory_cap(tmp_path):
    e = Enumerator(G.get("BOOL-AON-n4"), tmp_path, **QUIET)
    k, state = e.run(40, max_gb=0.001)
    assert state == "budget_exhausted" and e.stop_reason == "memory" and k < 35


def test_s2_enumeration_phases():
    """REQ-MEA-20: Study 2 enumerations write under data/enumeration_s2/, never into the archived pilot's."""
    from reach.runner.phases import PHASES
    for name in ("s2_enum_real", "s2_enum_ieee", "s2_enum_bool", "s2_precision", "s2_enum_extra", "s2_precision_extra"):
        tasks = PHASES[name]["build"]()
        assert tasks
        for t in tasks:
            assert "data/enumeration_s2" in " ".join(t["argv"]), t["key"]


# ------------------------------------------------------------------ the set list (D052, D053)
def test_s2_set_list_is_consistent():
    """One source of truth for the function sets: every set exists, roles are known, the draw pools match D053,
    and every set has an enumeration phase."""
    from reach import s2_sets as S
    from reach.runner.phases import PHASES
    assert set(S.ROLES) <= {"anchor", "control", "size_check", "eligible"}
    for name, s in S.SETS.items():
        assert name in G.GRAMMARS and s["role"] in S.ROLES and s["domain"] == G.get(name).kind
    assert len(S.eligible("real")) == 8 and len(S.eligible("bool")) == 5
    enumerated = {t["meta"]["grammar"] for p in ("s2_enum_real", "s2_enum_bool", "s2_enum_extra")
                  for t in PHASES[p]["build"]()}
    assert set(S.SETS) <= enumerated
    for t in PHASES["s2_enum_ceiling"]["build"]():
        assert S.SETS[t["meta"]["grammar"]]["root"] == t["meta"]["root"]


def test_aon_n6_matches_brute_force(tmp_path):
    g = G.get("BOOL-AON-n6")
    e = Enumerator(g, tmp_path, **QUIET)
    e.run(5)
    assert {int(k): v[0] for k, v in _ours(e).items()} == _brute_min_sizes(g, 5)

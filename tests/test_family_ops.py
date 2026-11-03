"""EML-family operators: identities from Odrzywolek's papers (positive controls) and agreement of the
vectorised evaluator with an independent scalar cmath reference on random trees (docs/07 #9)."""
import cmath
import math
import random

import numpy as np
import pytest

from reach import grammar as G
from reach.sample import make_sampler

FAMILY = ["F-EML", "F-EDL", "F-NEML", "F-LDE0", "F-LDE1", "F-PLI", "F-PLM"]
XS = G.grid_sample()


def ev(name, s):
    g = G.get(name)
    return G.evaluate(g, G.parse_expr(s, g), XS)


@pytest.mark.parametrize("name,expr,f", [
    ("F-EML", "eml(x, 1)", np.exp),
    ("F-EDL", f"edl(x, {math.e!r})", np.exp),                  # e^x / ln e
    ("F-NEML", "neml(x, -inf)", np.log),                     # ln x - exp(-inf)
    ("F-LDE0", "lde(x, 0)", np.log),                         # ln x / e^0
    ("F-PLI", "pli(x, 1)", np.log),                          # (ln x)^1
    ("F-PLM", "plm(x, 1)", lambda x: 1 / np.log(x)),         # (ln x)^-1
])
def test_identities(name, expr, f):
    assert np.allclose(ev(name, expr), f(XS), rtol=1e-12)


class BranchCut(Exception):
    """A log argument lies exactly on the negative real axis: the result then depends on the sign of a
    zero imaginary part, which numpy and cmath propagate differently. All grammars share numpy's
    convention (as EML-GP did), so such evaluations are excluded from the agreement check."""


def _ref(op, a, b):
    """Scalar reference with the same conventions as numpy: log(0) = -inf; errors -> non-finite."""
    def log(z):
        if z.imag == 0 and z.real < 0:
            raise BranchCut
        return complex(-math.inf, 0) if z == 0 else cmath.log(z)
    try:
        if op == "eml":
            return cmath.exp(a) - log(b)
        if op == "edl":
            return cmath.exp(a) / log(b)
        if op == "neml":
            return log(a) - cmath.exp(b)
        if op == "lde":
            return log(a) / cmath.exp(b)
        if op == "pli":
            return cmath.exp(log(log(a)) / b) if log(a) != 0 else complex(math.nan, math.nan)
        if op == "plm":
            return cmath.exp(-b * log(log(a))) if log(a) != 0 else complex(math.nan, math.nan)
    except (OverflowError, ValueError, ZeroDivisionError):
        return complex(math.nan, math.nan)


def _scalar(g, tree, x):
    if not tree.is_operator:
        return complex(x) if isinstance(tree.value, str) else complex(tree.value)
    a, b = _scalar(g, tree.left, x), _scalar(g, tree.right, x)
    if not (cmath.isfinite(a) or (a.real == -math.inf and a.imag == 0)) or \
       not (cmath.isfinite(b) or (b.real == -math.inf and b.imag == 0)):
        return complex(math.nan, math.nan)        # non-finite inputs: only classification is compared
    return _ref(tree.value, a, b)


@pytest.mark.parametrize("name", FAMILY)
def test_vectorised_matches_scalar_reference(name):
    g, rng = G.get(name), random.Random(7)
    sample = make_sampler(g)
    xs = XS[::10]
    checked = 0
    for _ in range(400):
        tree = sample(rng.choice([3, 5, 7, 9]), 10, rng)
        v = G.evaluate(g, tree, xs)
        for x, vi in zip(xs, v):
            try:
                r = _scalar(g, tree, x)
            except BranchCut:
                continue
            if np.isfinite(vi) and cmath.isfinite(r):
                assert abs(vi - r) <= 1e-9 * max(1.0, abs(r)), (G.expr_str(tree), x, vi, r)
                checked += 1
    assert checked > 500


def test_exact_hash_ignores_signed_zero():
    """Review 2026-10-15: -0.0 and +0.0 imaginary parts must hash alike (equal functions were scored 'not exact')."""
    import numpy as np
    from reach import grammar as G
    a = np.array([1 + 0j, 2.5 + 0j])
    b = np.array([complex(1, -0.0), complex(2.5, -0.0)])
    assert G.scaled_func_hash(a, 1.0) == G.scaled_func_hash(b, 1.0)
    assert G.scaled_func_hash(a, 1.0) != G.scaled_func_hash(a * (1 + 1e-6), 1.0)

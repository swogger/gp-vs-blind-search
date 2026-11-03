"""Grammar specs, trees and evaluation (docs/04 "Grammars"; decisions D008, D011).

Real grammars: one variable x, constant 1, complex128 semantics on the domain (2, 3).
Non-finite values propagate (numpy, errors ignored); division is IEEE (D011).
Boolean grammars: inputs x1..xn; a value is a whole truth table packed into an int
(bit r = output on row r), so one evaluation covers every row. Constant 1 = all ones.

Trees are galib Nodes (unary operators use the left child only). Expression strings are
prefix calls, e.g. `eml(x, 1)`, `+(x, exp(x))`, `nand(x1, x2)`, matching EML-GP's format.
"""
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np

from reach.galib.node import Node

# ---------------------------------------------------------------- inherited constants
MAX_NODES = 2_000
DOMAIN_LO, DOMAIN_HI = 2.0, 3.0
N_POINTS = 100
HASH_PRECISION = 9


@dataclass(frozen=True)
class Op:
    name: str            # the name galib uses
    arity: int           # 1 or 2
    fn: Callable         # real: fn(a[, b]) on complex arrays; bool: fn(mask, a[, b]) on ints


@dataclass(frozen=True)
class Grammar:
    name: str
    kind: str            # "real" or "bool"
    ops: tuple
    variables: tuple
    constants: tuple = ()
    n_inputs: int = 1

    @property
    def operators(self):
        return [o.name for o in self.ops]

    @property
    def op(self):
        return {o.name: o for o in self.ops}

    def arity_counts(self):
        """(a0, a1, a2): number of terminals, unary and binary symbols."""
        return (len(self.variables) + len(self.constants),
                sum(o.arity == 1 for o in self.ops), sum(o.arity == 2 for o in self.ops))

    @property
    def mask(self):
        return (1 << (1 << self.n_inputs)) - 1


# ---------------------------------------------------------------- real operators
def _eml(a, b):
    return np.exp(a) - np.log(b)


REAL_OPS = {
    "eml": Op("eml", 2, _eml),
    "+": Op("+", 2, lambda a, b: a + b),
    "-": Op("-", 2, lambda a, b: a - b),
    "*": Op("*", 2, lambda a, b: a * b),
    "/": Op("/", 2, lambda a, b: a / b),
    "exp": Op("exp", 1, np.exp),
    "log": Op("log", 1, np.log),
}


# ---------------------------------------------------------------- faithful semantics (EML family, D021)
# Exact extended-real values are allowed (ln 0 = -inf, exp(-inf) = 0, exp(+inf) = +inf), but an operation
# that turns finite inputs into an overflow or an underflow loses information and yields NaN. Plain IEEE
# arithmetic lets a tree reach a shorter "identity" through exp(huge) = inf ... exp(-inf) = 0 (found for
# -x and 1/x in EML at 13 nodes, against 15 for the exact constructions).
EXP_HI = float(np.log(np.finfo(float).max))     # 709.78
EXP_LO = float(np.log(np.finfo(float).tiny))    # -708.40: below this exp() is subnormal or 0


def _guard(r, *inputs):
    fin = np.ones(np.shape(r), bool)
    for x in inputs:
        fin &= np.isfinite(x)
    return np.where(fin & ~np.isfinite(r), np.nan, r)


def fexp(a):
    r = np.exp(a)
    lossy = np.isfinite(a) & ((np.real(a) > EXP_HI) | (np.real(a) < EXP_LO))
    return np.where(lossy, np.nan, r)


def fdiv(a, b):
    r = a / b
    fin = np.isfinite(a) & np.isfinite(b)
    under = fin & (a != 0) & (np.abs(r) < np.finfo(float).tiny)
    return np.where((fin & (b != 0) & ~np.isfinite(r)) | under, np.nan, r)


FAMILY_OPS = {   # Odrzywolek, arXiv 2603.21852 eq. 4; arXiv 2609.11210 Table 1
    "eml": Op("eml", 2, lambda a, b: _guard(fexp(a) - np.log(b), fexp(a), np.log(b))),  # e^x - ln y; 1
    "edl": Op("edl", 2, lambda a, b: fdiv(fexp(a), np.log(b))),                         # e^x / ln y; e
    "neml": Op("neml", 2, lambda a, b: _guard(np.log(a) - fexp(b), np.log(a), fexp(b))),  # ln x - e^y; -inf
    "lde": Op("lde", 2, lambda a, b: fdiv(np.log(a), fexp(b))),                         # ln x / e^y; 0 or 1
    "pli": Op("pli", 2, lambda a, b: fexp(fdiv(np.log(np.log(a)), b))),                 # (ln x)^(1/y); 1
    "plm": Op("plm", 2, lambda a, b: fexp(-b * np.log(np.log(a)))),                     # (ln x)^(-y); 1
}


# ---------------------------------------------------------------- held-out candidates (D027)
def fmul(a, b):
    r = a * b
    fin = np.isfinite(a) & np.isfinite(b)
    under = fin & (a != 0) & (b != 0) & (np.abs(r) < np.finfo(float).tiny)
    return np.where((fin & ~np.isfinite(r)) | under, np.nan, r)


# f(x, y) = A(x) o B(y): A, B in {x, e^x, ln x}, o in {+, -, *, /, ^}; named e.g. "esubx" = e^x - y
_UN = {"x": lambda a: a, "e": fexp, "l": np.log}
_BIN = {"add": lambda a, b: _guard(a + b, a, b), "sub": lambda a, b: _guard(a - b, a, b), "mul": fmul,
        "div": fdiv, "pow": lambda a, b: fexp(fmul(b, np.log(a)))}
CAND_CONSTS = {"1": 1.0, "0": 0.0, "m1": -1.0, "e": float(np.e)}
CONST_NAMES = dict(CAND_CONSTS, minf=float("-inf"))   # minf only for positive controls (NEML)
KNOWN_FORMS = {"esubl", "edivl", "lsube", "ldive"}     # eml, edl, neml, lde: already in the family


def cand_forms():
    return [f"{A}{o}{B}" for o in _BIN for A in _UN for B in _UN
            if not (A == B == "x" or (o in ("add", "mul") and A > B) or f"{A}{o}{B}" in KNOWN_FORMS)]


EXTRA_FORMS = {   # literature operators outside the A(x) o B(y) space (D027 fallback pool)
    "cosacos": lambda a, b: _guard(np.cos(a) - np.arccos(b), np.cos(a), np.arccos(b)),   # Stachowiak: cos x - arccos y
}


def _cand(name):
    """'S-<form>-<const>' (leaves x, c) or 'S2-<form>-<const>' (leaves x, y, c)."""
    kind, form, c = name.split("-")
    if form in EXTRA_FORMS:
        op = Op(form, 2, EXTRA_FORMS[form])
    else:
        A, o, B = form[0], form[1:-1], form[-1]
        op = Op(form, 2, lambda a, b: _BIN[o](_UN[A](a), _UN[B](b)))
    return Grammar(name, "real", (op,), ("x",) if kind == "S" else ("x", "y"), (CONST_NAMES[c],))


# ---------------------------------------------------------------- Study 2 real sets: guarded semantics (D048 C4)
# One policy for every real operator, as for the EML family (D021): exact extended reals are kept (ln 0 = -inf,
# exp(-inf) = 0, x/0 non-finite), but finite inputs that overflow or underflow give NaN. Tests: tests/test_phase1.py.
def _gtrig(f):
    return lambda a: _guard(f(a), a)


GUARDED_OPS = {
    "+": Op("+", 2, lambda a, b: _guard(a + b, a, b)),
    "-": Op("-", 2, lambda a, b: _guard(a - b, a, b)),
    "*": Op("*", 2, fmul),
    "/": Op("/", 2, fdiv),
    "exp": Op("exp", 1, fexp),
    "log": Op("log", 1, np.log),
    "sin": Op("sin", 1, _gtrig(np.sin)),
    "cos": Op("cos", 1, _gtrig(np.cos)),
    "eml": FAMILY_OPS["eml"],
    "reml": Op("reml", 2, lambda a, b: FAMILY_OPS["eml"].fn(b, a)),   # mirrored eml: redundancy control (D052)
    "sqrt": Op("sqrt", 1, _gtrig(np.sqrt)),
    "square": Op("square", 1, lambda a: fmul(a, a)),
}


def _guarded(name, ops, const=1.0):
    return Grammar(name, "real", tuple(GUARDED_OPS[o] for o in ops), ("x",), (const,))


# ---------------------------------------------------------------- Boolean operators
BOOL_OPS = {
    "nand": Op("nand", 2, lambda m, a, b: m ^ (a & b)),
    "nor": Op("nor", 2, lambda m, a, b: m ^ (a | b)),
    "and": Op("and", 2, lambda m, a, b: a & b),
    "or": Op("or", 2, lambda m, a, b: a | b),
    "xor": Op("xor", 2, lambda m, a, b: a ^ b),
    "xnor": Op("xnor", 2, lambda m, a, b: m ^ a ^ b),
    "not": Op("not", 1, lambda m, a: m ^ a),
    "imp": Op("imp", 2, lambda m, a, b: (m ^ a) | b),         # a -> b; with constant 0 complete (C9)
    "rimp": Op("rimp", 2, lambda m, a, b: (m ^ b) | a),       # b -> a: mirrored imp, redundancy control (D052)
    "nimp": Op("nimp", 2, lambda m, a, b: a & (m ^ b)),       # a and not b (random-set universe, D053)
    "rnimp": Op("rnimp", 2, lambda m, a, b: (m ^ a) & b),     # not a and b
}


def _real(name, ops):
    return Grammar(name, "real", tuple(REAL_OPS[o] for o in ops), ("x",), (1.0,))


def _bool(name, ops, n, constants=()):
    return Grammar(f"{name}-n{n}", "bool", tuple(BOOL_OPS[o] for o in ops),
                   tuple(f"x{i + 1}" for i in range(n)), constants, n)


def _family(name, op, const, variables=("x",)):
    return Grammar(name, "real", (FAMILY_OPS[op],), variables, (const,))


GRAMMARS = {g.name: g for g in [
    _real("EML", ["eml"]),
    # the EML family (faithful semantics): one binary operator, the variable x, the distinguished constant
    _family("F-EML", "eml", 1.0),
    _family("F-EDL", "edl", float(np.e)),
    _family("F-NEML", "neml", float("-inf")),
    _family("F-LDE0", "lde", 0.0),
    _family("F-LDE1", "lde", 1.0),
    _family("F-PLI", "pli", 1.0),
    _family("F-PLM", "plm", 1.0),
    # two variables (leaves x, y, c): for x + y, x * y, x / y, ...
    _family("F2-EML", "eml", 1.0, ("x", "y")),
    _family("F2-EDL", "edl", float(np.e), ("x", "y")),
    _family("F2-NEML", "neml", float("-inf"), ("x", "y")),
    _family("F2-LDE0", "lde", 0.0, ("x", "y")),
    _family("F2-LDE1", "lde", 1.0, ("x", "y")),
    _family("F2-PLI", "pli", 1.0, ("x", "y")),
    _family("F2-PLM", "plm", 1.0, ("x", "y")),
    _real("ARITH", ["+", "-", "*"]),
    _real("ELEM6", ["+", "-", "*", "/", "exp", "log"]),
    _real("HYB", ["+", "*", "eml"]),
    # Study 2 real sets (guarded semantics; docs/design/function_sets.md)
    _guarded("R-ARITH", ["+", "-", "*"]),
    _guarded("R-ARITH4", ["+", "-", "*", "/"]),
    _guarded("R-ELEM6", ["+", "-", "*", "/", "exp", "log"]),
    _guarded("R-ELEM4", ["+", "*", "exp", "log"]),
    _guarded("R-HYB", ["+", "*", "eml"]),
    _guarded("R-TRIG", ["+", "*", "sin", "cos"]),
    # D052: practitioner's set (SRBench-style) and the real redundancy control (same functions as F-EML)
    _guarded("R-SRB", ["+", "-", "*", "/", "exp", "log", "sin", "cos", "sqrt", "square"]),
    _guarded("F-EML-R", ["eml", "reml"]),
]}
for _n in (3, 4, 5):
    for _g in [_bool("BOOL-NAND", ["nand"], _n),
               _bool("BOOL-NOR", ["nor"], _n),                       # duality control (D008)
               _bool("BOOL-AON", ["and", "or", "not"], _n),
               _bool("BOOL-AN", ["and", "not"], _n),
               _bool("BOOL-AONX", ["and", "or", "not", "xor"], _n),
               _bool("BOOL-AX1", ["and", "xor"], _n, constants=(1,)),
               _bool("BOOL-RICH", ["and", "or", "nand", "nor", "xor", "xnor", "not"], _n),
               _bool("BOOL-IMP0", ["imp"], _n, constants=(0,)),          # Study 2 (C9)
               _bool("BOOL-IMP0R", ["imp", "rimp"], _n, constants=(0,))]:  # redundancy control (D052)
        GRAMMARS[_g.name] = _g
GRAMMARS["BOOL-NAND-n6"] = _bool("BOOL-NAND", ["nand"], 6)                 # problem-size check (D052)
GRAMMARS["BOOL-AON-n6"] = _bool("BOOL-AON", ["and", "or", "not"], 6)        # problem-size contrast (D053)


def get(name):
    if name not in GRAMMARS and name.split("-")[0] in ("S", "S2"):
        GRAMMARS[name] = _cand(name)
    if name not in GRAMMARS and (name.startswith("RS-") or name.startswith("BOOL-RS-")):   # random held-out sets (T26)
        from reach.tasks.s2_random_sets import grammar_from_name
        GRAMMARS[name] = grammar_from_name(name)
    return GRAMMARS[name]


# ---------------------------------------------------------------- trees
def leaf(v):
    return Node(v, is_operator=False)


def count_nodes(tree):
    n, stack = 0, [tree]
    while stack:
        node = stack.pop()
        n += 1
        if node.is_operator:
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)
    return n


def tree_depth(tree):
    """galib convention: a bare leaf has depth 1."""
    if tree is None:
        return 0
    return 1 + max(tree_depth(tree.left), tree_depth(tree.right))


def expr_str(tree):
    if not tree.is_operator:
        v = tree.value
        if isinstance(v, str):
            return v
        return str(int(v)) if float(v).is_integer() else repr(v)
    if tree.right is None:
        return f"{tree.value}({expr_str(tree.left)})"
    return f"{tree.value}({expr_str(tree.left)}, {expr_str(tree.right)})"


def parse_expr(s, grammar):
    s = s.strip()
    i = s.find("(")
    if i < 0:
        if s in grammar.variables:
            return leaf(s)
        return leaf(float(s) if grammar.kind == "real" else int(s))
    name, inner = s[:i], s[i + 1:-1]
    op = grammar.op[name]
    depth = 0
    for j, ch in enumerate(inner):
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            return Node(name, parse_expr(inner[:j], grammar), parse_expr(inner[j + 1:], grammar), True)
    if op.arity != 1:
        raise ValueError(f"bad expression {s!r}")
    return Node(name, parse_expr(inner, grammar), None, True)


# ---------------------------------------------------------------- evaluation
def var_masks(n):
    """Truth-table column of each input: row r has x_i = bit (n-1-i) of r."""
    rows = 1 << n
    return {f"x{i + 1}": sum(1 << r for r in range(rows) if (r >> (n - 1 - i)) & 1) for i in range(n)}


def evaluate(grammar, tree, xs=None):
    """Real: complex128 vector over xs. Bool: packed truth table (int)."""
    if grammar.kind == "real":
        with np.errstate(all="ignore"):
            return _eval_real(grammar.op, tree, xs)
    return _eval_bool(grammar.op, tree, grammar.mask, var_masks(grammar.n_inputs))


def _shape(xs):
    return next(iter(xs.values())).shape if isinstance(xs, dict) else xs.shape


def _eval_real(ops, tree, xs):
    """xs: one sample array (one variable) or a dict {variable: array} (several variables)."""
    if not tree.is_operator:
        if isinstance(tree.value, str):
            return xs[tree.value] if isinstance(xs, dict) else xs
        return np.full(_shape(xs), complex(tree.value), dtype=np.complex128)
    op = ops[tree.value]
    if op.arity == 1:
        return op.fn(_eval_real(ops, tree.left, xs))
    return op.fn(_eval_real(ops, tree.left, xs), _eval_real(ops, tree.right, xs))


def _eval_bool(ops, tree, mask, vm):
    if not tree.is_operator:
        return vm[tree.value] if isinstance(tree.value, str) else (mask if tree.value else 0)
    op = ops[tree.value]
    if op.arity == 1:
        return op.fn(mask, _eval_bool(ops, tree.left, mask, vm))
    return op.fn(mask, _eval_bool(ops, tree.left, mask, vm), _eval_bool(ops, tree.right, mask, vm))


def mse_against(pred, target, penalty=1000.0):
    """EML-GP's error: mean squared |residual|, a non-finite residual costs `penalty` (EML-GP: 1000)."""
    with np.errstate(all="ignore"):
        diff = np.abs(pred - target)
        bad = ~np.isfinite(diff)
        sq = np.where(bad, 0.0, diff) ** 2
        return float((np.where(bad, penalty, sq)).sum() / diff.size)


def grid_sample():
    step = (DOMAIN_HI - DOMAIN_LO) / (N_POINTS + 1)
    return np.array([DOMAIN_LO + step * (i + 1) for i in range(N_POINTS)], dtype=np.complex128)


def validation_sample(seed):
    rng = random.Random(seed)
    return np.array([complex(rng.uniform(DOMAIN_LO, DOMAIN_HI)) for _ in range(N_POINTS)],
                    dtype=np.complex128)


def grid_sample2():
    """Two variables: the 10 x 10 grid on (2, 3)^2, endpoints excluded (100 points, as in one variable)."""
    t = np.array([DOMAIN_LO + (DOMAIN_HI - DOMAIN_LO) * (i + 1) / 11 for i in range(10)])
    X, Y = np.meshgrid(t, t, indexing="ij")
    return {"x": X.ravel().astype(np.complex128), "y": Y.ravel().astype(np.complex128)}


def validation_sample2(seed):
    rng = random.Random(seed)
    pts = [(rng.uniform(DOMAIN_LO, DOMAIN_HI), rng.uniform(DOMAIN_LO, DOMAIN_HI)) for _ in range(N_POINTS)]
    return {"x": np.array([complex(p[0]) for p in pts]), "y": np.array([complex(p[1]) for p in pts])}


def sample_grid(g):
    return grid_sample() if len(g.variables) == 1 else grid_sample2()


def sample_val(g, seed):
    return validation_sample(seed) if len(g.variables) == 1 else validation_sample2(seed)


def scaled_func_hash(vec, scale, precision=HASH_PRECISION):
    import hashlib
    with np.errstate(all="ignore"):
        v = np.asarray(vec, dtype=np.complex128) / scale
        # + 0.0 maps -0.0 to +0.0: without it, equal functions whose zero imaginary parts carried different
        # signs hashed differently (false "not exact"; found in review 2026-10-15). The enumerator key already did this.
        r, i = np.round(v.real, precision) + 0.0, np.round(v.imag, precision) + 0.0
        r[~np.isfinite(r)] = np.nan
        i[~np.isfinite(i)] = np.nan
    return hashlib.sha1(r.tobytes() + i.tobytes()).hexdigest()


# ---------------------------------------------------------------- sizes
def syntactic_counts(grammar, kmax):
    """N_G(K) for K = 0..kmax: N(1) = a0, N(K) = a1 N(K-1) + a2 sum_{i+j=K-1} N(i) N(j)."""
    a0, a1, a2 = grammar.arity_counts()
    n = [0] * (kmax + 1)
    if kmax >= 1:
        n[1] = a0
    for k in range(2, kmax + 1):
        n[k] = a1 * n[k - 1] + a2 * sum(n[i] * n[k - 1 - i] for i in range(1, k - 1))
    return n


# arity of every real symbol in the random-set universe (D053; tool T26)
REAL_SYMBOL_ARITY = {o: GUARDED_OPS[o].arity for o in GUARDED_OPS}

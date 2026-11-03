"""Logistic reach K50 with a target (cluster) bootstrap: port of legacy analyze_p1.logit / k50.

Separation (guardrail row 118, amended by D088): with complete or quasi-complete separation the maximum-likelihood
slope of the one-predictor fit is -inf (successes at the smaller sizes) or +inf. `fit1` (D088, "keep the sign") returns
that signed limit and K50 at the midpoint of the gap between the largest size with a success and the smallest size
with a failure on the separating side (falling: (max success size + min failure size) / 2; rising: (max failure size +
min success size) / 2; quasi-complete at a shared size s: s). Only truly degenerate data (one outcome, one size, a
singular step) have no fit. `mle` keeps row 118's rule (None when separated) for the fitters that D088 leaves on it."""
from collections import defaultdict
from contextlib import contextmanager
from typing import NamedTuple

import numpy as np


def logit(x, y, iters=300):
    """IRLS fit of P(y) = 1 / (1 + exp(-(b0 + b1 x))). Returns (b0, b1)."""
    X = np.column_stack([np.ones(len(x)), np.asarray(x, float)])
    b = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = np.clip(p * (1 - p), 1e-9, None)
        nb = np.linalg.solve((X * W[:, None]).T @ X, (X * W[:, None]).T @ (X @ b + (y - p) / W))
        if np.max(np.abs(nb - b)) < 1e-12:
            return nb
        b = nb
    return b


def separation(x, y):
    """-1 when successes and failures do not overlap in x except at one shared value and the successes lie at the
    smaller x (complete or quasi-complete separation, falling: the maximum-likelihood slope is -inf), +1 when they lie
    at the larger x (rising: +inf), 0 otherwise. Exact for one predictor with an intercept; IRLS would stop at its
    iteration limit with a large finite slope (guardrail row 118). y may be a rate: y > 0 counts as a success at x and
    y < 1 as a failure. 0 when one outcome is absent (that degenerate case is the caller's own check)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    s, f = x[y > 0], x[y < 1]
    if len(s) == 0 or len(f) == 0:
        return 0
    return -1 if s.max() <= f.min() else 1 if f.max() <= s.min() else 0


def separated(x, y):
    """True when `separation` finds complete or quasi-complete separation in either direction."""
    return separation(x, y) != 0


def separated_columns(X, y, cols=None, tol=1e-6):
    """The columns among `cols` (default: all) of the design X whose logistic coefficient has no finite
    maximum-likelihood value because of complete or quasi-complete separation (guardrail row 118): there is a direction
    w with (2y - 1) x_i . w >= 0 for every row and w_j != 0, along which the likelihood never falls. Rows with the same
    design point and both outcomes force x . w = 0, so when those points have full column rank no such w exists and no
    linear program is needed (the usual case). Otherwise each column is checked by two linear programs (max and min of
    w_j within the box |w| <= 1) on columns scaled to unit maximum. y may be a rate (y > 0 a success, y < 1 a
    failure). Assumes X has full column rank (a collinear design is a different defect)."""
    from scipy.optimize import linprog
    X, y = np.asarray(X, float), np.asarray(y, float)
    k = X.shape[1]
    cols = list(range(k)) if cols is None else list(cols)
    pts, inv = np.unique(X, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    succ = np.bincount(inv, weights=(y > 0).astype(float), minlength=len(pts)) > 0
    fail = np.bincount(inv, weights=(y < 1).astype(float), minlength=len(pts)) > 0
    mixed = pts[succ & fail]
    if len(mixed) and np.linalg.matrix_rank(mixed) == k:
        return []
    scale = np.abs(X).max(axis=0)
    scale[scale == 0] = 1.0
    Z = np.vstack([pts[succ] / scale, -pts[fail] / scale])             # z . w >= 0 for every row
    out = []
    for j in cols:
        for sign in (1.0, -1.0):
            c = np.zeros(k)
            c[j] = -sign                                                 # maximise sign * w_j
            r = linprog(c, A_ub=-Z, b_ub=np.zeros(len(Z)), bounds=[(-1, 1)] * k, method="highs")
            if r.status == 0 and -r.fun > tol:
                out.append(j)
                break
    return out


class Fit(NamedTuple):
    """A one-predictor fit (D088): b0, b1 (b1 = -inf or +inf and b0 NaN when separated), k50 (-b0/b1, or the gap
    midpoint when separated) and separated (-1 falling, +1 rising, 0 none)."""
    b0: float
    b1: float
    k50: float
    separated: int


TALLY = {}          # {label: dict(fits, falling, rising)}: one-predictor fits counted under `counting` (D088 report)
_LABELS = []


@contextmanager
def counting(label):
    """Count every non-degenerate `fit1` (and so `mle`) call made inside the block under `label` in TALLY: fits, and
    separated ones falling and rising. Nested blocks count under the innermost label; calls outside any block are not
    counted. The caller clears TALLY."""
    _LABELS.append(label)
    try:
        yield
    finally:
        _LABELS.pop()


def _count(sep):
    if _LABELS:
        c = TALLY.setdefault(_LABELS[-1], dict(fits=0, falling=0, rising=0))
        c["fits"] += 1
        if sep:
            c["falling" if sep < 0 else "rising"] += 1


def gap_k50(x, y, sep):
    """K50 of a separated fit (D088): the midpoint of the gap between the largest size with any success and the
    smallest size with any failure (sep = -1), or between the largest size with any failure and the smallest size with
    any success (sep = +1, the wrong direction). Quasi-complete separation at a shared size s gives s."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    s, f = x[y > 0], x[y < 1]
    return float((s.max() + f.min()) / 2) if sep < 0 else float((f.max() + s.min()) / 2)


def fit1(x, y):
    """D088's one-predictor fit: None when truly degenerate (no data, one outcome, one size, or a singular step);
    Fit(nan, -inf or +inf, gap midpoint, -1 or +1) when separated; otherwise `logit`'s (b0, b1) with k50 = -b0/b1."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(y) == 0 or y.min() == y.max() or x.min() == x.max():
        return None
    sep = separation(x, y)
    if sep:
        _count(sep)
        return Fit(float("nan"), sep * float("inf"), gap_k50(x, y, sep), sep)
    try:
        b = logit(x, y)
    except np.linalg.LinAlgError:
        return None
    _count(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return Fit(b[0], b[1], -b[0] / b[1], 0)


def mle(x, y):
    """Row 118's rule: `logit`'s (b0, b1), or None when the maximum-likelihood fit does not exist or cannot be computed
    (one outcome, one size, separation or a singular step). For the fitters D088 leaves on row 118 (H8c's base
    curves)."""
    f = fit1(x, y)
    return None if f is None or f.separated else np.array([f.b0, f.b1])


def k50(size, y, cluster, boot=1000, seed=20260926):
    """Midpoint -b0/b1 on `size` (any unit), 95% CI from resampling clusters (targets).
    Returns (estimate, lo, hi, slope); NaN when the fit is degenerate. A separated fit (D088) gives the gap midpoint
    and slope -inf (+inf in the wrong direction); a degenerate or rising replicate is skipped, a separated falling one
    enters at its gap midpoint."""
    size, y = np.asarray(size, float), np.asarray(y, float)
    nan4 = (float("nan"),) * 4
    f = fit1(size, y)
    if f is None:
        return nan4
    est = f.k50
    by = defaultdict(list)
    for i, c in enumerate(cluster):
        by[c].append(i)
    ids, rng, bs = list(by), np.random.default_rng(seed), []
    for _ in range(boot):
        idx = np.concatenate([by[ids[j]] for j in rng.integers(0, len(ids), len(ids))])
        bb = fit1(size[idx], y[idx])
        if bb is not None and bb.b1 < 0:
            bs.append(bb.k50)
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (float("nan"), float("nan"))
    return float(est), float(lo), float(hi), float(f.b1)

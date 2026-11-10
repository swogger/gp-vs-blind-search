"""Isotonic interpolated reach (REQ-ANA-108; Stage 5 Rule S2's secondary estimator, stage5_design.md Section 1).
EXPLORATORY until a registration names it; it never replaces the registered logistic K50.

Per level (one size value in bits), the mean of the targets' success probabilities; a non-increasing fit in bits by
pool-adjacent-violators (PAV), weighted by the number of targets at each level; the reach is where the fitted curve
crosses 1/2, interpolated linearly between the two adjacent levels that bracket the crossing: the last level whose
fitted value is at least 1/2 and the next one. Not crossing: every fitted value at least 1/2 gives the top level,
flagged "at_least_top"; the lowest level's fitted value below 1/2 gives the lowest level, flagged "at_most_lowest".

Properties (tests/test_iso_reach.py): adding a success never lowers the reach (PAV is monotone in its data and the
interpolation is monotone in both bracketing values); adding or removing a level on its own side of 1/2 (above the
crossing with a rate below 1/2, below it with a rate of at least 1/2) keeps the reach between the two bracketing
levels, where the logistic K50 moves with every level that enters the fit.
"""
import math
from collections import defaultdict

import numpy as np

FLAGS = ("bracketed", "at_least_top", "at_most_lowest", "empty")


def pav_nonincreasing(y, w):
    """Weighted least-squares non-increasing fit of y (in the order given) by pool-adjacent-violators."""
    blocks = []                                   # [weighted sum, weight, count]
    for yi, wi in zip(y, w):
        blocks.append([float(yi) * float(wi), float(wi), 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
            s, ww, n = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += ww
            blocks[-1][2] += n
    out = []
    for s, ww, n in blocks:
        out += [s / ww] * n
    return np.array(out, float)


def levels(sizes, probs, ndigits=9):
    """(x, mean, weight): one entry per distinct size (rounded to `ndigits`, so float noise does not split a level),
    the mean of its targets' probabilities and its number of targets; NaN probabilities are left out, so a level
    with none never appears."""
    by = defaultdict(list)
    for s, p in zip(sizes, probs):
        if p is not None and np.isfinite(p) and np.isfinite(s):
            by[round(float(s), ndigits)].append(float(p))
    x = sorted(by)
    return (np.array(x, float), np.array([np.mean(by[k]) for k in x], float),
            np.array([len(by[k]) for k in x], float))


def reach(sizes, probs, ndigits=9):
    """The isotonic interpolated reach of targets with sizes `sizes` (bits) and success probabilities `probs`."""
    x, m, w = levels(sizes, probs, ndigits)
    if not len(x):
        return dict(reach=float("nan"), flag="empty", lo=None, hi=None, n_levels=0, levels=[], fitted=[], means=[])
    f = pav_nonincreasing(m, w)
    base = dict(n_levels=len(x), levels=x.tolist(), fitted=f.tolist(), means=m.tolist())
    if f[0] < 0.5:
        return dict(base, reach=float(x[0]), flag="at_most_lowest", lo=None, hi=float(x[0]))
    i = int(np.nonzero(f >= 0.5)[0][-1])           # the last level at or above 1/2 (f is non-increasing)
    if i == len(x) - 1:
        return dict(base, reach=float(x[-1]), flag="at_least_top", lo=float(x[-1]), hi=None)
    r = x[i] + (f[i] - 0.5) / (f[i] - f[i + 1]) * (x[i + 1] - x[i])
    return dict(base, reach=float(r), flag="bracketed", lo=float(x[i]), hi=float(x[i + 1]))


def target_probs(records, unit="bits_sem"):
    """(sizes, probs) per target from 0/1 records (one per run, each with target_id, y and units[unit]): a target's
    probability is the share of its runs that succeeded. Records of one set only (target ids are per set)."""
    by, size = defaultdict(list), {}
    for r in records:
        by[r["target_id"]].append(r["y"])
        size[r["target_id"]] = r["units"][unit]
    ids = sorted(by)
    return [size[t] for t in ids], [float(np.mean(by[t])) for t in ids]


def reach_records(records, unit="bits_sem"):
    return reach(*target_probs(records, unit))


def gap(res, null_bits):
    """reach - null with the flag carried: "at_least_top" makes the gap a lower bound, "at_most_lowest" an upper."""
    return dict(gap=res["reach"] - null_bits if math.isfinite(res["reach"]) else float("nan"), flag=res["flag"])

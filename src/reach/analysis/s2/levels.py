"""Per-level recovery rates with design-effect Wilson intervals, and level suggestions (tool T16; REQ-ANA-03).

Runs are nested in targets: a level's interval uses n_eff = n / DEFF with DEFF = 1 + (m - 1)·ICC (m runs per target);
when a level's ICC is undefined (all outcomes equal), the set-wide ICC is used as a floor (paper 1, lesson 15)."""
from collections import defaultdict

import numpy as np
from scipy.stats import norm


def icc(groups):
    """One-way ANOVA ICC for binary outcomes; groups = list of per-target outcome lists. NaN if undefined."""
    groups = [np.asarray(g, float) for g in groups if len(g)]
    k = len(groups)
    n = sum(len(g) for g in groups)
    if k < 2 or n <= k:
        return float("nan")
    grand = np.concatenate(groups).mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    msb, msw = ssb / (k - 1), ssw / (n - k)
    m = (n - sum(len(g) ** 2 for g in groups) / n) / (k - 1)
    den = msb + (m - 1) * msw
    return float("nan") if den == 0 else float((msb - msw) / den)


def wilson(k, n, deff=1.0, conf=0.95):
    if n == 0:
        return float("nan"), float("nan")
    ne = n / deff
    p = k / n
    z = norm.ppf(0.5 + conf / 2)
    c = (p + z * z / (2 * ne)) / (1 + z * z / ne)
    h = z * np.sqrt(p * (1 - p) / ne + z * z / (4 * ne * ne)) / (1 + z * z / ne)
    return max(0.0, c - h), min(1.0, c + h)


def level_rates(records, size="K"):
    """records: dicts with `size`, target_id, y. Returns one row per level: rate, Wilson interval with DEFF."""
    by = defaultdict(lambda: defaultdict(list))
    for r in records:
        by[r[size]][r["target_id"]].append(r["y"])
    all_groups = [g for lv in by.values() for g in lv.values()]
    floor = icc(all_groups)
    floor = 0.0 if np.isnan(floor) else max(0.0, floor)
    rows = []
    for k in sorted(by):
        groups = list(by[k].values())
        y = np.concatenate([np.asarray(g) for g in groups])
        rho = icc(groups)
        rho = floor if np.isnan(rho) else max(rho, 0.0)
        m = len(y) / len(groups)
        deff = 1 + (m - 1) * rho
        lo, hi = wilson(int(y.sum()), len(y), deff)
        rows.append(dict(level=k, n=len(y), targets=len(groups), rate=float(y.mean()), lo=lo, hi=hi, icc=rho, deff=deff))
    for r in rows:
        r[size] = r.pop("level")
    return rows


def suggest_levels(available, b0, b1, n=10, p_hi=0.95, p_lo=0.05):
    """n levels from `available`, evenly spaced between the sizes where logit P = b0 + b1·K gives p_hi and p_lo."""
    logit = lambda p: np.log(p / (1 - p))                         # noqa: E731
    k_hi, k_lo = (logit(p_hi) - b0) / b1, (logit(p_lo) - b0) / b1
    want = np.linspace(min(k_hi, k_lo), max(k_hi, k_lo), n)
    av = np.array(sorted(available))
    out = []
    for w in want:
        for k in av[np.argsort(np.abs(av - w))]:
            if int(k) not in out:
                out.append(int(k))
                break
    return sorted(out)

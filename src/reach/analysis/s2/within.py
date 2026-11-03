"""H3: the same function in different sets (tool T16, `within`; docs/design/hypotheses.md, H3; brief §9.5).

Input: tidy records of the shared-function pool (`load.tidy` on `s2_cal_pool`, regime A at E0). A pool target's id is
"<set>_<pool_id>", so the function is read from the target id.

- `fixed_effects`: logit P(exact) = a_f - beta * size, one intercept per function f, standard errors clustered by
  function. Functions whose runs all succeed or all fail carry no information on beta and are dropped (counted).
  This part of H3 holds if the coefficient on size is negative (upper 95% bound `hi` < 0). If every kept function
  is separated by size in the same direction (`jointly_separated`), the coefficient has no maximum-likelihood value
  and the fit is refused (ValueError: "not estimable"; guardrail row 118).
- `rank_share`: per function, Kendall's tau-b between the set's size and the set's recovery rate. The share of
  functions with tau < 0 among all functions with runs is compared with the registered share; an undefined tau (no
  size spread across sets, or all rates equal) counts as not negative and stays in the denominator (as filed; D079 b).
  The share among functions with a size spread is reported beside it (exploratory). Functions with tau > 0 are
  listed as counter-instances.
- `pooled_slope`: the same logit without function intercepts (shows how much of the slope is between functions).
"""
from collections import defaultdict

import numpy as np
import statsmodels.api as sm
from scipy.stats import kendalltau


def pool_id(target_id):
    return target_id.rsplit("_", 1)[1]


def _one_domain(recs):
    from reach import grammar as G
    kinds = set()
    for name in {r["set"] for r in recs}:
        try:
            kinds.add(G.get(name).kind)
        except (KeyError, ValueError):                     # synthetic names in tests
            pass
    if len(kinds) > 1:
        raise ValueError("H3 runs per domain: pool ids repeat across domains")


def _size(r, unit):
    return r["K"] if unit == "nodes" else r["units"][unit]


def jointly_separated(groups):
    """True when the model with an intercept per group and one common slope has no maximum-likelihood slope: every
    group's successes and failures are separated by x in the same direction (complete or quasi-complete; a group with
    one x value fits either direction). `groups`: [(x, y)], each group with both outcomes. Exact: if one group
    overlaps, or two groups are separated in opposite directions, the likelihood falls as the slope grows either
    way, so the slope is finite."""
    down = up = True
    for x, y in groups:
        x, y = np.asarray(x, float), np.asarray(y, float)
        s, f = x[y > 0], x[y < 1]
        down = down and s.max() <= f.min()
        up = up and f.max() <= s.min()
    return bool(groups) and (down or up)


def fixed_effects(recs, unit="nodes"):
    _one_domain(recs)
    by = defaultdict(list)
    for r in recs:
        by[pool_id(r["target_id"])].append(r)
    keep = {f: rr for f, rr in by.items() if len({r["y"] for r in rr}) == 2}
    if jointly_separated([([_size(r, unit) for r in rr], [r["y"] for r in rr]) for rr in keep.values()]):
        raise ValueError("every function is separated by size in the same direction: no maximum-likelihood slope")
    fs = sorted(keep)
    rows = [(i, r) for i, f in enumerate(fs) for r in keep[f]]
    y = np.array([r["y"] for _, r in rows], float)
    X = np.zeros((len(rows), len(fs) + 1))
    for j, (i, r) in enumerate(rows):
        X[j, i] = 1.0
        X[j, -1] = _size(r, unit)
    groups = np.array([i for i, _ in rows])
    res = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds=dict(groups=groups))
    b, se = float(res.params[-1]), float(res.bse[-1])
    return dict(beta=-b, coef=b, se=se, lo=b - 1.96 * se, hi=b + 1.96 * se, functions=len(fs),
                dropped_constant=len(by) - len(fs), n=len(y))


def pooled_slope(recs, unit="nodes"):
    y = np.array([r["y"] for r in recs], float)
    X = sm.add_constant(np.array([_size(r, unit) for r in recs], float))
    return float(sm.GLM(y, X, family=sm.families.Binomial()).fit().params[1])


def rank_share(recs, unit="nodes"):
    _one_domain(recs)
    cell = defaultdict(list)
    for r in recs:
        cell[(pool_id(r["target_id"]), r["set"])].append(r)
    per = defaultdict(list)
    for (f, s), rr in cell.items():
        per[f].append((_size(rr[0], unit), float(np.mean([r["y"] for r in rr]))))
    neg, undefined, counter, spread = 0, 0, [], 0
    for f, pts in sorted(per.items()):
        sizes, rates = zip(*pts)
        if max(sizes) == min(sizes):
            continue
        spread += 1
        tau = kendalltau(sizes, rates)[0] if len(set(rates)) > 1 else float("nan")
        if not np.isfinite(tau):
            undefined += 1
        elif tau < 0:
            neg += 1
        elif tau > 0:
            counter.append(f)
    return dict(functions=len(per), with_spread=spread, no_spread=len(per) - spread, negative=neg,
                undefined=undefined, share_negative=neg / len(per) if per else float("nan"),
                share_negative_spread_only=neg / spread if spread else float("nan"), counter_instances=counter)

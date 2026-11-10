"""Stage 5 statistics library (stage5_design.md Sections 1 and 6; REQ-ANA-109 to REQ-ANA-111). No phase is read here:
the driver that loads the Stage 5 phases comes with them. Until Stage 5 is registered every use is EXPLORATORY.

- `shared_levels` / `contrast` (Rule S1, guardrail row 170): a contrast within a set fits both arms on the identical
  set of (set, level, target) rows; anything else is refused.
- `histogram`, `union_expected`, `union_share`, `restart_twin_shares` (REQ-ANA-109, Section 6): the expected union
  coverage of R restarts drawn without replacement from a target's runs, exact by the hypergeometric formula.
- `target_bootstrap`, `noninferiority`, `tost`, `conjunction`, `holm`, `set_level_interval` (REQ-ANA-110, Rule S3):
  Stages 3 and 4's inference, reusing `h6d.equivalence`, `h6d.holm`, `remedy.p_beyond` and `remedy.conjunction_p`;
  Stage 4's H11 replicates and p values are reproduced exactly through these functions.
"""
import math
from collections import defaultdict
from fractions import Fraction
from math import comb

import numpy as np

from reach.analysis.s2 import h6d, remedy


def row_keys(rows, unit="bits_sem", ndigits=9):
    """{(set, level, target_id)} of an arm: records with set, target_id and units[unit], or such triples."""
    out = set()
    for r in rows:
        if isinstance(r, tuple):
            s, lv, t = r
        else:
            s, lv, t = r["set"], r["units"][unit], r["target_id"]
        out.add((s, round(float(lv), ndigits), t))
    return out


def shared_levels(a, b, name="contrast", unit="bits_sem"):
    """The rows both arms are fitted on; ValueError when the arms differ in any (set, level, target) row (Rule S1)."""
    ka, kb = row_keys(a, unit), row_keys(b, unit)
    if ka != kb:
        only_a, only_b = sorted(ka - kb), sorted(kb - ka)
        raise ValueError(f"{name}: the arms differ in (set, level, target) rows: {len(only_a)} only in the first "
                         f"(e.g. {only_a[:2]}), {len(only_b)} only in the second (e.g. {only_b[:2]}); Rule S1 refuses "
                         "a contrast fitted on different rows")
    return sorted(ka)


def contrast(a, b, estimate, name="contrast", unit="bits_sem"):
    """estimate(a) - estimate(b) after the shared-level check."""
    shared_levels(a, b, name, unit)
    return estimate(a) - estimate(b)


# ---------------------------------------------------------------- union coverage of restarts (REQ-ANA-109; Section 6)
def histogram(key_sets, level_of):
    """{level: [functions covered by exactly m of the runs, m = 0..n]} from the runs' sets of distinct keys
    (`level_of(key)` gives a key's level; m = 0 holds 0: a function no run covered is not in any set)."""
    n, count = len(key_sets), defaultdict(int)
    for ks in key_sets:
        for k in ks:
            count[k] += 1
    out = defaultdict(lambda: [0] * (n + 1))
    for k, m in count.items():
        out[level_of(k)][m] += 1
    return dict(out)


def union_expected(hist, R):
    """Expected number of distinct functions R runs cover when drawn without replacement from the n runs of the
    histogram (hist[m] functions covered by exactly m of them): sum_m hist[m] (1 - C(n - m, R) / C(n, R)), exact
    (a Fraction). A function covered by m runs is missed only when all R draws avoid those m runs."""
    n = len(hist) - 1
    if not 1 <= R <= n:
        raise ValueError(f"R = {R} restarts from {n} runs: the hypergeometric reading needs 1 <= R <= n")
    tot = comb(n, R)
    return sum((Fraction(h) * (1 - Fraction(comb(n - m, R), tot)) for m, h in enumerate(hist) if m), Fraction(0))


def union_share(hist, R, U):
    """The share of a level's U functions that R restarts cover on average (union_expected / U)."""
    return float(union_expected(hist, R) / U) if U else 0.0


def restart_twin_shares(targets, R, U):
    """The twin of R restarts per level (memo Section 6; the conditioning rule is the design point the pilot settles):
    `targets` [dict(level=the target's level, n_prime=its runs that did not hit, hist={K: histogram over those runs})];
    per target and level K, the share of K that R of its n_prime runs cover; a target with n_prime < R takes the mean
    of its level's targets that have R or more (pooled); levels of targets weighted by their targets. Returns
    dict(shares={K: share}, eligible, pooled, unpooled_levels)."""
    by = defaultdict(list)
    for t in targets:
        by[t["level"]].append(t)
    acc, weight, eligible, pooled, empty = defaultdict(float), 0, 0, 0, []
    for lv, ts in sorted(by.items()):
        ok = [t for t in ts if t["n_prime"] >= R]
        if not ok:
            empty.append(lv)
            continue
        eligible += len(ok)
        pooled += len(ts) - len(ok)
        for K, u in U.items():
            vals = [union_share(t["hist"].get(K, [0] * (t["n_prime"] + 1)), R, u) for t in ok]
            acc[K] += len(ts) * float(np.mean(vals))
        weight += len(ts)
    shares = {K: acc[K] / weight for K in U} if weight else {K: float("nan") for K in U}
    return dict(shares=shares, eligible=eligible, pooled=pooled, unpooled_levels=empty)


# ---------------------------------------------------------------- inference (REQ-ANA-110; Rule S3)
SEED, BOOT, KEEP, ALPHA = 20261020, 2000, 0.9, 0.05


def target_bootstrap(ids_by_set, stat, boot=BOOT, seed=SEED):
    """Replicates of `stat({set: [target ids]})` with targets resampled within sets, the same draw for every arm a
    target appears in: per replicate, per set in sorted order, len(ids) draws over the set's sorted ids (the order of
    Stage 4's `remedy.h11`, so its replicates are reproduced exactly)."""
    rng = np.random.default_rng(seed)
    sets = sorted(ids_by_set)
    ids = {g: sorted(ids_by_set[g]) for g in sets}
    reps = []
    for _ in range(boot):
        pick = {}
        for g in sets:
            pick[g] = [ids[g][j] for j in rng.integers(0, len(ids[g]), len(ids[g]))]
        reps.append(stat(pick))
    return reps


def noninferiority(point, reps, margin, side="above", boot=None):
    """One-sided test that the statistic lies beyond `margin` (`remedy.p_beyond`: p = (replicates at or on the wrong
    side + 1)/(kept + 1)); estimable when at least 90% of the replicates are kept and the point is finite."""
    t = dict(remedy.p_beyond(reps, margin, side, boot or len(reps)), estimate=point, margin=margin, side=side)
    t["estimable"] = bool(t["estimable"] and math.isfinite(point))
    return t


def tost(point, reps, delta):
    """Equivalence within +-delta: the larger of the two one-sided bootstrap p's (`h6d.equivalence`, Stage 3)."""
    return h6d.equivalence(point, reps, delta)


def conjunction(tests):
    """'Every member holds': p = the largest member p (`remedy.conjunction_p`), estimable when every member is."""
    return dict(p=remedy.conjunction_p([t["p"] for t in tests.values()]), members=sorted(tests),
                estimable=all(t["estimable"] for t in tests.values()))


def holm(tests, alpha=ALPHA):
    """Holm within a family (`h6d.holm`): a test that is not estimable counts as p = 1 and is not supported."""
    return h6d.holm(tests, alpha)


def set_level_interval(values, level=0.90):
    """Reported beside a domain mean, never deciding: the t interval over the per-set estimates."""
    from scipy.stats import t as tdist
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if len(v) < 2:
        return dict(mean=float(np.mean(v)) if len(v) else float("nan"), lo=float("nan"), hi=float("nan"), n=len(v))
    m, se = float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))
    q = tdist.ppf(0.5 + level / 2, len(v) - 1)
    return dict(mean=m, se=se, lo=m - q * se, hi=m + q * se, n=len(v))

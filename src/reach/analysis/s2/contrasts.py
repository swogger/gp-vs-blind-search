"""Mechanism contrasts (tool T16; brief §9.5; H5, H6): paired K50 differences on the same targets and seeds with a
paired target bootstrap; exact McNemar tests per level on seed-matched pairs; Holm within declared families whose size
counts untestable members (lesson 16, REQ-ANA-07); Wilcoxon and Friedman summaries across sets (Derrac et al. 2011)."""
from collections import defaultdict
from math import comb

import numpy as np
from scipy import stats

from reach.analysis.k50 import fit1


def holm(pvals, m):
    """Holm-adjusted p-values for a family of size m; None entries are untestable members (they still count in m)."""
    idx = sorted((p, i) for i, p in enumerate(pvals) if p is not None)
    out = [None] * len(pvals)
    running = 0.0
    for rank, (p, i) in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * p))
        out[i] = running
    return out


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p for b and c discordant pairs; None when there are none (untestable)."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def mcnemar_per_level(records, a, b, size="K"):
    """Stage 1: "exact McNemar per level" (H5 support; design 9.5: seed-matched). Pairs are runs of algorithms a and b on
    the same target and seed; per size level: pairs, b (a exact, b not), c (b exact, a not), ties ("both or neither
    exact carries no information") and the two-sided exact p (None without discordant pairs: untestable). A run
    without a partner is left out."""
    runs = defaultdict(dict)
    for r in records:
        if r["algorithm"] in (a, b):
            runs[(r["target_id"], r["seed"])][r["algorithm"]] = r
    out = {}
    for pair in runs.values():
        if a not in pair or b not in pair:
            continue
        lv = out.setdefault(pair[a][size], dict(pairs=0, b=0, c=0, ties=0))
        ya, yb = pair[a]["y"], pair[b]["y"]
        lv["pairs"] += 1
        lv["b"] += int(ya == 1 and yb == 0)
        lv["c"] += int(ya == 0 and yb == 1)
        lv["ties"] += int(ya == yb)
    for lv in out.values():
        lv["p"] = mcnemar_exact(lv["b"], lv["c"])
    return dict(sorted(out.items()))


def _k50(K, y):
    """K50 = -b0/b1; the gap midpoint when separated with successes at the smaller sizes (D088); NaN when degenerate
    or the slope is not negative."""
    f = fit1(K, y)
    return float(f.k50) if f is not None and f.b1 < 0 else float("nan")


def paired_k50_difference(records, a, b, boot=2000, seed=20261020, size="K", keep_reps=False):
    """K50(a) - K50(b) on targets present for both, with a bootstrap of those targets. keep_reps: also return the
    finite replicates (`reps`), for a p by CI inversion (H5's Holm step; the caller removes them)."""
    by = defaultdict(lambda: defaultdict(list))
    for r in records:
        by[r["target_id"]][r["algorithm"]].append((r[size], r["y"]))
    tids = [t for t in by if by[t][a] and by[t][b]]

    def diff(ts):
        A = [v for t in ts for v in by[t][a]]
        B = [v for t in ts for v in by[t][b]]
        return _k50(*zip(*A)) - _k50(*zip(*B))
    est = diff(tids)
    rng = np.random.default_rng(seed)
    reps = [diff([tids[i] for i in rng.integers(0, len(tids), len(tids))]) for _ in range(boot)]
    ok = [x for x in reps if np.isfinite(x)]
    lo, hi = np.percentile(ok, [2.5, 97.5]) if ok else (float("nan"),) * 2      # saturated: not estimable
    out = dict(diff=est, lo=float(lo), hi=float(hi), targets=len(tids), dropped=boot - len(ok),
               estimable=len(ok) >= 0.9 * boot)
    return dict(out, reps=ok) if keep_reps else out


def wilcoxon_across_sets(diffs):
    """Signed-rank test of per-set K50 differences (one value per set)."""
    d = [x for x in diffs if np.isfinite(x)]
    return None if len(d) < 2 else float(stats.wilcoxon(d).pvalue)


def friedman_across_sets(table):
    """table: list of per-set rows, one K50 per algorithm (same order). Friedman chi-square p."""
    cols = list(zip(*table))
    return None if len(table) < 2 or len(cols) < 3 else float(stats.friedmanchisquare(*cols).pvalue)

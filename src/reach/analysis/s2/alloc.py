"""Best split of a budget into independent runs (tool T16; REQ-ANA-06; paper 1's restarts analysis, D038, generalised).

Runs at E0 (200 generations) per target give, by the prefix property, a run of any length L <= 200. A budget of B x E0
spent as r = B*200/L runs of L generations succeeds on a target with probability
    1 - C(n - s, r) / C(n, r)      (unbiased, r <= n; s of n runs succeeded by L)
    1 - (1 - s/n)^r                (plug-in when r > n; conservative and labelled).
K50 comes from a logistic fit of the per-target probabilities on size. The best split is the length with the highest K50;
inside the bootstrap it is re-selected in every replicate (paper 1 did the same), so its CI includes the selection.
A run's outcome at L generations is `load.outcome`'s: CPU-limited runs (and runs killed without a result) count as
failures (primary), or are left out where censored (censored_as_failure=False: the Stage 1 sensitivity analysis)."""
from collections import defaultdict
from math import comb

import numpy as np

from reach.analysis.k50 import fit1
from reach.analysis.s2.load import outcome


def p_at_least_one(s, n, r):
    if r <= n:
        return 1 - comb(n - s, r) / comb(n, r), False
    return 1 - (1 - s / n) ** r, True


def _per_target(runs):
    """Error rows are left out; a row killed without a result (no gen_exact) counts as a failure (primary rule)."""
    by = defaultdict(list)
    for r in runs:
        if r.get("terminal_state") != "error":
            by[r["target_id"]].append(r)
    return by


def _k50(K, p):
    """K50 of the split's success curve; +inf when every target is solved (reach above the levels, which is the best a
    split can do), -inf when none is, NaN when not estimable (review 2, row 68). Separated rates (y > 0 a success, y < 1
    a failure) with successes at the smaller sizes give the gap midpoint (D088); separated the other way, NaN as any
    rising curve."""
    K, p = np.asarray(K, float), np.asarray(p, float)
    if len(p) == 0 or K.min() == K.max():
        return float("nan")
    if p.min() >= 1 - 1e-12:
        return float("inf")
    if p.max() <= 1e-12:
        return float("-inf")
    f = fit1(K, p)
    if f is None:
        return float("nan")
    if f.separated:
        return float(f.k50) if f.b1 < 0 else float("nan")
    k = float(f.k50) if f.b1 < 0 else float("nan")
    span = K.max() - K.min()
    return k if abs(k - (K.max() + K.min()) / 2) <= 10 * span else float("nan")   # near separation: not estimable


def _pct(xs, q):
    """Nearest-rank percentile (copes with +-inf)."""
    xs = sorted(xs)
    return xs[int(round(q / 100 * (len(xs) - 1)))]


def counts(rr, L, censored_as_failure=True):
    """(successes, runs) of one target's (non-error) runs at L generations: a success is an exact hit before L; every
    other run is a failure (primary), or, with censored_as_failure=False (the Stage 1 sensitivity analysis), left out
    when censored at L (`load.outcome`: a CPU stop before L, or killed without a result)."""
    s = n = 0
    for x in rr:
        if x.get("gen_exact") is not None and x["gen_exact"] < L:
            s += 1
        elif not censored_as_failure and outcome(x, L) == "censored":
            continue
        n += 1
    return s, n


def _table(by, budget, lengths, censored_as_failure=True):
    out, plugs = {}, {}
    for L in lengths:
        r = int(round(budget * 200 / L))
        Ks, ps, plug = [], [], False
        for tid, rr in by.items():
            s, n = counts(rr, L, censored_as_failure)
            if n == 0:
                continue
            p, pl = p_at_least_one(s, n, r)
            Ks.append(rr[0]["K"])
            ps.append(p)
            plug |= pl
        out[L], plugs[L] = _k50(Ks, ps), plug
    return out, plugs


def best_split(runs, budget, lengths=(25, 50, 100, 200), censored_as_failure=True):
    k, plugs = _table(_per_target(runs), budget, lengths, censored_as_failure)
    finite = [L for L in lengths if not np.isnan(k[L]) and k[L] > float("-inf")]  # +inf (all solved) may win
    best = max(finite, key=lambda L: k[L]) if finite else None                    # nothing solved: none
    return dict(k50=k, plug_in=plugs, best_length=best, best_k50=k[best] if best else float("nan"))


def best_split_ci(runs, budget, lengths=(25, 50, 100, 200), boot=1000, seed=20261020, censored_as_failure=True):
    by = _per_target(runs)
    tids = list(by)
    rng = np.random.default_rng(seed)
    est = best_split(runs, budget, lengths, censored_as_failure)
    reps, chosen = [], []
    for _ in range(boot):
        draw = rng.choice(len(tids), len(tids), replace=True)
        bb = {f"{tids[i]}#{j}": by[tids[i]] for j, i in enumerate(draw)}
        k, _ = _table(bb, budget, lengths, censored_as_failure)
        finite = [L for L in lengths if not np.isnan(k[L]) and k[L] > float("-inf")]
        if finite:
            L = max(finite, key=lambda x: k[x])
            reps.append(k[L])
            chosen.append(L)
    lo, hi = (_pct(reps, 2.5), _pct(reps, 97.5)) if reps else (float("nan"),) * 2
    return dict(k50=est["best_k50"], best_length=est["best_length"], lo=float(lo), hi=float(hi), chosen_lengths=chosen,
                dropped=boot - len(reps), estimable=len(reps) >= 0.9 * boot)

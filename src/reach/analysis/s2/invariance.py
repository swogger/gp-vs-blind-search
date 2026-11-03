"""H2: is a size unit invariant across function sets? (tool T16; docs/design/hypotheses.md, H2 rule)

Per unit u and set G (one domain): LOGO log-loss = a logistic fit of recovery on s_u over the other sets, scored on G's
runs; within log-loss = target-fold cross-validation inside G. Excess(u) = mean over sets of (LOGO - within). The unit is
invariant if the upper 95% bound of the excess (joint bootstrap of targets within every set) is at most Delta.

A LOGO or within-set fit on separated data (`k50.separated`: no maximum-likelihood fit; guardrail row 118) makes that
set's log-loss, and so the excess, not estimable: a replicate with such a fit is degenerate (dropped and counted), and
with an undefined point estimate or more than 10% dropped the unit is not estimable (no p, upper95 = inf, not
invariant). The fields `dropped` and `estimable` are added only when a replicate was dropped or the excess is not
estimable, so results without separation are unchanged."""
from collections import defaultdict

import numpy as np

from reach.analysis.k50 import logit, separated
from reach.analysis.s2 import law


def _ll(b, x, y):
    if b is None:
        return float("nan")
    p = np.clip(1 / (1 + np.exp(-(b[0] + b[1] * x))), 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _fit(x, y):
    """(b0, b1); the clipped-mean intercept when y has one value; None when separated (no maximum-likelihood fit)."""
    if len(set(y)) < 2:
        m = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        return np.array([np.log(m / (1 - m)), 0.0])
    if separated(x, y):
        return None
    return logit(x, y)


def _arrays(recs, unit):
    by = defaultdict(lambda: defaultdict(list))
    for r in recs:
        by[r["set"]][r["target_id"]].append((r["units"][unit], r["y"]))
    return by


def _excess_from(by, folds=5, rng=None):
    return float(np.mean(list(_per_set_from(by, folds, rng).values())))


def _per_set_from(by, folds=5, rng=None, train=None):
    """{set: LOGO - within log-loss}; LOGO fits use the other sets of `train` (default: all sets)."""
    return {s: logo - within for s, (logo, within) in _per_set_parts(by, folds, rng, train).items()}


def _per_set_parts(by, folds=5, rng=None, train=None):
    """{set: (LOGO log-loss, within-set cross-validated log-loss)}."""
    sets = sorted(by)
    xs = {s: np.array([v[0] for t in by[s] for v in by[s][t]], float) for s in sets}
    ys = {s: np.array([v[1] for t in by[s] for v in by[s][t]], float) for s in sets}
    out = {}
    for s in sets:
        others = [o for o in (sorted(train) if train else sets) if o != s]
        b = _fit(np.concatenate([xs[o] for o in others]), np.concatenate([ys[o] for o in others]))
        logo = _ll(b, xs[s], ys[s])
        tids = list(by[s])
        order = np.arange(len(tids)) if rng is None else rng.permutation(len(tids))
        fold = {tids[i]: k % folds for k, i in enumerate(order)}
        losses, n = 0.0, 0
        for f in range(folds):
            tr = [v for t in tids if fold[t] != f for v in by[s][t]]
            te = [v for t in tids if fold[t] == f for v in by[s][t]]
            if not te or not tr:
                continue
            bt = _fit(np.array([v[0] for v in tr], float), np.array([v[1] for v in tr], float))
            losses += _ll(bt, np.array([v[0] for v in te]), np.array([v[1] for v in te], float)) * len(te)
            n += len(te)
        out[s] = (logo, losses / n)
    return out


def excess(recs, unit, folds=5):
    return _excess_from(_arrays(recs, unit), folds)


def invariance_test(recs, units, delta, boot=2000, seed=20261020, folds=5):
    """{unit: dict(excess, upper95, invariant, p_below_delta)}; the same bootstrap draws for every unit (joint).
    p_below_delta inverts the one-sided upper 95% bound against Delta (p < 0.05 exactly when upper95 < Delta)."""
    rng = np.random.default_rng(seed)
    base = {u: _arrays(recs, u) for u in units}
    sets = sorted(base[units[0]])
    draws = []
    for _ in range(boot):
        draws.append({s: rng.choice(list(base[units[0]][s]), size=len(base[units[0]][s]), replace=True) for s in sets})
    res = {}
    for u in units:
        parts = _per_set_parts(base[u], folds)
        est = float(np.mean([lo - wi for lo, wi in parts.values()]))
        reps = []
        for d in draws:
            by = {s: {f"{t}#{k}": base[u][s][t] for k, t in enumerate(d[s])} for s in sets}
            reps.append(_excess_from(by, folds, rng))
        ok = [x for x in reps if np.isfinite(x)]
        estimable = bool(np.isfinite(est) and len(ok) >= 0.9 * boot)
        up = float(np.percentile(ok, 95)) if estimable else float("inf")
        res[u] = dict(excess=est, upper95=up, invariant=estimable and up <= delta, boot=boot,
                      p_below_delta=law.p_bound(ok, delta, "below", 1) if estimable else float("nan"),  # Holm's p
                      logo=float(np.mean([lo for lo, _ in parts.values()])),    # rule 4's "LOGO log-loss" (D079 a)
                      within=float(np.mean([wi for _, wi in parts.values()])))
        if len(ok) < boot or not estimable:
            res[u].update(dropped=boot - len(ok), estimable=estimable)
    return res


def preferred_unit(inv, alpha=0.05):
    """H2's preferred unit (D079 a; hypotheses.md rule at the registered commit 3b8aa5e): "4. If several units are
    invariant, the preferred one has the lowest LOGO log-loss." "6. If none is invariant, report the ranking of sets in
    bits_sem". Invariant = supported in the registered H2 Holm family (the units of `inv`, one-sided bound against
    Delta; D078). Ties go to the first unit in `inv`'s order. Returns preferred (None under rule 6), unit (the unit the
    per-set report uses: bits_sem under rule 6), rule, invariant (the Holm-invariant units) and the Holm members."""
    from reach.analysis.s2 import families
    units = list(inv)
    mem = [dict(id=[u], kind="reject", p=inv[u]["p_below_delta"], untestable=None) for u in units]
    families.holm_family(mem, alpha)
    ok = [u for u, mm in zip(units, mem) if mm["supported"]]
    holm = {u: mm["p_holm"] for u, mm in zip(units, mem)}
    if not ok:
        return dict(preferred=None, unit="bits_sem", invariant=[], holm=holm,
                    rule="6: no unit invariant (Holm); no preferred unit (rule 6): per-set report in bits_sem")
    best = min(ok, key=lambda u: inv[u]["logo"])                # min keeps the first of equal values
    rule = "only invariant unit (Holm)" if len(ok) == 1 else "4: lowest LOGO log-loss among the Holm-invariant units"
    return dict(preferred=best, unit=best, invariant=ok, holm=holm, rule=rule)


def per_set_test(recs, unit, boot=2000, seed=20261020, folds=5, train=None):
    """{set: dict(excess, upper95[, dropped, estimable])} for one unit, targets resampled within every set (H2 deviating
    sets; H2s with `train` = the sets the law is fitted on). A set's excess is not estimable when its point estimate is
    undefined or more than 10% of its replicates are (separated fits); its upper95 is then inf, never within a limit."""
    rng = np.random.default_rng(seed)
    base = _arrays(recs, unit)
    est = _per_set_from(base, folds, train=train)
    reps = {s: [] for s in base}
    for _ in range(boot):
        by = {s: {f"{t}#{k}": base[s][t] for k, t in enumerate(rng.choice(list(base[s]), size=len(base[s])))}
              for s in base}
        for s, v in _per_set_from(by, folds, rng, train=train).items():
            reps[s].append(v)
    out = {}
    for s in base:
        ok = [x for x in reps[s] if np.isfinite(x)]
        estimable = bool(np.isfinite(est[s]) and len(ok) >= 0.9 * boot)
        out[s] = dict(excess=est[s], upper95=float(np.percentile(ok, 95)) if estimable else float("inf"))
        if len(ok) < boot or not estimable:
            out[s].update(dropped=boot - len(ok), estimable=estimable)
    return out

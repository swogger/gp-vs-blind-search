"""Pooled law of recovery (tool T16; brief §9.3): logit P = alpha - beta*s_u + gamma*log2(E/E0), per domain and
algorithm, with standard errors clustered by target (REQ-RULE-05). rate = gamma/beta is the reach gained per budget
doubling in unit u (nodes or bits per doubling)."""
import math
from collections import defaultdict

import numpy as np
import statsmodels.api as sm
from scipy.stats import norm

from reach.analysis.k50 import fit1, separated_columns


SET_MODEL, ONE_MODEL = "set intercepts, common slope", "one intercept"


def _design(records, cols, set_specific):
    """Columns `cols` after the intercept(s): one intercept, or one per set (set_specific: alpha_G, D079 c). Returns
    X, the intercept names (sets, or None) and the number of intercept columns."""
    sets = sorted({str(r.get("set")) for r in records})
    if set_specific and len(sets) > 1:
        D = np.array([[str(r.get("set")) == g for g in sets] for r in records], float)
        return np.column_stack([D] + cols), sets, len(sets)
    return np.column_stack([np.ones(len(records))] + cols), None, 1


def fit_law(records, unit="nodes", set_specific=False):
    """set_specific (the registered reading, D079 c): logit P = alpha_G - beta*s + gamma*log2(E/E0) with an intercept
    per set and common beta and gamma, so gamma/beta is the slope of K50 on log2 budget in every set. Otherwise one
    intercept for all sets (reported beside it). If separation leaves beta or gamma without a finite
    maximum-likelihood value (`separated_columns`; guardrail row 118), beta, gamma, rate and their standard errors are
    NaN (not estimable)."""
    s = np.array([r["units"][unit] if unit in r.get("units", {}) else r["K"] for r in records], float)
    lb = np.array([math.log2(r["budget"]) for r in records], float)
    y = np.array([r["y"] for r in records], float)
    groups = np.unique([r["target_id"] for r in records], return_inverse=True)[1]
    X, names, k = _design(records, [-s, lb], set_specific)
    model = sm.GLM(y, X, family=sm.families.Binomial())
    naive = model.fit()
    clus = model.fit(cov_type="cluster", cov_kwds=dict(groups=groups))
    beta, gamma = clus.params[k], clus.params[k + 1]
    se = [float(clus.bse[k]), float(naive.bse[k]), float(clus.bse[k + 1])]
    if separated_columns(X, y, [k, k + 1]):
        beta = gamma = float("nan")
        se = [float("nan")] * 3
    out = dict(beta=float(beta), gamma=float(gamma), rate=float(gamma / beta),
               se_beta_cluster=se[0], se_beta_naive=se[1], se_gamma_cluster=se[2], n=len(y), targets=int(groups.max() + 1),
               intercepts="set" if names else "one", model=SET_MODEL if names else ONE_MODEL)
    if names:
        out.update(alpha=float("nan"), alphas={g: float(a) for g, a in zip(names, clus.params[:k])})
    else:
        out.update(alpha=float(clus.params[0]))
    return out


def _size(r, unit):
    return r["units"][unit] if unit != "nodes" and unit in r.get("units", {}) else r["K"]


def _by_target(records):
    by = defaultdict(list)
    for r in records:
        by[r["target_id"]].append(r)
    return by


def _slope(rr, unit):
    """Logistic slope on size; NaN when the fit is degenerate (one outcome, one size, singular); -inf or +inf when
    separated (D088: the maximum-likelihood limit keeps its sign)."""
    x, y = np.array([_size(r, unit) for r in rr], float), np.array([r["y"] for r in rr], float)
    f = fit1(x, y)
    return float("nan") if f is None else float(f.b1)


def percentile(x, q):
    """np.percentile (linear interpolation) on the extended reals (D088): identical to np.percentile when every value
    is finite. Otherwise, at a position between two order statistics, an infinite neighbour is the limit of the
    interpolation (-inf below a finite value gives -inf, +inf above one gives +inf, two equal infinities give that
    infinity) and -inf next to +inf has none (NaN, never beyond any bound). Positions with two finite neighbours are
    numpy's own value (finite stand-ins for the infinities)."""
    x = np.asarray(x, float)
    if np.all(np.isfinite(x)):
        return np.percentile(x, q)
    r = np.sort(x)
    fin = r[np.isfinite(r)]
    lo_s, hi_s = (fin.min() - 1, fin.max() + 1) if len(fin) else (-1.0, 1.0)
    out = np.atleast_1d(np.percentile(np.where(r == -np.inf, lo_s, np.where(r == np.inf, hi_s, r)), q)).astype(float)
    n = len(r)
    for j, qq in enumerate(np.atleast_1d(np.asarray(q, float))):
        h = qq / 100 * (n - 1)
        i = min(int(np.floor(h)), n - 1)
        a, b, t = r[i], r[min(i + 1, n - 1)], h - i
        if t == 0 or a == b:
            v = a if np.isinf(a) else None
        elif np.isinf(a) and np.isinf(b):
            v = float("nan")
        else:
            v = a if a == -np.inf else b if b == np.inf else None
        if v is not None:
            out[j] = v
    return out


def slope_ci(records, unit="nodes", boot=1000, seed=20261020):
    """H1 in one cell (set x algorithm x budget): logistic slope of exact recovery on size, 95% CI from resampling
    targets. Degenerate replicates (one outcome, one size, singular) are dropped and counted (more than 10% dropped:
    not estimable). A separated cell or replicate keeps its sign (D088): slope -inf (successes at the smaller sizes) or
    +inf, kept in the percentile CI (`percentile`) and in the CI-inversion p. Only when separation occurs, the result
    also has `separated` ('falling' or 'rising': the point fit) and `separated_reps` (replicates)."""
    by = _by_target(records)
    ids, rng = list(by), np.random.default_rng(seed)
    reps = [_slope([r for j in rng.integers(0, len(ids), len(ids)) for r in by[ids[j]]], unit) for _ in range(boot)]
    ok = [x for x in reps if not np.isnan(x)]
    lo, hi = percentile(ok, [2.5, 97.5]) if ok else (float("nan"),) * 2
    slope = _slope(records, unit)
    out = dict(slope=slope, lo=float(lo), hi=float(hi), dropped=boot - len(ok),
               estimable=bool(len(ok) >= 0.9 * boot and not np.isnan(slope)), targets=len(ids), n=len(records),
               p_below0=p_ci_inversion(ok), saturated=len({r["y"] for r in records}) <= 1)
    if np.isinf(slope):
        out["separated"] = "falling" if slope < 0 else "rising"
    n_inf = sum(np.isinf(x) for x in ok)
    if n_inf:
        out["separated_reps"] = int(n_inf)
    return out


def separation_count(cells, boot):
    """D088's report line for a slope family: cells whose point fit is separated (falling, rising) and separated
    replicates among cells x boot (`slope_ci`'s `separated` and `separated_reps`)."""
    return dict(cells=len(cells), separated_cells=sum(bool(c.get("separated")) for c in cells),
                falling=sum(c.get("separated") == "falling" for c in cells),
                rising=sum(c.get("separated") == "rising" for c in cells), replicates=len(cells) * boot,
                separated_replicates=sum(c.get("separated_reps", 0) for c in cells))


def p_ci_inversion(reps):
    """Stage 1: "P-values for Holm come from the bootstrap by CI inversion"; "Directional effects: the 95% CI excludes 0
    in the predicted direction". The smallest alpha at which the two-sided 100(1 - alpha)% percentile CI, computed as
    slope_ci computes it (np.percentile, linear interpolation), lies wholly below 0. So p < 0.05 exactly when slope_ci's
    hi < 0. A CI wholly above 0 is not evidence (p = 1); p = 0 when every replicate is below 0; NaN without replicates."""
    return p_bound(reps, 0.0, "below", 2)


def p_bound(reps, null=0.0, direction="below", sided=2):
    """The registered bound inverted (REQ-ANA-29): the smallest alpha at which the percentile bound of level 1 - alpha
    lies strictly beyond `null` in `direction`. sided=2: the end of the two-sided 100(1 - alpha)% CI (percentile
    100(1 - alpha/2) for "below", 100 alpha/2 for "above"); sided=1: the one-sided bound (percentile 100(1 - alpha) or
    100 alpha), as `invariance` computes H2's "upper 95% bound". np.percentile's linear interpolation, so p < alpha
    exactly when that bound is beyond null. p = 0 when every replicate is beyond null; p = 1 when none is; with j
    replicates not beyond null p lies in [sided (j - 1) / (n - 1), sided j / (n - 1)] (capped at 1). NaN without
    replicates. Replicates may be infinite (D088), read as `percentile` reads them: a crossing next to -inf lies at
    the finite neighbour; one between -inf and +inf (no interpolated value) at -inf's position."""
    x = np.asarray(reps, float)
    r = np.sort(x - null if direction == "below" else null - x)
    n = len(r)
    if n == 0:
        return float("nan")
    k = int(np.searchsorted(r, 0.0, side="left"))          # replicates strictly beyond null
    if k == 0:
        return 1.0
    if k == n:
        return 0.0
    a, b = r[k - 1], r[k]                                   # a < 0 <= b: the interpolated quantile is < 0 before h0
    if a == -np.inf:                                        # D088: -inf next to b; before b the bound is -inf
        h0 = float(k) if np.isfinite(b) else float(k - 1)
    else:
        h0 = (k - 1) + (-a) / (b - a)                       # b = +inf gives k - 1
    return float(min(1.0, max(0.0, sided * (1.0 - h0 / (n - 1)))))


def _cell_key(c):
    return (c["set"], c["algorithm"], c.get("config"), c["budget"])


def holm_cells(cells, domain_of, alpha, declared=None):
    """The registered Holm step for H1 (Stage 1, Inference criteria) and H10a ("as for H1"): one family per domain,
    "each counting untestable members". `cells` are slope_ci results with set, algorithm, config (or none) and budget;
    `declared` the family's cell keys fixed before the runs (None: the cells present, for the DRYRUN). A cell enters
    with its p_below0 if estimable; saturated, not-estimable and missing cells are untestable members (in m, never
    supported). A cell is supported when its Holm-adjusted p is below alpha (strict, as "the CI excludes 0"). Adds
    p_holm, holm_supported and holm_status to each cell; returns per domain: m, counts, missing cells and
    all_testable_supported (every member other than a saturated cell supported). An undeclared cell is refused."""
    from reach.analysis.s2 import contrasts
    present = {_cell_key(c) for c in cells}
    if len(present) != len(cells):
        raise ValueError("Holm family: a cell appears twice")
    missing = []
    if declared is not None:
        declared = [tuple(k) for k in declared]
        extra = present - set(declared)
        if extra:
            raise ValueError(f"Holm family: cells not in the declared family: {sorted(map(str, extra))}")
        missing = [k for k in declared if k not in present]
    out = {}
    for d in sorted({domain_of(c["set"]) for c in cells} | {domain_of(k[0]) for k in missing}):
        fam = [c for c in cells if domain_of(c["set"]) == d]
        miss = [k for k in missing if domain_of(k[0]) == d]
        m = len(fam) + len(miss)
        testable = [bool(c["estimable"]) and np.isfinite(c["p_below0"]) for c in fam]
        adj = contrasts.holm([c["p_below0"] if t else None for c, t in zip(fam, testable)], m=m)
        for c, t, q in zip(fam, testable, adj):
            c["p_holm"] = q
            c["holm_supported"] = bool(t and q < alpha)
            c["holm_status"] = ("supported" if c["holm_supported"] else "not supported") if t else (
                "saturated (untestable)" if c.get("saturated") else "not estimable (not supported)")
        st = [c["holm_status"] for c in fam]
        sat = st.count("saturated (untestable)")
        sup = st.count("supported")
        out[d] = dict(alpha=alpha, m=m, family="declared" if declared is not None else "observed", supported=sup,
                      not_supported=st.count("not supported"), saturated=sat,
                      not_estimable=st.count("not estimable (not supported)"), missing=[list(k) for k in miss],
                      all_testable_supported=sup == m - sat)
    return out


def _k50(rr, unit):
    """K50 = -b0/b1; the gap midpoint when separated with successes at the smaller sizes (D088); NaN when degenerate
    or the slope is not negative (a rising curve, separated or not)."""
    x, y = np.array([_size(r, unit) for r in rr], float), np.array([r["y"] for r in rr], float)
    f = fit1(x, y)
    return float(f.k50) if f is not None and f.b1 < 0 else float("nan")


def _per_doubling(by, ids, unit, budgets):
    pts = []
    for b in budgets:
        k = _k50([r for t in ids for r in by[t] if r["budget"] == b], unit)
        if np.isfinite(k):
            pts.append((math.log2(b), k))
    if len(pts) < 3:
        return float("nan")
    x, y = np.array(pts).T
    return float(np.polyfit(x, y, 1)[0])


def reach_per_doubling(records, unit="nodes", budgets=(1, 2, 4, 8, 16), boot=1000, seed=20261020):
    """H4, H6a: least-squares slope of K50 (in `unit`) on log2 budget; 95% CI from resampling targets, the same draw
    for every budget (a target's runs at all budgets come from the same runs, as prefixes)."""
    by = _by_target([r for r in records if r["budget"] in budgets])
    ids, rng = list(by), np.random.default_rng(seed)
    reps = [_per_doubling(by, [ids[j] for j in rng.integers(0, len(ids), len(ids))], unit, budgets) for _ in range(boot)]
    ok = [x for x in reps if np.isfinite(x)]
    lo, hi = np.percentile(ok, [2.5, 97.5]) if ok else (float("nan"),) * 2
    return dict(rate=_per_doubling(by, ids, unit, budgets), lo=float(lo), hi=float(hi), dropped=boot - len(ok),
                estimable=len(ok) >= 0.9 * boot, targets=len(ids))


def curvature(records, unit="nodes", set_specific=False):
    """H4a: the pooled law with a quadratic term in log2 budget; returns its coefficient with a target-clustered 95%
    CI. Linear in log2 budget if the CI covers 0 (equivalence margin registered at G2). p_wald inverts that clustered CI
    (two-sided normal p; H4a's p for Holm, D078): p < 0.05 exactly when |q/se| > 1.95996, so it can differ from the
    1.96 bounds only for |q/se| in (1.95996, 1.96]. set_specific: intercepts per set (D079 c). Separation that leaves
    a size or budget coefficient without a finite maximum-likelihood value (guardrail row 118) gives NaN (no p)."""
    s = np.array([_size(r, unit) for r in records], float)
    lb = np.array([math.log2(r["budget"]) for r in records], float)
    y = np.array([r["y"] for r in records], float)
    groups = np.unique([r["target_id"] for r in records], return_inverse=True)[1]
    X, names, _ = _design(records, [-s, lb, lb ** 2], set_specific)
    res = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds=dict(groups=groups))
    q, se = float(res.params[-1]), float(res.bse[-1])
    if separated_columns(X, y, range(X.shape[1] - 3, X.shape[1])):
        q = se = float("nan")
    p = float(2 * norm.sf(abs(q / se))) if se > 0 and np.isfinite(q / se) else float("nan")
    return dict(quad=q, se=se, lo=q - 1.96 * se, hi=q + 1.96 * se, n=len(y), p_wald=p,
                model=SET_MODEL if names else ONE_MODEL)


def _irls(X, y, iters=100):
    """Maximum-likelihood logistic coefficients by Newton steps; None if not converged or not finite."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = np.clip(p * (1 - p), 1e-12, None)
        try:
            step = np.linalg.solve((X * W[:, None]).T @ X, X.T @ (y - p))
        except np.linalg.LinAlgError:
            return None
        b = b + step
        if not np.all(np.isfinite(b)):
            return None
        if np.max(np.abs(step)) < 1e-10:
            return b
    return None


def rate_ci(records, unit="bits_sem", boot=2000, seed=20261020, set_specific=False):
    """H4b and H6a (REQ-ANA-29): the pooled law's reach per budget doubling, rate = gamma/beta in `unit` (fit_law), with
    a target bootstrap, targets resampled within sets (Stage 1, Statistical models). A replicate is degenerate (dropped)
    when the fit does not converge or beta <= 0 (no size effect: no reach); more than 10% dropped: not estimable.
    Returns the estimate, the two-sided 95% CI (lo, hi), the one-sided upper 95% bound (upper95, as H2's) and
    p_below1 (H4b: upper 95% bound below 1, one-sided) and p_differs1 (H6a: differs from 1, two-sided).
    set_specific: an intercept per set with common beta and gamma (the registered reading, D079 c); every set is in
    every replicate (targets are resampled within sets), so no intercept column is ever empty."""
    fit = fit_law(records, unit, set_specific)
    by = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(records):
        by[r.get("set")][r["target_id"]].append(i)
    s = np.array([_size(r, unit) for r in records], float)
    X, _, k = _design(records, [-s, np.array([math.log2(r["budget"]) for r in records], float)], set_specific)
    y = np.array([r["y"] for r in records], float)
    rng = np.random.default_rng(seed)
    strata = [(list(t.values())) for _, t in sorted(by.items(), key=lambda kv: str(kv[0]))]
    reps = []
    for _ in range(boot):
        idx = np.concatenate([np.asarray(t[j]) for t in strata for j in rng.integers(0, len(t), len(t))])
        b = _irls(X[idx], y[idx])
        reps.append(b[k + 1] / b[k] if b is not None and b[k] > 0 else float("nan"))
    ok = [x for x in reps if np.isfinite(x)]
    lo, hi = np.percentile(ok, [2.5, 97.5]) if ok else (float("nan"),) * 2
    from reach.analysis.s2.families import p_differs
    return dict(rate=fit["rate"], lo=float(lo), hi=float(hi),
                upper95=float(np.percentile(ok, 95)) if ok else float("nan"), dropped=boot - len(ok),
                estimable=bool(len(ok) >= 0.9 * boot and np.isfinite(fit["rate"])), boot=boot, unit=unit,
                targets=sum(len(t) for t in strata),
                model=fit["model"],
                p_below1=p_bound(ok, 1.0, "below", 1), p_differs1=p_differs(ok, 1.0))


def slope_equality(band, budgets, margin, unit="bits_sem", boot=2000, seed=20261020):
    """H4c (REQ-ANA-49; reading D081 a). Filed: "per-set slopes (bits_sem per doubling) are equal within each domain
    within +-0.5 (real) or +-1.0 (Boolean)"; "Equivalence: the 95% CI lies within the margin". Per set, the slope of K50
    on log2 budget (`reach_per_doubling`'s estimator) on its budget-level records `band[g]` at `budgets[g]`; every pair
    of sets must differ by less than the margin: the two-sided 95% target-bootstrap CI of the pair's difference inside
    (-margin, margin) (targets resampled within sets, one generator for all sets). Pair p = the larger of the two
    one-sided inversions (as H4d); member p = the largest pair p (intersection-union). A pair with more than 10% of its
    replicates dropped (a set's slope undefined) or an undefined point estimate makes the member untestable."""
    sets = sorted(band)
    by = {g: _by_target([r for r in band[g] if r["budget"] in budgets[g]]) for g in sets}
    point = {g: _per_doubling(by[g], list(by[g]), unit, budgets[g]) for g in sets}
    rng = np.random.default_rng(seed)
    reps = {g: [] for g in sets}
    for _ in range(boot if len(sets) > 1 else 0):
        for g in sets:
            ids = list(by[g])
            reps[g].append(_per_doubling(by[g], [ids[j] for j in rng.integers(0, len(ids), len(ids))], unit,
                                         budgets[g]))
    out_sets = {}
    for g in sets:
        ok = [x for x in reps[g] if np.isfinite(x)]
        lo, hi = np.percentile(ok, [2.5, 97.5]) if ok else (float("nan"),) * 2
        out_sets[g] = dict(rate=point[g], lo=float(lo), hi=float(hi), budgets=list(budgets[g]))
    pairs = []
    for i, a in enumerate(sets):
        for b in sets[i + 1:]:
            d = [x - y for x, y in zip(reps[a], reps[b]) if np.isfinite(x) and np.isfinite(y)]
            lo, hi = np.percentile(d, [2.5, 97.5]) if d else (float("nan"),) * 2
            est = point[a] - point[b]
            ok = len(d) >= 0.9 * boot and bool(np.isfinite(est))
            p = max(p_bound(d, -margin, "above", 2), p_bound(d, margin, "below", 2)) if ok else None
            pairs.append(dict(sets=[a, b], diff=float(est), lo=float(lo), hi=float(hi), dropped=boot - len(d),
                              estimable=ok, p=p, holds=bool(ok and -margin < lo and hi < margin)))
    if len(sets) < 2:
        why = "fewer than 2 sets"
    else:
        bad = [" - ".join(pr["sets"]) for pr in pairs if not pr["estimable"]]
        why = ("not estimable: " + ", ".join(bad[:4]) + ("…" if len(bad) > 4 else "")) if bad else None
    return dict(unit=unit, margin=margin, sets=out_sets, pairs=pairs, boot=boot,
                p=None if why else max(pr["p"] for pr in pairs), untestable=why,
                holds=bool(why is None and all(pr["holds"] for pr in pairs)))

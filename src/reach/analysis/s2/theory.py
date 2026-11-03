"""Fit of the analytic model (tool T24; docs/design/theory_model.md; REQ-ANA-08).

Data: one record per run with the target's size b (bits_sem by default), gen_exact (generation of the first exact match,
or None) and gens_reached / generations (censoring). With T in generations (success by L generations iff T <= L):
- weibull (the model): cumulative hazard H(T) = exp(c_set - alpha*b*ln2) * max(T - t0, 0)^kappa, so that
  cloglog P(T <= L) = kappa*ln(L - t0) - alpha*b*ln2 + c_set (A1, A2);
- lognormal (registered alternative, Barrero et al.): ln(T - t0) ~ Normal(m_set + m_b*b, sigma).
Discrete time: an event at generation g contributes S(g) - S(g + 1); a run without success contributes S(gens_reached).
Predictions: bits per budget doubling = kappa/alpha (T2); best run length L* = t0/(1 - kappa) for kappa < 1 (T5).
- Targets differ (A3): a target effect u ~ N(0, sigma) adds to the log hazard (Weibull) or subtracts from the mean log
  time (lognormal), shared by all runs on the target and integrated out by Gauss-Hermite quadrature. Without it kappa
  and alpha are biased low (a mixture of easy and hard targets looks like a falling hazard), so restarts would look
  useful when they are not (guardrail row 48). It is on whenever some target has more than one run.
- A hit at generation 0 is the initial random population, not evolution: such runs are left out of the run-time fit
  and counted (`gen0_hits`), and every other run is conditioned on no hit at generation 0 (left truncation), so the
  fit stays unbiased when the process itself can hit before generation 1. Otherwise they force t0 below one.
  Evolution can first hit at generation 1 (T in (1, 2]), so t0 >= 1; below that t0 is not identified from
  generation counts.
"""
import math

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

LN2 = math.log(2)


GH_X, GH_W = np.polynomial.hermite.hermgauss(20)
U_NODES, LOG_W = math.sqrt(2) * GH_X, np.log(GH_W / math.sqrt(math.pi))      # E f(u), u ~ N(0, 1)


def _arrays(runs, unit, sets=None):
    sets = sorted({r.get("set") for r in runs}) if sets is None else list(sets)
    sid = np.array([sets.index(r.get("set")) for r in runs], dtype=int)
    b = np.array([r["b"] if "b" in r else r["units"][unit] for r in runs], float)
    ev = np.array([r.get("gen_exact") is not None for r in runs])
    g = np.array([r["gen_exact"] if r.get("gen_exact") is not None else r.get("gens_reached", r.get("generations"))
                  for r in runs], float)
    tid = np.unique([str(r.get("target_id", i)) for i, r in enumerate(runs)], return_inverse=True)[1]
    return sets, sid, b, ev, g, tid


def _logS(T, eta, kappa, t0):
    return -np.exp(np.minimum(eta, 700.0)) * np.maximum(T - t0, 0.0) ** kappa


def _ll_runs(ls0, ls1, ev, ls_first):
    # event: log(S(g) - S(g+1)) = log S(g) + log(1 - exp(log S(g+1) - log S(g))); every kept run is conditioned on
    # no hit at generation 0 (left truncation at 1), since runs with such a hit are left out
    ll_ev = ls0 + np.log(-np.expm1(np.minimum(ls1 - ls0, -1e-300)))
    return np.where(ev, ll_ev, ls0) - ls_first


def _marginal(ll_fn, shift, tid, n_tid, sigma_u):
    """-log likelihood with the target effect integrated out; ll_fn(offset) gives per-run log-likelihoods."""
    if sigma_u is None:
        return -ll_fn(0.0).sum()
    per = np.stack([np.bincount(tid, ll_fn(shift * sigma_u * x), minlength=n_tid) for x in U_NODES])   # nodes x targets
    m = per.max(axis=0)
    return -(m + np.log(np.exp(per - m[None, :] + LOG_W[:, None]).sum(axis=0))).sum()


def _nll_weibull(p, sid, b, ev, g, n_sets, tid, n_tid, frailty):
    kappa, alpha, t0 = math.exp(p[0]), p[1], math.exp(p[2])
    eta = p[3:3 + n_sets][sid] - alpha * b * LN2
    ll = lambda off: _ll_runs(_logS(g, eta + off, kappa, t0), _logS(g + 1, eta + off, kappa, t0), ev,   # noqa
                              _logS(1.0, eta + off, kappa, t0))
    return _marginal(ll, 1.0, tid, n_tid, math.exp(p[-1]) if frailty else None)


def _nll_lognormal(p, sid, b, ev, g, n_sets, tid, n_tid, frailty):
    sigma, mb, t0 = math.exp(p[0]), p[1], math.exp(p[2])
    mu = p[3:3 + n_sets][sid] + mb * b

    def ll(off):
        logS = lambda T: norm.logsf((np.log(np.maximum(T - t0, 1e-12)) - mu - off) / sigma)   # noqa
        return _ll_runs(logS(g), logS(g + 1), ev, logS(np.ones_like(g)))
    return _marginal(ll, -1.0, tid, n_tid, math.exp(p[-1]) if frailty else None)


def _check(runs):
    if any(r.get("algorithm") == "CGP" for r in runs):
        raise ValueError("theory.fit: CGP rows have generations of 4-5 evaluations, not tree GP generations; "
                         "fit them on an explicit evaluation axis instead")


def _kept(runs):
    """Generation-0 hits are left out (initial population); so are rows killed without a reached generation, whose
    time is unknown (they must not count as surviving the whole run; review, row 56)."""
    return [r for r in runs if r.get("gen_exact") != 0 and (r.get("gen_exact") is not None or "gens_reached" in r)]


def fit(runs, family="weibull", unit="bits_sem", frailty="auto", x0=None):
    """frailty: True, False, or "auto" (on when some target has more than one run). Tree GP rows only: CGP rows
    count generations of 4-5 evaluations, not of a population, so they are refused (guardrail row 87). x0: one start
    from these parameters (a fit's "x"; the bootstrap's warm start) instead of the three default starts."""
    _check(runs)
    n0 = sum(1 for r in runs if r.get("gen_exact") == 0)
    runs = _kept(runs)
    sets, sid, b, ev, g, tid = _arrays(runs, unit)
    n, n_tid = len(sets), int(tid.max()) + 1
    if frailty == "auto":
        frailty = n_tid < len(runs)
    nll = _nll_weibull if family == "weibull" else _nll_lognormal
    bounds = [(-3.0, 2.0), (-5.0, 5.0), (0.0, math.log(200.0))] + [(-60.0, 60.0)] * n      # t0 >= 1 generation
    if frailty:
        bounds = bounds + [(math.log(1e-3), math.log(5.0))]
    starts = []
    for t0_start in (1.0, 3.0, 10.0):                                 # a few starts: t0 is weakly identified
        st = np.r_[0.0, 0.5 if family == "weibull" else 0.2, math.log(t0_start), np.zeros(n)]
        starts.append(np.r_[st, math.log(0.5)] if frailty else st)
    if x0 is not None:
        starts = [np.clip(np.asarray(x0, float), [lo for lo, _ in bounds], [hi for _, hi in bounds])]
        if len(starts[0]) != len(bounds):
            raise ValueError("theory.fit: x0 does not match the model's parameters")
    best = None
    for st in starts:
        res = minimize(nll, st, args=(sid, b, ev, g, n, tid, n_tid, frailty), method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    p = best.x
    out = dict(family=family, unit=unit, sets=sets, loglik=float(-best.fun), n=len(b), events=int(ev.sum()),
               t0=float(math.exp(p[2])), set_terms=dict(zip(sets, map(float, p[3:3 + n]))), converged=bool(best.success),
               frailty=bool(frailty), sigma_target=float(math.exp(p[-1])) if frailty else None, gen0_hits=n0,
               targets=n_tid, x=[float(v) for v in p])
    if family == "weibull":
        out.update(kappa=float(math.exp(p[0])), alpha=float(p[1]))
        out["bits_per_doubling"] = out["kappa"] / out["alpha"] if out["alpha"] > 0 else float("nan")
    else:
        out.update(sigma=float(math.exp(p[0])), m_b=float(p[1]))
    return out


def best_run_length(f):
    """L* = t0 / (1 - kappa) for kappa < 1 (T5); None when restarts cannot help (kappa >= 1)."""
    return f["t0"] / (1 - f["kappa"]) if f["kappa"] < 1 else None


def loglik(f, runs):
    """Log-likelihood of `runs` under the fitted model `f` (its family, parameters, sets and target effect), with the
    fit's rules: generation-0 hits and rows without a reached generation left out, the target effect integrated out
    per target (so a new target is scored by its marginal likelihood). Runs of a set the fit does not know are refused."""
    _check(runs)
    runs = _kept(runs)
    if not runs:
        return 0.0
    unknown = {r.get("set") for r in runs} - set(f["sets"])
    if unknown:
        raise ValueError(f"theory.loglik: sets not in the fit: {sorted(map(str, unknown))}")
    sets, sid, b, ev, g, tid = _arrays(runs, f["unit"], f["sets"])
    nll = _nll_weibull if f["family"] == "weibull" else _nll_lognormal
    return float(-nll(np.asarray(f["x"], float), sid, b, ev, g, len(sets), tid, int(tid.max()) + 1, f["frailty"]))


def heldout_loglik(train, test, family="weibull", unit="bits_sem", frailty="auto"):
    """Stage 1: "compared ... by held-out log-likelihood" (D080 d): fit on `train`, score `test`. Refuses a test run on
    a (set, target) used in the fit: that would be an in-sample comparison."""
    seen = {(r.get("set"), str(r.get("target_id"))) for r in train}
    both = sorted({(str(r.get("set")), str(r.get("target_id"))) for r in test if (r.get("set"), str(r.get("target_id")))
                   in seen})
    if both:
        raise ValueError(f"theory.heldout_loglik: {len(both)} test targets were used in the fit (in-sample), e.g. "
                         f"{both[0]}")
    return loglik(fit(train, family, unit, frailty), test)


def compare_families(runs, folds=5, unit="bits_sem", families=("weibull", "lognormal")):
    """The registered comparison (D080 d): per set, targets in folds (k % folds in order of first appearance, as H2's
    within-set log-loss); each fold scored by models fitted on the other folds of the same set; held-out
    log-likelihoods summed over folds and sets. Returns the sums, the preferred family (the higher) and the counts."""
    by = {}
    for r in runs:
        by.setdefault(r.get("set"), {}).setdefault(str(r.get("target_id")), []).append(r)
    total = {fam: 0.0 for fam in families}
    n_folds = 0
    for g in sorted(by, key=str):
        tids = list(by[g])
        fold = {t: k % folds for k, t in enumerate(tids)}
        for k in range(min(folds, len(tids))):
            test = [r for t in tids if fold[t] == k for r in by[g][t]]
            train = [r for t in tids if fold[t] != k for r in by[g][t]]
            if not _kept(test) or not _kept(train):
                continue
            n_folds += 1
            for fam in families:
                total[fam] += heldout_loglik(train, test, fam, unit)
    pref = max(families, key=lambda fam: total[fam]) if n_folds else None
    return dict(heldout_loglik=total, preferred=pref, folds=folds, scored_folds=n_folds, sets=sorted(map(str, by)),
                runs=len(_kept(runs)), difference=total[families[0]] - total[families[1]] if n_folds else float("nan"),
                note="held-out log-likelihood: 5-fold cross-validation over targets within each set (D080 d)")

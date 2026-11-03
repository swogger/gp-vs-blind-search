"""T24 analytic-model fit (written before the code). REQ-ANA-08: recovers known parameters from simulated runs."""
import math

import numpy as np
import pytest

from reach.analysis.s2 import theory as M


def simulate(kappa, alpha, c, t0, bits, n_per, max_gen, seed=1):
    """Weibull run times with H(T) = exp(c - alpha*b*ln2) * (T - t0)^kappa; event generation g = ceil(T) - 1."""
    rng = np.random.default_rng(seed)
    runs = []
    for b in bits:
        lam = math.exp(c - alpha * b * math.log(2))
        u = rng.random(n_per)
        T = t0 + (-np.log(u) / lam) ** (1 / kappa)                  # S(T) = exp(-lam (T - t0)^kappa)
        for t in T:
            g = int(math.ceil(t)) - 1
            runs.append(dict(b=b, set="S", gen_exact=g if g < max_gen else None, gens_reached=min(g, max_gen),
                             generations=max_gen))
    return runs


@pytest.mark.parametrize("kappa,alpha", [(0.6, 0.8), (1.0, 1.0), (0.4, 0.5)])
def test_weibull_fit_recovers_known_parameters(kappa, alpha):
    runs = simulate(kappa, alpha, c=2.0, t0=4.0, bits=np.arange(6, 22, 2), n_per=400, max_gen=800)
    f = M.fit(runs, family="weibull")
    assert f["kappa"] == pytest.approx(kappa, abs=0.06)
    assert f["alpha"] == pytest.approx(alpha, abs=0.06)
    assert f["t0"] == pytest.approx(4.0, abs=2.0)
    assert f["bits_per_doubling"] == pytest.approx(kappa / alpha, rel=0.12)


def test_best_run_length_prediction():
    f = dict(kappa=0.5, t0=10.0)
    assert M.best_run_length(f) == pytest.approx(20.0)              # t0 / (1 - kappa)
    assert M.best_run_length(dict(kappa=1.1, t0=10.0)) is None      # no restarts gain when kappa >= 1


def test_lognormal_alternative_fits_and_compares():
    runs = simulate(0.6, 0.8, c=2.0, t0=4.0, bits=np.arange(6, 22, 2), n_per=300, max_gen=800)
    w, ln = M.fit(runs, family="weibull"), M.fit(runs, family="lognormal")
    assert w["loglik"] > ln["loglik"]                                 # Weibull data: Weibull fits better


def test_lognormal_data_prefer_the_lognormal_model():
    """Negative control for the comparison: lognormal run times must not be taken for Weibull."""
    rng = np.random.default_rng(3)
    runs = []
    for b in np.arange(6, 22, 2):
        T = 4.0 + np.exp(rng.normal(1.0 + 0.25 * b, 0.8, 300))
        for t in T:
            gg = int(math.ceil(t)) - 1
            runs.append(dict(b=b, set="S", gen_exact=gg if gg < 800 else None, gens_reached=min(gg, 800), generations=800))
    assert M.fit(runs, family="lognormal")["loglik"] > M.fit(runs, family="weibull")["loglik"]


def simulate_targets(kappa, alpha, c, t0, sd, bits, targets=20, seeds=5, max_gen=800, p_gen0=0.0, seed=1):
    """Runs grouped by target, with a target effect u ~ N(0, sd) on the log hazard (A3) and, optionally, hits by the
    initial population at generation 0 with probability p_gen0 for targets of 6 bits or fewer."""
    rng = np.random.default_rng(seed)
    runs = []
    for b in bits:
        for t in range(targets):
            u = rng.normal(0, sd)
            lam = math.exp(c + u - alpha * b * math.log(2))
            for s in range(seeds):
                if b <= 6 and rng.random() < p_gen0:
                    g = 0
                else:
                    g = int(math.ceil(t0 + (-math.log(rng.random()) / lam) ** (1 / kappa))) - 1
                runs.append(dict(b=float(b), set="S", target_id=f"b{b}_t{t}", gen_exact=g if g < max_gen else None,
                                 gens_reached=min(g, max_gen), generations=max_gen))
    return runs


def test_target_frailty_removes_the_bias_in_kappa():
    """A3: targets differ. Without the random effect, kappa is biased low and restarts look useful when they are not
    (negative control); with it, kappa = 1 is recovered and no best run length is predicted."""
    runs = simulate_targets(1.0, 0.5, c=1.0, t0=10.0, sd=1.5, bits=np.arange(4, 26, 2))
    naive = M.fit(runs, family="weibull", frailty=False)
    assert naive["kappa"] < 0.85
    f = M.fit(runs, family="weibull")
    assert f["frailty"] and f["kappa"] == pytest.approx(1.0, abs=0.12) and f["alpha"] == pytest.approx(0.5, abs=0.1)
    assert f["sigma_target"] == pytest.approx(1.5, abs=0.45)
    lstar = M.best_run_length(f)
    assert lstar is None or lstar > 100


def test_generation_zero_hits_do_not_pin_the_warm_up():
    """A hit at generation 0 is the initial population, not evolution: it is left out of the run-time fit and counted."""
    runs = simulate_targets(0.7, 0.5, c=1.0, t0=10.0, sd=0.0, bits=np.arange(4, 26, 2), p_gen0=0.3)
    f = M.fit(runs, family="weibull")
    assert f["gen0_hits"] > 0 and f["t0"] == pytest.approx(10.0, abs=3.0) and f["kappa"] == pytest.approx(0.7, abs=0.1)


def test_killed_runs_without_a_reached_generation_are_left_out():
    """Review finding 7: a row killed by the queue has no gens_reached; it must not count as surviving its full run."""
    runs = simulate(0.6, 0.8, c=2.0, t0=4.0, bits=np.arange(6, 22, 2), n_per=200, max_gen=800)
    killed = [dict(b=6.0, set="S", gen_exact=None, generations=800) for _ in range(300)]
    f, g = M.fit(runs, family="weibull"), M.fit(runs + killed, family="weibull")
    assert g["n"] == f["n"] and g["alpha"] == pytest.approx(f["alpha"], abs=1e-6)


def test_cgp_rows_are_refused():
    """Guardrail row 87 (independent review, 2026-10-26): the fit's time axis is tree GP generations; a CGP generation
    is 4 or 5 evaluations, so a CGP row in the fit would put another time scale into kappa, t0 and the set terms."""
    runs = simulate(0.6, 0.8, c=2.0, t0=4.0, bits=np.arange(6, 22, 2), n_per=50, max_gen=800)
    M.fit(runs, family="weibull")                                         # tree GP rows (no algorithm field): fitted
    cgp_row = dict(runs[0], algorithm="CGP", evals_per_gen=4)
    for family in ("weibull", "lognormal"):
        with pytest.raises(ValueError, match="CGP"):
            M.fit(runs + [cgp_row], family=family)

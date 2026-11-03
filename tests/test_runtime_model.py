"""The run-time model's registered tests (REQ-ANA-43 to 47; D080 d-h; guardrail row 106), written before the code.
Synthetic data only.

Stage 1 (OSF a3spg), as filed: "Run-time model form: the Weibull model (the theory) is compared with a lognormal
alternative by held-out log-likelihood; either result is reported." "H4d: the run-time model's kappa/alpha matches the
measured slope within the same margins." "H4e: restarts help (best split better than one run) exactly in the cells
where kappa-hat < 1, in at least 75% of cells." "H4f: the measured best run length lies within the CI of
t0/(1 - kappa-hat)."
"""
import math

import numpy as np
import pytest

from reach.analysis.s2 import runtime as RT
from reach.analysis.s2 import theory as M
from reach.analysis.s2.load import tidy


def sim_set(g, kappa, alpha, c, t0, sizes, targets=8, seeds=5, gens=3200, alloc_seeds=0, sd=0.5, seed=1,
            bits_per_node=2.0, lognormal=False):
    """Regime A rows of one set: Weibull run times with a target effect (A1-A3), bits_sem = bits_per_node x K; plus
    `alloc_seeds` more runs per target at E0 (200 generations) for the best split. lognormal: ln(T - t0) normal."""
    rng = np.random.default_rng(seed)
    rows, al = [], []
    for K in sizes:
        b = bits_per_node * K
        for t in range(targets):
            u = rng.normal(0, sd)
            for s in range(seeds + alloc_seeds):
                if lognormal:
                    T = t0 + math.exp(rng.normal(-1.0 + 0.3 * b - u, 0.9))
                else:
                    lam = math.exp(c + u - alpha * b * math.log(2))
                    T = t0 + (-math.log(rng.random()) / lam) ** (1 / kappa)
                ge = int(math.ceil(T)) - 1
                n = gens if s < seeds else 200
                ok = ge < n
                r = dict(key=f"{g}|{K}|{t}|{s}", set=g, target_id=f"{g}_K{K}_{t}", seed=s, K=K, K_construction=K,
                         K_proven=K, algorithm="A", config="base", pop_size=100,
                         units=dict(nodes=float(K), bits_uniform=3.0 * K, bits_syn=2.5 * K, bits_sem=b),
                         generations=n, gen_exact=ge if ok else None, gens_reached=ge if ok else n,
                         terminal_state="converged" if ok else "budget_exhausted", cpu_s=1.0)
                (rows if s < seeds else al).append(r)
    return rows, al


SIZES = range(4, 14)          # bits_sem 8-26: K50 inside the range at 1-16 x E0 for the parameters below


def _model(g, kappa, alpha, c, t0, boot, alloc_seeds=10, seed=1):
    rows, al = sim_set(g, kappa, alpha, c, t0, SIZES, alloc_seeds=alloc_seeds, seed=seed)
    runs, _ = RT.fit_runs(rows)
    band = tidy(rows, (1, 2, 4, 8, 16))
    return RT.set_model(runs, band, rows + al if alloc_seeds else None, (1, 2, 4, 8, 16), boot=boot)


# ------------------------------------------------------------------ Weibull against lognormal (held-out)


def test_heldout_loglik_refuses_targets_used_in_the_fit():
    rows, _ = sim_set("S", 0.6, 0.5, 2.0, 2.0, SIZES, targets=4)
    runs, _ = RT.fit_runs(rows)
    train = [r for r in runs if not r["target_id"].endswith("_0")]
    test = [r for r in runs if r["target_id"].endswith("_0")]
    assert np.isfinite(M.heldout_loglik(train, test))
    with pytest.raises(ValueError, match="in-sample"):
        M.heldout_loglik(train, test + train[:1])
    with pytest.raises(ValueError, match="in-sample"):
        M.heldout_loglik(runs, runs)


def test_loglik_reproduces_the_fit_loglik():
    rows, _ = sim_set("S", 0.6, 0.5, 2.0, 2.0, SIZES, targets=4)
    runs, _ = RT.fit_runs(rows)
    for fam in ("weibull", "lognormal"):
        f = M.fit(runs, fam)
        assert M.loglik(f, runs) == pytest.approx(f["loglik"], rel=1e-9)
        g = M.fit(runs, fam, x0=f["x"])                                     # warm start: the same optimum
        assert g["loglik"] == pytest.approx(f["loglik"], abs=1e-3)
    with pytest.raises(ValueError, match="sets not in the fit"):
        M.loglik(f, [dict(runs[0], set="OTHER")])


def test_heldout_comparison_picks_the_true_family_in_both_directions():
    """Positive and negative control: Weibull run times prefer the Weibull model out of sample, lognormal ones the
    lognormal model; the folds hold whole targets."""
    w, _ = sim_set("S", 0.6, 0.5, 2.0, 3.0, SIZES, targets=4, seed=4)
    ln, _ = sim_set("S", 0.6, 0.5, 2.0, 3.0, SIZES, targets=4, seed=5, lognormal=True)
    cw = M.compare_families(RT.fit_runs(w)[0])
    cl = M.compare_families(RT.fit_runs(ln)[0])
    assert cw["preferred"] == "weibull" and cw["difference"] > 0 and cw["scored_folds"] == 5
    assert cl["preferred"] == "lognormal" and cl["difference"] < 0


def test_fit_runs_apply_the_censoring_rule():
    base = dict(set="S", target_id="t", K=5, units=dict(bits_sem=10.0), generations=3200)
    rows = [dict(base, gen_exact=40, gens_reached=40, terminal_state="converged"),
            dict(base, gen_exact=None, gens_reached=900, terminal_state="timeout"),       # CPU stop, no hit
            dict(base, gen_exact=None, terminal_state="timeout"),                         # killed, no generation
            dict(base, terminal_state="error")]
    runs, n = RT.fit_runs(rows)
    assert len(runs) == 2 and runs[1]["gens_reached"] == 3200 and n["cpu_failure"] == 1        # failure (primary)
    assert n["no_reached_generation"] == 1 and n["error"] == 1
    sens, m = RT.fit_runs(rows, censored_as_failure=False)
    assert len(sens) == 1 and m["cpu_left_out"] == 1                                            # left out


# ------------------------------------------------------------------ H4d


def _fake(diff_reps, point=0.1, kappa=0.5, lstar_reps=(), cells=None, help_reps=None, boot=None, t0=10.0,
          kappa_reps=None, t0_reps=None):
    boot = boot or len(diff_reps)
    return dict(point=dict(diff=point, ratio=1.0 + point, slope=1.0, kappa=kappa, lstar=10.0, t0=t0), boot=boot,
                reps=dict(diff=list(diff_reps), kappa=list(kappa_reps) if kappa_reps is not None else [kappa] * boot,
                          t0=list(t0_reps) if t0_reps is not None else [t0] * boot,
                          lstar=list(lstar_reps) or [10.0] * boot, help=help_reps or {}), cells=cells or {})


def test_h4d_equivalence_p_inverts_the_two_sided_ci_within_the_margin():
    """Known answer: replicates evenly spaced on [-0.8, 0.2] (101 values): the two-sided CI's upper end is below 0.5
    always; its lower end is above -0.5 from alpha = 0.6 (2.5% and 97.5% at alpha 0.05: -0.775, 0.175)."""
    reps = np.linspace(-0.8, 0.2, 101)
    d = RT.h4d_set(_fake(reps), 0.5)
    assert d["p"] == pytest.approx(0.6, abs=1e-9) and not d["holds"]
    assert d["lo"] == pytest.approx(-0.775) and d["hi"] == pytest.approx(0.175)
    e = RT.h4d_set(_fake(reps / 2), 0.5)                                    # [-0.4, 0.1]: inside
    assert e["p"] == 0.0 and e["holds"]
    nan = RT.h4d_set(_fake(list(reps[:85]) + [float("nan")] * 16), 0.5)    # 16% degenerate: not estimable
    assert not nan["estimable"] and not nan["holds"] and nan["dropped"] == 16
    t = RT.domain_tests({"A": _fake(reps / 2), "B": _fake(reps)}, 0.5, 0.75)["H4d"]
    assert t["p"] == pytest.approx(0.6) and not t["holds"]                 # conjunction: the larger p
    u = RT.domain_tests({"A": _fake(reps / 2), "B": _fake(list(reps[:85]) + [float("nan")] * 16)}, 0.5, 0.75)["H4d"]
    assert u["p"] is None and u["untestable"] == "not estimable in B"


@pytest.mark.parametrize("kappa,alpha", [(0.6, 0.5), (1.2, 0.8)])
def test_h4d_recovers_kappa_over_alpha_against_the_measured_slope(kappa, alpha):
    """Positive control: Weibull data give kappa/alpha (bits_sem per doubling) and the same measured slope; negative
    control: the same data fail a margin of 0.01."""
    m = _model("S", kappa, alpha, 2.0, 2.0, boot=20, alloc_seeds=0)
    assert m["point"]["ratio"] == pytest.approx(kappa / alpha, rel=0.15)
    assert m["point"]["slope"] == pytest.approx(kappa / alpha, rel=0.2) and abs(m["point"]["diff"]) < 0.3
    assert RT.h4d_set(m, 1.0)["holds"] and not RT.h4d_set(m, 0.01)["holds"]
    assert len(m["reps"]["diff"]) == 20 and m["cells"] == {}
    assert np.std(m["reps"]["slope"]) > 0 and np.std(m["reps"]["kappa"]) > 0          # both resampled


# ------------------------------------------------------------------ H4e


def test_h4e_share_and_p_known_answer():
    """Two sets x 2 cells: 3 of 4 agree (an undefined cell does not); at least 0.75: a replicate share of exactly
    0.75 meets it; a replicate with a failed fit is dropped."""
    cells = {1: dict(help=True, length=25), 2: dict(help=None, length=None)}
    a = _fake([0.0] * 10, kappa=0.5, cells=cells, help_reps={1: [True] * 10, 2: [True] * 10})
    b = _fake([0.0] * 10, kappa=1.5, cells={1: dict(help=False, length=200), 2: dict(help=False, length=400)},
              help_reps={1: [False] * 10, 2: [True] * 5 + [False] * 5})
    e = RT.domain_tests({"A": a, "B": b}, 0.5, 0.75)["H4e"]
    assert e["share"] == 0.75 and e["holds_point"] and len(e["cells"]) == 4      # "at least": 0.75 meets it
    assert e["p"] == pytest.approx(0.0)                       # replicate shares 0.75 and 1.0: all meet 0.75
    b["reps"]["kappa"] = [float("nan")] * 2 + [1.5] * 8
    e2 = RT.domain_tests({"A": a, "B": b}, 0.5, 0.75)["H4e"]
    assert e2["dropped"] == 2 and not e2["estimable"] and e2["untestable"] == "not estimable"
    none = RT.domain_tests({"A": _fake([0.0] * 10)}, 0.5, 0.75)["H4e"]
    assert none["untestable"] == "no allocation cells" and none["p"] is None


def test_h4e_restarts_help_only_when_kappa_below_one():
    """Data: a decelerating search (kappa 0.5, short warm-up) gains from restarts at every budget; an accelerating one
    (kappa 1.6) loses; the cells agree with kappa-hat < 1 in both (positive control), and a rule read the other way
    would agree in none (negative control)."""
    dec = _model("D", 0.5, 0.4, 1.0, 1.5, boot=5, seed=2)
    acc = _model("A", 1.6, 1.0, -2.0, 1.5, boot=5, seed=3)
    assert dec["point"]["kappa"] < 1 < acc["point"]["kappa"]
    for b in (1, 4, 16):
        assert dec["cells"][b]["help"] is True and dec["cells"][b]["length"] < 200 * b
        assert acc["cells"][b]["help"] is False and acc["cells"][b]["length"] == 200 * b
    e = RT.domain_tests({"D": dec, "A": acc}, 1.0, 0.75)["H4e"]
    assert e["share"] == 1.0 and e["holds_point"]
    inverted = [c["help"] == (c["kappa"] > 1) for c in e["cells"]]
    assert not any(inverted)
    f = RT.domain_tests({"D": dec, "A": acc}, 1.0, 0.75)["H4f"]           # H4f: kappa-hat > 1 -> the whole budget
    for c in f["cells"]:
        if c["set"] == "A":
            assert c["model_length"] == c["length"] == 200 * c["budget"] and c["p"] == 1.0 and c["among"]


def test_one_run_at_e0_is_the_200_generation_split(monkeypatch):
    """Guardrail row 108 (CI, Linux): at E0 the one run is the split of length 200; computed by a second formula, its
    K50 differed from the split's in the last bits, so "restarts help" was decided by float noise. Planted: a one-run
    K50 smaller by 1e-12 must not make restarts help."""
    rows, al = sim_set("A", 1.6, 1.0, -2.0, 1.5, SIZES, alloc_seeds=10, seed=3)
    true = RT.one_run_k50
    monkeypatch.setattr(RT, "one_run_k50", lambda *a, **k: true(*a, **k) - 1e-12)
    c = RT.cell(rows + al, 1)
    assert c["best_split_length"] == 200 and c["help"] is False and c["length"] == 200
    assert c["one_run_k50"] == c["best_split_k50"]


# ------------------------------------------------------------------ H4f


GRID = (25, 50, 100, 200)        # the split lengths at E0, where the whole budget (200 generations) is one of them


def _cell(length, whole=200, grid=GRID):
    return dict(help=True, length=length, grid=sorted(set(grid) | {whole}), whole_budget_length=whole)


def test_model_length_is_the_grid_argmax_at_the_fixed_budget_not_lstar():
    """T5 (theory_model.md at 3b8aa5e): "the run length that maximises success at a fixed total budget"; "r = E/L runs
    give total hazard ∝ (E/L)·(L − t₀)^κ". On the measured grid the model's best length is the argmax of
    (B/L)(L - t0)^kappa: kappa 0.6, t0 30 gives L* = 75, which is not a measured length; the model picks 100 (values
    0, 24.1, 25.6, 21.8 at 25-200 for B = 200), also at B = 3200. kappa >= 1: the whole budget (T4). No candidate
    longer than t0: none."""
    assert RT.model_length(0.6, 30.0, GRID, 200) == 100
    assert RT.model_length(0.6, 30.0, GRID + (3200,), 3200) == 100
    assert RT.model_length(0.9, 50.0, GRID, 200) == 200 and RT.model_length(0.9, 50.0, GRID + (3200,), 3200) == 200
    assert RT.model_length(0.6, 20.0, GRID, 200) == 50 and RT.model_length(0.6, 5.0, GRID, 200) == 25
    assert RT.model_length(1.2, 5.0, GRID + (3200,), 3200) == 3200 and RT.model_length(1.0, 5.0, GRID, 200) == 200
    assert RT.model_length(0.6, 200.0, GRID, 200) is None and RT.model_length(float("nan"), 5.0, GRID, 200) is None


def test_h4f_holds_when_the_theory_is_right_and_lstar_exceeds_the_budget():
    """Review (2026-10-27): kappa 0.9, t0 50 gives L* = 500 > B = 200, so one run of the whole budget is the model's
    best length; data that agree (measured best = B) must hold. The unclipped CI of L* (about [455, 980]) excluded
    200 with p = 0 (the old reading)."""
    rng = np.random.default_rng(0)
    ks = np.clip(rng.normal(0.9, 0.01, 2000), 0, 0.98)
    m = _fake([0.0] * 2000, kappa=0.9, t0=50.0, kappa_reps=ks, t0_reps=[50.0] * 2000,
              lstar_reps=[RT._lstar(k, 50.0) for k in ks], cells={1: _cell(200)}, help_reps={1: [False] * 2000})
    f = RT.domain_tests({"G": m}, 0.5, 0.75)["H4f"]
    c = f["cells"][0]
    assert c["model_length"] == 200 and c["central"] == [200] and c["lstar_lo"] > 400
    assert c["p"] == 1.0 and c["among"] and f["p"] == 1.0 and f["holds_point"] and f["untestable"] is None
    from reach.analysis.s2.families import holm_family
    mem = holm_family([dict(id=["H4f"], kind="retain", p=f["p"], untestable=None)], 0.05)["members"][0]
    assert mem["supported"]


def test_h4f_fails_when_the_measured_best_disagrees_with_the_model():
    """kappa 0.6, t0 30: the model picks 100 in every replicate; a measured best of 200 (or 25) is not among the
    model's lengths: p = 0, the retain member is not supported. A measured 100 holds."""
    m = _fake([0.0] * 200, kappa=0.6, t0=30.0, kappa_reps=np.linspace(0.58, 0.62, 200), t0_reps=[30.0] * 200,
              cells={1: _cell(200), 2: _cell(25), 4: _cell(100)}, help_reps={b: [True] * 200 for b in (1, 2, 4)})
    f = RT.domain_tests({"S": m}, 0.5, 0.75)["H4f"]
    by = {c["budget"]: c for c in f["cells"]}
    assert all(c["model_length"] == 100 and c["central"] == [100] for c in by.values())
    assert by[1]["p"] == 0 and not by[1]["among"] and by[2]["p"] == 0 and not by[2]["among"]
    assert by[4]["p"] == 1.0 and by[4]["among"]
    assert f["p"] == 0 and not f["holds_point"]
    from reach.analysis.s2.families import holm_family
    assert not holm_family([dict(id=["H4f"], kind="retain", p=f["p"], untestable=None)], 0.05)["members"][0]["supported"]


def test_h4f_retain_member_known_answer():
    """100 replicates of (kappa, t0) whose model lengths on the grid 25-200 (B = 200) are 50 (2), 100 (60) and 200
    (38). p = the smallest alpha at which the measured length leaves the central 100(1 - alpha)% of the replicates'
    choices = min(1, 2 min(P(choice <= L), P(choice >= L))), 0 when no replicate chose L: 100 -> 1, 200 -> 0.76,
    50 -> 0.04 (outside the central 95%), 25 -> 0 (never chosen). Member p = the smallest cell p; a cell without a
    measured length, or more than 10% of replicates without a model length, makes the member untestable."""
    ks = [0.6] * 2 + [0.6] * 60 + [0.9] * 38
    t0s = [20.0] * 2 + [30.0] * 60 + [50.0] * 38
    cells = {1: _cell(100), 2: _cell(200), 4: _cell(50), 8: _cell(25)}
    m = _fake([0.0] * 100, kappa=0.6, t0=30.0, kappa_reps=ks, t0_reps=t0s, cells=cells,
              help_reps={b: [True] * 100 for b in cells})
    f = RT.domain_tests({"S": m}, 1.0, 0.75)["H4f"]
    by = {c["budget"]: c for c in f["cells"]}
    assert by[1]["p"] == pytest.approx(1.0) and by[1]["among"]
    assert by[2]["p"] == pytest.approx(0.76) and by[2]["among"]
    assert by[4]["p"] == pytest.approx(0.04) and not by[4]["among"]
    assert by[8]["p"] == 0 and not by[8]["among"]
    assert by[1]["central"] == [100, 200] and by[1]["model_length"] == 100 and by[1]["dropped"] == 0
    assert f["p"] == 0 and not f["holds_point"]
    assert RT.p_among([50] * 2 + [100] * 60 + [200] * 38, 200) == pytest.approx(0.76)
    half = [50] * 50 + [200] * 50                    # 100 lies between the chosen lengths but is never chosen: p = 0
    assert RT.p_among(half, 100) == 0 and RT.central(half) == [50, 200] and RT.p_among(half, 200) == 1.0
    m["cells"][8]["length"] = None
    g = RT.domain_tests({"S": m}, 1.0, 0.75)["H4f"]
    assert g["p"] is None and g["untestable"].startswith("no measured length")
    m["cells"][8]["length"] = 25
    m["reps"]["kappa"] = [float("nan")] * 11 + ks[11:]                      # 11% degenerate: not estimable
    h = RT.domain_tests({"S": m}, 1.0, 0.75)["H4f"]
    assert h["p"] is None and all(not c["estimable"] and c["dropped"] == 11 for c in h["cells"])
    m["reps"]["kappa"] = [float("nan")] * 10 + ks[10:]                      # 10%: estimable
    assert RT.domain_tests({"S": m}, 1.0, 0.75)["H4f"]["p"] == 0


def test_lstar_follows_t0_over_one_minus_kappa():
    m = _model("S", 0.5, 0.4, 1.0, 40.0, boot=8, alloc_seeds=0, seed=6)
    assert m["point"]["kappa"] == pytest.approx(0.5, abs=0.12) and m["point"]["t0"] == pytest.approx(40, abs=12)
    assert m["point"]["lstar"] == pytest.approx(m["point"]["t0"] / (1 - m["point"]["kappa"]))
    assert RT._lstar(1.2, 5.0) == RT.LSTAR_CAP and math.isnan(RT._lstar(float("nan"), 5.0))


def test_each_set_draws_its_own_bootstrap_targets():
    """Guardrail row 113 (independent review, 2026-10-27): `set_model` seeded every set's target bootstrap with the
    same seed, so two sets with the same number of targets drew the same target indices in every replicate and the
    H4e share pooled over sets was correlated across sets. Two sets with identical data (only the names differ) must
    get different replicates; every registered calibration set gets its own stream."""
    from reach.runner.phases import CAL_SETS
    rows, _ = sim_set("S1", 0.6, 0.5, 2.0, 2.0, range(4, 9), targets=4, seed=1)
    out = {}
    for g in ("S1", "S2"):
        rr = [dict(r, set=g, target_id=r["target_id"].replace("S1", g)) for r in rows]
        out[g] = RT.set_model(RT.fit_runs(rr)[0], tidy(rr, (1, 2, 4, 8, 16)), None, (1, 2, 4, 8, 16), boot=4)
    assert out["S1"]["point"]["kappa"] == pytest.approx(out["S2"]["point"]["kappa"])          # same data
    assert out["S1"]["reps"]["slope"] != out["S2"]["reps"]["slope"]                           # different draws
    draws = {g: tuple(np.random.default_rng(RT.set_seed(g)).integers(0, 40, 40)) for g in CAL_SETS}
    assert len(set(draws.values())) == len(CAL_SETS)


def _planted_cell(monkeypatch, k50):
    best = max((L for L in k50 if not math.isnan(k50[L]) and k50[L] > float("-inf")), key=lambda L: k50[L])
    monkeypatch.setattr(RT.alloc, "best_split", lambda *a, **k: dict(k50=dict(k50), plug_in={}, best_length=best,
                                                                      best_k50=k50[best]))
    return RT.cell([], 1)


def test_h4f_grid_keeps_lengths_that_solve_nothing(monkeypatch):
    """D087 (stricter than D080 h): a split length whose K50 is -inf (nothing solved) stays in the model's candidate
    grid, so the model's choice cannot silently move to the next length. kappa 0.6, t0 5: the model picks 25 on the
    full grid; 25 solved nothing and the measured best is 50, so the cell fails (p 0). Planted: dropping -inf from the
    grid makes the model pick 50 and the cell hold. An undefined (NaN) K50 is still left out."""
    c = _planted_cell(monkeypatch, {25: float("-inf"), 50: 9.0, 100: 8.0, 200: 7.0})
    assert c["grid"] == [25, 50, 100, 200] and c["length"] == 50
    m = _fake([0.0] * 200, kappa=0.6, t0=5.0, kappa_reps=np.linspace(0.58, 0.62, 200), t0_reps=[5.0] * 200,
              cells={1: c}, help_reps={1: [True] * 200})
    f = RT.domain_tests({"S": m}, 0.5, 0.75)["H4f"]
    x = f["cells"][0]
    assert x["model_length"] == 25 and x["central"] == [25] and x["p"] == 0 and not x["among"]
    assert f["p"] == 0 and not f["holds_point"]
    c = _planted_cell(monkeypatch, {25: float("nan"), 50: 9.0, 100: 8.0, 200: 7.0})
    assert c["grid"] == [50, 100, 200] and c["length"] == 50

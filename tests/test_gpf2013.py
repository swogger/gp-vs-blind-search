"""H7 comparator 1 aligned with Graff, Poli and Flores (2013) (D083, D085; docs/design/gpf2013_comparator.md): the
Boolean indicators theta_i (Eq. 3) and phi_i (Eq. 5), the continuous indicators rho_i (Eq. 7, every lag) and
varsigma_j (Eq. 8, j = 2..N as in Eq. 10), and the linear model of Eqs. (9) and (10) fitted as in section 3.4 (LARS
order, number of indicators by 5-fold cross-validated RSE, least squares). Known answers worked by hand."""
import json
import math

import numpy as np
import pytest

from reach import grammar as G
from reach.analysis.s2 import gp2010
from reach.analysis.s2 import gpf2013 as M

CAL_REAL, CAL_BOOL = "R-ARITH4", "BOOL-NAND-n5"


# ---------------------------------------------------------------- Boolean indicators (section 3.2)
def test_theta_and_phi_by_hand_on_two_inputs():
    """Rows r = 0..3 (bits are the inputs). AND = (0,0,0,1): of the 8 ordered pairs at distance 1, 4 differ ->
    theta_1 = 4 / (4 * 2) = 0.5; of the 4 at distance 2 (0<->3, 1<->2), 2 differ -> theta_2 = 2 / (4 * 1) = 0.5;
    the second difference is the XOR of all four outputs, 1 at every x -> phi_2 = 4. XOR: theta = (1, 0), phi_2 = 0."""
    assert M.theta([0, 0, 0, 1], 2) == pytest.approx([0.5, 0.5])
    assert M.phi([0, 0, 0, 1], 2) == pytest.approx([4.0])
    assert M.theta([0, 1, 1, 0], 2) == pytest.approx([1.0, 0.0])
    assert M.phi([0, 1, 1, 0], 2) == pytest.approx([0.0])
    assert M.theta([1, 1, 1, 1], 2) == [0.0, 0.0] and M.phi([1, 1, 1, 1], 2) == [0.0]


def test_theta_and_phi_reproduce_the_papers_three_input_example():
    """The function of the 2013 paper's Table 1 (outputs of abc = 000..111: 1,1,0,1,0,0,0,1). By hand: 6 of the 12
    cube edges differ -> theta_1 = 12 / (8 * 3) = 0.5; 8 of the 12 distance-2 pairs -> theta_2 = 16 / 24 = 2/3; 2 of
    the 4 antipodal pairs -> theta_3 = 4 / 8 = 0.5 (the paper: proportional to 4). 4 of the 6 square faces have odd
    parity, 4 points each -> phi_2 = 16; the XOR of all 8 outputs is 0 -> phi_3 = 0 (the paper: f_abc = 0)."""
    f = [1, 1, 0, 1, 0, 0, 0, 1]
    assert M.theta(f, 3) == pytest.approx([0.5, 2 / 3, 0.5])
    assert M.phi(f, 3) == pytest.approx([16.0, 0.0])


def test_phi_over_sets_equals_the_multiset_sum():
    """Eq. (5) sums over multisets; a repeated variable gives a zero difference (footnote 3), so the sum over sets is
    the same. Brute force over multisets with repetition on random 4-input functions."""
    from itertools import combinations_with_replacement
    rng = np.random.default_rng(5)
    for _ in range(5):
        f = list(rng.integers(0, 2, 16))
        for i in (2, 3, 4):
            tot = 0
            for a in combinations_with_replacement(range(4), i):
                d = list(f)
                for v in a:
                    d = [d[x] ^ d[x ^ (1 << v)] for x in range(16)]
                tot += sum(d)
            assert M.phi(f, 4)[i - 2] == tot


def test_boolean_indicators_of_grammar_targets():
    """Five inputs: not x5 has theta_i = C(4, i-1) / C(5, i) = i/5 and no higher difference; parity of five has
    theta_i = 1 for odd i and 0 for even i; and(x1, x2) has phi_2 = 32 (one pair, every x) and phi_3..5 = 0."""
    names, x = M.indicators(G.get(CAL_BOOL), gp2010.target_vector(G.get(CAL_BOOL), "nand(x5, x5)"))
    assert names == [f"theta_{i}" for i in range(1, 6)] + [f"phi_{i}" for i in range(2, 6)]     # 2N - 1 = 9
    assert x[:5] == pytest.approx([i / 5 for i in range(1, 6)]) and x[5:] == [0.0] * 4
    g = G.get("BOOL-RICH-n5")
    _, x = M.indicators(g, gp2010.target_vector(g, "xor(xor(xor(xor(x1, x2), x3), x4), x5)"))
    assert x[:5] == [1.0, 0.0, 1.0, 0.0, 1.0] and x[5:] == [0.0] * 4
    _, x = M.indicators(g, gp2010.target_vector(g, "and(x1, x2)"))
    assert x[5:] == [32.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------- continuous indicators (section 3.3)
def test_rho_by_hand_on_one_variable():
    """f = x^2 at 0, 0.5, 1, 1.5 stored as (0, 1, 4, 9) with h = 0.5: lag 1: (1 + 3 + 5) / 0.5 = 18; lag 2:
    (4 + 8) / 1.0 = 12; lag 3: 9 / 1.5 = 6. Pairs that leave the grid add 0 (footnote 5)."""
    assert M.rho(np.array([0.0, 1.0, 4.0, 9.0]), 0.5) == pytest.approx([18.0, 12.0, 6.0])


def test_rho_and_varsigma_reproduce_the_papers_two_variable_example():
    """The 2013 paper's Table 3 example (x1, x2 in {-0.5, 0, 0.5}; F[i, j] = f(x1_i, x2_j)). By hand: rho_1 = (4.0 +
    4.4) / 0.5 = 16.8 (the paper's |f(0, -0.5) - f(-0.5, -0.5)| = 0.5 is one term); rho_2 = (3.0 + 3.6) / 1.0 = 6.6;
    varsigma_2 over {x1x1, x1x2, x2x2}: (2.0 + 4.0 + 2.0) / 0.5^2 = 32."""
    F = np.array([[-1.1, -0.5, 0.1], [-0.6, 1.0, 0.6], [-0.1, 0.5, 1.1]])
    assert M.rho(F, 0.5) == pytest.approx([16.8, 6.6])
    assert M.varsigma(F, 0.5, 2) == pytest.approx(32.0)


def test_varsigma_by_hand_on_a_product():
    """F = x * y on {0, 1, 2}^2, h = 1: only the mixed difference is non-zero, 1 at each of the 4 inner squares."""
    t = np.arange(3.0)
    F = np.outer(t, t)
    assert M.varsigma(F, 1.0, 2) == 4.0
    assert M.rho(F, 1.0) == pytest.approx([12.0, 6.0])


def test_one_variable_has_no_varsigma_and_every_lag():
    """Eq. (10) sums varsigma_j for j = 2..N, so one variable has none (D085); rho over all 99 lags of the 100-point
    grid, h = 1/101. f = x: rho_i = (100 - i) * (i h) / (i h) = 100 - i."""
    g = G.get(CAL_REAL)
    assert M.grid(g) == ((100,), pytest.approx(1 / 101))
    names, x = M.indicators(g, gp2010.target_vector(g, "x"))
    assert names == [f"rho_{i}" for i in range(1, 100)]
    assert x == pytest.approx([100.0 - i for i in range(1, 100)])


def test_standardised_form_is_scale_free():
    """Real indicators on the standardised vector (adaptation, D085): a rescaled and shifted target gives the same
    values; raw values scale with it. The Boolean domain has the raw form only."""
    g = G.get(CAL_REAL)
    v = np.array(gp2010.target_vector(g, "*(x, x)"))
    _, a = M.indicators(g, v, form="std")
    _, b = M.indicators(g, 1e6 * v + 3.0, form="std")
    assert a == pytest.approx(b)
    _, r = M.indicators(g, 2 * v)
    assert r == pytest.approx(2 * np.array(M.indicators(g, v)[1]))
    with pytest.raises(ValueError, match="raw"):
        M.indicators(G.get(CAL_BOOL), [0.0] * 32, form="std")


# ---------------------------------------------------------------- the model (section 3.4)
def test_fit_budget_known_answer_by_hand():
    """X = (x0, x1), P = 0.2 + 0.1 x0 exactly; x1 uncorrelated. LARS enters x0 first; least squares gives a0 = 0.2,
    a = 0.1."""
    x0 = np.array([0.0, 1, 2, 3, 4, 5])
    x1 = np.array([1.0, -1, -1, 1, 1, -1])
    mb = M.fit_budget(np.column_stack([x0, x1]), 0.2 + 0.1 * x0, ["x0", "x1"], m=1)
    assert mb["selected"] == ["x0"] and mb["a0"] == pytest.approx(0.2) and mb["coef"] == pytest.approx([0.1])


def test_number_of_indicators_is_the_first_minimum_of_the_cv_curve():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(50, 6))
    P = 0.5 + 0.1 * X[:, 1] + 0.05 * X[:, 4] + 0.05 * rng.normal(size=50)
    parts = gp2010.folds(50)
    curve = M.cv_curve(X, P, parts)
    mb = M.fit_budget(X, P, [f"c{i}" for i in range(6)], parts)
    assert mb["m"] == int(np.argmin(curve)) + 1 < 6 and mb["cv_rse"] == pytest.approx(float(curve.min()))
    assert mb["selected"][:2] == ["c1", "c4"]


def test_cv_curve_by_brute_force():
    """For every fold: LARS order on the training rows, least squares with the first m indicators, prediction of the
    left-out rows; RSE over all rows for each m."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 5))
    P = 0.5 + 0.1 * X[:, 0] - 0.05 * X[:, 2] + 0.02 * rng.normal(size=40)
    parts = gp2010.folds(40)
    curve = M.cv_curve(X, P, parts)
    for m in range(1, 6):
        Q = np.empty(40)
        for te in parts:
            tr = np.setdiff1d(np.arange(40), te)
            order = gp2010.lar_order(X[tr], P[tr])[:m]
            Z = np.column_stack([np.ones(len(tr)), X[np.ix_(tr, order)]])
            b = np.linalg.lstsq(Z, P[tr], rcond=None)[0]
            Q[te] = np.column_stack([np.ones(len(te)), X[np.ix_(te, order)]]) @ b
        assert curve[m - 1] == pytest.approx(((P - Q) ** 2).sum() / ((P - P.mean()) ** 2).sum())
    assert curve[1] < 0.5 * curve[0]                                       # the second planted indicator helps


def _bool_recs(n_per_set=40, seed=0, sets=(CAL_BOOL, "BOOL-NOR-n5"), runs=20):
    """Random truth tables; P = 0.9 - 0.8 theta_1 exactly in run counts (runs per target)."""
    rng = np.random.default_rng(seed)
    recs, vec = [], {}
    g = G.get(CAL_BOOL)
    for s in sets:
        for t in range(n_per_set):
            v = [float(b) for b in rng.random(32) < rng.uniform(0.05, 0.95)]
            vec[(s, f"{s}_{t}")] = v
            k = round(runs * (0.9 - 0.8 * M.indicators(g, v)[1][0]))
            for b in (1, 2):
                recs += [dict(set=s, target_id=f"{s}_{t}", budget=b, seed=i, y=int(i < k)) for i in range(runs)]
    return recs, vec


def test_fit_domain_recovers_a_planted_indicator_model():
    recs, vec = _bool_recs()
    model = M.fit_domain(recs, vec, "bool", budgets=(1, 2))
    assert model["form"] == "raw" and model["kind"] == "graff_poli_flores_2013"
    for b in ("1", "2"):
        mb = model["budgets"][b]
        assert mb["selected"][0] == "theta_1" and mb["cv_rse"] < 0.05 and mb["m"] <= 9
    p = M.predict(model, recs, vec)
    rates = np.array([0.9 - 0.8 * M.indicators(G.get(CAL_BOOL), vec[(r["set"], r["target_id"])])[1][0]
                      for r in recs])
    assert np.abs(p - rates).max() < 0.06
    assert model["budgets_without_model"] == []


def test_real_form_is_chosen_by_cross_validated_rse():
    """Amplitudes over six orders of magnitude: when recovery depends on the shape (frequency), the standardised form
    wins; when it depends on the raw amplitude, the raw form wins (ties go to raw)."""
    rng = np.random.default_rng(3)
    x = np.real(G.sample_grid(G.get(CAL_REAL)))
    for depends_on, want in (("shape", "std"), ("amplitude", "raw")):
        recs, vec = [], {}
        for s in (CAL_REAL, "R-TRIG"):
            for t in range(40):
                A = 10 ** rng.uniform(0, 6) if depends_on == "shape" else rng.uniform(1, 2)
                w = rng.uniform(1, 6) if depends_on == "shape" else 3.0
                vec[(s, str(t))] = list(A * np.sin(w * x))
                P = 0.1 + 0.15 * (w - 1) if depends_on == "shape" else 0.9 * (A - 1)
                k = round(20 * P)
                recs += [dict(set=s, target_id=str(t), budget=1, seed=i, y=int(i < k)) for i in range(20)]
        model = M.fit_domain(recs, vec, "real", budgets=(1,))
        assert model["form"] == want, depends_on
        assert set(model["form_cv_rse"]) == {"raw", "std"}


def test_fit_refuses_held_out_sets_and_mixed_lengths():
    recs, vec = _bool_recs(n_per_set=12)
    for bad in ("BOOL-AN-n5", "BOOL-AX1-n5", "R-ARITH"):
        r2 = [dict(r, set=bad) if r["set"] == "BOOL-NOR-n5" else r for r in recs]
        v2 = {(bad if k[0] == "BOOL-NOR-n5" else k[0], k[1]): v for k, v in vec.items()}
        with pytest.raises(PermissionError, match="calibration sets only"):
            M.fit_domain(r2, v2, "bool", budgets=(1, 2))
    model = M.fit_domain(recs, vec, "bool", budgets=(1, 2))
    with pytest.raises(ValueError, match="no model at budget"):
        M.predict(model, [dict(recs[0], budget=4)], vec)
    with pytest.raises(ValueError, match="indicators"):
        M.predict(model, [dict(set="BOOL-NAND-n6", target_id="z", budget=1)], {("BOOL-NAND-n6", "z"): [0.0] * 64})


def test_freeze_writes_a_checksum_and_never_replaces(tmp_path):
    recs, vec = _bool_recs(n_per_set=15)
    model = M.fit_domain(recs, vec, "bool", budgets=(1,))
    path = tmp_path / "gpf2013.json"
    sha = M.freeze(dict(bool=model), path)
    assert path.with_suffix(".sha256").read_text().split()[0] == sha
    assert M.freeze(dict(bool=model), path) == sha
    back = json.loads(path.read_text())["bool"]
    one = [r for r in recs if r["budget"] == 1]
    assert M.predict(back, one, vec) == pytest.approx(M.predict(model, one, vec))
    model["budgets"]["1"]["a0"] += 1e-6
    with pytest.raises(RuntimeError, match="never replaced"):
        M.freeze(dict(bool=model), path)


def test_comparator1_is_scored_with_the_h7_criteria_on_clipped_predictions():
    """As comparator 2: H7 criteria on predictions clipped to [0, 1]; RSE per budget on target rates, unclipped; the
    share clipped. Hand-built model P = 5.0 - 9.0 theta_1 (theta_1 of random tables lies near 0.5: both sides clip)."""
    from reach.analysis.s2 import predictor as Pr
    from test_gp2010 import _ho_recs
    recs, vec = _ho_recs()
    names = M.indicator_names(G.get("BOOL-AN-n5"))
    model = dict(kind="graff_poli_flores_2013", form="raw", names=names,
                 budgets={"1": dict(a0=5.0, coef=[-9.0], selected=["theta_1"])})
    base = dict(unit="nodes", budget_term=False, params=[0.5, -0.05])
    res = M.score(model, recs, vec, base, 0.1, boot=50, min_sets_ok=2)
    th = {k: M.theta([int(b) for b in v], 5)[0] for k, v in vec.items()}
    raw = np.array([5.0 - 9.0 * th[(r["set"], r["target_id"])] for r in recs])
    assert (raw < 0).any() and (raw > 1).any()
    want = Pr.score_p(recs, np.clip(raw, 0, 1), Pr.predict(base, recs), 0.1, boot=50, min_sets_ok=2)
    assert {k: res[k] for k in want} == want
    assert res["clipped_share"] == pytest.approx(np.mean((raw < 0) | (raw > 1))) and 0 < res["clipped_share"] < 1
    rates = {}
    for r, q in zip(recs, raw):
        rates.setdefault((r["set"], r["target_id"]), [q, []])[1].append(r["y"])
    P_ = np.array([np.mean(v[1]) for v in rates.values()])
    Q_ = np.array([v[0] for v in rates.values()])
    assert res["rse"]["1"] == pytest.approx(((P_ - Q_) ** 2).sum() / ((P_ - P_.mean()) ** 2).sum())
    assert not math.isnan(res["rse"]["1"])


# ---------------------------------------------------------------- wiring: confirm's H7 block and Stage 1 isolation
def _calibration_with_targets(tmp_path, monkeypatch):
    from reach.analysis.s2 import invariance as I
    from test_dryrun import SETS, _rows, _write
    rows = [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0, i)]
    _write(tmp_path, "s2_cal_a16", rows)
    units = {r["target_id"]: r["units"] for r in rows}
    for g in SETS:
        d = tmp_path / f"data/targets_s2/{g}"
        d.mkdir(parents=True)
        tids = sorted({r["target_id"] for r in rows if r["set"] == g})
        (d / "calibration.json").write_text(json.dumps([dict(target_id=t, units=units[t], expression=t) for t in tids]))
    from reach.analysis.s2 import confirm as C
    monkeypatch.setattr(I, "invariance_test", lambda *a, **k: {u: dict(excess=0.0, upper95=0.001, invariant=False,
                        boot=2000, p_below_delta=0.5, logo=0.1) for u in C.UNITS})
    monkeypatch.setattr(gp2010, "target_vector", lambda g, e: list(np.random.default_rng(abs(hash(e)) % 2**32)
                                                                    .integers(0, 2, 32).astype(float)))


def test_confirm_fits_comparator1_on_the_h7_records(tmp_path, monkeypatch):
    """confirm fits comparator 1 (D085) per domain on exactly the records and target vectors comparator 2 gets (the
    H7 records), stores it as `comparator` and names it in the H7 line."""
    from reach.analysis.s2 import confirm as C
    _calibration_with_targets(tmp_path, monkeypatch)
    f1, f2 = [], []
    real1, real2 = M.fit_domain, gp2010.fit_domain
    monkeypatch.setattr(M, "fit_domain", lambda recs, vec, d, *a, **k: f1.append((list(recs), vec, d))
                        or real1(recs, vec, d, *a, **k))
    monkeypatch.setattr(gp2010, "fit_domain", lambda recs, vec, d, *a, **k: f2.append((list(recs), vec, d))
                        or real2(recs, vec, d, *a, **k))
    res = C.build(dict(C.PHASE_MAPS["calibration"], levels=None), tmp_path, boot=20, allow_incomplete=True)
    assert len(f1) == len(f2) == 1 and f1[0][2] == "bool" and f1[0][1] == f2[0][1]
    key = lambda rs: sorted((r["set"], r["target_id"], r["budget"], r["seed"]) for r in rs)   # noqa: E731
    assert key(f1[0][0]) == key(f2[0][0])
    c1 = res["h7"]["bool"]["comparator"]
    assert c1["kind"] == "graff_poli_flores_2013" and c1["form"] == "raw" and c1["left_out"] == []
    assert set(c1["budgets"]) == {"1", "2", "4"} and "loso_logloss" not in c1
    line = next(x for x in C.render(res, "t", "s").splitlines() if x.startswith("- **bool:** unit "))
    assert "Graff, Poli and Flores (2013) comparator, raw indicators, 5-fold RSE 1x " in line
    assert "style" not in line


def test_stage1_results_do_not_depend_on_the_h7_comparators(tmp_path, monkeypatch):
    """Isolation (D085): the comparators are fitted after every Stage 1 test of the domain and draw no global random
    numbers, so with both replaced by stubs every result except H7's is identical."""
    from reach.analysis.s2 import confirm as C
    from reach.analysis.s2 import dryrun as D
    from test_dryrun import SETS, _rows, _write
    _write(tmp_path, "s2_pilot_reach_a", [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0, i)])
    _write(tmp_path, "s2_pilot_mech", _rows("BOOL-NAND-n5", "B", 0.5, 9))
    rows = [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0, i)]
    for g in SETS:
        d = tmp_path / f"data/targets_s2/{g}"
        d.mkdir(parents=True)
        tids = sorted({r["target_id"] for r in rows if r["set"] == g})
        (d / "pilot.json").write_text(json.dumps([dict(target_id=t, units=dict(nodes=1.0), expression=t) for t in tids]))
    monkeypatch.setattr(gp2010, "target_vector", lambda g, e: list(np.random.default_rng(abs(hash(e)) % 2**32)
                                                                    .integers(0, 2, 32).astype(float)))
    a = json.loads(json.dumps(D.build(tmp_path, delta=0.01, boot=30), default=float))
    assert a["h7"]["bool"]["comparator"]["kind"] == "graff_poli_flores_2013"
    stub = lambda *x, **k: dict(budgets={}, stub=True, form="stub", measure="stub")          # noqa: E731
    monkeypatch.setattr(M, "fit_domain", stub)
    monkeypatch.setattr(gp2010, "fit_domain", stub)
    monkeypatch.setattr(gp2010, "summary", lambda m: m)
    b = json.loads(json.dumps(D.build(tmp_path, delta=0.01, boot=30), default=float))
    assert b["h7"]["bool"]["comparator"]["stub"]
    assert set(a) == set(b) and {k: a[k] for k in a if k != "h7"} == {k: b[k] for k in b if k != "h7"}
    other = lambda t: [x for x in C.render(t, "t", "s").splitlines() if "LOSO log-loss" not in x]   # noqa: E731
    assert other(a) == other(b)


STAGE1_SHA_656369A = "6ae72e67168889a4223c13d622162feba5bce23e40d8cadede486cc88b7c6261"   # before D086
STAGE1_SHA_D086 = "b2f4eb3b523b521a582270906e1038e7520f539ba351fd107d6a9cce87a1ecac"   # before D088
STAGE1_SHA_D088 = "8ae9b6e6003803483155f919be9f322395ef5003bf313f614b98ad583721283a"


def test_pilot_dryrun_stage1_part_unchanged_since_656369a():
    """The committed DRYRUN (data/analysis/dryrun.json, regenerated with comparator 1 aligned) has the same Stage 1
    part, every result except H7's, as the DRYRUN of the code at 656369a (recomputed from a clean export of that
    commit on 2026-10-27, 700 s): canonical JSON without "h7", SHA-256. The full DRYRUN is too slow for the suite; a
    change to anything outside H7 that is committed with a regenerated DRYRUN fails here (D085). Moved once, by D086:
    the separation fix changes exactly the pilot cells with separated fits or replicates (with the checks disabled the
    fixed code reproduces 656369a's DRYRUN byte for byte); the lock is now on that result. Moved again, by D088: separated
    one-predictor fits keep their sign and K50 sits at the gap midpoint, which changes exactly the pilot cells with
    separated fits or replicates (with separated fits treated as row 118 did, the D088 code reproduces the DRYRUN of
    82e192f byte for byte, apart from the new "separation" counts); the lock is now on the D088 result."""
    import hashlib
    from pathlib import Path
    res = json.loads((Path(__file__).resolve().parents[1] / "data/analysis/dryrun.json").read_text())
    body = json.dumps({k: v for k, v in res.items() if k != "h7"}, sort_keys=True)
    assert hashlib.sha256(body.encode()).hexdigest() == STAGE1_SHA_D088
    assert res["h7"]["bool"]["comparator"]["kind"] == "graff_poli_flores_2013"

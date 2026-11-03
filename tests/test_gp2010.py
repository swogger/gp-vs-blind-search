"""H7 comparator 2: Graff and Poli (2010), Eq. (5): P(t) = a0 + sum_{p in S} a_p d(p, t) (D084;
docs/design/gp2010_comparator.md). Known answers worked by hand, the LAR property, refusals and freezing."""
import json
import math

import numpy as np
import pytest

from reach import grammar as G
from reach.analysis.s2 import gp2010 as M

CAL_REAL, CAL_BOOL = "R-ARITH4", "BOOL-NAND-n5"


def test_measures_reproduce_table4_by_hand():
    """Table 4 (p.1263) on t = (1, 0), p = (0, 2): differences (1, -2)."""
    t, p = [[1.0, 0.0]], [[0.0, 2.0]]
    want = dict(mse=(1 + 4) / 2, l1=3.0, dot2=0.0, dot3=0.0, gauss=math.exp(-5), log1p=math.log(2) + math.log(3))
    for k, v in want.items():
        assert M.distances(k, t, p)[0, 0] == pytest.approx(v)
    assert M.distances("dot2", [[1.0, 1.0]], [[1.0, 2.0]])[0, 0] == 9.0
    assert list(M.MEASURES) == ["mse", "l1", "dot2", "dot3", "gauss", "log1p"]
    assert M.CANDIDATES == tuple(M.MEASURES) + tuple("std:" + k for k in M.MEASURES)


def test_standardised_measures_by_hand():
    """Adaptation 7: footnote 8's normalisation, (t - mean) / sd (population sd); a constant vector becomes zeros.
    t = (1, 3) -> (-1, 1); p = (2, 2, ...) constant -> zeros; q = (0, 4) -> (-1, 1)."""
    assert M.standardise([[1.0, 3.0], [2.0, 2.0]]).tolist() == [[-1.0, 1.0], [0.0, 0.0]]
    assert M.distances("std:l1", [[1.0, 3.0]], [[2.0, 2.0], [0.0, 4.0]]).tolist() == [[2.0, 0.0]]
    assert M.distances("std:dot2", [[1.0, 3.0]], [[0.0, 4.0]])[0, 0] == 4.0


def test_equation5_known_answer_by_hand():
    """Worked by hand. T = {t1 = (1,0), t2 = (0,1), t3 = (1,1)}, d = (t.p)^2, P = (0.5, 0.5, 0.2). The columns
    d(t_j, .) are (1,0,1), (0,1,1), (1,1,4); centred P is (.1, .1, -.2); the correlations are -.1/sqrt(2/3) for the
    first two and -.6/sqrt(6) for the third, so LARS enters t3 first. OLS of P on (1, 1, 4): slope -0.6/6 = -0.1,
    intercept 0.4 + 0.2 = 0.6. Eq. (5) then predicts 0.6 for (0,0) and 0.6 - 0.1*4 = 0.2 for (1,1)."""
    V = [[1, 0], [0, 1], [1, 1]]
    ids = [["S", "t1"], ["S", "t2"], ["S", "t3"]]
    mb = M.fit_budget(V, [0.5, 0.5, 0.2], ids, "dot2", m=1)
    assert mb["refs"] == [["S", "t3"]] and mb["m"] == 1
    assert mb["a0"] == pytest.approx(0.6) and mb["coef"] == pytest.approx([-0.1])
    model = dict(measure="dot2", budgets={"1": mb})
    assert M.predict_vectors(model, [[0, 0], [1, 1], [1, 0]], 1) == pytest.approx([0.6, 0.2, 0.5])


def test_lar_is_equiangular_on_a_correlated_design():
    """The defining LAR property (Efron et al. 2004): when a column enters, every active column and the new one have
    the same absolute correlation with the residual, and no other column has more."""
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(60, 8))
    X = Z + 0.8 * Z[:, [0]]                                     # correlated columns
    y = X @ np.array([3, -2, 0, 1, 0, 0, 0.5, 0]) + rng.normal(size=60)
    order, mus = M.lar_order(X, y, trace=True)
    assert sorted(order) == list(range(8))
    Xs = (X - X.mean(0)) / np.sqrt(((X - X.mean(0)) ** 2).sum(0))
    for k in range(1, len(order)):
        c = np.abs(Xs.T @ (y - y.mean() - mus[k]))
        act = order[:k + 1]
        assert np.allclose(c[act], c[act].max(), rtol=1e-8)
        assert c.max() <= c[act].max() * (1 + 1e-8)


def test_lar_stays_equiangular_along_a_long_path_on_distances():
    """A full path on an L1 distance matrix (149 steps): the bordered inverse stays accurate."""
    rng = np.random.default_rng(9)
    V = rng.uniform(0, 5, size=(150, 20))
    X, y = M.distances("l1", V, V), rng.random(150)
    order, mus = M.lar_order(X, y, trace=True)
    assert len(order) == 149
    Xs = (X - X.mean(0)) / np.sqrt(((X - X.mean(0)) ** 2).sum(0))
    for k in range(1, len(order)):
        c = np.abs(Xs.T @ (y - y.mean() - mus[k]))
        act = order[:k + 1]
        assert c[act].min() >= c[act].max() * (1 - 1e-6) and c.max() <= c[act].max() * (1 + 1e-6)


def test_lar_on_an_orthogonal_design_enters_by_correlation():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(40, 5))
    Q, _ = np.linalg.qr(A - A.mean(0))                           # centred, orthonormal columns
    y = Q @ np.array([0.5, -3.0, 1.0, 2.0, -0.2])
    corr = np.abs(((Q - Q.mean(0)) / np.sqrt(((Q - Q.mean(0)) ** 2).sum(0))).T @ (y - y.mean()))
    assert M.lar_order(Q, y) == list(np.argsort(-corr))


def test_lar_skips_a_duplicate_column():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 3))
    X = np.column_stack([X, X[:, 0]])
    order = M.lar_order(X, X[:, 0] * 2 + X[:, 1] + rng.normal(size=30) * 0.1)
    assert len(order) == 3 and not {0, 3} <= set(order)


def test_nested_ols_matches_lstsq_for_every_m():
    rng = np.random.default_rng(6)
    X, y = rng.normal(size=(25, 6)), rng.normal(size=25)
    order = [3, 0, 5, 1]
    ols = M.NestedOLS(X, y, order)
    for m in range(1, 5):
        Z = np.column_stack([np.ones(25), X[:, order[:m]]])
        assert ols.coef(m) == pytest.approx(np.linalg.lstsq(Z, y, rcond=None)[0])
    Xn = rng.normal(size=(7, 6))
    want = [ols.coef(m)[0] + Xn[:, order[:m]] @ ols.coef(m)[1:] for m in range(1, 5)]
    assert ols.predict_all(Xn) == pytest.approx(np.array(want))


def _planted(n=120, L=16, seed=7):
    """P(t) = 0.98 - 0.023 * L1(p*, t) exactly (P in [0, 1]), p* = problem 11 (real-valued vectors, so that no other
    Table 4 measure is a linear function of L1): the procedure must find p* and the coefficients."""
    rng = np.random.default_rng(seed)
    V = rng.uniform(0, 5, size=(n, L))
    P = 0.98 - 0.023 * M.distances("l1", V, V[[11]])[:, 0]
    return V, P


def test_cross_validation_finds_a_planted_reference_and_m():
    V, P = _planted()
    ids = [["S", f"t{i}"] for i in range(len(P))]
    mb = M.fit_budget(V, P, ids, "l1", M.folds(len(P)))
    assert mb["refs"][0] == ["S", "t11"] and 1 <= mb["m"] < 20 and mb["cv_rse"] < 0.5   # p* is absent from one fold
    model = dict(measure="l1", budgets={"1": mb})
    assert M.predict_vectors(model, V[:20], 1) == pytest.approx(P[:20], abs=1e-9)


def test_cv_curve_by_brute_force_and_m_is_its_first_minimum():
    """Section 3.4 rebuilt with lstsq: per fold, LAR on the training rows and columns only (A2), OLS on the first m,
    RSE over all problems; fit_budget takes the smallest m with the lowest RSE (A4)."""
    rng = np.random.default_rng(10)
    V = rng.uniform(0, 5, size=(40, 6))
    P = np.clip(0.9 - 0.02 * M.distances("l1", V, V[[3]])[:, 0] + rng.normal(0, 0.05, 40), 0, 1)
    D, parts = M.distances("l1", V, V), M.folds(40)
    curve = M.cv_curve(D, P, parts)
    Q = np.zeros((len(curve), 40))
    for te in parts:
        tr = np.setdiff1d(np.arange(40), te)
        order = M.lar_order(D[np.ix_(tr, tr)], P[tr])
        for m in range(1, len(curve) + 1):
            Z = np.column_stack([np.ones(len(tr)), D[np.ix_(tr, tr[order[:m]])]])
            cf = np.linalg.lstsq(Z, P[tr], rcond=None)[0]
            Q[m - 1, te] = cf[0] + D[np.ix_(te, tr[order[:m]])] @ cf[1:]
    want = ((Q - P) ** 2).sum(axis=1) / ((P - P.mean()) ** 2).sum()
    assert curve[:20] == pytest.approx(want[:20], rel=1e-6)
    mb = M.fit_budget(V, P, [["S", str(i)] for i in range(40)], "l1", parts)
    assert mb["m"] == int(np.argmin(curve)) + 1 and mb["cv_rse"] == pytest.approx(curve.min())


def _recs(V, P, sets=(CAL_REAL,), runs=5, budgets=(1, 2)):
    """Run records whose share of successes is P rounded to 1/runs (deterministic)."""
    recs, vec = [], {}
    for i, (v, p) in enumerate(zip(V, P)):
        s = sets[i % len(sets)]
        vec[(s, f"t{i}")] = list(v)
        for b in budgets:
            for k in range(runs):
                recs.append(dict(set=s, target_id=f"t{i}", budget=b, y=int(k < round(p * runs))))
    return recs, vec


def test_fit_domain_chooses_the_measure_by_cross_validated_rse():
    """Section 6.1: the planted measure (l1) wins; one model per budget; P(t) = share of runs with y = 1."""
    V, P = _planted(n=150)
    recs, vec = _recs(V, P, runs=100)
    m = M.fit_domain(recs, vec, "real", budgets=(1, 2))
    assert m["measure"] == "l1" and set(m["budgets"]) == {"1", "2"}
    assert m["budgets"]["1"]["refs"][0] == [CAL_REAL, "t11"]
    assert min(m["measure_cv_rse"], key=m["measure_cv_rse"].get) == "l1" and set(m["measure_cv_rse"]) == set(M.CANDIDATES)
    with pytest.raises(ValueError, match="no model at budget"):
        M.predict_vectors(m, V[:2], 4)


def test_fit_refuses_held_out_and_benchmark_sets():
    V, P = _planted(n=30)
    for bad in ("R-ARITH", "BOOL-AN-n5", "RS-add_sqrt"):
        recs, vec = _recs(V, P, sets=(CAL_REAL, bad))
        with pytest.raises(PermissionError, match="calibration sets only"):
            M.fit_domain(recs, vec, "real", budgets=(1, 2))


def test_mixed_vector_lengths_are_refused():
    recs = [dict(set=CAL_BOOL, target_id="a", budget=1, y=1), dict(set=CAL_BOOL, target_id="b", budget=1, y=0)]
    with pytest.raises(ValueError, match="different lengths"):
        M.problems(recs, {(CAL_BOOL, "a"): [0.0] * 32, (CAL_BOOL, "b"): [0.0] * 64}, budgets=(1,))
    model = dict(measure="l1", budgets={"1": dict(a0=0.5, coef=[0.1], ref_vectors=[[0.0] * 32])})
    with pytest.raises(ValueError, match="vector length"):
        M.predict_vectors(model, [[0.0] * 64], 1)


def test_target_vectors_are_grid_values_and_truth_tables():
    gb = G.get(CAL_BOOL)
    tt = M.target_vector(gb, "nand(x5, x5)")                   # not x5: grammar.evaluate puts x5 in bit 0 of the row
    assert len(tt) == 32 and tt[:4] == [1.0, 0.0, 1.0, 0.0] and M.target_vector(gb, "nand(x1, x1)")[:16] == [1.0] * 16
    gr = G.get(CAL_REAL)
    v = M.target_vector(gr, "*(x, x)")
    grid = G.sample_grid(gr).real
    assert v == pytest.approx(list(grid ** 2))
    with pytest.raises(ValueError, match="not finite"):
        M.target_vector(gr, "/(x, -(x, x))")


def test_freeze_writes_a_checksum_and_never_replaces(tmp_path):
    V, P = _planted(n=40)
    model = dict(measure="l1", budgets={"1": M.fit_budget(V, P, [["S", str(i)] for i in range(40)], "l1", m=2)})
    path = tmp_path / "gp2010.json"
    sha = M.freeze(dict(real=model), path)
    assert path.with_suffix(".sha256").read_text().split()[0] == sha
    assert M.freeze(dict(real=model), path) == sha                       # the same content: unchanged
    back = json.loads(path.read_text())["real"]
    assert M.predict_vectors(back, V[:5], 1) == pytest.approx(M.predict_vectors(model, V[:5], 1))
    model["budgets"]["1"]["a0"] += 1e-6
    with pytest.raises(RuntimeError, match="never replaced"):
        M.freeze(dict(real=model), path)
    assert path.with_suffix(".sha256").read_text().split()[0] == sha


# ---------------------------------------------------------------- scoring beside the frozen model and comparator 1
def _ho_recs():
    """Held-out-like records (scoring only): two sets, targets with known vectors, 4 runs each at budget 1."""
    rng = np.random.default_rng(11)
    recs, vec = [], {}
    for s in ("BOOL-AN-n5", "BOOL-AX1-n5"):
        for t in range(12):
            vec[(s, f"{s}{t}")] = list(rng.integers(0, 2, 32).astype(float))
            for k in range(4):
                recs.append(dict(set=s, target_id=f"{s}{t}", budget=1, y=int(rng.random() < 0.3 + 0.04 * t),
                                 units=dict(nodes=float(5 + t)), pfeatures=dict(f=float(t))))
    return recs, vec


def test_score_p_is_the_frozen_models_score():
    from reach.analysis.s2 import predictor as Pr
    recs, _ = _ho_recs()
    model = dict(unit="nodes", budget_term=False, params=[1.0, -0.1])
    base = dict(unit="nodes", budget_term=False, params=[0.5, -0.05])
    a = Pr.score(model, recs, base, 0.1, boot=50, min_sets_ok=2)
    b = Pr.score_p(recs, Pr.predict(model, recs), Pr.predict(base, recs), 0.1, boot=50, min_sets_ok=2)
    assert a == b


def test_comparator2_is_scored_with_the_h7_criteria_on_clipped_predictions():
    """Adaptation 10: H7 criteria on predictions clipped to [0, 1]; RSE per budget on target rates, unclipped; the
    share clipped. Hand-built model: P = 2.4 - 0.1 * |t|_1 against the all-zero reference with d = l1."""
    from reach.analysis.s2 import predictor as Pr
    recs, vec = _ho_recs()
    model = dict(measure="l1", budgets={"1": dict(a0=2.4, coef=[-0.1], ref_vectors=[[0.0] * 32], refs=[["S", "z"]])})
    base = dict(unit="nodes", budget_term=False, params=[0.5, -0.05])
    res = M.score(model, recs, vec, base, 0.1, boot=50, min_sets_ok=2)
    raw = np.array([2.4 - 0.1 * sum(vec[(r["set"], r["target_id"])]) for r in recs])
    want = Pr.score_p(recs, np.clip(raw, 0, 1), Pr.predict(base, recs), 0.1, boot=50, min_sets_ok=2)
    assert {k: res[k] for k in want} == want
    assert res["clipped_share"] == pytest.approx(np.mean((raw < 0) | (raw > 1))) and 0 < res["clipped_share"] < 0.5
    rates = {}
    for r, q in zip(recs, raw):
        rates.setdefault((r["set"], r["target_id"]), [q, []])[1].append(r["y"])
    P_ = np.array([np.mean(v[1]) for v in rates.values()])
    Q_ = np.array([v[0] for v in rates.values()])
    assert res["rse"]["1"] == pytest.approx(((P_ - Q_) ** 2).sum() / ((P_ - P_.mean()) ** 2).sum())


def test_score_beside_scores_the_three_models_against_one_baseline():
    """Comparator 1 is the Graff, Poli and Flores (2013) model (D085; it replaced D070's logistic model, D083)."""
    from reach.analysis.s2 import gpf2013 as C1
    from reach.analysis.s2 import predictor as Pr
    recs, vec = _ho_recs()
    frozen = dict(unit="nodes", budget_term=False, params=[1.0, -0.1])
    base = dict(unit="nodes", budget_term=False, params=[0.5, -0.05])
    comp1 = dict(kind="graff_poli_flores_2013", form="raw", names=C1.indicator_names(G.get("BOOL-AN-n5")),
                 budgets={"1": dict(a0=1.6, coef=[-2.0], selected=["theta_1"])})
    comp2 = dict(measure="dot2", budgets={"1": dict(a0=0.4, coef=[0.001], ref_vectors=[[1.0] * 32], refs=[["S", "o"]])})
    out = M.score_beside(frozen, comp1, comp2, recs, vec, base, 0.1, boot=50, min_sets_ok=2)
    pb = Pr.predict(base, recs)
    assert out["frozen"] == Pr.score_p(recs, Pr.predict(frozen, recs), pb, 0.1, boot=50, min_sets_ok=2)
    assert out["comparator1"] == C1.score(comp1, recs, vec, base, 0.1, boot=50, min_sets_ok=2)
    p1 = np.clip(C1.predict(comp1, recs, vec), 0, 1)
    assert {k: out["comparator1"][k] for k in ("auc", "brier")} == {k: Pr.score_p(recs, p1, pb, 0.1, boot=50, min_sets_ok=2)[k]
                                                                   for k in ("auc", "brier")}
    assert out["comparator2"] == M.score(comp2, recs, vec, base, 0.1, boot=50, min_sets_ok=2)


def test_summary_and_attach_vectors_round_trip():
    V, P = _planted(n=40)
    ids = [["S", str(i)] for i in range(40)]
    model = dict(measure="l1", budgets={"1": M.fit_budget(V, P, ids, "l1", m=3)})
    s = M.summary(model)
    assert "ref_vectors" not in s["budgets"]["1"] and "ref_vectors" in model["budgets"]["1"]
    back = M.attach_vectors(json.loads(json.dumps(s)), {("S", str(i)): list(V[i]) for i in range(40)})
    assert M.predict_vectors(back, V[:6], 1) == pytest.approx(M.predict_vectors(model, V[:6], 1))


def test_confirm_fits_comparator2_on_the_h7_records(tmp_path, monkeypatch):
    """Wiring (D084): confirm fits comparator 2 per domain on exactly the records its H7 selection uses (every H7
    budget), with a vector for each of their targets, stores the summary beside comparator 1 and reports it."""
    from reach.analysis.s2 import confirm as C
    from reach.analysis.s2 import invariance as I
    from reach.analysis.s2 import predictor as Pr
    from test_dryrun import SETS, _rows, _write
    rows = [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0, i)]
    _write(tmp_path, "s2_cal_a16", rows)
    for g in SETS:
        d = tmp_path / f"data/targets_s2/{g}"
        d.mkdir(parents=True)
        tids = sorted({r["target_id"] for r in rows if r["set"] == g})
        units = {r["target_id"]: r["units"] for r in rows}
        (d / "calibration.json").write_text(json.dumps([dict(target_id=t, units=units[t], expression=t) for t in tids]))
    monkeypatch.setattr(I, "invariance_test", lambda *a, **k: {u: dict(excess=0.0, upper95=0.001, invariant=False,
                        boot=2000, p_below_delta=0.5, logo=0.1) for u in C.UNITS})
    monkeypatch.setattr(M, "target_vector", lambda g, e: list(np.random.default_rng(abs(hash(e)) % 2**32)
                                                               .integers(0, 2, 32).astype(float)))
    seen, fits = [], []
    real_sel, real_fit = Pr.select, M.fit_domain
    monkeypatch.setattr(Pr, "select", lambda recs, *a, **k: seen.append(list(recs)) or real_sel(recs, *a, **k))
    monkeypatch.setattr(M, "fit_domain", lambda recs, vec, d, *a, **k: fits.append((list(recs), vec, d))
                        or real_fit(recs, vec, d, *a, **k))
    res = C.build(dict(C.PHASE_MAPS["calibration"], levels=None), tmp_path, boot=20, allow_incomplete=True)
    assert len(fits) == 1 and fits[0][2] == "bool"
    key = lambda rs: sorted((r["set"], r["target_id"], r["budget"], r["seed"]) for r in rs)   # noqa: E731
    assert key(fits[0][0]) == key(seen[0])
    have = {r["budget"] for r in seen[0]}                                # the synthetic runs reach 1, 2 and 4 x E0
    assert have == {1, 2, 4} and set(C.H4_BUDGETS) > have
    assert set(fits[0][1]) == {(r["set"], r["target_id"]) for r in seen[0]}
    c2 = res["h7"]["bool"]["comparator2"]
    assert set(c2["budgets"]) == {str(b) for b in have} and c2["left_out"] == []
    assert c2["budgets_without_model"] == sorted(set(C.H4_BUDGETS) - have)
    assert all("ref_vectors" not in mb and "refs" in mb for mb in c2["budgets"].values())
    line = next(x for x in C.render(res, "t", "s").splitlines() if x.startswith("- **bool:** unit "))
    assert f"Graff and Poli (2010) comparator, d {c2['measure']}" in line


def test_lar_survives_raw_real_targets_of_extreme_scale():
    """Guardrail row 117: on raw grid values of R-ELEM4's calibration targets (|t| up to about 4e13; target files only,
    synthetic outcomes) the mse columns are numerically collinear and the bordered inverse broke down (division by
    zero). The path must stop cleanly, and the steps it takes must still be equiangular."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    g = G.get("R-ELEM4")
    ts = json.loads((root / "data/targets_s2/R-ELEM4/calibration.json").read_text())
    V = np.array([M.target_vector(g, t["expression"]) for t in ts])
    y = np.random.default_rng(0).random(len(V))
    X = M.distances("mse", V, V)
    order, mus = M.lar_order(X, y, trace=True)
    assert 1 <= len(order) < len(V)
    Xs = (X - X.mean(0)) / np.sqrt(((X - X.mean(0)) ** 2).sum(0))
    for k in range(1, len(order)):
        c = np.abs(Xs.T @ (y - y.mean() - mus[k]))
        assert c[order[:k + 1]].min() >= c[order[:k + 1]].max() * (1 - 1e-6)

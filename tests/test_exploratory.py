"""The four exploratory analyses filed in Stage 1 ("Exploratory and labelled: a grammar-level explanation of set
intercepts; Boolean Walsh-spectrum measures; the shape of the allocation curve; bloat"; design at 3b8aa5e, 9.6 and
hypotheses.md "Exploratory"), built by D083 in `reach.analysis.s2.exploratory` (choices in D085). Synthetic data only;
never the calibration runs."""
import json
import math

import numpy as np
import pytest

from reach.analysis.s2 import exploratory as X

E0 = 20_000


def _sig(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------- Walsh-spectrum measures
def test_walsh_measures_by_hand():
    """F = (-1)^f, W(w) = sum_x F(x) (-1)^(w.x); energy share W^2 / 4^n. One input variable: all energy at one order-1
    coefficient. AND of two inputs: W = (2, 2, 2, -2), shares 1/4 each: mean order (0 + 1 + 1 + 2) / 4 = 1, degree 2,
    entropy 2 bits, nonlinearity 2 - 2/2 = 1. Parity of 3: all energy at order 3."""
    m = X.walsh_measures([0, 1, 0, 1], 2)                       # f = the input in bit 0
    assert m == dict(nonlinearity=0.0, walsh_order=1.0, walsh_degree=1.0, spectral_entropy=0.0, essential_inputs=1.0)
    m = X.walsh_measures([0, 0, 0, 1], 2)
    assert m == pytest.approx(dict(nonlinearity=1.0, walsh_order=1.0, walsh_degree=2.0, spectral_entropy=2.0,
                                   essential_inputs=2.0))
    par = [bin(r).count("1") & 1 for r in range(8)]
    assert X.walsh_measures(par, 3) == pytest.approx(dict(nonlinearity=0.0, walsh_order=3.0, walsh_degree=3.0,
                                                          spectral_entropy=0.0, essential_inputs=3.0))


def _bool_recs(theta=-1.0, seed=0, sets=("BOOL-NAND-n5", "BOOL-NOR-n5", "BOOL-AON-n5"), n=40, runs=5):
    """E0 records: logit P = a_G - 0.5 K + theta * z, z a planted measure (standardised), independent of K."""
    rng = np.random.default_rng(seed)
    recs, meas = [], {}
    for j, g in enumerate(sets):
        for t in range(n):
            K = int(rng.integers(5, 21))
            z = float(rng.normal())
            tid = f"{g}_{t}"
            meas[(g, tid)] = dict(nonlinearity=z, walsh_order=float(rng.normal()), walsh_degree=float(K),
                                  spectral_entropy=float(rng.normal()), essential_inputs=float(rng.integers(1, 6)))
            p = _sig(5.0 + 0.3 * j - 0.5 * K + theta * z)
            for s in range(runs):
                recs.append(dict(set=g, target_id=tid, seed=s, K=K, budget=1, units=dict(nodes=float(K),
                                 bits_sem=1.3 * K), y=int(rng.random() < p)))
    return recs, meas


def test_walsh_effect_beyond_size_is_recovered():
    recs, meas = _bool_recs(theta=-1.0)
    res = X.walsh(recs, "bits_sem", meas, boot=100)
    nl = res["measures"]["nonlinearity"]
    assert nl["theta"] < 0 and nl["hi"] < 0 and nl["cv_logloss_gain"] > 0
    wo = res["measures"]["walsh_order"]                          # no effect planted: the CI covers 0
    assert wo["lo"] < 0 < wo["hi"]
    assert res["measures"]["walsh_degree"]["spearman_size"] == pytest.approx(1.0)
    assert res["unit"] == "bits_sem" and res["targets"] == 120 and res["sets"] == 3


# ---------------------------------------------------------------- grammar-level intercepts (design 9.6)
def _e0_recs(seed=1, n_sets=6, n=30, runs=5):
    rng = np.random.default_rng(seed)
    sets = ["F-EML", "R-ARITH4", "R-ELEM4", "R-HYB", "F-LDE0", "R-TRIG", "F-EML-R"][:n_sets]
    prop1 = {g: float(i) for i, g in enumerate(sets)}
    prop2 = {g: float(rng.normal()) for g in sets}
    alpha = {g: 2.0 + 1.5 * prop1[g] for g in sets}
    recs = []
    for g in sets:
        for t in range(n):
            K = int(rng.integers(3, 30))
            for s in range(runs):
                recs.append(dict(set=g, target_id=f"{g}_{t}", seed=s, K=K, budget=1, units=dict(nodes=float(K)),
                                 y=int(rng.random() < _sig(alpha[g] - 0.5 * K))))
    feats = {g: dict(alphabet=prop1[g], redundancy=prop2[g], n_vars=1) for g in sets}
    return recs, feats, alpha


def test_intercepts_recover_set_alphas_and_choose_the_explaining_property():
    recs, feats, alpha = _e0_recs()
    res = X.intercepts(recs, "nodes", feats, boot=50)
    assert res["properties"] == ["alphabet", "redundancy"]                    # n_vars is constant: not usable
    for g, a in alpha.items():
        e = res["alpha"][g]
        assert e["est"] == pytest.approx(a, abs=1.5) and e["lo"] < e["est"] < e["hi"] and e["lo"] < a < e["hi"]
        assert e["k50"] == pytest.approx(e["est"] / res["beta"])
    assert res["beta"] == pytest.approx(0.5, abs=0.1)
    assert res["chosen"]["properties"] == ["alphabet"] and res["chosen"]["coef"][1] == pytest.approx(1.5, abs=0.5)
    names = [c["properties"] for c in res["candidates"]]
    assert [] in names and ["alphabet", "redundancy"] in names                # pairs allowed with 6 sets
    null = next(c for c in res["candidates"] if c["properties"] == [])
    assert res["chosen"]["loso_rmse"] < null["loso_rmse"]
    small = X.intercepts([r for r in recs if r["set"] in ("F-EML", "R-ARITH4", "R-ELEM4", "R-HYB", "F-LDE0")],
                         "nodes", feats, boot=10)
    assert all(len(c["properties"]) <= 1 for c in small["candidates"])       # fewer than 6 sets: one property at most


def test_separated_fits_are_not_estimable():
    """Guardrail row 118 in the exploratory fits: a set in which every run is exact has no finite alpha_G and is left
    out and named; a measure that separates the outcomes has no finite theta; such bootstrap replicates are dropped."""
    recs, feats, alpha = _e0_recs()
    recs = [dict(r, y=1) if r["set"] == "R-TRIG" else r for r in recs]
    res = X.intercepts(recs, "nodes", feats, boot=20)
    assert res["separated_sets"] == ["R-TRIG"] and "R-TRIG" not in res["sets"]
    assert math.isnan(res["alpha"]["R-TRIG"]["est"]) and "separated" in res["alpha"]["R-TRIG"]["note"]
    assert res["alpha"]["F-EML"]["est"] == pytest.approx(alpha["F-EML"], abs=1.5)
    b, meas = _bool_recs(theta=0.0)
    for r in b:                                           # nonlinearity now decides every run: complete separation
        meas[(r["set"], r["target_id"])]["nonlinearity"] = 1.0 if r["target_id"].endswith(("0", "2", "4")) else 0.0
    rows = [dict(r, y=int(meas[(r["set"], r["target_id"])]["nonlinearity"] == 1.0)) for r in b]
    w = X.walsh(rows, "bits_sem", meas, boot=20)
    nl = w["measures"]["nonlinearity"]
    assert math.isnan(nl["theta"]) and not nl["estimable"] and nl["dropped"] == 20


def test_fit_refuses_separation_even_when_the_solver_returns_numbers(monkeypatch):
    """Row 118: k50.logit stopped at its iteration limit with a large finite slope. `_fit` must not depend on the solver
    failing: with a solver that always returns numbers, a separated design still gives None, an overlapping one not."""
    monkeypatch.setattr(X.law, "_irls", lambda Xd, y: np.full(Xd.shape[1], 25.0))
    x = np.array([3] * 10 + [4] * 10 + [6] * 10 + [7] * 10, float)
    Xd = np.column_stack([np.ones(40), -x])
    assert X._fit(Xd, np.array([1] * 20 + [0] * 20, float)) is None
    assert X._fit(Xd, np.array([1, 0] * 20, float)) is not None


def test_loso_rmse_by_hand():
    """Three sets, y = (1, 2, 4), x = (0, 1, 2): leaving out each set and fitting a line through the other two predicts
    0, 2.5 and 3 against 1, 2, 4: RMSE = sqrt((1 + 0.25 + 1) / 3). The null model predicts the others' mean: 3, 2.5,
    1.5: RMSE = sqrt((4 + 0.25 + 6.25) / 3)."""
    y, x = np.array([1.0, 2.0, 4.0]), np.array([[0.0], [1.0], [2.0]])
    assert X.loso_rmse(y, x) == pytest.approx(math.sqrt((1 + 0.25 + 1) / 3))
    assert X.loso_rmse(y, np.zeros((3, 0))) == pytest.approx(math.sqrt((4 + 0.25 + 6.25) / 3))


# ---------------------------------------------------------------- allocation-curve shape
def _alloc_runs(kappa, seed=2, n=20, seeds=32, gens=200):
    """Runs of 200 generations (and 3,200 for the first 5 seeds) with a Weibull first-hit time of shape kappa whose
    scale grows with size K."""
    rng = np.random.default_rng(seed)
    out = []
    for t in range(n):
        K = int(rng.integers(6, 20))
        scale = 20 * 2 ** ((K - 6) / 2)
        for s in range(seeds):
            T = scale * (-math.log(rng.random())) ** (1 / kappa)
            g = 3200 if s < 5 else gens
            ge = math.ceil(T) if T < g else None
            out.append(dict(set="F-EML", target_id=f"t{t}", seed=s, K=K, generations=g, gen_exact=ge,
                            gens_reached=ge if ge is not None else g,
                            terminal_state="converged" if ge is not None else "budget_exhausted"))
    return out


def test_allocation_curve_shape_follows_kappa():
    """kappa < 1 (decreasing hazard): restarts help, the shortest split is best and the curve falls; kappa > 1: the
    longest run is best and the curve rises (T4 of the theory model)."""
    from reach.analysis.s2 import alloc
    short = X.allocation_shape(_alloc_runs(0.3), budgets=(4,), boot=40)["4"]
    assert short["best_length"] == 25 and short["shape"] == "falls"
    long_ = X.allocation_shape(_alloc_runs(2.5), budgets=(4,), boot=40)["4"]
    assert long_["best_length"] == 800 and long_["shape"] == "rises"
    assert list(long_["k50"]) == ["25", "50", "100", "200", "800"]               # whole budget as one run (5 seeds)
    runs = _alloc_runs(2.5)
    k, _ = alloc._table(alloc._per_target(runs), 4, (25, 50, 100, 200))
    assert all(long_["k50"][str(L)] == pytest.approx(k[L]) for L in (25, 50, 100, 200))
    assert sum(long_["chosen_share"].values()) == pytest.approx(1.0)


def test_whole_budget_point_uses_only_runs_that_reached_it():
    """A 200-generation run says nothing about success by 800 generations: only the 3,200-generation runs enter."""
    runs = _alloc_runs(2.5)
    pt = X.whole_budget_k50(runs, 4)
    only = [r for r in runs if r["generations"] >= 800]
    assert pt == pytest.approx(X.whole_budget_k50(only, 4)) and pt["runs"] == len(only)


# ---------------------------------------------------------------- bloat
def _bloat_rows():
    rows = []
    for t in range(20):
        K = 5 + t % 5
        for s, g in enumerate((200, 200, 3200, 3200)):
            fail = dict(set="F-EML", algorithm="A", target_id=f"t{t}", seed=s, K_construction=K, K_proven=K,
                        generations=g, gens_reached=g, gen_exact=None, exact=False, terminal_state="budget_exhausted",
                        final_tree_size=40 if g == 200 else 320)
            rows.append(fail)
        rows.append(dict(fail, seed=9, exact=True, gen_exact=10 + t, gens_reached=10 + t, terminal_state="converged",
                         final_tree_size=K + (2 if t < 10 else 0)))
        rows.append(dict(fail, seed=10, gens_reached=50, terminal_state="timeout", final_tree_size=5000))   # censored
    return rows


def test_bloat_summary_and_growth_by_hand():
    """Exact programs are K + 2 nodes for half of the targets and minimal for the rest; failed runs end at 40 nodes
    after 200 generations and 320 after 3,200 (8 times as large); CPU-stopped runs are left out."""
    res = X.bloat(_bloat_rows(), boot=50)
    cell = {(c["algorithm"], c["generations"]): c for c in res["cells"]}
    assert cell[("A", 200)]["failed"] == 40 and cell[("A", 200)]["failed_median_size"] == 40
    assert cell[("A", 3200)]["failed_median_size"] == 320
    assert (cell[("A", 3200)]["runs"], cell[("A", 3200)]["failed"], cell[("A", 3200)]["exact"]) == (60, 40, 20)
    ex = res["exact"]["A"]
    assert ex["n"] == 20 and ex["share_minimal"] == 0.5 and ex["median_excess"] == 1.0
    g = res["growth"]["A"]
    assert g["ratio"] == pytest.approx(8.0) and g["lo"] == pytest.approx(8.0) and g["hi"] == pytest.approx(8.0)
    assert (g["short"], g["long"]) == (200, 3200) and res["left_out_censored"] == 20


# ---------------------------------------------------------------- entry point, labels and isolation
OPS = {"BOOL-NAND-n5": "nand(x1, x2)", "BOOL-NOR-n5": "nor(x1, x2)", "BOOL-AON-n5": "and(x1, x2)"}


def _pilot_root(tmp_path):
    from test_dryrun import SETS, _rows, _write
    rows = [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0, i)]
    for r in rows:
        r["final_tree_size"] = int(r["K_construction"]) + 3
        r["exact"] = r["gen_exact"] is not None
    _write(tmp_path, "s2_pilot_reach_a", rows)
    _write(tmp_path, "s2_pilot_mech", [])
    for g in SETS:
        d = tmp_path / f"data/targets_s2/{g}"
        d.mkdir(parents=True)
        tids = sorted({r["target_id"] for r in rows if r["set"] == g})
        (d / "pilot.json").write_text(json.dumps([dict(target_id=t, units=rows[0]["units"],
                                                       expression=OPS[g] if "_K6" in t else "x3")
                                                  for t in tids]))
    (tmp_path / "data/analysis").mkdir(parents=True)
    (tmp_path / "data/analysis/properties.json").write_text(json.dumps(
        {g: dict(alphabet=float(i), redundancy=float(i * i)) for i, g in enumerate(SETS)}))
    (tmp_path / "docs/status").mkdir(parents=True)
    return rows


def test_dryrun_entry_point_writes_labelled_reports(tmp_path):
    _pilot_root(tmp_path)
    X.main(["--dryrun", "--boot", "20"], root=tmp_path)
    res = json.loads((tmp_path / "data/analysis/dryrun_exploratory.json").read_text())
    assert res["label"].startswith("EXPLORATORY") and "pilot" in res["label"]
    assert set(res) >= {"intercepts", "walsh", "allocation", "bloat"}
    for name in X.REPORTS:
        text = (tmp_path / f"docs/status/dryrun_exploratory_{name}.md").read_text()
        assert text.startswith("# EXPLORATORY") and "not confirmatory" in text and len(text.splitlines()) <= 200
    assert "bool" in res["intercepts"] and "BOOL-NAND-n5" in res["allocation"]
    assert res["walsh"]["targets"] > 0 and res["bloat"]["BOOL-NAND-n5"]["exact"]["A"]["median_excess"] == 3.0


def test_calibration_mode_reads_the_calibration_phases_and_writes_its_own_files(tmp_path, monkeypatch):
    """Calibration mode reads s2_cal_* (tested on a temporary root only) and never writes the confirmatory report."""
    from test_dryrun import SETS, _rows, _write
    rows = [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0, i)]
    for r in rows:
        r["final_tree_size"] = 9
    _write(tmp_path, "s2_cal_a16", rows)
    for g in SETS:
        d = tmp_path / f"data/targets_s2/{g}"
        d.mkdir(parents=True)
        tids = sorted({r["target_id"] for r in rows if r["set"] == g})
        (d / "calibration.json").write_text(json.dumps([dict(target_id=t, units=rows[0]["units"], expression="x1")
                                                        for t in tids]))
    (tmp_path / "docs/status").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="confirmatory calibration analysis first"):
        X.main(["--calibration", "--boot", "10"], root=tmp_path)          # the H2-preferred unit comes from it
    (tmp_path / "data/analysis").mkdir(parents=True)
    (tmp_path / "data/analysis/calibration.json").write_text(json.dumps({"h2": {"bool": {"report_unit": "nodes"}}}))
    seen = []
    real = X.confirm.read_phase
    monkeypatch.setattr(X.confirm, "read_phase", lambda ph, root=None: seen.append(ph) or real(ph, root))
    res = X.main(["--calibration", "--boot", "10"], root=tmp_path)
    assert "s2_cal_a16" in seen and not any(p.startswith("s2_pilot") for p in seen)
    assert res["units"]["bool"]["unit"] == "nodes" and res["walsh"]["unit"] == "nodes"
    assert (tmp_path / "data/analysis/calibration_exploratory.json").exists()
    assert (tmp_path / "docs/status/calibration_exploratory_bloat.md").read_text().startswith("# EXPLORATORY")
    assert not (tmp_path / "docs/status/calibration_report.md").exists()


def test_exploratory_analyses_stay_out_of_the_confirmatory_report(tmp_path):
    """The confirmatory analysis neither imports nor reports them (Stage 1: exploratory and labelled)."""
    import inspect
    from reach.analysis.s2 import confirm, dryrun
    for mod in (confirm, dryrun):
        assert "s2.exploratory" not in inspect.getsource(mod) and "import exploratory" not in inspect.getsource(mod)
    _pilot_root(tmp_path)
    text = dryrun.render(dryrun.build(tmp_path, delta=0.01, boot=20))
    assert "Walsh" not in text and "bloat" not in text.lower() and "allocation-curve" not in text


def test_reports_fit_200_lines_at_calibration_size():
    """At calibration size (allocation: 16 sets x 5 budgets; bloat: 16 sets x 3 algorithms x 2 run lengths;
    intercepts: 7 real and 9 Boolean sets with 6 properties, every candidate model) each report stays within 200
    lines."""
    from reach.runner.phases import CAL_SETS
    shape = X.allocation_shape(_alloc_runs(0.3), budgets=(1, 2, 4, 8, 16), boot=5)
    rows = _bloat_rows()
    rows += [dict(r, algorithm=a, generations=200, gens_reached=min(r["gens_reached"], 200)) for a in ("B", "HC")
             for r in rows if r["generations"] == 200]
    bl = X.bloat(rows, boot=5)
    props = ("alphabet", "redundancy", "bits_per_node", "undefined_share_initial", "neutral_share_mutation", "n_vars")
    inter = {}
    for d, n in (("real", 7), ("bool", 9)):
        recs, feats, _ = _e0_recs(n_sets=min(n, 7))
        feats = {g: {p: float(np.random.default_rng(hash((g, p)) % 2**32).normal()) for p in props} for g in feats}
        inter[d] = X.intercepts(recs, "nodes", feats, boot=5)
    recs, meas = _bool_recs()
    res = dict(label="EXPLORATORY (test)", intercepts=inter, walsh=X.walsh(recs, "bits_sem", meas, boot=5),
               allocation={g: shape for g in CAL_SETS}, bloat={g: bl for g in CAL_SETS}, units={})
    assert len(bl["cells"]) == 4 and all(len(v["candidates"]) == 22 for v in inter.values())
    for name in X.REPORTS:
        assert len(X.render(res, name, "t").splitlines()) <= 200, name

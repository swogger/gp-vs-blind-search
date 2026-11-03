"""H10a and H10b analysis hooks (D076; REQ-ANA-26, REQ-ANA-27), written before the code. Synthetic data only."""
import copy
import hashlib
import json

import numpy as np
import pytest

from reach.analysis.s2 import cgp_confirm as C
from reach.analysis.s2 import predictor

E0 = 20_000


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _rows(g, a, b, levels=(5, 7, 9, 11, 13, 15), n_targets=30, seeds=5, seed=0, saturated=False):
    """CGP rows with P(exact by E0) = sig(a + b K) and P(exact by 16 E0) = sig(a + 2 + b K), prefix-consistent. A
    planted bits_sem that falls with K: a test in that unit would see the opposite slope (H10a is in nodes)."""
    rng = np.random.default_rng(seed)
    rows = []
    for K in levels:
        for i in range(n_targets):
            for s in range(seeds):
                u = rng.random()
                if saturated:
                    ev = E0 // 4
                else:
                    ev = E0 if u < _sig(a + b * K) else 16 * E0 if u < _sig(a + 2 + b * K) else None
                rows.append(dict(algorithm="CGP", set=g, target_id=f"{g}_{K}_{i}", seed=s, config="n500",
                                 K_construction=K, K_proven=K, units=dict(nodes=float(K), bits_sem=100.0 - 2 * K), budget=16,
                                 budget_E=16 * E0, exact=ev is not None, evals_exact=ev, evals_reached=16 * E0,
                                 terminal_state="converged" if ev else "budget_exhausted"))
    return rows


def test_h10a_tests_every_set_at_e0_and_16x_on_nodes():
    rows = (_rows("F-EML", 3.0, -0.5, seed=1) + _rows("BOOL-NAND-n5", 3.0, -0.4, seed=2)
            + _rows("BOOL-NOR-n5", -6.0, 0.5, seed=3)                       # planted: recovery rises with size
            + _rows("BOOL-AON-n5", 0, 0, saturated=True))                   # every run exact: no slope
    res = C.h10a(rows, boot=200)
    cells = {(c["set"], c["budget"]): c for c in res["cells"]}
    assert set(cells) == {(g, b) for g in ("F-EML", "BOOL-NAND-n5", "BOOL-NOR-n5", "BOOL-AON-n5") for b in (1, 16)}
    for g in ("F-EML", "BOOL-NAND-n5"):
        for b in (1, 16):
            c = cells[(g, b)]
            assert c["supports"] and c["hi"] < 0 and c["estimable"] and c["unit"] == "nodes" and c["algorithm"] == "CGP"
    assert not cells[("BOOL-NOR-n5", 1)]["supports"] and cells[("BOOL-NOR-n5", 1)]["lo"] > 0
    sat = cells[("BOOL-AON-n5", 1)]
    assert sat["untestable"] and not sat["supports"] and not sat["estimable"]       # reported, never dropped
    assert cells[("F-EML", 1)]["domain"] == "real" and cells[("BOOL-NAND-n5", 16)]["domain"] == "bool"
    s = res["summary"]
    assert s["real"] == dict(cells=2, supported=2, untestable=0)
    assert s["bool"] == dict(cells=6, supported=2, untestable=2)
    # E0 and 16 x E0 are read from the same runs by prefix: the 16x slope sits to the right (higher recovery)
    assert cells[("F-EML", 16)]["slope"] < 0


def test_h10a_refuses_rows_of_other_engines():
    rows = _rows("F-EML", 3.0, -0.5, levels=(5, 7), n_targets=3, seeds=1)
    rows[0] = dict(rows[0], algorithm="A")                                   # planted: one tree-GP row
    with pytest.raises(ValueError, match="CGP"):
        C.h10a(rows, boot=10)


FROZEN = dict(unit="nodes", budget_term=True, feature=None, params=[4.0, -0.5, 0.6], loso_logloss=0.3,
              sha256="f" * 64)


def _recs(sets, c0, c1, seed, levels=(3, 5, 7, 9, 11, 13), n_targets=20, shift=None):
    """Records (as load.tidy gives them) with logit P = c0 + c1 * eta, eta from FROZEN."""
    rng = np.random.default_rng(seed)
    out = []
    for g in sets:
        for K in levels:
            for i in range(n_targets):
                for b in C.FIT_BUDGETS:
                    eta = FROZEN["params"][0] + FROZEN["params"][1] * K + FROZEN["params"][2] * np.log2(b)
                    p = _sig(c0 + (shift or {}).get(g, 0.0) + c1 * eta)
                    out.append(dict(set=g, target_id=f"{g}_{K}_{i}", seed=0, algorithm="CGP", K=K,
                                    units=dict(nodes=float(K)), budget=b, y=int(rng.random() < p)))
    return out


def test_h10b_transfer_recovers_the_planted_intercept_and_slope():
    frozen = copy.deepcopy(FROZEN)
    cal = _recs([f"S{i}" for i in range(6)], -1.0, 0.7, seed=4)
    t = C.fit_transfer(frozen, cal)
    assert frozen == FROZEN                                                  # the frozen tree model is untouched
    assert t["transfer"]["c0"] == pytest.approx(-1.0, abs=0.2) and t["transfer"]["c1"] == pytest.approx(0.7, abs=0.1)
    assert t["base_sha256"] == FROZEN["sha256"] and t["unit"] == "nodes" and t["budget_term"] is True
    assert t["budgets"] == list(C.FIT_BUDGETS) and t["n"] == len(cal)
    # the composite model is c0 + c1 * eta exactly, and predictor.predict reads it unchanged
    eta = np.log(predictor.predict(FROZEN, cal) / (1 - predictor.predict(FROZEN, cal)))
    want = _sig(t["transfer"]["c0"] + t["transfer"]["c1"] * eta)
    assert np.allclose(predictor.predict(t, cal), want)
    # only budgets 1-16 x E0 enter the fit; records of other algorithms are refused
    with pytest.raises(ValueError, match="CGP"):
        C.fit_transfer(FROZEN, [dict(cal[0], algorithm="A")] + cal[1:])
    quarter = [dict(r, budget=0.25) for r in cal[:50]]
    assert C.fit_transfer(FROZEN, cal + quarter)["n"] == len(cal)


def test_h10b_scores_with_the_h7_criteria():
    sets = [f"S{i}" for i in range(6)]
    cal = _recs(sets, -1.0, 0.7, seed=5)
    t = C.fit_transfer(FROZEN, cal)
    good = C.h10b(t, cal, _recs([f"H{i}" for i in range(6)], -1.0, 0.7, seed=6), boot=200)
    assert good["passes"] and good["auc"] >= 0.8 and good["sets_ok"] >= 5
    assert good["criteria"] == dict(auc=0.80, slope=[0.7, 1.3], eps_set=0.10, min_sets_ok=5)
    base = predictor.fit_baseline(cal, budget_term=True)    # D090: nodes + log2 budget, on CGP calibration runs
    assert good["baseline"] == t["baseline"] and t["baseline"]["params"] == pytest.approx(base["params"])
    # planted: two held-out sets shifted by 2 logits fail the per-set criterion (at least 5 of 6 needed)
    ho = _recs([f"H{i}" for i in range(6)], -1.0, 0.7, seed=6, shift={"H0": 2.0, "H1": 2.0})
    bad = C.h10b(t, cal, ho, boot=200)
    assert not bad["passes"] and bad["sets_ok"] <= 4
    # the untransferred tree model (c0 = 0, c1 = 1) is badly calibrated on these CGP runs
    raw = predictor.score(FROZEN, _recs([f"H{i}" for i in range(6)], -1.0, 0.7, seed=6), base, 0.10, boot=50,
                          min_sets_ok=5)
    assert not raw["passes"]


def test_h10b_transfer_is_frozen_with_its_checksum(tmp_path, monkeypatch):
    from reach.runner import phases as P
    cal = _recs([f"S{i}" for i in range(6)], -1.0, 0.7, seed=7)
    models = {"real": C.fit_transfer(FROZEN, cal), "bool": C.fit_transfer(FROZEN, cal)}
    reg = tmp_path / "data/registration"
    reg.mkdir(parents=True)
    out = C.freeze_transfer(models, reg / "cgp_transfer.json")
    body = (reg / "cgp_transfer.json").read_bytes()
    assert (reg / "cgp_transfer.sha256").read_text().split()[0] == hashlib.sha256(body).hexdigest() == out["sha256"]
    assert json.loads(body)["real"]["transfer"] == models["real"]["transfer"]
    C.freeze_transfer(models, tmp_path / "again.json")
    assert (tmp_path / "again.json").read_bytes() == body                     # reproducible
    monkeypatch.setattr(P, "ROOT", tmp_path)
    assert P.cgp_transfer_frozen()["real"]["transfer"] == models["real"]["transfer"]   # the phase guard accepts it


def test_h10b_transfer_freeze_refuses_replacement_and_invalid_models(tmp_path):
    """Guardrail row 91: predictor.freeze overwrote a frozen transfer model with other content, and the phase guard
    checked only the file against its own .sha256, so {"real": {}, "bool": {}} passed (independent review)."""
    cal = _recs([f"S{i}" for i in range(6)], -1.0, 0.7, seed=8)
    models = {"real": C.fit_transfer(FROZEN, cal), "bool": C.fit_transfer(FROZEN, cal)}
    C.check_transfer(models)                                                 # control: a fitted pair is valid
    path = tmp_path / "cgp_transfer.json"
    C.freeze_transfer(models, path)
    body, sha = path.read_bytes(), path.with_suffix(".sha256").read_bytes()
    C.freeze_transfer(copy.deepcopy(models), path)                           # same content: no error, no change
    other = dict(models, real=C.fit_transfer(FROZEN, _recs([f"S{i}" for i in range(6)], -0.5, 0.9, seed=9)))
    with pytest.raises(RuntimeError, match="never replaced"):                # planted: a different, valid model
        C.freeze_transfer(other, path)
    assert path.read_bytes() == body and path.with_suffix(".sha256").read_bytes() == sha
    good = models["real"]
    bad_models = [
        {"real": {}, "bool": {}},                                             # the review's example
        {"real": good},                                                       # a domain missing
        dict(models, extra=good),                                             # an unknown key
        dict(models, bool=dict(good, params=[float("nan")] + good["params"][1:])),
        dict(models, bool=dict(good, params=good["params"][:-1])),            # wrong number of coefficients
        dict(models, bool=dict(good, params=[good["params"][0] + 1.0] + good["params"][1:])),   # not c0 + c1 * eta
        dict(models, bool=dict(good, transfer=dict(c0=good["transfer"]["c0"]))),
        dict(models, bool=dict(good, unit="leaves")),
        dict(models, bool=dict(good, budgets=[1, 16])),                       # not the registered fit budgets
    ]
    for i, bad in enumerate(bad_models):
        with pytest.raises(ValueError):
            C.check_transfer(bad)
        with pytest.raises(ValueError):
            C.freeze_transfer(bad, tmp_path / f"bad{i}.json")
        assert not (tmp_path / f"bad{i}.json").exists()


def test_h10b_uses_the_registered_bootstrap_count(monkeypatch):
    """Guardrail row 92: h10b defaulted to 1,000 bootstrap replicates; Stage 1 registers 2,000 (confirm.REGISTERED)."""
    from reach.analysis.s2 import confirm
    cal = _recs([f"S{i}" for i in range(6)], -1.0, 0.7, seed=10)
    t = C.fit_transfer(FROZEN, cal)
    seen = []
    real_score = predictor.score

    def spy(*a, **k):
        seen.append(k.get("boot"))
        return real_score(*a, **dict(k, boot=20))
    monkeypatch.setattr(predictor, "score", spy)
    C.h10b(t, cal, cal)
    assert seen == [confirm.REGISTERED["boot"]] == [2000]


# ------------------------------------------------------------------ H10a for Boolean CGP at 1/16 x E0 (D083)
def _rows_sixteenth(g, a, b, levels=(5, 7, 9, 11, 13, 15), n_targets=30, seeds=5, seed=0, all_by=None):
    """CGP rows with P(exact by E0/16) = sig(a + b K) and later hits at E0; all_by: every run exact by that many
    evaluations (a cell saturated at 1/16 x E0 when all_by <= 1,250)."""
    rng = np.random.default_rng(seed)
    rows = []
    for K in levels:
        for i in range(n_targets):
            for s in range(seeds):
                u = rng.random()
                ev = all_by if all_by else (E0 // 16 if u < _sig(a + b * K) else E0 if u < _sig(a + 3 + b * K) else None)
                rows.append(dict(algorithm="CGP", set=g, target_id=f"{g}_{K}_{i}", seed=s, config="n500",
                                 K_construction=K, K_proven=K, units=dict(nodes=float(K)), budget=16, budget_E=16 * E0,
                                 exact=ev is not None, evals_exact=ev, evals_reached=16 * E0,
                                 terminal_state="converged" if ev else "budget_exhausted"))
    return rows


def test_h10a_sixteenth_is_a_separate_boolean_family_of_nine():
    """D083: Boolean CGP at 1/16 x E0 (1,250 evaluations, read by prefix from the same runs), the H1 code, Holm over
    the 9 Boolean calibration sets; saturated and missing cells are untestable members; real rows get no cells."""
    assert C.H10A_SIXTEENTH == 1 / 16
    fam = C.declared_h10a_sixteenth_cells()
    from reach import grammar as G
    from reach.runner.phases import CAL_SETS
    assert len(fam) == 9 and {k[0] for k in fam} == {g for g in CAL_SETS if G.get(g).kind == "bool"}
    assert all(k[1:] == ("CGP", None, 1 / 16) for k in fam)
    rows = (_rows_sixteenth("BOOL-NAND-n5", 3.0, -0.5, seed=1)
            + _rows_sixteenth("BOOL-NOR-n5", -6.0, 0.5, seed=2)                    # planted: rises with size
            + _rows_sixteenth("BOOL-AON-n5", 0, 0, all_by=E0 // 32)                # every run exact by 1/16 x E0
            + _rows_sixteenth("F-EML", 3.0, -0.5, seed=3))                          # real: no family
    res = C.h10a_sixteenth(rows, boot=200)
    cells = {c["set"]: c for c in res["cells"]}
    assert set(cells) == {"BOOL-NAND-n5", "BOOL-NOR-n5", "BOOL-AON-n5"} and res["budget"] == 1 / 16
    assert all(c["budget"] == 1 / 16 and c["unit"] == "nodes" and c["algorithm"] == "CGP" for c in cells.values())
    assert cells["BOOL-NAND-n5"]["holm_supported"] and cells["BOOL-NAND-n5"]["hi"] < 0
    assert not cells["BOOL-NOR-n5"]["holm_supported"]
    assert cells["BOOL-AON-n5"]["holm_status"] == "saturated (untestable)"
    h = res["holm"]
    assert set(h) == {"bool"} and h["bool"]["m"] == 9 and len(h["bool"]["missing"]) == 6
    # the outcome is read at 1,250 evaluations: a hit at exactly E0/16 counts, one evaluation later does not
    one = _rows_sixteenth("BOOL-NAND-n5", 0, 0, levels=(5,), n_targets=1, seeds=1, all_by=E0 // 16)
    assert C.records(one, (C.H10A_SIXTEENTH,))[0]["y"] == 1
    one[0]["evals_exact"] = E0 // 16 + 1
    assert C.records(one, (C.H10A_SIXTEENTH,))[0]["y"] == 0


def test_h10a_sixteenth_leaves_h10a_and_its_families_unchanged():
    """The new family does not dilute H10a: h10a still declares 32 cells (real 14, Boolean 18), its results are the same
    with or without the 1/16 analysis, and the two families share no cell."""
    rows = (_rows_sixteenth("BOOL-NAND-n5", 3.0, -0.5, seed=1) + _rows("F-EML", 3.0, -0.5, seed=4))
    before = C.h10a(copy.deepcopy(rows), boot=100)
    six = C.h10a_sixteenth(copy.deepcopy(rows), boot=100)
    after = C.h10a(copy.deepcopy(rows), boot=100)
    assert before == after
    assert after["holm"]["real"]["m"] == 14 and after["holm"]["bool"]["m"] == 18
    assert {c["budget"] for c in after["cells"]} == {1, 16}
    keys = lambda cs: {(c["set"], c["budget"]) for c in cs}                                # noqa: E731
    assert not keys(after["cells"]) & keys(six["cells"])
    assert not set(map(tuple, C.declared_h10a_cells())) & set(map(tuple, C.declared_h10a_sixteenth_cells()))


def test_h10a_sixteenth_refuses_rows_of_other_engines():
    rows = _rows_sixteenth("BOOL-NAND-n5", 3.0, -0.5, levels=(5, 7), n_targets=3, seeds=1)
    rows[0] = dict(rows[0], algorithm="A")
    with pytest.raises(ValueError, match="CGP"):
        C.h10a_sixteenth(rows, boot=10)


def test_h10b_baseline_is_frozen_with_the_transfer_model():
    """D090: the Brier baseline (nodes + log2 budget, fitted on the CGP calibration records) is frozen in the transfer
    model and h10b scores against it, not against a refit; a transfer model without a valid baseline is refused."""
    cal = _recs([f"S{i}" for i in range(6)], -1.0, 0.7, seed=11)
    t = C.fit_transfer(FROZEN, cal)
    assert t["baseline"] == predictor.fit_baseline(cal, budget_term=True)
    ho = _recs([f"H{i}" for i in range(6)], -1.0, 0.7, seed=12)
    planted = dict(t, baseline=dict(t["baseline"], params=[0.0, 0.0, 0.0]))     # planted: a different frozen baseline
    a, b = C.h10b(t, cal, ho, boot=50), C.h10b(planted, cal, ho, boot=50)
    assert b["brier_baseline"] == pytest.approx(0.25) and a["brier_baseline"] != pytest.approx(0.25)
    models = {"real": t, "bool": t}
    C.check_transfer(models)
    for bad in (dict(t, baseline=None), dict(t, baseline=dict(t["baseline"], budget_term=False)),
                dict(t, baseline=dict(t["baseline"], params=[1.0, float("nan"), 0.0]))):
        with pytest.raises(ValueError):
            C.check_transfer({"real": bad, "bool": t})


def test_h10a_refuses_records_that_are_already_tidy():
    """Guardrail row 134 (audit 2026-10-28, finding 1): `load()` returns tidy records, and h10a and h10a_sixteenth
    tidied them again: without `evals_reached` every record became censored (a failure), so all 32 cells were
    saturated and untestable, with no error. Tidy records are refused; raw rows still work."""
    rows = _rows("F-EML", 3.0, -0.5, seed=1) + _rows_sixteenth("BOOL-NAND-n5", 2.0, -0.3, seed=4)
    recs = C.records(rows, C.FIT_BUDGETS)                                   # what load() returns
    for f in (lambda: C.h10a(recs, boot=50, declared=None), lambda: C.h10a_sixteenth(recs, boot=50, declared=None),
              lambda: C.records(recs, C.H10A_BUDGETS)):
        with pytest.raises(ValueError, match="already tidy"):
            f()
    assert C.h10a(rows, boot=50, declared=None)["summary"]["real"]["supported"] == 2      # control: raw rows

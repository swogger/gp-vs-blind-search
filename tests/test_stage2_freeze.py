"""Step 3c driver (`reach.analysis.s2.stage2_freeze`; REQ-ANA-74), written before the code. Synthetic roots only.

From the confirmatory calibration result (data/analysis/calibration.json) and calibration data only: the H7 model per
domain with its Brier baseline (D090), both comparators, the H9b transfer model and the held-out levels, each written
once with its .sha256 and never replaced with other content."""
import hashlib
import json

import numpy as np
import pytest

from reach.analysis.s2 import confirm as C
from reach.analysis.s2 import predictor
from reach.analysis.s2 import stage2_freeze as S


def _cal_root(tmp_path):
    from test_dryrun import SETS, _rows, _write
    rows = [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0, i)]
    fails = [r for r in rows if r["gen_exact"] is None]
    for r in fails[::5]:                                   # CPU-stopped runs: failures in the primary analysis
        r.update(terminal_state="timeout", gens_reached=150)
    _write(tmp_path, "s2_cal_a16", rows)
    res = C.build(dict(C.PHASE_MAPS["calibration"], levels=None), tmp_path, boot=20, allow_incomplete=True)
    return tmp_path, res


def test_output_paths_are_the_registered_ones():
    assert S.PATHS == dict(h7="data/registration/h7_model.json", gpf2013="data/registration/gpf2013_comparator.json",
                           gp2010="data/registration/gp2010_comparator.json",
                           operon="data/registration/operon_transfer.json",
                           levels="data/registration/heldout_levels.json",
                           predictions="data/registration/heldout_predictions.json")          # REQ-ANA-77 (D092)


def test_h7_records_are_those_confirm_selected_on(tmp_path):
    """The baseline must be fitted on exactly the H7 selection records: the driver rebuilds them and refits the selected
    model, which must reproduce calibration.json's coefficients; a planted change of the coefficients is refused."""
    root, res = _cal_root(tmp_path)
    h7 = res["h7"]["bool"]
    recs = S.h7_records(root)
    assert set(recs) == {"bool"} and {r["budget"] for r in recs["bool"]} == {1, 2, 4}     # synthetic runs end at 4 x E0
    S.check_reproduces(h7, recs["bool"])                                       # same records, same fit
    with pytest.raises(ValueError, match="reproduce"):
        S.check_reproduces(dict(h7, params=[h7["params"][0] + 1e-4] + h7["params"][1:]), recs["bool"])
    with pytest.raises(ValueError, match="reproduce"):                         # planted: records at another budget
        S.check_reproduces(h7, recs["bool"] + [dict(r, budget=0.25) for r in recs["bool"][:200]])


def test_h7_models_carry_the_baseline_and_are_frozen_once(tmp_path):
    root, res = _cal_root(tmp_path)
    recs = S.h7_records(root)
    models = S.h7_models(res, recs, "c" * 64)
    m = models["bool"]
    assert m["params"] == res["h7"]["bool"]["params"] and m["unit"] == res["h7"]["bool"]["selected_unit"]
    assert m["baseline"] == predictor.fit_baseline(recs["bool"], budget_term=True)
    assert m["fit_records"]["n"] == len(recs["bool"]) and m["calibration_sha256"] == "c" * 64
    path = tmp_path / "h7_model.json"
    frozen = S.freeze_h7(models, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.with_suffix(".sha256").read_text().split()[0] == sha and frozen["bool"]["sha256"] == sha
    assert S.load_h7(path)["bool"]["sha256"] == sha
    S.freeze_h7(json.loads(json.dumps(models)), path)                          # same content: no change
    other = dict(models, bool=dict(m, params=[m["params"][0] + 0.01] + m["params"][1:]))
    with pytest.raises(RuntimeError, match="never replaced"):
        S.freeze_h7(other, path)
    path.write_text(path.read_text().replace('"domain"', '"domain" '))         # planted: changed after freezing
    with pytest.raises(ValueError, match="checksum"):
        S.load_h7(path)
    with pytest.raises(ValueError):                                            # no baseline: refused (D090)
        S.freeze_h7(dict(bool=dict(m, baseline=None)), tmp_path / "other.json")


def test_failed_or_unfitted_comparators_are_refused():
    ok = dict(kind="x", budgets={"1": dict(a0=0.5, coef=[], selected=[])})
    with pytest.raises(ValueError, match="comparator"):
        S.comparator_models(dict(h7=dict(real=dict(comparator=dict(error="boom"), comparator2=ok),
                                         bool=dict(comparator=ok, comparator2=ok))), {})
    with pytest.raises(ValueError, match="comparator"):
        S.comparator_models(dict(h7=dict(real=dict(comparator=None, comparator2=ok),
                                         bool=dict(comparator=ok, comparator2=ok))), {})


def test_comparator2_gets_its_reference_vectors_back():
    c2 = dict(measure="l1", budgets={"1": dict(a0=0.3, coef=[0.1], refs=[["S", "t1"]])})
    c1 = dict(kind="graff_poli_flores_2013", budgets={"1": dict(a0=0.5, coef=[], selected=[])})
    cal = dict(h7=dict(real=dict(comparator=c1, comparator2=c2), bool=dict(comparator=c1, comparator2=c2)))
    gpf, gp = S.comparator_models(cal, {("S", "t1"): [1.0, 2.0]})
    assert gpf == dict(real=c1, bool=c1)
    assert gp["real"]["budgets"]["1"]["ref_vectors"] == [[1.0, 2.0]] and "ref_vectors" not in c2["budgets"]["1"]


def test_heldout_levels_come_from_the_frozen_model_with_each_sets_feature(tmp_path):
    """Levels per set from `freeze.heldout_levels` with the domain's frozen model, its units and its feature value;
    written once with its .sha256 and the model's SHA-256."""
    from reach.analysis.s2 import freeze as F
    model = dict(unit="bits_sem", budget_term=True, feature="z", params=[6.0, -0.5, 0.4, -3.0], rule6=True,
                 unit_source="no preferred unit, rule 6: bits_sem; D082", sha256="d" * 64)
    units = {"A": lambda K: dict(nodes=float(K), bits_sem=0.8 * K), "B": lambda K: dict(nodes=float(K), bits_sem=1.3 * K)}
    av = {g: list(range(3, 60)) for g in units}
    feats = {"A": dict(z=0.2), "B": dict(z=0.6)}
    lv = S.heldout_levels_for({"bool": model}, {"A": "bool", "B": "bool"}, units, av, feats)
    for g in ("A", "B"):
        want = F.heldout_levels(model, units[g], av[g], 200, feats[g])
        assert (lv[g]["main"], lv[g]["budget"]) == (want["main"], want["budget"]) and lv[g]["domain"] == "bool"
    assert lv["_h7_model_sha256"] == {"bool": "d" * 64}
    path = tmp_path / "heldout_levels.json"
    S.write_once(lv, path)
    assert path.with_suffix(".sha256").read_text().split()[0] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="never replaced"):
        S.write_once(dict(lv, A=dict(lv["A"], main=[3])), path)
    with pytest.raises(ValueError, match="feature"):
        S.heldout_levels_for({"bool": model}, {"A": "bool"}, units, av, {"A": {}})


def test_operon_records_are_calibration_operon_rows_at_16x(tmp_path):
    from test_dryrun import _write
    rows = []
    for g in ("R-ARITH4", "R-TRIG"):
        for K in (5, 7):
            for cfg in ("symbolic", "default"):
                rows.append(dict(key=f"{g}{K}{cfg}", set=g, target_id=f"{g}_{K}", seed=1, algorithm="operon",
                                 config=cfg, budget=16, K_construction=K, K_proven=K, exact=K == 5,
                                 units=dict(nodes=float(K), bits_uniform=1.0, bits_syn=1.0, bits_sem=2.0 * K),
                                 terminal_state="converged", cpu_s=1.0))
    _write(tmp_path, "s2_cal_operon", rows)
    recs = S.operon_records(tmp_path)
    assert len(recs) == 8 and {r["budget"] for r in recs} == {16} and {r["algorithm"] for r in recs} == {"operon"}
    assert np.mean([r["y"] for r in recs]) == 0.5


def test_committed_stage2_files_agree_with_the_calibration_result():
    """The files the driver wrote (step 4, D090) match their checksums, calibration.json and one another."""
    from reach.analysis.s2 import h9b
    from reach.paths import ROOT
    from reach.runner.phases import heldout_sets
    reg = {k: ROOT / p for k, p in S.PATHS.items()}
    for p in reg.values():
        assert p.with_suffix(".sha256").read_text().split()[0] == hashlib.sha256(p.read_bytes()).hexdigest(), p.name
    cal_path = ROOT / S.CALIBRATION
    cal = json.loads(cal_path.read_text())
    h7 = S.load_h7(reg["h7"])
    for d in ("real", "bool"):
        m, h = h7[d], cal["h7"][d]
        assert (m["unit"], m["budget_term"], m["feature"], m["params"]) == (h["selected_unit"], h["budget_term"],
                                                                            h["feature"], h["params"])
        assert m["calibration_sha256"] == hashlib.sha256(cal_path.read_bytes()).hexdigest()
        assert m["baseline"]["unit"] == "nodes" and m["baseline"]["budget_term"] is True
    lv = json.loads(reg["levels"].read_text())
    assert lv["_h7_model_sha256"] == {"real": h7["real"]["sha256"], "bool": h7["bool"]["sha256"]}
    for g in heldout_sets():
        main, budget = lv[g]["main"], lv[g]["budget"]
        assert len(set(main)) == len(set(budget)) == 5 and not set(main) & set(budget), g
    op = json.loads(reg["operon"].read_text())
    h9b.check_transfer(op)
    assert {m["base_sha256"] for m in op.values()} == {h7["real"]["sha256"]}
    assert json.loads(reg["gpf2013"].read_text()) == {d: cal["h7"][d]["comparator"] for d in ("real", "bool")}

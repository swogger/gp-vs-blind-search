"""H7's unit (D081 f, decided by D082; REQ-ANA-56; guardrail row 111).

Design at the registered commit 3b8aa5e, docs/design/analysis.md 9.7: "Candidate models: the H2-preferred unit alone;
plus the budget term; plus number of variables; plus one grammar property. Choose by the best leave-one-set-out
log-loss on calibration data". hypotheses.md, H7: "Model: fitted on calibration data only (the H2-preferred unit, the
H4 budget term, and variables chosen by a registered rule)". confirm had passed all four units to `predictor.select`.

Author decision (D082, 2026-10-27, before any calibration outcome was seen): the candidates use the H2-preferred unit
only (D079 a); with no preferred unit (D079 rule 6) they use bits_sem, labelled so in the report and the frozen model.
`predictor.select` itself is unchanged (budget term and set-level feature by the leave-one-set-out rule).
"""
import json

import pytest

from reach.analysis.s2 import confirm as C
from reach.analysis.s2 import freeze as F
from reach.analysis.s2 import invariance as I
from reach.analysis.s2 import predictor as P

FACTOR = {"BOOL-NAND-n5": 1.0, "BOOL-NOR-n5": 1.6, "BOOL-AON-n5": 2.4}


def _root(tmp_path):
    """Three sets on one curve in nodes; the bit units scale nodes by a different factor per set, so leave-one-set-out
    selection over all four units picks nodes."""
    from test_dryrun import SETS, _rows, _write
    rows = []
    for i, g in enumerate(SETS):
        for r in _rows(g, "A", 2.0, i):
            K = r["K_construction"]
            r["units"] = dict(nodes=float(K), bits_uniform=FACTOR[g] * K, bits_syn=FACTOR[g] * K + 1,
                              bits_sem=FACTOR[g] * K + 2)
            rows.append(r)
    _write(tmp_path, "s2_cal_a16", rows)
    return tmp_path


def _inv(invariant):
    return {u: dict(excess=0.0, upper95=0.001, invariant=u == invariant, boot=2000,
                    p_below_delta=0.0 if u == invariant else 0.5, logo=0.1) for u in C.UNITS}


def _build(tmp_path, monkeypatch, invariant):
    root = _root(tmp_path)
    monkeypatch.setattr(I, "invariance_test", lambda *a, **k: _inv(invariant))
    return C.build(dict(C.PHASE_MAPS["calibration"], levels=None), root, boot=20, allow_incomplete=True)


@pytest.mark.parametrize("invariant, expected", [("bits_uniform", "bits_uniform"), (None, "bits_sem")])
def test_h7_unit_is_the_h2_preferred_unit(tmp_path, monkeypatch, invariant, expected):
    res = _build(tmp_path, monkeypatch, invariant)
    assert res["h2"]["bool"]["preferred"] == invariant and res["h2"]["bool"]["report_unit"] == expected
    assert res["h7"]["bool"]["selected_unit"] == expected


@pytest.mark.parametrize("invariant, expected", [("bits_uniform", "bits_uniform"), (None, "bits_sem")])
def test_confirm_passes_one_unit_to_select(tmp_path, monkeypatch, invariant, expected):
    """D082: confirm passes the preferred unit (bits_sem under rule 6) alone; the budget term and the set-level feature
    are still chosen by `predictor.select`. Fails on the planted four-unit call."""
    seen = []
    real = P.select

    def spy(recs, units, *a, **k):
        seen.append(list(units))
        return real(recs, units, *a, **k)
    monkeypatch.setattr(P, "select", spy)
    _build(tmp_path, monkeypatch, invariant)
    assert seen == [[expected]]


@pytest.mark.parametrize("invariant", ["bits_uniform", None])
def test_report_and_frozen_model_label_the_rule6_fallback(tmp_path, monkeypatch, invariant):
    res = _build(tmp_path, monkeypatch, invariant)
    h = res["h7"]["bool"]
    rule6 = invariant is None
    assert h["rule6"] is rule6 and ("rule 6" in h["unit_source"]) is rule6 and "D082" in h["unit_source"]
    line = next(x for x in C.render(res, "t", "s").splitlines() if x.startswith("- **bool:** unit "))
    assert (f"unit {h['selected_unit']} (no preferred unit, rule 6" in line) is rule6
    assert ("H2-preferred" in line) is not rule6
    assert not any("H7: the unit" in x for x in res["not_run"])
    m = F.freeze_model(F.frozen_model(h, "bool"), tmp_path / "model_bool.json")
    body = json.loads((tmp_path / "model_bool.json").read_text())
    assert body["unit"] == m["unit"] == h["selected_unit"] and body["rule6"] is rule6 and body["domain"] == "bool"
    assert body["unit_source"] == h["unit_source"]
    assert body["params"] == h["params"] and body["budget_term"] == h["budget_term"] and body["feature"] == h["feature"]


def test_freeze_refuses_a_model_without_a_consistent_unit_source(tmp_path):
    base = dict(unit="bits_sem", budget_term=True, feature=None, params=[6.0, -0.5, 0.4], domain="bool")
    bad = [base,                                                                     # no rule-6 flag
           dict(base, unit="nodes", rule6=True, unit_source="no preferred unit (rule 6): bits_sem (D082)"),
           dict(base, rule6=True, unit_source="H2-preferred unit (D082)"),
           dict(base, rule6=False, unit_source="no preferred unit (rule 6): bits_sem (D082)")]
    for i, m in enumerate(bad):
        with pytest.raises(ValueError):
            F.freeze_model(m, tmp_path / f"m{i}.json")
    assert not any(tmp_path.iterdir())

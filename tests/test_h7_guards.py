"""H7 is not a Stage 1 test: a failure in comparator 1, comparator 2 or the H7 selection must never abort the Stage 1
report (independent review of 8e66047..4ac972b, 2026-10-27; guardrail row 119). The error is recorded in the H7 result
and in the report line; every Stage 1 result is unchanged."""
import json
import math

import pytest

from reach.analysis.s2 import gp2010, predictor
from reach.analysis.s2 import gpf2013 as M
from test_gpf2013 import _calibration_with_targets


def _build(tmp_path):
    from reach.analysis.s2 import confirm as C
    res = C.build(dict(C.PHASE_MAPS["calibration"], levels=None), tmp_path, boot=20, allow_incomplete=True)
    return json.loads(json.dumps(res, default=float)), C


def _stage1(res):
    return {k: v for k, v in res.items() if k != "h7"}


def _boom(*a, **k):
    raise FloatingPointError("planted")


@pytest.mark.parametrize("mod,key,name", [(gp2010, "comparator2", "Graff and Poli (2010)"),
                                          (M, "comparator", "Graff, Poli and Flores (2013)")])
def test_a_failing_comparator_never_aborts_the_stage1_report(tmp_path, monkeypatch, mod, key, name):
    _calibration_with_targets(tmp_path, monkeypatch)
    ok, _ = _build(tmp_path)
    monkeypatch.setattr(mod, "fit_domain", _boom)
    res, C = _build(tmp_path)
    assert _stage1(res) == _stage1(ok) and res["h1"] and res["holm"]
    h = res["h7"]["bool"]
    assert h[key]["error"] == "FloatingPointError: planted"
    other = "comparator" if key == "comparator2" else "comparator2"
    assert h[other] == ok["h7"]["bool"][other] and "error" not in h[other]
    line = next(x for x in C.render(res, "t", "s").splitlines() if x.startswith("- **bool:** unit "))
    assert f"{name} comparator failed (FloatingPointError: planted)" in line


def test_a_failing_h7_selection_never_aborts_the_stage1_report_and_is_never_frozen(tmp_path, monkeypatch):
    from reach.analysis.s2 import freeze
    _calibration_with_targets(tmp_path, monkeypatch)
    ok, _ = _build(tmp_path)
    monkeypatch.setattr(predictor, "select", lambda *a, **k: (_ for _ in ()).throw(ValueError("separated data")))
    res, C = _build(tmp_path)
    assert _stage1(res) == _stage1(ok)
    h = res["h7"]["bool"]
    assert h["error"] == "ValueError: separated data" and h["selected_unit"] is None and math.isnan(h["loso_logloss"])
    line = next(x for x in C.render(res, "t", "s").splitlines() if x.startswith("- **bool:** H7 selection failed"))
    assert "(ValueError: separated data)" in line and "Graff and Poli (2010) comparator, d " in line
    with pytest.raises(ValueError, match="no H7 model"):
        freeze.frozen_model(h, "bool")

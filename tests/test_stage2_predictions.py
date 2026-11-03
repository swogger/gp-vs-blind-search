"""Stage 2 prediction file (REQ-ANA-77; guardrail row 132): the frozen H7 model's P(exact) for every held-out target
and benchmark cell at every scored budget, written once with its SHA-256 and never replaced with other content."""
import hashlib
import json

import pytest

from reach.analysis.s2 import freeze as F
from reach.analysis.s2 import predictor
from reach.analysis.s2 import stage2_freeze as S
from reach.paths import ROOT

MODELS = {
    "real": dict(domain="real", unit="bits_sem", budget_term=True, feature=None, params=[3.0, -0.4, 0.2],
                 rule6=True, unit_source="no preferred unit, rule 6: bits_sem; D082"),
    "bool": dict(domain="bool", unit="bits_sem", budget_term=True, feature="z", params=[4.0, -0.1, 0.5, -4.0],
                 rule6=True, unit_source="no preferred unit, rule 6: bits_sem; D082"),
}
SETS = {"R-ARITH": "real", "BOOL-AN-n5": "bool"}
Z = {"BOOL-AN-n5": 0.4}


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))
    p.with_suffix(".sha256").write_text(f"{_sha(p)}  {p.name}\n")


def _root(tmp_path):
    """A repository with a frozen model file, held-out levels (with the Boolean set's feature value), held-out and
    benchmark target files with their .sha256, and the set properties."""
    reg = tmp_path / "data/registration"
    reg.mkdir(parents=True)
    body = json.dumps(MODELS, indent=1, sort_keys=True)
    (reg / "h7_model.json").write_text(body)
    (reg / "h7_model.sha256").write_text(f"{hashlib.sha256(body.encode()).hexdigest()}  h7_model.json\n")
    lv = {g: dict(main=[5, 7], budget=[9, 11], domain=d,
                  rule=dict(feature="z", feature_value=Z[g]) if d == "bool" else {}) for g, d in SETS.items()}
    _write(reg / "heldout_levels.json", lv)
    for g in SETS:
        held = [dict(target_id=f"{g}_K{K:02d}_{i:02d}", K_construction=K, units=dict(nodes=K, bits_sem=1.5 * K + i))
                for K in (5, 7, 9, 11) for i in range(3)]
        _write(tmp_path / f"data/targets_s2/{g}/heldout.json", held)
        _write(tmp_path / f"data/targets_s2/{g}/bench.json",
               [dict(target_id=f"{g}_bench_b", benchmark="b", K_construction=7, units=dict(nodes=7, bits_sem=9.0))])
    props = {g: dict(z=Z.get(g, 0.1)) for g in SETS}
    (tmp_path / "data/analysis").mkdir(parents=True)
    (tmp_path / "data/analysis/properties.json").write_text(json.dumps(props))
    return tmp_path


@pytest.fixture
def features(monkeypatch):
    from reach.analysis.s2 import confirm as C
    monkeypatch.setattr(C, "set_features", lambda root, sets: {g: {"z": Z.get(g, 0.1)} for g in sets})


def test_prediction_file_is_never_replaced_with_other_content(tmp_path):
    """Guardrail row 132: `freeze.write_predictions` overwrote an existing prediction file silently (the frozen models
    refuse, row 123). Same content: no change; other content: refused, the file unchanged."""
    m = F.freeze_model(dict(MODELS["real"]), tmp_path / "model.json")
    t = [dict(set="S", target_id=f"t{K}", units=dict(nodes=float(K), bits_sem=1.6 * K)) for K in (5, 9)]
    p = F.write_predictions(m, t, (1, 16), tmp_path / "pred.json")
    before = p.read_bytes()
    F.write_predictions(m, t, (1, 16), tmp_path / "pred.json")                # same content: allowed, unchanged
    assert p.read_bytes() == before
    with pytest.raises(RuntimeError, match="never replaced"):
        F.write_predictions(m, t[:1], (1, 16), tmp_path / "pred.json")
    assert p.read_bytes() == before and (tmp_path / "pred.sha256").read_text().split()[0] == _sha(p)


def test_prediction_file_covers_every_heldout_target_and_benchmark_cell(tmp_path, features):
    root = _root(tmp_path)
    sha = S.write_prediction_file(root)
    p = root / S.PATHS["predictions"]
    assert (root / "data/registration/heldout_predictions.sha256").read_text().split()[0] == sha == _sha(p)
    out = json.loads(p.read_text())
    model_sha = _sha(root / "data/registration/h7_model.json")
    assert out["model_sha256"] == model_sha and out["budgets"] == list(predictor.SCORED["H7"]["budgets"])
    rows = out["predictions"]
    assert len(rows) == 2 * (12 + 1) * 5 and all(r["model_sha256"] == model_sha for r in rows)
    assert {r["kind"] for r in rows} == {"heldout", "bench"}
    for g, d in SETS.items():                                                 # P from the frozen model of its domain
        for kind in ("heldout", "bench"):
            ts = json.loads((root / f"data/targets_s2/{g}/{kind}.json").read_text())
            for t in ts:
                got = sorted((r["budget"], r["p"]) for r in rows if r["set"] == g and r["target_id"] == t["target_id"])
                recs = [dict(units=t["units"], budget=b, features={"z": Z[g]} if d == "bool" else {})
                        for b, _ in got]
                assert [q for _, q in got] == pytest.approx(list(predictor.predict(MODELS[d], recs)), abs=1e-15)
    src = out["sources"]
    assert src["data/targets_s2/R-ARITH/heldout.json"] == _sha(root / "data/targets_s2/R-ARITH/heldout.json")
    assert out["features"] == {"BOOL-AN-n5": {"z": 0.4}, "R-ARITH": {}}
    assert S.write_prediction_file(root) == sha                               # rerun: same content, no change


def test_prediction_file_refuses_unchecked_or_inconsistent_inputs(tmp_path, features, monkeypatch):
    """A target file that does not match its .sha256, a changed model file, or a feature value other than the one the
    held-out levels were placed with is refused before anything is written (planted violations)."""
    root = _root(tmp_path)
    t = root / "data/targets_s2/BOOL-AN-n5/bench.json"
    t.write_text(t.read_text().replace("9.0", "9.5"))
    with pytest.raises(ValueError, match="checksum"):
        S.write_prediction_file(root)
    root = _root(tmp_path / "b")
    m = root / "data/registration/h7_model.json"
    m.write_text(m.read_text().replace("-0.4", "-0.41"))
    with pytest.raises(ValueError, match="checksum"):
        S.write_prediction_file(root)
    root = _root(tmp_path / "c")
    from reach.analysis.s2 import confirm as C
    monkeypatch.setattr(C, "set_features", lambda r, sets: {g: {"z": 0.45} for g in sets})
    with pytest.raises(ValueError, match="feature"):
        S.write_prediction_file(root)
    assert not (root / S.PATHS["predictions"]).exists()


def test_committed_prediction_file_matches_the_frozen_model_and_targets():
    """The committed file (when written) holds every held-out target and benchmark cell of the 12 held-out sets at
    every scored budget, under the frozen model's SHA-256, and is reproduced byte for byte from the committed inputs."""
    p = ROOT / S.PATHS["predictions"]
    if not p.exists():
        pytest.skip("prediction file not written yet")
    assert p.with_suffix(".sha256").read_text().split()[0] == _sha(p)
    out = json.loads(p.read_text())
    assert out["model_sha256"] == _sha(ROOT / S.PATHS["h7"])
    lv = json.loads((ROOT / S.PATHS["levels"]).read_text())
    sets = sorted(g for g in lv if not g.startswith("_"))
    assert len(sets) == 12
    want = set()
    for g in sets:
        for kind in ("heldout", "bench"):
            f = ROOT / f"data/targets_s2/{g}/{kind}.json"
            if f.exists():
                want |= {(g, t["target_id"], b) for t in json.loads(f.read_text())
                         for b in predictor.SCORED["H7"]["budgets"]}
    got = [(r["set"], r["target_id"], r["budget"]) for r in out["predictions"]]
    assert len(got) == len(set(got)) and set(got) == want
    again = json.loads(S.prediction_body(ROOT))                               # reproduced (Linux maths: last bits)
    assert [r["p"] for r in again["predictions"]] == pytest.approx([r["p"] for r in out["predictions"]], abs=1e-12)
    assert {k: v for k, v in again.items() if k != "predictions"} == {k: v for k, v in out.items() if k != "predictions"}


@pytest.mark.study_machine
def test_committed_prediction_file_is_reproduced_byte_for_byte():
    p = ROOT / S.PATHS["predictions"]
    if not p.exists():
        pytest.skip("prediction file not written yet")
    assert S.prediction_body(ROOT) == p.read_text()

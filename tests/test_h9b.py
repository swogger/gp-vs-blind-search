"""H9b (D090): transfer of the frozen H7 law to Operon, written before the code. Synthetic records only.

Author decision (D090, "Transfer model"): as H10b for CGP, logit P = c0 + c1 x eta (the frozen H7 model's linear
predictor), fitted by maximum likelihood per Operon configuration on the finished s2_cal_operon runs (16 x E0, the only
budget Operon runs at), frozen before the Stage 2 filing, scored on the held-out Operon runs with the full H7 criteria
(both held-out Operon sets must meet the per-set criterion). The Brier baseline is fitted on the same records and frozen
with it; with one budget, log2 budget is a constant, so it is nodes only. Benchmarks: descriptive, pooled."""
import copy
import hashlib
import json
import math

import numpy as np
import pytest

from reach.analysis.s2 import h9b as H
from reach.analysis.s2 import predictor

FROZEN = dict(unit="bits_sem", budget_term=True, feature=None, params=[3.85, -0.38, 0.16], sha256="a" * 64,
              rule6=True, unit_source="no preferred unit, rule 6: bits_sem; D082")
CAL = ["R-ARITH4", "R-ELEM4", "R-TRIG"]
HO = ["R-ARITH", "R-ELEM6"]
LAM = {"R-ARITH4": 1.0, "R-ELEM4": 1.9, "R-TRIG": 1.4, "R-ARITH": 1.15, "R-ELEM6": 1.7}   # bits per node per set


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _recs(sets, coefs, seed, shift=None, n_targets=12, seeds=5, prefix="t"):
    """Operon records (as load.tidy gives them, 16 x E0) with logit P = c0 + c1 * eta per configuration."""
    rng = np.random.default_rng(seed)
    out = []
    for g in sets:
        for K in (5, 9, 13, 17, 21, 25, 29):
            for i in range(n_targets):
                s = LAM[g] * K
                eta = FROZEN["params"][0] + FROZEN["params"][1] * s + FROZEN["params"][2] * 4
                for cfg, (c0, c1) in coefs.items():
                    for sd in range(seeds):
                        p = _sig(c0 + c1 * eta + (shift or {}).get(g, 0.0))
                        out.append(dict(set=g, target_id=f"{g}_{prefix}{K}_{i}", seed=sd, algorithm="operon",
                                        config=cfg, K=K, units=dict(nodes=float(K), bits_sem=s), budget=16,
                                        y=int(rng.random() < p)))
    return out


COEFS = dict(symbolic=(0.5, 0.8), default=(-0.5, 1.2))


def test_h9b_transfer_recovers_the_planted_intercept_and_slope_per_configuration():
    frozen = copy.deepcopy(FROZEN)
    cal = _recs(CAL, COEFS, seed=1)
    t = H.fit_transfer(frozen, cal)
    assert frozen == FROZEN and set(t) == {"symbolic", "default"}
    for cfg, (c0, c1) in COEFS.items():
        m = t[cfg]
        assert m["transfer"]["c0"] == pytest.approx(c0, abs=0.35) and m["transfer"]["c1"] == pytest.approx(c1, abs=0.2)
        assert m["budgets"] == [16] and m["base_sha256"] == FROZEN["sha256"] and m["unit"] == "bits_sem"
        assert m["n"] == sum(r["config"] == cfg for r in cal)
        rc = [r for r in cal if r["config"] == cfg]
        eta = np.log(predictor.predict(FROZEN, rc) / (1 - predictor.predict(FROZEN, rc)))
        assert np.allclose(predictor.predict(m, rc), _sig(m["transfer"]["c0"] + m["transfer"]["c1"] * eta))
        # the baseline: nodes only (one budget, so log2 budget is constant), fitted on the same records (D090)
        assert m["baseline"] == predictor.fit_baseline(rc, budget_term=False)
    H.check_transfer(t)


def test_h9b_fit_refuses_other_engines_held_out_sets_and_separated_data():
    cal = _recs(CAL, COEFS, seed=2)
    with pytest.raises(ValueError, match="operon"):
        H.fit_transfer(FROZEN, [dict(cal[0], algorithm="A")] + cal[1:])
    with pytest.raises(ValueError, match="calibration"):
        H.fit_transfer(FROZEN, cal + _recs(["R-ARITH"], COEFS, seed=3))
    with pytest.raises(ValueError, match="configuration"):
        H.fit_transfer(FROZEN, [r for r in cal if r["config"] == "symbolic"])
    sep = [dict(r, y=int(r["K"] <= 13)) for r in cal]                         # planted: separated on eta
    with pytest.raises(ValueError, match="separated"):
        H.fit_transfer(FROZEN, sep)


def test_h9b_transfer_is_frozen_checked_and_never_replaced(tmp_path):
    t = H.fit_transfer(FROZEN, _recs(CAL, COEFS, seed=4))
    path = tmp_path / "operon_transfer.json"
    out = H.freeze_transfer(t, path)
    body = path.read_bytes()
    assert path.with_suffix(".sha256").read_text().split()[0] == hashlib.sha256(body).hexdigest() == out["sha256"]
    H.freeze_transfer(copy.deepcopy(t), path)                                 # same content: no change
    other = H.fit_transfer(FROZEN, _recs(CAL, dict(symbolic=(0.0, 1.0), default=(-0.5, 1.2)), seed=5))
    with pytest.raises(RuntimeError, match="never replaced"):
        H.freeze_transfer(other, path)
    assert path.read_bytes() == body
    good = t["symbolic"]
    for i, bad in enumerate([dict(symbolic=good),                                             # a configuration missing
                             dict(t, symbolic=dict(good, budgets=[1, 16])),                    # not the fit budget
                             dict(t, symbolic=dict(good, baseline=None)),                      # no frozen baseline
                             dict(t, symbolic=dict(good, baseline=dict(good["baseline"], budget_term=True))),
                             dict(t, symbolic=dict(good, params=[good["params"][0] + 1] + good["params"][1:]))]):
        with pytest.raises(ValueError):
            H.freeze_transfer(bad, tmp_path / f"bad{i}.json")
        assert not (tmp_path / f"bad{i}.json").exists()


def test_h9b_is_scored_with_the_full_h7_criteria_on_both_held_out_sets(monkeypatch):
    t = H.fit_transfer(FROZEN, _recs(CAL, COEFS, seed=6, n_targets=30))
    seen = []
    real = predictor.score_p

    def spy(*a, **k):
        seen.append(k)
        return real(*a, **dict(k, boot=100))
    monkeypatch.setattr(predictor, "score_p", spy)
    good = H.h9b(t, _recs(HO, COEFS, seed=7))
    assert set(good) == {"symbolic", "default"} and all(good[c]["passes"] for c in good)
    assert all(k["boot"] == 2000 and k["min_sets_ok"] == 2 and k["eps_set"] == 0.10 for k in seen)
    assert good["symbolic"]["test"] == "H9b" and good["symbolic"]["n_sets_registered"] == 2
    bad = H.h9b(t, _recs(HO, COEFS, seed=7, shift={"R-ELEM6": 2.5}))          # planted: one set off by 2.5 logits
    assert all(not bad[c]["passes"] and bad[c]["sets_ok"] == 1 for c in bad)
    with pytest.raises(ValueError, match="held-out"):
        H.h9b(t, _recs(CAL[:1], COEFS, seed=8))
    with pytest.raises(ValueError, match="bench"):
        H.h9b(t, _recs(HO, COEFS, seed=7, prefix="bench_"))


def test_h9b_benchmarks_are_descriptive_and_pooled():
    t = H.fit_transfer(FROZEN, _recs(CAL, COEFS, seed=9))
    d = H.describe_bench(t, _recs(HO, COEFS, seed=10, n_targets=1, prefix="bench_"))
    for cfg in COEFS:
        assert "passes" not in d[cfg] and "sets_ok" not in d[cfg] and d[cfg]["descriptive"]
        assert len(d[cfg]["per_cell_error"]) == 2 * 7 and math.isfinite(d[cfg]["auc"])

"""The node-count baseline's ranking quality on the held-out records (descriptive; D117; REQ-ANA-104).

The registered H7 scoring (predictor.score_model) reports the frozen model's AUC and the baseline's Brier score,
nothing else about the baseline. The paper's Table A.5 and Section 4.5 also quote the baseline's AUC, so this module
computes it the same way: the frozen baseline (logit P = a + b nodes + g log2 budget, frozen with the model, D090)
predicts the H7 records of s2_ho_a0 and s2_ho_a16 (censored runs as failures, the registered budgets, the frozen
features), and the AUC is predictor._auc, the statistic the registered scoring uses. The frozen model's AUC is
recomputed beside it as a check against stage2.json. Output: data/analysis/baseline_auc.json.
"""
import json

import numpy as np

from reach.analysis.s2 import cgp_confirm, predictor
from reach.analysis.s2 import stage2_report as S


def auc_of(model, recs):
    """AUC of a model's predictions on tidy records with outcome y, as the registered scoring computes it."""
    y = np.array([r["y"] for r in recs], float)
    return predictor._auc(y, predictor.predict(model, recs))


def build(root):
    inp = S.inputs(root)
    rows, status, _ = S.phases(root)
    why = S._pending("H7", status, False)
    if why:
        raise RuntimeError(why)
    recs = []
    for ph in S.TESTS["H7"]:
        S._check_regime(ph, rows[ph])
        recs += S._records(root, ph, rows[ph], predictor.SCORED["H7"]["budgets"], True)
    recs = S._features(root, inp, recs)
    h7 = inp["h7"]
    out = {}
    for d, rr in sorted(cgp_confirm.by_domain(recs).items()):
        out[d] = dict(auc=auc_of(h7[d]["baseline"], rr), frozen_auc=auc_of(h7[d], rr), n=len(rr),
                      sets=sorted({r["set"] for r in rr}))
    return dict(label="DESCRIPTIVE: the frozen node-count baseline's AUC on the H7 records (D117)", **out)


def main():
    from reach.paths import ROOT
    out = build(ROOT)
    (ROOT / "data/analysis/baseline_auc.json").write_text(json.dumps(out, indent=1) + "\n")
    for d in ("real", "bool"):
        print(f"{d}: baseline AUC {out[d]['auc']:.4f}, frozen model AUC {out[d]['frozen_auc']:.4f}, n {out[d]['n']}")


if __name__ == "__main__":
    main()

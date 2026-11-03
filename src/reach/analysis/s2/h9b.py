"""H9b (D090, "Transfer model"): transfer of the frozen H7 law to Operon (Stage 2).

As H10b for CGP: per Operon configuration (symbolic, default), logit P = c0 + c1 * eta, with eta the frozen real H7
model's linear predictor, fitted by maximum likelihood on the finished s2_cal_operon runs (the standard-operator real
calibration sets, 16 x E0, the only budget Operon runs at) and frozen with SHA-256 (data/registration/
operon_transfer.json) before the Stage 2 filing, so no addendum is needed. The Brier baseline is fitted on the same
records and frozen with it (D090); with one budget, log2 budget is a constant, so it is nodes only. Scored on the
held-out Operon runs (s2_ho_operon, held-out targets of R-ARITH and R-ELEM6) with the full H7 criteria
(`predictor.score_model("H9b")`: AUC >= 0.80; calibration slope in [0.7, 1.3] with an intercept CI covering 0; Brier
below the baseline; per-set absolute error <= 0.10 in both sets; B = 2,000). Operon's benchmark cells are described,
pooled, with no pass or fail (`describe_bench`). Separated data are refused (guardrail row 118, D088). This replaces
the form's "same size ordering" and the design's AUC criterion (hypotheses.md H9b), a logged deviation from the design.
"""
from collections import defaultdict

from reach.analysis.s2 import predictor, transfer

CAL_PHASE, HO_PHASE = "s2_cal_operon", "s2_ho_operon"
BUDGETS = (16,)
CONFIGS = ("symbolic", "default")
TRANSFER_FILE = "data/registration/operon_transfer.json"


def _only_operon(recs):
    other = {r.get("algorithm") for r in recs} - {"operon"}
    if other:
        raise ValueError(f"H9b reads operon rows only; found algorithm(s) {sorted(map(str, other))}")


def _by_config(recs):
    by = defaultdict(list)
    for r in recs:
        by[r["config"]].append(r)
    missing = [c for c in CONFIGS if c not in by]
    if missing or set(by) - set(CONFIGS):
        raise ValueError(f"H9b: configurations {sorted(by)}, expected {list(CONFIGS)}")
    return by


def _is_bench(r):
    return "_bench_" in r["target_id"]


def fit_transfer(frozen, cal_recs):
    """Both configurations' transfer models from Operon calibration records (calibration sets only)."""
    from reach.runner.phases import CAL_SETS
    _only_operon(cal_recs)
    outside = sorted({r["set"] for r in cal_recs} - set(CAL_SETS))
    if outside:
        raise ValueError(f"H9b is fitted on calibration sets only; found {outside}")
    return {c: transfer.fit(frozen, rr, BUDGETS, f"H9b transfer, Operon {c}", baseline_budget_term=False)
            for c, rr in sorted(_by_config(cal_recs).items())}


def check_transfer(models):
    transfer.check(models, CONFIGS, BUDGETS, "H9b transfer model", baseline_budget_term=False)


def freeze_transfer(models, path):
    return transfer.freeze(models, path, check_transfer)


def _heldout(recs):
    from reach.runner.phases import OPERON_SETS, heldout_sets
    allowed = set(heldout_sets()) & set(OPERON_SETS)
    outside = sorted({r["set"] for r in recs} - allowed)
    if outside:
        raise ValueError(f"H9b scores the held-out Operon sets {sorted(allowed)} only; found {outside}")


def h9b(models, ho_recs):
    """Per configuration, the registered H7 scoring of the held-out targets (benchmarks are described separately)."""
    _only_operon(ho_recs)
    _heldout(ho_recs)
    if any(_is_bench(r) for r in ho_recs):
        raise ValueError("H9b scores held-out targets only; benchmark cells go to describe_bench")
    return {c: predictor.score_model("H9b", models[c], rr, models[c]["baseline"])
            for c, rr in sorted(_by_config(ho_recs).items())}


def describe_bench(models, bench_recs):
    """Operon on the real benchmark cells (D049, D090): descriptive, pooled, per configuration."""
    _only_operon(bench_recs)
    _heldout(bench_recs)
    if not all(_is_bench(r) for r in bench_recs):
        raise ValueError("describe_bench reads benchmark cells only")
    return {c: predictor.describe_model(models[c], rr, models[c]["baseline"])
            for c, rr in sorted(_by_config(bench_recs).items())}

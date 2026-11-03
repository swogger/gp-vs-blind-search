"""T16 analysis: the single outcome definition (REQ-ANA-05), written before the code."""
import pytest

from reach.analysis.s2.load import outcome, outcome_cgp, tidy


def row(**kw):
    base = dict(key="k", set="S", target_id="t", K_construction=5, K_proven=5, units=dict(nodes=5.0, bits_sem=7.0),
                algorithm="A", config="base", seed=1, pop_size=100, generations=800, terminal_state="budget_exhausted",
                gen_exact=None, gens_reached=800)
    base.update(kw)
    return base


def test_success_by_prefix():
    r = row(terminal_state="converged", gen_exact=150, gens_reached=150)
    assert outcome(r, 150) == "failure" and outcome(r, 151) == "success" and outcome(r, 800) == "success"


def test_budget_exhausted_is_failure_up_to_its_length():
    r = row()
    assert outcome(r, 200) == "failure" and outcome(r, 800) == "failure"
    assert outcome(r, 1600) == "beyond"                                   # the run never went that far


def test_success_beyond_the_runs_own_length_is_beyond():
    """Guardrail row 47: beyond a run's length only successes are known; dropping the failures alone biases up."""
    r = row(terminal_state="converged", gen_exact=150, gens_reached=150)
    assert outcome(r, 1600) == "beyond"
    assert {x["budget"] for x in tidy([r, row()], (1, 4, 8, 16))} == {1, 4}


def test_cpu_stop_censors_longer_budgets_only():
    r = row(terminal_state="timeout", gens_reached=300)
    assert outcome(r, 200) == "failure" and outcome(r, 300) == "failure" and outcome(r, 400) == "censored"


def test_queue_kill_without_result_is_censored_everywhere():
    r = dict(key="k", terminal_state="timeout", queue_status="cpu_limit", meta=dict(set="S"))
    assert outcome(r, 50) == "censored"


def test_error_rows_are_excluded():
    assert outcome(dict(key="k", terminal_state="error"), 200) == "error"


def test_tidy_reads_budgets_as_multiples_of_E0():
    rows = [row(terminal_state="converged", gen_exact=90, gens_reached=90), row(seed=2)]
    recs = tidy(rows, budgets=(0.25, 0.5, 1, 2, 4))
    by = {(r["seed"], r["budget"]): r for r in recs}
    assert by[(1, 0.25)]["y"] == 0 and by[(1, 0.5)]["y"] == 1 and by[(2, 4)]["y"] == 0
    assert {r["budget"] for r in recs} == {0.25, 0.5, 1, 2, 4}
    assert all(r["censored"] is False for r in recs)


def test_operon_rows_are_read_at_their_own_budget_only():
    """Guardrail row 53: Operon records have no gen_exact or gens_reached (no prefix property); an exact Operon run was
    read as censored, i.e. a failure. Its outcome is `exact`, at its own budget, and nowhere else."""
    r = dict(key="o", set="R-ARITH4", target_id="t", K_construction=9, K_proven=9, units=dict(nodes=9.0),
             algorithm="operon", config="symbolic", seed=1, budget=16, exact=True, terminal_state="converged")
    t = tidy([r, dict(r, key="o2", exact=False, terminal_state="budget_exhausted")], (1, 4, 16))
    assert [(x["budget"], x["y"], x["censored"]) for x in t] == [(16, 1, False), (16, 0, False)]
    assert tidy([dict(r, terminal_state="error")], (16,)) == []


def test_killed_operon_rows_are_censored_with_their_meta_size():
    """Review finding 4: a killed Operon row has only the task meta (K, no K_construction, no exact)."""
    r = dict(key="o", set="R-ARITH4", target_id="t", K=9, K_proven=9, algorithm="operon", config="default", seed=1,
             budget=16, terminal_state="timeout")
    t = tidy([r], (16,))
    assert t[0]["K"] == 9 and t[0]["censored"] and t[0]["y"] == 0
    assert tidy([r], (16,), censored_as_failure=False) == []


def test_cgp_rows_are_read_by_evaluations():
    """REQ-ANA-22: a CGP run to 16 x E0 gives every budget by prefix, counted in evaluations, not generations."""
    base = dict(key="c", set="BOOL-NAND-n5", target_id="t", K_construction=9, K_proven=9, units=dict(nodes=9.0),
                algorithm="CGP", config="n500", seed=1, budget=16, budget_E=320_000, pop_size=5)
    ok = dict(base, exact=True, evals_exact=20_001, evals_reached=20_001, terminal_state="converged")
    assert [(x["budget"], x["y"]) for x in tidy([ok], (0.25, 1, 2, 16))] == [(0.25, 0), (1, 0), (2, 1), (16, 1)]
    assert outcome_cgp(dict(ok, evals_exact=20_000), 20_000) == "success"
    fail = dict(base, exact=False, evals_exact=None, evals_reached=319_997, terminal_state="budget_exhausted")
    assert {x["y"] for x in tidy([fail], (1, 16))} == {0} and outcome_cgp(fail, 32 * 20_000) == "beyond"
    cpu = dict(base, exact=False, evals_exact=None, evals_reached=50_001, terminal_state="timeout")
    assert outcome_cgp(cpu, 40_000) == "failure" and outcome_cgp(cpu, 80_000) == "censored"
    rejected = dict(base, exact=False, claim_rejected=True, evals_exact=None, evals_reached=5_001,
                    terminal_state="early_stopped")
    assert outcome_cgp(rejected, 5_000) == "failure" and outcome_cgp(rejected, 20_000) == "censored"
    killed = dict(base, terminal_state="timeout", child_signal=24)          # no evals_reached: hard CPU kill
    assert all(x["censored"] for x in tidy([killed], (0.25, 1, 16)))
    assert tidy([dict(base, terminal_state="error")], (1,)) == []

"""QA audits of the calibration runs (D068): reproducibility and independent success labels."""
from reach.analysis.s2 import audit as A


def test_compare_rows_ignores_timing_and_finds_any_other_difference():
    a = dict(key="k", exact=True, gen_exact=12, gens_reached=12, terminal_state="converged", final_error=0.0,
             final_tree_size=7, best_expr="nand(x1, x2)", cpu_seconds=1.0, wall_seconds=2.0, cpu_s=1.1, wall_s=2.2)
    b = dict(a, key="audit|k", cpu_seconds=9.0, wall_seconds=9.0, cpu_s=9.0, wall_s=9.0, finished="later")
    assert A.compare_rows(a, b) == []
    assert A.compare_rows(a, dict(b, gen_exact=13)) == ["gen_exact"]
    assert A.compare_rows(a, dict(b, best_expr="nand(x2, x1)", exact=False)) == ["best_expr", "exact"]


def test_audit_sample_is_deterministic_and_stratified():
    rows = {f"p|{i}": dict(terminal_state="converged" if i % 2 else "budget_exhausted") for i in range(100)}
    s1, s2 = A.sample_keys(rows, 20, seed=1), A.sample_keys(rows, 20, seed=1)
    assert s1 == s2 and len(s1) == 20 and len(set(s1)) == 20
    assert A.sample_keys(dict(rows, bad=dict(terminal_state="error")), 101, seed=1).count("bad") == 0


def test_repro_report_pairs_each_rerun_with_its_original(tmp_path):
    import json
    for ph, rows in (("s2_cal_a0", [dict(key="x", exact=True, gen_exact=3)]),
                     ("s2_audit_repro", [dict(key="audit|x", audit_of="s2_cal_a0", exact=True, gen_exact=4)])):
        d = tmp_path / f"data/runs/{ph}"
        d.mkdir(parents=True)
        (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    rep = A.repro_report(tmp_path)
    assert rep == {"s2_cal_a0": dict(checked=1, identical=0, differing={"x": ["gen_exact"]})}


def test_independent_boolean_check_agrees_and_catches_a_planted_wrong_answer():
    from reach.analysis.s2 import audit_labels as L
    assert L.same_function_bool("nand(x1, nand(x2, x2))", "nand(x1, nand(x2, x2))", 3)
    assert L.same_function_bool("imp(x1, x2)", "or(not(x1), x2)", 2)                # different trees, same function
    assert not L.same_function_bool("nand(x1, x2)", "nor(x1, x2)", 2)               # planted wrong answer
    assert L.same_function_bool("rimp(x1, x2)", "imp(x2, x1)", 2) and L.same_function_bool("nimp(x1, x2)", "rnimp(x2, x1)", 2)


def test_independent_real_check_agrees_and_catches_a_planted_wrong_answer():
    from reach.analysis.s2 import audit_labels as L
    assert L.same_function_real("eml(x, 1)", "-(exp(x), log(1))", scale=20.1)       # e^x - ln 1 = e^x
    assert L.same_function_real("*(x, +(x, 1))", "+(*(x, x), x)", scale=12.0)
    assert not L.same_function_real("*(x, +(x, 1))", "+(*(x, x), 1)", scale=12.0)   # planted wrong answer
    assert L.same_function_real("+(x, 1)", "+(x, 1.0)", scale=4.0)                    # numbers parse as values


def test_extended_reals_and_near_misses():
    """Validation on pilot data (D068): ln 0 = -inf, exp(-inf) = 0 is an exact match (engine semantics); a function
    9e-10 away from the target is not the target."""
    from reach.analysis.s2 import audit_labels as L
    assert L.same_function_real("*(x, +(exp(*(log(log(1)), x)), exp(1)))", "*(x, exp(1))", scale=10.0)
    assert not L.same_function_real("+(x, 1)", "+(x, +(1, 9e-10))", scale=4.0)


def test_three_classes_separate_bugs_from_registered_near_misses():
    from reach.analysis.s2 import audit_labels as L
    assert L.classify(L.max_rel_diff_real("+(x, 1)", "+(x, 1.0)", 4.0)) == "identical"
    assert L.classify(L.max_rel_diff_real("+(x, 1)", "+(x, +(1, 4e-11))", 1.0)) == "near"
    assert L.classify(L.max_rel_diff_real("+(x, 1)", "+(x, 2)", 1.0)) == "different"


# ---- QA check 2 on every calibration phase with exact labels (D091; REQ-ANA-75; guardrail row 128) ----

def test_operon_infix_parser_reads_operons_printed_models():
    """Operon prints infix with negative numbers in parentheses; values checked against a hand-written formula."""
    import math
    from reach.analysis.s2 import audit_labels as L
    s = ("((-0.78174161911010742) + ((-0.31277334690093994) * (sin(((-3.00000000000000000) * X1)) "
         "* ((-4.00000000000000000) + (0.70710676908493042 * X1))))) / exp(log(X1)) - 2 ^ 3 ^ 0.5")
    x = 2.37
    want = (-0.78174161911010742 + (-0.31277334690093994) * (math.sin(-3 * x) * (-4 + 0.70710676908493042 * x))) \
        / math.exp(math.log(x)) - 2 ** (3 ** 0.5)
    assert abs(float(L.operon_value(L.parse_infix(s), x)) - want) < 1e-12
    assert float(L.operon_value(L.parse_infix("-X1 * 2 - 3"), 2.5)) == -8.0              # unary minus, left assoc.


def test_operon_check_refits_scale_and_intercept_and_catches_a_planted_wrong_answer():
    from reach.analysis.s2 import audit_labels as L
    t = "sin(*(x, cos(1)))"
    exact = "(0.25 + ((-3.00000000000000000) * sin((X1 * cos(1.00000000000000000)))))"   # affine terms are free
    assert L.classify(L.operon_diff(exact, t, 1.0)) == "identical"
    f32 = "(0.1 + (2.0 * sin((X1 * 0.54030230586813977))))"                  # cos(1) printed to 17 digits
    assert L.classify(L.operon_diff(f32, t, 1.0)) == "identical"
    near = "(0.1 + (2.0 * sin((X1 * 0.54030230588813977))))"                 # 2e-11 off inside sin
    assert L.classify(L.operon_diff(near, t, 1.0)) == "near"
    assert L.classify(L.operon_diff("(0.1 + (2.0 * sin((X1 * 0.54))))", t, 1.0)) == "different"   # planted
    assert L.classify(L.operon_diff("(1.0 + (0.0 * X1))", t, 1.0)) == "different"     # constant: no scale to fit


def test_operon_check_treats_undefined_values_as_not_exact():
    """Operon is real-valued: log or sqrt of a negative number, a division by zero or an infinite result has no
    value, so such a model cannot be exact."""
    from reach.analysis.s2 import audit_labels as L
    assert L.operon_diff("log((X1 - 2.5))", "log(x)", 1.0) == float("inf")
    assert L.operon_diff("sqrt((X1 - 2.5))", "x", 1.0) == float("inf")
    assert L.operon_diff("(X1 / (X1 - X1))", "x", 1.0) == float("inf")
    assert L.classify(L.operon_diff("exp(log(X1))", "x", 1.0)) == "identical"
    assert L.operon_diff("log((3 - X1))", "log(-(x, 3))", 1.0) == float("inf")    # complex target: never exact


def test_audit_checks_operon_records_and_never_skips_a_row_silently():
    """Row 128: the audit kept only rows with `best_expr`, so Operon records (`model`) were dropped without a word and a
    check of s2_cal_operon would have reported 0 checked, 0 disputed. Every completed row is now checked or counted
    as unchecked with its reason; planted wrong labels are caught in both directions."""
    from reach.analysis.s2 import audit_labels as L
    tg = {("R-TRIG", "t1"): dict(expression="sin(*(x, cos(1)))", target_scale=1.0)}
    base = dict(set="R-TRIG", target_id="t1", algorithm="operon", terminal_state="converged")
    rows = [dict(base, key="ok", exact=True, model="(0.5 + (2.0 * sin((X1 * cos(1.0)))))"),
            dict(base, key="wrong", exact=True, model="(0.5 + (2.0 * sin((X1 * 0.54))))"),
            dict(base, key="missed", exact=False, terminal_state="budget_exhausted", model="sin((X1 * cos(1.0)))"),
            dict(base, key="nomodel", exact=False, terminal_state="budget_exhausted"),
            dict(base, key="err", exact=False, terminal_state="error")]
    a = L.audit(rows, tg)
    assert a["exact_checked"] == 2 and a["exact_disputed"] == ["wrong"]
    assert a["failures_checked"] == 1 and a["failures_disputed"] == ["missed"]
    assert a["unchecked"] == {"nomodel": "no expression recorded"}


def test_pool_phase_reads_its_own_targets(tmp_path):
    """s2_cal_pool runs pool.json targets; the old main read every phase's targets from one file name."""
    import json
    from reach.analysis.s2 import audit_labels as L
    assert L.targets_file("s2_cal_pool", "calibration") == "pool"
    assert L.targets_file("s2_cal_mech", "calibration") == "calibration"
    d = tmp_path / "data/runs/s2_cal_pool"
    d.mkdir(parents=True)
    (d / "results.jsonl").write_text(json.dumps(dict(
        key="k", set="BOOL-AONX-n5", target_id="p1", algorithm="A", exact=True, terminal_state="converged",
        best_expr="and(x1, x2)")) + "\n")
    g = tmp_path / "data/targets_s2/BOOL-AONX-n5"
    g.mkdir(parents=True)
    (g / "pool.json").write_text(json.dumps([dict(target_id="p1", expression="and(x2, x1)")]))
    res = L.run(["s2_cal_pool"], "calibration", tmp_path)
    assert res["s2_cal_pool"]["exact_checked"] == 1 and not res["s2_cal_pool"]["exact_disputed"]


def test_operon_models_exact_only_in_ieee_arithmetic_are_their_own_class():
    """D056 scores Operon in IEEE double precision, where x/0 = inf and 0/inf = 0, so a dead subtree dividing by an
    exact zero vanishes; in real arithmetic the model is undefined there. Found on s2_cal_operon (12 R-ARITH4 models,
    D091): such a model is neither identical nor disputed but 'ieee_only' (exact by the registered rule, reported);
    a failure that IEEE arithmetic finds exact is still a disagreement."""
    from reach.analysis.s2 import audit_labels as L
    t = "sin(*(x, cos(1)))"
    dead = "((0.0 / (2.0 / (0.0 * X1))) + (2.0 * sin((X1 * cos(1.0)))))"
    assert L.operon_diff(dead, t, 1.0) == float("inf")                               # undefined in real arithmetic
    assert L.classify(L.operon_diff_ieee(dead, t, 1.0)) == "identical"
    assert L.classify(L.operon_diff_ieee("((0.0 / (2.0 / (0.0 * X1))) + sin(X1))", t, 1.0)) == "different"
    assert L.operon_diff_ieee("((1.0 / (0.0 * X1)) + sin(X1))", t, 1.0) == float("inf")     # inf survives
    tg = {("R-TRIG", "t1"): dict(expression=t, target_scale=1.0)}
    base = dict(set="R-TRIG", target_id="t1", algorithm="operon")
    rows = [dict(base, key="ieee", exact=True, terminal_state="converged", model=dead),
            dict(base, key="miss", exact=False, terminal_state="budget_exhausted", model=dead)]
    a = L.audit(rows, tg)
    assert a["exact_ieee_only"] == ["ieee"] and a["exact_disputed"] == [] and a["failures_disputed"] == ["miss"]


def test_exponential_towers_overflow_as_in_ieee_instead_of_hanging():
    """A recorded failure in s2_cal_robust (F-EML, R5) is exp(exp(exp(~7))); 30-digit mpmath never returned on it, so
    the audit could not finish (D091). exp of an argument beyond +-1e5 is +inf or 0, as in IEEE double (where the
    limit is about 710): no finite double value changes."""
    import time
    from reach.analysis.s2 import audit_labels as L
    t0 = time.time()
    assert L.max_rel_diff_real("eml(eml(eml(eml(x, x), 1), 1), eml(1, eml(1, x)))", "x", 1.0) == float("inf")
    assert L.same_function_real("eml(-(x, eml(eml(eml(x, x), 1), 1)), 1)", "-(x, x)", 1.0)   # exp(-huge) = 0
    assert L.operon_diff("exp(exp(exp(exp(X1))))", "x", 1.0) == float("inf")
    assert time.time() - t0 < 10
    assert L.same_function_real("exp(x)", "eml(x, 1)", 1.0)                                      # ordinary values


def test_cgp_timing_fields_are_timing():
    """Guardrail row 137 (audit 2026-10-28, finding 5): CGP rows also carry child_cpu_seconds and rust_cpu_seconds; two
    reruns of one CGP run differ in exactly those, so a repro audit of s2_cal_cgp flagged every run. Any other field
    still counts."""
    a = dict(key="k", exact=True, evals_exact=4321, cpu_seconds=1.0, child_cpu_seconds=0.9, rust_cpu_seconds=0.8)
    b = dict(a, cpu_seconds=1.2, child_cpu_seconds=1.1, rust_cpu_seconds=1.0)
    assert A.compare_rows(a, b) == []
    assert A.compare_rows(a, dict(b, evals_exact=4322)) == ["evals_exact"]
    assert {"child_cpu_seconds", "rust_cpu_seconds"} <= A.BIGTREE_IGNORE


def test_tree_gp_runs_exact_only_in_extended_reals_are_their_own_class():
    """Guardrail row 180 (2026-11-08, held-out re-scoring): Study 2's real sets keep exact extended reals (ln 0 = -inf,
    exp(-inf) = 0, x/0 non-finite; D048 C4, D021), so a tree whose dead subtree divides by log(1) = 0 is exact by the
    registered rule while 30-digit real arithmetic finds it undefined: 10 s4_restart R-ELEM6 runs were reported as
    disputed. Such a run is its own class, 'extended_only', checked by an independent float64 evaluation of the guarded
    rules; a tree that is undefined in reals and different in extended reals stays disputed, and a failure that the
    extended evaluation finds exact is still a disagreement."""
    from reach.analysis.s2 import audit_labels as L
    t, dead = "-(exp(exp(1)), x)", "-(exp(/(-(exp(1), exp(exp(1))), log(1))), -(x, exp(exp(1))))"   # R-ELEM6_K05_05
    assert L.max_rel_diff_real(dead, t, 1.0) == float("inf")                     # the old instrument disputes it
    assert L.classify(L.max_rel_diff_extended(dead, t, 1.0)) == "identical"
    assert L.classify(L.max_rel_diff_extended("+(/(x, log(1)), x)", "x", 1.0)) == "different"   # inf survives
    assert L.max_rel_diff_extended("exp(*(x, +(+(1, 1), 1000)))", "x", 1.0) == float("inf")    # lossy overflow: NaN
    assert L.max_rel_diff_extended("*(x, exp(-(1, exp(exp(exp(1))))))", "x", 1.0) == float("inf")  # lossy underflow
    tg = {("R-ELEM6", "t1"): dict(expression=t, target_scale=1.0)}
    base = dict(set="R-ELEM6", target_id="t1")
    rows = [dict(base, key="ext", exact=True, terminal_state="converged", best_expr=dead),
            dict(base, key="bad", exact=True, terminal_state="converged", best_expr="+(/(x, log(1)), x)"),
            dict(base, key="miss", exact=False, terminal_state="budget_exhausted", best_expr=dead)]
    a = L.audit(rows, tg)
    assert a["exact_extended_only"] == ["ext"] and a["exact_disputed"] == ["bad"] and a["failures_disputed"] == ["miss"]

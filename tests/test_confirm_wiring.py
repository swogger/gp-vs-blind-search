"""Filed Stage 1 items wired into confirm.py (D079; REQ-ANA-34 to 38; guardrail rows 101 to 104), written before the
code. Synthetic data only; no calibration outcome is read.

Stage 1 (OSF a3spg), as filed: "Runs stopped by their CPU limit count as failures (primary) and are excluded in a
sensitivity analysis." "Every run in the registered task lists is included; no exclusion after the fact." "A run killed
from outside (not by its CPU limit) is an error and is rerun once; if still missing it is reported and excluded."
"Function fixed-effects logistic model (H3)." "Paired target bootstrap for contrasts; exact McNemar per level."
"allocation (one run or a best split from 32 seeds)"; "the extra allocation seeds enter only the best split." "Sizes
above a set's enumeration ceiling are constructed sizes (upper bounds) and are labelled so."
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from reach.analysis.s2 import alloc as A
from reach.analysis.s2 import confirm as C
from reach.analysis.s2 import contrasts as CT
from reach.analysis.s2 import within as W
from reach.analysis.s2.load import tidy
from reach.paths import ROOT
from reach.s2_lock import task_list_hash
from test_dryrun import SETS, _rows

CAL = dict(C.PHASE_MAPS["calibration"], levels=None)
REG = ("s2_cal_a16", "s2_cal_a0", "s2_cal_mech", "s2_cal_alloc", "s2_cal_robust", "s2_cal_pool", "s2_cal_rand",
       "s2_cal_operon")


def _phase(root, phase, rows, keys=None):
    """A phase directory with a manifest listing its task keys (default: the rows' keys)."""
    d = root / "data/runs" / phase
    d.mkdir(parents=True, exist_ok=True)
    keys = [r["key"] for r in rows] if keys is None else keys
    (d / "manifest.json").write_text(json.dumps(dict(n_tasks=len(keys), tasks=[dict(key=k) for k in keys])))
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return keys


def _register(root, keys):
    """data/registration/stage1_tasks.json for the 8 Stage 1 phases (phases without tasks: an empty list)."""
    reg = {ph: dict(n_tasks=len(keys.get(ph, [])), sha256=task_list_hash([dict(key=k) for k in keys.get(ph, [])]))
           for ph in REG}
    p = root / "data/registration/stage1_tasks.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(reg))


def _base(root, extra_a16=(), extra_keys=()):
    rows = [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0 + i, i)] + list(extra_a16)
    keys = {"s2_cal_a16": _phase(root, "s2_cal_a16", rows, [r["key"] for r in rows] + list(extra_keys))}
    return rows, keys


# ------------------------------------------------------------------ termination table and completeness


def _odd_rows():
    base = dict(set="BOOL-NAND-n5", target_id="BOOL-NAND-n5_K6_0", K_construction=6, K_proven=6, algorithm="A",
                config="base", pop_size=100, generations=800)
    return [dict(base, key="crash", seed=101, terminal_state="error", queue_status="error",
                 error_tail="Traceback\nValueError: boom"),
            dict(base, key="outside", seed=102, terminal_state="error", queue_status="killed_signal_9"),
            dict(base, key="cpu", seed=103, terminal_state="timeout", queue_status="cpu_limit", gens_reached=150,
                 gen_exact=None, units=dict(nodes=6.0, bits_uniform=12.0, bits_syn=9.0, bits_sem=6.0))]


def test_termination_table_counts_every_state_per_phase(tmp_path):
    rows, keys = _base(tmp_path, extra_a16=_odd_rows())
    op = [dict(key="op-err", set="R-TRIG", target_id="R-TRIG_K5_0", K=5, K_proven=5, algorithm="operon",
               config="default", budget=16, seed=1, terminal_state="error", queue_status="error",
               error_tail="RuntimeError: operon")]
    keys["s2_cal_operon"] = _phase(tmp_path, "s2_cal_operon", op)
    _register(tmp_path, keys)
    t = C.termination(tmp_path, REG, registered=tmp_path / "data/registration/stage1_tasks.json")
    a16 = t["phases"]["s2_cal_a16"]
    n = len(rows)
    assert a16["tasks"] == n and a16["rows"] == n and a16["missing"] == 0 and a16["complete"]
    assert a16["states"]["error"] == 1 and a16["states"]["killed"] == 1 and a16["states"]["timeout"] == 1
    assert a16["states"]["converged"] == sum(r["terminal_state"] == "converged" for r in rows)
    assert a16["states"]["budget_exhausted"] == sum(r["terminal_state"] == "budget_exhausted" for r in rows)
    assert sum(a16["states"].values()) == n                              # every row in exactly one column
    assert t["phases"]["s2_cal_operon"]["states"]["error"] == 1 and t["complete"]
    assert t["phases"]["s2_cal_mech"]["tasks"] == 0 and t["phases"]["s2_cal_mech"]["complete"]


def test_confirmatory_report_is_refused_when_tasks_lack_rows(tmp_path):
    """A registered task without a row: confirm refuses to report confirmatory results unless explicitly flagged; the
    flag marks the result not confirmatory and names the missing key. (Planted: counting rows instead of comparing
    the key list with the registered hash passes a phase with a foreign row in place of a registered one.)"""
    rows, keys = _base(tmp_path, extra_keys=["lost-task"])
    _register(tmp_path, keys)
    with pytest.raises(C.IncompleteError, match="s2_cal_a16"):
        C.build(CAL, tmp_path, boot=20)
    res = C.build(CAL, tmp_path, boot=20, allow_incomplete=True)
    comp = res["completeness"]
    assert not comp["complete"] and comp["phases"]["s2_cal_a16"]["missing"] == 1
    assert comp["phases"]["s2_cal_a16"]["missing_keys"] == ["lost-task"] and "INCOMPLETE" in res["label"]
    assert "INCOMPLETE" in C.render(res, "t", "s")
    # same count, wrong key: a foreign row in place of the registered one is not complete
    (tmp_path / "data/runs/s2_cal_a16/results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows + [dict(rows[0], key="foreign")]))
    t = C.termination(tmp_path, REG, registered=tmp_path / "data/registration/stage1_tasks.json")
    assert not t["phases"]["s2_cal_a16"]["complete"] and t["phases"]["s2_cal_a16"]["extra_keys"] == ["foreign"]
    # complete: no refusal
    rows2, keys2 = _base(tmp_path / "ok")
    _register(tmp_path / "ok", keys2)
    res = C.build(CAL, tmp_path / "ok", boot=20)
    assert res["completeness"]["complete"] and "INCOMPLETE" not in res["label"]


def test_excluded_rows_are_listed_with_reasons(tmp_path):
    rows, keys = _base(tmp_path, extra_a16=_odd_rows())
    op = [dict(key="op-err", set="R-TRIG", target_id="R-TRIG_K5_0", K=5, K_proven=5, algorithm="operon",
               config="default", budget=16, seed=1, terminal_state="error", queue_status="error",
               error_tail="RuntimeError: operon")]
    keys["s2_cal_operon"] = _phase(tmp_path, "s2_cal_operon", op)
    _register(tmp_path, keys)
    res = C.build(CAL, tmp_path, boot=20)
    ex = {e["key"]: e for e in res["exclusions"]}
    assert set(ex) == {"crash", "outside", "op-err"}                     # the CPU-limited run is not excluded
    assert ex["crash"]["phase"] == "s2_cal_a16" and "error" in ex["crash"]["reason"] and "boom" in ex["crash"]["detail"]
    assert "killed from outside" in ex["outside"]["reason"] and "signal 9" in ex["outside"]["reason"]
    assert ex["op-err"]["algorithm"] == "operon" and ex["op-err"]["phase"] == "s2_cal_operon"
    text = C.render_termination(res, "t", "s")
    assert all(k in text for k in ex) and len(text.splitlines()) <= 200


# ------------------------------------------------------------------ CPU-limit sensitivity


def _censor(rows, g, n):
    """Turn n failures of set g into CPU-limited runs stopped at generation 100 (censored at E0 and above)."""
    out, done = [], 0
    for r in rows:
        if done < n and r["set"] == g and r["gen_exact"] is None:
            r = dict(r, terminal_state="timeout", gens_reached=100)
            done += 1
        out.append(r)
    return out


def test_cpu_limit_sensitivity_excludes_censored_records(tmp_path):
    rows = _censor([r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0 + i, i)], "BOOL-NOR-n5", 7)
    _phase(tmp_path, "s2_cal_a16", rows)
    res = C.build(CAL, tmp_path, boot=20, allow_incomplete=True)
    s = res["sensitivity_cpu_excluded"]
    # runs, not records: each of the 7 runs is censored at 1, 2 and 4 x E0 (3 records), counted once (row 116)
    assert s["recomputed"] and s["censored_runs"] == 7 and s["by_phase_group"]["reach"] == 7
    assert "censored_records" not in s
    prim = {(c["set"], c["budget"]): c["n"] for c in res["h1"]}
    sens = {(c["set"], c["budget"]): c["n"] for c in s["h1"]}
    assert sens[("BOOL-NOR-n5", 1)] == prim[("BOOL-NOR-n5", 1)] - 7
    assert sens[("BOOL-NOR-n5", 0.25)] == prim[("BOOL-NOR-n5", 0.25)]       # not censored at 50 generations
    assert sens[("BOOL-NAND-n5", 1)] == prim[("BOOL-NAND-n5", 1)]
    assert {"h1_holm", "h2", "h4", "holm"} <= set(s) and "sensitivity_cpu_excluded" not in s
    assert "CPU-limited runs excluded" in C.render(res, "t", "s")
    assert "CPU-limited runs in the primary analysis (counted as failures where censored): 7 (" in C.render(res, "t", "s")


def test_cpu_limit_sensitivity_is_not_recomputed_without_censored_runs(tmp_path):
    _phase(tmp_path, "s2_cal_a16", [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0 + i, i)])
    s = C.build(CAL, tmp_path, boot=20, allow_incomplete=True)["sensitivity_cpu_excluded"]
    assert s["censored_runs"] == 0 and not s["recomputed"] and "identical" in s["note"]


def test_best_split_can_exclude_censored_runs():
    """One target, 4 runs: exact at generation 10; killed without a result; CPU stop at generation 30; full failure.
    Primary: 1 success of 4 at every length. Sensitivity: the killed run is censored everywhere, the CPU stop beyond
    30 generations (at 25: a failure)."""
    runs = [dict(target_id="t", K=5, gen_exact=10, gens_reached=10, generations=200, terminal_state="converged"),
            dict(target_id="t", K=5, generations=200, terminal_state="timeout"),
            dict(target_id="t", K=5, gen_exact=None, gens_reached=30, generations=200, terminal_state="timeout"),
            dict(target_id="t", K=5, gen_exact=None, gens_reached=200, generations=200,
                 terminal_state="budget_exhausted")]
    assert A.counts(runs, 25) == (1, 4) and A.counts(runs, 50) == (1, 4)
    assert A.counts(runs, 25, censored_as_failure=False) == (1, 3)
    assert A.counts(runs, 50, censored_as_failure=False) == (1, 2)


# ------------------------------------------------------------------ H3 on the pool


def _pool_rows(sets, beta_within=0.5, n_fun=30, seeds=10, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for f in range(n_fun):
        m = rng.integers(4, 13)
        a = 2.0 + rng.normal(0, 0.5)
        for g in rng.choice(sets, size=3, replace=False):
            d = int(rng.integers(0, 7))
            p = 1 / (1 + math.exp(-(a - beta_within * d)))
            for k in range(seeds):
                ok = rng.random() < p
                out.append(dict(key=f"pool|{g}|{f}|{k}", set=str(g), target_id=f"{g}_pool{f:03d}", seed=k,
                                K_construction=int(m + d), K_proven=int(m + d), algorithm="A", config="base",
                                pop_size=100, generations=200, gen_exact=5 if ok else None,
                                gens_reached=5 if ok else 200,
                                terminal_state="converged" if ok else "budget_exhausted"))
    return out


def test_h3_is_reported_per_domain_from_the_pool_phase(tmp_path):
    _phase(tmp_path, "s2_cal_a16", [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0 + i, i)])
    pool = _pool_rows(["BOOL-NAND-n5", "BOOL-NOR-n5", "BOOL-AON-n5", "BOOL-RICH-n5"])
    _phase(tmp_path, "s2_cal_pool", pool)
    res = C.build(CAL, tmp_path, boot=20, allow_incomplete=True)
    h = res["h3"]["bool"]
    recs = tidy(pool, (1,))
    fe, rk = W.fixed_effects(recs), W.rank_share(recs)
    assert h["fixed_effects"]["coef"] == pytest.approx(fe["coef"]) and h["rank_share"] == rk
    assert h["fe_supported"] == (fe["hi"] < 0) and h["fe_supported"]                 # planted slope 0.5 within
    assert h["share_supported"] == (rk["share_negative"] >= 0.6) and h["registered_share"] == 0.6
    assert h["holds"] == (h["fe_supported"] and h["share_supported"]) and h["unit"] == "nodes"
    assert res["h3"]["real"]["status"] == "missing: no data" and not res["h3"]["real"]["holds"]
    assert not any(x.startswith("H3") for x in res["not_run"])
    # a weak effect: negative point estimate, CI covering 0 -> the fixed-effects part is not supported (planted:
    # deciding on the point estimate)
    weak = _pool_rows(["BOOL-NAND-n5", "BOOL-NOR-n5", "BOOL-AON-n5", "BOOL-RICH-n5"], beta_within=0.05, seed=2)
    _phase(tmp_path, "s2_cal_pool", weak)
    w = C.build(CAL, tmp_path, boot=20, allow_incomplete=True)["h3"]["bool"]
    assert w["fixed_effects"]["coef"] < 0 < w["fixed_effects"]["hi"] and not w["fe_supported"] and not w["holds"]


# ------------------------------------------------------------------ exact McNemar per level


def test_mcnemar_per_level_known_answer():
    """Pairs on the same target and seed. Level 5: 6 pairs x only, 1 pair y only, 3 ties -> p = 2 (1 + 7) / 128;
    level 7: ties only -> untestable (None). A record without a partner is ignored. (Planted: counting ties as
    discordant, or pairing by target only, changes b and c.)"""
    recs = []
    for i, (xv, yv) in enumerate([(1, 0)] * 6 + [(0, 1)] + [(1, 1), (0, 0), (1, 1)]):
        recs += [dict(algorithm="X", target_id=f"t{i % 5}", seed=i, K=5, y=xv),
                 dict(algorithm="Y", target_id=f"t{i % 5}", seed=i, K=5, y=yv)]
    recs += [dict(algorithm=a, target_id="u", seed=s, K=7, y=s % 2) for a in ("X", "Y") for s in range(4)]
    recs += [dict(algorithm="X", target_id="lonely", seed=99, K=5, y=1)]
    m = CT.mcnemar_per_level(recs, "X", "Y")
    assert m[5]["pairs"] == 10 and m[5]["b"] == 6 and m[5]["c"] == 1 and m[5]["ties"] == 3
    assert m[5]["p"] == pytest.approx(2 * 8 / 128)
    assert m[7]["pairs"] == 4 and m[7]["b"] == m[7]["c"] == 0 and m[7]["p"] is None


def test_confirm_reports_mcnemar_per_level_for_h5(tmp_path):
    _phase(tmp_path, "s2_cal_a16", [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0 + i, i)])
    _phase(tmp_path, "s2_cal_mech", _rows("BOOL-NAND-n5", "B", 0.5, 9) + _rows("BOOL-NAND-n5", "HC", 2.0, 8))
    res = C.build(CAL, tmp_path, boot=20, allow_incomplete=True)
    m = res["h5_mcnemar"]["BOOL-NAND-n5"]
    assert set(m) == {"A-B", "HC-A"}
    assert all(set(v) >= {"pairs", "b", "c", "ties", "p"} for v in m["A-B"].values())
    assert sum(v["b"] for v in m["A-B"].values()) > sum(v["c"] for v in m["A-B"].values())   # A beats B
    assert "McNemar" in C.render_supplement(res, "t", "s")


# ------------------------------------------------------------------ best split from the allocation seeds


def test_best_split_uses_the_main_and_allocation_seeds_of_the_alloc_targets(tmp_path):
    """32 seeds per allocation target: the 5 main seeds (s2_cal_a16, read at E0 by prefix) and 27 allocation seeds;
    main-seed runs on other targets are not used. (Planted: the allocation seeds alone, or every a16 target.)"""
    a16 = [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0 + i, i)]
    _phase(tmp_path, "s2_cal_a16", a16)
    targets = sorted({r["target_id"] for r in a16 if r["set"] == "BOOL-NAND-n5"})[::2]
    rng = np.random.default_rng(3)
    alloc = []
    for t in targets:
        K = next(r["K_construction"] for r in a16 if r["target_id"] == t)
        for s in range(27):
            ge = int(rng.integers(0, 400))
            ok = ge < 200
            alloc.append(dict(key=f"alloc|{t}|{s}", set="BOOL-NAND-n5", target_id=t, seed=1000 + s, K_construction=K,
                              K_proven=K, algorithm="A", config="base", pop_size=100, generations=200,
                              gen_exact=ge if ok else None, gens_reached=ge if ok else 200,
                              terminal_state="converged" if ok else "budget_exhausted"))
    _phase(tmp_path, "s2_cal_alloc", alloc)
    res = C.build(CAL, tmp_path, boot=20, allow_incomplete=True)
    b = res["alloc"]["BOOL-NAND-n5"]
    assert b["targets"] == len(targets) and b["seeds_per_target"] == [3 + 27, 3 + 27]    # _rows has 3 main seeds
    runs = [dict(r, K=r["K_construction"]) for r in a16 + alloc if r["target_id"] in set(targets)]
    exp = A.best_split_ci(runs, 4, boot=20)
    assert b["budgets"]["4"]["k50"] == pytest.approx(exp["k50"]) and b["budgets"]["4"]["best_length"] == exp["best_length"]
    assert set(b["budgets"]) == {"1", "2", "4", "8", "16"} and "chosen_lengths" not in b["budgets"]["4"]
    assert "BOOL-NOR-n5" not in res["alloc"]                         # no allocation runs: no best split
    assert "Best split" in C.render_supplement(res, "t", "s")


# ------------------------------------------------------------------ constructed sizes


def test_constructed_sizes_are_labelled(tmp_path):
    rows = [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0 + i, i)]
    rows = [dict(r, K_proven=None) if r["K_construction"] >= 24 else r for r in rows]
    _phase(tmp_path, "s2_cal_a16", rows)
    res = C.build(CAL, tmp_path, boot=20, allow_incomplete=True)
    sz = res["sizes"]["BOOL-NAND-n5"]
    assert sz["constructed_levels"] == [24, 26] and sz["proven_levels"] == list(range(6, 24, 2))
    assert sz["constructed_targets"] == 2 * 8
    text = C.render_supplement(res, "t", "s")
    assert "constructed K" in text and "K_proven" in text and "K_construction" in text
    line = next(x for x in text.splitlines() if x.startswith("| BOOL-NAND-n5 |") and "24, 26" in x)
    assert "upper bound" in line or "constructed K" in line
    assert "minimal" not in text.replace("not minimal", "")


def test_h3_share_decision_uses_the_full_denominator(tmp_path):
    """Guardrail row 115 (independent review, 2026-10-27): which share decides H3 in confirm was untested; deciding by
    the spread-only share went unnoticed. Functions without a size spread (the same size in every set: tau undefined,
    not negative, in the denominator; D079 b) are added until the full share falls below 0.6 while the spread-only
    share stays at or above it: H3's share part must not hold."""
    sets = ["BOOL-NAND-n5", "BOOL-NOR-n5", "BOOL-AON-n5", "BOOL-RICH-n5"]
    _phase(tmp_path, "s2_cal_a16", [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0 + i, i)])
    pool = _pool_rows(sets, beta_within=1.0, seed=3)
    rk = W.rank_share(tidy(pool, (1,)))
    neg, n = rk["negative"], rk["functions"]
    assert neg / rk["with_spread"] >= 0.6
    n0 = next(k for k in range(1, 100) if neg / (n + k) < 0.6)
    for f in range(n0):                                     # no size spread: K 9 in every set
        for g in sets[:3]:
            for k in range(10):
                ok = k < 5
                pool.append(dict(key=f"pool|{g}|flat{f}|{k}", set=g, target_id=f"{g}_pool{900 + f:03d}", seed=k,
                                 K_construction=9, K_proven=9, algorithm="A", config="base", pop_size=100,
                                 generations=200, gen_exact=5 if ok else None, gens_reached=5 if ok else 200,
                                 terminal_state="converged" if ok else "budget_exhausted"))
    _phase(tmp_path, "s2_cal_pool", pool)
    h = C.build(CAL, tmp_path, boot=20, allow_incomplete=True)["h3"]["bool"]
    r = h["rank_share"]
    assert r["no_spread"] == rk["no_spread"] + n0 and r["share_negative"] < 0.6 <= r["share_negative_spread_only"]
    assert not h["share_supported"] and not h["holds"]


def test_censored_runs_count_each_run_once_in_every_phase_group():
    """Guardrail row 116 (independent review, 2026-10-27): the sensitivity summary counted censored records (one per
    run and budget), not runs. One run per group, censored at several budgets, counts once; an uncensored run and an
    error row count zero."""
    base = dict(set="BOOL-NAND-n5", target_id="t1", seed=0, algorithm="A", config="base", pop_size=100,
                generations=3200, K_construction=6, K_proven=6)
    rows = [dict(base, gen_exact=None, gens_reached=100, terminal_state="timeout"),            # censored at 1-16 x E0
            dict(base, seed=1, gen_exact=None, gens_reached=3200, terminal_state="budget_exhausted"),
            dict(base, seed=2, terminal_state="error"),
            dict(base, seed=3)]                                                                 # killed: censored everywhere
    assert C.censored_runs(tidy(rows, (1, 2, 4, 8, 16))) == 2
    assert C.censored_runs(tidy(rows[:1], (1, 2, 4, 8, 16))) == 1
    assert C.censored_runs(tidy([dict(base, seed=s, config=c, gen_exact=None, gens_reached=100, terminal_state="timeout",
                                      generations=800) for s in (0, 1) for c in ("R1", "R2")], (1, 4))) == 4


# ---------------------------------------------- guardrail row 136: units of killed rows in a phase with two target files
def _two_files(root):
    d = root / "data/targets_s2/R-ARITH"
    d.mkdir(parents=True)
    u = dict(nodes=9.0, bits_uniform=30.0, bits_syn=20.0, bits_sem=10.0)
    (d / "heldout.json").write_text(json.dumps([dict(target_id="R-ARITH_K09_00", units=u, K_construction=9)]))
    (d / "bench.json").write_text(json.dumps([dict(target_id="R-ARITH_bench_N1", units=dict(u, bits_sem=11.0),
                                                   K_construction=9)]))
    killed = dict(algorithm="A", set="R-ARITH", target_id="R-ARITH_K09_00", seed=1, config="base", K_construction=9,
                  K_proven=9, terminal_state="killed", pop_size=100)                  # no result line: no units
    bench = dict(killed, target_id="R-ARITH_bench_N1", terminal_state="budget_exhausted", gens_reached=200,
                 generations=200, gen_exact=None, units=dict(u, bits_sem=11.0))
    return [killed, bench]


def test_killed_rows_keep_their_units_when_a_phase_reads_two_target_files(tmp_path):
    """Audit 2026-10-28, finding 4: `_load_units` cleared its map per target-file name, so a report of s2_ho_operon or
    s2_ho_bench that loaded "bench" after "heldout" dropped killed held-out rows (no units in the row) into _NO_UNITS
    instead of counting them as failures. One call now loads every file a phase reads (STAGE2_TARGET_FILES)."""
    from reach.analysis.s2.load import tidy
    rows = _two_files(tmp_path)
    C._NO_UNITS.clear()
    C._load_units(tmp_path, ["R-ARITH"], ("heldout", "bench"))
    recs = C._with_units(tidy(rows, (1,), True))
    assert C._NO_UNITS == [] and len(recs) == 2
    k = next(r for r in recs if r["target_id"] == "R-ARITH_K09_00")
    assert k["y"] == 0 and k["censored"] and k["units"]["bits_sem"] == 10.0          # a failure, with its units
    assert C.STAGE2_TARGET_FILES["s2_ho_operon"] == ("heldout", "bench")
    C._NO_UNITS.clear()


def test_one_target_id_in_two_files_with_other_units_is_refused(tmp_path):
    _two_files(tmp_path)
    p = tmp_path / "data/targets_s2/R-ARITH/bench.json"
    t = json.loads(p.read_text())
    t[0]["target_id"] = "R-ARITH_K09_00"
    p.write_text(json.dumps(t))
    with pytest.raises(ValueError, match="two target files"):
        C._load_units(tmp_path, ["R-ARITH"], ("heldout", "bench"))


def test_stage2_target_files_are_the_ones_each_phase_reads(tmp_path, monkeypatch, request):
    """STAGE2_TARGET_FILES names exactly the target files each Stage 2 phase's registered tasks read (--targets)."""
    from reach import s2_lock as L
    from reach.runner.phases import registration_build
    if not (ROOT / "data/registration/heldout_levels.json").exists():
        pytest.skip("no held-out levels")
    request.getfixturevalue("cgp_exe")
    lock = tmp_path / "stage.txt"
    lock.write_text("test build only\n")
    monkeypatch.setattr(L, "STAGE1_FILE", lock)
    monkeypatch.setattr(L, "STAGE2_FILE", lock)
    assert set(C.STAGE2_TARGET_FILES) == set(L.STAGE2_PHASES)
    for name in L.STAGE2_PHASES:
        files = {Path(t["argv"][t["argv"].index("--targets") + 1]).stem for t in registration_build(name)()}
        assert files == set(C.STAGE2_TARGET_FILES[name]), name

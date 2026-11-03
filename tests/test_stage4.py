"""Stage 4, the remedy test (D100; docs/design/remedy.md; REQ-RUN-41, REQ-ANA-95). Written before the code."""
import json
import math
from pathlib import Path

import pytest

from reach import s2_lock as L
from reach.runner import phases as P

ROOT = Path(__file__).resolve().parents[1]
REAL_HO = ["R-ARITH", "R-ELEM6", "RS-add_sqrt", "RS-eml_sin_sqrt_square", "RS-mul_div_eml_sin_square", "RS-mul_eml_sin"]


def _open_stage4(monkeypatch, tmp_path):
    f = tmp_path / "stage4_registration.txt"
    f.write_text("Stage 4 filed (test)\n")
    monkeypatch.setattr(L, "STAGE4_FILE", f)


def test_restart_seed_slice_is_fresh():
    from reach.seeds import S2_SLICES, TARGET_SEEDS, s2_seeds
    seeds = s2_seeds("restart")
    assert len(seeds) == len(set(seeds)) == 64
    others = set(s2_seeds("main"))
    for name in S2_SLICES:
        if name != "restart":
            others |= set(s2_seeds(name))
    assert not set(seeds) & others
    assert TARGET_SEEDS["restart"] not in {v for k, v in TARGET_SEEDS.items() if k != "restart"}


@pytest.mark.study_machine
def test_level_rule_adds_only_rs_add_sqrt_levels():
    rule = {g: P.restart_extra_levels(g) for g in REAL_HO}
    assert rule == {g: ([18, 19, 20] if g == "RS-add_sqrt" else []) for g in REAL_HO}


@pytest.mark.study_machine
def test_restart_tasks(monkeypatch, tmp_path):
    _open_stage4(monkeypatch, tmp_path)
    if not (ROOT / "data/targets_s2/RS-add_sqrt/restart.json").exists():
        pytest.skip("restart targets not drawn yet (s4_targets)")
    from reach.seeds import s2_seeds
    tasks = P.PHASES["s4_restart"]["build"]()
    seeds = set(s2_seeds("restart"))
    assert len(tasks) == 55 * 30 * 64 and len({t["key"] for t in tasks}) == len(tasks)
    sets = {t["meta"]["set"] for t in tasks}
    assert sets == set(REAL_HO)
    for t in tasks:
        a = t["argv"]
        assert a[a.index("--generations") + 1] == "100" and a[a.index("--pop_size") + 1] == "100"
        assert a[a.index("--algorithm") + 1] == "A" and a[a.index("--config") + 1] == "base"
        assert a[a.index("--stop_on") + 1] == "exact" and a[a.index("--penalty") + 1] == "scaled"
        assert a[a.index("--timeout") + 1] == "900" and t["cpu_limit"] == 1000
        assert int(a[a.index("--seed") + 1]) in seeds
        assert t["meta"]["K"] == t["meta"]["K_proven"]                      # constructed levels left out
    per_set = {g: len({t["meta"]["K"] for t in tasks if t["meta"]["set"] == g}) for g in REAL_HO}
    assert per_set == {"R-ARITH": 8, "R-ELEM6": 9, "RS-add_sqrt": 13, "RS-eml_sin_sqrt_square": 8,
                       "RS-mul_div_eml_sin_square": 8, "RS-mul_eml_sin": 9}


def test_stage4_phases_are_locked_and_routed(monkeypatch, tmp_path):
    monkeypatch.setattr(L, "STAGE4_FILE", tmp_path / "not_recorded.txt")
    with pytest.raises(PermissionError):
        P.PHASES["s4_restart"]["build"]()
    assert L.registered_tasks_file("s4_targets") is None                    # target draw: no search, not locked
    assert L.registered_tasks_file("s4_restart") == L.TASKS4_FILE
    assert L.registered_tasks_file("s2_ho_a16") == L.TASKS2_FILE              # Stage 2 routing unchanged


# ---------------------------------------------------------------- H11 statistics on synthetic sets
def _set_runs(levels, hit_rate, n=64, targets=6):
    """{target: [runs]} and {target: bits}: per level, `targets` targets x n 100-generation runs, a share hit_rate(b)
    of them exact at generation 10."""
    runs, size = {}, {}
    for b in levels:
        for t in range(targets):
            tid = f"b{b}_t{t}"
            size[tid] = b
            k = int(round(hit_rate(b) * n))
            runs[tid] = [dict(target_id=tid, generations=100, gens_reached=100, terminal_state="budget_exhausted",
                              gen_exact=10 if s < k else None) for s in range(n)]
    return runs, size


def _one_run(levels, k50, n=5, targets=6):
    out = []
    for b in levels:
        for t in range(targets):
            for s in range(n):
                out.append(dict(target_id=f"b{b}_t{t}", y=int(b < k50 and s < 4 or b < k50 - 2), bits=b))
    return out


def test_h11_non_inferiority_and_bracketing():
    from reach.analysis.s2 import remedy
    levels = [10, 12, 14, 16, 18, 20, 22]
    # per-run hit rate falling with size so that 32 restarts cross 1/2 near 18.3 bits
    rate = lambda b: max(0.0, min(1.0, 2 ** (-(b - 12) / 1.2) * 0.2))          # noqa: E731
    runs, size = _set_runs(levels, rate)
    k = remedy.restart_k50(runs, size, R=32)
    assert not k["extrapolated"] and 12 < k["k50"] < 22
    # every target solved: K50 +inf -> replaced by the top level, flagged
    allhit, size2 = _set_runs(levels, lambda b: 0.5)
    k2 = remedy.restart_k50(allhit, size2, R=32)
    assert k2["extrapolated"] and k2["k50"] == 22 and k2["raw"] == math.inf


def test_h11_gain_is_a_conjunction_over_sets():
    from reach.analysis.s2 import remedy
    p = remedy.conjunction_p([0.01, 0.03, 0.2])
    assert p == 0.2


def test_h11_holm_and_estimability():
    from reach.analysis.s2 import remedy
    d = remedy.decide({"H11-1": dict(p=0.01, estimable=True), "H11-2": dict(p=0.04, estimable=True)})
    assert d["H11-1"]["supported"] and d["H11-2"]["supported"]               # 0.01 < 0.025; 0.04 < 0.05
    d = remedy.decide({"H11-1": dict(p=0.03, estimable=True), "H11-2": dict(p=0.04, estimable=True)})
    assert not d["H11-1"]["supported"] and not d["H11-2"]["supported"]       # Holm stops at the first
    d = remedy.decide({"H11-1": dict(p=0.001, estimable=False), "H11-2": dict(p=0.001, estimable=True)})
    assert not d["H11-1"]["supported"]                                       # not estimable: never supported
    # guardrail row 155: a non-estimable test counts as p = 1 (Stage 3's rule), so it does not block the other
    d = remedy.decide({"H11-1": dict(p=0.001, estimable=False), "H11-2": dict(p=0.02, estimable=True)})
    assert d["H11-2"]["supported"] and not d["H11-1"]["supported"]
    d = remedy.decide({"H11-1": dict(p=0.025, estimable=True), "H11-2": dict(p=0.05, estimable=True)})
    assert d["H11-1"]["supported"] and d["H11-2"]["supported"]               # p <= threshold, as h6d.holm
    est = remedy.p_beyond([-0.5, -0.8, -1.2, float("nan")], margin=-1.0, side="above", boot=4)
    assert est["kept"] == 3 and not est["estimable"]                         # 3 of 4 kept < 90%
    assert est["p"] == pytest.approx((1 + 1) / (3 + 1))                      # one replicate at or below -1


def test_h11_end_to_end_on_synthetic_sets():
    """Two sets whose 32-restart K50 lies near the null's 18.29 bits and whose one run reaches about 12: D near 0
    (H11-1 supported), gains near 6 (H11-2 supported); D is the mean of the per-set restart K50 minus the null."""
    from reach.analysis.s2 import remedy
    levels = [10, 12, 14, 16, 18, 20, 22]
    rate = lambda b: 0.2 * 2 ** (-(b - 12) / 1.95)                           # noqa: E731
    restart, one = {}, {}
    for g in ("A", "B"):
        runs, size = _set_runs(levels, rate, targets=8)
        restart[g] = (runs, size)
        one[g] = [dict(target_id=t, y=int(size[t] < 12 or (size[t] == 12 and s < 3)), units={"bits_sem": size[t]})
                  for t in runs for s in range(5)]
    res = remedy.h11(restart, one, boot=60)
    k = {g: remedy.restart_k50(*restart[g])["k50"] for g in restart}
    assert res["tests"]["H11-1"]["estimate"] == pytest.approx(sum(k.values()) / 2 - math.log2(320_000))
    assert -1.0 < res["tests"]["H11-1"]["estimate"] < 2.0                  # near the null, above the margin
    assert res["tests"]["H11-1"]["supported"] and res["tests"]["H11-2"]["supported"]
    assert all(v["gain"] > 4 for v in res["per_set"].values())


def test_stage4_registration_files_match_the_builds(monkeypatch, tmp_path):
    """Once written (scripts/s2_task_lists.py --stage 4), stage4_tasks.json holds s4_restart's count, key hash and
    full hash as the phase builds now."""
    reg = ROOT / "data/registration/stage4_tasks.json"
    if not reg.exists():
        pytest.skip("Stage 4 task list not written yet")
    _open_stage4(monkeypatch, tmp_path)
    want = json.loads(reg.read_text())
    assert sorted(want) == sorted(L.STAGE4_PHASES)
    for name in L.STAGE4_PHASES:
        tasks = P.PHASES[name]["build"]()
        assert want[name] == dict(n_tasks=len(tasks), sha256=L.task_list_hash(tasks),
                                  sha256_full=L.task_list_full_hash(tasks))

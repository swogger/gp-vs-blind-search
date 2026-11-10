"""Stage 5 phases (D125; docs/design/stage5_*.md): REQ-RUN-42 to REQ-RUN-49. Task lists are built, never run.
Written before the code; every check is shown to fire on a planted violation."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from reach import s2_lock as L
from reach.runner import phases as P

ROOT = Path(__file__).resolve().parents[1]
REAL_HO = ["R-ARITH", "R-ELEM6", "RS-add_sqrt", "RS-eml_sin_sqrt_square", "RS-mul_div_eml_sin_square", "RS-mul_eml_sin"]
MAIN_REAL = ["F-EML", "R-ARITH4", "R-ELEM4", "R-HYB", "F-LDE0", "F-EML-R", "R-TRIG"]
MAIN_BOOL = ["BOOL-NAND-n5", "BOOL-NOR-n5", "BOOL-IMP0-n5", "BOOL-AON-n5", "BOOL-AONX-n5", "BOOL-RICH-n5",
             "BOOL-IMP0R-n5"]
PILOTS = ("s5_pilot_grid", "s5_pilot_wide", "s5_pilot_boolsub", "s5_pilot_selfree", "s5_pilot_pop1000",
          "s5_pilot_keys", "s5_pilot_robust")
PILOTS_ALL = PILOTS + ("s5_pilot_restart10",)   # ten s4_restart re-executions (QA; coordinator, 2026-11-08)


@pytest.fixture
def stage5(monkeypatch, tmp_path):
    f = tmp_path / "stage5_registration.txt"
    f.write_text("Stage 5 filed (test)\n")
    monkeypatch.setattr(L, "STAGE5_FILE", f)
    return f


def _arg(t, flag):
    a = t["argv"]
    return a[a.index(flag) + 1]


def _orig(phase):
    return {t["key"]: t for t in P._manifest_tasks(phase)}


# ---------------------------------------------------------------- REQ-RUN-42 lock, routing, order
def test_stage5_phases_are_locked_and_routed(monkeypatch, tmp_path):
    monkeypatch.setattr(L, "STAGE5_FILE", tmp_path / "not_recorded.txt")
    monkeypatch.setattr(P, "_low_targets", lambda g: [])
    for name in L.STAGE5_PHASES + L.STAGE5_PAUSED:             # registered and paused (D126): all locked
        with pytest.raises(PermissionError):
            P.PHASES[name]["build"]()
        assert L.registered_tasks_file(name) == L.TASKS5_FILE
    assert L.registered_tasks_file("s5_unlisted") == L.TASKS5_FILE          # any other s5 phase must be registered
    for name in ("s5_targets",) + PILOTS_ALL:
        assert L.registered_tasks_file(name) is None
    assert L.registered_tasks_file("s4_restart") == L.TASKS4_FILE             # earlier stages unchanged
    assert L.registered_tasks_file("s2_revisit_a0") == L.TASKS3_FILE
    assert L.registered_tasks_file("s2_ho_a16") == L.TASKS2_FILE
    assert L.registered_tasks_file("s2_cal_a0") == L.TASKS_FILE


def test_the_runner_checks_stage5_phases(tmp_path, monkeypatch):
    import argparse
    from reach.runner import __main__ as M
    monkeypatch.setitem(M.PHASES, "s5_fake", dict(build=lambda: [dict(key="a", argv=["x"], cpu_limit=1)],
                                                  description="", verify=None))
    monkeypatch.setattr(L, "TASKS5_FILE", tmp_path / "none.json")
    with pytest.raises(PermissionError):
        M.cmd_plan(argparse.Namespace(phase="s5_fake", runs=tmp_path, force=False), quiet=True)


def test_priority_order_lists_every_stage5_phase_once():
    flat = [p for group in P.STAGE5_ORDER for p in group]
    assert sorted(flat) == sorted(n for n in P.PHASES if n.startswith("s5_")) and len(flat) == len(set(flat))
    assert P.STAGE5_ORDER[0] == ("s5_targets",) and set(P.STAGE5_ORDER[1]) == set(PILOTS_ALL)
    assert P.STAGE5_ORDER[2][:2] == ("s5_grid", "s5_boolsub") and P.STAGE5_ORDER[-1] == ("s5_frozen",)
    assert flat.index("s5_long_extra") < flat.index("s5_revisit_ho16")
    assert flat.index("s5_revisit_restart") < flat.index("s5_union")
    assert not [p for p in flat if "ipop" in p]                               # IPOP is not run (D125)
    assert set(flat) - {"s5_targets"} - set(PILOTS_ALL) == set(L.STAGE5_PHASES + L.STAGE5_PAUSED)


def test_stage5_seed_slices_are_fresh():
    from reach.seeds import S2_SLICES, TARGET_SEEDS, s2_seeds
    assert S2_SLICES["s5_pop1000"] == (784, 789) and S2_SLICES["s5_pilot"] == (810, 830)
    for name in ("s5_pop1000", "s5_pilot"):
        mine = set(s2_seeds(name))
        others = set(s2_seeds("main")).union(*(set(s2_seeds(n)) for n in S2_SLICES if n != name))
        assert mine and not mine & others, name
    assert TARGET_SEEDS["low"] == 20261108 and len(set(TARGET_SEEDS.values())) == len(TARGET_SEEDS)


# ---------------------------------------------------------------- REQ-RUN-43 the low levels
LOW = {"F-EML": [5], "F-EML-R": [5], "R-ARITH4": [3, 5], "R-ELEM4": [3, 4, 5], "R-HYB": [3, 5], "F-LDE0": [5, 7],
       "R-TRIG": [3, 4], "BOOL-NAND-n5": [3, 5, 7], "BOOL-NOR-n5": [3, 5], "BOOL-IMP0-n5": [3], "BOOL-IMP0R-n5": [3],
       "BOOL-AON-n5": [2, 3, 4], "BOOL-AONX-n5": [2, 3], "BOOL-RICH-n5": [2]}


def test_low_level_rule():
    assert {g: P.low_levels(g) for g in P.S5_MAIN_SETS} == LOW
    assert sorted(P.S5_MAIN_SETS) == sorted(MAIN_REAL + MAIN_BOOL)
    tasks = P.PHASES["s5_targets"]["build"]()                                # not locked: no search
    from reach.seeds import TARGET_SEEDS
    assert {t["key"] for t in tasks} == set(LOW)
    for t in tasks:
        assert _arg(t, "--name") == "low" and _arg(t, "--n") == "30"
        assert _arg(t, "--seed") == str(TARGET_SEEDS["low"])
        assert _arg(t, "--levels") == ",".join(map(str, LOW[t["key"]]))


def test_targets_take_every_eligible_function_when_fewer_than_n(tmp_path):
    """s2_targets.generate on a level with fewer eligible functions than asked takes all of them."""
    from reach import grammar as G
    from reach import s2_targets as T
    from reach.enumerate import Enumerator
    g = G.get("BOOL-NAND-n3")
    Enumerator(g, tmp_path, log=lambda *_: None).run(5)
    ts, log = T.generate(g, tmp_path, [3, 5], 30, 1)
    for k in ("3", "5"):
        e = log["levels"][k]
        assert e["found"] == e["n_depth_ok"] - sum(e["rejected"].values()) < 30, k
    assert len({t["expression"] for t in ts}) == len(ts)


# ---------------------------------------------------------------- REQ-RUN-44 fitness-information phases
@pytest.mark.parametrize("name,config,domain,count", [("s5_grid", "FG", "real", True), ("s5_wide", "WIDE", "real", True),
                                                      ("s5_frozen", "FB", "real", False),
                                                      ("s5_boolsub", "BS24", "bool", True)])
def test_fitness_information_phases(stage5, name, config, domain, count):
    from reach.seeds import s2_seeds
    tasks = P.PHASES[name]["build"]()
    sets = MAIN_REAL if domain == "real" else MAIN_BOOL
    excluded = set(WIDE_EXCLUDED) if name == "s5_wide" else set()            # review F1 (row 173)
    assert len(tasks) == len({t["key"] for t in tasks}) == len(sets) * 10 * 30 * 5 - 5 * len(excluded)
    a0, a16 = _orig("s2_cal_a0"), _orig("s2_cal_a16")
    want = {(o["meta"]["set"], o["meta"]["target_id"], o["meta"]["seed"]) for o in list(a0.values()) + list(a16.values())
            if o["meta"]["set"] in sets and o["meta"]["target_id"] not in excluded}
    assert {(t["meta"]["set"], t["meta"]["target_id"], t["meta"]["seed"]) for t in tasks} == want
    for t in tasks:
        assert _arg(t, "--config") == config and _arg(t, "--generations") == "200" and _arg(t, "--pop_size") == "100"
        assert _arg(t, "--timeout") == "900" and P.PRINT_TREE in t["argv"]
        assert ("--count_distinct" in t["argv"]) == count
        assert t["cpu_limit"] == (1000 * P.REVISIT_CPU_FACTOR if count else 1000)
        assert int(_arg(t, "--seed")) in s2_seeds("main")
        src = a0 if t["meta"]["pair_of"] == "s2_cal_a0" else a16
        o = src[t["meta"]["pair_key"]]                                    # the paired registered base run exists
        assert (o["meta"]["target_id"], o["meta"]["seed"]) == (t["meta"]["target_id"], t["meta"]["seed"])


WIDE_EXCLUDED = ("F-EML_K19_03", "F-EML_K19_22", "F-EML_K21_04", "F-EML_K23_03", "F-EML_K25_09", "F-EML-R_K15_18",
                 "F-EML-R_K21_26")                 # variance on the wide 100-point grid overflows (review F1)


def test_wide_phases_exclude_targets_whose_normaliser_is_not_finite(stage5):
    """Review F1 (guardrail row 173): a target whose variance on the wide interval's 100-point grid is not finite in
    double precision gets no s5_wide or s5_pilot_wide task (paired exclusion from both arms of H12-3); planted case
    F-EML_K19_03; the other H12 arms keep every target."""
    wide, grid = P.PHASES["s5_wide"]["build"](), P.PHASES["s5_grid"]["build"]()
    assert len(wide) == 10_500 - 35 and len(grid) == 10_500
    gone = {t["meta"]["target_id"] for t in grid} - {t["meta"]["target_id"] for t in wide}
    assert gone == set(WIDE_EXCLUDED) and "F-EML_K19_03" in gone
    assert P.wide_excluded() == sorted(WIDE_EXCLUDED)
    pilot, pilot_grid = P.PHASES["s5_pilot_wide"]["build"](), P.PHASES["s5_pilot_grid"]["build"]()
    gone = {t["meta"]["target_id"] for t in pilot_grid} - {t["meta"]["target_id"] for t in pilot}
    assert gone == {"F-EML_K15_04", "F-EML_K23_02", "F-EML-R_K21_05"} and len(pilot) == len(pilot_grid) - 6


# ---------------------------------------------------------------- REQ-RUN-45 selection-free phases
def test_selection_free_levels_and_tasks(stage5, monkeypatch):
    lv = {g: P.selfree_levels(g) for g in P.S5_MAIN_SETS}
    assert sum(len(lv[g]) for g in MAIN_REAL) == 37 and sum(len(lv[g]) for g in MAIN_BOOL) == 22
    for g in P.S5_MAIN_SETS:
        bits = {t["K_construction"]: t["units"]["bits_sem"]
                for t in json.loads((ROOT / f"data/targets_s2/{g}/calibration.json").read_text())}
        assert all(bits[k] <= math.log2(16 * 20_000) for k in lv[g])
        cal = json.loads((ROOT / "data/registration/levels.json").read_text())[g]
        assert [k for k in sorted(cal["main"] + cal["budget"]) if bits[k] <= math.log2(16 * 20_000)] == lv[g]
    if not all((ROOT / f"data/targets_s2/{g}/low.json").exists() for g in LOW):
        with pytest.raises(FileNotFoundError):                             # the low levels must be drawn first
            P.PHASES["s5_selfree"]["build"]()
        monkeypatch.setattr(P, "_low_targets", lambda g: [])
        n_low = 0
    else:
        n_low = sum(len(json.loads((ROOT / f"data/targets_s2/{g}/low.json").read_text())) for g in LOW)
    sf, base = P.PHASES["s5_selfree"]["build"](), P.PHASES["s5_low_base"]["build"]()
    assert len(sf) == 8_850 + 5 * n_low and len(base) == 5 * n_low
    for t in sf:
        assert _arg(t, "--config") == "SF" and _arg(t, "--generations") == "200" and "--count_distinct" in t["argv"]
    for t in base:
        assert _arg(t, "--config") == "base" and t["meta"]["targets"] == "low"


# ---------------------------------------------------------------- REQ-RUN-46 population 1,000
def test_population_1000_on_the_restart_levels(stage5, monkeypatch, tmp_path):
    from reach.seeds import s2_seeds
    tasks = P.PHASES["s5_pop1000"]["build"]()
    assert len(tasks) == len({t["key"] for t in tasks}) == 55 * 30 * 5
    for t in tasks:
        assert _arg(t, "--config") == "R2" and _arg(t, "--pop_size") == "1000" and _arg(t, "--generations") == "320"
        assert _arg(t, "--timeout") == "7200" and t["cpu_limit"] == 7300 and int(_arg(t, "--seed")) in s2_seeds("s5_pop1000")
    s4 = {(t["meta"]["set"], t["meta"]["target_id"]) for t in P._manifest_tasks("s4_restart")}
    assert {(t["meta"]["set"], t["meta"]["target_id"]) for t in tasks} == s4          # rule S1: the same targets


# ---------------------------------------------------------------- REQ-RUN-47, 48 re-execution phases
@pytest.mark.parametrize("name,src,n,keys", [("s5_revisit_robust", "s2_cal_robust", 6_000, False),
                                             ("s5_revisit_restart", "s4_restart", 105_600, True),
                                             ("s5_revisit_ho16", "s2_ho_a16", 900, False)])
def test_reexecution_tasks_are_the_originals_plus_flags(stage5, name, src, n, keys):
    from reach.analysis.s2.phase1 import enum_dir
    orig = _orig(src)
    tasks = P.PHASES[name]["build"]()
    assert len(tasks) == n and len({t["key"] for t in tasks}) == n
    paths = set()
    for t in tasks:
        o = orig[t["meta"]["audit_key"]]
        extra = ["--count_distinct", "--enum_dir", str(enum_dir(o["meta"]["set"]))]
        if keys:
            extra += ["--keys_out", P.keys_path(o["meta"]["target_id"], o["meta"]["seed"])]
            paths.add(extra[-1])
        assert t["key"] == f"revisit|{o['key']}" and t["meta"]["audit_of"] == src
        assert t["argv"] == o["argv"] + extra and t["cpu_limit"] == o["cpu_limit"] * P.REVISIT_CPU_FACTOR
        assert {k: v for k, v in t["meta"].items() if k not in ("audit_of", "audit_key")} == o["meta"]
        assert t.get("cwd", ".") == o.get("cwd", ".") and t.get("env", {}) == o.get("env", {})
    assert len(paths) == (n if keys else 0)
    if name == "s5_revisit_robust":
        assert {t["meta"]["config"] for t in tasks} == {"R1", "R2"}
    if name == "s5_revisit_ho16":
        from reach.seeds import s2_seeds
        assert {t["meta"]["seed"] for t in tasks} == {s2_seeds("heldout")[0]}
        assert {t["meta"]["set"] for t in tasks} == set(REAL_HO)


def test_reexecution_verify_ignores_keys_and_checks_the_side_files(tmp_path):
    from reach.analysis.s2.audit import revisit_report
    f = tmp_path / "k.npz"
    f.write_bytes(b"side file")
    sha = hashlib.sha256(b"side file").hexdigest()
    base = dict(set="R-ARITH", exact=False, gen_exact=None, gens_reached=100, pop_size=100,
                terminal_state="budget_exhausted", best_expr="x")
    _write(tmp_path, "s4_restart", [dict(key="k1"), dict(key="k2")], [dict(base, key="k1"), dict(base, key="k2")])
    tasks = [dict(key=f"revisit|{k}", argv=["{python}", "-m", "reach.tasks.s2_run", "--keys_out", f"{k}.npz"],
                  meta=dict(audit_of="s4_restart", audit_key=k)) for k in ("k1", "k2")]
    rv = dict(stop=dict(n=10_000))
    good = [dict(base, key=f"revisit|{k}", rv=rv, keys=dict(file=str(f), n=3, sha256=sha)) for k in ("k1", "k2")]
    _write(tmp_path, "s5_revisit_restart", tasks, good)
    assert revisit_report(tmp_path, "s5_revisit_restart", extra_ignore=("keys",), check_keys=True)["ok"]
    assert not revisit_report(tmp_path, "s5_revisit_restart")["ok"]                  # "keys" is a new field
    bad = [good[0], dict(good[1], keys=dict(file=str(f), n=3, sha256="0" * 64))]   # planted: a changed side file
    _write(tmp_path, "s5_revisit_restart", tasks, bad)
    rep = revisit_report(tmp_path, "s5_revisit_restart", extra_ignore=("keys",), check_keys=True)
    assert not rep["ok"] and rep["sources"]["s4_restart"]["differing"]["k2"] == ["side file differs from its row"]
    _write(tmp_path, "s5_revisit_restart", tasks, [good[0], {k: v for k, v in good[1].items() if k != "keys"}])
    rep = revisit_report(tmp_path, "s5_revisit_restart", extra_ignore=("keys",), check_keys=True)
    assert rep["sources"]["s4_restart"]["differing"]["k2"] == ["no side file"]


def _write(root, name, tasks, rows):
    d = root / f"data/runs/{name}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(dict(phase=name, n_tasks=len(tasks), tasks=tasks)))
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_union_phase_has_one_task_per_restart_target(stage5):
    tasks = P.PHASES["s5_union"]["build"]()
    s4 = {(t["meta"]["set"], t["meta"]["target_id"]) for t in P._manifest_tasks("s4_restart")}
    assert len(tasks) == len(s4) == 1650 and {(t["meta"]["set"], t["meta"]["target_id"]) for t in tasks} == s4
    for t in tasks:
        assert "reach.tasks.s5_union" in t["argv"] and _arg(t, "--n_runs") == "64"


def test_union_histograms_equal_brute_force():
    from reach.tasks.s5_union import histograms
    rng = np.random.default_rng(3)
    lv = rng.integers(1, 6, 500)                                           # 500 lookup functions on levels 1..5
    runs = []
    for i in range(9):
        idx = rng.choice(500, 60, replace=False)
        runs.append((idx, np.sort(rng.integers(1, 10_001, 60)), None if i % 3 else 4_000 + i))
    h = histograms(runs, lv, 5)
    for p in (5_000, 10_000):
        for name, group in (("all", runs), ("nohit", [r for r in runs if r[2] is None or r[2] > p])):
            for k in range(1, 6):
                cover = [sum(int(j in set(idx[first <= p])) for idx, first, _ in group) for j in np.flatnonzero(lv == k)]
                want = [sum(1 for c in cover if c == m) for m in range(len(group) + 1)]
                assert h[str(p)][name][str(k)] == want, (p, name, k)
            assert h[str(p)][f"n_{name}"] == len(group)


def test_union_task_refuses_missing_or_changed_side_files(tmp_path, monkeypatch):
    import subprocess
    import sys
    import os
    from reach import grammar as G
    from reach import revisit as V
    from reach.enumerate import Enumerator
    g = G.get("BOOL-NAND-n3")
    d = tmp_path / "enum"
    Enumerator(g, d, log=lambda *_: None).run(7)
    lk = V.Lookup(g, d)
    rows = []
    for s in range(3):
        f = tmp_path / f"k{s}.npz"
        c = V.Counter(g, None, lk, pop_size=100, record_keys=True)
        c.first = {int(k): 100 * (i + 1) for i, k in enumerate(lk.keys[s:s + 5])}
        info = c.write_keys(f)
        rows.append(dict(key=f"revisit|s4_restart|x|t0|{s}", set=g.name, target_id="t0", exact=s == 0,
                         gen_exact=10 if s == 0 else None, pop_size=100, terminal_state="converged", keys=info,
                         rv=dict(lookup=lk.describe())))
    rows.append(dict(rows[1], key="revisit|s4_restart|x|t1|0", target_id="t1"))   # another target's row is ignored
    res = tmp_path / "results.jsonl"
    reach_root = tmp_path / "root"
    (reach_root / "data/registration").mkdir(parents=True)

    def run(rr, n=3):
        res.write_text("".join(json.dumps(r) + "\n" for r in rr))
        return subprocess.run([sys.executable, "-m", "reach.tasks.s5_union", "--set", g.name, "--target_id", "t0",
                               "--rows", str(res), "--enum_dir", str(d), "--n_runs", str(n)], cwd=ROOT,
                              capture_output=True, text=True,
                              env=dict(os.environ, PYTHONPATH=str(ROOT / "src"), REACH_ROOT=str(reach_root)))
    p = run(rows)
    assert p.returncode == 0, p.stderr[-400:]
    out = json.loads(next(l for l in p.stdout.splitlines() if l.startswith("RESULT "))[7:])
    assert out["n_runs"] == 3 and out["hist"]["10000"]["n_nohit"] == 2
    assert run(rows[:2]).returncode != 0                                     # a missing row
    planted = [rows[0], rows[1], dict(rows[2], keys=dict(rows[2]["keys"], sha256="0" * 64))]
    assert run(planted).returncode != 0                                      # a changed side file
    # review F3 (guardrail row 175): the side files' lookup must be the loaded one, and the registered one if recorded
    other = V.Lookup(g, d, bound_bits=float(np.log2(lk.keys.size)) - 1)      # a smaller lookup of the same set
    assert other.sha256 != lk.sha256
    planted = [rows[0], rows[1], dict(rows[2], rv=dict(lookup=other.describe()))]
    p = run(planted)
    assert p.returncode != 0 and "lookup" in p.stdout + p.stderr
    (reach_root / "data/registration/stage5_lookups.json").write_text(json.dumps({g.name: "0" * 64}))
    p = run(rows)
    assert p.returncode != 0 and "stage5_lookups" in p.stdout + p.stderr
    (reach_root / "data/registration/stage5_lookups.json").write_text(json.dumps({g.name: lk.sha256}))
    assert run(rows).returncode == 0


# ---------------------------------------------------------------- REQ-RUN-49 one long run on the extra levels
EXTRA = {"R-ARITH": [19, 21, 23], "R-ELEM6": [10, 11, 12, 13], "RS-add_sqrt": [10, 12, 14, 16, 17, 18, 19, 20],
         "RS-eml_sin_sqrt_square": [10, 11, 12], "RS-mul_div_eml_sin_square": [10, 11, 12],
         "RS-mul_eml_sin": [11, 12, 13, 14]}


def test_long_extra_tasks(stage5):
    from reach.seeds import s2_seeds
    assert {g: P.long_extra_levels(g) for g in REAL_HO} == EXTRA
    tasks = P.PHASES["s5_long_extra"]["build"]()
    assert len(tasks) == len({t["key"] for t in tasks}) == 25 * 30 * 5
    a0 = _orig("s2_ho_a0")
    for t in tasks:
        m = t["meta"]
        assert _arg(t, "--config") == "base" and _arg(t, "--generations") == "3200" and _arg(t, "--timeout") == "7200"
        assert "--count_distinct" not in t["argv"] and m["seed"] in s2_seeds("heldout")    # no counter (D126)
        if m["set"] == "RS-add_sqrt" and m["K"] >= 18:
            assert "prefix_key" not in m                                     # no Stage 2 run at E0 on these levels
        else:
            o = a0[m["prefix_key"]]
            assert (o["meta"]["target_id"], o["meta"]["seed"]) == (m["target_id"], m["seed"])
    assert {t["cpu_limit"] for t in tasks} == {7300}                       # soft + 100 s, not doubled (D126)
    assert sum("prefix_key" in t["meta"] for t in tasks) == 22 * 30 * 5


def _lrow(key, **kw):
    r = dict(key=key, exact=False, gen_exact=None, gens_reached=3200, terminal_state="budget_exhausted",
             best_expr="x", final_tree_size=1, generations=3200)
    r.update(kw)
    return r


def test_long_extra_prefix_check_fires_on_planted_mismatches(tmp_path):
    from reach.analysis.s2.audit import long_extra_report
    short = [dict(_lrow("a"), generations=200, gens_reached=200),
             dict(_lrow("b"), generations=200, exact=True, gen_exact=50, gens_reached=50, terminal_state="converged",
                  best_expr="y", final_tree_size=3),
             dict(_lrow("c"), generations=200, gens_reached=200)]
    _write(tmp_path, "s2_ho_a0", [dict(key=r["key"]) for r in short], short)
    tasks = [dict(key=f"L{k}", meta=dict(prefix_of="s2_ho_a0", prefix_key=k)) for k in "abc"] + [dict(key="Ld", meta={})]
    good = [_lrow("La", exact=True, gen_exact=900, gens_reached=900, terminal_state="converged"),
            _lrow("Lb", exact=True, gen_exact=50, gens_reached=50, terminal_state="converged", best_expr="y",
                  final_tree_size=3), _lrow("Lc"), _lrow("Ld")]
    _write(tmp_path, "s5_long_extra", tasks, good)
    rep = long_extra_report(tmp_path, "s5_long_extra")
    assert rep["ok"] and rep["compared"] == 3 and rep["mismatch"] == {} and rep["no_prefix"] == 1
    for bad, why in ((dict(good[0], gen_exact=120), "a"), (dict(good[1], gen_exact=51), "b"),
                     (dict(good[1], best_expr="z"), "b")):
        rows = [bad if r["key"] == bad["key"] else r for r in good]
        _write(tmp_path, "s5_long_extra", tasks, rows)
        rep = long_extra_report(tmp_path, "s5_long_extra")
        assert not rep["ok"] and list(rep["mismatch"]) == [f"L{why}"], bad
    _write(tmp_path, "s5_long_extra", tasks, good[1:])                       # a missing row
    assert not long_extra_report(tmp_path, "s5_long_extra")["ok"]


# ---------------------------------------------------------------- pilots (REQ-RUN-42: never locked, pilot data only)
def test_pilot_phases_use_pilot_targets_and_seeds(monkeypatch, tmp_path):
    from reach.seeds import s2_seeds
    monkeypatch.setattr(L, "STAGE5_FILE", tmp_path / "not_recorded.txt")    # not locked
    pilot_seeds = set(s2_seeds("pilot")) | set(s2_seeds("s5_pilot"))
    sizes = {}
    for name in PILOTS:
        tasks = P.PHASES[name]["build"]()
        sizes[name] = len(tasks)
        assert tasks and len({t["key"] for t in tasks}) == len(tasks), name
        for t in tasks:
            assert _arg(t, "--targets").endswith("/pilot.json") and int(_arg(t, "--seed")) in pilot_seeds, name
    seeds = lambda name: {int(_arg(t, "--seed")) for t in P.PHASES[name]["build"]()}   # noqa: E731
    assert seeds("s5_pilot_grid") == set(s2_seeds("pilot")[:2]) == seeds("s5_pilot_selfree")
    assert seeds("s5_pilot_pop1000") == set(s2_seeds("pilot")[8:10])
    assert seeds("s5_pilot_keys") == set(s2_seeds("s5_pilot")[:16])
    assert sizes["s5_pilot_robust"] == 2 * sum(1 for t in P._manifest_tasks("s2_pilot_robust")
                                               if t["meta"]["config"] == "R2")


def test_stage5_registration_files_match_the_builds(stage5):
    """Once written (scripts/s2_task_lists.py --stage 5), stage5_tasks.json holds each locked phase's count, key hash
    and full hash as the phase builds now."""
    reg = ROOT / "data/registration/stage5_tasks.json"
    if not reg.exists():
        pytest.skip("Stage 5 task lists not written yet")
    want = json.loads(reg.read_text())
    assert sorted(want) == sorted(L.STAGE5_PHASES)
    for name in L.STAGE5_PHASES:
        tasks = P.PHASES[name]["build"]()
        assert want[name] == dict(n_tasks=len(tasks), sha256=L.task_list_hash(tasks),
                                  sha256_full=L.task_list_full_hash(tasks))


def test_task_list_script_writes_the_stage5_lists(tmp_path, monkeypatch):
    """scripts/s2_task_lists.py --stage 5 builds every locked Stage 5 phase (locks lifted in memory only) and writes
    its count, key hash and full hash, and the lookups of the counted sets."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("s2_task_lists", ROOT / "scripts/s2_task_lists.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "data/registration").mkdir(parents=True)
    built = []

    def build(name):
        def f():
            built.append(name)
            assert L.STAGE5_FILE.read_text().strip()                       # the lock is lifted for the build only
            return [dict(key=f"{name}|a", argv=["x"], cpu_limit=1, meta={})]
        return f
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "registration_build", build)
    real = L.STAGE5_FILE
    mod.main(["--stage", "5"])
    assert L.STAGE5_FILE == real and built == list(L.STAGE5_PHASES)
    out = json.loads((tmp_path / "data/registration/stage5_tasks.json").read_text())
    assert sorted(out) == sorted(L.STAGE5_PHASES) and all(len(v["sha256_full"]) == 64 for v in out.values())
    assert not (tmp_path / "data/registration/stage5_lookups.json").exists()   # nothing counted: none (D126)


def _gz(root, name, rows):
    import gzip
    d = root / f"data/runs/{name}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.jsonl.gz").write_bytes(gzip.compress("".join(json.dumps(r) + "\n" for r in rows).encode()))


def test_keyless_reexecution_passes_its_verify_step(tmp_path, monkeypatch):
    """Review F2 (guardrail row 174): the verify step checked a side file on every row, so every correct row of
    s5_revisit_robust and s5_revisit_ho16 (no --keys_out) failed. A side file is checked exactly when the task's argv
    has --keys_out; a `keys` field on a task without it is a problem."""
    from types import SimpleNamespace
    base = dict(set="F-EML", exact=False, gen_exact=None, gens_reached=200, pop_size=100,
                terminal_state="budget_exhausted", best_expr="x")
    argv = ["{python}", "-m", "reach.tasks.s2_run", "--count_distinct", "--enum_dir", "d"]
    rv = dict(stop=dict(n=20_000))
    monkeypatch.setattr(P, "ROOT", tmp_path)
    for name, src in (("s5_revisit_robust", "s2_cal_robust"), ("s5_revisit_ho16", "s2_ho_a16")):
        _write(tmp_path, src, [dict(key="k1")], [dict(base, key="k1")])
        tasks = [dict(key="revisit|k1", argv=argv, meta=dict(audit_of=src, audit_key="k1"))]
        _write(tmp_path, name, tasks, [dict(base, key="revisit|k1", rv=rv)])
        assert P.PHASES[name]["verify"](SimpleNamespace(name=name)) == 0, name
        _write(tmp_path, name, tasks, [dict(base, key="revisit|k1", rv=rv, keys=dict(file="x", n=1, sha256="0"))])
        assert P.PHASES[name]["verify"](SimpleNamespace(name=name)) == 1, name      # planted: keys on a keyless task
    from reach.analysis.s2.audit import revisit_report
    rep = revisit_report(tmp_path, "s5_revisit_ho16", extra_ignore=("keys",), check_keys=True)
    assert rep["sources"]["s2_ho_a16"]["differing"]["k1"] == ["keys without --keys_out"]
    tasks = [dict(key="revisit|k1", argv=argv + ["--keys_out", "k.npz"], meta=dict(audit_of="s2_ho_a16", audit_key="k1"))]
    _write(tmp_path, "s5_revisit_ho16", tasks, [dict(base, key="revisit|k1", rv=rv)])
    rep = revisit_report(tmp_path, "s5_revisit_ho16", extra_ignore=("keys",), check_keys=True)
    assert rep["sources"]["s2_ho_a16"]["differing"]["k1"] == ["no side file"]        # planted: the file is missing


def test_reproduction_checks_read_gzip_and_report_a_missing_original(tmp_path):
    """Review F4 (guardrail row 176): the checks read only results.jsonl (s4_restart is committed gzipped) and a
    missing original row raised KeyError."""
    from reach.analysis.s2.audit import long_extra_report, revisit_report
    base = dict(set="R-ARITH", exact=False, gen_exact=None, gens_reached=100, pop_size=100,
                terminal_state="budget_exhausted", best_expr="x")
    _gz(tmp_path, "s4_restart", [dict(base, key="k1"), dict(base, key="k2")])
    tasks = [dict(key=f"revisit|{k}", argv=["x"], meta=dict(audit_of="s4_restart", audit_key=k)) for k in ("k1", "k2")]
    rows = [dict(base, key=f"revisit|{k}", rv=dict(stop=dict(n=10_000))) for k in ("k1", "k2")]
    _write(tmp_path, "s5_revisit_restart", tasks, rows)
    assert revisit_report(tmp_path, "s5_revisit_restart")["ok"]                     # the originals from the gzip
    _gz(tmp_path, "s4_restart", [dict(base, key="k1")])
    rep = revisit_report(tmp_path, "s5_revisit_restart")
    assert not rep["ok"] and rep["sources"]["s4_restart"]["missing_original"] == ["k2"]
    short = [dict(_lrow("a"), generations=200, gens_reached=200)]
    _gz(tmp_path, "s2_ho_a0", short)
    ltasks = [dict(key="La", meta=dict(prefix_of="s2_ho_a0", prefix_key="a")),
              dict(key="Lb", meta=dict(prefix_of="s2_ho_a0", prefix_key="b"))]
    _write(tmp_path, "s5_long_extra", ltasks, [_lrow("La"), _lrow("Lb")])
    rep = long_extra_report(tmp_path, "s5_long_extra")
    assert rep["compared"] == 1 and rep["missing"] == ["Lb"] and not rep["ok"]


def test_restart_pilot_is_ten_reexecutions_with_key_sets(monkeypatch, tmp_path):
    """The memo's reproduction check (coordinator, 2026-11-08): ten s4_restart tasks, drawn as D068's sample (seed
    20261022), re-executed with the counter and key sets as a pilot (QA; their outcomes are known); not locked; each
    task is its original plus the flags, its side file under the pilot's own folder."""
    from reach.analysis.s2.audit import sample_keys
    monkeypatch.setattr(L, "STAGE5_FILE", tmp_path / "not_recorded.txt")
    tasks = P.PHASES["s5_pilot_restart10"]["build"]()
    orig = _orig("s4_restart")
    chosen = sample_keys(P.phase_rows("s4_restart"), 10)
    assert sorted(t["meta"]["audit_key"] for t in tasks) == chosen and len(tasks) == 10
    for t in tasks:
        o = orig[t["meta"]["audit_key"]]
        extra = t["argv"][len(o["argv"]):]
        assert t["argv"][:len(o["argv"])] == o["argv"] and extra[:2] == ["--count_distinct", "--enum_dir"]
        assert extra[3:] == ["--keys_out", P.keys_path(o["meta"]["target_id"], o["meta"]["seed"], "s5_pilot_restart10")]
        assert t["cpu_limit"] == o["cpu_limit"] * P.REVISIT_CPU_FACTOR and t["meta"]["audit_of"] == "s4_restart"
    assert P.PHASES["s5_pilot_restart10"]["verify"] is P.PHASES["s5_revisit_restart"]["verify"]

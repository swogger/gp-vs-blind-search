"""Held-out phases (Phase 7; REQ-RUN-21): locked until the Stage 2 registration is recorded, built like calibration."""
import hashlib
import json

import pytest

from reach import s2_lock as L
from reach.runner import phases as P

SETS = ["R-ARITH", "BOOL-AN-n5"]


def _setup(tmp_path, monkeypatch, stage2=True):
    reg = tmp_path / "data/registration"
    reg.mkdir(parents=True)
    if stage2:
        (reg / "stage2_registration.txt").write_text("https://osf.io/xxxxx\n")
    (reg / "heldout_levels.json").write_text(json.dumps({g: {"main": [3, 5], "budget": [7, 9]} for g in SETS}))
    (reg / "heldout_levels.sha256").write_text(                       # written with its checksum (row 124)
        hashlib.sha256((reg / "heldout_levels.json").read_bytes()).hexdigest() + "  heldout_levels.json\n")
    for g in SETS:
        d = tmp_path / f"data/targets_s2/{g}"
        d.mkdir(parents=True)
        for name, ks in (("heldout", (3, 5, 7, 9)), ("bench", (5,))):
            ts = [dict(target_id=f"{g}_{name}_{K}_{i}", expression="x", K_construction=K, K_proven=K)
                  for K in ks for i in range(30 if name == "heldout" else 1)]
            (d / f"{name}.json").write_text(json.dumps(ts))
            (d / f"{name}.sha256").write_text(hashlib.sha256((d / f"{name}.json").read_bytes()).hexdigest() + "  x\n")
    monkeypatch.setattr(P, "ROOT", tmp_path)
    monkeypatch.setattr(L, "STAGE2_FILE", reg / "stage2_registration.txt")
    monkeypatch.setattr(P, "heldout_sets", lambda: SETS)


def test_heldout_phases_refuse_before_stage2(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, stage2=False)
    for name in ("s2_ho_a0", "s2_ho_a16", "s2_ho_bench", "s2_ho_operon"):
        with pytest.raises(PermissionError):
            P.PHASES[name]["build"]()


def test_heldout_phases_follow_the_plan(tmp_path, monkeypatch):
    from reach.seeds import s2_seeds
    _setup(tmp_path, monkeypatch)
    a0, a16 = P.PHASES["s2_ho_a0"]["build"](), P.PHASES["s2_ho_a16"]["build"]()
    assert len(a0) == len(a16) == 2 * 2 * 30 * 5                       # sets x levels x targets x held-out seeds
    assert {t["meta"]["seed"] for t in a0} == set(s2_seeds("heldout"))
    assert {t["meta"]["budget"] for t in a16} == {16} and {t["meta"]["K"] for t in a16} == {7, 9}
    bench = P.PHASES["s2_ho_bench"]["build"]()
    assert len(bench) == 2 * 30 and {t["meta"]["seed"] for t in bench} == set(s2_seeds("benchmark"))
    op = P.PHASES["s2_ho_operon"]["build"]()
    assert {t["meta"]["set"] for t in op} == {"R-ARITH"} and all("|operon|" in t["key"] for t in op)


def test_heldout_targets_phase_uses_the_frozen_levels_and_heldout_seed(tmp_path, monkeypatch):
    from reach.seeds import TARGET_SEEDS
    _setup(tmp_path, monkeypatch, stage2=False)                       # not a search: allowed before Stage 2
    t = P.PHASES["s2_ho_targets"]["build"]()
    a = t[0]["argv"]
    assert a[a.index("--name") + 1] == "heldout" and a[a.index("--seed") + 1] == str(TARGET_SEEDS["heldout"])
    assert a[a.index("--levels") + 1] == "3,5,7,9"


def test_heldout_levels_are_checked_against_their_checksum(tmp_path, monkeypatch):
    """Guardrail row 124: `_ho_levels` read heldout_levels.json without its .sha256 (cgp_settings.json is checked)."""
    _setup(tmp_path, monkeypatch, stage2=False)
    reg = tmp_path / "data/registration"
    assert P.PHASES["s2_ho_targets"]["build"]()                             # control: the checksummed file is read
    body = json.loads((reg / "heldout_levels.json").read_text())
    body["R-ARITH"]["main"] = [3, 7]
    (reg / "heldout_levels.json").write_text(json.dumps(body))
    with pytest.raises(ValueError, match="checksum"):
        P.PHASES["s2_ho_targets"]["build"]()
    (reg / "heldout_levels.sha256").unlink()
    with pytest.raises(ValueError, match="checksum"):
        P.PHASES["s2_ho_targets"]["build"]()


def test_heldout_operon_includes_the_real_benchmarks(tmp_path, monkeypatch):
    """Guardrail row 127: D049 (phase7_heldout.md) runs Operon "on the held-out sets among the five standard real sets,
    and on the real benchmarks"; s2_ho_operon left the benchmarks out (inventory 5e). Benchmark cells of the Operon
    sets get both configurations at 16 x E0 with the 30 benchmark seeds, as s2_ho_bench; Boolean benchmarks none."""
    from reach.seeds import s2_seeds
    _setup(tmp_path, monkeypatch)
    op = P.PHASES["s2_ho_operon"]["build"]()
    bench = [t for t in op if "_bench_" in t["meta"]["target_id"]]
    held = [t for t in op if "_heldout_" in t["meta"]["target_id"]]
    assert len(held) == 4 * 30 * 2 * 5 and len(bench) == 1 * 2 * 30 and len(op) == len(held) + len(bench)
    assert {t["meta"]["set"] for t in bench} == {"R-ARITH"}
    assert {t["meta"]["seed"] for t in bench} == set(s2_seeds("benchmark"))
    assert {t["meta"]["config"] for t in bench} == {"symbolic", "default"} and {t["meta"]["budget"] for t in bench} == {16}
    a = bench[0]["argv"]
    assert a[a.index("--targets") + 1] == "data/targets_s2/R-ARITH/bench.json"
    assert len({t["key"] for t in op}) == len(op)


# ------------------------------------------------- D092: Stage 2 tree-GP runs record the whole best tree (row 131)
def test_heldout_tree_gp_tasks_print_the_whole_tree(tmp_path, monkeypatch):
    """D092 (guardrail row 131; REQ-RUN-33): every tree-GP task of the held-out phases carries `--print_tree`
    (REQ-ENG-22: the whole best tree after the search, nothing else changed), so exact held-out runs with best trees
    over 200 nodes can be label-checked. Operon tasks are not tree-GP runs and are left as they are."""
    _setup(tmp_path, monkeypatch)
    for name in ("s2_ho_a0", "s2_ho_a16", "s2_ho_bench"):
        tasks = P.PHASES[name]["build"]()
        assert tasks and all(t["argv"].count("--print_tree") == 1 for t in tasks), name
        assert P.tree_gp_without_full_tree(tasks) == [], name
    op = P.PHASES["s2_ho_operon"]["build"]()
    assert op and not any("--print_tree" in t["argv"] or "reach.tasks.s2_run" in t["argv"] for t in op)


def test_full_tree_check_fires_on_a_planted_task_without_the_option(tmp_path, monkeypatch):
    """The check itself (planted violation): a held-out tree-GP task without `--print_tree` is named."""
    _setup(tmp_path, monkeypatch)
    tasks = P.PHASES["s2_ho_a0"]["build"]()
    tasks[3] = dict(tasks[3], argv=[a for a in tasks[3]["argv"] if a != "--print_tree"])
    assert P.tree_gp_without_full_tree(tasks) == [tasks[3]["key"]]


def test_every_stage2_tree_gp_task_prints_the_whole_tree(tmp_path, monkeypatch, request):
    """D092 on the committed data: every task of every Stage 2 phase (stage2_tasks.json's phases, as registered) that
    runs tree GP (reach.tasks.s2_run) carries `--print_tree`, whatever the phase; at least one phase has such tasks.
    The locks are lifted in memory only; nothing runs."""
    from reach.runner.phases import registration_build
    if not (P.ROOT / "data/registration/heldout_levels.json").exists():
        pytest.skip("no held-out levels")
    request.getfixturevalue("cgp_exe")                    # CGP phases need the binary to plan (REQ-RUN-27)
    lock = tmp_path / "stage.txt"
    lock.write_text("test build only\n")
    monkeypatch.setattr(L, "STAGE1_FILE", lock)
    monkeypatch.setattr(L, "STAGE2_FILE", lock)
    n = 0
    for name in L.STAGE2_PHASES:
        tasks = registration_build(name)()
        assert P.tree_gp_without_full_tree(tasks) == [], name
        n += sum("reach.tasks.s2_run" in t["argv"] for t in tasks)
    assert n > 0

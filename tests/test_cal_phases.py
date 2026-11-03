"""Calibration phases (T14; REQ-RUN-18): locked until the Stage 1 registration is recorded; built from the G2 levels."""
import hashlib
import json

import pytest

from reach import s2_lock as L
from reach.runner import phases as P


def _setup(tmp_path, monkeypatch, stage1=True):
    reg = tmp_path / "data/registration"
    reg.mkdir(parents=True)
    if stage1:
        (reg / "stage1_registration.txt").write_text("https://osf.io/xxxxx\n")
    (reg / "levels.json").write_text(json.dumps({"BOOL-NAND-n5": {"main": [5, 7, 9, 11, 13], "budget": [15, 17, 19, 21, 23]}}))
    d = tmp_path / "data/targets_s2/BOOL-NAND-n5"
    d.mkdir(parents=True)
    ts = [dict(target_id=f"T{K}_{i}", expression="nand(x1, x2)", K_construction=K, K_proven=K)
          for K in (5, 7, 9, 11, 13, 15, 17, 19, 21, 23) for i in range(30)]
    (d / "calibration.json").write_text(json.dumps(ts))
    (d / "calibration.sha256").write_text(hashlib.sha256((d / "calibration.json").read_bytes()).hexdigest() + "  x\n")
    monkeypatch.setattr(P, "ROOT", tmp_path)
    monkeypatch.setattr(L, "STAGE1_FILE", reg / "stage1_registration.txt")
    monkeypatch.setattr(P, "CAL_SETS", ["BOOL-NAND-n5"])


def test_confirmatory_phases_refuse_before_stage1(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, stage1=False)
    for name in ("s2_cal_a16", "s2_cal_a0", "s2_cal_mech", "s2_cal_alloc"):
        with pytest.raises(PermissionError):
            P.PHASES[name]["build"]()


def test_calibration_blocks_follow_the_plan(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    a16 = P.PHASES["s2_cal_a16"]["build"]()
    a0 = P.PHASES["s2_cal_a0"]["build"]()
    mech = P.PHASES["s2_cal_mech"]["build"]()
    alloc = P.PHASES["s2_cal_alloc"]["build"]()
    assert len(a16) == 5 * 30 * 5 and all(t["meta"]["budget"] == 16 for t in a16)
    assert len(a0) == 5 * 30 * 5 and all(t["meta"]["K"] in (5, 7, 9, 11, 13) for t in a0)
    assert len(mech) == 10 * 30 * 5 * 2 and {t["meta"]["algorithm"] for t in mech} == {"B", "HC"}
    assert len(alloc) == 5 * 20 * 27 and all(t["meta"]["budget"] == 1 for t in alloc)
    keys = [t["key"] for t in a16 + a0 + mech + alloc]
    assert len(keys) == len(set(keys))


def test_controls_and_size_checks_get_regime_A_only(tmp_path, monkeypatch):
    """D053: F-EML-R, IMP0-R and the n = 6 sets get a0 and a16 only, never mech, alloc, pool or robust."""
    assert set(P.REDUCED_SETS) == {"F-EML-R", "BOOL-IMP0R-n5", "BOOL-NAND-n6", "BOOL-AON-n6"}
    assert not set(P.REDUCED_SETS) & set(P.FULL_SETS) and set(P.FULL_SETS) | set(P.REDUCED_SETS) == set(P.CAL_SETS)


def test_pool_rand_and_robust_phases(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(P, "FULL_SETS", ["BOOL-NAND-n5"])
    d = tmp_path / "data/targets_s2/BOOL-NAND-n5"
    pool = [dict(target_id=f"BOOL-NAND-n5_pool{i:03d}", pool_id=f"pool{i:03d}", expression="nand(x1, x2)", K_construction=3,
                 K_proven=3) for i in range(4)]
    (d / "pool.json").write_text(json.dumps(pool))
    (d / "pool.sha256").write_text(hashlib.sha256((d / "pool.json").read_bytes()).hexdigest() + "  x\n")
    lv = json.loads((tmp_path / "data/registration/levels.json").read_text())
    lv["_robust"] = {"bool": ["BOOL-NAND-n5"], "real": []}
    (tmp_path / "data/registration/levels.json").write_text(json.dumps(lv))
    pool_t = P.PHASES["s2_cal_pool"]["build"]()
    assert len(pool_t) == 4 * 30 and all(t["meta"]["budget"] == 1 for t in pool_t)
    rand_t = P.PHASES["s2_cal_rand"]["build"]()
    assert len(rand_t) == 4 and all("--n" in t["argv"] for t in rand_t)
    rob = P.PHASES["s2_cal_robust"]["build"]()
    assert len(rob) == 6 * 5 * 20 * 5 and {t["meta"]["config"] for t in rob} == {"R1", "R2", "R3", "R4", "R5", "R6"}
    assert all(t["meta"]["budget"] == 4 for t in rob)
    for name in ("s2_cal_pool", "s2_cal_rand", "s2_cal_robust"):
        monkeypatch.setattr(L, "STAGE1_FILE", tmp_path / "missing.txt")
        with pytest.raises(PermissionError):
            P.PHASES[name]["build"]()


def test_calibration_targets_use_the_registered_levels_and_seed(tmp_path, monkeypatch):
    """Target generation is not a search: it runs before Stage 1, from levels.json, 30 per level, calibration seed."""
    from reach.seeds import TARGET_SEEDS
    _setup(tmp_path, monkeypatch, stage1=False)
    tasks = P.PHASES["s2_targets_cal"]["build"]()
    assert [t["key"] for t in tasks] == ["BOOL-NAND-n5"]
    a = tasks[0]["argv"]
    assert a[a.index("--name") + 1] == "calibration" and a[a.index("--n") + 1] == "30"
    assert a[a.index("--seed") + 1] == str(TARGET_SEEDS["calibration"])
    assert a[a.index("--levels") + 1] == "5,7,9,11,13,15,17,19,21,23"
    pool = P.PHASES["s2_targets_pool"]["build"]()
    assert sorted(t["key"] for t in pool) == ["bool", "real"]
    assert all(t["argv"][t["argv"].index("--seed") + 1] == str(TARGET_SEEDS["pool"]) for t in pool)


def test_operon_phase_is_locked_and_runs_in_the_ext_environment(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, stage1=False)
    with pytest.raises(PermissionError):
        P.PHASES["s2_cal_operon"]["build"]()
    _setup_operon = tmp_path / "data/targets_s2/R-ARITH4"
    _setup_operon.mkdir(parents=True)
    ts = [dict(target_id=f"A{K}_{i}", expression="+(x, 1)", K_construction=K, K_proven=K)
          for K in (3, 5) for i in range(30)]
    (_setup_operon / "calibration.json").write_text(json.dumps(ts))
    (_setup_operon / "calibration.sha256").write_text(
        hashlib.sha256((_setup_operon / "calibration.json").read_bytes()).hexdigest() + "  x\n")
    (tmp_path / "data/registration/levels.json").write_text(json.dumps({"R-ARITH4": {"main": [3], "budget": [5]}}))
    (tmp_path / "data/registration/stage1_registration.txt").write_text("https://osf.io/xxxxx\n")
    monkeypatch.setattr(P, "CAL_SETS", ["R-ARITH4", "BOOL-NAND-n5"])
    tasks = P.PHASES["s2_cal_operon"]["build"]()
    assert len(tasks) == 2 * 30 * 5 * 2                                  # levels x targets x main seeds x configs
    t = tasks[0]
    assert t["argv"][0].endswith(".venv-ext/bin/python") and "operon-libs" in t["env"]["DYLD_LIBRARY_PATH"]
    assert t["argv"][t["argv"].index("--budget") + 1] == "16" and "|operon|" in t["key"]
    assert {x["meta"]["set"] for x in tasks} == {"R-ARITH4"}             # Operon covers standard real sets only

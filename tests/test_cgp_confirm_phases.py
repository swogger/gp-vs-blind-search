"""Confirmatory CGP phases s2_cal_cgp and s2_ho_cgp (D076; REQ-RUN-28 to 31), written before the code. Built in a
temporary root with synthetic targets; nothing is planned into data/runs and nothing runs."""
import argparse
import hashlib
import json

import pytest

from reach import s2_lock as L
from reach.runner import phases as P
from reach.seeds import s2_seeds

CAL = ["F-EML", "BOOL-NAND-n5"]
HO = ["R-ARITH", "BOOL-AN-n5"]
SETTINGS = dict(genome_length=dict(real=500, bool=100), soft_cpu_limit_s=dict(real=900, bool=600), levels="all",
                rule="synthetic", pilot_phase="s2_pilot_cgp", pilot_results_sha256="0" * 64)
_BASE, _C0, _C1 = [4.0, -0.5, 0.6], -1.0, 0.5
_MODEL = dict(unit="nodes", budget_term=True, feature=None, base_params=_BASE, transfer=dict(c0=_C0, c1=_C1),
              params=[_C0 + _C1 * _BASE[0]] + [_C1 * x for x in _BASE[1:]], budgets=[1, 2, 4, 8, 16],
              baseline=dict(unit="nodes", budget_term=True, feature=None, params=[3.0, -0.4, 0.5]))   # D090
TRANSFER = dict(real=_MODEL, bool=_MODEL)                     # a valid H10b transfer model (row 91)


def _write_checked(p, obj):
    p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    p.with_suffix(".sha256").write_text(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n")


def _targets(d, name, levels, per_level):
    d.mkdir(parents=True, exist_ok=True)
    ts = [dict(target_id=f"{d.name}_{name}_{K}_{i}", expression="x", K_construction=K, K_proven=K)
          for K in levels for i in range(per_level)]
    (d / f"{name}.json").write_text(json.dumps(ts))
    (d / f"{name}.sha256").write_text(hashlib.sha256((d / f"{name}.json").read_bytes()).hexdigest() + "  x\n")


def _setup(tmp_path, monkeypatch, stage2=True, settings=True, ho_levels=True, transfer=True):
    reg = tmp_path / "data/registration"
    reg.mkdir(parents=True)
    if stage2:
        (reg / "stage2_registration.txt").write_text("https://osf.io/xxxxx\n")
    (reg / "levels.json").write_text(json.dumps({g: {"main": [3, 5], "budget": [7, 9]} for g in CAL}))
    if ho_levels:
        _write_checked(reg / "heldout_levels.json", {g: {"main": [3, 5], "budget": [7, 9]} for g in HO})   # row 124
    if settings:
        _write_checked(reg / "cgp_settings.json", SETTINGS)
    if transfer:
        _write_checked(reg / "cgp_transfer.json", TRANSFER)
    for g in CAL:
        _targets(tmp_path / f"data/targets_s2/{g}", "calibration", (3, 5, 7, 9, 11), 32)   # 32: first 30 used
    for g in HO:
        _targets(tmp_path / f"data/targets_s2/{g}", "heldout", (3, 5, 7, 9), 30)
    monkeypatch.setattr(P, "ROOT", tmp_path)
    monkeypatch.setattr(P, "CAL_SETS", CAL)
    monkeypatch.setattr(P, "heldout_sets", lambda: HO)
    monkeypatch.setattr(L, "STAGE2_FILE", reg / "stage2_registration.txt")
    return reg


def _arg(argv, flag):
    return argv[argv.index(flag) + 1]


def test_cgp_phases_refuse_before_stage2(tmp_path, monkeypatch, cgp_exe):
    _setup(tmp_path, monkeypatch, stage2=False)
    for name in ("s2_cal_cgp", "s2_ho_cgp"):
        with pytest.raises(PermissionError, match="Stage 2"):
            P.PHASES[name]["build"]()
    with pytest.raises(PermissionError, match="Stage 2"):
        P.registration_build("s2_ho_cgp")()


def test_cgp_phases_refuse_without_the_settings_file(tmp_path, monkeypatch, cgp_exe):
    from reach.runner.__main__ import main
    _setup(tmp_path, monkeypatch, settings=False)
    for name in ("s2_cal_cgp", "s2_ho_cgp"):
        with pytest.raises(FileNotFoundError, match=r"cgp_settings\.json.*reach\.analysis\.s2\.cgp_pilot"):
            P.PHASES[name]["build"]()
    runs = tmp_path / "runs"
    with pytest.raises(FileNotFoundError):
        main(["--runs", str(runs), "plan", "s2_cal_cgp"])
    assert not (runs / "s2_cal_cgp/manifest.json").exists()


def test_cgp_phases_refuse_a_changed_or_invalid_settings_file(tmp_path, monkeypatch, cgp_exe):
    reg = _setup(tmp_path, monkeypatch)
    p = reg / "cgp_settings.json"
    assert P.cgp_settings()["genome_length"] == SETTINGS["genome_length"]            # control: the valid file passes
    p.write_text(p.read_text().replace('"real": 500', '"real": 2000'))                 # planted: edited after writing
    with pytest.raises(ValueError, match="checksum"):
        P.PHASES["s2_cal_cgp"]["build"]()
    p.with_suffix(".sha256").unlink()
    with pytest.raises(ValueError, match="checksum"):
        P.cgp_settings()
    for bad in (dict(SETTINGS, genome_length=dict(real=300, bool=100)),               # a length the pilot never ran
                dict(SETTINGS, genome_length=dict(real=500)),                          # a domain missing
                dict(SETTINGS, levels="main"),                                         # not a registered choice
                dict(SETTINGS, soft_cpu_limit_s=dict(real=100, bool=600))):            # below the 600 s floor
        _write_checked(p, bad)
        with pytest.raises(ValueError):
            P.cgp_settings()


def test_cal_cgp_follows_the_plan(tmp_path, monkeypatch, cgp_exe):
    reg = _setup(tmp_path, monkeypatch)
    tasks = P.PHASES["s2_cal_cgp"]["build"]()
    seeds = s2_seeds("cgp_calibration")
    assert len(seeds) == 5
    for other in ("main", "calibration", "heldout", "pilot", "cgp_pilot", "cgp_validation", "benchmark", "pool"):
        assert not set(seeds) & set(s2_seeds(other)), other
    assert len(tasks) == len({t["key"] for t in tasks}) == 2 * 4 * 30 * 5           # sets x 10->4 levels x 30 x 5
    cells = set()
    for t in tasks:
        a, m = t["argv"], t["meta"]
        dom = "real" if m["set"] == "F-EML" else "bool"
        assert _arg(a, "--targets") == f"data/targets_s2/{m['set']}/calibration.json"
        assert int(_arg(a, "--n_nodes")) == SETTINGS["genome_length"][dom] and float(_arg(a, "--budget")) == 16
        assert float(_arg(a, "--timeout")) == SETTINGS["soft_cpu_limit_s"][dom] and t["cpu_limit"] > 600
        assert (_arg(a, "--levels_back"), _arg(a, "--mutation"), _arg(a, "--strategy"), _arg(a, "--stop_on"),
                _arg(a, "--penalty")) == ("unrestricted", "single", "1+4", "exact", "scaled")
        assert m["K"] in (3, 5, 7, 9) and m["seed"] in seeds and m["block"] == "cal_cgp"
        assert int(m["target_id"].rsplit("_", 1)[1]) < 30                              # first 30 per level
        cells.add((m["set"], m["K"]))
    assert cells == {(g, K) for g in CAL for K in (3, 5, 7, 9)}
    _write_checked(reg / "cgp_settings.json", dict(SETTINGS, levels="budget"))       # the 100 CPU-h fallback
    tasks = P.PHASES["s2_cal_cgp"]["build"]()
    assert len(tasks) == 2 * 2 * 30 * 5 and {t["meta"]["K"] for t in tasks} == {7, 9}


def test_ho_cgp_follows_the_plan(tmp_path, monkeypatch, cgp_exe):
    _setup(tmp_path, monkeypatch)
    tasks = P.PHASES["s2_ho_cgp"]["build"]()
    seeds = s2_seeds("cgp_heldout")
    assert len(seeds) == 5 and not set(seeds) & (set(s2_seeds("cgp_calibration")) | set(s2_seeds("heldout")))
    assert len(tasks) == len({t["key"] for t in tasks}) == 2 * 4 * 30 * 5
    for t in tasks:
        a, m = t["argv"], t["meta"]
        dom = "real" if m["set"] == "R-ARITH" else "bool"
        assert _arg(a, "--targets") == f"data/targets_s2/{m['set']}/heldout.json" and m["block"] == "ho_cgp"
        assert int(_arg(a, "--n_nodes")) == SETTINGS["genome_length"][dom] and float(_arg(a, "--budget")) == 16
        assert m["seed"] in seeds and m["set"] in HO
    assert sorted(t["key"] for t in P.registration_build("s2_ho_cgp")()) == sorted(t["key"] for t in tasks)
    tmp2 = tmp_path / "nolevels"
    _setup(tmp2, monkeypatch, ho_levels=False)
    with pytest.raises(FileNotFoundError, match=r"heldout_levels\.json is missing \(written by the freeze step"):
        P.PHASES["s2_ho_cgp"]["build"]()


def test_ho_cgp_waits_for_the_frozen_transfer_model(tmp_path, monkeypatch, cgp_exe):
    reg = _setup(tmp_path, monkeypatch, transfer=False)
    with pytest.raises(PermissionError, match="H10b"):
        P.PHASES["s2_ho_cgp"]["build"]()
    assert P.registration_build("s2_ho_cgp")()                                       # registration needs no model
    _write_checked(reg / "cgp_transfer.json", TRANSFER)
    assert P.PHASES["s2_ho_cgp"]["build"]()                                          # control: frozen and valid
    (reg / "cgp_transfer.json").write_text("{}")                                     # planted: changed after freezing
    with pytest.raises(PermissionError, match="H10b"):
        P.PHASES["s2_ho_cgp"]["build"]()


def _tasks(keys):
    return [dict(key=k, argv=["x"], cpu_limit=1) for k in keys]


def test_the_runner_checks_cgp_phases_against_the_stage2_list(tmp_path, monkeypatch):
    """Guardrail row 89: s2_cal_cgp starts with s2_cal_ but is registered in Stage 2. A matching entry planted in the
    Stage 1 list must not pass it; the Stage 2 list must."""
    from reach.runner import __main__ as M
    one = _tasks(["a"])
    entry = {n: dict(n_tasks=1, sha256=L.task_list_hash(one)) for n in ("s2_cal_cgp", "s2_ho_cgp")}
    s1, s2 = tmp_path / "s1.json", tmp_path / "s2.json"
    s1.write_text(json.dumps(entry))
    monkeypatch.setattr(L, "TASKS_FILE", s1)
    monkeypatch.setattr(L, "TASKS2_FILE", s2)
    for name in ("s2_cal_cgp", "s2_ho_cgp"):
        monkeypatch.setitem(M.PHASES, name, dict(build=lambda: _tasks(["a"]), description="", verify=None))
        assert L.registered_tasks_file(name) == s2
        with pytest.raises(PermissionError, match="no registered task list"):
            M.cmd_plan(argparse.Namespace(phase=name, runs=tmp_path / "runs", force=False), quiet=True)
    s2.write_text(json.dumps(entry))
    for name in ("s2_cal_cgp", "s2_ho_cgp"):
        M.cmd_plan(argparse.Namespace(phase=name, runs=tmp_path / "runs", force=False), quiet=True)
    assert L.registered_tasks_file("s2_cal_a16") == s1 and L.registered_tasks_file("s2_ho_a0") == s2
    assert L.registered_tasks_file("s2_ho_targets") is None and L.registered_tasks_file("s2_pilot_cgp") is None


def test_cgp_tasks_accept_registered_targets_after_stage2(tmp_path, monkeypatch, cgp_exe):
    _setup(tmp_path, monkeypatch)
    t = dict(target_id="F-EML_calibration_3_0", K_construction=3, K_proven=3)
    assert P._cgp_task("x", "F-EML", "calibration", t, 100, 1, 16, 600)
    t = dict(target_id="R-ARITH_heldout_3_0", K_construction=3, K_proven=3)
    assert P._cgp_task("x", "R-ARITH", "heldout", t, 100, 1, 16, 600)
    for tname in ("pool", "benchmark", "bench"):                                       # never a CGP target file
        with pytest.raises(PermissionError):
            P._cgp_task("x", "F-EML", tname, t, 100, 1, 16, 600)


def test_stage2_task_list_script_covers_every_stage2_phase():
    import importlib.util
    from reach.paths import ROOT
    spec = importlib.util.spec_from_file_location("s2_task_lists", ROOT / "scripts/s2_task_lists.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert set(mod.STAGE2) == set(L.STAGE2_PHASES)
    assert {"s2_cal_cgp", "s2_ho_a0", "s2_ho_a16", "s2_ho_bench", "s2_ho_operon", "s2_ho_cgp"} == set(L.STAGE2_PHASES)
    assert all(n in P.PHASES for n in L.STAGE2_PHASES)
    assert not set(mod.CONFIRMATORY) & set(L.STAGE2_PHASES)
    assert sorted(mod.CONFIRMATORY + [n for n in L.STAGE2_PHASES if n.startswith("s2_cal_")]) == \
        sorted(n for n in P.PHASES if n.startswith("s2_cal_"))
    assert {n for n in P.PHASES if n.startswith("s2_ho_") and n != "s2_ho_targets"} <= set(L.STAGE2_PHASES)


def test_ho_cgp_refuses_a_checksummed_but_invalid_transfer_model(tmp_path, monkeypatch, cgp_exe):
    """Guardrail row 91: a file matching its own .sha256 is not enough; both domains need a valid transfer model."""
    reg = _setup(tmp_path, monkeypatch)
    assert P.PHASES["s2_ho_cgp"]["build"]()                                         # control: the valid model passes
    for bad in (dict(real={}, bool={}), dict(real=TRANSFER["real"]),
                dict(TRANSFER, bool=dict(TRANSFER["bool"], params=[0.0, 0.0, 0.0]))):
        _write_checked(reg / "cgp_transfer.json", bad)                               # planted: checksummed, invalid
        with pytest.raises(PermissionError, match="H10b"):
            P.PHASES["s2_ho_cgp"]["build"]()


def test_docs_name_the_cgp_seed_slices():
    """Guardrail row 94: cgp.md said the calibration uses the "main" seeds, and stage2_form.md gave slice positions
    as seed numbers ("seeds 710-714"); both name the slices of reach.seeds."""
    import re
    from pathlib import Path
    from reach.seeds import S2_SLICES
    root = Path(__file__).resolve().parents[1]
    form = (root / "docs/registration/osf/stage2_form.md").read_text()
    cgp = (root / "docs/design/cgp.md").read_text()
    for name in ("cgp_calibration", "cgp_heldout"):
        a, b = S2_SLICES[name]
        assert f"{name}, S2_SEEDS[{a}:{b}]" in form, name
    assert not re.search(r"seeds \d+-\d+", form)
    for name in ("cgp_validation", "cgp_pilot", "cgp_calibration", "cgp_heldout"):
        a, b = S2_SLICES[name]
        assert f"`{name}` (S2_SEEDS[{a}:{b}])" in cgp, name
    assert not re.search(r"[Cc]alibration uses the .main. seeds", cgp)

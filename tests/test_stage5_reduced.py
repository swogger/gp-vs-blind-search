"""Stage 5 reduced to H11-2L (D126): only s5_long_extra is registered, the other Stage 5 phases stay locked; the
long runs carry no counter; the confirmatory driver analyses only H11-2L and refuses to run without the registration
files (review M2), with a bootstrap other than the registered one (m7), on rows without a target checksum or a
counted row without its lookup (m3); the registered test table is pinned (m1). Guardrail rows 202 to 205. Written
before the code; every check is shown to fire on a planted case."""
import argparse
import hashlib
import json
import shutil

import pytest

import _stage5_synth as Y
from reach import s2_lock as L
from reach.analysis.s2 import stage5
from reach.analysis.s2 import stage5_report as D
from reach.analysis.s2 import stage5_stats as S
from reach.runner import phases as P

PAUSED = ("s5_grid", "s5_boolsub", "s5_wide", "s5_selfree", "s5_low_base", "s5_pop1000", "s5_revisit_robust",
          "s5_revisit_restart", "s5_union", "s5_revisit_ho16", "s5_frozen")


# ---------------------------------------------------------------- the lock (REQ-RUN-42; guardrail row 202)
def test_only_the_long_runs_are_registered_and_the_paused_phases_stay_locked(tmp_path, monkeypatch):
    from reach.runner import __main__ as M
    assert L.STAGE5_PHASES == ("s5_long_extra",)
    assert sorted(L.STAGE5_PAUSED) == sorted(PAUSED) and not set(L.STAGE5_PAUSED) & set(L.STAGE5_PHASES)
    reg = tmp_path / "stage5_registration.txt"
    reg.write_text("https://osf.io/xxxxx (test)\n")
    monkeypatch.setattr(L, "STAGE5_FILE", reg)
    fake = [dict(key="a", argv=["x"], cpu_limit=1)]
    entry = dict(n_tasks=1, sha256=L.task_list_hash(fake), sha256_full=L.task_list_full_hash(fake))
    tasks5 = tmp_path / "stage5_tasks.json"
    tasks5.write_text(json.dumps({"s5_long_extra": entry}))
    monkeypatch.setattr(L, "TASKS5_FILE", tasks5)
    for name in PAUSED:                       # registration recorded, phase not in the task list: refused at plan
        assert L.registered_tasks_file(name) == L.TASKS5_FILE
        monkeypatch.setitem(M.PHASES, name, dict(build=lambda: [dict(t) for t in fake], description="", verify=None))
        with pytest.raises(PermissionError, match="no registered task list"):
            M.cmd_plan(argparse.Namespace(phase=name, runs=tmp_path / "runs", force=False), quiet=True)
    # the plant: the same paused phase listed in the task file would plan (so the task list is what refuses it)
    tasks5.write_text(json.dumps({"s5_long_extra": entry, "s5_grid": entry}))
    M.cmd_plan(argparse.Namespace(phase="s5_grid", runs=tmp_path / "runs", force=False), quiet=True)


def test_long_runs_carry_no_counter(monkeypatch, tmp_path):
    """D126: H11-2L needs no counts; hard CPU limit = soft + 100 s on every long run."""
    f = tmp_path / "stage5_registration.txt"
    f.write_text("filed (test)\n")
    monkeypatch.setattr(L, "STAGE5_FILE", f)
    tasks = P.PHASES["s5_long_extra"]["build"]()
    assert len(tasks) == 3750
    assert not [t for t in tasks if "--count_distinct" in t["argv"] or "--enum_dir" in t["argv"]]
    assert {t["cpu_limit"] for t in tasks} == {7300}


# ---------------------------------------------------------------- the registered test table (guardrail row 203)
def test_the_registered_test_table_is_pinned():
    want = {
        "H12-1": dict(family="H12", kind="tost", delta=1.0),
        "H12-2": dict(family="H12", kind="above", margin=0.0),
        "H12-3": dict(family="H12", kind="tost", delta=1.0),
        "H13-1": dict(family="H13", kind="tost", delta=1.0),
        "H13-2": dict(family="H13", kind="tost", delta=1.0),
        "H14-1": dict(family="H14", kind="above", margin=-1.0),
        "H14-2": dict(family="H14", kind="above", margin=-1.0),
        "H15-1": dict(family="H15", kind="above", margin=-1.0),
        "H15-2": dict(family="H15", kind="above", margin=0.5),
        "H16-1": dict(family="H16", kind="conj_tost", delta=1.0),
        "H16-2": dict(family="H16", kind="tost", delta=0.25, center=1.0),
        "H11-2L": dict(family="H11-2L", kind="conj_above", margin=2.0),
    }
    assert S.TESTS == want
    assert (stage5.BOOT, stage5.SEED, stage5.KEEP, stage5.ALPHA) == (2000, 20261020, 0.9, 0.05)
    assert D.CONF["registered_families"] == ("H11-2L",) and D.CONF["registered"] is True


# ---------------------------------------------------------------- the confirmatory driver (guardrail rows 204, 205)
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("s5reduced")
    cfg, levels, extra = Y.make_root(root)
    return root, cfg, levels


def _entry(root, phase):
    from reach.runner.phases import _manifest_tasks
    tasks = _manifest_tasks(phase, root)
    return dict(n_tasks=len(tasks), sha256=L.task_list_hash(tasks), sha256_full=L.task_list_full_hash(tasks))


def _register(root, tasks_sha=None):
    reg = root / "data/registration"
    reg.mkdir(parents=True, exist_ok=True)
    (reg / "stage5_registration.txt").write_text("https://osf.io/xxxxx (test)\n")
    (reg / "stage5_tasks.json").write_text(json.dumps({"s5_long_extra": _entry(root, "s5_long_extra")}))
    (reg / "stage4_tasks.json").write_text(json.dumps({"s4_restart": _entry(root, "s4_restart")}))
    (reg / "stage2_tasks.json").write_text(json.dumps({"s2_ho_a16": _entry(root, "s2_ho_a16")}))
    sha = tasks_sha or hashlib.sha256((reg / "stage5_tasks.json").read_bytes()).hexdigest()
    (reg / "stage5_checksums.txt").write_text(f"{sha}  data/registration/stage5_tasks.json\n# commit test\n")


def test_confirmatory_build_needs_the_registration_files(synth, tmp_path, monkeypatch):
    src, cfg, levels = synth
    root = tmp_path / "root"
    shutil.copytree(src, root)
    conf = dict(cfg, registered=True, registered_families=("H11-2L",))
    monkeypatch.setattr(stage5, "BOOT", 3)
    reg = root / "data/registration"

    def build(**kw):
        return D.build(root, conf, boot=kw.get("boot", 3), families=kw.get("families"), levels_given=levels)

    with pytest.raises(RuntimeError, match="stage5_registration.txt"):          # plant 1: nothing registered
        build()
    _register(root)
    res = build()                                                               # the good case
    assert set(res["tests"]) == {"H11-2L"} and res["families"] == ["H11-2L"]
    for missing in ("stage5_tasks.json", "stage5_checksums.txt", "stage4_tasks.json", "stage2_tasks.json"):
        keep = (reg / missing).read_text()
        (reg / missing).unlink()                                                # plant 2: one file missing
        with pytest.raises(RuntimeError, match=missing):
            build()
        (reg / missing).write_text(keep)
    _register(root, tasks_sha="0" * 64)                                        # plant 3: tasks file off its checksum
    with pytest.raises(RuntimeError, match="stage5_checksums.txt"):
        build()
    _register(root)
    (reg / "stage4_tasks.json").write_text(json.dumps({"s4_other": _entry(root, "s4_restart")}))
    with pytest.raises(RuntimeError, match="s4_restart: not in stage4_tasks.json"):   # plant 4: earlier stage
        build()
    _register(root)
    with pytest.raises(RuntimeError, match="not registered"):                   # plant 5: a paused family
        build(families=["H12"])
    with pytest.raises(RuntimeError, match="bootstrap"):                        # plant 6: another B (m7)
        build(boot=5)
    # without `registered` (the synthetic and pilot configurations) nothing changes
    (reg / "stage5_registration.txt").unlink()
    assert "H11-2L" in D.build(root, cfg, boot=3, families=["H11-2L"], levels_given=levels)["tests"]


def test_rows_without_a_target_checksum_or_a_counted_row_without_its_lookup_are_refused():
    tables = {"R-X": {"t0": dict(target_id="t0", _file="calibration")}}
    shas = {"R-X": {"calibration": "a" * 64}}
    good = dict(key="k", set="R-X", target_id="t0", targets_sha256="a" * 64)
    D.check_rows_targets([good], tables, shas)
    with pytest.raises(RuntimeError, match="targets_sha256"):                   # plant: the field missing
        D.check_rows_targets([dict(good, targets_sha256=None)], tables, shas)
    counted = dict(key="c", set="R-X", rv=dict(lookup=dict(sha256="b" * 64)))
    assert D.check_lookups([counted]) == {"R-X": "b" * 64}
    with pytest.raises(RuntimeError, match="no lookup"):                        # plant: a counted row, no lookup
        D.check_lookups([counted, dict(key="d", set="R-X", rv=dict(stop=dict(n=1)))])
    assert D.check_lookups([dict(key="e", set="R-X")]) == {}                     # an uncounted row has none


# ---------------------------------------------------------------- like for like with Stage 4 (guardrail row 206)
def _logistic_arm(sizes, k50, slope=1.5):
    import math
    return {f"t{i}": (float(b), 1 / (1 + math.exp(-slope * (k50 - b))), 5) for i, b in enumerate(sizes)}


def test_h11_2l_brackets_the_restart_arm_only_as_stage4_did():
    """Stage 4 replaced a restart K50 above the top tested level by that level and read one run's K50 by the logistic
    unbracketed (remedy.h11: law._k50). H11-2L does the same: a one-run K50 above its top level stays extrapolated."""
    sizes = [8.0, 9.0, 10.0, 11.0, 12.0] * 3
    restart, long = _logistic_arm(sizes, 13.0), _logistic_arm(sizes, 12.6)     # both above the top level, 12
    ids = {"H-A": sorted(restart)}
    point = S.heldout(dict(restart={"H-A": restart}, long={"H-A": long}), ["H-A"])(ids)
    v = point["per_set"]["H-A"]
    assert v["restart"]["k50"] == 12.0 and v["restart"]["bracketed"]           # capped, as Stage 4
    assert v["long"]["k50"] == pytest.approx(12.6, abs=1e-3) and not v["long"]["bracketed"]   # not capped
    assert point["values"]["H11-2L"]["primary"]["H-A"] == pytest.approx(12.0 - 12.6, abs=1e-3)


@pytest.mark.study_machine
def test_h11_2l_path_on_stage4s_arms_reproduces_the_registered_h11_2():
    """Instrument check (independent review M3, 2026-11-09): H11-2L's statistic path (heldout, run_block; 2,000
    replicates, seed 20261020) on Stage 4's own arms (the restart arm on its 55 levels, one run on Stage 2's five
    budget levels; Rule S1's guard bypassed on purpose) reproduces h11.json's per-set gains, intervals and p's and
    H11-2's p. It failed while the one-run arm was bracketed (RS-add_sqrt p 0.0010 against 0.0015)."""
    from reach.analysis.s2.load import read_phase
    from reach.paths import ROOT
    H = D.HELDOUT
    tables = {g: D.target_table(ROOT, g, ("heldout", "restart"))[0] for g in H}
    sizes = {g: {t: v["units"]["bits_sem"] for t, v in tables[g].items() if D.proven(v)} for g in H}
    rb = D._by([r for r in read_phase("s4_restart", ROOT) if r["set"] in H], "set", "target_id")
    one16 = read_phase("s2_ho_a16", ROOT)
    arms = dict(restart={g: S.restart_arm({t: rr for (h, t), rr in rb.items() if h == g}, sizes[g], 32, 100)
                         for g in H},
                long={g: S.one_run_arm([r for r in one16 if r["set"] == g], 3200, sizes[g]) for g in H})
    ids = {g: sorted(arms["restart"][g]) for g in H}
    res, point = S.run_block(["H11-2L"], ids, S.heldout(arms, H), stage5.BOOT)
    reg = json.loads((ROOT / "data/analysis/h11.json").read_text())["h11"]
    pr = res["H11-2L"]["tests"]["primary"]
    assert pr["p"] == pytest.approx(reg["tests"]["H11-2"]["p"], abs=1e-12)
    for g in H:
        m, w = pr["member_tests"][g], reg["per_set_gain_tests"][g]
        assert m["p"] == pytest.approx(w["p"], abs=1e-12), g
        assert (m["lo90"], m["hi90"]) == pytest.approx((w["lo90"], w["hi90"]), abs=1e-9), g
        assert point["values"]["H11-2L"]["primary"][g] == pytest.approx(reg["per_set"][g]["gain"], abs=1e-9), g


# ---------------------------------------------------------------- the registration files (guardrail row 207)
def test_stage5_task_list_matches_its_checksum_and_lists_only_the_registered_phase():
    """Once filed: the runner trusts stage5_tasks.json, so it must equal the file the checksum list registers and name
    s5_long_extra alone (re-running the task-list script after a change to STAGE5_PHASES would otherwise unlock a
    paused phase)."""
    from reach.paths import ROOT
    reg = ROOT / "data/registration"
    if not (reg / "stage5_tasks.json").exists() or not (reg / "stage5_checksums.txt").exists():
        pytest.skip("Stage 5 registration files not written yet")
    assert sorted(json.loads((reg / "stage5_tasks.json").read_text())) == ["s5_long_extra"]
    sums = D.registered_checksums(ROOT)
    assert sums["data/registration/stage5_tasks.json"] == hashlib.sha256((reg / "stage5_tasks.json").read_bytes()
                                                                         ).hexdigest()
    assert not (reg / "stage5_lookups.json").exists()                           # nothing is counted (D126)


def test_strict_driver_checks_earlier_stages_full_hash_and_the_prefix_phase(synth, tmp_path, monkeypatch):
    """Review minor: the registered driver checked s4_restart and s2_ho_a16 by count and key hash only, and never
    checked s2_ho_a0 (read by the prefix check) against its registration."""
    src, cfg, levels = synth
    root = tmp_path / "root"
    shutil.copytree(src, root)
    conf = dict(cfg, registered=True, registered_families=("H11-2L",))
    monkeypatch.setattr(stage5, "BOOT", 3)
    reg = root / "data/registration"
    short = [dict(key=f"ho_a0|H-A|x{i}|A|base|200|1", set="H-A", target_id=f"x{i}", seed=1) for i in range(2)]
    Y.write_phase(root, "s2_ho_a0", short)                                      # a prefix phase the long runs name
    monkeypatch.setattr(D, "_prefix_sources", lambda r, ph: ["s2_ho_a0"])
    _register(root)
    e2 = json.loads((reg / "stage2_tasks.json").read_text())
    e2["s2_ho_a0"] = _entry(root, "s2_ho_a0")
    (reg / "stage2_tasks.json").write_text(json.dumps(e2))
    D.build(root, conf, boot=3, levels_given=levels)                            # the good case
    bad = dict(e2, s2_ho_a16=dict(e2["s2_ho_a16"], sha256_full="0" * 64))       # plant 1: other settings
    (reg / "stage2_tasks.json").write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="full hash"):
        D.build(root, conf, boot=3, levels_given=levels)
    del e2["s2_ho_a0"]                                                          # plant 2: prefix phase unregistered
    (reg / "stage2_tasks.json").write_text(json.dumps(e2))
    with pytest.raises(RuntimeError, match="s2_ho_a0: not in stage2_tasks.json"):
        D.build(root, conf, boot=3, levels_given=levels)


# ---------------------------------------------------------------- the public form (guardrail row 208)
def _form_fields():
    from reach.paths import ROOT
    text = (ROOT / "docs/registration/osf/stage5_form.md").read_text()
    return text.split("\n---\n", 1)[1].split("## Before filing (checklist)", 1)[0]


def test_the_form_names_every_phase_run_on_the_registered_targets():
    """Independent review B1 (2026-11-09): the reduced form named s4_restart, s2_ho_a16 and s2_ho_a0 but not the
    Cartesian GP and Operon phases that ran on the same targets at 16 x E0. Every phase with a task on one of the 1,650
    targets must be named in the public fields (pilot and re-execution phases included where present)."""
    from reach.paths import ROOT
    rt = {(g, t["target_id"]) for g, _, t in P.restart_targets()}
    fields, missing = _form_fields(), []
    for d in sorted((ROOT / "data/runs").iterdir()):
        m = d / "manifest.json"
        if not m.exists() or d.name == "s5_long_extra":
            continue
        tasks = json.loads(m.read_text()).get("tasks", [])
        if any((t.get("meta", {}).get("set"), t.get("meta", {}).get("target_id")) in rt for t in tasks):
            if d.name not in fields:
                missing.append(d.name)
    assert not missing, f"phases run on the registered targets but not named in the form: {missing}"
    # the plant: the check fires on a form without one of them
    assert "s2_ho_cgp" in fields and "s2_ho_cgp" not in fields.replace("s2_ho_cgp", "")


def test_the_public_fields_carry_no_internal_labels():
    """Independent review M1 (2026-11-09): 'deviation 3' is an internal number no public record defines; the public
    fields also never name internal decisions, guardrails, requirements or private review material (D126)."""
    import re
    fields = _form_fields()
    bad = [p for p in (r"\bdeviation \d", r"\bD\d{3}\b", r"guardrail", r"REQ-", r"simulated", r"editorial",
                       r"assessment", r"reviewer", "—") if re.search(p, fields, re.I)]
    assert not bad, bad
    assert re.search(r"\bdeviation \d", "found as deviation 3", re.I)              # the plant fires

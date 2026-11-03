"""H6d re-run phases (REQ-RUN-39; D098, Phase 7b): the registered calibration tasks plus the counter, locked until
Stage 3; the pilot re-run; the row-by-row comparison with the originals. Written before the code."""
import json
from pathlib import Path

import pytest

from reach import s2_lock as L
from reach.runner import phases as P

ROOT = Path(__file__).resolve().parents[1]


def _open_stage3(monkeypatch, tmp_path):
    f = tmp_path / "stage3_registration.txt"
    f.write_text("Stage 3 filed (test)\n")
    monkeypatch.setattr(L, "STAGE3_FILE", f)


def test_revisit_tasks_are_the_registered_tasks_plus_the_option(monkeypatch, tmp_path):
    from reach.analysis.s2.phase1 import enum_dir
    _open_stage3(monkeypatch, tmp_path)
    for name, src in (("s2_revisit_a0", "s2_cal_a0"), ("s2_revisit_a16", "s2_cal_a16")):
        orig = {t["key"]: t for t in json.loads((ROOT / f"data/runs/{src}/manifest.json").read_text())["tasks"]}
        tasks = P.PHASES[name]["build"]()
        assert len(tasks) == len(orig) == 12000
        assert {t["meta"]["audit_key"] for t in tasks} == set(orig)
        for t in tasks:
            o = orig[t["meta"]["audit_key"]]
            assert t["key"] == f"revisit|{o['key']}" and t["meta"]["audit_of"] == src
            assert t["argv"] == o["argv"] + ["--count_distinct", "--enum_dir", str(enum_dir(o["meta"]["set"]))]
            assert t["cpu_limit"] == o["cpu_limit"] * P.REVISIT_CPU_FACTOR
            assert {k: v for k, v in t["meta"].items() if k not in ("audit_of", "audit_key")} == o["meta"]
            assert t.get("cwd", ".") == o.get("cwd", ".") and t.get("env", {}) == o.get("env", {})


def test_revisit_phases_are_locked_until_stage3(monkeypatch, tmp_path):
    monkeypatch.setattr(L, "STAGE3_FILE", tmp_path / "not_recorded.txt")
    for name in ("s2_revisit_a0", "s2_revisit_a16"):
        with pytest.raises(PermissionError):
            P.PHASES[name]["build"]()
        assert L.registered_tasks_file(name) == L.TASKS3_FILE
    assert L.registered_tasks_file("s2_revisit_pilot") is None
    assert P.PHASES["s2_revisit_pilot"]["build"]()                          # the pilot re-run is not locked


def test_pilot_tasks_cover_every_set_and_level_first():
    tasks = P.PHASES["s2_revisit_pilot"]["build"]()
    srcs = {"s2_pilot_reach_a", "s2_pilot_reach_b"}
    n = sum(json.loads((ROOT / f"data/runs/{s}/manifest.json").read_text())["n_tasks"] for s in srcs)
    assert len(tasks) == n and {t["meta"]["audit_of"] for t in tasks} == srcs
    cells = {(t["meta"]["set"], t["meta"]["K"]) for t in tasks}
    first = tasks[:len(cells)]
    assert {(t["meta"]["set"], t["meta"]["K"]) for t in first} == cells       # one task per set and level first
    for t in tasks:
        assert t["argv"][-3:-1] == ["--count_distinct", "--enum_dir"]
        assert (Path(t["argv"][-1]) / "counts.json").exists()


def _row(key, **kw):
    r = dict(key=key, block="cal_a0", set="BOOL-NAND-n3", domain="bool", algorithm="A", exact=True, gen_exact=3,
             gens_reached=3, pop_size=100, terminal_state="converged", final_error=0.0, best_expr="x1",
             cpu_seconds=1.0, wall_seconds=1.0, cpu_s=1.0)
    r.update(kw)
    return r


def _write(root, name, tasks, rows):
    d = root / f"data/runs/{name}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(dict(phase=name, n_tasks=len(tasks), tasks=tasks)))
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_revisit_report_flags_any_difference(tmp_path):
    from reach.analysis.s2.audit import revisit_report
    keys = ["k1", "k2", "k3", "k4", "k5", "k6"]
    near = dict(exact=False, gen_exact=None, gens_reached=800, terminal_state="budget_exhausted", converged=True)
    orig = [_row(k) for k in keys[:5]] + [_row("k6", **near)]      # k6: met the tolerance, no exact hit (review m1)
    _write(tmp_path, "s2_cal_a0", [dict(key=k) for k in keys], orig)
    tasks = [dict(key=f"revisit|{k}", meta=dict(audit_of="s2_cal_a0", audit_key=k)) for k in keys]
    rv = dict(stop=dict(n=400))                                   # converged at generation 3: 4 generations of 100
    good = [_row(f"revisit|{k}", rv=rv, cpu_seconds=2.0, wall_seconds=3.0, cpu_s=2.5) for k in keys[:5]]
    good.append(_row("revisit|k6", rv=dict(stop=dict(n=80000)), **near))   # 800 generations, not 801
    _write(tmp_path, "s2_revisit_a0", tasks, good)
    rep = revisit_report(tmp_path, "s2_revisit_a0")
    assert rep["ok"] and rep["sources"]["s2_cal_a0"]["identical"] == 6
    bad = [good[0], dict(good[1], gen_exact=4), {k: v for k, v in good[2].items() if k != "rv"},
           dict(good[3], rv=dict(stop=dict(n=300))), good[5]]          # k5 missing
    _write(tmp_path, "s2_revisit_a0", tasks, bad)
    rep = revisit_report(tmp_path, "s2_revisit_a0")
    e = rep["sources"]["s2_cal_a0"]
    assert not rep["ok"] and e["identical"] == 2 and e["missing"] == ["k5"]
    assert e["differing"]["k2"] == ["gen_exact"]
    assert e["differing"]["k3"] == ["no rv"]
    assert e["differing"]["k4"] == ["rv counts 300 evaluations, the run made 400"]


def test_stage3_registration_files_match_the_builds(monkeypatch, tmp_path):
    """Once written (scripts/s2_task_lists.py --stage 3), stage3_tasks.json holds each re-run phase's count, key hash
    and full hash as the phase builds now, and stage3_lookups.json every set's lookup SHA-256 (review m4)."""
    reg = ROOT / "data/registration/stage3_tasks.json"
    if not reg.exists():
        pytest.skip("Stage 3 task lists not written yet")
    _open_stage3(monkeypatch, tmp_path)
    want = json.loads(reg.read_text())
    assert sorted(want) == sorted(L.STAGE3_PHASES)
    for name in L.STAGE3_PHASES:
        tasks = P.PHASES[name]["build"]()
        assert want[name] == dict(n_tasks=len(tasks), sha256=L.task_list_hash(tasks),
                                  sha256_full=L.task_list_full_hash(tasks))
    lk = json.loads((ROOT / "data/registration/stage3_lookups.json").read_text())
    sets = {t["meta"]["set"] for name in L.STAGE3_PHASES for t in P.PHASES[name]["build"]()}
    assert set(lk) == sets

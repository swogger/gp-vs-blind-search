"""Big-tree reruns (D091; guardrail row 131): exact runs whose best tree has over 200 nodes recorded no expression.
REQ-ENG-22 (print option changes nothing else), REQ-RUN-32 (rerun phase), REQ-ANA-76 (comparison and label check).
Written before the code."""
import json
import random
import subprocess
import sys
from pathlib import Path

from reach import grammar as G
from reach.engines import treegp as E

ROOT = Path(__file__).resolve().parents[1]
TIMING = {"cpu_seconds", "wall_seconds"}
NEW = "best_expr_full"
BOOL_TGT = "nand(nand(x1, nand(x2, x3)), nand(x3, nand(x1, x2)))"


def _cases():
    gb = G.get("BOOL-NAND-n3")
    tb = G.parse_expr(BOOL_TGT, gb)
    gr = G.get("R-ARITH4")
    tr = G.parse_expr("/(*(x, x), -(x, 1))", gr)
    grid, val = G.sample_grid(gr), G.sample_val(gr, 20260722)
    real = dict(val_xs=val, val_target=G.evaluate(gr, tr, val), var_norm=0.023, grid_xs=grid,
                grid_target=G.evaluate(gr, tr, grid), grid_scale=4.49)
    out = []
    for seed in (1, 2):
        out += [(gb, tb, "A", seed, dict(generations=25)),
                (gb, tb, "B", seed, dict(generations=25)),
                (gb, tb, "A", seed, dict(generations=25, crossover_mode="headless")),
                (gb, tb, "A", seed, dict(generations=15, selection="lexicase")),
                (gb, tb, "A", seed, dict(generations=15, max_depth=17)),
                (gr, tr, "A", seed, dict(generations=4, **real))]
    return out


def _traced(monkeypatch, case, full, plant=False):
    """One run; returns (result, final state of every random generator the run created, global random state)."""
    g, t, regime, seed, kw = case
    made = []

    class Tracked(random.Random):
        def __init__(self, *a):
            super().__init__(*a)
            made.append(self)

    random.seed(20261028)
    with monkeypatch.context() as m:
        m.setattr(random, "Random", Tracked)
        if plant:                                      # the option consumes one random number of the run's stream
            orig = G.expr_str
            m.setattr(E.G, "expr_str", lambda tree: (made[-1].random(), orig(tree))[1])
        r = E.run_gp(g, t, regime, seed, stop_on="exact", timeout_seconds=600, full_expr=full, **kw)
    return r, [x.getstate() for x in made], random.getstate()


def _differences(a, b):
    """What differs between a run without the option (a) and with it (b), apart from timing and the new field."""
    (ra, sa, ga), (rb, sb, gb) = a, b
    out = [k for k in sorted((set(ra) | set(rb)) - TIMING - {NEW})
           if json.dumps(ra.get(k), default=str) != json.dumps(rb.get(k), default=str)]
    if sa != sb:
        out.append("generator states")
    if ga != gb:
        out.append("global random state")
    if NEW in ra:
        out.append("new field without the option")
    return out


def test_full_tree_option_changes_no_random_draw_and_no_other_field(monkeypatch):
    monkeypatch.setattr(E, "EXPR_MAX_NODES", 5)        # so that most runs exceed the limit (the real one is 200)
    assert E.EXPR_MAX_NODES == 5
    over = 0
    for case in _cases():
        a, b = _traced(monkeypatch, case, False), _traced(monkeypatch, case, True)
        assert len(a[1]) == 3                         # master, population and evolution generators all tracked
        assert _differences(a, b) == [], (case[2], case[4])
        r = b[0]
        assert r[NEW] == G.expr_str(G.parse_expr(r[NEW], case[0]))
        assert G.count_nodes(G.parse_expr(r[NEW], case[0])) == r["final_tree_size"]
        if r["final_tree_size"] > 5:
            over += 1
            assert r["best_expr"] == "" and r[NEW]
        else:
            assert r["best_expr"] == r[NEW]
    assert over >= 3


def test_full_tree_check_fires_when_the_option_consumes_a_random_number(monkeypatch):
    """Validation of the check above on a planted violation: the draw comes after the search, so every recorded
    field is unchanged and only the generator states reveal it."""
    monkeypatch.setattr(E, "EXPR_MAX_NODES", 5)
    case = _cases()[0]
    a, b = _traced(monkeypatch, case, False), _traced(monkeypatch, case, True, plant=True)
    assert _differences(a, b) == ["generator states"]
    assert a[0]["final_tree_size"] == b[0]["final_tree_size"]


def test_engine_default_limit_is_200_nodes():
    assert E.EXPR_MAX_NODES == 200


def _s2_run(tmp_path, extra=()):
    import hashlib
    p = tmp_path / "targets.json"
    p.write_text(json.dumps([dict(target_id="t0", expression="nand(x1, x2)", K_construction=3, K_proven=3,
                                  units=dict(nodes=3.0))]))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    args = ["--set", "BOOL-NAND-n3", "--targets", str(p), "--target_id", "t0", "--targets_sha256", sha,
            "--algorithm", "A", "--config", "base", "--seed", "7", "--generations", "50", "--pop_size", "100",
            "--penalty", "scaled", "--stop_on", "exact", "--seed_leaves", "off", "--timeout", "60", *extra]
    pr = subprocess.run([sys.executable, "-m", "reach.tasks.s2_run", *args], cwd=ROOT, capture_output=True, text=True)
    assert pr.returncode == 0, pr.stderr[-500:]
    return json.loads(next(l for l in pr.stdout.splitlines() if l.startswith("RESULT "))[7:])


def test_s2_run_print_tree_adds_only_the_full_expression(tmp_path):
    a, b = _s2_run(tmp_path), _s2_run(tmp_path, ["--print_tree"])
    assert NEW not in a and b[NEW] == b["best_expr"] != ""
    assert {k: v for k, v in a.items() if k not in TIMING} == {k: v for k, v in b.items() if k not in TIMING | {NEW}}


# ---------------------------------------------------------------- the rerun phase (REQ-RUN-32)

def _write_phase(root, name, tasks, rows):
    d = root / f"data/runs/{name}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(dict(phase=name, tasks=tasks)))
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _task(key, algo="A", module="reach.tasks.s2_run"):
    return dict(key=key, argv=["{python}", "-m", module, "--set", "BOOL-NAND-n3", "--seed", "1"], cwd=".",
                cpu_limit=1000, weight=1, meta=dict(block="cal_a0", set="BOOL-NAND-n3", algorithm=algo))


def _row(key, exact=True, expr="", cpu=1.0, domain="bool", state="converged"):
    return dict(key=key, block="cal_a0", set="BOOL-NAND-n3", domain=domain, algorithm="A", exact=exact,
                terminal_state=state, best_expr=expr, final_tree_size=15, gen_exact=3 if exact else None, cpu_s=cpu,
                cpu_seconds=cpu)


def test_bigtree_tasks_are_exactly_the_exact_runs_without_an_expression(tmp_path):
    import pytest
    from reach.runner import phases as P
    keys = ["big1", "big2", "small", "fail", "real"]
    _write_phase(tmp_path, "s2_cal_a0", [_task(k) for k in keys],
                 [_row("big1", cpu=3.0), _row("big2", cpu=2.0), _row("small", expr="nand(x1, x2)"),
                  _row("fail", exact=False), _row("real", cpu=9.0, domain="real")])
    tasks = P.bigtree_tasks(tmp_path, ("s2_cal_a0",))
    assert [t["key"] for t in tasks] == ["bigtree|big1", "bigtree|big2", "bigtree|real"]
    t = tasks[0]
    orig = _task("big1")
    assert t["argv"] == orig["argv"] + ["--print_tree"]
    assert t["meta"] == dict(orig["meta"], audit_of="s2_cal_a0", audit_key="big1")
    assert (t["cwd"], t["cpu_limit"]) == (orig["cwd"], orig["cpu_limit"])
    sample = P.bigtree_tasks(tmp_path, ("s2_cal_a0",), sample=1)
    assert [t["key"] for t in sample] == ["bigtree|big2", "bigtree|real"]   # cheapest per phase + cheapest real
    _write_phase(tmp_path, "s2_cal_a0", [_task("big1", module="reach.tasks.s2_operon")], [_row("big1")])
    with pytest.raises(ValueError):                                            # only tree-GP run tasks
        P.bigtree_tasks(tmp_path, ("s2_cal_a0",))


def test_bigtree_phases_on_the_calibration_records():
    from collections import Counter
    from reach.runner import phases as P
    full = P.PHASES["s2_audit_bigtrees"]["build"]()
    assert Counter(t["meta"]["audit_of"] for t in full) == {
        "s2_cal_a0": 227, "s2_cal_a16": 374, "s2_cal_alloc": 634, "s2_cal_mech": 1067, "s2_cal_robust": 1008,
        "s2_cal_pool": 170}
    assert len({t["key"] for t in full}) == 3480
    for ph in P.BIGTREE_PHASES:
        man = {t["key"]: t for t in json.loads((ROOT / f"data/runs/{ph}/manifest.json").read_text())["tasks"]}
        rows = {}                                   # last row per key, parsed only where best_expr is empty
        for line in open(ROOT / f"data/runs/{ph}/results.jsonl"):
            if '"best_expr": ""' in line:
                r = json.loads(line)
                rows[r["key"]] = r
        mine = [t for t in full if t["meta"]["audit_of"] == ph]
        for t in mine:
            k = t["meta"]["audit_key"]
            assert t["key"] == f"bigtree|{k}" and t["argv"] == man[k]["argv"] + ["--print_tree"]
            assert rows[k]["exact"] is True and not rows[k]["best_expr"]
    sample = P.PHASES["s2_audit_bigtrees_sample"]["build"]()
    assert {t["key"] for t in sample} <= {t["key"] for t in full} and 18 <= len(sample) <= 19
    assert sum(t["meta"]["set"].startswith("R-") for t in sample) == 1


# ---------------------------------------------------------------- the comparison (REQ-ANA-76)

def test_bigtree_compare_lists_every_difference_except_timing_and_bookkeeping():
    from reach.analysis.s2 import audit as A
    o = dict(_row("k"), cpu_limit=1000, queue_status="ok", exit=0, signal=0, wall_s=1, finished="a", code_hash="x",
             git="g", final_error=0.0)
    full = "nand(nand(nand(x1, x2), nand(x1, x2)), nand(nand(x1, x2), nand(x1, x2)))"
    r = dict(o, key="bigtree|k", audit_of="s2_cal_a0", audit_key="k", cpu_s=9, cpu_seconds=9, wall_s=9,
             finished="b", code_hash="y", git="h", queue_status="ok", **{NEW: full})
    assert A.bigtree_compare(o, r) == []
    assert A.bigtree_compare(o, dict(r, gen_exact=4, final_error=1e-9)) == ["final_error", "gen_exact"]
    assert A.bigtree_compare(o, dict(r, best_expr="x")) == ["best_expr"]
    assert A.bigtree_compare(o, {k: v for k, v in r.items() if k != NEW}) == ["no full expression"]
    assert A.bigtree_compare(o, dict(r, **{NEW: "nand(x1, x2)"})) == ["full expression has 3 nodes, not 15"]
    assert A.bigtree_compare(dict(o, **{NEW: full}), r) == ["full expression in the original"]


def test_bigtree_report_fires_on_planted_mismatches(tmp_path):
    from reach.analysis.s2 import audit as A
    full = "nand(nand(nand(x1, x2), nand(x1, x2)), nand(nand(x1, x2), nand(x1, x2)))"     # = nand(x1, x2), 15 nodes
    wrong = "nand(nand(nand(x1, x2), nand(x1, x2)), nand(nand(x1, x3), nand(x1, x3)))"    # 15 nodes, not the target
    d = tmp_path / "data/targets_s2/BOOL-NAND-n3"
    d.mkdir(parents=True)
    (d / "calibration.json").write_text(json.dumps([dict(target_id="t0", expression="nand(x1, x2)",
                                                         target_scale=1.0)]))
    orig = [dict(_row(k), target_id="t0") for k in ("k1", "k2")]
    _write_phase(tmp_path, "s2_cal_a0", [_task(k) for k in ("k1", "k2")], orig)

    def report(**plant):
        reruns = []
        for r in orig:
            x = dict(r, key=f"bigtree|{r['key']}", audit_of="s2_cal_a0", audit_key=r["key"], cpu_s=5.0,
                     **{NEW: full})
            x.update(plant.get(r["key"], {}))
            if "drop" not in x:
                reruns.append({k: v for k, v in x.items() if v is not None})
        _write_phase(tmp_path, "s2_audit_bigtrees", [dict(key=f"bigtree|{r['key']}", meta=dict(
            audit_of="s2_cal_a0", audit_key=r["key"])) for r in orig], reruns)
        return A.bigtree_report(tmp_path, "s2_audit_bigtrees")

    clean = report()
    assert clean["ok"] and clean["s2_cal_a0"]["identical"] == 2 and clean["s2_cal_a0"]["labels"]["exact_checked"] == 2
    bad = report(k2=dict(gen_exact=4))
    assert not bad["ok"] and bad["s2_cal_a0"]["differing"] == {"k2": ["gen_exact"]}
    bad = report(k2=dict(drop=True))
    assert not bad["ok"] and bad["s2_cal_a0"]["missing"] == ["k2"]
    bad = report(k2={NEW: None})
    assert not bad["ok"] and bad["s2_cal_a0"]["differing"] == {"k2": ["no full expression"]}
    bad = report(k2={NEW: "nand(x1, x2)"})
    assert not bad["ok"] and "k2" in bad["s2_cal_a0"]["differing"]
    bad = report(k2={NEW: wrong})
    assert not bad["ok"] and bad["s2_cal_a0"]["labels"]["exact_disputed"] == ["bigtree|k2"]

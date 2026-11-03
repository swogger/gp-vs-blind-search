"""T26 random held-out function sets (written before the code)."""
import pytest

from reach import grammar as G
from reach.enumerate import Enumerator
from reach.tasks import s2_random_sets as R


def _complete_by_enumeration(ops, consts, tmp_path):
    g = R.make_grammar("bool", ops, consts, n=3)
    e = Enumerator(g, tmp_path / "_".join(ops + [str(c) for c in consts]), log=lambda *_: None)
    e.run(40)
    return len(e.known) == 256


@pytest.mark.parametrize("ops,consts", [(["nand"], []), (["and", "or"], []), (["xor"], [1]), (["imp"], [0]),
                                        (["nimp"], [1]), (["xor", "and"], []), (["xor", "and"], [1]),
                                        (["and", "not"], []), (["or", "xnor"], []), (["rnimp", "xor"], [])])
def test_post_criterion_agrees_with_enumeration(ops, consts, tmp_path):
    assert R.is_complete(ops, consts) == _complete_by_enumeration(ops, consts, tmp_path)


def test_draw_is_deterministic_and_respects_the_rules():
    a = R.draw_candidates("bool", seed=7, n=8)
    assert a == R.draw_candidates("bool", seed=7, n=8)
    listed = {frozenset(s) for s in R.listed_symbol_sets("bool")}
    for c in a:
        assert R.is_complete(c["ops"], c["constants"]) and frozenset(c["ops"] + [f"c{x}" for x in c["constants"]]) not in listed
        assert 2 <= c["k"] <= 4
    r = R.draw_candidates("real", seed=7, n=8)
    for c in r:
        assert any(G.REAL_SYMBOL_ARITY[o] == 2 for o in c["ops"]) and 2 <= c["k"] <= 5
        assert frozenset(c["ops"]) not in {frozenset(s) for s in R.listed_symbol_sets("real")}


def test_names_are_path_safe_and_rebuild_in_a_fresh_process():
    """The enumeration task runs in a new process: G.get must rebuild a random set from its name alone."""
    import subprocess
    import sys
    from reach import grammar as G
    g = R.make_grammar("real", ["+", "/", "exp"], [])
    b = R.make_grammar("bool", ["nand", "xor"], [1], n=5)
    for name in (g.name, b.name):
        assert "/" not in name and "*" not in name and " " not in name
    code = (f"from reach import grammar as G; g = G.get({g.name!r}); b = G.get({b.name!r}); "
            "print(sorted(g.operators), b.n_inputs, b.constants)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "['+', '/', 'exp'] 5 (1,)"
    assert G.get(g.name).kind == "real" and sorted(G.get(b.name).operators) == ["nand", "xor"]


def test_seed_comes_from_the_stage1_identifier():
    import hashlib
    assert R.draw_seed("a3spg") == int.from_bytes(hashlib.sha256(b"a3spg").digest()[:8], "big")
    assert R.draw_seed("A3SPG") == R.draw_seed("a3spg")


def test_every_drawn_candidate_is_locked(tmp_path, monkeypatch):
    """Random held-out sets stay locked like the listed ones until Stage 2 (REQ-RUN-10)."""
    import json
    from reach import s2_lock as L
    p = tmp_path / "random_draw.json"
    p.write_text(json.dumps({"real": {"candidates": [{"name": "RS-add_exp"}]},
                             "bool": {"candidates": [{"name": "BOOL-RS-nand_xor_c1-n5"}]}}))
    monkeypatch.setattr(L, "RANDOM_FILE", p)
    monkeypatch.setattr(L, "STAGE2_FILE", tmp_path / "not_recorded.txt")   # before Stage 2, whatever the repo holds
    assert {"RS-add_exp", "BOOL-RS-nand_xor_c1-n5"} <= L.held_out()
    with pytest.raises(PermissionError):
        L.check_search_allowed("RS-add_exp")


def test_draw_record_is_reproducible_and_named():
    from reach.tasks import s2_random_draw as D
    a, b = D.draw("a3spg"), D.draw("A3SPG")
    assert a == b and a["real"]["seed"] + 1 == a["bool"]["seed"]
    for dom in ("real", "bool"):
        names = [c["name"] for c in a[dom]["candidates"]]
        assert len(names) == D.N_CANDIDATES == len(set(names))
        assert all(G.get(n).kind == dom for n in names)


def test_enumeration_phase_alternates_domains_in_draw_order(tmp_path, monkeypatch):
    import json
    from reach.runner import phases as P
    from reach.tasks import s2_random_draw as D
    reg = tmp_path / "data/registration"
    reg.mkdir(parents=True)
    rec = D.draw("a3spg")
    (reg / "random_draw.json").write_text(json.dumps(rec))
    monkeypatch.setattr(P, "ROOT", tmp_path)
    tasks = P.PHASES["s2_enum_random"]["build"]()
    keys = [t["key"] for t in tasks]
    assert len(tasks) == 12 and keys[0] == rec["real"]["candidates"][0]["name"]
    assert keys[1] == rec["bool"]["candidates"][0]["name"] and keys[2] == rec["real"]["candidates"][1]["name"]


def test_aliases_of_locked_sets_are_locked_too(tmp_path, monkeypatch):
    """Review finding 6: the lock compares operator signatures, not names (reordered or renamed copies are refused)."""
    import json
    from reach import s2_lock as L
    draw = tmp_path / "heldout_draw.json"
    draw.write_text(json.dumps({"real": {"held_out": ["R-ARITH"]}, "bool": {"held_out": ["BOOL-AN-n5"]}}))
    rnd = tmp_path / "random_draw.json"
    rnd.write_text(json.dumps({"real": {"candidates": [{"name": "RS-mul_eml_sin"}]},
                               "bool": {"candidates": [{"name": "BOOL-RS-rimp_c0-n5"}]}}))
    monkeypatch.setattr(L, "DRAW_FILE", draw)
    monkeypatch.setattr(L, "RANDOM_FILE", rnd)
    monkeypatch.setattr(L, "STAGE2_FILE", tmp_path / "none.txt")
    for alias in ("RS-add_sub_mul", "RS-eml_mul_sin", "BOOL-RS-and_not-n5", "BOOL-RS-c0_rimp-n5"):
        with pytest.raises(PermissionError):
            L.check_search_allowed(alias)
    for free in ("R-ARITH4", "F-EML", "BOOL-NAND-n5", "RS-add_sqrt"):
        L.check_search_allowed(free)


def test_a_disk_stop_is_an_error_not_a_ceiling(tmp_path):
    """Guardrail row 70: on 2026-10-21 a memory spike filled the disk with swap and nine enumerations stopped at size 1
    "for disk", recorded as results. A disk stop says nothing about the grammar: the task must fail (exit != 0), so
    the queue records an error and the watchdog retries it."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "reach.tasks.enumerate_task", "--grammar", "BOOL-NAND-n3", "--root",
                        str(tmp_path), "--kmax", "5", "--min_free_gb", "100000"], capture_output=True, text=True)
    assert r.returncode != 0 and "disk" in (r.stdout + r.stderr)


def test_rerun_phase_takes_disk_stops_and_kills_with_a_lower_boolean_cap(tmp_path, monkeypatch):
    import json
    from reach.runner import phases as P
    from reach.tasks import s2_random_draw as D
    reg = tmp_path / "data/registration"
    reg.mkdir(parents=True)
    rec = D.draw("a3spg")
    (reg / "random_draw.json").write_text(json.dumps(rec))
    runs = tmp_path / "data/runs/s2_enum_random"
    runs.mkdir(parents=True)
    r0, b0, b1 = rec["real"]["candidates"][0]["name"], rec["bool"]["candidates"][0]["name"], rec["bool"]["candidates"][1]["name"]
    rows = [dict(key=r0, terminal_state="budget_exhausted", stop_reason="disk", k_reached=1),
            dict(key=b0, terminal_state="error", queue_status="killed_signal_9"),
            dict(key=b1, terminal_state="budget_exhausted", stop_reason="cpu", k_reached=24)]
    (runs / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(P, "ROOT", tmp_path)
    tasks = P.PHASES["s2_enum_random_b"]["build"]()
    assert [t["key"] for t in tasks] == [r0, b0]                      # real first; valid results are not rerun
    boo = tasks[1]["argv"]
    assert boo[boo.index("--max_gb") + 1] == "6" and tasks[0]["argv"][tasks[0]["argv"].index("--max_gb") + 1] == "8"


def test_heldout_random_sets_are_the_first_four_with_ceiling_ten(tmp_path):
    """Registered rule: in draw order, the first 4 candidates per domain whose proven ceiling is at least 10."""
    import json
    from reach.tasks import s2_random_draw as D
    rec = {"real": {"candidates": [{"name": f"RS-r{i}"} for i in range(6)]},
           "bool": {"candidates": [{"name": f"BOOL-RS-b{i}-n5"} for i in range(6)]}}
    ceil = {"RS-r0": 14, "RS-r1": 9, "RS-r2": 12, "RS-r3": 20, "RS-r4": 25, "RS-r5": 30,
            "BOOL-RS-b0-n5": 21, "BOOL-RS-b1-n5": 21, "BOOL-RS-b2-n5": 23, "BOOL-RS-b3-n5": 29}
    out = D.heldout_sets(rec, ceil)
    assert out["real"] == ["RS-r0", "RS-r2", "RS-r3", "RS-r4"]                 # r1 fails the ceiling rule
    assert out["bool"] == ["BOOL-RS-b0-n5", "BOOL-RS-b1-n5", "BOOL-RS-b2-n5", "BOOL-RS-b3-n5"]
    with pytest.raises(ValueError):
        D.heldout_sets(rec, {k: v for k, v in ceil.items() if k != "RS-r2"})   # order matters: r2 not enumerated


def test_enumeration_folder_of_a_random_set():
    from reach.analysis.s2.phase1 import enum_dir
    assert enum_dir("RS-add_sqrt").parts[-2:] == ("random", "RS-add_sqrt")
    assert enum_dir("BOOL-RS-rimp_c0-n5").parts[-2] == "random"


def test_operator_probe_merges_into_its_file(tmp_path, monkeypatch):
    """Running the probe on new sets must keep the values of sets already probed (the analysis uses them)."""
    import json
    from reach.tasks import s2_opprobe as O
    p = tmp_path / "opprobe.json"
    p.write_text(json.dumps({"F-EML": {"keep": 1}}))
    O.merge_results(p, {"BOOL-NAND-n3": {"new": 2}})
    assert json.loads(p.read_text()) == {"F-EML": {"keep": 1}, "BOOL-NAND-n3": {"new": 2}}

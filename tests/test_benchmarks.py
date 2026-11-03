"""Benchmark targets (T3; D069): truth tables packed as the engine packs them; a benchmark enters a set only with its
proven minimum size."""
import pytest

from reach import grammar as G
from reach import s2_benchmarks as B
from reach.analysis.s2 import audit_labels as A
from reach.enumerate import Enumerator
from reach.targets import load_structure


def test_truth_tables_match_an_independent_evaluator():
    formulas = {"parity5": "xor(xor(xor(xor(x1, x2), x3), x4), x5)", "add2_s0": "xor(xor(x2, x4), x5)",
                "mux": "or(and(x1, x3), and(not(x1), x2))",
                "majority5": None, "cmp2_gt": "or(and(x1, not(x3)), and(xnor(x1, x3), and(x2, not(x4))))"}
    g = G.get("BOOL-RICH-n5")
    for name, expr in formulas.items():
        if expr:
            tt = G.evaluate(g, G.parse_expr(expr, g))
            assert B.truth_table(B.BOOL[name], 5) == tt, name
            assert A.same_function_bool(expr, expr, 5)
    assert bin(B.truth_table(B.BOOL["majority5"], 5)).count("1") == 16              # C(5,3)+C(5,4)+C(5,5)


def test_real_benchmarks_are_finite_and_real_on_the_grid():
    import numpy as np
    x = G.grid_sample().real
    for name, f in B.REAL.items():
        v = f(x)
        assert np.isfinite(v).all() and np.isrealobj(v), name


def test_a_benchmark_enters_only_with_its_proven_size(tmp_path):
    g = G.get("BOOL-AONX-n3")
    Enumerator(g, tmp_path / g.name, log=lambda *_: None).run(9)
    e = load_structure(g, tmp_path / g.name)
    specs = {"parity3": lambda x: x[0] ^ x[1] ^ x[2], "maj3": lambda x: int(sum(x) >= 2)}
    ts, log = B.bench_targets(g, e, specs)
    assert {t["benchmark"] for t in ts} == {"parity3", "maj3"}, log
    par = next(t for t in ts if t["benchmark"] == "parity3")
    assert par["K_proven"] == 5 and G.evaluate(g, G.parse_expr(par["expression"], g)) == B.truth_table(specs["parity3"], 3)
    tiny = load_structure(g, tmp_path / g.name)
    tiny.keys = {k: v for k, v in tiny.keys.items() if k <= 3}
    _, log2 = B.bench_targets(g, tiny, {"parity3": specs["parity3"]})
    assert log2["parity3"] == "not within the ceiling"


def test_bench_phase_covers_the_twelve_heldout_sets():
    from reach.runner.phases import PHASES, heldout_sets
    hs = heldout_sets()
    assert len(hs) == 12 and {"R-ARITH", "R-ELEM6", "BOOL-AN-n5", "BOOL-AX1-n5"} <= set(hs)
    assert [t["key"] for t in PHASES["s2_targets_bench"]["build"]()] == hs


# ------------------------------------------------------------------ guardrail row 126: D050 noise correction
def test_real_benchmark_units_carry_the_noise_correction(tmp_path):
    """Row 126: real benchmark sizes skipped the D050 float-noise correction that the target generator applies, so
    bits_sem equalled bits_sem_uncorrected (inventory 5b). The rows are required for a real set."""
    import numpy as np
    from reach.units import SizeUnits
    g = G.get("R-ARITH")
    Enumerator(g, tmp_path / g.name, log=lambda *_: None).run(7)
    e = load_structure(g, tmp_path / g.name)
    specs = {"sq": lambda x: x * x + x}
    with pytest.raises(ValueError, match="D050"):
        B.bench_targets(g, e, specs)
    noise = [dict(K=3, noise=1, U=next(st["U"] for st in e.stats if st["K"] == 3))]
    ts, log = B.bench_targets(g, e, specs, noise_rows=noise)
    (t,) = ts
    assert t["units"] == SizeUnits(g, e.stats, noise_rows=noise).all(t["K_proven"])
    assert t["units"]["bits_sem"] < t["units"]["bits_sem_uncorrected"]
    gb = G.get("BOOL-AONX-n3")                                               # Boolean sets have no noise rows
    Enumerator(gb, tmp_path / gb.name, log=lambda *_: None).run(5)
    eb = load_structure(gb, tmp_path / gb.name)
    with pytest.raises(ValueError, match="D050"):
        B.bench_targets(gb, eb, {"p": lambda x: x[0] ^ x[1]}, noise_rows=noise)
    assert np.isfinite(B.bench_targets(gb, eb, {"p": lambda x: x[0] ^ x[1]})[0][0]["units"]["bits_sem"])


def test_committed_real_benchmarks_carry_the_noise_correction():
    """The committed real bench files record the D050 rows they used (bench_log.json) and every target's units are
    those of the set's size units with those rows (counts.json is committed; level files are not needed)."""
    import json
    from reach.analysis.s2.phase1 import enum_dir
    from reach.paths import ROOT
    from reach.runner.phases import heldout_sets
    from reach.units import SizeUnits
    real = [g for g in heldout_sets() if G.get(g).kind == "real"]
    assert len(real) == 6
    for g in real:
        d = ROOT / f"data/targets_s2/{g}"
        log = json.loads((d / "bench_log.json").read_text())
        assert isinstance(log.get("noise_rows"), list), g
        su = SizeUnits(G.get(g), json.loads((enum_dir(g) / "counts.json").read_text())["levels"],
                       noise_rows=log["noise_rows"])
        for t in json.loads((d / "bench.json").read_text()):
            assert t["units"] == pytest.approx(su.all(t["K_proven"])), (g, t["target_id"])


# ------------------------------------------------------------------ D091: SRBench and EML-GP one-variable targets
def _ref(name):
    import json
    from reach.paths import ROOT
    return ROOT / "data/reference" / name, json.loads((ROOT / "data/reference" / name / "source.json").read_text())


def test_reference_lists_match_their_recorded_checksums():
    import hashlib
    d, src = _ref("srbench_groundtruth")
    assert hashlib.sha256((d / "equations.csv").read_bytes()).hexdigest() == src["equations_csv_sha256"]
    assert hashlib.sha256((d / "feynman_I_6_2a.yaml").read_bytes()).hexdigest() == \
        src["pmlb"]["metadata_sha256"]["feynman_I_6_2a"]
    assert len(src["pmlb"]["metadata_sha256"]) == 130 and src["pmlb"]["commit"] and src["srbench"]["commit"]
    _, eml = _ref("eml_gp_targets")
    assert len(eml["targets"]) == 12 and len(eml["file_sha256"]) == 64 and eml["commit"]


def test_every_srbench_and_eml_gp_item_is_included_or_excluded_with_a_reason():
    """Each one-variable item is a candidate (tested against D069's rule per set by s2_bench); each other item is
    excluded with its reason; no item is silently missing."""
    c = B.candidates()
    names = {r["item"] for r in c}
    assert len([r for r in c if r["source"] == "SRBench"]) == 130 and len([r for r in c if r["source"] == "EML-GP"]) == 12
    assert len(names) == 142
    for r in c:
        if r["n_variables"] == 1:
            assert r["benchmark"] in B.REAL and r["excluded"] is None, r
        else:
            assert r["benchmark"] is None and "variables" in r["excluded"], r
    assert {r["item"] for r in c if r["benchmark"]} == {"feynman_I_6_2a"} | {f"EML-GP {k}" for k in
                                                         ("1.1", "1.2", "1.4", "2.1", "3.1", "3.2", "3.3", "3.4", "3.5", "4.1")}


def test_srbench_and_eml_gp_functions_equal_their_sources():
    """Our functions against the source formulas, read from the reference files (the PMLB equation, EML-GP's printed
    names), evaluated on the grid with a different evaluator."""
    import csv
    import re
    import numpy as np
    x = G.grid_sample().real
    ns = dict(exp=np.exp, sqrt=np.sqrt, sin=np.sin, cos=np.cos, sinh=np.sinh, arctan=np.arctan, ln=np.log, pi=np.pi)
    d, _ = _ref("srbench_groundtruth")
    row = next(r for r in csv.DictReader(open(d / "equations.csv")) if r["dataset"] == "feynman_I_6_2a")
    rhs = row["equation"].split("=", 1)[1]
    assert np.allclose(eval(rhs, dict(ns, theta=x)), B.REAL["feynman_I_6_2a"](x), rtol=1e-14, atol=0)  # noqa: S307
    _, eml = _ref("eml_gp_targets")
    for t in eml["targets"]:
        if t["variables"] != ["x"]:
            continue
        f = re.sub(r"\[.*\]", "", t["name"].split("=", 1)[1]).replace("^", "**")
        f = re.sub(r"(\d)x", r"\1*x", f)
        assert np.allclose(eval(f, dict(ns, x=x)), B.REAL[B.eml_name(t["key"])](x), rtol=1e-14, atol=0), t  # noqa: S307


def test_new_benchmarks_are_appended_after_the_frozen_ones():
    """Existing bench files keep their entries and order: new specs come after K3 (only additions can differ)."""
    keys = list(B.REAL)
    assert keys[:10] == ["N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "K2", "K3"]
    assert set(keys[10:]) == {"feynman_I_6_2a"} | {B.eml_name(k) for k in
                                                   ("1.1", "1.2", "1.4", "2.1", "3.1", "3.2", "3.3", "3.4", "3.5", "4.1")}
    assert list(B.BOOL) == ["parity5", "majority5", "add2_s0", "add2_s1", "add2_cout", "cmp2_gt", "mux"]


# ------------------------------------------------------------------ D091: regeneration through the queue (row 129)
@pytest.fixture(autouse=True)
def _unregistered(tmp_path, monkeypatch):
    """Guardrail row 133: the regeneration tests read the repository's own Stage 2 lock files, so writing
    stage2_tasks.json (a registration step, D092) made them fail although the code was right. Each test starts from
    an unregistered state in its own directory; the refusal test sets its own lock file."""
    from reach import s2_lock
    monkeypatch.setattr(s2_lock, "TASKS2_FILE", tmp_path / "no_stage2_tasks.json")
    monkeypatch.setattr(s2_lock, "STAGE2_FILE", tmp_path / "no_stage2_registration.txt")


def _frozen(tmp_path, ts, log):
    from reach.s2_targets import write_frozen
    write_frozen(tmp_path, "bench", ts, log)
    return {f: (tmp_path / f).read_bytes() for f in ("bench.json", "bench_log.json", "bench.sha256")}


def test_regeneration_reproduces_or_only_appends(tmp_path):
    old = [dict(target_id="S_bench_N1", benchmark="N1", units=dict(bits_sem=1.5))]
    log = dict(grammar="S", benchmarks={"N1": "proven K=5"}, noise_rows=[dict(K=3, noise=1, U=7)])
    before = _frozen(tmp_path, old, log)
    r = B.regenerate(tmp_path, old, log)
    assert r["reproduced"] and r["added"] == [] and r["bench_sha256"] == before["bench.sha256"].decode().split()[0]
    assert {f: (tmp_path / f).read_bytes() for f in before} == before                  # nothing rewritten
    new = old + [dict(target_id="S_bench_eml_1_1", benchmark="eml_1_1", units=dict(bits_sem=2.0))]
    log2 = dict(log, benchmarks=dict(log["benchmarks"], eml_1_1="proven K=3", feynman_I_6_2a="not within the ceiling"))
    r = B.regenerate(tmp_path, new, log2)
    assert not r["reproduced"] and r["added"] == ["eml_1_1"] and r["old_bench_sha256"] != r["bench_sha256"]
    import hashlib
    import json
    raw = (tmp_path / "bench.json").read_bytes()
    assert json.loads(raw) == new and hashlib.sha256(raw).hexdigest() == r["bench_sha256"]
    assert (tmp_path / "bench.sha256").read_text().split()[0] == r["bench_sha256"]


def test_regeneration_refuses_any_change_to_existing_entries(tmp_path):
    import pytest as _pt
    old = [dict(target_id="S_bench_N1", benchmark="N1", units=dict(bits_sem=1.5))]
    log = dict(grammar="S", benchmarks={"N1": "proven K=5"}, noise_rows=[dict(K=3, noise=1, U=7)])
    before = _frozen(tmp_path, old, log)
    for ts, lg, why in [([dict(old[0], units=dict(bits_sem=1.6))], log, "entries"),
                        ([], dict(log, benchmarks={}), "entries"),
                        (old, dict(log, noise_rows=[]), "noise"),
                        (old, dict(log, benchmarks={"N1": "not within the ceiling"}), "log")]:
        with _pt.raises(ValueError, match=why):
            B.regenerate(tmp_path, ts, lg)
        assert {f: (tmp_path / f).read_bytes() for f in before} == before


def test_regeneration_is_refused_once_stage2_is_registered(tmp_path, monkeypatch):
    import pytest as _pt
    from reach import s2_lock
    reg = tmp_path / "stage2_tasks.json"
    reg.write_text("{}")
    monkeypatch.setattr(s2_lock, "TASKS2_FILE", reg)
    with _pt.raises(PermissionError):
        B.regenerate(tmp_path, [], dict(benchmarks={}))


def test_bench_regeneration_phase_covers_the_twelve_heldout_sets():
    from reach.runner.phases import PHASES, heldout_sets
    tasks = PHASES["s2_targets_bench2"]["build"]()
    assert [t["key"] for t in tasks] == heldout_sets()
    assert all(t["argv"][-3:] == ["--grammar", t["key"], "--regenerate"] for t in tasks)


def _bench_tree(tmp_path, content=b"[]", records=(), phase="s2_targets_bench2"):
    import hashlib
    import json
    d = tmp_path / "data/targets_s2/R-X"
    d.mkdir(parents=True)
    (d / "bench.json").write_bytes(content)
    (d / "bench.sha256").write_text(hashlib.sha256(content).hexdigest() + "  bench.json\n")
    r = tmp_path / f"data/runs/{phase}"
    r.mkdir(parents=True)
    (r / "results.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records))
    return hashlib.sha256(content).hexdigest()


def test_bench_record_check_fires_on_planted_violations(tmp_path):
    """Row 129: every committed bench file must have a completed queue record whose recorded SHA-256 is the file's."""
    import hashlib
    sha = hashlib.sha256(b"[]").hexdigest()
    ok = dict(key="R-X", grammar="R-X", terminal_state="completed", queue_status="ok", bench_sha256=sha)
    for recs, bad in [([ok], False),
                      ([], True),                                                   # no record at all
                      ([dict(ok, bench_sha256="0" * 64)], True),                    # file changed after the run
                      ([dict(ok, terminal_state="error", queue_status="error")], True),
                      ([{k: v for k, v in ok.items() if k != "bench_sha256"}], True)]:   # old record without a hash
        import shutil
        shutil.rmtree(tmp_path / "data", ignore_errors=True)
        _bench_tree(tmp_path, records=recs)
        assert bool(B.bench_record_problems(tmp_path)) == bad, recs
    shutil.rmtree(tmp_path / "data")
    _bench_tree(tmp_path, records=[ok])
    (tmp_path / "data/targets_s2/R-X/bench.sha256").write_text("0" * 64 + "  bench.json\n")
    assert B.bench_record_problems(tmp_path)                                         # sidecar disagrees


def test_committed_bench_files_have_a_queue_record():
    """Row 129: the 6 real bench files were regenerated outside the queue in 9a3f56d, so no queue record had their
    content (a strict-xfail known defect until the s2_targets_bench2 records were committed, D091)."""
    from reach.paths import ROOT
    assert B.bench_record_problems(ROOT) == []


def test_bench_task_result_line_is_built_from_the_regeneration_record(tmp_path):
    """Row 130: s2_bench --regenerate passed bench_sha256 twice to the RESULT dict (explicitly and in the comparison),
    so every task of the first s2_targets_bench2 run crashed with a TypeError after writing its files (2026-10-28)."""
    import hashlib
    ts = [dict(target_id="S_bench_N1", benchmark="N1")]
    log = dict(grammar="S", benchmarks={"N1": "proven K=5"}, noise_rows=None)
    cmp = B.regenerate(tmp_path, ts, log)
    rec = B.result_record("S", ts, log["benchmarks"], tmp_path / "bench.json", cmp, root=tmp_path)
    assert rec["bench_sha256"] == hashlib.sha256((tmp_path / "bench.json").read_bytes()).hexdigest()
    assert rec["terminal_state"] == "completed" and rec["targets"] == 1 and rec["added"] == []          # new file
    assert B.result_record("S", ts, {}, tmp_path / "bench.json", {}, root=tmp_path)["bench_sha256"] == rec["bench_sha256"]

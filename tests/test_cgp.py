"""Cartesian GP engine and task (D072; docs/design/cgp.md), written before the code: REQ-ENG-16 to 21, REQ-RUN-22 to 24.
The Rust-Python semantics cross-check is in test_cgp_semantics.py."""
import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from reach import cgp
from reach import grammar as G

ROOT = Path(__file__).resolve().parents[1]
TIMING = {"cpu_seconds", "wall_seconds", "child_cpu_seconds", "rust_cpu_seconds"}
S2_RUN_FIELDS = {"set", "domain", "algorithm", "config", "seed", "target_id", "K_proven", "K_construction", "units",
                 "budget_E", "generations", "pop_size", "evals_per_gen", "penalty", "stop_on", "seed_leaves", "exact",
                 "converged", "gen_exact", "gen_tol", "gens_reached", "terminal_state", "final_error",
                 "final_tree_size", "cpu_seconds", "wall_seconds", "best_expr", "targets_sha256"}
CGP_FIELDS = {"n_nodes", "levels_back", "mutation", "strategy", "evals_exact", "eval_index_exact", "evals_reached",
              "claim_exact", "claim_rejected", "missed_exact", "active_nodes", "dag_size", "target_dag_nodes",
              "binary_sha256", "rust_source_sha256", "child_cpu_seconds", "best_graph"}


# ------------------------------------------------------------------ helpers
def _targets(tmp_path, expr, **extra):
    t = [dict(target_id="t0", expression=expr, K_construction=3, K_proven=3, units=dict(nodes=3.0), **extra)]
    p = tmp_path / "targets.json"
    p.write_text(json.dumps(t))
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def _args(path, sha, s="BOOL-NAND-n3", **over):
    a = dict(set=s, targets=str(path), target_id="t0", targets_sha256=sha, seed="7", n_nodes="100",
             levels_back="unrestricted", mutation="single", strategy="1+4", budget="1", penalty="scaled",
             stop_on="exact", timeout="60")
    a.update(over)
    return [x for k, v in a.items() if v is not None for x in (f"--{k}", v)]


def _env(**extra):
    """The task's environment: this checkout's src first (the shared editable .venv points at another worktree)."""
    return dict(os.environ, PYTHONPATH=str(ROOT / "src"), **extra)


def _task(args, env=None):
    return subprocess.run([sys.executable, "-m", "reach.tasks.s2_cgp", *args], cwd=ROOT, capture_output=True,
                          text=True, env=env or _env())


def test_task_subprocesses_import_this_checkout():
    """Guardrail row 88 (independent review, 2026-10-26): the task subprocesses ran without PYTHONPATH, so in the dev
    worktree they imported the main checkout through the shared editable .venv (the hook's PYTHONPATH hid it)."""
    p = subprocess.run([sys.executable, "-c", "import reach.cgp as m; print(m.__file__)"], cwd=ROOT,
                       capture_output=True, text=True, env=_env())
    assert p.returncode == 0, p.stderr[-500:]
    assert Path(p.stdout.strip()).resolve() == (ROOT / "src/reach/cgp.py").resolve()


def _row(p):
    assert p.returncode == 0, p.stderr[-800:]
    line = next(l for l in p.stdout.splitlines() if l.startswith("RESULT "))
    return json.loads(line[7:])


def _engine(exe, g, expr, seed=1, n_nodes=100, max_evals=20_000, cpu_limit=60.0, **ctx_kw):
    """Run the binary directly (no task): returns the engine's report and the context."""
    tree = G.parse_expr(expr, g)
    ctx = cgp.context(g, tree, **ctx_kw)
    status, _, out, err = cgp.run_engine(exe, "run", cgp.run_spec(g, ctx, seed=seed, n_nodes=n_nodes,
                                                                  max_evals=max_evals, cpu_limit=cpu_limit))
    assert status == 0, err[-800:]
    return json.loads(out), ctx


def _real_kw(g, expr):
    v = G.evaluate(g, G.parse_expr(expr, g), G.sample_grid(g))
    return dict(var_target=float(v.real.var()), target_scale=float(abs(v).max()))


# ------------------------------------------------------------------ engine (REQ-ENG-16 to 21)
def test_cargo_unit_tests_pass(cgp_exe):
    """The crate's own tests (RNG reference outputs, single active mutation, guards, truth tables)."""
    p = subprocess.run([cgp.cargo(), "test", "--release", "--locked", "--offline", "--quiet"], cwd=cgp.CRATE,
                       capture_output=True, text=True)
    assert p.returncode == 0, (p.stdout + p.stderr)[-2000:]


def test_no_size_term_in_cgp_selection(cgp_exe):
    """Guardrail row 79 (REQ-RULE-01): with operators {and, or} on one input every program computes x1, so every
    offspring has the parent's fitness. Without a size term every generation replaces the parent, whatever the sizes;
    any size penalty or size tie-break would refuse the larger offspring."""
    g = G.Grammar("BOOL-AO-n1", "bool", (G.BOOL_OPS["and"], G.BOOL_OPS["or"]), ("x1",), (), 1)
    ctx = cgp.context(g, G.parse_expr("x1", g))
    ctx["tt"] = g.mask ^ ctx["tt"]                     # target: not x1, which no program computes (never stops)
    status, _, out, err = cgp.run_engine(cgp_exe, "run", cgp.run_spec(g, ctx, seed=3, n_nodes=60, max_evals=8_001,
                                                                      cpu_limit=60))
    assert status == 0, err
    r = json.loads(out)
    assert r["terminal_state"] == "budget_exhausted" and r["gens_reached"] == 2_000
    assert r["replacements"] == r["neutral_replacements"] == r["gens_reached"]
    assert len(r["parent"]["nodes"]) > 0                  # the parent drifted away from the bare input


def neutral_drift_sizes(exe, seeds=range(1, 51), n_nodes=60, gens=2_000):
    """Mean active nodes of the initial parent and of the parent after `gens` generations of pure neutral drift, over
    `seeds` (the landscape of test_no_size_term_in_cgp_selection: every offspring ties)."""
    g = G.Grammar("BOOL-AO-n1", "bool", (G.BOOL_OPS["and"], G.BOOL_OPS["or"]), ("x1",), (), 1)
    ctx = cgp.context(g, G.parse_expr("x1", g))
    ctx["tt"] = g.mask ^ ctx["tt"]
    out = []
    for max_evals in (1, 1 + 4 * gens):
        sz = []
        for seed in seeds:
            st, _, o, err = cgp.run_engine(exe, "run", cgp.run_spec(g, ctx, seed=seed, n_nodes=n_nodes,
                                                                    max_evals=max_evals, cpu_limit=60))
            assert st == 0, err
            sz.append(len(json.loads(o)["parent"]["nodes"]))
        out.append(sum(sz) / len(sz))
    return tuple(out)


def test_no_size_tie_break_among_offspring(cgp_exe):
    """Guardrail row 84 (REQ-RULE-01; found by the independent review, 2026-10-26): row 79's test cannot see a size
    tie-break AMONG tied offspring, since some offspring still replaces the parent every generation. On the neutral
    landscape every offspring ties, so the tie-break alone picks the next parent. A uniform pick is size-blind: the
    parent's active size stays near that of a random genome (the initial parents of the same 50 seeds; measured 11.1
    initially, 13.8 after 2,000 generations; single active mutation is not symmetric between genomes, hence the small
    rise). A size tie-break drives it to an extreme: smallest-first 2.1, largest-first 48.0 (planted crates). The test
    requires the final mean within a factor of 2 of the initial mean."""
    initial, final = neutral_drift_sizes(cgp_exe)
    assert initial > 5                                    # precondition: random genomes are not trivially small
    assert 0.5 <= final / initial <= 2.0, (initial, final)


def test_evaluation_count_and_budget_stop(cgp_exe):
    """C8: 1 for the initial parent, 4 (Boolean) or 5 (real) per generation; no generation exceeds the budget."""
    g = G.Grammar("BOOL-AO-n1", "bool", (G.BOOL_OPS["and"], G.BOOL_OPS["or"]), ("x1",), (), 1)
    ctx = cgp.context(g, G.parse_expr("x1", g))
    ctx["tt"] = g.mask ^ ctx["tt"]                     # unreachable target: the run uses its whole budget
    for max_evals, gens in ((1_001, 250), (1_004, 250), (1_005, 251)):
        spec = cgp.run_spec(g, ctx, seed=5, n_nodes=50, max_evals=max_evals, cpu_limit=60)
        r = json.loads(cgp.run_engine(cgp_exe, "run", spec)[2])
        assert r["terminal_state"] == "budget_exhausted"
        assert (r["gens_reached"], r["evals_reached"], r["evals_per_gen"]) == (gens, 1 + 4 * gens, 4)
    gr = G.get("F-EML")
    r, _ = _engine(cgp_exe, gr, "eml(eml(x, x), eml(1, x))", seed=5, max_evals=1_003,
                   **_real_kw(gr, "eml(eml(x, x), eml(1, x))"))
    assert r["terminal_state"] != "converged"
    assert (r["gens_reached"], r["evals_reached"], r["evals_per_gen"]) == (200, 1 + 5 * 200, 5)


def _solved(exe, g, expr, seeds, **kw):
    for seed in seeds:
        r, ctx = _engine(exe, g, expr, seed=seed, **kw)
        if r["claim_exact"] and r["evals_exact"] > 1:
            return seed, r
    pytest.fail(f"precondition: no seed solves {expr} in {g.name}")


@pytest.mark.parametrize("name,expr", [("BOOL-AONX-n5", "xor(and(x1, x2), or(x3, not(x4)))"),
                                       ("R-ELEM4", "+(x, *(x, x))")])
def test_prefix_property_cgp(cgp_exe, name, expr):
    """REQ-ENG-03 for CGP: the run to a budget is the prefix of a longer run with the same seed, so the first exact
    match is found at the same evaluation by every budget that contains it, and missed by every shorter one."""
    g = G.get(name)
    kw = _real_kw(g, expr) if g.kind == "real" else {}
    seed, full = _solved(cgp_exe, g, expr, range(1, 30), max_evals=40_000, **kw)
    e, per = full["evals_exact"], full["evals_per_gen"]
    assert e == 1 + per * full["gen_exact"]                                  # whole generations are counted
    assert e - per < full["eval_index_exact"] <= e                           # the hitting offspring is in the last one
    short, _ = _engine(cgp_exe, g, expr, seed=seed, max_evals=e - 1, **kw)
    just, _ = _engine(cgp_exe, g, expr, seed=seed, max_evals=e, **kw)
    assert not short["claim_exact"] and short["evals_reached"] == e - per
    assert just["claim_exact"] and just["evals_exact"] == e and just["best"] == full["best"]


def test_same_seed_same_record_different_seed_differs(tmp_path, cgp_exe):
    """REQ-ENG-20: the record is a function of the seed and the inputs (timing fields aside)."""
    path, sha = _targets(tmp_path, "xor(and(x1, x2), or(x3, not(x4)))")
    a, b, c = (_row(_task(_args(path, sha, s="BOOL-AONX-n5", seed=s, budget="0.2"))) for s in ("11", "11", "12"))
    drop = lambda r: {k: v for k, v in r.items() if k not in TIMING}                              # noqa: E731
    assert drop(a) == drop(b)
    assert drop(a) != drop(c)
    assert (a["evals_exact"], a["best_graph"]) != (c["evals_exact"], c["best_graph"])


def test_rejected_claim_is_not_exact(cgp_exe):
    """REQ-ENG-19 (guardrail row 76): the engine is given a different target than Python's (a planted disagreement);
    its exact claim must be recorded as not exact, flagged, with the run counted as stopped early."""
    g = G.get("BOOL-NAND-n5")
    r, ctx = _engine(cgp_exe, g, "nand(x1, x2)", seed=2)
    assert r["claim_exact"]
    rec = cgp.outcome(g, ctx, r)
    assert rec["exact"] and not rec["claim_rejected"] and rec["evals_exact"] == r["evals_exact"]
    python_ctx = cgp.context(g, G.parse_expr("nand(x1, x3)", g))
    rec = cgp.outcome(g, python_ctx, r)
    assert not rec["exact"] and rec["claim_rejected"] and rec["evals_exact"] is None
    assert rec["terminal_state"] == "early_stopped" and rec["gen_exact"] is None
    # a missed exact match is visible: the engine reports no claim, Python finds the parent exact
    rec = cgp.outcome(g, ctx, dict(r, claim_exact=False, terminal_state="budget_exhausted"))
    assert rec["missed_exact"] and not rec["exact"]


def test_dag_and_active_node_measures(cgp_exe):
    """REQ-ENG-21 (guardrail row 80): XOR from NAND as a tree has 9 nodes and 6 distinct subtrees; the merged DAG has
    exactly those 6 nodes and computes the tree's function (so dag_nodes bounds the minimum DAG from above)."""
    g = G.get("BOOL-NAND-n3")
    tree = G.parse_expr("nand(nand(x1, nand(x1, x2)), nand(x2, nand(x1, x2)))", g)
    prog = cgp.tree_to_program(g, tree)
    s = cgp.sizes(prog)
    assert (s["active_nodes"], s["dag_size"], s["expanded_size"]) == (4, 6, G.count_nodes(tree))
    assert cgp.evaluate(g, prog) == G.evaluate(g, tree)
    assert cgp.expression(g, prog) == G.expr_str(tree)
    assert cgp.expression(g, prog, limit=8) == ""
    # a found graph: active nodes as listed, expansion consistent with the printed expression
    r, ctx = _engine(cgp_exe, G.get("BOOL-AONX-n5"), "xor(and(x1, x2), or(x3, not(x4)))", seed=4)
    rec = cgp.outcome(G.get("BOOL-AONX-n5"), ctx, r)
    assert rec["active_nodes"] == len(r["best"]["nodes"]) and rec["dag_size"] > rec["active_nodes"]
    if rec["best_expr"]:
        t = G.parse_expr(rec["best_expr"], G.get("BOOL-AONX-n5"))
        assert G.count_nodes(t) == rec["final_tree_size"]
        assert G.evaluate(G.get("BOOL-AONX-n5"), t) == cgp.evaluate(G.get("BOOL-AONX-n5"), r["best"])


# ------------------------------------------------------------------ task (REQ-RUN-22 to 24)
def test_cgp_record_has_the_s2_run_fields(tmp_path, cgp_exe):
    path, sha = _targets(tmp_path, "nand(x1, nand(x2, x3))")
    r = _row(_task(_args(path, sha)))
    assert S2_RUN_FIELDS | CGP_FIELDS <= set(r)
    assert r["algorithm"] == "CGP" and r["exact"] and r["terminal_state"] == "converged"
    assert r["rust_source_sha256"] == cgp.RUST_SOURCE_SHA256 and r["binary_sha256"] == cgp.file_sha256(cgp_exe)
    rpath, rsha = _targets(tmp_path, "+(x, 1)", var_target=1 / 12, target_scale=4.0)
    r = _row(_task(_args(rpath, rsha, s="R-ARITH4")))
    assert r["exact"] and r["evals_per_gen"] == 5 and r["domain"] == "real"


def test_cgp_task_requires_every_setting(tmp_path, cgp_exe):
    """C6: no defaults for scientific settings."""
    path, sha = _targets(tmp_path, "nand(x1, x2)")
    for missing in ("seed", "n_nodes", "levels_back", "mutation", "strategy", "budget", "penalty", "stop_on",
                    "timeout", "targets_sha256"):
        p = _task(_args(path, sha, **{missing: None}))
        assert p.returncode != 0 and f"--{missing}" in p.stderr, missing
    p = _task(_args(path, "0" * 64))
    assert p.returncode != 0 and "checksum" in p.stderr


def test_cgp_held_out_set_is_refused(tmp_path, cgp_exe):
    """REQ-RUN-10: R-ARITH and BOOL-AN-n5 are held out until Stage 2."""
    from reach.s2_lock import STAGE2_FILE, held_out
    if "R-ARITH" not in held_out() or STAGE2_FILE.exists():
        pytest.skip("draw not present or Stage 2 already filed")
    path, sha = _targets(tmp_path, "+(x, 1)", var_target=0.08, target_scale=4.0)
    p = _task(_args(path, sha, s="R-ARITH"))
    assert p.returncode != 0 and "held-out" in p.stderr
    path, sha = _targets(tmp_path, "and(x1, x2)")
    p = _task(_args(path, sha, s="BOOL-AN-n5"))
    assert p.returncode != 0 and "held-out" in p.stderr


def test_rust_source_hash_is_current(cgp_exe):
    """REQ-RUN-23: the Python code names the hash of the Rust source in this tree, and the binary embeds it. After
    any change to rust/cgp, update RUST_SOURCE_SHA256 (scripts/cgp_rehash.py), which changes the pinned code hash."""
    assert cgp.source_hash() == cgp.RUST_SOURCE_SHA256
    assert cgp.embedded_hash(cgp_exe) == cgp.RUST_SOURCE_SHA256
    assert cgp_exe == cgp.binary()


def test_cgp_constants_are_tree_gp_constants():
    """Guardrail row 86 (independent review, 2026-10-26): reach.cgp copied tree GP's batch size, convergence threshold,
    validation seed and E0 without a test. They are imported now; this pins them to the values tree GP runs with
    (s2_run parses arguments when imported, so its VAL_SEED is read from its source)."""
    import ast
    from reach.analysis.s2 import load
    from reach.engines import treegp
    from reach.tasks.s2_configs import CONFIGS
    src = (ROOT / "src/reach/tasks/s2_run.py").read_text()
    s2_val_seed = next(eval(compile(ast.Expression(n.value), "s2_run", "eval"), {}) for n in ast.parse(src).body
                       if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "VAL_SEED" for t in n.targets))
    assert cgp.VAL_SEED == s2_val_seed == 20260721 + 1
    assert (cgp.BATCH, cgp.TOL) == (treegp.BATCH_SIZE, treegp.CONV_THRESHOLD) == (75, 1e-6)
    assert cgp.E0 == load.E0 == CONFIGS["base"]["pop_size"] * CONFIGS["base"]["generations_per_E0"] == 20_000


def test_rewritten_constant_is_not_hidden_by_cached_bytecode(tmp_path):
    """Guardrail row 82 (found while validating this file's tests on planted violations): rewriting the source-hash
    constant keeps the file's size, and within the same second Python kept using the cached bytecode (kept under
    sys.pycache_prefix by the system Python), so the old constant won. drop_bytecode must remove that cache."""
    mod = tmp_path / "pinmod.py"
    mod.write_text('H = "' + "a" * 64 + '"\n')
    st = mod.stat()
    run = lambda: subprocess.run([sys.executable, "-c", "import pinmod; print(pinmod.H[:4])"], cwd=tmp_path,  # noqa
                                 capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(tmp_path))).stdout
    assert run().strip() == "aaaa"                               # compiles and caches the bytecode
    mod.write_text('H = "' + "b" * 64 + '"\n')
    os.utime(mod, ns=(st.st_atime_ns, st.st_mtime_ns))           # same size, same second: the cache looks valid
    assert run().strip() == "aaaa"                               # the hazard itself (precondition of this test)
    cgp.drop_bytecode(mod)
    assert run().strip() == "bbbb"


def test_binary_from_other_source_is_refused(tmp_path, cgp_exe):
    """Guardrail row 77: a binary built from changed source embeds another hash and is refused in place of the pinned
    one, by the library and by the task."""
    crate = tmp_path / "crate"
    shutil.copytree(cgp.CRATE, crate, ignore=shutil.ignore_patterns("target"))
    with open(crate / "src" / "main.rs", "a") as fh:
        fh.write("\n// a planted change\n")
    other = cgp.build(crate=crate, store=tmp_path / "store", profile="dev")
    assert cgp.embedded_hash(other) == cgp.source_hash(crate) != cgp.RUST_SOURCE_SHA256
    with pytest.raises(cgp.BinaryError, match="no CGP binary"):
        cgp.binary(store=tmp_path / "store")
    fake_root = tmp_path / "root"
    planted = fake_root / "rust" / "bin" / cgp.RUST_SOURCE_SHA256 / "cgp"
    planted.parent.mkdir(parents=True)
    shutil.copy2(other, planted)
    with pytest.raises(cgp.BinaryError, match="built from source"):
        cgp.binary(store=fake_root / "rust" / "bin")
    path, sha = _targets(tmp_path, "nand(x1, x2)")
    p = _task(_args(path, sha), env=_env(REACH_ROOT=str(fake_root)))
    assert p.returncode != 0 and "built from source" in p.stderr


def test_cpu_soft_limit_records_the_evaluations_reached(tmp_path, cgp_exe):
    """REQ-RUN-15 for CGP: a run stopped by its soft CPU limit records the evaluations it reached."""
    path, sha = _targets(tmp_path, "eml(eml(eml(1, x), eml(x, 1)), eml(eml(x, x), 1))", var_target=1.0, target_scale=1.0)
    r = _row(_task(_args(path, sha, s="F-EML", timeout="0", budget="16")))
    assert r["terminal_state"] == "timeout" and r["evals_reached"] == 1 and r["evals_exact"] is None
    r = _row(_task(_args(path, sha, s="F-EML", timeout="0.3", budget="16")))
    assert r["terminal_state"] == "timeout" and 1 < r["evals_reached"] < 16 * cgp.E0
    assert r["rust_cpu_seconds"] >= 0.3 and r["cpu_seconds"] >= r["child_cpu_seconds"] >= r["rust_cpu_seconds"]


def test_engine_panic_message_is_kept(tmp_path, cgp_exe):
    """Guardrail row 85 (independent review, 2026-10-26): the crate aborts on a panic (SIGABRT), and the task printed
    only "killed by signal 6", losing the panic message. Planted panic: a genome of 0 nodes. The task must still die by
    SIGABRT, and its stderr and record must carry the engine's message."""
    path, sha = _targets(tmp_path, "nand(x1, x2)")
    p = _task(_args(path, sha, n_nodes="0"))
    assert p.returncode == -signal.SIGABRT, p.stderr[-500:]
    assert "panicked" in p.stderr and "n_nodes must be at least 1" in p.stderr
    row = json.loads(next(l for l in p.stdout.splitlines() if l.startswith("RESULT "))[7:])
    assert row["child_signal"] == signal.SIGABRT and "n_nodes must be at least 1" in row["engine_stderr"]
    assert "evals_reached" not in row


def test_child_killed_by_sigkill_is_passed_on(tmp_path, cgp_exe):
    """Guardrail row 83 (found by CI): Linux ends a process at its hard CPU limit with SIGKILL, whose handler cannot be
    set; the task raised there and became an error row. A SIGKILL of the engine must end the task by SIGKILL."""
    import time
    path, sha = _targets(tmp_path, "eml(eml(eml(1, x), eml(x, 1)), eml(eml(x, x), 1))", var_target=1.0, target_scale=1.0)
    decoy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", f"{cgp_exe} run decoy"])
    p = subprocess.Popen([sys.executable, "-m", "reach.tasks.s2_cgp", *_args(path, sha, s="F-EML", budget="16",
                                                                               timeout="600")],
                         cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=_env())
    try:
        for _ in range(100):                                      # wait until our engine runs, then kill it alone:
            time.sleep(0.1)                                       # -P, children of this task only (guardrail row 152:
            if subprocess.run(["pkill", "-9", "-P", str(p.pid), "-f", f"{cgp_exe} run"]).returncode == 0:
                break                                             # the old pattern killed every engine on the machine)
        out, err = p.communicate(timeout=60)
        assert decoy.poll() is None, "the test killed an engine process that was not its own"
    finally:
        decoy.kill()
        decoy.wait()
    assert p.returncode == -signal.SIGKILL, err[-500:]
    row = json.loads(next(l for l in out.splitlines() if l.startswith("RESULT "))[7:])
    assert row["child_signal"] == signal.SIGKILL and "evals_reached" not in row


def test_cpu_of_the_child_and_hard_kills_reach_the_queue(tmp_path, cgp_exe):
    """Guardrail row 78: through the queue, the row's CPU seconds include the engine child's, and a child killed by
    the hard CPU limit (ulimit -t) makes the row a timeout with that signal and no evaluations reached, so the run is
    censored at every budget."""
    from reach.analysis.s2.load import tidy
    from reach.runner.queue import Phase, Queue
    path, sha = _targets(tmp_path, "eml(eml(eml(1, x), eml(x, 1)), eml(eml(x, x), 1))", var_target=1.0, target_scale=1.0)
    argv = ["{python}", "-m", "reach.tasks.s2_cgp", *_args(path, sha, s="F-EML", budget="16")]
    soft = [a if a != "60" else "1.0" for a in argv]                       # soft limit 1 CPU s, hard 30
    hard = [a if a != "60" else "1000" for a in argv]                      # soft limit far above the hard 1 CPU s
    ph = Phase("cgp", runs_dir=tmp_path / "runs")
    ph.write_manifest([dict(key="soft", argv=soft, cpu_limit=30), dict(key="hard", argv=hard, cpu_limit=1)])
    Queue(ph, workers=2, install_signals=False, out=io.StringIO()).run()
    rows = ph.results()
    s, h = rows["soft"], rows["hard"]
    assert s["terminal_state"] == "timeout" and s["queue_status"] == "ok" and s["evals_reached"] > 1
    assert s["cpu_s"] >= s["child_cpu_seconds"] >= 1.0
    assert h["terminal_state"] == "timeout" and h["queue_status"] == "cpu_limit"
    assert h["signal"] in (signal.SIGXCPU, signal.SIGKILL) and h["child_signal"] == h["signal"]
    assert "evals_reached" not in h and h["cpu_s"] >= 1.0
    t = tidy([h], (0.25, 1, 16))
    assert len(t) == 3 and all(x["censored"] and x["y"] == 0 for x in t)

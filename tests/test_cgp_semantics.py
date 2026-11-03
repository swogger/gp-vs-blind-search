"""Rust-Python agreement of the CGP engine's semantics (REQ-ENG-17; guardrail rows 75, 76), written before the code.

Two levels: every real operator on special and random complex pairs (`cgp ops`), and whole programs (`cgp eval`):
every target of every CGP set (pilot, calibration, pool, benchmark files) scored against its own values, the first
20 of them also against perturbed values near the exact rule's edges, and 1,000 random genomes per set. Boolean:
truth tables and errors identical. Real, on every platform: single operators within 1e-12 relative (absolute below
magnitude 1); for programs the same finite / +inf / -inf / NaN class for every real and imaginary part and identical
exact decisions. Program values are not bounded off the study machine: a last-bit difference of the maths library
grows along deep random chains (CI, Linux: up to 2.6e-6 relative in an R-ELEM6 random genome, 9.4e-12 in F-LDE0).
On the study machine, where every study run is made (guardrail row 38), everything is bit-identical (study_machine
tests).
"""
import functools
import json
import os
import zlib
from pathlib import Path

import numpy as np
import pytest

from reach import cgp
from reach import grammar as G

ROOT = Path(__file__).resolve().parents[1]
REAL_OPS = sorted(o for o in cgp.RUST_OPS if o in G.GUARDED_OPS or o == "lde")
SPECIALS = [0.0, -0.0, 1.0, -1.0, 2.5, np.pi, np.inf, -np.inf, np.nan, 5e-324, -1e-310, 1e-200, 1e300, -1e300,
            709.0, 709.8, 710.0, -708.0, -708.5, -746.0]
PERTURB = [(0.0, 0.0), (0.0, 1e-12), (0.0, 1e-10), (0.0, 1e-9), (0.0, 1e-8), (1e-4, 0.0), (3e-4, 0.0)]
N_RANDOM = 1_000
STATS = {}                        # largest differences seen, for the report (printed with -s)


def _python_op(name):
    return G.GUARDED_OPS[name] if name in G.GUARDED_OPS else G.FAMILY_OPS[name]


def _hex_to_complex(s):
    a = np.frombuffer(bytes.fromhex(s), dtype=">u8").astype(np.uint64).view(np.float64)
    return a.view(np.complex128)                          # never re + 1j * im: 1j * inf is nan + inf j


def _hex_f64(s):
    return float(np.array([int(s, 16)], dtype=np.uint64).view(np.float64)[0])


def _classes(v):
    """finite 0, +inf 1, -inf 2, NaN 3, for real and imaginary parts."""
    out = []
    for part in (v.real, v.imag):
        c = np.zeros(part.shape, int)
        c[part == np.inf], c[part == -np.inf], c[np.isnan(part)] = 1, 2, 3
        out.append(c)
    return np.stack(out)


def _assert_values_agree(rust, py, what, tol=1e-12):
    """Same class for every part; finite parts within `tol` relative (absolute below magnitude 1). tol None: the
    class pattern only, the largest difference is recorded."""
    cr, cp = _classes(rust), _classes(py)
    bad = np.argwhere(cr != cp)
    assert len(bad) == 0, f"{what}: non-finite pattern differs at {bad[:3].tolist()}: {rust[bad[0][1]]} vs {py[bad[0][1]]}"
    for r, p in ((rust.real, py.real), (rust.imag, py.imag)):
        fin = np.isfinite(r)
        d = np.abs(r[fin] - p[fin]) / np.maximum(np.maximum(np.abs(r[fin]), np.abs(p[fin])), 1.0)
        m = float(d.max()) if d.size else 0.0
        STATS[what.split(":")[0]] = max(STATS.get(what.split(":")[0], 0.0), m)
        assert tol is None or m <= tol, f"{what}: values differ by {m:.3g} relative"


def _bits_equal(rust, py):
    """Bit identity with NaN equal to NaN whatever its payload; signed zeros must match."""
    for r, p in ((rust.real, py.real), (rust.imag, py.imag)):
        same = (r.view(np.uint64) == p.view(np.uint64)) | (np.isnan(r) & np.isnan(p))
        if not same.all():
            return False
    return True


# ------------------------------------------------------------------ operators
def _op_pairs():
    rng = np.random.default_rng(20261026)
    sp = np.array([complex(a, b) for a in SPECIALS for b in SPECIALS[:12]])
    a, b = np.repeat(sp, len(sp)), np.tile(sp, len(sp))             # every pair of special values
    n = 20_000
    scale = 10.0 ** rng.integers(-3, 3, (4, n))
    ra = rng.normal(size=n) * scale[0] + 1j * rng.normal(size=n) * scale[1] * rng.choice([0, 1], n)
    rb = rng.normal(size=n) * scale[2] + 1j * rng.normal(size=n) * scale[3] * rng.choice([0, 1], n)
    return np.concatenate([a, ra, rng.uniform(2, 3, 5000)]), np.concatenate([b, rb, rng.uniform(-750, 750, 5000)])


@pytest.fixture(scope="module")
def op_results(cgp_exe):
    a, b = _op_pairs()
    spec = "\n".join(["ops " + " ".join(REAL_OPS),
                      f"guards {cgp.bits(G.EXP_HI)} {cgp.bits(G.EXP_LO)} {cgp.bits(np.finfo(float).tiny)}",
                      "a " + cgp.cbits(a), "b " + cgp.cbits(b)]) + "\n"
    status, _, out, err = cgp.run_engine(cgp_exe, "ops", spec)
    assert status == 0, err[-500:]
    rust = {d["op"]: _hex_to_complex(d["values"]) for d in map(json.loads, out.splitlines())}
    with np.errstate(all="ignore"):
        py = {o: (_python_op(o).fn(a) if _python_op(o).arity == 1 else _python_op(o).fn(a, b)) for o in REAL_OPS}
    return rust, py


def test_every_real_operator_agrees_with_python(op_results):
    rust, py = op_results
    assert set(rust) == set(REAL_OPS)
    for o in REAL_OPS:
        _assert_values_agree(rust[o], np.asarray(py[o], dtype=np.complex128), f"op {o}")


@pytest.mark.study_machine
def test_every_real_operator_is_bit_identical_on_the_study_machine(op_results):
    """Guardrail row 75: numpy's fused multiply-adds, SIMD hypot and libm complex functions, reproduced exactly."""
    rust, py = op_results
    for o in REAL_OPS:
        assert _bits_equal(rust[o], np.asarray(py[o], dtype=np.complex128)), o


# ------------------------------------------------------------------ programs
def test_every_cgp_set_is_covered():
    """Every calibration set, every held-out set (listed and random) and the validation set is cross-checked, and every
    one of their operators has Rust semantics."""
    from reach.runner.phases import CAL_SETS
    from reach.s2_lock import held_out
    sets = cgp.cgp_sets()
    assert set(CAL_SETS) | held_out() | {cgp.VALIDATION_SET} <= set(sets)
    assert {"R-ARITH", "R-ELEM6", "BOOL-AN-n5", "BOOL-AX1-n5", "RS-add_sqrt", "BOOL-RS-rimp_c0-n5"} <= set(sets)
    for s in sets:
        assert set(G.get(s).operators) <= cgp.RUST_OPS, s


def _set_targets(name):
    out = []
    for f in sorted((ROOT / "data/targets_s2" / name).glob("*.json")):
        if not f.name.endswith("_log.json"):
            out += json.loads(f.read_text())
    return out


def _reference(g, targets):
    """The target the batch fitness is measured against: the set's first target, else a small expression."""
    if targets:
        return G.parse_expr(targets[0]["expression"], g)
    binop = next(o for o in g.ops if o.arity == 2).name
    leaves = ("x", "x") if g.kind == "real" else ("x1", "x2")
    return G.parse_expr(f"{binop}({leaves[0]}, {leaves[1]})", g)


def _real_norms(g, tree, t=None):
    if t is not None and "var_target" in t:
        return t["var_target"], t["target_scale"]
    v = G.evaluate(g, tree, G.sample_grid(g))
    return float(np.var(v.real)) or 1.0, float(np.abs(v).max()) or 1.0


@functools.lru_cache(maxsize=None)
def _crosscheck(exe, name):
    """Score every target and 1,000 random genomes of a set in Rust and in Python. Asserts the agreement that must
    hold on every platform; returns whether every value was also bit-identical."""
    g = G.get(name)
    targets = _set_targets(name)
    if name == cgp.VALIDATION_SET:                     # even-3 parity: not(odd parity), odd parity as minterms
        odd = "or(and(x1, nor(x2, x3)), or(and(x2, nor(x1, x3)), or(and(x3, nor(x1, x2)), and(x1, and(x2, x3)))))"
        targets = [dict(target_id="even3", expression=f"nor({odd}, {odd})")]
    ref = _reference(g, targets)
    kw = dict(zip(("var_target", "target_scale"), _real_norms(g, ref, targets[0] if targets else None))) \
        if g.kind == "real" else {}
    ctx = cgp.context(g, ref, **kw)
    lines = cgp.spec_lines(g, ctx)
    trees = [G.parse_expr(t["expression"], g) for t in targets]
    progs = [cgp.tree_to_program(g, t) for t in trees]
    cases = []
    if g.kind == "real":
        bx = np.random.default_rng(7).uniform(G.DOMAIN_LO, G.DOMAIN_HI, cgp.BATCH).astype(np.complex128)
        by = G.evaluate(g, ref, bx)
        lines += ["batch_x " + cgp.cbits(bx), "batch_y " + cgp.cbits(by)]
        for i, (t, tree, prog) in enumerate(zip(targets, trees, progs)):
            var, scale = _real_norms(g, tree, t)
            vy, gy = G.evaluate(g, tree, ctx["val_x"]), G.evaluate(g, tree, ctx["grid_x"])
            for dv, dg in (PERTURB if i < 20 else PERTURB[:1]):
                c = dict(ctx, var_norm=var, scale=scale, penalty=1000.0 * var, val_y=vy * (1 + dv), grid_y=gy * (1 + dg))
                cases.append((prog, c))
                lines.append(f"case {cgp.bits(scale)} {cgp.bits(var)} {cgp.bits(1000.0 * var)} {cgp.cbits(c['val_y'])} "
                             f"{cgp.cbits(c['grid_y'])} " + " ".join(cgp.program_tokens(prog)))
        pts = np.concatenate([bx, ctx["val_x"], ctx["grid_x"]])          # elementwise: one evaluation for all three
        nb, nv = len(bx), len(ctx["val_x"])
    lines += ["program " + " ".join(cgp.program_tokens(p)) for p in progs]
    lines.append(f"sample {N_RANDOM} 500 {zlib.crc32(name.encode())}")
    status, _, out, err = cgp.run_engine(exe, "eval", "\n".join(lines) + "\n")
    assert status == 0, err[-500:]
    rows = [json.loads(l) for l in out.splitlines()]
    n_p, n_c = len(progs), len(cases)                    # the engine prints programs, then cases, then samples
    assert len(rows) == n_p + n_c + N_RANDOM
    checked = ([(r, ctx) for r in rows[:n_p]] + [(r, c) for r, (_, c) in zip(rows[n_p:n_p + n_c], cases)]
               + [(r, ctx) for r in rows[n_p + n_c:]])
    bitwise = True
    for k, (r, c) in enumerate(checked):
        prog = r["program"]
        if k < n_p + n_c:
            assert prog == {"n_in": len(cgp.leaves(g)), **(progs[k] if k < n_p else cases[k - n_p][0])}, "round trip"
        if g.kind == "bool":
            tt = cgp.evaluate(g, prog)
            assert int(r["tt"], 16) == tt, f"{name}: truth table differs for {prog}"
            assert r["fitness"] == bin((tt ^ ctx["tt"]) & g.mask).count("1")
            continue
        v = cgp.evaluate(g, prog, pts)
        pb, pv, pg = v[:nb], v[nb:nb + nv], v[nb + nv:]
        rb, rg = _hex_to_complex(r["batch"]), _hex_to_complex(r["grid"])
        _assert_values_agree(rb, pb, f"programs: batch values of {prog}", tol=None)     # see the module docstring
        _assert_values_agree(rg, pg, f"programs: grid values of {prog}", tol=None)
        fit = G.mse_against(pb, by, c["penalty"]) / c["var_norm"]
        val = G.mse_against(pv, c["val_y"], c["penalty"]) / c["var_norm"]
        exact = val <= cgp.TOL and G.scaled_func_hash(pg, c["scale"]) == G.scaled_func_hash(c["grid_y"], c["scale"])
        rf, rv = _hex_f64(r["fitness"]), _hex_f64(r["val"])
        assert np.isfinite(rf) == np.isfinite(fit) and np.isfinite(rv) == np.isfinite(val), (name, prog, rf, fit)
        assert r["exact"] == exact, f"{name}: exact decision differs for {prog}"
        for key, a, b in (("fitness", rf, fit), ("val", rv, val)):
            STATS[key] = max(STATS.get(key, 0.0), abs(a - b) / max(abs(a), abs(b), 1e-300))
        STATS["exact_true" if exact else "exact_false"] = STATS.get("exact_true" if exact else "exact_false", 0) + 1
        bitwise &= _bits_equal(rb, pb) and _bits_equal(rg, pg) and rf == fit and rv == val
    if g.kind == "real":
        tb = _hex_to_complex(rows[0]["target_batch"])
        _assert_values_agree(tb, by, f"{name}: the engine's target values", tol=None)
        bitwise &= _bits_equal(tb, by)
    return bitwise


@pytest.mark.parametrize("name", cgp.cgp_sets())
def test_programs_agree_with_python(cgp_exe, name):
    _crosscheck(cgp_exe, name)
    if os.environ.get("CGP_STATS"):
        print(name, STATS)


@pytest.mark.study_machine
def test_programs_are_bit_identical_on_the_study_machine(cgp_exe):
    """Guardrail row 75: on the study machine every value, nMSE and decision is bit-identical to Python's."""
    assert [s for s in cgp.cgp_sets() if not _crosscheck(cgp_exe, s)] == []

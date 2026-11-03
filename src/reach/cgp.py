"""Cartesian GP (D072; docs/design/cgp.md): the Rust engine's binary and pinned source, the run specification, and
the Python re-scoring of what the engine reports.

Pinning (REQ-RUN-23): the queue pins src/reach, not rust/. This module therefore names the SHA-256 of the Rust source
it expects (RUST_SOURCE_SHA256, tested to equal the source in the tree). The binary embeds the hash of the source it
was built from (build.rs); `build()` installs it in a store keyed by that hash, and `binary()` refuses a binary whose
embedded hash differs. A phase pinned to older Python code thus runs the binary built from its own Rust source.

Re-scoring (REQ-ENG-19): the engine reports active graphs (compact node lists). Python evaluates them with
reach.grammar's operators, which is the tree evaluation of their expansion, and applies the registered exact rule.
"""
import hashlib
import os
import shutil
import subprocess
import struct
import tempfile
from pathlib import Path

import numpy as np

from reach import grammar as G
from reach.analysis.s2.load import E0                         # evaluations per budget unit (2x10^4)
from reach.engines.treegp import BATCH_SIZE as BATCH          # fresh batch points per generation (75)
from reach.engines.treegp import CONV_THRESHOLD as TOL        # validation nMSE threshold (1e-6)
from reach.paths import ROOT
from reach.targets import VAL_SEED                            # validation stream; equal to s2_run.VAL_SEED (tested)

RUST_SOURCE_SHA256 = "b2e05fb873510249fcda743c1560afa390241892e3d53bf73f1b9e5a9dae0c05"
CRATE = ROOT / "rust" / "cgp"
STORE = ROOT / "rust" / "bin"                     # untracked; one immutable binary per source hash
EXPR_LIMIT = 2_000                                # best_expr only for expansions up to this many nodes
RUST_OPS = {"+", "-", "*", "/", "exp", "log", "sin", "cos", "sqrt", "square", "eml", "reml", "lde",
            "and", "or", "nand", "nor", "xor", "xnor", "not", "imp", "rimp", "nimp", "rnimp"}

# validation set of Goldman and Punch (2015, 2013): {AND, OR, NAND, NOR} on 3 inputs (docs/design/cgp.md)
VALIDATION_SET = "BOOL-AONN-n3"
G.GRAMMARS.setdefault(VALIDATION_SET, G.Grammar(VALIDATION_SET, "bool", tuple(G.BOOL_OPS[o] for o in
                                                ("and", "or", "nand", "nor")), ("x1", "x2", "x3"), (), 3))


class BinaryError(RuntimeError):
    """The CGP binary is missing or was not built from the expected source."""


# ------------------------------------------------------------------ source, build and store
def source_files(crate=CRATE):
    """The files whose bytes define the engine (as build.rs): the manifests, build.rs and every src/**/*.rs."""
    crate = Path(crate)
    files = ["Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain.toml"]
    files += [str(p.relative_to(crate)).replace(os.sep, "/") for p in (crate / "src").rglob("*.rs")]
    return sorted(files)


def source_hash(crate=CRATE):
    """SHA-256 over (path, NUL, length as u64 LE, bytes) of every source file, sorted by path (as build.rs)."""
    h = hashlib.sha256()
    for rel in source_files(crate):
        data = (Path(crate) / rel).read_bytes()
        h.update(rel.encode() + b"\0" + struct.pack("<Q", len(data)) + data)
    return h.hexdigest()


def drop_bytecode(path):
    """Remove the cached bytecode of a source file (guardrail row 82). Python trusts a cache whose recorded size and
    modification second match the source, so a rewrite of the same size within the same second (the source-hash
    constant) is not recompiled; the system Python keeps these caches under sys.pycache_prefix, not next to the file."""
    import importlib.util
    Path(importlib.util.cache_from_source(str(path))).unlink(missing_ok=True)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def embedded_hash(exe):
    """The source hash compiled into a binary (`cgp --source-hash`)."""
    p = subprocess.run([str(exe), "--source-hash"], capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        raise BinaryError(f"{exe} --source-hash failed: {p.stderr[-300:]}")
    return p.stdout.strip()


def cargo():
    exe = shutil.which("cargo") or str(Path.home() / ".cargo" / "bin" / "cargo")
    if not Path(exe).exists():
        raise BinaryError("cargo not found: install Rust (docs/runbooks/setup.md)")
    return exe


def build(crate=CRATE, store=STORE, profile="release"):
    """Build the crate (pinned toolchain, locked, offline) and install the binary in the store under its source hash.
    Returns the installed path. Refuses if the built binary does not embed the hash of the source on disk."""
    crate, store = Path(crate), Path(store)
    p = subprocess.run([cargo(), "build", f"--{profile}" if profile == "release" else "--profile=dev", "--locked",
                        "--offline", "--quiet"], cwd=crate, capture_output=True, text=True)
    if p.returncode != 0:
        raise BinaryError(f"cargo build failed:\n{p.stderr[-2000:]}")
    built = crate / "target" / ("release" if profile == "release" else "debug") / "cgp"
    want, got = source_hash(crate), embedded_hash(built)
    if got != want:
        raise BinaryError(f"built binary embeds {got}, the source on disk hashes to {want}")
    dest = store / got / "cgp"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(f".tmp{os.getpid()}")
        shutil.copy2(built, tmp)
        tmp.rename(dest)
    return dest


def binary(expected=RUST_SOURCE_SHA256, store=STORE):
    """The store's binary for the expected source; refused unless it embeds exactly that hash."""
    exe = Path(store) / expected / "cgp"
    if not exe.exists():
        raise BinaryError(f"no CGP binary for source {expected[:16]} in {store}: build it with reach.cgp.build() "
                          "from a checkout of that source")
    got = embedded_hash(exe)
    if got != expected:
        raise BinaryError(f"binary {exe} was built from source {got[:16]}, not the pinned {expected[:16]}")
    return exe


# ------------------------------------------------------------------ programs (compact active graphs)
def leaves(g):
    """The program inputs: the set's leaves, variables first (as tree GP's leaves)."""
    return list(g.variables) + list(g.constants)


def leaf_str(v):
    if isinstance(v, str):
        return v
    return str(int(v)) if float(v).is_integer() else repr(v)


def tree_to_program(g, tree):
    """A tree as a compact graph with identical subtrees merged (hash-consing). The number of distinct subtrees,
    leaves included, is `n_in_used + len(nodes)` (dag_nodes; an upper bound on the minimum DAG size)."""
    ins = leaves(g)
    nodes, memo = [], {}

    def visit(t):
        if not t.is_operator:
            return next(i for i, v in enumerate(ins) if isinstance(v, str) == isinstance(t.value, str) and v == t.value)
        a = visit(t.left)
        key = (t.value, a) if g.op[t.value].arity == 1 else (t.value, a, visit(t.right))
        if key not in memo:
            nodes.append(list(key))
            memo[key] = len(ins) + len(nodes) - 1
        return memo[key]
    out = visit(tree)
    return dict(n_in=len(ins), out=out, nodes=nodes)


def program_tokens(prog):
    return [str(prog["out"])] + [",".join(str(x) for x in n) for n in prog["nodes"]]


def evaluate(g, prog, xs=None):
    """Values of a program with reach.grammar's operators: the same calls, in the same order, as G.evaluate on its
    expansion (real: complex array over xs; Boolean: packed truth table)."""
    if g.kind == "real":
        with np.errstate(all="ignore"):
            vals = [xs] + [np.full(xs.shape, complex(c), dtype=np.complex128) for c in g.constants]
            for n in prog["nodes"]:
                op = g.op[n[0]]
                vals.append(op.fn(vals[n[1]]) if op.arity == 1 else op.fn(vals[n[1]], vals[n[2]]))
            return vals[prog["out"]]
    m = g.mask
    vm = G.var_masks(g.n_inputs)
    vals = [vm[v] for v in g.variables] + [m if c else 0 for c in g.constants]
    for n in prog["nodes"]:
        op = g.op[n[0]]
        vals.append(op.fn(m, vals[n[1]]) if op.arity == 1 else op.fn(m, vals[n[1]], vals[n[2]]))
    return vals[prog["out"]]


def sizes(prog):
    """Found-solution sizes: active function nodes; DAG size (active nodes + distinct inputs used); expanded tree
    size, counted by memoisation (never by expansion)."""
    n_in, nodes = prog["n_in"], prog["nodes"]
    used = {prog["out"]} if prog["out"] < n_in else set()
    size = [1] * n_in
    for n in nodes:
        size.append(1 + sum(size[a] for a in n[1:]))
        used |= {a for a in n[1:] if a < n_in}
    return dict(active_nodes=len(nodes), dag_size=len(nodes) + len(used), expanded_size=size[prog["out"]])


def expression(g, prog, limit=EXPR_LIMIT):
    """The expanded prefix expression (reach.grammar's format), or '' when the expansion exceeds `limit` nodes."""
    if sizes(prog)["expanded_size"] > limit:
        return ""
    s = [leaf_str(v) for v in leaves(g)]
    for n in prog["nodes"]:
        s.append(f"{n[0]}({', '.join(s[a] for a in n[1:])})")
    return s[prog["out"]]


# ------------------------------------------------------------------ targets, context, specification
def context(g, tree, var_target=None, target_scale=None, penalty="scaled"):
    """Everything the engine and the re-scoring need about a target, computed by reach.grammar (as s2_run)."""
    ctx = dict(set=g.name, kind=g.kind, program=tree_to_program(g, tree))
    if g.kind == "real":
        grid, val = G.sample_grid(g), G.sample_val(g, VAL_SEED)
        ctx.update(val_x=val, val_y=G.evaluate(g, tree, val), grid_x=grid, grid_y=G.evaluate(g, tree, grid),
                   var_norm=float(var_target), scale=float(target_scale),
                   penalty={"scaled": 1000.0 * float(var_target), "fixed": 1000.0}[penalty])
    else:
        ctx.update(tt=G.evaluate(g, tree))
    return ctx


def bits(x):
    return f"{struct.unpack('<Q', struct.pack('<d', float(x)))[0]:016x}"


def cbits(arr):
    return " ".join(f"{bits(z.real)} {bits(z.imag)}" for z in np.asarray(arr, dtype=np.complex128))


def spec_lines(g, ctx):
    """The problem part of a specification (rust/cgp/src/spec.rs)."""
    bad = [o for o in g.operators if o not in RUST_OPS]
    if bad:
        raise ValueError(f"{g.name}: operators {bad} have no Rust implementation")
    lines = [f"domain {g.kind}", "ops " + " ".join(g.operators), f"n_vars {len(g.variables)}"]
    if g.kind == "real":
        if len(g.variables) != 1:
            raise ValueError(f"{g.name}: the CGP engine takes one real variable")
        lines += ["consts " + " ".join(bits(c) for c in g.constants),
                  f"guards {bits(G.EXP_HI)} {bits(G.EXP_LO)} {bits(np.finfo(float).tiny)}",
                  f"penalty {bits(ctx['penalty'])}", f"var_norm {bits(ctx['var_norm'])}", f"tol {bits(TOL)}",
                  f"scale {bits(ctx['scale'])}", f"round_factor {bits(10.0 ** G.HASH_PRECISION)}",
                  f"domain_lo {bits(G.DOMAIN_LO)}", f"domain_hi {bits(G.DOMAIN_HI)}", f"batch_n {BATCH}",
                  "target " + " ".join(program_tokens(ctx["program"])),
                  "val_x " + cbits(ctx["val_x"]), "val_y " + cbits(ctx["val_y"]),
                  "grid_x " + cbits(ctx["grid_x"]), "grid_y " + cbits(ctx["grid_y"])]
    else:
        lines += ["consts " + " ".join(str(int(c)) for c in g.constants), f"target_tt {ctx['tt']:x}"]
    return lines


def run_spec(g, ctx, *, seed, n_nodes, max_evals, cpu_limit):
    return "\n".join(spec_lines(g, ctx) + [f"seed {int(seed)}", f"n_nodes {int(n_nodes)}",
                                            f"max_evals {int(max_evals)}", f"cpu_limit {float(cpu_limit)!r}"]) + "\n"


# ------------------------------------------------------------------ re-scoring (the registered exact rule)
def rescore(g, prog, ctx):
    """(score, exact) by Python: real (validation nMSE, nMSE <= 1e-6 and equal 9-decimal scale-aware grid hash, as
    tree GP's stop rule); Boolean (Hamming errors, truth table equal)."""
    if g.kind == "real":
        val = G.mse_against(evaluate(g, prog, ctx["val_x"]), ctx["val_y"], ctx["penalty"]) / ctx["var_norm"]
        exact = val <= TOL and (G.scaled_func_hash(evaluate(g, prog, ctx["grid_x"]), ctx["scale"])
                                == G.scaled_func_hash(ctx["grid_y"], ctx["scale"]))
        return val, bool(exact)
    tt = evaluate(g, prog)
    return float(bin((tt ^ ctx["tt"]) & g.mask).count("1")), tt == ctx["tt"]


# ------------------------------------------------------------------ running the engine and reading its claims
def run_engine(exe, mode, spec_text):
    """Run the binary once as a child process. Returns (wait status, child CPU seconds, stdout, stderr); the CPU time
    comes from wait4, so it is the child's own (REQ-RUN-24)."""
    with tempfile.TemporaryDirectory(prefix="cgp-") as tmp:
        spec, out, err = (os.path.join(tmp, n) for n in ("spec.txt", "out.txt", "err.txt"))
        Path(spec).write_text(spec_text)
        with open(out, "w") as fo, open(err, "w") as fe:
            proc = subprocess.Popen([str(exe), mode, spec], stdout=fo, stderr=fe)
            _, status, ru = os.wait4(proc.pid, 0)
            proc.returncode = status                       # reaped here, with its resource usage
        return status, ru.ru_utime + ru.ru_stime, Path(out).read_text(), Path(err).read_text()


def outcome(g, ctx, r):
    """Fields of a run record from the engine's report `r`, every claim re-scored by Python. A claim Python rejects is
    not exact: flagged `claim_rejected`, terminal state early_stopped (the run stopped before its budget). A final
    parent or best that Python finds exact without a claim is flagged `missed_exact`."""
    best_val, best_exact = rescore(g, r["best"], ctx)
    _, parent_exact = rescore(g, r["parent"], ctx)
    claim = bool(r["claim_exact"])
    exact = claim and best_exact
    sz = sizes(r["best"])
    return dict(
        exact=exact, claim_exact=claim, claim_rejected=claim and not best_exact,
        missed_exact=(not claim) and (best_exact or parent_exact), parent_exact=parent_exact,
        converged=r["gen_tol"] is not None,
        terminal_state="early_stopped" if claim and not best_exact else r["terminal_state"],
        rust_terminal_state=r["terminal_state"], evals_per_gen=r["evals_per_gen"],
        generations=(r["max_evals"] - 1) // r["evals_per_gen"],
        evals_exact=r["evals_exact"] if exact else None, eval_index_exact=r["eval_index_exact"] if exact else None,
        gen_exact=r["gen_exact"] if exact else None, gen_tol=r["gen_tol"], gens_reached=r["gens_reached"],
        evals_reached=r["evals_reached"], final_error=best_val, rust_final_error=r["best_val"],
        final_tree_size=sz["expanded_size"], active_nodes=sz["active_nodes"], dag_size=sz["dag_size"],
        parent_active_nodes=len(r["parent"]["nodes"]), replacements=r["replacements"],
        neutral_replacements=r["neutral_replacements"], rust_cpu_seconds=r["cpu_seconds"],
        best_expr=expression(g, r["best"]), best_graph=r["best"])


def cgp_sets():
    """Every set the CGP may run on: the calibration sets, the held-out sets (listed and random candidates, locked
    until Stage 2) and the validation set. All of their operators must have Rust semantics (REQ-ENG-17)."""
    from reach.runner.phases import CAL_SETS
    from reach.s2_lock import held_out
    return list(CAL_SETS) + sorted(held_out()) + [VALIDATION_SET]

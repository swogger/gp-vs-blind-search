"""Study 2 targets (tool T4; docs/design/targets.md; REQ-MEA-06 to -10, -16, -22). Paper 1's `reach.targets` is kept.

Within a set's enumeration ceiling: draw uniformly among the distinct functions of minimum size K that have a minimal
tree of depth <= DEPTH_LIMIT (the enumerator stores the shallowest minimal tree; C5); the excluded share is logged.
Above it: compose an operator with minimal parts whose sizes sum to K - 1 (constructed; K_proven None).

Real filters, in order:
1. finite on the grid and the validation sample; variance; magnitude ratio (paper 1);
2. real-valued (imaginary parts <= 1e-12 of the scale; REQ-MEA-22);
3. 50-digit screen: double-precision grid values equal 50-digit values within 1e-9 relative (paper 1 S3);
4. dead subtree (paper 1, D020);
5. near duplicate: no smaller enumerated function within nMSE <= 1e-6 (D022; also removes D050's noise duplicates).
Boolean: non-constant; essential inputs, algebraic degree and nonlinearity are recorded.
Every target records nodes, bits_uniform, bits_syn and bits_sem (tool T5). Files are frozen with a SHA-256.
"""
import hashlib
import json
import random
from pathlib import Path

import mpmath as mp
import numpy as np

from reach import grammar as G
from reach.engines.treegp import MAX_DEPTH
from reach.family_metrics import near_duplicates
from reach.galib.node import Node
from reach.targets import VAL_SEED, _real_info, dead_subtree, load_structure
from reach.units import SizeUnits

DEPTH_LIMIT = MAX_DEPTH
COMPLEX_TOL = 1e-12
FP_TOL = 1e-9
NEAR_DUP_NMSE = 1e-6
_MP_OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b, "/": lambda a, b: a / b,
           "exp": mp.exp, "log": mp.log, "sin": mp.sin, "cos": mp.cos, "sqrt": mp.sqrt, "square": lambda a: a * a,
           "eml": lambda a, b: mp.exp(a) - mp.log(b), "reml": lambda a, b: mp.exp(b) - mp.log(a),
           "lde": lambda a, b: mp.log(a) / mp.exp(b)}


# ------------------------------------------------------------------ real filters
def _mp_eval(tree, x):
    if not tree.is_operator:
        return mp.mpc(x) if isinstance(tree.value, str) else mp.mpc(tree.value)
    f = _MP_OPS[tree.value]
    return f(_mp_eval(tree.left, x)) if tree.right is None else f(_mp_eval(tree.left, x), _mp_eval(tree.right, x))


def fp_dependent(tree, xs, v):
    """True if the double-precision values differ from 50-digit values by more than FP_TOL (relative to max(1, |.|))."""
    with mp.workdps(50):
        for x, d in zip(xs, v):
            try:
                e = complex(_mp_eval(tree, mp.mpf(float(np.real(x)))))
            except (ZeroDivisionError, ValueError, OverflowError):
                return True
            if not np.isfinite(e) or abs(d - e) > FP_TOL * max(1.0, abs(e)):
                return True
    return False


def real_filters(g, tree, K, vals=None):
    """(fields, "") if the target passes, else (None, reason)."""
    info, why = _real_info(g, tree)
    if info is None:
        return None, why
    v = info.pop("vals")
    vv = G.evaluate(g, tree, G.sample_val(g, VAL_SEED))
    if max(np.abs(v.imag).max(), np.abs(vv.imag).max()) > COMPLEX_TOL * info["target_scale"]:
        return None, "complex"
    if fp_dependent(tree, G.sample_grid(g), v):
        return None, "fp_dependent"
    if dead_subtree(g, tree, v, info["var_target"]):
        return None, "dead_subtree"
    if vals is not None and near_duplicates(vals, v, info["var_target"], K)[1][6] > 0:
        return None, "near_duplicate"
    return info, ""


# ------------------------------------------------------------------ Boolean properties
def boolean_properties(tt, n):
    rows = 1 << n
    f = [(tt >> r) & 1 for r in range(rows)]
    essential = sum(any(f[r] != f[r ^ (1 << (n - 1 - i))] for r in range(rows)) for i in range(n))
    a = f[:]                                                  # algebraic normal form (Moebius transform)
    for i in range(n):
        for r in range(rows):
            if r >> i & 1:
                a[r] ^= a[r ^ (1 << i)]
    degree = max((bin(r).count("1") for r in range(rows) if a[r]), default=0)
    walsh = [sum((-1) ** (f[r] ^ (bin(r & w).count("1") & 1)) for r in range(rows)) for w in range(rows)]
    return dict(essential_inputs=essential, degree=degree, nonlinearity=(rows // 2) - max(abs(x) for x in walsh) // 2,
                weight=sum(f))


# ------------------------------------------------------------------ generation
def _units(g, e, noise_rows):
    return SizeUnits(g, e.stats, noise_rows=noise_rows)


def generate(g, enum_dir, levels, n_per_level, seed, vals=None, noise_rows=None, max_attempts=200_000):
    """Returns (targets, log). `vals`: enumerated values by level (real near-duplicate filter); `noise_rows`: D050 rows."""
    e = load_structure(g, enum_dir)
    ceiling = max(k for k in e.keys if len(e.keys[k]))
    su = _units(g, e, noise_rows)
    rng = random.Random(seed)
    targets, log = [], {}
    for K in levels:
        rej, got = {}, []

        def take(tree, k_proven, depth, seen=None, key=None):
            if g.kind == "real":
                fields, why = real_filters(g, tree, K, vals)
                if fields is None:
                    rej[why] = rej.get(why, 0) + 1
                    return
                extra = fields
            else:
                tt = G.evaluate(g, tree)
                if tt in (0, g.mask):
                    rej["constant"] = rej.get("constant", 0) + 1
                    return
                extra = dict(properties=boolean_properties(tt, g.n_inputs))
            if seen is not None:
                seen.add(key)
            got.append(dict(expression=G.expr_str(tree), K_construction=K, K_proven=k_proven, constructed=k_proven is None,
                            depth=depth, units=su.all(K), **extra))

        if K <= ceiling:
            n_level = len(e.keys.get(K, []))
            ok = [i for i in range(n_level) if e.depth[K][i] <= DEPTH_LIMIT]
            rng.shuffle(ok)
            for i in ok:
                if len(got) == n_per_level:
                    break
                tree = G.parse_expr(e.expr(K, i), g)
                take(tree, K, int(e.depth[K][i]))
            log[str(K)] = dict(proven=True, n_level=n_level, n_depth_ok=len(ok),
                               excluded_share=(1 - len(ok) / n_level) if n_level else 0.0, found=len(got), rejected=rej)
        else:
            seen = set()
            jobs = [(o, K - 1, None) for o in g.ops if o.arity == 1 and len(e.keys.get(K - 1, []))]
            jobs += [(o, i, K - 1 - i) for o in g.ops if o.arity == 2 for i in range(1, K - 1)
                     if len(e.keys.get(i, [])) and len(e.keys.get(K - 1 - i, []))]
            w = [len(e.keys[i]) * (1 if j is None else len(e.keys[j])) for _, i, j in jobs]
            for _ in range(max_attempts if jobs else 0):
                if len(got) == n_per_level:
                    break
                o, i, j = rng.choices(jobs, weights=w)[0]
                left = G.parse_expr(e.expr(i, rng.randrange(len(e.keys[i]))), g)
                right = None if j is None else G.parse_expr(e.expr(j, rng.randrange(len(e.keys[j]))), g)
                tree = Node(o.name, left, right, True)
                depth = G.tree_depth(tree)
                if depth > DEPTH_LIMIT:
                    rej["depth"] = rej.get("depth", 0) + 1
                    continue
                v = G.evaluate(g, tree, G.sample_grid(g)) if g.kind == "real" else G.evaluate(g, tree)
                key = int(e._key(v[None, :] if g.kind == "real" else np.array([v], dtype=np.uint64))[0])
                if key in seen:
                    rej["duplicate"] = rej.get("duplicate", 0) + 1
                    continue
                pos = np.searchsorted(e.known, np.uint64(key))
                if pos < len(e.known) and int(e.known[pos]) == key:
                    rej["smaller_in_enumeration"] = rej.get("smaller_in_enumeration", 0) + 1
                    continue
                take(tree, None, depth, seen, key)
            log[str(K)] = dict(proven=False, found=len(got), rejected=rej)
        for j, r in enumerate(got):
            r["target_id"] = f"{g.name}_K{K:02d}_{j:02d}"
        targets += got
    return targets, dict(grammar=g.name, seed=seed, ceiling=ceiling, depth_limit=DEPTH_LIMIT, levels=log)


def pilot_levels(levels, max_proven=12):
    """Pilot sizes from a set's counts: up to `max_proven` evenly spaced non-empty proven sizes from 3 to the ceiling,
    plus constructed sizes ceiling + 2, + 4, + 8 (reach can lie above the proven range, as for NAND)."""
    ks = [st["K"] for st in levels if st["U"] > 0 and st["K"] >= 3]
    ceiling = max(ks)
    if len(ks) > max_proven:
        idx = np.unique(np.round(np.linspace(0, len(ks) - 1, max_proven)).astype(int))
        ks = [ks[i] for i in idx]
    return ks, [ceiling + 2, ceiling + 4, ceiling + 8]


def write_frozen(out_dir, name, targets, log):
    """Write <name>.json, <name>_log.json and <name>.sha256 once; refuse to overwrite (frozen; REQ-MEA-10)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.json"
    if p.exists():
        raise FileExistsError(f"{p} is frozen")
    p.write_text(json.dumps(targets, indent=1, default=float))
    (out / f"{name}_log.json").write_text(json.dumps(log, indent=1))
    (out / f"{name}.sha256").write_text(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n")
    return p

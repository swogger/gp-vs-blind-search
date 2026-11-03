"""EML-family feasibility pilot (decision D021): minimal constructions of calculator functions per
operator, and the target sets for the GP scout.

Every family grammar has the same syntax (one binary operator, leaves {x, c}), so its syntactic
space is identical; only the semantics differ. Minimal sizes are proven by enumeration on EML-GP's
100-point grid on (2, 3) (9-digit semantic key), i.e. minimal among trees that agree with the
function on that domain; each found tree is re-verified by evaluation.
"""
import json
import math

import numpy as np

from reach import grammar as G
from reach.paths import ROOT
from reach.targets import _real_info, generate, load_structure

FAMILY = ["F-EML", "F-EDL", "F-NEML", "F-LDE0", "F-LDE1", "F-PLI", "F-PLM"]

# calculator functions of one variable on (2, 3), and constants (as constant functions)
CALC = {
    "exp(x)": np.exp, "ln(x)": np.log, "-x": lambda x: -x, "1/x": lambda x: 1 / x,
    "x^2": lambda x: x ** 2, "sqrt(x)": np.sqrt, "x+1": lambda x: x + 1, "x-1": lambda x: x - 1,
    "2x": lambda x: 2 * x, "x/2": lambda x: x / 2, "exp(-x)": lambda x: np.exp(-x),
    "x^x": lambda x: x ** x, "exp(exp(x))": lambda x: np.exp(np.exp(x)), "ln(ln(x))": lambda x: np.log(np.log(x)),
}
# calculator functions of two variables (F2-* grammars, leaves x, y, c)
CALC2 = {
    "x+y": lambda x, y: x + y, "x-y": lambda x, y: x - y, "x*y": lambda x, y: x * y, "x/y": lambda x, y: x / y,
    "x^y": lambda x, y: x ** y, "log_x(y)": lambda x, y: np.log(y) / np.log(x), "(x+y)/2": lambda x, y: (x + y) / 2,
    "sqrt(x^2+y^2)": lambda x, y: np.sqrt(x * x + y * y),
}
CONSTANTS = {"0": 0.0, "1": 1.0, "-1": -1.0, "2": 2.0, "1/2": 0.5, "e": math.e, "pi": math.pi, "i": 1j}


def enum_dir(name):
    return ROOT / "data/enumeration" / name


def minimal_table(name):
    """{function: (K, expression) or None} for every calculator function and constant."""
    g = G.get(name)
    xs = G.sample_grid(g)
    e = load_structure(g, enum_dir(name))
    if len(g.variables) == 1:
        targets = {k: f(xs) for k, f in CALC.items()}
        targets.update({k: np.full(xs.shape, complex(c)) for k, c in CONSTANTS.items()})
    else:
        targets = {k: f(xs["x"], xs["y"]) for k, f in CALC2.items()}
    out = {}
    for fname, vec in targets.items():
        key = e._key(vec[None, :])[0]
        hit = None
        for k in sorted(e.keys):
            idx = np.nonzero(e.keys[k] == key)[0]
            if len(idx):
                expr = e.expr(k, int(idx[0]))
                got = G.evaluate(g, G.parse_expr(expr, g), xs)
                ok = bool(np.allclose(got, vec, rtol=1e-9, atol=1e-12))     # guards against key collisions
                hit = dict(K=k, expr=expr, verified=ok)
                break
        out[fname] = hit
    ceiling = max(e.keys)
    return dict(grammar=name, ceiling=ceiling, table=out)


def scout_levels(ceiling, top=21):
    c = min(ceiling, top)
    return sorted(set(range(3, c + 1, 2)) | {c + 2, c + 4})


def prepare(name, n_per_level=10, seed=20261013):
    """Minimal table + scout targets (targets_fam) + calculator targets (calc) for one operator."""
    out = ROOT / "data/targets" / name
    out.mkdir(parents=True, exist_ok=True)
    res_dir = ROOT / "data/family"
    res_dir.mkdir(parents=True, exist_ok=True)
    mt = minimal_table(name)
    (res_dir / f"minimal_{name}.json").write_text(json.dumps(mt, indent=1))
    if not (out / "targets_fam.json").exists():
        ts, log = generate(G.get(name), enum_dir(name), scout_levels(mt["ceiling"]), n_per_level, seed)
        (out / "targets_fam.json").write_text(json.dumps(ts, indent=1))
        (out / "targets_fam_log.json").write_text(json.dumps(log, indent=1))
    if not (out / "calc.json").exists():
        g, calc = G.get(name), []
        for fname, hit in mt["table"].items():
            if hit is None or not hit["verified"] or fname in CONSTANTS:
                continue
            tree = G.parse_expr(hit["expr"], g)
            info, why = _real_info(g, tree)
            if info is None:
                continue
            info.pop("vals")
            calc.append(dict(target_id=f"{name}_calc_{fname}", function=fname, expression=hit["expr"],
                             K_construction=hit["K"], K_proven=hit["K"], depth=G.tree_depth(tree), **info))
        (out / "calc.json").write_text(json.dumps(calc, indent=1))
    return mt

"""Universality screen for one held-out candidate (D027) -> data/screen/S-<form>-<const>.json + RESULT line.

Sufficient condition (the reduction used for EML): the one-variable grammar builds e^x and ln x, and the
two-variable grammar builds x - y or x / y (x - y = ln(e^x / e^y)), or x + y together with -x (D027 amendment). Minimal sizes by enumeration on the usual
grids; every construction is re-verified on an extended real domain."""
import argparse
import json

import numpy as np

from reach import grammar as G
from reach.enumerate import Enumerator
from reach.paths import ROOT
from reach.targets import load_structure

ap = argparse.ArgumentParser()
ap.add_argument("--form", required=True)
ap.add_argument("--const", required=True)
ap.add_argument("--k1", type=int, default=15)
ap.add_argument("--k2", type=int, default=15)
ap.add_argument("--max_gb", type=float, default=3.0)
a = ap.parse_args()

rng = np.random.default_rng(0)
X_EXT = rng.uniform(0.5, 6.0, 60).astype(np.complex128)
Y_EXT = rng.uniform(0.5, 6.0, 60).astype(np.complex128)


def find(g, fns):
    """{name: dict(K, expr, ext_ok) or None}, smallest K on the enumeration grid."""
    d = ROOT / "data/enumeration" / g.name
    e = Enumerator(g, d, log=lambda *_: None)
    k_reached, state = e.run(a.k1 if len(g.variables) == 1 else a.k2, a.max_gb)
    e = load_structure(g, d)
    xs = G.sample_grid(g)
    ext = X_EXT if len(g.variables) == 1 else {"x": X_EXT, "y": Y_EXT}
    out = {}
    for name, f in fns.items():
        vec = f(xs) if len(g.variables) == 1 else f(xs["x"], xs["y"])
        key = e._key(vec[None, :])[0]
        out[name] = None
        for k in sorted(e.keys):
            idx = np.nonzero(e.keys[k] == key)[0]
            if len(idx):
                expr = e.expr(k, int(idx[0]))
                t = G.parse_expr(expr, g)
                want = f(ext) if len(g.variables) == 1 else f(ext["x"], ext["y"])
                got = G.evaluate(g, t, ext)
                fin = np.isfinite(got) & np.isfinite(want)
                ok = bool(fin.mean() >= 0.9 and np.allclose(got[fin], want[fin], rtol=1e-8, atol=1e-10))
                out[name] = dict(K=k, expr=expr, grid_ok=bool(np.allclose(G.evaluate(g, t, xs), vec, rtol=1e-9, atol=1e-12)),
                                 ext_ok=ok, ext_finite=float(fin.mean()))
                break
    return out, dict(k_reached=k_reached, state=state)


g1 = G.get(f"S-{a.form}-{a.const}")
one, info1 = find(g1, {"exp(x)": np.exp, "ln(x)": np.log, "-x": lambda x: -x})
good = lambda h: h is not None and h["grid_ok"] and h["ext_ok"]
res = dict(form=a.form, const=a.const, one=one, enum1=info1, two=None, enum2=None)
if good(one["exp(x)"]) and good(one["ln(x)"]):
    two, info2 = find(G.get(f"S2-{a.form}-{a.const}"), {"x-y": lambda x, y: x - y, "x/y": lambda x, y: x / y,
                                                               "x+y": lambda x, y: x + y})
    res.update(two=two, enum2=info2)
    routes = [two[f]["K"] for f in ("x-y", "x/y") if good(two[f])]
    if good(two["x+y"]) and good(one["-x"]):
        routes.append(two["x+y"]["K"] + one["-x"]["K"])
    sub_ok = bool(routes)
else:
    sub_ok = False
res["passed"] = bool(good(one["exp(x)"]) and good(one["ln(x)"]) and sub_ok)
if res["passed"]:
    res["score"] = one["exp(x)"]["K"] + one["ln(x)"]["K"] + min(routes)
out = ROOT / "data/screen"
out.mkdir(parents=True, exist_ok=True)
(out / f"{g1.name}.json").write_text(json.dumps(res, indent=1))
print("RESULT " + json.dumps(dict(grammar=g1.name, terminal_state="completed", passed=res["passed"],
                                  score=res.get("score"))), flush=True)

"""Enumeration-based target metrics for the EML family (D022).

near-duplicates   the nearest smaller function in GP's own fitness (nMSE, non-finite = 1000 per point)
building blocks   share of the target's non-leaf proper subtrees that are finite on the grid
locality          fitness of the target's one-edit neighbours (each proper subtree replaced by a leaf, each
                  leaf flipped): does search get graded partial credit near the target, or a needle?
"""
import json

import numpy as np

from reach import grammar as G
from reach.enumerate import Enumerator, read_level
from reach.paths import ROOT
from reach.targets import _replace, _subtrees


def load_values(g, upto):
    """Values of every enumerated level K <= upto, rebuilt from parent pointers."""
    e = Enumerator(g, ROOT / "data/enumeration" / g.name, log=lambda *_: None)
    for st in json.loads((e.out / "counts.json").read_text())["levels"]:
        k = st["K"]
        if k > upto or not st.get("vals_stored", True):
            continue
        par = read_level(e.out / f"level_{k:03d}.npz")[1]
        if k == 1:
            vals, _ = e._leaves()
        else:
            vals = np.empty((len(par), G.N_POINTS), np.complex128)
            for oi, i, j in np.unique(par[:, [0, 1, 3]], axis=0):
                r = np.where((par[:, 0] == oi) & (par[:, 1] == i) & (par[:, 3] == j))[0]
                vals[r] = e._apply(g.ops[oi], e.vals[i][par[r, 2]], e.vals[j][par[r, 4]])
        e.vals[k] = vals
    return e.vals


def nmse_rows(V, f, var):
    with np.errstate(all="ignore"):
        d = np.abs(V - f[None, :])
        bad = ~np.isfinite(d)
        return (np.where(bad, 1000.0, np.where(bad, 0.0, d) ** 2)).mean(axis=1) / var


def near_duplicates(vals, f, var, K):
    """(d_min, counts within 1e-6 / 1e-4 / 1e-2) against all functions of size < K."""
    best, cnt = np.inf, {6: 0, 4: 0, 2: 0}
    for k in sorted(vals):
        if k >= K:
            continue
        V = vals[k]
        for s in range(0, len(V), 200_000):
            nm = nmse_rows(V[s:s + 200_000], f, var)
            if len(nm):
                best = min(best, float(nm.min()))
            for p in cnt:
                cnt[p] += int((nm <= 10.0 ** -p).sum())
    return best, cnt


def target_metrics(g, tree, var, vals, xs=None):
    xs = G.sample_grid(g) if xs is None else xs
    K = G.count_nodes(tree)
    f = G.evaluate(g, tree, xs)
    d_min, cnt = near_duplicates(vals, f, var, K)
    paths = [(p, n) for p, n in _subtrees(tree) if p]
    inner = [n for p, n in paths if n.is_operator]
    sub_viable = float(np.mean([np.isfinite(G.evaluate(g, n, xs)).all() for n in inner])) if inner else 1.0
    leaves = [G.leaf(v) for v in list(g.variables) + list(g.constants)]
    nb = []
    for p, n in paths:
        for lf in leaves:
            if not n.is_operator and n.value == lf.value:
                continue
            nb.append(G.evaluate(g, _replace(tree, p, lf.copy()), xs))
    nbm = nmse_rows(np.array(nb), f, var) if nb else np.array([np.inf])
    fin = np.array([np.isfinite(v).all() for v in nb]) if nb else np.array([False])
    mag = np.abs(f)
    return dict(K=K, depth=G.tree_depth(tree), d_min=d_min, n_near_6=cnt[6], n_near_4=cnt[4], n_near_2=cnt[2],
                sub_viable=sub_viable, dyn_range=float(np.log10(mag.max() / mag.min())),
                nb_n=len(nb), nb_nonfinite=float(1 - fin.mean()),
                nb_med=float(np.median(np.log10(nbm[fin]))) if fin.any() else 3.0,
                nb_good=float((nbm <= 1e-2).mean()), nb_best=float(np.log10(max(nbm.min(), 1e-20))))

"""H7 comparator 1 (D070, aligned with the paper by D083 and D085; docs/design/gpf2013_comparator.md): the
difficulty-indicator models of Graff, Poli and Flores (2013, Evolutionary Computation 21(4):533-560,
doi 10.1162/EVCO_a_00096; author manuscript, section 3):

    Boolean:     P(f) = a0 + sum_{i=1..N} a_i theta_i(f) + sum_{j=2..N} b_j phi_j(f)          (Eq. 9)
    continuous:  P(f) = a0 + sum_i a_i rho_i(f) + sum_{j=2..N} b_j varsigma_j(f)              (Eq. 10)

theta_i (Eq. 3): the share of ordered input pairs at Hamming distance i whose outputs differ (normalised by
1 / (2^N C(N, i))). phi_j (Eq. 5): the sum over inputs x and variable multisets a of size j of the j-th order Boolean
difference; a repeated variable gives 0 (footnote 3), so the sum runs over sets. rho_i (Eq. 7): the sum over grid
points and variables of |f(x + h_i e_j) - f(x)| / h_i, h_i = i x the grid step, for every lag; terms outside the grid
are 0 (footnote 5). varsigma_j (Eq. 8): the sum over grid points and variable multisets of size j of |j-th order
discrete derivative| at the grid step; Eq. (10) uses j = 2..N, so one variable has none (D085).

Fitted as in section 3.4: indicators ordered by LARS, the number of indicators by 5-fold cross-validated relative
squared error (RSE), coefficients by least squares (reach.analysis.s2.gp2010's LAR and nested least squares). Adapted as
comparator 2: problems are calibration targets, P is the share of a target's runs with exact recovery at a budget,
one model per domain and H7 budget, predictions clipped to [0, 1] for the H7 criteria. Real indicators are computed on
raw and on standardised target vectors, and the form with the lower cross-validated RSE is kept (D085).
"""
import hashlib
import json
from itertools import combinations, combinations_with_replacement
from math import comb
from pathlib import Path

import numpy as np

from reach.analysis.s2 import gp2010

BUDGETS = gp2010.BUDGETS
FOLDS, SEED = gp2010.FOLDS, gp2010.SEED              # the same folds as comparator 2
FROZEN_FILE = "data/registration/gpf2013_comparator.json"  # step 3c: both domains, with its .sha256
FORMS = {"bool": ("raw",), "real": ("raw", "std")}   # ties go to the first


# ---------------------------------------------------------------- Boolean indicators (section 3.2)
def theta(f, n):
    """[theta_1, ..., theta_n] (Eq. 3) of a truth table f (row r: the inputs are the bits of r)."""
    f = [int(v) for v in f]
    rows = 1 << n
    out = []
    for i in range(1, n + 1):
        masks = [sum(1 << v for v in a) for a in combinations(range(n), i)]
        s = sum(f[x] != f[x ^ m] for x in range(rows) for m in masks)          # ordered pairs (j, k), H(j, k) = i
        out.append(s / (rows * comb(n, i)))
    return out


def phi(f, n):
    """[phi_2, ..., phi_n] (Eq. 5): sum over x and variable sets a of size j of the j-th order difference, the XOR of
    f over the 2^j corners x XOR e_T, T a subset of a."""
    f = [int(v) for v in f]
    rows = 1 << n
    out = []
    for j in range(2, n + 1):
        tot = 0
        for a in combinations(range(n), j):
            corners = [sum(1 << a[k] for k in range(j) if t >> k & 1) for t in range(1 << j)]
            for x in range(rows):
                d = 0
                for c in corners:
                    d ^= f[x ^ c]
                tot += d
        out.append(float(tot))
    return out


# ---------------------------------------------------------------- continuous indicators (section 3.3)
def rho(F, h):
    """[rho_1, ..., rho_L] (Eq. 7) on a regular grid F (one axis per variable, step h on every axis): lag i adds
    |F(x + i e_j) - F(x)| / (i h) over the points where x + i e_j is on the grid (others are 0, footnote 5); L = the
    longest axis minus 1."""
    F = np.asarray(F, float)
    L = max(F.shape) - 1
    out = []
    for i in range(1, L + 1):
        s = 0.0
        for ax in range(F.ndim):
            n = F.shape[ax]
            if i < n:
                s += float(np.abs(np.take(F, range(i, n), axis=ax) - np.take(F, range(0, n - i), axis=ax)).sum())
        out.append(s / (i * h))
    return out


def varsigma(F, h, j):
    """varsigma_j (Eq. 8): sum over variable multisets a of size j and grid points of |the composed forward
    differences along a| / h^j, at points where every corner is on the grid (others are 0)."""
    F = np.asarray(F, float)
    s = 0.0
    for a in combinations_with_replacement(range(F.ndim), j):
        D = F
        for ax in a:
            D = np.diff(D, axis=ax)
        s += float(np.abs(D).sum())
    return s / h ** j


def grid(g):
    """(shape, step) of the fitness grid of a real set (reach.grammar.sample_grid): 100 points on (2, 3) for one
    variable, 10 x 10 for two."""
    from reach import grammar as G
    span = G.DOMAIN_HI - G.DOMAIN_LO
    if len(g.variables) == 1:
        return (G.N_POINTS,), span / (G.N_POINTS + 1)
    return (10,) * len(g.variables), span / 11


def indicator_names(g):
    if g.kind == "bool":
        n = g.n_inputs
        return [f"theta_{i}" for i in range(1, n + 1)] + [f"phi_{j}" for j in range(2, n + 1)]
    shape, _ = grid(g)
    return [f"rho_{i}" for i in range(1, max(shape))] + [f"varsigma_{j}" for j in range(2, len(shape) + 1)]


def indicators(g, vector, form="raw"):
    """(names, values) of one target: its truth table (Boolean) or its grid values (real; gp2010.target_vector).
    form "std": the real vector standardised first (gp2010.standardise; D085)."""
    v = np.asarray(vector, float)
    if g.kind == "bool":
        if form != "raw":
            raise ValueError("Boolean indicators have the raw form only")
        return indicator_names(g), theta(v, g.n_inputs) + phi(v, g.n_inputs)
    if form == "std":
        v = gp2010.standardise(v[None, :])[0]
    shape, h = grid(g)
    F = v.reshape(shape)
    return indicator_names(g), rho(F, h) + [varsigma(F, h, j) for j in range(2, len(shape) + 1)]


def _matrix(ids, vectors, form, names=None):
    from reach import grammar as G
    rows = []
    for k in ids:
        nm, x = indicators(G.get(k[0]), vectors[tuple(k)], form)
        if names is not None and nm != names:
            raise ValueError(f"comparator 1: {k[0]} has indicators {nm[:2]}... ({len(nm)}), the model {len(names)}")
        names = nm
        rows.append(x)
    return names, np.array(rows, float)


# ---------------------------------------------------------------- the model (section 3.4)
def cv_curve(X, P, parts):
    """Cross-validated RSE for n = 1..M indicators: in each fold LARS orders the indicators on the training rows and
    least squares fits the first n; M = the shortest fold path."""
    X, P = np.asarray(X, float), np.asarray(P, float)
    preds = []
    for te in parts:
        tr = np.setdiff1d(np.arange(len(P)), te)
        order = gp2010.lar_order(X[tr], P[tr])
        preds.append(gp2010.NestedOLS(X[tr], P[tr], order).predict_all(X[te]))
    M = min(len(q) for q in preds)
    den = float(((P - P.mean()) ** 2).sum())
    if M == 0 or den == 0:
        return np.array([])
    Q = np.empty((M, len(P)))
    for te, q in zip(parts, preds):
        Q[:, te] = q[:M]
    return ((Q - P) ** 2).sum(axis=1) / den


def fit_budget(X, P, names, parts=None, m=None):
    """One model (one domain, one budget): n by the cross-validated RSE (the smallest on ties), then LARS on every
    target for n steps and least squares. `m` given: no cross-validation (tests)."""
    X, P = np.asarray(X, float), np.asarray(P, float)
    curve = None
    if m is None:
        curve = cv_curve(X, P, parts if parts is not None else gp2010.folds(len(P)))
        if len(curve) == 0 or not np.isfinite(curve).any():
            return dict(m=0, a0=float(P.mean()), coef=[], selected=[], cv_rse=None, train_rse=None,
                        note="no variation in P or too few targets")
        m = int(np.nanargmin(curve)) + 1
    order = gp2010.lar_order(X, P, max_steps=m)
    cf = gp2010.NestedOLS(X, P, order).coef(len(order))
    return dict(m=len(order), a0=float(cf[0]), coef=[float(c) for c in cf[1:]], selected=[names[i] for i in order],
                cv_rse=None if curve is None else float(curve[m - 1]), train_rse=None)


def fit_domain(recs, vectors, domain, budgets=BUDGETS, seed=SEED):
    """Comparator 1 for one domain: per form (raw; real also standardised) the cross-validated RSE averaged over the
    budgets; the lower form is kept (ties: raw); then one model per budget. Records of a set outside CAL_SETS are
    refused (gp2010.problems; no held-out data)."""
    ids, V, P = gp2010.problems(recs, vectors, budgets)
    keep = {b: np.flatnonzero(np.isfinite(P[b])) for b in budgets}
    skipped = [b for b in budgets if len(keep[b]) < 2 * FOLDS]
    budgets = tuple(b for b in budgets if b not in skipped)
    if not budgets:
        raise ValueError("comparator 1: too few targets at every budget")
    parts = {b: gp2010.folds(len(keep[b]), seed=seed) for b in budgets}
    forms = FORMS[domain]
    score, Xs = {}, {}
    for form in forms:
        names, Xs[form] = _matrix(ids, vectors, form)
        cs = []
        for b in budgets:
            curve = cv_curve(Xs[form][keep[b]], P[b][keep[b]], parts[b])
            cs.append(float(np.min(curve)) if len(curve) else float("nan"))
        score[form] = float(np.nanmean(cs)) if np.isfinite(cs).any() else float("inf")
    form = min(forms, key=lambda f: (score[f], forms.index(f)))
    X = Xs[form]
    model = dict(kind="graff_poli_flores_2013", domain=domain, form=form, form_cv_rse=score, names=names, seed=seed,
                 folds=FOLDS, budgets={}, budgets_without_model=list(skipped))
    for b in budgets:
        k = keep[b]
        mb = fit_budget(X[k], P[b][k], names, parts[b])
        model["budgets"][str(b)] = mb
        mb["train_rse"] = gp2010.rse(P[b][k], _predict_rows(model, X[k], b))
        mb["n_targets"] = int(len(k))
    return model


def _predict_rows(model, X, budget):
    mb = model["budgets"].get(str(budget))
    if mb is None:
        raise ValueError(f"comparator 1: no model at budget {budget} (budgets {sorted(model['budgets'])})")
    if not mb["coef"]:
        return np.full(len(X), mb["a0"])
    ix = [model["names"].index(s) for s in mb["selected"]]
    return mb["a0"] + np.asarray(X, float)[:, ix] @ np.asarray(mb["coef"], float)


def predict(model, recs, vectors):
    """Eq. (9) or (10) for each record's target at its budget (unclipped). A target whose indicators differ from the
    model's (another number of inputs or variables), or a budget without a model, is refused."""
    out = np.empty(len(recs))
    keys = sorted({(r["set"], r["target_id"]) for r in recs})
    _, X = _matrix(keys, vectors, model["form"], model["names"])
    row = {k: i for i, k in enumerate(keys)}
    for b in sorted({r["budget"] for r in recs}):
        ix = [i for i, r in enumerate(recs) if r["budget"] == b]
        out[ix] = _predict_rows(model, X[[row[(recs[i]["set"], recs[i]["target_id"])] for i in ix]], b)
    return out


def summary(model):
    """The model as stored in confirm's result (it holds no target vectors, so it is the model itself)."""
    return dict(model)


def freeze(models, path):
    """Both domains in one JSON file with its .sha256 (as gp2010.freeze); an existing file with other content is never
    replaced (a frozen comparator changes only by a recorded decision and an OSF update)."""
    path = Path(path)
    body = json.dumps(models, indent=1, sort_keys=True)
    if path.exists() and path.read_text() != body:
        raise RuntimeError(f"{path.name} exists with other content: a frozen comparator is never replaced. If it must "
                           "change, record a decision and an OSF update, and remove it by hand.")
    path.write_text(body)
    sha = hashlib.sha256(body.encode()).hexdigest()
    path.with_suffix(".sha256").write_text(f"{sha}  {path.name}\n")
    return sha


def score(model, recs, vectors, baseline, eps_set, *, boot, min_sets_ok, seed=20261020):
    """H7 criteria (predictor.score_p) on predictions clipped to [0, 1], plus the paper's RSE per budget on target
    rates (unclipped) and the share of predictions clipped (as gp2010.score)."""
    from collections import defaultdict
    from reach.analysis.s2 import predictor
    raw = predict(model, recs, vectors)
    p = np.clip(raw, 0.0, 1.0)
    res = predictor.score_p(recs, p, predictor.predict(baseline, recs), eps_set, boot=boot, seed=seed,
                            min_sets_ok=min_sets_ok)
    tgt = defaultdict(list)
    for i, r in enumerate(recs):
        tgt[(r["budget"], r["set"], r["target_id"])].append(i)
    rse_b = {}
    for b in sorted({k[0] for k in tgt}):
        ks = [k for k in tgt if k[0] == b]
        rse_b[str(b)] = gp2010.rse([np.mean([recs[i]["y"] for i in tgt[k]]) for k in ks], [raw[tgt[k][0]] for k in ks])
    res.update(rse=rse_b, clipped_share=float(np.mean(raw != p)), form=model["form"])
    return res

"""H7 comparator 2 (D072, D084; docs/design/gp2010_comparator.md): the performance model of Graff and Poli (2010,
Artificial Intelligence 174(15):1254-1276, doi 10.1016/j.artint.2010.07.005), Eq. (5), p.1258:

    P(t) = a0 + sum_{p in S} a_p * d(p, t)

t is a problem (here a calibration target: its values on the fitness grid, or its truth table), S a set of reference
problems drawn from the training targets (Sigma = T), d one of the six measures of the paper's Table 4. As in the paper
(section 3.4, p.1259): S = the first m columns entered by LARS (plain LAR, Efron et al. 2004); m by 5-fold
cross-validated relative squared error (RSE); coefficients by ordinary least squares; no regularisation. d is chosen
per domain by the same RSE (section 6.1). One model per domain and per budget; fitted on calibration targets only.
"""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

BUDGETS = (1, 2, 4, 8, 16)                  # the H7 budgets (= confirm.H4_BUDGETS)
FROZEN_FILE = "data/registration/gp2010_comparator.json"        # step 3c: both domains, with its .sha256
FOLDS = 5
SEED = 20261027


def _sq(T, R):
    return ((T[:, None, :] - R[None, :, :]) ** 2).sum(axis=2)


MEASURES = {                                # Table 4, p.1263, in the paper's order (ties go to the first)
    "mse": lambda T, R: _sq(T, R) / T.shape[1],
    "l1": lambda T, R: np.abs(T[:, None, :] - R[None, :, :]).sum(axis=2),
    "dot2": lambda T, R: (T @ R.T) ** 2,
    "dot3": lambda T, R: (T @ R.T) ** 3,
    "gauss": lambda T, R: np.exp(-_sq(T, R)),
    "log1p": lambda T, R: np.log1p(np.abs(T[:, None, :] - R[None, :, :])).sum(axis=2),
}


CANDIDATES = tuple(MEASURES) + tuple("std:" + k for k in MEASURES)   # raw first; "std:" = on standardised vectors


def standardise(V):
    """Each vector shifted and scaled to mean 0 and standard deviation 1, as the paper normalises for NBRF (footnote 8,
    p.1260); a constant vector becomes zeros (adaptation 7)."""
    V = np.asarray(V, float)
    sd = V.std(axis=1, keepdims=True)
    return np.where(sd > 0, (V - V.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1.0), 0.0)


def distances(measure, T, R):
    """d(p, t) for every problem t (rows of T) and reference p (rows of R): a len(T) x len(R) matrix. A measure named
    "std:<m>" applies <m> to the standardised vectors."""
    T, R = np.asarray(T, float), np.asarray(R, float)
    if measure.startswith("std:"):
        measure, T, R = measure[4:], standardise(T), standardise(R)
    out = np.empty((len(T), len(R)))
    step = max(1, 2_000_000 // max(1, R.size))         # blocks keep the |t - p| arrays small
    for i in range(0, len(T), step):
        out[i:i + step] = MEASURES[measure](T[i:i + step], R)
    return out


def target_vector(g, expression):
    """The problem t of one target (adaptation 1): real, its values on the fitness grid; Boolean, its truth table in
    {0, 1}, row r = bit r (the order of comparator.features). A non-finite or complex value is refused."""
    from reach import grammar as G
    tree = G.parse_expr(expression, g)
    if g.kind == "bool":
        tt = G.evaluate(g, tree)
        return [float((tt >> r) & 1) for r in range(1 << g.n_inputs)]
    with np.errstate(all="ignore"):
        v = np.asarray(G.evaluate(g, tree, G.sample_grid(g)), complex)
    if not np.all(np.isfinite(v)) or np.any(np.abs(v.imag) > 1e-12 * np.maximum(1.0, np.abs(v.real))):
        raise ValueError(f"{expression}: not finite and real on the grid; the comparator needs a real vector")
    return [float(x) for x in v.real]


# ---------------------------------------------------------------- LARS (plain LAR) and nested least squares
def lar_order(X, y, max_steps=None, tol=1e-8, trace=False):
    """Columns of X in the order plain LAR (Efron et al. 2004, section 2) enters them; columns standardised to mean 0
    and unit length and y centred (ambiguity A1). A column within 1e-8 (squared distance, unit-length columns) of the
    active span is skipped (A6); the path stops if the active set becomes numerically singular. With trace=True also
    returns the fitted vector after each step (for the equiangular check)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, p = X.shape
    if n < 2 or p == 0:
        return ([], []) if trace else []
    Xc = X - X.mean(axis=0)
    norms = np.sqrt((Xc ** 2).sum(axis=0))
    ok = norms > tol * max(norms.max(), 1e-300)
    cols = np.flatnonzero(ok)
    Xs = Xc[:, cols] / norms[cols]
    r0 = y - y.mean()
    limit = min(n - 1, len(cols)) if max_steps is None else min(max_steps, n - 1, len(cols))
    mu = np.zeros(n)
    c = Xs.T @ r0
    Gm = Xs.T @ Xs                                                     # Gram matrix: each step costs O(p m), not O(n p)
    GA = np.zeros((len(cols), limit), order="F")                      # Gram columns of the active set, in entry order
    Ginv = np.zeros((limit, limit))                                    # (X_A' X_A)^-1, grown by bordering
    cand = np.ones(len(cols), bool)
    active, mus = [], []
    j = int(np.argmax(np.abs(c)))
    while len(active) < limit:
        cand[j] = False                                                # try to add column j
        k = len(active)
        b = GA[j, :k]
        e = Ginv[:k, :k] @ b
        d2 = 1.0 - b @ e                                               # squared distance of column j from the active span
        if d2 > tol:
            Ginv[:k, :k] += np.outer(e, e) / d2
            Ginv[:k, k] = Ginv[k, :k] = -e / d2
            Ginv[k, k] = 1.0 / d2
            GA[:, k] = Gm[:, j]
            active.append(j)
            if trace:
                mus.append(mu.copy())
        if len(active) == limit or not cand.any():
            break
        m = len(active)
        s = np.sign(c[active])
        ginv1 = s * (Ginv[:m, :m] @ s)                                 # G_A^-1 1 with G_A the signed Gram matrix
        tot = float(ginv1.sum())
        if not (np.isfinite(tot) and tot > 0):                          # numerically singular: stop the path here
            break
        A = 1.0 / math.sqrt(tot)
        w = A * ginv1
        a = GA[:, :m] @ (s * w)                                        # = X' u, u = X_A (s w) the equiangular vector
        C = float(np.max(np.abs(c[active])))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            g1 = (C - c) / (A - a)
            g2 = (C + c) / (A + a)
        gam = np.where(cand, np.fmin(np.where(g1 > 1e-12, g1, np.inf), np.where(g2 > 1e-12, g2, np.inf)), np.inf)
        j = int(np.argmin(gam))
        if not np.isfinite(gam[j]):
            break
        with np.errstate(over="ignore", invalid="ignore"):
            c_new = c - gam[j] * a
        if not np.all(np.isfinite(c_new)):                             # numerical breakdown: stop the path here
            break
        if trace:
            mu = mu + gam[j] * (Xs[:, active] @ (s * w))
        c = c_new
    order = [int(cols[i]) for i in active]
    return (order, mus) if trace else order


class NestedOLS:
    """Least squares on [1, X[:, order[:m]]] for every m at once (one QR; Golub and Van Loan): coefficients of the
    model with the first m entered columns, m = 1..M."""

    def __init__(self, X, y, order):
        self.order = list(order)
        Z = np.column_stack([np.ones(len(y)), np.asarray(X, float)[:, self.order]])
        self.Q, self.R = np.linalg.qr(Z)
        self.qy = self.Q.T @ np.asarray(y, float)

    def coef(self, m):
        """[a0, a_1..a_m] for the first m columns of the order."""
        return solve_triangular(self.R[:m + 1, :m + 1], self.qy[:m + 1])

    def predict_all(self, Xnew):
        """Predictions for the rows of Xnew (all columns of X, in X's order) by the models with m = 1..M columns: an
        M x len(Xnew) array. With R upper triangular, the leading block of R^-1 inverts the leading block of R, so the
        prediction with m columns is the cumulative sum over k <= m of (Z R^-1)[:, k] * (Q'y)[k]."""
        Z = np.column_stack([np.ones(len(Xnew)), np.asarray(Xnew, float)[:, self.order]])
        Rinv = solve_triangular(self.R, np.eye(self.R.shape[0]))
        return np.cumsum((Z @ Rinv) * self.qy, axis=1)[:, 1:].T


def folds(n, k=FOLDS, seed=SEED):
    """A seeded random split of n problems into k near-equal parts (ambiguity A3); a list of index arrays."""
    return np.array_split(np.random.default_rng(seed).permutation(n), k)


def rse(P, Q):
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    den = float(((P - P.mean()) ** 2).sum())
    return float(((P - Q) ** 2).sum() / den) if den > 0 else float("nan")


def cv_curve(D, P, parts):
    """Cross-validated RSE for m = 1..M (section 3.4). D = d(p, t) for all problems (rows t) and references (columns
    p, the same problems: Sigma = T). In each fold, rows and candidate references are the training parts only (A2),
    LAR orders the columns, OLS gives coefficients for every m; M = the smallest fold's limit (A4)."""
    P = np.asarray(P, float)
    preds = []
    for te in parts:
        tr = np.setdiff1d(np.arange(len(P)), te)
        Dtr = D[np.ix_(tr, tr)]
        order = lar_order(Dtr, P[tr])
        preds.append(NestedOLS(Dtr, P[tr], order).predict_all(D[np.ix_(te, tr)]))
    M = min(len(q) for q in preds)
    den = float(((P - P.mean()) ** 2).sum())
    if M == 0 or den == 0:
        return np.array([])
    Q = np.empty((M, len(P)))
    for te, q in zip(parts, preds):
        Q[:, te] = q[:M]
    return ((Q - P) ** 2).sum(axis=1) / den


def fit_budget(V, P, ids, measure, parts=None, m=None, D=None):
    """One model (one domain, one budget): m by the cross-validated RSE (smallest m on ties), then LAR on all of T
    for m steps and OLS (section 3.4). `m` given: no cross-validation (tests). D: the distance matrix, if computed."""
    V, P = np.asarray(V, float), np.asarray(P, float)
    D = distances(measure, V, V) if D is None else D
    curve = None
    if m is None:
        curve = cv_curve(D, P, parts if parts is not None else folds(len(P)))
        if len(curve) == 0 or not np.isfinite(curve).any():
            return dict(m=0, a0=float(P.mean()), coef=[], refs=[], ref_vectors=[], cv_rse=None, train_rse=None,
                        note="no variation in P or too few targets")
        m = int(np.nanargmin(curve)) + 1
    order = lar_order(D, P, max_steps=m)
    cf = NestedOLS(D, P, order).coef(len(order))
    return dict(m=len(order), a0=float(cf[0]), coef=[float(x) for x in cf[1:]], refs=[ids[i] for i in order],
                ref_vectors=[[float(x) for x in V[i]] for i in order],
                cv_rse=None if curve is None else float(curve[m - 1]), train_rse=None)


# ---------------------------------------------------------------- data, fitting, prediction
def _cal_sets():
    from reach.runner.phases import CAL_SETS
    return set(CAL_SETS)


def problems(recs, vectors, budgets=BUDGETS):
    """Per target: its vector and P(t) at each budget = share of its runs with y = 1 (adaptation 2). Refuses records of
    any set outside CAL_SETS (no held-out data, C1) and vectors of more than one length."""
    bad = sorted({r["set"] for r in recs} - _cal_sets())
    if bad:
        raise PermissionError(f"GP2010 comparator fitted on calibration sets only; refused {bad} (C1, D084)")
    ys = defaultdict(list)
    for r in recs:
        if r["budget"] in budgets:
            ys[(r["set"], r["target_id"], r["budget"])].append(r["y"])
    ids = sorted({(s, t) for s, t, _ in ys})
    if len({len(vectors[k]) for k in ids}) > 1:
        raise ValueError("GP2010 comparator: target vectors of different lengths in one model (adaptation 5)")
    V = np.array([vectors[k] for k in ids], float) if ids else np.zeros((0, 0))
    P = {b: np.array([np.mean(ys[(s, t, b)]) if ys.get((s, t, b)) else np.nan for s, t in ids]) for b in budgets}
    return [list(k) for k in ids], V, P


def fit_domain(recs, vectors, domain, budgets=BUDGETS, seed=SEED, measures=CANDIDATES):
    """The comparator for one domain: d by the lowest cross-validated RSE averaged over the budgets (section 6.1), among
    Table 4's six measures on raw and on standardised vectors (adaptation 7), then one model per budget. `vectors`:
    (set, target_id) -> list of floats."""
    ids, V, P = problems(recs, vectors, budgets)
    keep = {b: np.flatnonzero(np.isfinite(P[b])) for b in budgets}
    skipped = [b for b in budgets if len(keep[b]) < 2 * FOLDS]           # no records at that budget: no model there
    budgets = tuple(b for b in budgets if b not in skipped)
    if not budgets:
        raise ValueError("GP2010 comparator: too few targets at every budget")
    parts = {b: folds(len(keep[b]), seed=seed) for b in budgets}          # local indices, same split for every d
    score, Ds = {}, {}
    for d in measures:
        Ds[d] = distances(d, V, V)
        cs = []
        for b in budgets:
            k = keep[b]
            curve = cv_curve(Ds[d][np.ix_(k, k)], P[b][k], parts[b])
            cs.append(float(np.min(curve)) if len(curve) else float("nan"))
        score[d] = float(np.nanmean(cs)) if np.isfinite(cs).any() else float("inf")
    best = min(measures, key=lambda d: (score[d], measures.index(d)))
    D = Ds[best]
    models = {}
    for b in budgets:
        k = keep[b]
        mb = fit_budget(V[k], P[b][k], [ids[i] for i in k], best, parts[b], D=D[np.ix_(k, k)])
        mb["train_rse"] = rse(P[b][k], predict_vectors(dict(measure=best, budgets={str(b): mb}), V[k], b))
        mb["n_targets"] = int(len(k))
        models[str(b)] = mb
    return dict(kind="graff_poli_2010", domain=domain, measure=best, measure_cv_rse=score, seed=seed,
                folds=FOLDS, vector_length=int(V.shape[1]) if V.size else 0, budgets=models,
                budgets_without_model=list(skipped))


def predict_vectors(model, V, budget):
    """Eq. (5) for each row of V at one budget (unclipped). A budget without a model, or a vector of another length
    than the references', is refused."""
    mb = model["budgets"].get(str(budget))
    if mb is None:
        raise ValueError(f"GP2010 comparator: no model at budget {budget} (budgets {sorted(model['budgets'])})")
    V = np.asarray(V, float)
    if not mb["coef"]:
        return np.full(len(V), mb["a0"])
    R = np.asarray(mb["ref_vectors"], float)
    if V.ndim != 2 or V.shape[1] != R.shape[1]:
        raise ValueError(f"GP2010 comparator: vector length {V.shape[-1]} != the references' {R.shape[1]}")
    return mb["a0"] + distances(model["measure"], V, R) @ np.asarray(mb["coef"], float)


def predict(model, recs, vectors):
    """Per record: Eq. (5) for its target at its budget (unclipped)."""
    out = np.empty(len(recs))
    by = defaultdict(list)
    for i, r in enumerate(recs):
        by[r["budget"]].append(i)
    for b, ix in by.items():
        out[ix] = predict_vectors(model, [vectors[(recs[i]["set"], recs[i]["target_id"])] for i in ix], b)
    return out


def vectors_for(recs, targets):
    """(set, target_id) -> vector for the records' targets, from target records with an expression (confirm's
    `_TARGETS`); targets without an expression or whose vector is refused are returned as left out."""
    from reach import grammar as G
    vec, out = {}, []
    for k in sorted({(r["set"], r["target_id"]) for r in recs}):
        t = targets.get(k, {})
        try:
            vec[k] = target_vector(G.get(k[0]), t["expression"])
        except (KeyError, ValueError):
            out.append(list(k))
    return vec, out


def summary(model):
    """The model without its reference vectors (for confirm's result; `attach_vectors` restores them)."""
    return dict(model, budgets={b: {k: v for k, v in mb.items() if k != "ref_vectors"}
                                for b, mb in model["budgets"].items()})


def attach_vectors(model, vectors):
    """Reference vectors back from the target vectors, by reference id (step 3c, before freezing)."""
    return dict(model, budgets={b: dict(mb, ref_vectors=[list(vectors[tuple(k)]) for k in mb["refs"]])
                                for b, mb in model["budgets"].items()})


# ---------------------------------------------------------------- freeze and score
def freeze(models, path):
    """Both domains in one JSON file with its .sha256 (as predictor.freeze). An existing file with other content is
    never replaced (a frozen comparator changes only by a recorded decision and an OSF update)."""
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
    """H7 criteria (predictor.score_p) on predictions clipped to [0, 1] (adaptation 10), plus the paper's RSE per
    budget on target-level rates, unclipped, and the share of predictions clipped."""
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
        rse_b[str(b)] = rse([np.mean([recs[i]["y"] for i in tgt[k]]) for k in ks], [raw[tgt[k][0]] for k in ks])
    res.update(rse=rse_b, clipped_share=float(np.mean(raw != p)), measure=model["measure"])
    return res


def score_beside(frozen, comp1, comp2, recs, vectors, baseline, eps_set, *, boot, min_sets_ok, seed=20261020):
    """H7 scoring of one domain's held-out records (adaptation 11): the frozen size model, comparator 1 (Graff, Poli
    and Flores 2013, `gpf2013`; D085) and comparator 2, each by `predictor.score_p` against the same nodes-only
    baseline; both comparators read the targets' vectors and are clipped to [0, 1]."""
    from reach.analysis.s2 import gpf2013, predictor
    kw = dict(boot=boot, seed=seed, min_sets_ok=min_sets_ok)
    pb = predictor.predict(baseline, recs)
    return dict(frozen=predictor.score_p(recs, predictor.predict(frozen, recs), pb, eps_set, **kw),
                comparator1=gpf2013.score(comp1, recs, vectors, baseline, eps_set, **kw),
                comparator2=score(comp2, recs, vectors, baseline, eps_set, **kw))



def score_beside_registered(frozen, comp1, comp2, recs, vectors, baseline):
    """`score_beside` with the registered H7 values (predictor.H7_RULE, SCORED["H7"]; guardrail row 125)."""
    from reach.analysis.s2 import predictor
    r = predictor.H7_RULE
    return score_beside(frozen, comp1, comp2, recs, vectors, baseline, r["eps_set"], boot=r["boot"], seed=r["seed"],
                        min_sets_ok=predictor.SCORED["H7"]["min_sets_ok"])

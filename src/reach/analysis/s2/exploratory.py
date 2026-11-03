"""The four exploratory analyses filed in Stage 1 (OSF a3spg): "Exploratory and labelled: a grammar-level explanation of
set intercepts; Boolean Walsh-spectrum measures; the shape of the allocation curve; bloat." Design at the registered
commit 3b8aa5e: analysis.md 9.6 ("Regress the set-level intercepts alpha_G (in the H2-preferred unit) on the grammar
properties of 7.4 ... keep to one or two predictors chosen by a registered rule, report leave-one-set-out prediction
error"); hypotheses.md "Exploratory": "Boolean difficulty beyond size: Walsh-spectrum measures and number of essential
inputs"; "Shape of the allocation curve; bloat and final tree size". Built by D083; choices the texts leave open: D085.

    python -m reach.analysis.s2.exploratory --dryrun        (pilot data; never quote)
    python -m reach.analysis.s2.exploratory --calibration   (after the confirmatory calibration analysis)

Each writes its own JSON and four reports labelled EXPLORATORY; nothing here enters the confirmatory report, and no
test here is a confirmatory test (no Holm family, no decision).
"""
import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from reach.analysis import k50
from reach.analysis.s2 import alloc, confirm, law
from reach.analysis.s2.load import tidy

REPORTS = ("intercepts", "walsh", "allocation", "bloat")
MODES = {
    "dryrun": dict(phmap="pilot", result="data/analysis/dryrun.json", out="data/analysis/dryrun_exploratory.json",
                   prefix="docs/status/dryrun_exploratory", boot=200,
                   label="EXPLORATORY (DRYRUN: pilot data, not confirmatory; never quote these numbers)"),
    "calibration": dict(phmap="calibration", result="data/analysis/calibration.json",
                        out="data/analysis/calibration_exploratory.json", prefix="docs/status/calibration_exploratory",
                        boot=2000, label="EXPLORATORY (calibration data; filed as exploratory in Stage 1; not "
                                         "confirmatory)"),
}
SEED = 20261027
LENGTHS = (25, 50, 100, 200)                 # split lengths of the best split (generations), as confirm.best_split
WALSH = ("nonlinearity", "walsh_order", "walsh_degree", "spectral_entropy", "essential_inputs")
TREE_GP = ("A", "B", "HC")


def _pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def _resample(by_set, rng):
    """Targets resampled within sets (the study's target bootstrap): a list of (set, target, copy) draws."""
    out = []
    for g, tids in by_set.items():
        for j, i in enumerate(rng.integers(0, len(tids), len(tids))):
            out.append((g, tids[i], j))
    return out


# ---------------------------------------------------------------- Walsh-spectrum measures (Boolean targets)
def walsh_measures(f, n):
    """Walsh spectrum W(w) = sum_x (-1)^(f(x) + w.x) of a truth table (row r: inputs are the bits of r); energy shares
    W^2 / 4^n sum to 1. Measures: nonlinearity (2^(n-1) - max|W|/2), the energy-weighted mean order of w, the highest
    order with energy (Walsh degree), the spectral entropy in bits, and the number of essential inputs."""
    rows = 1 << n
    W = np.array([1.0 - 2.0 * int(v) for v in f])
    h = 1
    while h < rows:                                         # fast Walsh-Hadamard transform
        for i in range(0, rows, 2 * h):
            a, b = W[i:i + h].copy(), W[i + h:i + 2 * h].copy()
            W[i:i + h], W[i + h:i + 2 * h] = a + b, a - b
        h *= 2
    share = W ** 2 / rows ** 2
    order = np.array([bin(w).count("1") for w in range(rows)], float)
    nz = share > 1e-12
    ess = sum(any(int(f[x]) != int(f[x ^ (1 << i)]) for x in range(rows)) for i in range(n))
    return dict(nonlinearity=float(rows / 2 - np.abs(W).max() / 2), walsh_order=float((order * share).sum()),
                walsh_degree=float(order[nz].max()) if nz.any() else 0.0,
                spectral_entropy=float(-(share[nz] * np.log2(share[nz])).sum()) + 0.0, essential_inputs=float(ess))


def _design(recs, unit, sets, z=None):
    D = np.array([[r["set"] == g for g in sets] for r in recs], float)
    s = np.array([law._size(r, unit) for r in recs], float)
    cols = [D, -s[:, None]] + ([np.asarray(z, float)[:, None]] if z is not None else [])
    return np.column_stack(cols)


def _fit(X, y):
    """Logistic maximum likelihood, or None when a coefficient has no finite maximum-likelihood value (complete or
    quasi-complete separation, `k50.separated_columns`; guardrail row 118) or the fit does not converge."""
    if k50.separated_columns(X, y):
        return None
    return law._irls(X, y)


def _separated_sets(recs, unit):
    """Sets whose intercept has no finite maximum-likelihood value (for example every run exact, or none): reported and
    left out of the fit, repeatedly until none is left (row 118)."""
    out = []
    while True:
        sets = sorted({r["set"] for r in recs} - set(out))
        rr = [r for r in recs if r["set"] in sets]
        if not rr:
            return out
        y = np.array([r["y"] for r in rr], float)
        bad = [sets[j] for j in k50.separated_columns(_design(rr, unit, sets), y, range(len(sets)))]
        if not bad:
            return out
        out += bad


def _ll(X, y, b):
    p = np.clip(1 / (1 + np.exp(-X @ b)), 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def walsh(recs, unit, measures, boot=200, seed=SEED, folds=5):
    """Boolean difficulty beyond size, per measure: logit P = alpha_G - beta s + theta z at E0 on the Boolean main sets
    (z = the measure standardised over targets), theta with a 95% target-bootstrap CI (targets within sets); the gain in
    5-fold (over targets) cross-validated log-loss over the size-only model; Spearman's rank correlation of the measure
    with size over targets."""
    recs = [r for r in recs if (r["set"], r["target_id"]) in measures]
    sep = _separated_sets(recs, unit)
    recs = [r for r in recs if r["set"] not in sep]
    sets = sorted({r["set"] for r in recs})
    y = np.array([r["y"] for r in recs], float)
    tkey = [(r["set"], r["target_id"]) for r in recs]
    targets = sorted(set(tkey))
    by_set = defaultdict(list)
    for k in targets:
        by_set[k[0]].append(k)
    by_t = defaultdict(list)
    for i, k in enumerate(tkey):
        by_t[k].append(i)
    perm = np.random.default_rng(seed).permutation(len(targets))
    part = {targets[i]: f for f, ix in enumerate(np.array_split(perm, folds)) for i in ix}
    size = {k: law._size(recs[by_t[k][0]], unit) for k in targets}
    out = {}
    for m in WALSH:
        raw = np.array([measures[k][m] for k in targets], float)
        sd = raw.std()
        row = dict(mean=float(raw.mean()), sd=float(sd), spearman_size=float(spearmanr(raw, [size[k] for k in
                   targets]).correlation) if sd > 0 else float("nan"))
        if sd == 0:
            out[m] = dict(row, theta=float("nan"), lo=float("nan"), hi=float("nan"), cv_logloss_gain=float("nan"),
                          note="constant over targets")
            continue
        zt = dict(zip(targets, (raw - raw.mean()) / sd))
        z = np.array([zt[k] for k in tkey])
        Xf, Xb = _design(recs, unit, sets, z), _design(recs, unit, sets)
        b = _fit(Xf, y)
        rng = np.random.default_rng(seed)
        reps = []
        for _ in range(boot):
            ix = [i for g, k, _j in _resample(by_set, rng) for i in by_t[k]]
            bb = _fit(Xf[ix], y[ix])
            if bb is not None:
                reps.append(bb[-1])
        gain = []
        for f in range(folds):
            te = np.array([part[k] == f for k in tkey])
            bf, bs = _fit(Xf[~te], y[~te]), _fit(Xb[~te], y[~te])
            if bf is not None and bs is not None and te.any():
                gain.append(_ll(Xb[te], y[te], bs) - _ll(Xf[te], y[te], bf))
        out[m] = dict(row, theta=float(b[-1]) if b is not None else float("nan"), lo=_pct(reps, 2.5),
                      hi=_pct(reps, 97.5), dropped=boot - len(reps), estimable=b is not None and len(reps) >= 0.9 * boot,
                      cv_logloss_gain=float(np.mean(gain)) if gain else float("nan"))
    return dict(unit=unit, sets=len(sets), targets=len(targets), runs=len(recs), measures=out, boot=boot,
                separated_sets=sep)


# ---------------------------------------------------------------- grammar-level intercepts (design 9.6)
def loso_rmse(y, X):
    """Leave-one-set-out RMSE of an OLS line (intercept + the columns of X) predicting each set from the others."""
    y, X = np.asarray(y, float), np.asarray(X, float).reshape(len(y), -1)
    err = []
    for i in range(len(y)):
        k = np.arange(len(y)) != i
        Z = np.column_stack([np.ones(k.sum()), X[k]])
        b = np.linalg.lstsq(Z, y[k], rcond=None)[0]
        err.append(y[i] - np.concatenate([[1.0], X[i]]) @ b)
    return float(np.sqrt(np.mean(np.square(err))))


def intercepts(recs, unit, feats, boot=200, seed=SEED):
    """Set-level intercepts alpha_G of logit P = alpha_G - beta s at E0 (common beta; the H2 report unit), with 95%
    target-bootstrap CIs, regressed on the grammar properties (confirm.SET_FEATURES present for every set and not
    constant). Rule (D085): candidates are no property, each property alone and, with at least 6 sets, each pair; the
    lowest leave-one-set-out RMSE is chosen (ties: fewer properties, then the SET_FEATURES order)."""
    sep = _separated_sets(recs, unit)
    recs = [r for r in recs if r["set"] not in sep]
    sets = sorted({r["set"] for r in recs})
    y = np.array([r["y"] for r in recs], float)
    X = _design(recs, unit, sets)
    b = _fit(X, y) if len(sets) >= 2 else None
    if b is None:
        return dict(unit=unit, sets=sets, separated_sets=sep, status="not estimable (separated, too few sets or no "
                    "convergence)")
    by_set = defaultdict(list)
    by_t = defaultdict(list)
    for i, r in enumerate(recs):
        by_t[(r["set"], r["target_id"])].append(i)
    for k in sorted(by_t):
        by_set[k[0]].append(k)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(boot):
        ix = [i for g, k, _j in _resample(by_set, rng) for i in by_t[k]]
        bb = _fit(X[ix], y[ix])
        if bb is not None:
            reps.append(bb)
    R = np.array(reps) if reps else np.zeros((0, len(b)))
    beta = float(b[len(sets)])
    alpha = {g: dict(est=float(b[j]), lo=_pct(R[:, j], 2.5), hi=_pct(R[:, j], 97.5), k50=float(b[j]) / beta)
             for j, g in enumerate(sets)}
    sub = {g: feats.get(g, {}) for g in sets}
    props = list(confirm.usable_features(sub)) if all(sub.values()) else []
    a = np.array([alpha[g]["est"] for g in sets])
    cands = [()] + [(p,) for p in props] + (list(combinations(props, 2)) if len(sets) >= 6 else [])
    rows = []
    for c in cands:
        F = np.array([[sub[g][p] for p in c] for g in sets], float).reshape(len(sets), len(c))
        rows.append(dict(properties=list(c), loso_rmse=loso_rmse(a, F)))
    best = min(range(len(rows)), key=lambda i: (rows[i]["loso_rmse"], len(rows[i]["properties"]), i))
    c = rows[best]["properties"]
    F = np.column_stack([np.ones(len(sets))] + [[sub[g][p] for g in sets] for p in c])
    coef = np.linalg.lstsq(F, a, rcond=None)[0]
    ss = float(((a - a.mean()) ** 2).sum())
    r2 = 1 - float(((a - F @ coef) ** 2).sum()) / ss if ss > 0 else float("nan")
    alpha.update({g: dict(est=float("nan"), lo=float("nan"), hi=float("nan"), k50=float("nan"),
                          note="separated: no finite alpha_G (row 118)") for g in sep})
    return dict(unit=unit, sets=sets, separated_sets=sep, beta=beta, alpha=alpha, dropped=boot - len(reps), boot=boot,
                properties=props,
                candidates=sorted(rows, key=lambda r: r["loso_rmse"]),
                chosen=dict(properties=c, coef=[float(x) for x in coef], r2=r2, loso_rmse=rows[best]["loso_rmse"]))


# ---------------------------------------------------------------- allocation-curve shape
def whole_budget_k50(runs, budget, censored_as_failure=True):
    """K50 of one run of the whole budget (200 x budget generations), from the runs that were planned that long."""
    L = int(round(200 * budget))
    by = alloc._per_target([r for r in runs if r.get("generations", 0) >= L])
    Ks, ps = [], []
    for tid, rr in by.items():
        s, n = alloc.counts(rr, L, censored_as_failure)
        if n:
            Ks.append(rr[0]["K"])
            ps.append(s / n)
    return dict(k50=alloc._k50(Ks, ps), runs=sum(len(v) for v in by.values()), length=L)


def _curve(by, budget, lengths):
    k, plugs = alloc._table(by, budget, lengths)
    out = {str(L): k[L] for L in lengths}
    if budget > 1:
        runs = [r for rr in by.values() for r in rr]
        w = whole_budget_k50(runs, budget)
        if w["runs"]:
            out[str(w["length"])] = w["k50"]
    return out, any(plugs.values())


def _shape(curve):
    Ls = [int(L) for L in curve]
    finite = [L for L in Ls if not math.isnan(curve[str(L)]) and curve[str(L)] > float("-inf")]
    if not finite:
        return None, "not estimable", float("nan")
    best = max(finite, key=lambda L: curve[str(L)])                       # ties: the shortest (as alloc.best_split)
    shape = "falls" if best == Ls[0] else "rises" if best == Ls[-1] else "interior peak"
    pts = [(math.log2(L), curve[str(L)]) for L in finite if math.isfinite(curve[str(L)])]
    c2 = float(np.polyfit(*zip(*pts), 2)[0]) if len(pts) >= 3 else float("nan")
    return best, shape, c2


def allocation_shape(runs, budgets=(1, 2, 4, 8, 16), lengths=LENGTHS, boot=200, seed=SEED):
    """Per budget, K50 (nodes) of the best-split curve at each split length and of one run of the whole budget; the best
    length (ties: shortest), the shape (falls: shortest best, rises: longest best, else interior peak), the quadratic
    curvature of K50 on log2 length, and over a target bootstrap the share choosing each length and the curvature CI."""
    by = alloc._per_target(runs)
    tids = list(by)
    out = {}
    for B in budgets:
        curve, plug = _curve(by, B, lengths)
        best, shape, c2 = _shape(curve)
        rng = np.random.default_rng(seed)
        chosen, cs = [], []
        for _ in range(boot):
            bb = {f"{tids[i]}#{j}": by[tids[i]] for j, i in enumerate(rng.integers(0, len(tids), len(tids)))}
            b2, _s, c = _shape(_curve(bb, B, lengths)[0])
            if b2 is not None:
                chosen.append(b2)
            if math.isfinite(c):
                cs.append(c)
        out[str(B)] = dict(k50=curve, plug_in=plug, best_length=best, shape=shape, curvature=c2,
                           curvature_lo=_pct(cs, 2.5), curvature_hi=_pct(cs, 97.5),
                           chosen_share={str(L): chosen.count(L) / len(chosen) for L in sorted(set(chosen))})
    return out


# ---------------------------------------------------------------- bloat and final tree size
def bloat(rows, boot=200, seed=SEED):
    """Final best-of-run tree size of tree-GP runs (regimes A, B, headless chicken): per algorithm and run length the
    exact and failed runs; for exact programs the excess over the proven minimum size and the share that is minimal;
    for runs that failed through their whole length, the growth of the median final size from the shortest to the
    longest run length (target-bootstrap CI). Runs stopped early by the CPU limit without a hit are left out."""
    keep, cens = [], 0
    for r in rows:
        if r.get("algorithm") not in TREE_GP or r.get("terminal_state") == "error" or r.get("final_tree_size") is None:
            continue
        if r.get("gen_exact") is None and r.get("gens_reached", 0) < r.get("generations", 0):
            cens += 1
            continue
        keep.append(r)
    K = lambda r: r.get("K_construction", r.get("K"))                                      # noqa: E731
    cells = defaultdict(list)
    for r in keep:
        cells[(r["algorithm"], r["generations"])].append(r)
    out_cells = []
    for (a, gens), rr in sorted(cells.items()):
        ex = [r for r in rr if r.get("gen_exact") is not None]
        fl = [r for r in rr if r.get("gen_exact") is None]
        out_cells.append(dict(algorithm=a, generations=gens, runs=len(rr), exact=len(ex), failed=len(fl),
                              exact_median_ratio=float(np.median([r["final_tree_size"] / K(r) for r in ex])) if ex
                              else float("nan"),
                              failed_median_size=float(np.median([r["final_tree_size"] for r in fl])) if fl
                              else float("nan"),
                              failed_median_ratio=float(np.median([r["final_tree_size"] / K(r) for r in fl])) if fl
                              else float("nan")))
    exact, growth = {}, {}
    for a in sorted({r["algorithm"] for r in keep}):
        ex = [r for r in keep if r["algorithm"] == a and r.get("gen_exact") is not None
              and r.get("K_proven") is not None and r["K_proven"] == K(r)]
        if ex:
            exc = np.array([r["final_tree_size"] - r["K_proven"] for r in ex], float)
            rho = spearmanr(exc, [r["gen_exact"] for r in ex]).correlation if exc.std() > 0 else float("nan")
            exact[a] = dict(n=len(ex), median_excess=float(np.median(exc)), share_minimal=float(np.mean(exc == 0)),
                            spearman_hit_generation=float(rho))
        fl = [r for r in keep if r["algorithm"] == a and r.get("gen_exact") is None]
        lens = sorted({r["generations"] for r in fl})
        if len(lens) >= 2:
            lo_L, hi_L = lens[0], lens[-1]
            by = defaultdict(lambda: ([], []))
            for r in fl:
                if r["generations"] in (lo_L, hi_L):
                    by[r["target_id"]][r["generations"] == hi_L].append(r["final_tree_size"])
            tids = sorted(by)

            def ratio(ts):
                s = [x for t in ts for x in by[t][0]]
                l_ = [x for t in ts for x in by[t][1]]
                return float(np.median(l_) / np.median(s)) if s and l_ else float("nan")
            rng = np.random.default_rng(seed)
            reps = [ratio([tids[i] for i in rng.integers(0, len(tids), len(tids))]) for _ in range(boot)]
            reps = [x for x in reps if math.isfinite(x)]
            growth[a] = dict(short=lo_L, long=hi_L, ratio=ratio(tids), lo=_pct(reps, 2.5), hi=_pct(reps, 97.5))
    return dict(cells=out_cells, exact=exact, growth=growth, left_out_censored=cens, boot=boot)


# ---------------------------------------------------------------- build, render, main
def _units(root, md, required):
    p = Path(root) / md["result"]
    if not p.exists():
        if required:
            raise FileNotFoundError(f"{md['result']} not found: run the confirmatory calibration analysis first (the "
                                    "H2-preferred unit comes from it; D083 order)")
        return {d: ("bits_sem", "no confirmatory result found: bits_sem") for d in ("real", "bool")}
    h2 = json.loads(p.read_text()).get("h2", {})
    return {d: (h2.get(d, {}).get("report_unit") or "bits_sem",
                f"H2 report unit of {md['result']}" if h2.get(d, {}).get("report_unit") else "bits_sem (no H2 result)")
            for d in ("real", "bool")}


def build(mode, root=None, boot=None):
    from reach import grammar as G
    from reach.paths import ROOT
    md = MODES[mode]
    root = Path(root or ROOT)
    boot = boot or md["boot"]
    phmap = confirm.PHASE_MAPS[md["phmap"]]
    units = _units(root, md, required=mode == "calibration")
    a_rows, m_rows = confirm._rows(root, phmap["reach"]), confirm._rows(root, phmap["mech"])
    al_rows = confirm._rows(root, phmap.get("alloc", ()))
    sets = sorted({r.get("set") for r in a_rows if r.get("set")})
    confirm._load_units(root, sets, phmap["targets"])
    e0 = {g: confirm._with_units(tidy([r for r in a_rows if r.get("set") == g], (1,), True)) for g in sets}
    props = Path(root) / "data/analysis/properties.json"
    feats = confirm.set_features(root, sets) if props.exists() else {}
    res = dict(label=md["label"], mode=mode, boot=boot, units={d: dict(unit=u, source=s) for d, (u, s) in units.items()},
               intercepts={}, walsh={}, allocation={}, bloat={})
    for d in ("real", "bool"):
        main = [g for g in sets if G.get(g).kind == d and g not in confirm.SIZE_CHECKS]
        recs = [r for g in main for r in e0[g]]
        if len(main) >= 2 and recs:
            res["intercepts"][d] = intercepts(recs, units[d][0], feats, boot)
        if d == "bool" and recs:
            meas = {}
            for k, t in confirm._TARGETS.items():
                if k[0] in main and "expression" in t:
                    g = G.get(k[0])
                    tt = G.evaluate(g, G.parse_expr(t["expression"], g))
                    meas[k] = walsh_measures([(tt >> r) & 1 for r in range(1 << g.n_inputs)], g.n_inputs)
            res["walsh"] = walsh(recs, units[d][0], meas, boot)
    for g, (_tids, runs) in confirm.alloc_runs(a_rows, al_rows).items():
        res["allocation"][g] = allocation_shape(runs, confirm.H4_BUDGETS, boot=boot)
    for g in sets:
        res["bloat"][g] = bloat([r for r in a_rows + m_rows if r.get("set") == g], boot)
    return res


def _f(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}" if isinstance(x, float) \
        else str(x)


def render(res, name, source):
    L = [f"# EXPLORATORY: {dict(intercepts='grammar-level set intercepts', walsh='Boolean Walsh-spectrum measures', allocation='allocation-curve shape', bloat='bloat and final tree size')[name]}",
         "", f"> {res['label']}. Generated by `{source}`; filed in Stage 1 as exploratory (D083, D085): no test, no "
         "Holm family, no decision; not confirmatory.", ""]
    if name == "intercepts":
        L += ["logit P = alpha_G - beta s at E0 (regime A, main sets), common beta; alpha_G with a 95% target-bootstrap "
              "CI; regressed on grammar properties, chosen by leave-one-set-out RMSE (D085).", ""]
        for d, r in res["intercepts"].items():
            if "alpha" not in r:
                L += [f"- **{d}:** {r.get('status')}"]
                continue
            L += [f"## {d} (unit {r['unit']}; beta {_f(r['beta'], 3)})", "", "| set | alpha_G [95% CI] | K50 |",
                  "|---|---|---|"]
            L += [f"| {g} | {_f(a['est'])} [{_f(a['lo'])}, {_f(a['hi'])}] | {_f(a['k50'])} |" for g, a in r["alpha"].items()]
            c = r["chosen"]
            if r.get("separated_sets"):
                L += ["", f"Left out, separated (no finite alpha_G; row 118): {', '.join(r['separated_sets'])}."]
            L += ["", f"Chosen: {', '.join(c['properties']) or 'no property'} (coefficients "
                  f"{', '.join(_f(x, 3) for x in c['coef'])}; R² {_f(c['r2'])}; LOSO RMSE {_f(c['loso_rmse'], 3)}).",
                  "", "| properties | LOSO RMSE |", "|---|---|"]
            L += [f"| {', '.join(x['properties']) or '(none)'} | {_f(x['loso_rmse'], 3)} |" for x in r["candidates"]]
            L += [""]
    elif name == "walsh":
        w = res["walsh"]
        if not w:
            L += ["No Boolean records."]
        else:
            L += [f"Boolean main sets at E0, unit {w['unit']}: {w['sets']} sets, {w['targets']} targets, {w['runs']} runs. "
                  "logit P = alpha_G - beta s + theta z, z the measure standardised over targets; theta with a 95% "
                  "target-bootstrap CI; gain in 5-fold cross-validated log-loss over size alone (positive: the measure "
                  "adds information)." + (f" Left out, separated (row 118): {', '.join(w['separated_sets'])}."
                                          if w.get("separated_sets") else ""), "", "| measure | mean (sd) | Spearman with size | theta [95% CI] | CV log-loss "
                  "gain |", "|---|---|---|---|---|"]
            L += [f"| {m} | {_f(v['mean'])} ({_f(v['sd'])}) | {_f(v['spearman_size'])} | {_f(v['theta'], 3)} "
                  f"[{_f(v['lo'], 3)}, {_f(v['hi'], 3)}] | {_f(v['cv_logloss_gain'], 4)} |" for m, v in w["measures"].items()]
    elif name == "allocation":
        L += ["K50 (nodes) of the best split at each run length (generations) and of one run of the whole budget (last "
              "column; runs planned that long only); shape: falls (shortest best), rises (longest best) or interior "
              "peak; curvature: quadratic coefficient of K50 on log2 length [95% target-bootstrap CI]; * plug-in used.",
              "", "| set | budget | K50 by length | best (bootstrap share) | shape | curvature [CI] |",
              "|---|---|---|---|---|---|"]
        for g, per in res["allocation"].items():
            for B, v in per.items():
                ks = ", ".join(f"{L_}: {_f(k, 1)}" for L_, k in v["k50"].items())
                L.append(f"| {g} | {B}x | {ks}{' *' if v['plug_in'] else ''} | {v['best_length']} "
                         f"({_f(v['chosen_share'].get(str(v['best_length']), float('nan')))}) | {v['shape']} | "
                         f"{_f(v['curvature'], 2)} [{_f(v['curvature_lo'], 2)}, {_f(v['curvature_hi'], 2)}] |")
    else:
        L += ["Final best-of-run tree size (nodes) of tree-GP runs; ratio = size / K (constructed K above the ceiling); "
              "excess = size - proven minimum (proven targets only); growth = median failed size at the longest run "
              "length over the shortest [95% target-bootstrap CI]. CPU-stopped runs without a hit are left out.", "",
              "| set | algorithm | generations | runs | exact: median ratio | failed: median size (ratio) |",
              "|---|---|---|---|---|---|"]
        for g, b in res["bloat"].items():
            L += [f"| {g} | {c['algorithm']} | {c['generations']} | {c['runs']} | {_f(c['exact_median_ratio'])} | "
                  f"{_f(c['failed_median_size'], 1)} ({_f(c['failed_median_ratio'])}) |" for c in b["cells"]]
        L += ["", "| set | algorithm | exact: n, median excess, share minimal, Spearman with hit generation | "
              "growth (lengths) [CI] |", "|---|---|---|---|"]
        for g, b in res["bloat"].items():
            for a in sorted(set(b["exact"]) | set(b["growth"])):
                e, gr = b["exact"].get(a, {}), b["growth"].get(a)
                L.append(f"| {g} | {a} | {e.get('n', 0)}, {_f(e.get('median_excess', float('nan')), 1)}, "
                         f"{_f(e.get('share_minimal', float('nan')))}, {_f(e.get('spearman_hit_generation', float('nan')))} | "
                         + (f"{_f(gr['ratio'])} ({gr['short']}-{gr['long']}) [{_f(gr['lo'])}, {_f(gr['hi'])}] |" if gr
                            else "n/a |"))
    return "\n".join(L) + "\n"


def main(argv=None, root=None):
    from reach.paths import ROOT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dryrun", action="store_true", help="pilot data (never quote)")
    g.add_argument("--calibration", action="store_true", help="calibration data, after the confirmatory analysis")
    ap.add_argument("--boot", type=int, default=None)
    a = ap.parse_args(argv)
    mode = "dryrun" if a.dryrun else "calibration"
    root = Path(root or ROOT)
    res = build(mode, root, a.boot)
    md = MODES[mode]
    (root / md["out"]).parent.mkdir(parents=True, exist_ok=True)
    (root / md["out"]).write_text(json.dumps(res, indent=1, default=float))
    for name in REPORTS:
        (root / f"{md['prefix']}_{name}.md").write_text(render(res, name, "reach.analysis.s2.exploratory --" + mode))
    print(f"written {md['out']} and {md['prefix']}_{{{','.join(REPORTS)}}}.md (EXPLORATORY; list them in docs/README.md)")
    return res


if __name__ == "__main__":
    main()

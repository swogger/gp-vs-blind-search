"""The run-time model's registered tests H4d, H4e and H4f (REQ-ANA-44 to 47; D080 e-h). The model and the
Weibull-lognormal comparison are in `theory`.

Stage 1 (OSF a3spg), as filed: "H4d: the run-time model's kappa/alpha matches the measured slope within the same
margins" (+-0.5 real, +-1.0 Boolean bits_sem per doubling); "H4e: restarts help (best split better than one run)
exactly in the cells where kappa-hat < 1, in at least 75% of cells"; "H4f: the measured best run length lies within
the CI of t0/(1 - kappa-hat)"; "Equivalence: the 95% CI lies within the margin"; "P-values for Holm come from the
bootstrap by CI inversion"; "Bootstrap replicates with degenerate fits are dropped for that quantity; if more than 10%
are dropped, the CI is not estimable and the hypothesis is not supported."

Per set (design: "a discrete-time survival model per set and algorithm"): the Weibull fit on regime A's runs of the
set's budget levels (`fit_runs`), the set's measured slope (`law.reach_per_doubling`'s estimator, bits_sem), and, where
the set has allocation runs, one cell per budget comparing the best split with one run of the whole budget (`cell`).
One target bootstrap per set (the same draws for the fit, the slope and the cells; the fit warm-started from the
point estimate) gives every CI. `domain_tests` turns the sets into the three Holm members of a domain.

H4f (D080 h, amended before any calibration outcome was seen): T5 maximises success at a fixed total budget, so the
model's best length in a cell is the argmax over the cell's measured candidate lengths (`model_length`), not the raw
L* = t0/(1 - kappa), which can exceed the budget; L* and its CI are reported, not tested. The candidate lengths keep
those that solve nothing (K50 -inf; D087, stricter than D080 h): if the model picks one, the data cannot match it and
the cell fails.
"""
import math
import zlib
from collections import defaultdict

import numpy as np

from reach.analysis.s2 import alloc, law, theory
from reach.analysis.s2.load import outcome

GENS_PER_E0 = 200                   # E0 = 2 x 10^4 evaluations at population 100
LENGTHS = (25, 50, 100, 200)        # the best split's run lengths (alloc; runs at E0)
LSTAR_CAP = 1e9                     # L* for kappa >= 1 (restarts cannot help): infinite, held at this value
H4F_LEVEL = 0.95                    # H4f: the central 95% of the replicates' model lengths


def fit_runs(rows, censored_as_failure=True):
    """Runs for the fit from regime A's rows (units on each row): one per row, with b = bits_sem. A run stopped by its
    CPU limit without a hit is a failure through its planned length (primary) or left out (censored_as_failure=False,
    the sensitivity analysis); error rows and rows without a reached generation stay out (guardrail row 56). Returns
    (runs, counts)."""
    out, n = [], dict(error=0, no_reached_generation=0, cpu_failure=0, cpu_left_out=0)
    for r in rows:
        if r.get("terminal_state") == "error":
            n["error"] += 1
            continue
        if "gens_reached" not in r:
            n["no_reached_generation"] += 1
            continue
        x = dict(set=r["set"], target_id=r["target_id"], b=float(r["units"]["bits_sem"]), gen_exact=r.get("gen_exact"),
                 gens_reached=r["gens_reached"], generations=r.get("generations", r["gens_reached"]))
        if r.get("gen_exact") is None and r.get("terminal_state") == "timeout" and x["gens_reached"] < x["generations"]:
            if not censored_as_failure:
                n["cpu_left_out"] += 1
                continue
            x["gens_reached"] = x["generations"]
            n["cpu_failure"] += 1
        out.append(x)
    return out, n


def one_run_k50(runs, budget, censored_as_failure=True):
    """K50 (nodes) of one run of the whole budget: per target the share of runs exact before budget x 200 generations
    (`load.outcome`; runs shorter than that are 'beyond' and left out), fitted as the best split's (`alloc._k50`)."""
    L = int(round(budget * GENS_PER_E0))
    by = defaultdict(list)
    for r in runs:
        o = outcome(r, L)
        if o in ("beyond", "error") or (o == "censored" and not censored_as_failure):
            continue
        by[r["target_id"]].append((r["K"], o == "success"))
    if not by:
        return float("nan")
    return alloc._k50([v[0][0] for v in by.values()], [np.mean([s for _, s in v]) for v in by.values()])


def cell(runs, budget, censored_as_failure=True):
    """One H4e/H4f cell: the best split (lengths 25-200) against one run of the whole budget. help: the best split's
    K50 strictly above the one run's (None when either is missing or both are 'all solved'); length: the measured best
    run length, the length with the highest K50 among the split lengths and the whole budget (None when it is not
    unique at 'all solved', or nothing is solved at any length). grid: the model's candidate lengths, every length
    with a defined K50, including those that solve nothing (-inf; D087, stricter than D080 h), so the model's choice
    cannot silently move to the next length; an undefined (NaN) K50 is left out."""
    split = alloc.best_split(runs, budget, LENGTHS, censored_as_failure)
    whole = int(round(budget * GENS_PER_E0))
    # where the whole budget is one of the split lengths (E0), the one run IS that split: take its value, so float noise
    # between two formulas for the same share (1 - (n - s)/n against s/n) cannot decide "help" (guardrail row 108)
    one = split["k50"][whole] if whole in split["k50"] else one_run_k50(runs, budget, censored_as_failure)
    best = split["best_k50"]
    if math.isnan(best) or math.isnan(one) or (best == one == float("inf")):
        hlp = None
    else:
        hlp = bool(best > one)
    cands = {L: k for L, k in split["k50"].items() if not math.isnan(k)}
    if whole not in cands and not math.isnan(one):
        cands[whole] = one
    solved = {L: k for L, k in cands.items() if k > float("-inf")}
    length = None
    if solved:
        top = max(solved.values())
        at = sorted(L for L, k in solved.items() if k == top)
        length = None if (top == float("inf") and len(at) > 1) else at[0]
    return dict(budget=budget, best_split_k50=best, best_split_length=split["best_length"], one_run_k50=one,
                help=hlp, length=length, whole_budget_length=whole, grid=sorted(cands))


def model_length(kappa, t0, grid, whole):
    """H4f (D080 h, amended): the model's best run length in a cell. T5 (theory_model.md at 3b8aa5e): "the run length
    that maximises success at a fixed total budget"; "r = E/L runs give total hazard ∝ (E/L)·(L − t₀)^κ". With the
    fitted model's cumulative hazard H(T) = exp(eta) max(T - t0, 0)^kappa (`theory`; eta drops out of the argmax, and
    the conditioning on no hit at generation 0 changes nothing because H(1) = 0 for t0 >= 1), whole/L runs of L
    generations succeed with probability 1 - exp(-(whole/L) H(L)): the best length in `grid` (the cell's measured
    candidate lengths, generations) maximises (whole/L)(L - t0)^kappa. Ties: the shortest (as the measured best).
    None when kappa or t0 is not finite or no candidate is longer than t0 (no success predicted at any length)."""
    if not (np.isfinite(kappa) and np.isfinite(t0)):
        return None
    val = {L: (whole / L) * (L - t0) ** kappa for L in sorted(grid) if L > t0}
    if not val:
        return None
    top = max(val.values())
    return min(L for L, v in val.items() if v == top)


def p_among(choices, length):
    """H4f's p in one cell: the smallest alpha at which `length` is not among the lengths chosen in the central
    100(1 - alpha)% of the replicates' model lengths = min(1, 2 min(P(choice <= length), P(choice >= length))), the
    two-sided inversion of a percentile interval on the ordered grid; 0 when no replicate chose `length`."""
    x = np.asarray(choices, float)
    if len(x) == 0:
        return float("nan")
    if not np.any(x == length):
        return 0.0
    return float(min(1.0, 2 * min(np.mean(x <= length), np.mean(x >= length))))


def central(choices, level=H4F_LEVEL):
    """The lengths chosen in the central `level` of the replicates (those with p_among >= 1 - level)."""
    return sorted(L for L in set(choices) if p_among(choices, L) >= 1 - level - 1e-12)


def _lstar(kappa, t0):
    if not (np.isfinite(kappa) and np.isfinite(t0)):
        return float("nan")
    return min(t0 / (1 - kappa), LSTAR_CAP) if kappa < 1 else LSTAR_CAP


def _model(f):
    k, a, t0 = f["kappa"], f["alpha"], f["t0"]
    ok = all(np.isfinite(v) for v in (k, a, t0))
    return dict(kappa=k if ok else float("nan"), alpha=a if ok else float("nan"), t0=t0 if ok else float("nan"),
                ratio=k / a if ok and a > 0 else float("nan"), lstar=_lstar(k, t0) if ok else float("nan"))


def _fit(runs, **kw):
    try:
        return _model(theory.fit(runs, "weibull", **kw))
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return _model(dict(kappa=float("nan"), alpha=float("nan"), t0=float("nan")))


SEED = 20261020


def set_seed(name, seed=SEED):
    """The set's own bootstrap stream: SeedSequence([seed, crc32 of the set's name]), so sets draw independently
    whatever their order or number (guardrail row 113: one seed for every set gave sets with the same number of
    targets the same index draws, which correlated H4e's share across sets)."""
    return np.random.SeedSequence([seed, zlib.crc32(str(name).encode())])


def set_model(runs, band, alloc_runs, budgets, boot, seed=SEED, censored_as_failure=True, name=None):
    """Per set: the Weibull fit (kappa, alpha, t0, sigma), kappa/alpha against the measured slope (bits_sem per
    doubling over `budgets`, budget-level records `band`), L* and the cells (one per budget, from `alloc_runs`; none
    without them), with `boot` target-bootstrap replicates of every quantity (targets of the band drawn with
    replacement, each draw a new target for the target effect; allocation runs follow their target). The draws come
    from the set's own stream (`set_seed`; `name` defaults to the runs' set)."""
    names = {r.get("set") for r in runs}
    if name is None and len(names) != 1:
        raise ValueError(f"runtime.set_model: one set's runs expected, got {sorted(map(str, names))}")
    name = names.pop() if name is None else name
    f = theory.fit(runs, "weibull")
    by_band = law._by_target(band)
    slope = law._per_doubling(by_band, list(by_band), "bits_sem", budgets)
    point = _model(f)
    point.update(sigma_target=f["sigma_target"], gen0_hits=f["gen0_hits"], runs=f["n"], events=f["events"],
                 targets=f["targets"], slope=slope, diff=point["ratio"] - slope)
    cells = {b: cell(alloc_runs, b, censored_as_failure) for b in budgets} if alloc_runs else {}
    run_by, al_by = defaultdict(list), defaultdict(list)
    for r in runs:
        run_by[r["target_id"]].append(r)
    for r in alloc_runs or ():
        al_by[r["target_id"]].append(r)
    tids = sorted(set(by_band) | set(run_by))
    rng = np.random.default_rng(set_seed(name, seed))
    reps = dict(kappa=[], ratio=[], slope=[], diff=[], lstar=[], t0=[], help={b: [] for b in cells})
    for _ in range(boot):
        draw = rng.integers(0, len(tids), len(tids))
        rr, bb, aa = [], {}, []
        for j, i in enumerate(draw):
            t, lab = tids[i], f"{tids[i]}#{j}"
            rr += [dict(x, target_id=lab) for x in run_by.get(t, ())]
            bb[lab] = by_band.get(t, [])
            aa += [dict(x, target_id=lab) for x in al_by.get(t, ())]
        m = _fit(rr, frailty=f["frailty"], x0=f["x"]) if rr else _model(dict(kappa=float("nan"), alpha=float("nan"),
                                                                            t0=float("nan")))
        sl = law._per_doubling(bb, list(bb), "bits_sem", budgets)
        for k in ("kappa", "ratio", "lstar", "t0"):
            reps[k].append(m[k])
        reps["slope"].append(sl)
        reps["diff"].append(m["ratio"] - sl)
        for b in cells:
            reps["help"][b].append(cell(aa, b, censored_as_failure)["help"] if aa else None)
    return dict(point=point, cells=cells, reps=reps, boot=boot, budgets=list(budgets),
                weibull=dict(loglik=f["loglik"], converged=f["converged"], frailty=f["frailty"]))


def _ci(xs, boot):
    ok = [x for x in xs if np.isfinite(x)]
    lo, hi = np.percentile(ok, [2.5, 97.5]) if ok else (float("nan"),) * 2
    return ok, dict(lo=float(lo), hi=float(hi), dropped=boot - len(ok), estimable=len(ok) >= 0.9 * boot)


def h4d_set(m, margin):
    """H4d in one set: equivalence of kappa/alpha and the measured slope; p = the smallest alpha at which the two-sided
    CI of the difference lies strictly inside (-margin, margin) (the larger one-sided inversion)."""
    ok, ci = _ci(m["reps"]["diff"], m["boot"])
    p = max(law.p_bound(ok, -margin, "above", 2), law.p_bound(ok, margin, "below", 2)) if ok else float("nan")
    est = m["point"]["diff"]
    estimable = ci["estimable"] and np.isfinite(est)
    return dict(diff=est, ratio=m["point"]["ratio"], slope=m["point"]["slope"], margin=margin, p=p,
                holds=bool(estimable and -margin < ci["lo"] and ci["hi"] < margin), **dict(ci, estimable=estimable))


def lstar_ci(m):
    ok, ci = _ci(m["reps"]["lstar"], m["boot"])
    return ok, dict(ci, lstar=m["point"]["lstar"])


def _agree(hlp, kappa):
    return hlp is not None and np.isfinite(kappa) and hlp == (kappa < 1)


def domain_tests(models, margin, share):
    """The three Holm members of one domain from its sets' `set_model` results:
    H4d: conjunction over the sets (p = the largest set p; a set not estimable makes it untestable);
    H4e: the share of cells whose 'restarts help' equals (kappa-hat < 1), undefined cells not agreeing; p inverts the
         two-sided 95% bootstrap CI of the share against `share` ("at least": a replicate at it meets it); a replicate
         in which a set's fit failed is dropped;
    H4f: 'retain' (D080 h, amended): per cell the model's best length on the cell's measured grid (`model_length`) in
         every replicate; the cell holds when the measured best length is among the lengths chosen in the central 95%
         (`central`); p = `p_among`; member p = the smallest cell p; a cell without a measured length, or with more
         than 10% of replicates without a model length (a failed fit), makes the member untestable."""
    sets = sorted(models)
    d = {g: h4d_set(models[g], margin) for g in sets}
    why = [g for g in sets if not d[g]["estimable"]]
    h4d = dict(sets=d, p=max(v["p"] for v in d.values()) if sets and not why else None,
               untestable=None if sets and not why else ("no sets" if not sets else f"not estimable in {', '.join(why)}"),
               holds=bool(sets and all(v["holds"] for v in d.values())), margin=margin)
    cells = [(g, b) for g in sets for b in models[g]["cells"]]
    agree = {(g, b): _agree(models[g]["cells"][b]["help"], models[g]["point"]["kappa"]) for g, b in cells}
    boot = models[sets[0]]["boot"] if sets else 0
    reps = []
    for r in range(boot if cells else 0):
        ks = {g: models[g]["reps"]["kappa"][r] for g in {g for g, _ in cells}}
        if not all(np.isfinite(k) for k in ks.values()):
            reps.append(float("nan"))
            continue
        reps.append(float(np.mean([_agree(models[g]["reps"]["help"][b][r], ks[g]) for g, b in cells])))
    ok, ci = _ci(reps, boot)
    pt = float(np.mean(list(agree.values()))) if cells else float("nan")
    h4e = dict(cells=[dict(set=g, budget=b, help=models[g]["cells"][b]["help"], kappa=models[g]["point"]["kappa"],
                           agree=agree[(g, b)]) for g, b in cells], share=pt, registered_share=share,
               holds_point=bool(cells and pt >= share), **ci,
               p=law.p_bound(ok, share - 1e-9, "above", 2) if cells and ok else None,
               untestable=None if cells and ci["estimable"] else ("no allocation cells" if not cells else "not estimable"))
    fcells, unt = [], []
    for g in sets:
        m = models[g]
        if not m["cells"]:
            continue
        lci = lstar_ci(m)[1]
        for b, c in m["cells"].items():
            grid, whole = c.get("grid") or (), c.get("whole_budget_length")
            ch = [model_length(k, t, grid, whole) for k, t in zip(m["reps"]["kappa"], m["reps"]["t0"])]
            ok = [L for L in ch if L is not None]
            est = bool(ok) and len(ok) >= 0.9 * m["boot"]
            x = dict(set=g, budget=b, length=c["length"], grid=list(grid), estimable=est, dropped=m["boot"] - len(ok),
                     model_length=model_length(m["point"]["kappa"], m["point"].get("t0", float("nan")), grid, whole),
                     central=central(ok) if ok else [], lstar=lci["lstar"], lstar_lo=lci["lo"], lstar_hi=lci["hi"])
            if c["length"] is None or not est:
                x["p"] = None
                unt.append(f"{g} x{b}")
            else:
                x["p"] = p_among(ok, c["length"])
                x["among"] = bool(x["p"] >= 1 - H4F_LEVEL - 1e-12)
            fcells.append(x)
    h4f = dict(cells=fcells, p=min(x["p"] for x in fcells) if fcells and not unt else None,
               untestable=None if fcells and not unt else ("no allocation cells" if not fcells else
                                                           "no measured length or no model length in "
                                                           + ", ".join(unt[:6]) + ("…" if len(unt) > 6 else "")),
               holds_point=bool(fcells and not unt and all(x["among"] for x in fcells)))
    return dict(H4d=h4d, H4e=h4e, H4f=h4f)

"""H6d (T25, D098; docs/design/h6_explanation.md version 2; REQ-ANA-87): H6's gap split exactly into guidance, size
spread, repetition and an estimator offset, gap = G - S - R + O.

For a set and a budget of E evaluations: gap = K50(GP) - log2 E, H6c's own convention (K50 on corrected bits_sem by
`law._k50`; the null at log2 E). D is the mean number of distinct keys a run evaluates within E. The unguided twin hits a
level-K target with probability c_K = D_K / U_K (the share of level K GP covers, the run's own target left out);
enumeration of D hits with probability 1 up to the level where it ends, the remainder there, 0 above. K50(twin) and
K50(enum D) use H6's estimator on the same targets with these probabilities as rates. R = log2 E - log2 D,
S = K50(enum D) - K50(twin), G = K50(GP) - K50(twin), O = K50(enum D) - log2 D.

A run is dict(target_id, K, bits_sem, y={E: 0/1}, rv=the counter's record or None). Counts over runs that stop at a hit
are life-table means stratified by the run's target level (`cumulative`)."""
import json
import math
from pathlib import Path

import numpy as np

from reach.analysis.s2 import law
from reach.revisit import PREFIXES

E0 = 20_000
SIZE_CHECKS = ("BOOL-NAND-n6", "BOOL-AON-n6")          # left out of H6c's domain means (`confirm.SIZE_CHECKS`)
SEED, BOOT, KEEP = 20261020, 2000, 0.9               # H6's bootstrap; Stage 1's estimability rule (`nullgap`)
DELTA = {"H6d-1": 1.0, "H6d-2": 0.15}                # proposed margins (design 6), confirmed before Gate G5
ALPHA = 0.05
LOG2 = math.log2


# ---------------------------------------------------------------- counts over runs
def _snap(rv, p):
    return rv["prefixes"].get(str(p)) if p is not None else None


def _life_table(runs, E, field, pooled_inc=None):
    """Mean count at the first prefix over the runs that reached it, plus, per later prefix up to E, the mean increment
    over the runs that reached that prefix; an interval no run reached takes pooled_inc[p] (NaN without it)."""
    total, prev = None, None
    for p in (q for q in PREFIXES if q <= E):
        pairs = [(_snap(r["rv"], p), _snap(r["rv"], prev)) for r in runs]
        alive = [(np.asarray(s[field], float), np.asarray(o[field], float) if o is not None else 0.0)
                 for s, o in pairs if s is not None]
        if alive:
            inc = np.mean([s - o for s, o in alive], axis=0)
        elif pooled_inc is not None and p in pooled_inc:
            inc = pooled_inc[p]
        else:
            return float("nan")
        total = inc if total is None else total + inc
        prev = p
    return float("nan") if total is None else total


def _increments(runs, E, field):
    out, prev = {}, None
    for p in (q for q in PREFIXES if q <= E):
        pairs = [(_snap(r["rv"], p), _snap(r["rv"], prev)) for r in runs]
        alive = [np.asarray(s[field], float) - (np.asarray(o[field], float) if o is not None else 0.0)
                 for s, o in pairs if s is not None]
        if alive:
            out[p] = np.mean(alive, axis=0)
        prev = p
    return out


def cumulative(runs, E, field, how="stratified"):
    """Mean count `field` ("d", "undef", "lv", "lv_ok") within E evaluations over the runs (those with an rv).
    stratified (primary): a life table per target level, levels weighted by their share of runs, an interval a level
    has no run in taking the pooled increment; pooled: one life table; carry: a stopped run keeps its last counts (a
    lower bound); never_hit: runs that reached E without a hit."""
    runs = [r for r in runs if r.get("rv")]
    if not runs:
        return float("nan")
    if how == "pooled":
        return _life_table(runs, E, field)
    if how == "carry":
        def last(rv):
            s = _snap(rv, E)
            return s if s is not None else rv["stop"]
        return _scalar(np.mean([np.asarray(last(r["rv"])[field], float) for r in runs], axis=0))
    if how == "never_hit":
        keep = [r for r in runs if r["y"].get(E) == 0 and _snap(r["rv"], E) is not None]
        if not keep:
            return float("nan")
        return _scalar(np.mean([np.asarray(_snap(r["rv"], E)[field], float) for r in keep], axis=0))
    pooled = _increments(runs, E, field)
    by = {}
    for r in runs:
        by.setdefault(r["K"], []).append(r)
    parts = [(len(rr), _life_table(rr, E, field, pooled)) for rr in by.values()]
    if any(np.ndim(v) == 0 and not np.isfinite(v) for _, v in parts):
        return float("nan")
    return _scalar(sum(n * np.asarray(v) for n, v in parts) / len(runs))


def _scalar(v):
    return float(v) if np.ndim(v) == 0 else v


# ---------------------------------------------------------------- hit probabilities and K50s
def twin_probs(lv, U):
    """{K: share of level K covered}, from mean distinct keys per level `lv` (index = level); 0 above the bound."""
    return {k: (min(1.0, float(lv[k]) / U[k]) if k < len(lv) and U[k] > 0 else 0.0) for k in U}


def enum_probs(D, U):
    """Size-ordered enumeration of D distinct functions: 1 for full levels, the remainder at the level where it ends."""
    out, left = {}, float(D)
    for k in sorted(U):
        out[k] = 1.0 if U[k] == 0 or left >= U[k] else max(0.0, left / U[k])
        left -= U[k]
    return out


def k50_rates(runs, probs):
    """H6's estimator (`law._k50`, bits_sem) on the runs' targets with hit probabilities as rates."""
    return law._k50([dict(units=dict(bits_sem=r["bits_sem"]), y=probs.get(r["K"], 0.0)) for r in runs], "bits_sem")


def k50_interp(probs, U):
    """Where the per-level probability crosses 1/2, interpolated linearly in log2 U(<=K) over every level (the primary
    null's rule, `null.reach_bits_sem`); a sensitivity for the twin (review S2)."""
    cum, pts = 0, []
    for k in sorted(U):
        cum += U[k]
        if U[k] > 0:
            pts.append((LOG2(cum), probs.get(k, 0.0)))
    for (x0, p0), (x1, p1) in zip(pts, pts[1:]):
        if p0 >= 0.5 > p1:
            return x0 + (p0 - 0.5) / (p0 - p1) * (x1 - x0)
    return float("nan")


def gap(runs, E):
    gp = [r for r in runs if E in r["y"]]
    return law._k50([dict(units=dict(bits_sem=r["bits_sem"]), y=r["y"][E]) for r in gp], "bits_sem") - LOG2(E)


def set_terms(runs, E, U, how="stratified"):
    """gap, R, S, G, O and their parts for one set at budget E."""
    gp = [r for r in runs if E in r["y"]]
    d = cumulative(runs, E, "d", how)
    lv = cumulative(runs, E, "lv", how)
    k_gp = gap(runs, E) + LOG2(E)
    k_twin = k50_rates(gp, twin_probs(lv, U)) if np.ndim(lv) else float("nan")
    k_enum = k50_rates(gp, enum_probs(d, U)) if np.isfinite(d) else float("nan")
    log2_d = LOG2(d) if np.isfinite(d) and d > 0 else float("nan")
    return dict(gap=k_gp - LOG2(E), k50_gp=k_gp, k50_twin=k_twin, k50_enum=k_enum, log2_d=log2_d,
                R=LOG2(E) - log2_d, S=k_enum - k_twin, G=k_gp - k_twin, O=k_enum - log2_d)


def sensitivities(runs, E, U, U_ok):
    """Reported beside the primary terms (design 3): other coverage means, eligible coverage, the interpolated twin,
    evaluations on undefined or constant functions, and runs whose first sight of the target's key and registered
    exact hit disagree."""
    gp = [r for r in runs if E in r["y"]]
    k_gp = gap(runs, E) + LOG2(E)
    out = {}
    for how in ("pooled", "carry", "never_hit"):
        lv = cumulative(runs, E, "lv", how)
        out[f"G_{how}"] = k_gp - k50_rates(gp, twin_probs(lv, U)) if np.ndim(lv) else float("nan")
    lv_ok = cumulative(runs, E, "lv_ok")
    out["G_eligible"] = k_gp - k50_rates(gp, twin_probs(lv_ok, U_ok)) if np.ndim(lv_ok) else float("nan")
    lv = cumulative(runs, E, "lv")
    out["k50_twin_interp"] = k50_interp(twin_probs(lv, U), U) if np.ndim(lv) else float("nan")
    out["G_interp"] = k_gp - out["k50_twin_interp"]
    out["undef_share"] = cumulative(runs, E, "undef") / E
    out["key_hit_disagree"] = sum(1 for r in gp if r.get("rv") and
                                  ((r["rv"].get("target_first") is not None and r["rv"]["target_first"] <= E) != bool(r["y"][E])))
    return out


# ---------------------------------------------------------------- tests
def equivalence(point, reps, delta):
    """Two one-sided tests by the bootstrap: p = max(P(rep <= -delta), P(rep >= delta)) over the kept replicates;
    equivalent when the 90% interval lies inside +-delta and the statistic is estimable (at least 90% of replicates kept
    and a finite point estimate, Stage 1's rule)."""
    reps = np.asarray(reps, float)
    kept = reps[np.isfinite(reps)]
    estimable = bool(len(kept) >= KEEP * len(reps) and np.isfinite(point))
    if not len(kept):
        return dict(estimate=point, lo90=float("nan"), hi90=float("nan"), p=1.0, equivalent=False, estimable=False,
                    dropped=len(reps), delta=delta)
    lo, hi = (float(x) for x in np.percentile(kept, [5, 95]))
    p = max((np.sum(kept <= -delta) + 1) / (len(kept) + 1), (np.sum(kept >= delta) + 1) / (len(kept) + 1))
    return dict(estimate=float(point), lo90=lo, hi90=hi, p=float(p), delta=delta, estimable=estimable,
                dropped=int(len(reps) - len(kept)), equivalent=bool(estimable and -delta < lo and hi < delta))


def power(point, reps, delta, n_pilot, n_registered):
    """Power of the bootstrap TOST at the registered sample size, taking the pilot's estimate as the truth: the pilot
    replicates' SD scaled by sqrt(n_pilot / n_registered) as the registered SE, power = Phi((delta - |point|) / SE -
    z_0.95) (the nearer bound dominates), and the margin giving 80% power, |point| + (z_0.95 + z_0.80) x SE."""
    from scipy.stats import norm
    kept = np.asarray(reps, float)
    kept = kept[np.isfinite(kept)]
    se = float(np.std(kept, ddof=1) * math.sqrt(n_pilot / n_registered)) if len(kept) > 1 else float("nan")
    z95, z80 = norm.ppf(0.95), norm.ppf(0.80)
    pw = float(norm.cdf((delta - abs(point)) / se - z95)) if se > 0 else float("nan")
    return dict(estimate=float(point), se=se, delta=delta, power=pw, delta_80=float(abs(point) + (z95 + z80) * se))


def holm(tests, alpha=ALPHA):
    """Holm over the family: tests sorted by p (not estimable: p = 1), each compared with alpha / (m - i); the first
    failure stops the procedure. Adds 'supported' to each test."""
    order = sorted(tests, key=lambda k: tests[k]["p"] if tests[k]["estimable"] else 1.0)
    out, going = {}, True
    for i, k in enumerate(order):
        p = tests[k]["p"] if tests[k]["estimable"] else 1.0
        going = going and p <= alpha / (len(order) - i)
        out[k] = dict(tests[k], supported=bool(going), threshold=alpha / (len(order) - i))
    return out


def _by_target(runs):
    by = {}
    for r in runs:
        by.setdefault(r["target_id"], []).append(r)
    return by


def _slope(budgets, values):
    x, y = np.log2(np.asarray(budgets, float)), np.asarray(values, float)
    return float(np.polyfit(x, y, 1)[0]) if np.all(np.isfinite(y)) and len(y) >= 2 else float("nan")


def _resample(by_domain, tgt, rng):
    out = {}
    for d, s in by_domain.items():
        out[d] = {}
        for g, (_, U) in s.items():
            ids = list(tgt[d][g])
            out[d][g] = ([r for j in rng.integers(0, len(ids), len(ids)) for r in tgt[d][g][ids[j]]], U)
    return out


def _twin_difference(by_domain, e0):
    """H6d-1's statistic: mean K50(twin) over the Boolean main sets minus over the real ones, at e0."""
    k = {d: np.mean([set_terms(rr, e0, U)["k50_twin"] for rr, U in s.values()]) for d, s in by_domain.items()}
    return float(k["bool"] - k["real"])


def _real_guidance_slope(real_sets, budgets):
    """H6d-2's statistic: least-squares slope on log2 budget of the mean real G over the real main sets."""
    g = [np.mean([set_terms(rr, b, U)["G"] for rr, U in real_sets.values()]) for b in budgets]
    return _slope(budgets, g)


def domain_tests(by_domain, budgets=(E0,), boot=BOOT, seed=SEED, slope_sets=None, secondary=None):
    """by_domain {"real"|"bool": {set: (runs, U)}} at budgets[0] (E0: s2_revisit_a0 and _a16 pooled, as H6c) for
    H6d-1 (the twins' K50 equal across domains, within DELTA); slope_sets {set: (runs, U)} of the real domain at every
    budget (s2_revisit_a16, as H6a; default: by_domain["real"]) for H6d-2 (the slope per doubling of real G within
    DELTA of 0). Target bootstrap within sets for each statistic, Holm over the two."""
    e0, slope_sets = budgets[0], slope_sets or by_domain["real"]
    per = {d: {g: set_terms(rr, e0, U) for g, (rr, U) in s.items()} for d, s in by_domain.items()}
    tgt = {d: {g: _by_target(rr) for g, (rr, _) in s.items()} for d, s in by_domain.items()}
    rng = np.random.default_rng(seed)
    reps1 = [_twin_difference(_resample(by_domain, tgt, rng), e0) for _ in range(boot)]
    stats = {"H6d-1": (_twin_difference(by_domain, e0), reps1)}
    if len(budgets) >= 2:
        sl = {"real": slope_sets}
        tg2 = {"real": {g: _by_target(rr) for g, (rr, _) in slope_sets.items()}}
        rng = np.random.default_rng(seed)
        reps2 = [_real_guidance_slope(_resample(sl, tg2, rng)["real"], budgets) for _ in range(boot)]
        stats["H6d-2"] = (_real_guidance_slope(slope_sets, budgets), reps2)
    tests = {k: equivalence(pt, rp, DELTA[k]) for k, (pt, rp) in stats.items()}
    per_budget = {g: {b: set_terms(rr, b, U) for b in budgets} for g, (rr, U) in slope_sets.items()}
    out = dict(per_set=per, per_budget_real=per_budget, replicates={k: list(map(float, rp)) for k, (_, rp) in stats.items()},
               **holm(tests))
    if secondary:                                    # the secondary family: other margins, the same replicates
        out["secondary"] = holm({k: equivalence(stats[k][0], stats[k][1], d) for k, d in secondary.items() if k in stats})
    return out


# ---------------------------------------------------------------- reading the rows
def rows_by_set(root, phases):
    """{set: last row per key} over the phases' results files."""
    from reach.runner.queue import read_results
    by = {}
    for ph in phases:
        for r in read_results(Path(root) / f"data/runs/{ph}/results.jsonl").values():
            by.setdefault(r["set"], []).append(r)
    return by


def main_sets(by, domain):
    return sorted(g for g, rr in by.items() if rr[0]["domain"] == domain and g not in SIZE_CHECKS)


def runs_from_rows(rows, budgets_E):
    """Runs from rows: y at each budget E (evaluations) up to the row's own budget, read by prefix: an exact hit within
    E. A run stopped by its CPU limit counts as a failure, as in the registered analysis."""
    out = []
    for r in rows:
        hit = (r["gen_exact"] + 1) * r["pop_size"] if r.get("exact") else None
        y = {E: int(hit is not None and hit <= E) for E in budgets_E if E <= r["budget_E"]}
        out.append(dict(target_id=r["target_id"], K=r["K"], bits_sem=r["units"]["bits_sem"], y=y, rv=r.get("rv")))
    return out


def levels_U(enum_dir):
    """(U, U_ok) per level from a set's counts.json: all distinct functions, and those finite and not constant."""
    lv = json.loads((Path(enum_dir) / "counts.json").read_text())["levels"]
    return {s["K"]: s["U"] for s in lv}, {s["K"]: s.get("U_finite_nonconst", s["U"]) for s in lv}

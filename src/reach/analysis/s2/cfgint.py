"""H8c as filed in Stage 1: the set x configuration interaction within Delta (REQ-ANA-53; reading D081 e).

Stage 1 (OSF a3spg), as filed: "H8: under each robustness configuration R1-R6, ..., and the set x configuration
interaction is within Delta." (Delta: H2's margin, "the mean log-loss cost of shifting the pooled E0 curve by 1 node
(real) or 2 nodes (Boolean)"); "Equivalence: the 95% CI lies within the margin". Design at 3b8aa5e: "the set x
configuration interaction in the H2-preferred unit stays within Delta_cfg: a configuration shifts reach about equally
in every set."

Per configuration and domain, at E0, on the robustness sets with that configuration (base runs on the same targets and
seeds): each set's base curve logit P = a_g + b_g s (s in the unit); the configuration shifts reach by dK in the unit,
logit P = a_g + b_g (s - dK). For set g, H2's construction: the excess log-loss of g's configuration runs under the
shift fitted on the other sets (leave-one-set-out) over g's own shift (5-fold cross-validation over targets, as H2's
within-set log-loss). The excess is about 0 when the configuration shifts reach equally in every set, and about Delta
when g's shift differs by about the margin's shift. Joint target bootstrap (targets resampled within every set; base
and configuration runs of a target together); a replicate in which a base curve is degenerate is dropped; supported when
the two-sided 95% CI lies inside (-Delta, Delta); p = the larger of the two one-sided inversions (as H4d).
"""
import math
from collections import defaultdict

import numpy as np
from scipy.special import expit

from reach.analysis.k50 import mle
from reach.analysis.s2 import law

SEED = 20261020
FOLDS = 5


def _base(x, y):
    """(a, b) of the base curve; None when degenerate (constant data, no fit, separated data (row 118), or a slope that
    is not negative)."""
    fit = mle(x, y)
    if fit is None:
        return None
    a, b = fit
    return (float(a), float(b)) if np.isfinite(a) and np.isfinite(b) and b < 0 else None


def _ll(o, w, y, dk):
    p = np.clip(expit(o + w * dk), 1e-9, 1 - 1e-9)                       # clipped as H2's log-loss (`invariance._ll`)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _shift(parts, bound):
    """dK maximising the likelihood of logit P = o + w dK over `parts` [(o, w, y)], w > 0, within +-bound: the score
    sum w (y - p) falls in dK, so its root is bracketed and found by Newton steps kept inside the bracket (bisection
    when a step leaves it or the curve is flat). All successes: +bound; all failures: -bound."""
    o = np.concatenate([p[0] for p in parts])
    w = np.concatenate([p[1] for p in parts])
    y = np.concatenate([p[2] for p in parts])

    def score(dk):
        p = expit(o + w * dk)
        return float((w * (y - p)).sum()), float((w * w * p * (1 - p)).sum())
    lo, hi = -bound, bound
    if score(lo)[0] <= 0:
        return lo
    if score(hi)[0] >= 0:
        return hi
    dk = 0.0
    for _ in range(200):
        g, h = score(dk)
        if g > 0:
            lo = dk
        else:
            hi = dk
        new = dk + g / h if h > 0 else float("nan")
        if not (lo < new < hi):
            new = 0.5 * (lo + hi)
        if abs(new - dk) < 1e-10 or hi - lo < 1e-10:
            return new
        dk = new
    return dk


def _arrays(by, unit):
    """{set: {target: (x, y)}} from records."""
    out = {}
    for g, rr in by.items():
        t = defaultdict(list)
        for r in rr:
            t[r["target_id"]].append((float(r["units"][unit]), float(r["y"])))
        out[g] = {k: (np.array([a for a, _ in v]), np.array([b for _, b in v])) for k, v in t.items()}
    return out


def _cat(d, ids):
    return (np.concatenate([d[t][0] for t in ids]), np.concatenate([d[t][1] for t in ids])) if ids else \
        (np.zeros(0), np.zeros(0))


def _excess(base, cfg, ids, rng=None, folds=FOLDS):
    """{set: excess} for one draw: `ids` {set: [(base target key, cfg target key, label)]}; NaN when a base curve is
    degenerate."""
    sets = sorted(ids)
    fit, parts = {}, {}
    span = 1.0
    for g in sets:
        bx, by = _cat(base[g], [b for b, _, _ in ids[g] if b in base[g]])
        fit[g] = _base(bx, by)
        if fit[g] is None:
            return {s: float("nan") for s in sets}
        span = max(span, float(bx.max() - bx.min()))
    for g in sets:
        a, b = fit[g]
        parts[g] = {}
        for _, c, lab in ids[g]:
            x, y = cfg[g][c]
            parts[g][lab] = (a + b * x, np.full(len(x), -b), y)
    bound = 10 * span
    out = {}
    for g in sets:
        own = parts[g]
        if not own:
            out[g] = float("nan")
            continue
        dk_logo = _shift([v for h in sets if h != g for v in parts[h].values()], bound)
        logo = float(np.concatenate([_ll(o, w, y, dk_logo) for o, w, y in own.values()]).mean())
        labs = list(own)
        order = np.arange(len(labs)) if rng is None else rng.permutation(len(labs))
        fold = {labs[i]: k % folds for k, i in enumerate(order)}
        loss, n = 0.0, 0
        for f in range(folds):
            tr = [own[t] for t in labs if fold[t] != f]
            te = [own[t] for t in labs if fold[t] == f]
            if not tr or not te:
                continue
            dk = _shift(tr, bound)
            ll = np.concatenate([_ll(o, w, y, dk) for o, w, y in te])
            loss += float(ll.sum())
            n += len(ll)
        out[g] = logo - loss / n
    return out


def _shifts(base, cfg, ids):
    sets = sorted(ids)
    fit = {g: _base(*_cat(base[g], [b for b, _, _ in ids[g] if b in base[g]])) for g in sets}
    span = max([1.0] + [float(np.ptp(_cat(base[g], list(base[g]))[0])) for g in sets])
    parts = {g: [((fit[g][0] + fit[g][1] * cfg[g][c][0]), np.full(len(cfg[g][c][0]), -fit[g][1]), cfg[g][c][1])
                 for _, c, _ in ids[g]] for g in sets if fit[g] is not None}
    own = {g: _shift(parts[g], 10 * span) if parts.get(g) else float("nan") for g in sets}
    logo = {g: _shift([v for h in parts if h != g for v in parts[h]], 10 * span) if parts.get(g) and len(parts) > 1
            else float("nan") for g in sets}
    return own, logo


def h8c(base_by_set, cfg_by_set, unit, delta, boot=2000, seed=SEED, folds=FOLDS):
    """H8c for one configuration in one domain: `base_by_set` and `cfg_by_set` {set: records at E0}, paired by target
    and seed by the caller. Returns {set: dict(excess, lo, hi, p, dropped, estimable, holds, untestable, shift_own,
    shift_logo, unit, delta)}."""
    sets = sorted(g for g in cfg_by_set if cfg_by_set[g])
    if len(sets) < 2:
        return {g: dict(excess=float("nan"), lo=float("nan"), hi=float("nan"), p=None, dropped=None, estimable=False,
                        holds=False, untestable="fewer than 2 robustness sets with this configuration", unit=unit,
                        delta=delta, shift_own=float("nan"), shift_logo=float("nan")) for g in sets}
    base, cfg = _arrays({g: base_by_set.get(g, []) for g in sets}, unit), _arrays({g: cfg_by_set[g] for g in sets}, unit)
    tids = {g: sorted(cfg[g]) for g in sets}
    point = _excess(base, cfg, {g: [(t, t, t) for t in tids[g]] for g in sets}, None, folds)
    own, logo = _shifts(base, cfg, {g: [(t, t, t) for t in tids[g]] for g in sets})
    rng = np.random.default_rng(seed)
    reps = {g: [] for g in sets}
    for _ in range(boot):
        ids = {g: [(tids[g][j], tids[g][j], f"{tids[g][j]}#{k}") for k, j in
                   enumerate(rng.integers(0, len(tids[g]), len(tids[g])))] for g in sets}
        for g, v in _excess(base, cfg, ids, rng, folds).items():
            reps[g].append(v)
    out = {}
    for g in sets:
        ok = [x for x in reps[g] if np.isfinite(x)]
        lo, hi = np.percentile(ok, [2.5, 97.5]) if ok else (float("nan"),) * 2
        est = len(ok) >= 0.9 * boot and math.isfinite(point[g])
        p = max(law.p_bound(ok, -delta, "above", 2), law.p_bound(ok, delta, "below", 2)) if est else None
        out[g] = dict(excess=point[g], lo=float(lo), hi=float(hi), p=p, dropped=boot - len(ok), estimable=bool(est),
                      holds=bool(est and -delta < lo and hi < delta), untestable=None if est else "not estimable",
                      unit=unit, delta=delta, shift_own=own[g], shift_logo=logo[g], _reps=ok)
    return out

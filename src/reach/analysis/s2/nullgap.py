"""GP against the exact blind-search null: H6b, H6c and H8b as filed in Stage 1 (REQ-ANA-50 to 52; readings D081 b-d).

Stage 1 (OSF a3spg), as filed: "H6b: the GP-null gap changes with size (size x method interaction). H6c (secondary):
the gap at E0 has opposite signs in the two domains."; "H8: under each robustness configuration R1-R6, H1 holds, the
sign of GP minus null is unchanged, ..."; "Directional effects: the 95% CI excludes 0 in the predicted direction";
"P-values for Holm come from the bootstrap by CI inversion"; "Bootstrap replicates with degenerate fits are dropped for
that quantity; if more than 10% are dropped, the CI is not estimable and the hypothesis is not supported."

The null is the primary one (`null`, enumeration over distinct functions in size order): with E evaluations a target of
minimum size K is hit with probability clip((E - U(<=K-1)) / U(K), 0, 1); its reach in bits_sem is log2 E.
- H6b (D081 b): per target, the gap in recovery = GP's share of exact runs at E0 minus the null's hit probability;
  regressed on size in nodes with an intercept per set (the size x method interaction of a pooled linear model with
  set x method intercepts); two-sided target-bootstrap CI of the slope (targets resampled within sets).
- H6c (D081 c): per domain, the gap in reach = mean over its main sets of GP's K50 in bits_sem at E0 minus log2 E0;
  "opposite signs": one domain's two-sided CI below 0 and the other's above 0; p = the better of the two orientations,
  each the larger of its two inversions (intersection-union; the two orientations are disjoint, each side 2.5%).
- H8b (D081 d): per set and configuration, at E0 and at 4 x E0 (the robustness runs' budgets, as H8a), GP's K50 gap in
  bits_sem under the configuration and under the base on the same targets and seeds (paired target bootstrap); "the
  sign is unchanged": both CIs on the same side of 0 (p as H6c, same side); member p = the larger over the budgets.
"""
import math
from collections import defaultdict

import numpy as np

from reach.analysis.s2 import law, null
from reach.analysis.s2.load import E0

SEED = 20261020


def null_p(levels, K, proven, E):
    """The primary null's hit probability for a target of size K with E evaluations. A constructed target (not
    proven) lies above the enumeration ceiling: 0 when U(<=ceiling) >= E (the null exhausts its budget below it),
    otherwise NaN (not computable from the counts)."""
    ks = {st["K"] for st in levels}
    if proven and K in ks:
        return float(null.p_hit(levels, K, E))
    top = max(st["U_cum"] for st in levels)
    return 0.0 if top >= E else float("nan")


def _p_dir(reps, direction):
    return law.p_bound(reps, 0.0, direction, 2)


def p_opposite(a, b):
    """'a' and 'b' have opposite signs: the better orientation (a below 0 and b above, or the reverse), each the larger
    of its two two-sided CI inversions against 0."""
    return min(max(_p_dir(a, "below"), _p_dir(b, "above")), max(_p_dir(a, "above"), _p_dir(b, "below")))


def p_same(a, b):
    """'a' and 'b' have the same sign: both CIs below 0 or both above 0 (the better side)."""
    return min(max(_p_dir(a, "below"), _p_dir(b, "below")), max(_p_dir(a, "above"), _p_dir(b, "above")))


def _ci(reps):
    ok = [x for x in reps if np.isfinite(x)]
    lo, hi = np.percentile(ok, [2.5, 97.5]) if ok else (float("nan"),) * 2
    return ok, float(lo), float(hi)


def _within_slope(rows):
    """Slope of gap on K with an intercept per set (within-set centring)."""
    by = defaultdict(list)
    for g, K, gap in rows:
        by[g].append((K, gap))
    sxy = sxx = 0.0
    for v in by.values():
        x, y = np.array([a for a, _ in v], float), np.array([b for _, b in v], float)
        x, y = x - x.mean(), y - y.mean()
        sxy += float((x * y).sum())
        sxx += float((x * x).sum())
    return sxy / sxx if sxx > 0 else float("nan")


def h6b(recs_by_set, levels_by_set, E=E0, boot=2000, seed=SEED):
    """H6b for one domain: `recs_by_set` {set: GP records at budget E (E/E0 x E0)}, `levels_by_set` {set: the set's
    enumeration counts}. A target whose null probability cannot be computed is left out and counted."""
    per, left = {}, 0
    for g in sorted(recs_by_set):
        by = defaultdict(list)
        for r in recs_by_set[g]:
            by[r["target_id"]].append(r)
        rows = []
        for t in sorted(by):
            rr = by[t]
            K = rr[0]["K"]
            proven = rr[0].get("K_proven") is not None and rr[0]["K_proven"] == K
            pn = null_p(levels_by_set[g], K, proven, E)
            if not math.isfinite(pn):
                left += 1
                continue
            rows.append((t, K, float(np.mean([r["y"] for r in rr])) - pn))
        per[g] = rows
    flat = [(g, K, gap) for g, rows in per.items() for _, K, gap in rows]
    theta = _within_slope(flat)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(boot):
        draw = []
        for g, rows in per.items():
            if rows:
                draw += [(g, rows[j][1], rows[j][2]) for j in rng.integers(0, len(rows), len(rows))]
        reps.append(_within_slope(draw))
    ok, lo, hi = _ci(reps)
    estimable = len(ok) >= 0.9 * boot and math.isfinite(theta)
    p = min(_p_dir(ok, "below"), _p_dir(ok, "above")) if ok else float("nan")
    per_set = {}
    for g, rows in per.items():
        lv = defaultdict(list)
        for _, K, gap in rows:
            lv[K].append(gap)
        levels = [[K, float(np.mean(v)), len(v)] for K, v in sorted(lv.items())]
        signs = [(K, np.sign(m)) for K, m, _ in levels if m != 0]
        per_set[g] = dict(levels=levels, sign_changes=[[a, b] for (a, sa), (b, sb) in zip(signs, signs[1:]) if sa != sb])
    return dict(theta=theta, lo=lo, hi=hi, p=p if estimable else None, dropped=boot - len(ok), estimable=bool(estimable),
                untestable=None if estimable else "not estimable", unit="nodes", budget_evaluations=E,
                targets=len(flat), left_out=left, sets=sorted(per), per_set=per_set, _reps=ok)


def _k50_bits(rr):
    return law._k50(rr, "bits_sem") if rr else float("nan")


def domain_gap(recs_by_set, E=E0, boot=2000, seed=SEED):
    """H6c's gap for one domain: mean over its sets of K50 in bits_sem (records at budget E) minus log2 E; targets
    resampled within sets; a replicate in which any set's K50 is undefined is dropped."""
    nul = math.log2(E)
    by = {g: law._by_target(rr) for g, rr in sorted(recs_by_set.items())}
    point = {g: _k50_bits([r for t in b for r in b[t]]) for g, b in by.items()}
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(boot):
        ks = []
        for g, b in by.items():
            ids = list(b)
            ks.append(_k50_bits([r for j in rng.integers(0, len(ids), len(ids)) for r in b[ids[j]]]))
        reps.append(float(np.mean(ks)) - nul if all(np.isfinite(ks)) else float("nan"))
    ok, lo, hi = _ci(reps)
    gap = float(np.mean(list(point.values()))) - nul if by and all(np.isfinite(list(point.values()))) else float("nan")
    return dict(gap=gap, lo=lo, hi=hi, null_bits_sem=nul, dropped=boot - len(ok),
                estimable=bool(len(ok) >= 0.9 * boot and math.isfinite(gap)), sets=sorted(by),
                per_set={g: k - nul for g, k in point.items()}, _reps=ok)


def h6c(gaps):
    """H6c from `domain_gap` per domain ({"real": ..., "bool": ...}); one claim, counted in both domains' families."""
    for d in ("real", "bool"):
        if d not in gaps:
            return dict(p=None, untestable=f"missing: no data in {d}", orientation=None)
        if not gaps[d]["estimable"]:
            return dict(p=None, untestable=f"not estimable in {d}", orientation=None)
    a, b = gaps["real"]["_reps"], gaps["bool"]["_reps"]
    p1 = max(_p_dir(a, "below"), _p_dir(b, "above"))
    p2 = max(_p_dir(a, "above"), _p_dir(b, "below"))
    return dict(p=min(p1, p2), untestable=None,
                orientation="real below 0, Boolean above 0" if p1 <= p2 else "real above 0, Boolean below 0",
                gaps={d: {k: v for k, v in gaps[d].items() if not k.startswith("_")} for d in ("real", "bool")})


H8_BUDGETS = (1, 4)


def h8b(base, cfg, budgets=H8_BUDGETS, E=E0, boot=2000, seed=SEED):
    """H8b for one (set, configuration): `base` the base configuration's records and `cfg` the configuration's, on the
    same targets and seeds (paired by the caller), at `budgets` x E0."""
    bb, cc = law._by_target(base), law._by_target(cfg)
    tids = sorted(cc)
    for b in budgets:
        if not any(r["budget"] == b for r in cfg) or not any(r["budget"] == b for r in base):
            return dict(p=None, untestable=f"missing at budget {b}", budgets={}, targets=len(tids))
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, len(tids), len(tids)) for _ in range(boot)]
    out = {}
    for b in budgets:
        nul = math.log2(b * E)

        def gap(by, ids):
            return _k50_bits([r for t in ids for r in by.get(t, ()) if r["budget"] == b]) - nul
        rb, rc = [], []
        for d in draws:
            ids = [tids[j] for j in d]
            x, y = gap(bb, ids), gap(cc, ids)
            if np.isfinite(x) and np.isfinite(y):
                rb.append(x)
                rc.append(y)
        pb, pc = gap(bb, tids), gap(cc, tids)
        _, blo, bhi = _ci(rb)
        _, clo, chi = _ci(rc)
        est = len(rb) >= 0.9 * boot and np.isfinite(pb) and np.isfinite(pc)
        out[b] = dict(base_gap=pb, base_lo=blo, base_hi=bhi, config_gap=pc, config_lo=clo, config_hi=chi,
                      null_bits_sem=nul, dropped=boot - len(rb), estimable=bool(est),
                      p=p_same(rb, rc) if est else None)
    bad = [b for b in budgets if out[b]["p"] is None]
    return dict(p=None if bad else max(out[b]["p"] for b in budgets),
                untestable=f"not estimable at budget {bad[0]}" if bad else None, budgets=out, targets=len(tids))

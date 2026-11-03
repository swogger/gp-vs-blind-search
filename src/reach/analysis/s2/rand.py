"""The random-sampling arm of the calibration (REQ-ANA-40 to 42; D080 a-c): records from `s2_cal_rand`, H2r and the
H5 contrast B - random sampling.

Stage 1 (OSF a3spg), as filed: "s2_cal_rand 48 (4 x 250,000 random trees per full set)"; "H2r: random sampling's hit
probability is invariant in bits_uniform (same criterion as H2)"; "H5: ... B minus random sampling within +-1 node or
positive". Readings (D080): the hit probability is P(hit within E0 draws) = 1 - (1 - H/N)^E0 with H hits and N draws
summed over the set's parts (one draw is one evaluation); one record per target, the probability as its response.

A part stopped by its CPU limit without a result counts as 250,000 draws without a hit (primary: "Runs stopped by
their CPU limit count as failures") and is left out with censored_as_failure=False (the sensitivity analysis). Error
rows are left out here and listed by `termination.exclusions`.
"""
import math
from collections import defaultdict

from reach.analysis.s2.load import E0

DRAWS_PER_PART = 250_000                # filed: "4 x 250,000 random trees per full set"
CONTRAST = "B-RS"


def records(rows, targets, censored_as_failure=True):
    """{set: [record]}: one record per target of the set's target file (`targets`: {(set, target_id): target}), with
    set, target_id, algorithm 'RS', seed None, K (K_construction), K_proven, units (nodes = K), hits, draws, y = P(hit
    within E0) and censored (a part without a result was counted as draws without a hit). Also returns the number of
    parts counted as failures (`censored_parts`) and left out (`left_out_parts`)."""
    hits, draws, cens = defaultdict(lambda: defaultdict(int)), defaultdict(int), defaultdict(int)
    n_cens = n_out = 0
    for r in rows:
        g = r.get("set") or (r.get("meta") or {}).get("set")
        if not g or r.get("terminal_state") == "error":
            continue
        if r.get("hits") is None or r.get("n") is None:                 # stopped without a result line
            if censored_as_failure:
                draws[g] += DRAWS_PER_PART
                cens[g] += 1
                n_cens += 1
            else:
                n_out += 1
            continue
        draws[g] += int(r["n"])
        for tid, h in r["hits"].items():
            hits[g][tid] += int(h)
    out = {}
    for g in sorted(draws):
        recs = []
        for (gg, tid), t in sorted(targets.items()):
            if gg != g or draws[g] == 0:
                continue
            K = t.get("K_construction", t.get("K"))
            h = hits[g].get(tid, 0)
            y = -math.expm1(E0 * math.log1p(-h / draws[g])) if h < draws[g] else 1.0
            recs.append(dict(set=g, target_id=tid, algorithm="RS", seed=None, config="base", K=K,
                             K_proven=t.get("K_proven"), units=dict(t.get("units") or {}, nodes=float(K)), hits=h,
                             draws=draws[g], y=y, budget=1, censored=cens[g] > 0))
        out[g] = recs
    return dict(by_set=out, censored_parts=n_cens, left_out_parts=n_out)


def h2r(recs_by_set, sets, delta, units, boot, alpha_unit="bits_uniform"):
    """H2r: H2's invariance test on the random-sampling records of `sets`; the decision reads `alpha_unit` only
    (bits_uniform, as filed); the other units are reported beside it."""
    from reach.analysis.s2 import invariance
    use = [g for g in sets if recs_by_set.get(g)]
    if len(use) < 3:
        return dict(status=f"not testable: random-sampling records for {len(use)} sets (at least 3 needed)",
                    sets=use, holds=False)
    recs = [r for g in use for r in recs_by_set[g]]
    inv = invariance.invariance_test(recs, list(units), delta, boot=boot)
    v = inv[alpha_unit]
    return dict(status="tested", sets=use, unit=alpha_unit, delta=delta, units=inv, excess=v["excess"],
                upper95=v["upper95"], holds=bool(v["upper95"] <= delta), targets=len(recs),
                note="one record per target: P(hit within E0 draws); decided in bits_uniform only (filed)")


def b_minus_rs(b_recs, rs_recs, boot, margin):
    """H5, B - random sampling: K50(B) - K50(RS) in nodes at E0 on the targets with both, paired by target (paired
    target bootstrap); p inverts the two-sided 95% CI against -margin from above (as HC - A; D078 reading 3)."""
    from reach.analysis.s2 import contrasts, law
    both = [dict(r, algorithm="B") for r in b_recs] + list(rs_recs)
    c = contrasts.paired_k50_difference(both, "B", "RS", boot=boot, keep_reps=True)
    c["p"] = law.p_bound(c.pop("reps"), -margin, "above", 2)
    return c

"""The Holm families filed in Stage 1 other than H1 (REQ-ANA-29; guardrail row 96; readings for the author: D078).

Stage 1 (OSF a3spg), Inference criteria, as filed: "Directional effects: the 95% CI excludes 0 in the predicted
direction. Equivalence: the 95% CI lies within the margin. Holm correction within families declared now, each counting
untestable members: per domain, H1 cells; H2 units (4); H4a-f; H5 contrasts x sets; H6a-c; H8 configurations x sets
(18); H9a configurations x sets (6). P-values for Holm come from the bootstrap by CI inversion."

H1 is built in `law.holm_cells` (D077); H9a's members are H1-type cells and use the same function. The other families are
built here from `confirm.build`'s results: each member's p inverts its registered bound (`law.p_bound`), Holm runs per
domain over the declared members (`contrasts.holm`), and an untestable member (no data, not estimable, saturated, or a
test not built yet) counts in m and is never supported. alpha = 0.05 (`confirm.REGISTERED["alpha"]`).

Member kinds: 'reject' (the hypothesis is the alternative; supported when the Holm-adjusted p < alpha) and 'retain'
(H4a, "the clustered CI of a quadratic term covers 0": supported when the Holm-adjusted p >= alpha).
"""
import json
import math

import numpy as np

H2_UNITS = ("nodes", "bits_uniform", "bits_syn", "bits_sem")
H4_MEMBERS = {"real": ("H4a", "H4b", "H4c", "H4d", "H4e", "H4f"),
              "bool": ("H4a", "H4c", "H4d", "H4e", "H4f")}             # "H4b: in the real domain"
H5_CONTRASTS = ("A-B", "HC-A", "B-RS")
H5_MARGIN = 1.0                                                         # "within +-1 node or positive"
H6_MEMBERS = ("H6a", "H6b", "H6c")
H8_CONFIGS = ("R1", "R2", "R3", "R4", "R5", "R6")
H8_BUDGETS = (1, 4)                                                     # the robustness runs go to 4 x E0
OPERON_CONFIGS = ("symbolic", "default")
NOT_BUILT = "not built"
MISSING = "missing: no data"
TESTS = {"H2": "one-sided 95% bound of the mean excess below Delta (bootstrap)",
         "H4a": "clustered Wald CI of the quadratic term covers 0 (retain)",
         "H4b": "one-sided 95% bound of the pooled bits_sem rate below 1 (bootstrap)",
         "H4c": "95% CI of every pair of per-set bits_sem slopes within the margin (largest pair p; bootstrap)",
         "H4d": "95% CI of kappa/alpha minus the measured slope within the margin, in every set (largest p; bootstrap)",
         "H4e": "95% CI of the share of cells where restarts help exactly when kappa < 1 at or above 0.75 (bootstrap)",
         "H4f": "measured best run length among the model's best grid lengths (argmax of (B/L)(L - t0)^kappa) in "
                "the central 95% of replicates, in every cell (retain; smallest p)",
         "A-B": "95% CI of K50(A) - K50(B) above 0 (paired bootstrap)",
         "HC-A": "95% CI of K50(HC) - K50(A) above -1 node (paired bootstrap)",
         "B-RS": "95% CI of K50(B) - K50(RS) above -1 node (paired bootstrap)",
         "H6a": "95% CI of the pooled bits_sem rate excludes 1 (two-sided; bootstrap)",
         "H6b": "95% CI of the slope of the per-target GP - null recovery gap on size (nodes, set intercepts) "
                "excludes 0 (two-sided; bootstrap)",
         "H6c": "gap in reach at E0 (bits_sem): one domain's 95% CI below 0 and the other's above 0 (better "
                "orientation, larger p; bootstrap)",
         "H8": "H8a, H8b and H8c all hold (largest part's p); H8a: slope CI below 0 at E0 and 4 x E0; H8b: GP - null "
               "gap CIs of base and configuration on the same side of 0 at E0 and 4 x E0; H8c: 95% CI of the excess "
               "log-loss of the other sets' reach shift within +-Delta",
         "H9a": "slope CI below 0 (target bootstrap), as H1"}


def p_differs(reps, null):
    """H6a ("differs from"): the smallest alpha at which the two-sided percentile CI excludes `null` on either side."""
    from reach.analysis.s2.law import p_bound
    return min(p_bound(reps, null, "below", 2), p_bound(reps, null, "above", 2))


def holm_family(members, alpha):
    """Holm over one family (one domain). `members`: dicts with id, kind ('reject' or 'retain'), p and untestable (a
    reason or None). A member without a finite p is untestable ('not estimable' unless a reason is given); every member
    counts in m. Adds p_holm, supported and status to each member; returns the family's summary."""
    from reach.analysis.s2 import contrasts
    ids = [json.dumps(mm["id"]) for mm in members]
    if len(set(ids)) != len(ids):
        raise ValueError("Holm family: a member appears twice")
    for mm in members:
        if mm.get("untestable") is None and (mm.get("p") is None or not math.isfinite(mm["p"])):
            mm["untestable"] = "not estimable"
    m = len(members)
    adj = contrasts.holm([None if mm.get("untestable") else mm["p"] for mm in members], m=m)
    for mm, q in zip(members, adj):
        mm["p_holm"] = q
        if q is None:
            mm["supported"], mm["status"] = False, f"untestable: {mm['untestable']} (not supported)"
            continue
        if mm["kind"] == "reject":
            mm["supported"] = bool(q < alpha)
        elif mm["kind"] == "retain":
            mm["supported"] = bool(q >= alpha)
        else:
            raise ValueError(f"unknown member kind {mm['kind']!r}")
        mm["status"] = "supported" if mm["supported"] else "not supported"
    reasons = {}
    for mm in members:
        if mm["p_holm"] is None:
            reasons[mm["untestable"]] = reasons.get(mm["untestable"], 0) + 1
    sup = sum(mm["supported"] for mm in members)
    unt = sum(mm["p_holm"] is None for mm in members)
    return dict(alpha=alpha, m=m, supported=sup, not_supported=m - sup - unt, untestable=unt, untestable_reasons=reasons,
                members=members)


def resolution(m, sided, boot, alpha):
    """Can B replicates resolve Holm's smallest threshold alpha/m? A percentile bound at level 1 - alpha/m has
    B alpha / (m sides) replicates beyond it; below 1 the decision rests on the one or two most extreme replicates
    (interpolated). Also: whether the conventional bootstrap p (r + 1)/(B + 1), doubled for a two-sided CI, could reach
    the threshold at all. sided None: an analytic p (no bootstrap limit)."""
    thr = alpha / m if m else float("nan")
    if sided is None:
        return dict(threshold=thr, boot=boot, sided=None, replicates_beyond=None, resolvable=True,
                    p_min_conventional=None, conventional_reaches=True)
    beyond = boot * thr / sided
    pmin = sided / (boot + 1)
    return dict(threshold=thr, boot=boot, sided=sided, replicates_beyond=beyond, resolvable=bool(beyond >= 1),
                p_min_conventional=pmin, conventional_reaches=bool(pmin < thr))


def conjunction(parts):
    """An H8 member holds when all its parts hold (intersection-union): p = the largest part's p. `parts`: {name: (p,
    reason)}; a part without a p makes the member untestable, with every such part named."""
    why = [f"{k} {r}" for k, (p, r) in parts.items() if p is None]
    if why:
        return None, ", ".join(why)
    return max(p for p, _ in parts.values()), None


def h8a_part(cells, key, budgets=H8_BUDGETS):
    """H8a for (set, config): "H1 holds" in its cells at E0 and 4 x E0 (as H1: slope CI below 0); p = the larger."""
    g, cfg = key
    ps = []
    for b in budgets:
        c = next((c for c in cells if c["set"] == g and c["config"] == cfg and c["budget"] == b), None)
        if c is None:
            return None, f"missing at budget {b}"
        if c.get("saturated"):
            return None, f"saturated at budget {b}"
        if not c.get("estimable") or not math.isfinite(c.get("p_below0", float("nan"))):
            return None, f"not estimable at budget {b}"
        ps.append(c["p_below0"])
    return max(ps), None


def declared():
    """The families as filed, per domain, from the registered set lists: {family: {domain: [member ids]}}."""
    from reach import grammar as G
    from reach.paths import ROOT
    from reach.runner.phases import CAL_SETS, FULL_SETS, OPERON_SETS
    dom = {d: [g for g in FULL_SETS if G.get(g).kind == d] for d in ("real", "bool")}
    robust = json.loads((ROOT / "data/registration/levels.json").read_text())["_robust"]
    return dict(H2={d: [(u,) for u in H2_UNITS] for d in ("real", "bool")},
                H4={d: [(h,) for h in H4_MEMBERS[d]] for d in ("real", "bool")},
                H5={d: [(g, c) for g in dom[d] for c in H5_CONTRASTS] for d in ("real", "bool")},
                H6={d: [(h,) for h in H6_MEMBERS] for d in ("real", "bool")},
                H8={d: [(g, c) for g in robust[d] for c in H8_CONFIGS] for d in ("real", "bool")},
                H9a={"real": [(g, "operon", c, 16) for g in CAL_SETS if g in OPERON_SETS for c in OPERON_CONFIGS]})


def _member(i, kind, p, untestable=None, sided=2, test=""):
    return dict(id=list(i), kind=kind, p=p, untestable=untestable, sided=sided, test=test)


def _h2(res, d, i):
    v = res["h2"].get(d, {}).get("units", {}).get(i[0])
    if v is None:
        return _member(i, "reject", None, MISSING, 1, TESTS["H2"])
    return _member(i, "reject", v["p_below_delta"], None, 1, TESTS["H2"])


SIDED = {"H4a": None, "H4b": 1}                  # others two-sided (H4c-f, H6a-c, H8)
RUNTIME_OFF = "not run in this build (runtime_model=False)"


def _h4(res, d, i):
    h = res["h4"].get(d)
    if h is None:
        return _member(i, "retain" if i[0] in ("H4a", "H4f") else "reject", None, MISSING, SIDED.get(i[0], 2))
    if i[0] == "H4a":
        return _member(i, "retain", h["curvature_nodes"]["p_wald"], None, None, TESTS["H4a"])
    if i[0] == "H4b":
        rc = h["rate_bits_sem"]
        return _member(i, "reject", rc["p_below1"], None if rc["estimable"] else "not estimable", 1, TESTS["H4b"])
    if i[0] == "H4c":                                                    # D081 a
        c = h.get("h4c")
        if c is None:
            return _member(i, "reject", None, MISSING, 2, TESTS["H4c"])
        return _member(i, "reject", c["p"], c["untestable"], 2, TESTS["H4c"])
    if i[0] in ("H4d", "H4e", "H4f"):                                    # the run-time model (D080 f-h)
        kind = "retain" if i[0] == "H4f" else "reject"
        if "note" in res.get("h4_model", {}):
            return _member(i, kind, None, RUNTIME_OFF, 2, TESTS[i[0]])
        t = res.get("h4_model", {}).get(d)
        if t is None:
            return _member(i, kind, None, MISSING, 2, TESTS[i[0]])
        v = t["tests"][i[0]]
        return _member(i, kind, v["p"], v["untestable"], 2, TESTS[i[0]])
    return _member(i, "reject", None, NOT_BUILT)


def _h5(res, d, i):
    g, c = i
    v = res["h5"].get(g, {}).get(c)
    if v is None:
        return _member(i, "reject", None, MISSING, 2, TESTS[c])
    return _member(i, "reject", v["p"], None if v["estimable"] else "not estimable", 2, TESTS[c])


def _h6(res, d, i):
    if i[0] == "H6a":
        h = res["h4"].get(d)
        if h is None:
            return _member(i, "reject", None, MISSING, 2, TESTS["H6a"])
        rc = h["rate_bits_sem"]
        return _member(i, "reject", rc["p_differs1"], None if rc["estimable"] else "not estimable", 2, TESTS["H6a"])
    if i[0] == "H6b":                                                    # D081 b
        v = res.get("h6b", {}).get(d)
        if v is None:
            return _member(i, "reject", None, MISSING, 2, TESTS["H6b"])
        return _member(i, "reject", v["p"], v["untestable"], 2, TESTS["H6b"])
    if i[0] == "H6c":                                                    # D081 c: one claim, in both families
        v = res.get("h6c") or dict(p=None, untestable=MISSING)
        return _member(i, "reject", v["p"], v["untestable"], 2, TESTS["H6c"])
    return _member(i, "reject", None, NOT_BUILT)


def _part(v):
    """(p, reason) of an H8b or H8c result; missing when the member has no data."""
    if v is None:
        return None, "missing"
    return (v["p"], None) if v["p"] is not None else (None, v["untestable"] or "not estimable")


def _h8(res, d, i):
    key = f"{i[0]}|{i[1]}"
    parts = {"H8a": h8a_part(res["h8"], i), "H8b": _part(res.get("h8b", {}).get(key)),
             "H8c": _part(res.get("h8c", {}).get(key))}
    p, why = conjunction(parts)
    mm = _member(i, "reject", p, why, 2, TESTS["H8"])
    mm.update(p_h8a=parts["H8a"][0], p_h8b=parts["H8b"][0], p_h8c=parts["H8c"][0])
    return mm


def _observed(res, domain_of):
    """The DRYRUN's families: the declared members restricted to what the data hold (as H1's observed family)."""
    h2 = {d: [(u,) for u in H2_UNITS] for d in res["h2"]}
    h4 = {d: [(h,) for h in H4_MEMBERS[d]] for d in res["h4"]}
    h6 = {d: [(h,) for h in H6_MEMBERS] for d in sorted(set(res["h4"]) | set(res.get("h6b", {})))}
    h5, h8 = {}, {}
    for g in sorted(res["h5"]):
        h5.setdefault(domain_of(g), []).extend((g, c) for c in H5_CONTRASTS)
    for g, c in sorted({(c["set"], c["config"]) for c in res["h8"]}):
        h8.setdefault(domain_of(g), []).append((g, c))
    return dict(H2=h2, H4=h4, H5=h5, H6=h6, H8=h8)


def build(res, domain_of, declared_mode, alpha, boot):
    """res["holm"]: {family: {domain: summary}} for H2, H4, H5, H6, H8 (here) and H9a (law.holm_cells on res["h9a"])."""
    from reach.analysis.s2 import law
    fams = declared() if declared_mode else _observed(res, domain_of)
    make = dict(H2=_h2, H4=_h4, H5=_h5, H6=_h6, H8=_h8)
    out = {}
    for f, fn in make.items():
        out[f] = {}
        for d, ids in fams[f].items():
            members = [fn(res, d, i) for i in ids]
            s = holm_family(members, alpha)
            sides = [mm["sided"] for mm in members if mm["sided"] is not None]      # not built: two-sided assumed
            out[f][d] = dict(s, family="declared" if declared_mode else "observed",
                             resolution=resolution(s["m"], max(sides) if sides else None, boot, alpha))
    h9 = law.holm_cells(res["h9a"], domain_of, alpha, declared=fams["H9a"]["real"] if declared_mode else None) \
        if (res["h9a"] or declared_mode) else {}
    out["H9a"] = {}
    for d, s in h9.items():
        mem = [dict(id=[c["set"], c["algorithm"], c["config"], c["budget"]], kind="reject", p=c["p_below0"],
                    p_holm=c["p_holm"], supported=c["holm_supported"], status=c["holm_status"], test=TESTS["H9a"])
               for c in res["h9a"] if domain_of(c["set"]) == d]
        mem += [dict(id=list(k), kind="reject", p=None, p_holm=None, supported=False,
                     status=f"untestable: {MISSING} (not supported)", test=TESTS["H9a"]) for k in s["missing"]]
        out["H9a"][d] = dict(s, members=mem, resolution=resolution(s["m"], 2, boot, alpha))
    return out


def fmt_p(x):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.4f}"

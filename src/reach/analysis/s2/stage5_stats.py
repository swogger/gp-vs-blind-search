"""Stage 5 test statistics (docs/design/stage5_analysis.md; REQ-ANA-123 to REQ-ANA-128) on arms the driver
(`stage5_report`) has loaded and checked; no file is read here.

An arm of one set is {target_id: (bits, p, n)}: the target's registered size (bits_sem), its success probability and
the runs behind it. One run of L generations: the share of its runs exact before L (`load.outcome`; an error row left
out of n, a CPU stop or a kill a failure). R restarts: 1 - C(n - s, R)/C(n, R) (Stage 4's `alloc.p_at_least_one`). A
twin: its hit probability at the target's level (n = 0). Every statistic is a function of `pick` ({set: [target ids]},
resampled within sets with duplicates kept, `stage5.target_bootstrap`) and returns, per test, its value under the
primary estimator (Rule S2: `alloc._k50` with Stage 4's bracketing) and under the isotonic reach (`iso.reach`), so that
both estimators read the same replicates. H16's primary is the twin reach interpolated over every enumerated level
(`h6d.k50_interp`), with the logistic twin and the isotonic reach beside.
"""
import math
from collections import defaultdict

import numpy as np

from reach.analysis.s2 import alloc, h6d, iso, remedy, stage5
from reach.analysis.s2.load import outcome

E0 = 20_000
E16 = 16 * E0
NULL_E0, NULL_16 = math.log2(E0), math.log2(E16)
POPS = (25, 100, 1000)
NAN = float("nan")
DEN_TOL = 1e-9         # H15-2: a mean GP gain at or below 0 (within float noise) is not kept

# The twelve registered tests (draft form, Hypotheses and Inference criteria): kind, margin and family.
TESTS = {
    "H12-1": dict(family="H12", kind="tost", delta=1.0),
    "H12-2": dict(family="H12", kind="above", margin=0.0),
    "H12-3": dict(family="H12", kind="tost", delta=1.0),
    "H13-1": dict(family="H13", kind="tost", delta=1.0),
    "H13-2": dict(family="H13", kind="tost", delta=1.0),
    "H14-1": dict(family="H14", kind="above", margin=-1.0),
    "H14-2": dict(family="H14", kind="above", margin=-1.0),
    "H15-1": dict(family="H15", kind="above", margin=-1.0),
    "H15-2": dict(family="H15", kind="above", margin=0.5),
    "H16-1": dict(family="H16", kind="conj_tost", delta=1.0),
    "H16-2": dict(family="H16", kind="tost", delta=0.25, center=1.0),
    "H11-2L": dict(family="H11-2L", kind="conj_above", margin=2.0),
}
FAMILIES = {f: tuple(k for k, v in TESTS.items() if v["family"] == f)
            for f in ("H12", "H13", "H14", "H15", "H16", "H11-2L")}


# ---------------------------------------------------------------- arms and their reach (Rule S2)
def reach(sizes, ps, cap=True):
    """Primary: the logistic K50 on per-target probabilities (`alloc._k50`) with Stage 4's bracketing rule (above the
    top tested size, or +inf: the top, flagged; NaN or -inf: not estimable, NaN). Secondary: the isotonic reach.
    cap=False: no bracketing (Stage 4 read one long run this way, `remedy.h11`: a K50 above the top stays
    extrapolated, +inf stays +inf; guardrail row 206)."""
    sizes, ps = [float(x) for x in sizes], [float(x) for x in ps]
    if not sizes:
        return dict(k50=NAN, raw=NAN, bracketed=False, top=NAN, iso=NAN, iso_flag="empty")
    raw = alloc._k50(sizes, ps)
    top = max(sizes)
    above = bool(cap and not math.isnan(raw) and raw > top)
    if cap:
        k = top if above else (raw if math.isfinite(raw) else NAN)
    else:
        k = raw if not (math.isnan(raw) or raw == -math.inf) else NAN
    r = iso.reach(sizes, ps)
    return dict(k50=float(k), raw=float(raw), bracketed=above, top=top, iso=float(r["reach"]), iso_flag=r["flag"])


def arm_reach(arm, ids, cap=True):
    ids = [t for t in ids if t in arm]
    return reach([arm[t][0] for t in ids], [arm[t][1] for t in ids], cap)


def one_run_arm(rows, L, size_of):
    """{target: (bits, share exact before L, n)}: error rows left out of n; a row shorter than L is refused."""
    by = defaultdict(list)
    for r in rows:
        o = outcome(r, L)
        if o == "beyond":
            raise ValueError(f"{r.get('key')}: a run of {r.get('generations')} generations read at {L}")
        if o != "error":
            by[r["target_id"]].append(o == "success")
    return {t: (float(size_of[t]), float(np.mean(v)), len(v)) for t, v in by.items() if v}


def restart_arm(rows_by_target, size_of, R, L):
    """{target: (bits, P(at least one of R restarts of L generations), n)} from all its runs (Stage 4's rule)."""
    out = {}
    for t, runs in rows_by_target.items():
        s, n = remedy.hits(runs, L)
        if n:
            out[t] = (float(size_of[t]), float(alloc.p_at_least_one(s, n, R)[0]), n)
    return out


def arm_rows(arms):
    """The (set, level, target) rows of {set: arm}, for Rule S1's check."""
    return [(g, bits, t) for g, arm in arms.items() for t, (bits, *_rest) in arm.items()]


def guard(name, a, b):
    """Rule S1 (guardrail row 170): both arms on the identical (set, level, target) rows, else ValueError."""
    return stage5.shared_levels(arm_rows(a), arm_rows(b), name)


def _mean(xs):
    xs = [float(x) for x in xs]
    return float(np.mean(xs)) if xs and all(math.isfinite(x) for x in xs) else NAN


def _ids(pick, arm):
    return [t for t in pick if t in arm]


# ---------------------------------------------------------------- the restart twin (form, Indices; REQ-ANA-126)
def twin_vector(hist, n_prime, U, R):
    """Per level K (sorted U), the expected share of K that R of the target's n' runs without a hit cover
    (`stage5.union_share`); None when n' < R (the target then takes its level's mean)."""
    if n_prime < R:
        return None
    return np.array([stage5.union_share(hist.get(K, [0] * (n_prime + 1)), R, U[K]) for K in sorted(U)], float)


def restart_twin(entries, ids, U):
    """{K: share} from precomputed (own level, vector or None) per target, over the picked ids (duplicates count):
    per own level the mean over its targets with a vector, levels weighted by their picked targets (pooled ones
    included), a level without any vector left out; the rule of `stage5.restart_twin_shares`."""
    by = defaultdict(list)
    for t in ids:
        lv, vec = entries[t]
        by[lv].append(vec)
    acc, weight = np.zeros(len(U)), 0
    for vecs in by.values():
        ok = [v for v in vecs if v is not None]
        if ok:
            acc += len(vecs) * np.mean(ok, axis=0)
            weight += len(vecs)
    keys = sorted(U)
    return {K: (float(acc[i] / weight) if weight else NAN) for i, K in enumerate(keys)}


def _twin_arm(rows, probs, ids):
    """A twin's arm on the picked targets: rows {target: (bits, K)}, probs {K: hit probability}."""
    return {t: (rows[t][0], float(probs.get(rows[t][1], 0.0)), 0) for t in set(ids) if t in rows}


def coverage_probs(cov, ids, U, E, how):
    """Stage 3's twin probabilities c_K from the counted runs of the picked targets ({target: [runs]}, `h6d`'s run
    dicts) at E evaluations; {} when no estimate exists (no run reached a needed prefix)."""
    runs = [r for t in ids for r in cov.get(t, ())]
    lv = h6d.cumulative(runs, E, "lv", how)
    return h6d.twin_probs(lv, U) if np.ndim(lv) else {}


# ---------------------------------------------------------------- statistics per block
def h12_contrast(name, new, base):
    """H12-1, H12-3: the mean over sets of K50(new) - K50(base) at E0 (both arms on the same rows)."""
    sets = sorted(new)

    def stat(pick):
        per = {}
        for g in sets:
            a, b = arm_reach(new[g], pick[g]), arm_reach(base[g], pick[g])
            per[g] = dict(new=a, base=b, d=a["k50"] - b["k50"], d_iso=a["iso"] - b["iso"])
        return dict(values={name: dict(primary=_mean(v["d"] for v in per.values()),
                                       iso=_mean(v["d_iso"] for v in per.values()))},
                    per_set=per, set_level={name: dict(t={g: v["d"] for g, v in per.items()})})
    return stat


def h12_sign(name, arm, base=None):
    """H12-2: the mean over the Boolean sets of K50(BS24) - log2 E0 on proven targets; the change against the base
    arm beside (descriptive)."""
    sets = sorted(arm)

    def stat(pick):
        per = {}
        for g in sets:
            a = arm_reach(arm[g], pick[g])
            per[g] = dict(arm=a, gap=a["k50"] - NULL_E0, gap_iso=a["iso"] - NULL_E0)
            if base is not None:
                b = arm_reach(base[g], pick[g])
                per[g].update(base=b, d=a["k50"] - b["k50"])
        return dict(values={name: dict(primary=_mean(v["gap"] for v in per.values()),
                                       iso=_mean(v["gap_iso"] for v in per.values()))},
                    per_set=per, set_level={name: dict(t={g: v["gap"] for g, v in per.items()})})
    return stat


def h13_domains(name, sf, real, boolean):
    """H13-1: mean K50(SF) over the Boolean sets minus over the real sets, each set on its own S_d."""
    def stat(pick):
        k = {g: arm_reach(sf[g], pick[g]) for g in list(real) + list(boolean)}
        vals = {}
        for est, key in (("primary", "k50"), ("iso", "iso")):
            vals[est] = _mean(k[g][key] for g in boolean) - _mean(k[g][key] for g in real)
        return dict(values={name: vals}, per_set=k,
                    set_level={name: dict(welch=({g: k[g]["k50"] for g in boolean}, {g: k[g]["k50"] for g in real}))})
    return stat


def h13_twin(name, sf, twin_rows, cov, U, gp=None, E=E0):
    """H13-2: the real mean of K50(SF) - K50(twin), the twin's coverage from Stage 3's counted runs of the picked
    targets ({set: {target: [runs]}}, stratified life table at E0) read on S_real's targets (twin_rows {set: {target:
    (bits, K)}}); T = K50(GP) - K50(SF) beside when gp (the base arm on S_real) is given."""
    sets = sorted(sf)

    def stat(pick):
        per = {}
        for g in sets:
            ids = _ids(pick[g], twin_rows[g])
            probs = coverage_probs(cov[g], pick[g], U[g], E, "stratified")
            tw = arm_reach(_twin_arm(twin_rows[g], probs, ids), ids) if probs else reach([], [])
            s = arm_reach(sf[g], pick[g])
            per[g] = dict(sf=s, twin=tw, d=s["k50"] - tw["k50"], d_iso=s["iso"] - tw["iso"])
            if gp is not None:
                p = arm_reach(gp[g], pick[g])
                per[g].update(gp=p, T=p["k50"] - s["k50"], T_iso=p["iso"] - s["iso"])
        out = dict(values={name: dict(primary=_mean(v["d"] for v in per.values()),
                                      iso=_mean(v["d_iso"] for v in per.values()))},
                   per_set=per, set_level={name: dict(t={g: v["d"] for g, v in per.items()})})
        if gp is not None:
            out["T"] = dict(primary=_mean(v["T"] for v in per.values()), iso=_mean(v["T_iso"] for v in per.values()))
        return out
    return stat


def heldout(arms, sets, U=None, E_long=E16):
    """H14-1, H14-2, H15-1, H15-2 and H11-2L on the 55 restart levels, from whichever arms are given:
    restart (32 restarts x 100 generations), r2 (population 1,000 at 16 x E0), one (one run of 100 generations), long
    (one run of 3,200 generations), twin (per target (own level, vector for 32, vector for 1), `twin_vector`), rows
    ({set: {target: (bits, K)}} of the twins), long_cov (the counted long runs, {set: {target: [runs]}})."""
    A = arms

    def stat(pick):
        per, vals, sl = {}, {}, {}
        for g in sets:
            ids = pick[g]
            v = dict(restart=arm_reach(A["restart"][g], ids))
            if "r2" in A:
                v["r2"] = arm_reach(A["r2"][g], ids)
            if "long" in A:                       # one long run unbracketed, as Stage 4's H11-2 (guardrail row 206)
                v["long"] = arm_reach(A["long"][g], ids, cap=False)
            if "twin" in A:
                v["one"] = arm_reach(A["one"][g], ids)
                tids = _ids(ids, A["rows"][g])
                for j, key in ((1, "twin32"), (2, "twin1")):
                    probs = restart_twin({t: (A["twin"][g][t][0], A["twin"][g][t][j]) for t in set(tids)}, tids, U[g])
                    v[key] = arm_reach(_twin_arm(A["rows"][g], probs, tids), tids)
            if "long_cov" in A:
                tids = _ids(ids, A["rows"][g])
                probs = coverage_probs(A["long_cov"][g], ids, U[g], E_long, "never_hit")
                v["twin_long"] = arm_reach(_twin_arm(A["rows"][g], probs, tids), tids) if probs else reach([], [])
            per[g] = v
        for est, key in (("primary", "k50"), ("iso", "iso")):
            def d(a, b, off=0.0):
                return {g: per[g][a][key] - (per[g][b][key] if b else off) for g in sets}
            if "r2" in A:
                vals.setdefault("H14-1", {})[est] = _mean(d("restart", "r2").values())
                vals.setdefault("H14-2", {})[est] = _mean(d("r2", None, NULL_16).values())
                if est == "primary":
                    sl["H14-1"], sl["H14-2"] = dict(t=d("restart", "r2")), dict(t=d("r2", None, NULL_16))
            if "twin" in A:
                g32, g1 = d("restart", "twin32"), d("one", "twin1")
                dg = {g: g32[g] - g1[g] for g in sets}
                vals.setdefault("H15-1", {})[est] = _mean(dg.values())
                if est == "primary":
                    sl["H15-1"] = dict(t=dg)
                    for g in sets:
                        per[g].update(G32=g32[g], G1=g1[g])
            if "long" in A:
                gain = d("restart", "long")
                vals.setdefault("H11-2L", {})[est] = gain
                if est == "primary":
                    sl["H11-2L"] = dict(t=gain)
                if "twin" in A and "long_cov" in A:
                    tg = d("twin32", "twin_long")
                    num, den = _mean(tg.values()), _mean(gain.values())
                    ok = math.isfinite(num) and math.isfinite(den) and den > DEN_TOL
                    vals.setdefault("H15-2", {})[est] = num / den if ok else NAN
                    if est == "primary":
                        sl["H15-2"] = dict(ratio_of_means=dict(numerator=tg, denominator=gain))
        return dict(values=vals, per_set=per, set_level=sl)
    return stat


def h16(cov, rows, U, real, boolean, E=E0):
    """H16-1 and H16-2: the twin reach at E0 per population (cov {pop: {set: {target: [runs]}}}, the counted runs read
    at E0, stratified life table), interpolated over every enumerated level (primary), with the logistic and the
    isotonic reach on the robustness targets (rows {set: {target: (bits, K)}}) beside."""
    sets = list(real) + list(boolean)

    def stat(pick):
        per = {str(p): {} for p in POPS}
        for p in POPS:
            for g in sets:
                probs = coverage_probs(cov[p][g], pick[g], U[g], E, "stratified")
                tids = _ids(pick[g], rows[g])
                r = arm_reach(_twin_arm(rows[g], probs, tids), tids) if probs else reach([], [])
                per[str(p)][g] = dict(primary=h6d.k50_interp(probs, U[g]) if probs else NAN, logistic=r["k50"],
                                      iso=r["iso"])
        vals, sl = {"H16-1": {}, "H16-2": {}}, {}
        for est in ("primary", "logistic", "iso"):
            diff = {str(p): _mean(per[str(p)][g][est] for g in boolean) - _mean(per[str(p)][g][est] for g in real)
                    for p in (25, 1000)}
            vals["H16-1"][est] = diff
            means = [_mean(per[str(p)][g][est] for g in sets) for p in POPS]
            vals["H16-2"][est] = _slope(POPS, means)
        sl["H16-1"] = dict(members={str(p): dict(welch=({g: per[str(p)][g]["primary"] for g in boolean},
                                                        {g: per[str(p)][g]["primary"] for g in real}))
                                    for p in (25, 1000)})
        sl["H16-2"] = dict(t={g: _slope(POPS, [per[str(p)][g]["primary"] for p in POPS]) for g in sets})
        return dict(values=vals, per_set=per, set_level=sl)
    return stat


def _slope(pops, values):
    y = np.asarray(values, float)
    if not np.all(np.isfinite(y)):
        return NAN
    return float(np.polyfit(np.log2(np.asarray(pops, float)), y, 1)[0])


# ---------------------------------------------------------------- decisions (Rule S3)
def decide(name, point, reps, boot):
    """The test of one estimator: point (a number, or {member: number} for a conjunction) and its replicates."""
    spec = TESTS[name]
    kind = spec["kind"]
    if kind == "tost":
        c = spec.get("center", 0.0)
        t = stage5.tost(point - c, [x - c for x in reps], spec["delta"])
        return dict(t, statistic=point, center=c)
    if kind == "above":
        return dict(stage5.noninferiority(point, reps, spec["margin"], "above", boot), statistic=point)
    members = {}
    for m in sorted(point):
        mr = [r[m] for r in reps]
        if kind == "conj_tost":
            members[m] = stage5.tost(point[m], mr, spec["delta"])
        else:
            members[m] = stage5.noninferiority(point[m], mr, spec["margin"], "above", boot)
    out = stage5.conjunction(members)
    return dict(out, member_tests=members, statistic=dict(point),
                estimate=min(point.values()) if kind == "conj_above" else max(abs(v) for v in point.values()))


def run_block(names, ids_by_set, stat, boot=stage5.BOOT, seed=stage5.SEED):
    """The point estimate, the replicates (one target bootstrap for every test of the block) and every test under
    every estimator the statistic returns. Returns (tests {name: {point, tests {est: test}, replicates {est: [..]}}},
    point detail)."""
    full = {g: sorted(v) for g, v in ids_by_set.items()}
    point = stat(full)
    reps = [r["values"] for r in stage5.target_bootstrap(ids_by_set, stat, boot, seed)]
    out = {}
    for n in names:
        if n not in point["values"]:
            continue
        pv = point["values"][n]
        out[n] = dict(point=pv, tests={est: decide(n, pv[est], [r[n][est] for r in reps], boot) for est in pv},
                      replicates={est: [r[n][est] for r in reps] for est in pv})
    return out, point


def families(results, alpha=stage5.ALPHA):
    """Holm within each family for every estimator (the primary decides; a test whose decision differs under the
    isotonic reach is flagged), and the stage-wide Holm over all twelve primary tests (reported beside; None until
    every test is present)."""
    out = dict(families={}, flagged=[], stage_wide=None)
    for f, names in FAMILIES.items():
        have = [n for n in names if n in results]
        if not have:
            continue
        fam = {}
        for est in sorted({e for n in have for e in results[n]["tests"]}):
            if all(est in results[n]["tests"] for n in have):
                fam[est] = stage5.holm({n: results[n]["tests"][est] for n in have}, alpha)
        out["families"][f] = fam
        for n in have:
            if "iso" in fam and fam["iso"][n]["supported"] != fam["primary"][n]["supported"]:
                out["flagged"].append(n)
    if all(n in results for n in TESTS):
        out["stage_wide"] = stage5.holm({n: results[n]["tests"]["primary"] for n in TESTS}, alpha)
    return out


# ---------------------------------------------------------------- set-level intervals (reported beside)
def welch(a, b, level=0.90):
    """Welch's t interval for mean(a) - mean(b) over sets (descriptive; as the paper's S7)."""
    from scipy.stats import t as tdist
    a = np.asarray([x for x in a if np.isfinite(x)], float)
    b = np.asarray([x for x in b if np.isfinite(x)], float)
    d = float(a.mean() - b.mean()) if len(a) and len(b) else NAN
    if len(a) < 2 or len(b) < 2:
        return dict(mean=d, lo=NAN, hi=NAN, n=(len(a), len(b)))
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    if se == 0:
        return dict(mean=d, lo=d, hi=d, se=0.0, df=NAN, n=(len(a), len(b)))
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    q = float(tdist.ppf(0.5 + level / 2, df))
    return dict(mean=d, lo=d - q * se, hi=d + q * se, se=se, df=float(df), n=(len(a), len(b)))


def set_level(spec):
    """The set-level interval of one test from its point's set_level entry: t over per-set values, Welch for a
    difference of domain means, per member for a conjunction, numerator and denominator for H15-2's ratio."""
    if "t" in spec:
        return dict(stage5.set_level_interval(list(spec["t"].values())), values=spec["t"])
    if "welch" in spec:
        a, b = spec["welch"]
        return dict(welch(list(a.values()), list(b.values())), first=a, second=b)
    if "members" in spec:
        return {m: set_level(s) for m, s in spec["members"].items()}
    if "ratio_of_means" in spec:
        r = spec["ratio_of_means"]
        return {k: dict(stage5.set_level_interval(list(v.values())), values=v) for k, v in r.items()}
    raise ValueError(f"unknown set-level spec {sorted(spec)}")

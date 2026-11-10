"""Exploratory analyses for the revision (D125; REQ-ANA-112, REQ-ANA-113). EXPLORATORY and descriptive on registered
records: no test, no decision, no run; nothing here is confirmatory.

    PYTHONPATH=src python -m reach.analysis.s2.revision_report

Each section re-implements one computation of the private audit scripts in package code and first reproduces the
deposited registered values it overlaps (an instrument check raises otherwise). Writes
data/analysis/revision_exploratory.json and docs/status/revision_exploratory.md.
"""
import json
import math
import statistics as st
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from reach.analysis.s2 import alloc, confirm, ert, h6d, iso, law, null, nullgap, preempt, remedy, stage5
from reach.analysis.s2 import h11_report as H11
from reach.analysis.s2.load import E0, read_phase, tidy
from reach.analysis.s2.phase1 import enum_dir

LABEL = "EXPLORATORY (D125; registered records, no new runs; not confirmatory)"
BUDGETS = (1, 2, 4, 8, 16)
CAL = ("s2_cal_a0", "s2_cal_a16")
REVISIT = ("s2_revisit_a0", "s2_revisit_a16")
TWIN_BUDGETS = tuple(5_000 * 2 ** i for i in range(7))          # 1/4 to 16 x E0, the counter's prefixes
L2 = math.log2


# ---------------------------------------------------------------- reading
def rows(root, phase):
    """A phase's rows (`load.read_phase`: the raw results.jsonl, or the committed results.jsonl.gz)."""
    return read_phase(phase, root)


def _json(root, name):
    return json.loads((Path(root) / f"data/analysis/{name}.json").read_text())


def main_sets(root):
    m = json.loads((Path(root) / "data/registration/h7_model.json").read_text())
    return {d: m[d]["fit_records"]["sets"] for d in ("real", "bool")}


def level_counts(g):
    return {s["K"]: s["U"] for s in json.loads((enum_dir(g) / "counts.json").read_text())["levels"]}


def _check(ok, what):
    if not ok:
        raise RuntimeError(f"instrument check failed: {what}")


def _close(a, b, tol=1e-9):
    return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, abs_tol=tol)


def _mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


# ---------------------------------------------------------------- 1. Boolean proven-only reading capped at the top
def bool_capped(root, sets):
    """Proven-only Boolean K50 at E0 (`preempt.bool_size_variants`, as preempt.json), each set's K50 capped at its top
    proven level; and the proven levels each budget's fit rests on."""
    _, recs = preempt._recs(root, CAL, sets, "calibration", BUDGETS)
    po = preempt.bool_size_variants(recs, *confirm._sensitivity_inputs(root, sets))["proven_only"]
    dep = _json(root, "preempt")["bool_bounds"]["readings"]["proven_only"]
    per, nlev = {}, {}
    for g in sets:
        e0 = [r for r in po if r["set"] == g and r["budget"] == 1]
        k = nullgap._k50_bits(e0)
        _check(_close(k, dep["per_set"][g]["k50_e0"]), f"{g} proven-only K50 {k} vs preempt.json")
        top = max(r["units"]["bits_sem"] for r in e0)
        per[g] = dict(k50=k, top_level_bits=top, capped=bool(k > top), k50_used=min(k, top))
        nlev[g] = {str(b): len({round(r["units"]["bits_sem"], 6) for r in po if r["set"] == g and r["budget"] == b})
                   for b in BUDGETS}
    return dict(mean_gap=_mean([v["k50_used"] for v in per.values()]) - L2(E0),
                mean_gap_uncapped=dep["gap"]["gap"], sets_above_top=sum(v["capped"] for v in per.values()),
                per_set=per, n_proven_levels_by_budget=nlev)


# ---------------------------------------------------------------- 2. G with the twin re-estimated on proven targets
def g_proven(root, rev, sets):
    """H6d's G on proven targets (K_proven set) with the twin re-estimated on them (`h6d.set_terms` on the re-executed
    rows), beside S8's G (preempt.json: proven-only K50 minus the registered twin)."""
    dep = _json(root, "preempt")["bool_bounds"]["readings"]["proven_only"]
    reg = _json(root, "h6d")["per_set"]["bool"]
    per = {}
    for g in sets:
        U = level_counts(g)
        runs = h6d.runs_from_rows([r for r in rev[g] if r.get("K_proven") is not None], TWIN_BUDGETS)
        t = h6d.set_terms(runs, E0, U)
        per[g] = dict(G=t["G"], k50_gp=t["k50_gp"], k50_twin=t["k50_twin"], k50_twin_registered=reg[g]["k50_twin"],
                      G_registered_twin=dep["per_set"][g]["G"], k50_gp_preempt=dep["per_set"][g]["k50_e0"])
    _check(_close(_mean([v["G_registered_twin"] for v in per.values()]), dep["G_mean"]), "S8 mean G")
    return dict(mean_g=_mean([v["G"] for v in per.values()]), mean_g_registered_twin=dep["G_mean"],
                max_abs_k50_gp_difference=max(abs(v["k50_gp"] - v["k50_gp_preempt"]) for v in per.values()),
                per_set=per)


# ---------------------------------------------------------------- 3. coverage within the same runs (Figure 4)
def coverage_within(root, rev16, g="R-ARITH4", early=10_000, late=E0 * 16):
    """Mean distinct functions of at most the ceiling (every enumerated level) a run of s2_revisit_a16 has evaluated
    by `early` and by `late` evaluations, on the runs that reached both prefixes (Figure 4's fields)."""
    K = [k for k, u in level_counts(g).items() if u > 0]
    rr = [r for r in rev16 if r.get("set") == g and r.get("rv")]

    def small(r, P):
        return sum(r["rv"]["prefixes"][str(P)]["lv"][k] for k in K)
    at_e = [r for r in rr if str(early) in r["rv"]["prefixes"]]
    at_l = [r for r in rr if str(late) in r["rv"]["prefixes"]]
    both = [r for r in at_e if str(late) in r["rv"]["prefixes"]]
    gone = [r for r in at_e if str(late) not in r["rv"]["prefixes"]]
    m_e, m_l = _mean([small(r, early) for r in both]), _mean([small(r, late) for r in both])
    return dict(set=g, early_evaluations=early, late_evaluations=late, n_runs=len(both), early=m_e, late=m_l,
                ratio=m_l / m_e, all_runs_early=dict(n=len(at_e), mean=_mean([small(r, early) for r in at_e])),
                all_runs_late=dict(n=len(at_l), mean=_mean([small(r, late) for r in at_l])),
                stopped_between=dict(n=len(gone), mean_early=_mean([small(r, early) for r in gone])))


# ---------------------------------------------------------------- 4. H11 on the shared five levels
def restart_data(root):
    """Stage 4's two arms as h11_report builds them (registered target files checked against their SHA-256)."""
    srows = rows(root, "s4_restart")
    sizes = H11.load_sizes(root, H11.SETS)
    restart = {}
    for g in H11.SETS:
        by = defaultdict(list)
        for r in srows:
            if r["set"] == g:
                by[r["target_id"]].append(r)
        restart[g] = dict(by)
    recs = tidy([r for r in rows(root, "s2_ho_a16") if r.get("set") in H11.SETS], (16,))
    one = {g: [dict(set=g, target_id=r["target_id"], y=r["y"], units={"bits_sem": sizes[g][r["target_id"]]})
               for r in recs if r["set"] == g] for g in H11.SETS}
    return restart, one, sizes, len(srows)


def _restart_recs(g, runs, size):
    return [dict(set=g, target_id=t, units={"bits_sem": size[t]}) for t in runs]


def h11_common(root, restart, one, sizes):
    """The restart arm refitted on the one-run arm's (level, target) rows (`stage5.shared_levels` checks them), the
    registered estimator and bracketing rule (`remedy.restart_k50`); gain = restart K50 - one-run K50."""
    reg = _json(root, "h11")["h11"]["per_set"]
    per = {}
    for g in H11.SETS:
        full = remedy.restart_k50(restart[g], {t: sizes[g][t] for t in restart[g]})
        k1 = law._k50(one[g], "bits_sem")
        _check(_close(full["k50"], reg[g]["k50"]) and _close(k1, reg[g]["one_run"]), f"{g} H11 arms vs h11.json")
        ids = sorted({r["target_id"] for r in one[g]})
        sub = {t: restart[g][t] for t in ids}
        stage5.shared_levels(_restart_recs(g, sub, sizes[g]), one[g], f"H11 common levels {g}")
        k = remedy.restart_k50(sub, {t: sizes[g][t] for t in ids})
        per[g] = dict(k50_all_levels=full["k50"], k50_common=k["k50"], raw_common=k["raw"],
                      extrapolated=k["extrapolated"], top_common=k["top"], one_run=k1, gain=k["k50"] - k1,
                      raw_gain=k["raw"] - k1,
                      gain_registered=reg[g]["gain"], n_levels_all=len({round(sizes[g][t], 6) for t in restart[g]}),
                      n_levels_common=len({round(sizes[g][t], 6) for t in ids}),
                      gap_common=k["k50"] - L2(16 * E0))
    br = [v["gain"] for v in per.values() if not v["extrapolated"]]
    lb = [v["gain"] for v in per.values() if v["extrapolated"]]
    return dict(per_set=per, bracketed=len(br), bracketed_gain_lo=min(br) if br else None,
                bracketed_gain_hi=max(br) if br else None, bound_lo=min(lb) if lb else None,
                bound_hi=max(lb) if lb else None, mean_gap_common=_mean([v["gap_common"] for v in per.values()]))


# ---------------------------------------------------------------- 5. S12's base on the configurations' targets
def s12_same_targets(root):
    """Tree GP base and each H8 configuration (s2_cal_robust) on the same targets, real sets, `remedy.null_gaps`; the
    base on the robustness targets must equal H8b's base_gap (calibration.json)."""
    rob = remedy._tidy(root, ("s2_cal_robust",), "calibration", (1, 4))
    tg = {(r["set"], r["target_id"]) for r in rob}
    base_all = remedy._tidy(root, CAL, "calibration", (1, 4), lambda r: r.get("algorithm") == "A" and
                            r.get("config") == "base")
    base = [r for r in base_all if (r["set"], r["target_id"]) in tg]
    h8b = _json(root, "calibration")["h8b"]
    real = sorted({r["set"] for r in rob if not r["set"].startswith("BOOL")})
    out = dict(by_budget={}, per_set={g: {} for g in real})
    for B in (1, 4):
        gb = remedy.null_gaps(base, B)
        ga = remedy.null_gaps([r for r in base_all if r["set"] in real], B)
        row = dict(base=_mean([gb[g]["gap"] for g in real]), base_all_levels=_mean([ga[g]["gap"] for g in real]))
        for g in real:
            _check(_close(gb[g]["gap"], h8b[f"{g}|R1"]["budgets"][str(B)]["base_gap"]), f"{g} H8b base at {B}")
            out["per_set"][g][str(B)] = dict(base=gb[g]["gap"], base_all_levels=ga[g]["gap"])
        for cfg in sorted({r["config"] for r in rob}):
            gc = remedy.null_gaps([r for r in rob if r["config"] == cfg], B)
            row[cfg] = _mean([gc[g]["gap"] for g in real])
            for g in real:
                out["per_set"][g][str(B)][cfg] = gc[g]["gap"]
        out["by_budget"][str(B)] = row
    e0 = out["by_budget"]["1"]
    return dict(base=e0["base"], base_all_levels=e0["base_all_levels"], sets=real,
                configs={k: v for k, v in e0.items() if k not in ("base", "base_all_levels")}, **out)


# ---------------------------------------------------------------- 6. tolerance successes by level (S10)
def tolerance_added(root, sets=("R-HYB", "R-ARITH4")):
    """Runs within tolerance at E0 (`preempt.tidy_tol`) but not exact (`load.tidy`), per level; the lower five and the
    upper five of each set's ten levels."""
    raw, ex = preempt._recs(root, CAL, list(sets), "calibration", (1,))
    _, tl = preempt._recs(root, CAL, list(sets), "calibration", (1,), tol=True)
    out = {}
    for g in sets:
        e = {(r["target_id"], r["seed"]): r for r in ex if r["set"] == g}
        t = {(r["target_id"], r["seed"]): r for r in tl if r["set"] == g}
        lv = sorted({round(r["units"]["bits_sem"], 6) for r in e.values()})
        add = defaultdict(int)
        for k, r in t.items():
            if r["y"] and not e[k]["y"]:
                add[round(r["units"]["bits_sem"], 6)] += 1
        lost = sum(1 for k, r in e.items() if r["y"] and not t[k]["y"])
        out[g] = dict(total=sum(add.values()), lower_five=sum(add[x] for x in lv[:5]),
                      upper_five=sum(add[x] for x in lv[5:]), per_level={f"{x:.2f}": add[x] for x in lv},
                      exact_not_tolerance=lost,
                      k50_exact=nullgap._k50_bits(list(e.values())), k50_tolerance=nullgap._k50_bits(list(t.values())),
                      violations=sum(1 for r in raw if r["set"] == g and r.get("exact")
                                     and (r.get("gen_tol") is None or r["gen_tol"] > r["gen_exact"])))
    first, rest = sets[0], sets[1:]
    return dict(out[first], set=first, **{g: out[g] for g in rest})


# ---------------------------------------------------------------- 7. the twin: extrapolation and budget
def twin(root, rev, sets):
    """The registered twin K50 against each set's lowest tested level; the interpolated twin (`h6d.k50_interp`, every
    enumerated level) per budget from the re-executed rows, reproducing h6d.json's k50_twin_interp at E0."""
    h = _json(root, "h6d")
    per, by_budget = {}, {str(E): {} for E in TWIN_BUDGETS}
    for d in ("real", "bool"):
        vals = {E: [] for E in TWIN_BUDGETS}
        for g in sets[d]:
            U = level_counts(g)
            runs = h6d.runs_from_rows(rev[g], TWIN_BUDGETS)
            low = min(r["units"]["bits_sem"] for r in rev[g] if r.get("units"))
            ti = {}
            for E in TWIN_BUDGETS:
                lv = h6d.cumulative(runs, E, "lv")
                ti[E] = h6d.k50_interp(h6d.twin_probs(lv, U), U) if np.ndim(lv) else float("nan")
                vals[E].append(ti[E])
            _check(_close(ti[E0], h["sensitivities"][g]["k50_twin_interp"]), f"{g} interpolated twin at E0")
            k = h["per_set"][d][g]["k50_twin"]
            per[g] = dict(domain=d, k50_twin=k, k50_twin_interp=ti[E0], lowest_level=low, below_lowest=bool(k < low),
                          distance_below=low - k, interp_by_budget={str(E): ti[E] for E in TWIN_BUDGETS})
        for E in TWIN_BUDGETS:
            by_budget[str(E)][d] = _mean(vals[E])
    for v in by_budget.values():
        v["diff"] = v["bool"] - v["real"]
    ir = _mean([h["sensitivities"][g]["k50_twin_interp"] for g in sets["real"]])
    ib = _mean([h["sensitivities"][g]["k50_twin_interp"] for g in sets["bool"]])
    return dict(interp_real=ir, interp_bool=ib, interp_diff=ib - ir,
                extrapolated_count=sum(v["below_lowest"] for v in per.values()), n_sets=len(per),
                by_budget_unit="evaluations", by_budget=by_budget, per_set=per)


# ---------------------------------------------------------------- 8. restarts against the realisable enumerator
def realisable_vs_restart(root):
    """Restarted GP's K50 at 16 x E0 (h11.json) against the realisable null (bottom-up enumeration with observational-
    equivalence pruning, every candidate counted): its reach = log2 E + (bottom-up reach - exact reach), both by
    `null.reach_bits_sem` (S9's shift convention, REQ-ANA-90)."""
    h, E = _json(root, "h11")["h11"]["per_set"], 16 * E0
    per = {}
    for g, v in h.items():
        lv = json.loads((enum_dir(g) / "counts.json").read_text())["levels"]
        ex, bu = null.reach_bits_sem(lv, E, "enumeration"), null.reach_bits_sem(lv, E, "bottom_up")
        real = L2(E) + bu - ex
        per[g] = dict(k50_restart=v["k50"], exact_reach=ex, bottom_up_reach=bu, realisable=real, diff=v["k50"] - real)
    return dict(mean=_mean([v["diff"] for v in per.values()]), sets_above=sum(v["diff"] > 0 for v in per.values()),
                n_sets=len(per), per_set=per)


# ---------------------------------------------------------------- 9. the isotonic interpolated reach (REQ-ANA-108)
def _iso_set(recs, U, E):
    """GP's isotonic reach per target (`iso.reach_records`) and the exact null's on the same targets, its success
    share within E at each target's level (`h6d.enum_probs`; 0 above the enumeration), read the same way."""
    gp = iso.reach_records(recs)
    per_run = iso.reach([r["units"]["bits_sem"] for r in recs], [r["y"] for r in recs])   # levels weighted by runs
    pn = h6d.enum_probs(E, U)
    by = {r["target_id"]: r for r in recs}
    nl = iso.reach([r["units"]["bits_sem"] for r in by.values()], [pn.get(r["K"], 0.0) for r in by.values()])
    return dict(reach=gp["reach"], flag=gp["flag"], lo=gp["lo"], hi=gp["hi"], gap=gp["reach"] - L2(E),
                gap_per_run=per_run["reach"] - L2(E), null_reach=nl["reach"], null_flag=nl["flag"],
                null_offset=nl["reach"] - L2(E),
                gap_like=gp["reach"] - nl["reach"], logistic=law._k50(recs, "bits_sem"))


def _iso_summary(per, sets):
    vs = [per[g] for g in sets]
    return dict(mean_gap=_mean([v["gap"] for v in vs]), mean_gap_per_run=_mean([v["gap_per_run"] for v in vs]),
                mean_gap_like=_mean([v["gap_like"] for v in vs]), mean_null_offset=_mean([v["null_offset"] for v in vs]),
                not_bracketed=[g for g in sets if per[g]["flag"] != "bracketed"])


def iso_section(root, sets, restart, one, sizes):
    _, recs = preempt._recs(root, CAL, sets["real"] + sets["bool"], "calibration", (1,))
    e0 = {g: _iso_set([r for r in recs if r["set"] == g], level_counts(g), E0) for d in sets for g in sets[d]}
    cal = _json(root, "calibration")["h6c"]["gaps"]
    for d in sets:
        for g in sets[d]:
            _check(_close(e0[g]["logistic"] - L2(E0), cal[d]["per_set"][g]), f"{g} logistic gap vs H6c")
    out = dict(e0=dict(per_set=e0, **{d: _iso_summary(e0, sets[d]) for d in sets}))
    K = _heldout_K(root, H11.SETS)
    E = 16 * E0
    r16 = {"all": {}, "shared": {}, "one_run": {}}
    for g in H11.SETS:
        U = level_counts(g)
        for arm, ids in (("all", sorted(restart[g])), ("shared", sorted({r["target_id"] for r in one[g]}))):
            sz, ps = [], []
            for t in ids:
                s, n = remedy.hits(restart[g][t], remedy.H11_L)
                ps.append(alloc.p_at_least_one(s, n, remedy.H11_R)[0])
                sz.append(sizes[g][t])
            gp = iso.reach(sz, ps)
            nl = iso.reach(sz, [h6d.enum_probs(E, U).get(K[g][t], 0.0) for t in ids])
            r16[arm][g] = dict(reach=gp["reach"], flag=gp["flag"], gap=gp["reach"] - L2(E), null_reach=nl["reach"],
                               null_flag=nl["flag"], gap_like=gp["reach"] - nl["reach"], n_levels=gp["n_levels"])
        o = iso.reach_records(one[g])
        r16["one_run"][g] = dict(reach=o["reach"], flag=o["flag"], gap=o["reach"] - L2(E), n_levels=o["n_levels"])
        r16["shared"][g]["gain"] = r16["shared"][g]["reach"] - o["reach"]
        r16["all"][g]["gain"] = r16["all"][g]["reach"] - o["reach"]
    for arm, v in r16.items():
        v["mean_gap"] = _mean([x["gap"] for x in v.values()])
        if arm != "one_run":
            v["mean_gap_like"] = _mean([x["gap_like"] for x in v.values() if isinstance(x, dict)])
    out["restart16"] = r16
    return out


def _heldout_K(root, sets):
    out = {}
    for g in sets:
        out[g] = {}
        for name in H11.TARGET_FILES:
            p = Path(root) / f"data/targets_s2/{g}/{name}.json"
            if p.exists():
                out[g].update({t["target_id"]: t["K_proven"] for t in json.loads(p.read_text())
                               if t["K_proven"] == t["K_construction"] and not t.get("constructed")})
    return out


# ---------------------------------------------------------------- 10. ERT per set and level (REQ-ANA-113)
def _ert_levels(rr, U, E, kind):
    by = defaultdict(list)
    for r in rr:
        if r.get("K_proven") is not None and r.get("K_proven") in U:
            by[r["K_proven"]].append(r)
    out = {}
    for K, runs in sorted(by.items()):
        e, n = ert.ert(runs, E, kind), ert.null_evals(U, K)
        if not e["runs"]:                       # no run of this level reaches E (main levels run at E0 only)
            continue
        out[str(K)] = dict(bits=float(L2(n["cover"])), **e, null_cover=n["cover"], null_expected=n["expected"],
                           gap_expected=ert.gap_bits(n["expected"], e["ert"]),
                           gap_cover=ert.gap_bits(n["cover"], e["ert"]),
                           gap_bound=(ert.gap_bits(n["expected"], e["lower_bound"]) if e["lower_bound"] else None))
    return out


def _ert_summary(per, sets):
    out = {}
    for d, gs in sets.items():
        cells = [v for g in gs if g in per for v in per[g].values()]
        fin = [v["gap_expected"] for v in cells if math.isfinite(v["gap_expected"])]
        meds = [st.median([v["gap_expected"] for v in per[g].values() if math.isfinite(v["gap_expected"])]) for g in gs
                if g in per and any(math.isfinite(v["gap_expected"]) for v in per[g].values())]
        out[d] = dict(n_cells=len(cells), n_no_success=len(cells) - len(fin),
                      median_gap_expected=st.median(fin) if fin else None,
                      median_gap_cover=st.median([v["gap_cover"] for v in cells if math.isfinite(v["gap_cover"])])
                      if fin else None, mean_of_set_medians=_mean(meds))
    return out


def ert_section(root, sets, cal_rows, cgp_rows):
    res = dict(tree={}, cgp={}, summary=dict(tree={}, cgp={}))
    allsets = sets["real"] + sets["bool"]
    for B in (1, 16):
        E = B * E0
        res["tree"][str(B)] = {g: _ert_levels([r for r in cal_rows if r["set"] == g], level_counts(g), E, "tree")
                               for g in allsets}
        res["cgp"][str(B)] = {g: _ert_levels([r for r in cgp_rows if r["set"] == g], level_counts(g), E, "cgp")
                              for g in allsets if any(r["set"] == g for r in cgp_rows)}
        for sysname in ("tree", "cgp"):
            res["summary"][sysname][str(B)] = _ert_summary(res[sysname][str(B)], sets)
    res["cgp_configs"] = {g: sorted({r["config"] for r in cgp_rows if r["set"] == g}) for g in allsets}
    dep = _json(root, "remedy")["systems"]
    for cfg in sorted({r["config"] for r in cgp_rows}):
        rr = [r for r in cgp_rows if r["config"] == cfg]
        confirm._load_units(root, sorted({r["set"] for r in rr}), "calibration")
        recs = [r for r in confirm._with_units(tidy(rr, (1, 16))) if r["set"] not in confirm.SIZE_CHECKS]
        for B in (1, 16):
            for g, v in remedy.null_gaps(recs, B).items():
                _check(_close(v["gap"], dep[f"Cartesian GP {cfg}, {B} x E0"][g]["gap"]), f"CGP {cfg} {g} {B} gap")
    return res


# ---------------------------------------------------------------- 11. benchmark cells (s2_ho_bench, E0)
def bench(root):
    """Per benchmark cell (one proven target, 30 runs at E0): GP's success share at E0 against the null's
    (`ert.null_success`), and the gap log2(null's expected evaluations) - log2(GP's ERT at E0)."""
    cells = defaultdict(list)
    for r in rows(root, "s2_ho_bench"):
        cells[(r["domain"], r["set"], r["target_id"])].append(r)
    per = {}
    for (d, g, t), rr in sorted(cells.items()):
        U, K = level_counts(g), rr[0]["K_proven"]
        e, n = ert.ert(rr, E0, "tree"), ert.null_evals(U, K)
        p_gp = e["successes"] / e["runs"]
        p_null = ert.null_success(U, K, E0)
        per[f"{g}|{t}"] = dict(domain=d, set=g, target_id=t, K=K, bits=rr[0]["units"]["bits_sem"], runs=e["runs"],
                               successes=e["successes"], p_gp=p_gp, p_null=p_null, ert=e["ert"],
                               null_expected=n["expected"], null_cover=n["cover"],
                               gap_expected=ert.gap_bits(n["expected"], e["ert"]),
                               gap_cover=ert.gap_bits(n["cover"], e["ert"]),
                               gap_bound=ert.gap_bits(n["expected"], e["lower_bound"]) if e["lower_bound"] else None,
                               floor=bool(n["expected"] <= 100))
    summ = {}
    for d in ("real", "bool"):
        vs = [v for v in per.values() if v["domain"] == d]
        fin = [v for v in vs if math.isfinite(v["gap_expected"])]
        summ[d] = dict(n_cells=len(vs), below=sum(v["p_gp"] < v["p_null"] for v in vs),
                       above=sum(v["p_gp"] > v["p_null"] for v in vs), tie=sum(v["p_gp"] == v["p_null"] for v in vs),
                       n_with_success=len(fin),
                       median_gap_expected=st.median([v["gap_expected"] for v in fin]) if fin else None,
                       median_gap_cover=st.median([v["gap_cover"] for v in fin]) if fin else None,
                       median_gap_expected_nonfloor=st.median([v["gap_expected"] for v in fin if not v["floor"]])
                       if any(not v["floor"] for v in fin) else None, n_floor=sum(v["floor"] for v in vs))
    reg = _json(root, "stage2")["phases"]["s2_ho_bench"]
    n_succ = sum(v["successes"] for v in per.values())
    _check(sum(v["runs"] for v in per.values()) == reg["rows"] and n_succ == reg["states"]["converged"],
           f"s2_ho_bench rows and successes vs stage2.json ({n_succ})")
    return dict(per_cell=per, **summ)


# ---------------------------------------------------------------- 12. Operon facts
OPERON_TASK = "src/reach/tasks/s2_operon.py"
PYOPERON = ".venv-ext/lib/python3.11/site-packages/pyoperon/sklearn.py"


def _main_root(root):
    r = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=root, capture_output=True, text=True)
    return (Path(root) / r.stdout.strip()).resolve().parent if r.returncode == 0 and r.stdout.strip() else Path(root)


def operon(root):
    """Operon's training data, budget use per mode, and its real gap at 16 x E0, from the task code, the installed
    pyoperon and the records (s2_cal_operon, s2_ho_operon); the gaps by `remedy.null_gaps`, equal to remedy.json's."""
    code = (Path(root) / OPERON_TASK).read_text()
    facts = dict(training_points=75, draws_once=code.count("rng.uniform(2.0, 3.0, 75)") == 1,
                 one_fit=code.count(".fit(") == 1, source=OPERON_TASK)
    py = next((p for p in (Path(root) / PYOPERON, _main_root(root) / PYOPERON) if p.exists()), None)
    if py is not None:
        t = py.read_text()
        facts["pyoperon_all_rows"] = "training_range        = op.Range(0, ds.Rows)" in t
        facts["pyoperon_batch_default_all"] = "0 uses all training samples" in t
    facts["pyoperon_checked"] = py is not None
    use = {}
    for ph in ("s2_cal_operon", "s2_ho_operon"):
        for r in rows(root, ph):
            if "evaluations" in r:
                use.setdefault(r["config"], []).append(r)
    budget = {c: dict(n=len(v), eval_share=st.median(r["evaluations"] / r["budget_E"] for r in v),
                      residual_share=st.median(max(r["residual_evaluations"], 0) / r["budget_E"] for r in v),
                      jacobian_share=st.median(max(r["jacobian_evaluations"], 0) / r["budget_E"] for r in v),
                      exact_share=_mean([float(r["exact"]) for r in v])) for c, v in sorted(use.items())}
    dep = _json(root, "remedy")["systems"]
    gaps = {}
    for cfg in ("symbolic", "default"):
        op = remedy._tidy(root, ("s2_cal_operon",), "calibration", (16,), lambda r: r.get("config") == cfg) + \
            remedy._tidy(root, ("s2_ho_operon",), "heldout", (16,), lambda r: r.get("config") == cfg)
        gg = remedy.null_gaps(op, 16)
        for g, v in gg.items():
            _check(_close(v["gap"], dep[f"Operon {cfg}, 16 x E0"][g]["gap"]), f"Operon {cfg} {g} vs remedy.json")
        gaps[cfg] = dict(mean_gap=_mean([v["gap"] for v in gg.values()]), per_set={g: v["gap"] for g, v in gg.items()},
                         n_sets=len(gg))
    return dict(facts=facts, budget_use=budget, gap_16=gaps, real_gap_16=gaps["symbolic"]["mean_gap"])


# ---------------------------------------------------------------- driver
def build(root):
    root = Path(root)
    t0 = time.process_time()
    sets = main_sets(root)
    rev = h6d.rows_by_set(root, REVISIT)
    rev16 = rows(root, "s2_revisit_a16")
    restart, one, sizes, n_restart = restart_data(root)
    cal_rows = [r for ph in CAL for r in rows(root, ph) if r.get("algorithm") == "A" and r.get("config") == "base"]
    cgp_rows = [r for r in rows(root, "s2_cal_cgp")]
    res = dict(label=LABEL)
    res["bool_capped"] = bool_capped(root, sets["bool"])
    res["g_proven_reestimated"] = g_proven(root, rev, sets["bool"])
    res["coverage_within"] = coverage_within(root, rev16)
    res["h11_common"] = h11_common(root, restart, one, sizes)
    res["s12_same_targets"] = s12_same_targets(root)
    res["tolerance_added"] = tolerance_added(root)
    res["twin"] = twin(root, rev, sets)
    res["realisable_vs_restart"] = realisable_vs_restart(root)
    res["iso"] = iso_section(root, sets, restart, one, sizes)
    res["ert"] = ert_section(root, sets, cal_rows, cgp_rows)
    res["bench"] = bench(root)
    res["operon"] = operon(root)
    res["rows_s4_restart"] = n_restart
    res["built"] = time.strftime("%Y-%m-%d %H:%M:%S")
    res["cpu_seconds"] = round(time.process_time() - t0, 1)
    return res


# ---------------------------------------------------------------- report
def _f(x, d=2, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, float) and math.isinf(x):
        return "+inf" if x > 0 else "-inf"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def render(res):
    b, g, c, h = res["bool_capped"], res["g_proven_reestimated"], res["coverage_within"], res["h11_common"]
    L = ["# Revision analyses (EXPLORATORY)", "",
         f"> Generated by `reach.analysis.s2.revision_report` ({res['built']}); every number from "
         "data/analysis/revision_exploratory.json (key in brackets). Do not edit by hand. Exploratory and descriptive "
         "on registered records (D125): no test, no decision, no new run; not confirmatory. Each section first "
         "reproduces the deposited registered values it overlaps (named below); the build stops otherwise. Gaps are "
         "in bits_sem against the exact null's reach log₂ E unless stated.", "",
         "## 1. Boolean proven-only K₅₀ capped at the top proven level [bool_capped]", "",
         "s2_cal_a0 + s2_cal_a16 (tree GP base), the 7 Boolean main sets, proven targets only "
         "(`preempt.bool_size_variants`); K₅₀ at E₀ by `law._k50` equals preempt.json's proven-only K₅₀ per set. A "
         "K₅₀ above the set's top proven level is replaced by that level; mean gap = mean K₅₀ − log₂ 20,000.", "",
         "| set | K₅₀ | top proven level | capped | proven levels at 1, 2, 4, 8, 16 × E₀ |", "|---|---|---|---|---|"]
    for s, v in b["per_set"].items():
        L.append(f"| {s} | {_f(v['k50'])} | {_f(v['top_level_bits'])} | {'yes' if v['capped'] else ''} | "
                 f"{', '.join(str(n) for n in b['n_proven_levels_by_budget'][s].values())} |")
    L += ["", f"Mean gap {_f(b['mean_gap'], sign=True)} capped ({_f(b['mean_gap_uncapped'], sign=True)} uncapped, "
          f"preempt.json); {b['sets_above_top']} sets above their top level. Above E₀ only the budget levels run, so "
          "the per-budget fits rest on 2 to 4 proven levels.", "",
          "## 2. G with the twin re-estimated on proven targets [g_proven_reestimated]", "",
          "`h6d.set_terms` at E₀ on the re-executed rows (s2_revisit_a0 + _a16) with K_proven set: mean G "
          f"{_f(g['mean_g'])}; S8's G (proven-only K₅₀ minus the registered twin, preempt.json) {_f(g['mean_g_registered_twin'])}."
          f" The two proven-only GP K₅₀ readings agree within {g['max_abs_k50_gp_difference']:.1e} bits.", "",
          "## 3. Coverage on the same runs [coverage_within]", "",
          f"s2_revisit_a16, {c['set']}: distinct functions at every enumerated level per run, on the {c['n_runs']} "
          f"runs that reached both prefixes: {_f(c['early'], 1)} at {c['early_evaluations']:,} and {_f(c['late'], 1)} "
          f"at {c['late_evaluations']:,} evaluations (×{_f(c['ratio'])}). Figure 4's means (all runs at each prefix): "
          f"{_f(c['all_runs_early']['mean'], 1)} over {c['all_runs_early']['n']} runs and "
          f"{_f(c['all_runs_late']['mean'], 1)} over {c['all_runs_late']['n']}; the {c['stopped_between']['n']} runs "
          f"that stopped in between had {_f(c['stopped_between']['mean_early'], 1)} at {c['early_evaluations']:,}.", "",
          "## 4. H11's restart arm on the one-run arm's levels [h11_common]", "",
          "s4_restart (32 restarts × 100 generations, `remedy.restart_k50`, bracketing rule) refitted on exactly the "
          "(level, target) rows of the one-run arm (s2_ho_a16 at 16 × E₀; `stage5.shared_levels` checks the rows); "
          "both arms on all their levels reproduce h11.json. `^`: above the top shared level, replaced by it (the gain "
          "is a lower bound).", "",
          "| set | levels (all / shared) | restart K₅₀ all | restart K₅₀ shared | one run | gain shared | registered gain |",
          "|---|---|---|---|---|---|---|"]
    for s, v in h["per_set"].items():
        L.append(f"| {s} | {v['n_levels_all']} / {v['n_levels_common']} | {_f(v['k50_all_levels'])} | "
                 f"{_f(v['k50_common'])}{'^' if v['extrapolated'] else ''} | {_f(v['one_run'])} | "
                 f"{_f(v['gain'], sign=True)} | {_f(v['gain_registered'], sign=True)} |")
    L += ["", f"Bracketed in {h['bracketed']} of 6 sets (gains {_f(h['bracketed_gain_lo'])} to "
          f"{_f(h['bracketed_gain_hi'])}); lower bounds {_f(h['bound_lo'])} to {_f(h['bound_hi'])} elsewhere; mean gap "
          f"on the shared levels {_f(h['mean_gap_common'], sign=True)} (biased down by the cap).", ""]
    s12 = res["s12_same_targets"]
    L += ["## 5. S12 on the configurations' own targets [s12_same_targets]", "",
          "Real sets " + ", ".join(s12["sets"]) + "; tree GP base (s2_cal_a0 + _a16) restricted to the targets of "
          "s2_cal_robust, gaps by `remedy.null_gaps` (each set's base equals H8b's base_gap in calibration.json); "
          "the configurations from s2_cal_robust (equal to remedy.json). Means over the three sets.", "",
          "| budget | base, same targets | base, all ten levels | " + " | ".join(s12["configs"]) + " |",
          "|---|---|---|" + "---|" * len(s12["configs"])]
    for B, v in s12["by_budget"].items():
        L.append(f"| {B} × E₀ | {_f(v['base'], sign=True)} | {_f(v['base_all_levels'], sign=True)} | " +
                 " | ".join(_f(v[k], sign=True) for k in s12["configs"]) + " |")
    t = res["tolerance_added"]
    a = t["R-ARITH4"]
    L += ["", "## 6. Tolerance successes by level [tolerance_added]", "",
          "Runs within tolerance at E₀ (`preempt.tidy_tol`, validation nMSE ≤ 10⁻⁶) but not exact (`load.tidy`), "
          "s2_cal_a0 + _a16, by level (bits): " +
          ", ".join(f"{k}: {n}" for k, n in t["per_level"].items()) +
          f" ({t['set']}). {t['set']}: {t['total']} added, {t['lower_five']} at the five lower levels, "
          f"{t['upper_five']} at the five upper; R-ARITH4: {a['total']}, {a['lower_five']} and {a['upper_five']}. "
          f"No exact run is outside tolerance ({t['violations'] + a['violations']} violations). Logistic K₅₀ exact → "
          f"tolerance: {t['set']} {_f(t['k50_exact'])} → {_f(t['k50_tolerance'])}, R-ARITH4 {_f(a['k50_exact'])} → "
          f"{_f(a['k50_tolerance'])}.", ""]
    w = res["twin"]
    L += ["## 7. The twin [twin]", "",
          f"h6d.json: the registered twin K₅₀ lies below the set's lowest tested level in {w['extrapolated_count']} "
          f"of {w['n_sets']} sets. Interpolated twin (`h6d.k50_interp` over every enumerated level, h6d.json "
          f"sensitivities): real {_f(w['interp_real'], 3)}, Boolean {_f(w['interp_bool'], 3)}, difference "
          f"{_f(w['interp_diff'], 3, sign=True)}. Per budget (evaluations; the re-executed rows' counts at each "
          "prefix, reproducing h6d.json at 20,000):", "",
          "| evaluations | " + " | ".join(f"{int(E):,}" for E in w["by_budget"]) + " |",
          "|---|" + "---|" * len(w["by_budget"])]
    for d in ("real", "bool", "diff"):
        L.append(f"| {d} | " + " | ".join(_f(v[d], sign=d == "diff") for v in w["by_budget"].values()) + " |")
    rv = res["realisable_vs_restart"]
    L += ["", "## 8. Restarts against the realisable enumerator [realisable_vs_restart]", "",
          "Restarted K₅₀ at 16 × E₀ (h11.json) minus the realisable null's reach, log₂ E + (bottom-up − exact reach) "
          "by `null.reach_bits_sem` (S9's convention): " +
          ", ".join(f"{s} {_f(v['diff'], sign=True)}" for s, v in rv["per_set"].items()) +
          f"; mean {_f(rv['mean'], sign=True)}, above in {rv['sets_above']} of {rv['n_sets']}.", ""]
    L += _render_iso(res["iso"]) + _render_ert(res["ert"]) + _render_bench(res["bench"]) + _render_operon(res["operon"])
    return "\n".join(L) + "\n"


def _render_iso(i):
    e = i["e0"]
    L = ["## 9. Isotonic interpolated reach [iso] (REQ-ANA-108)", "",
         "Per target the share of its runs that succeeded; per level the targets' mean; a non-increasing PAV fit "
         "weighted by targets; the 1/2 crossing interpolated between the two bracketing levels (`iso.reach`). The "
         "exact null read the same way: its success share within E at each target's level (`h6d.enum_probs`). E₀: "
         "s2_cal_a0 + _a16, the 14 main sets, registered units; the logistic gaps reproduce H6c's per-set gaps.", "",
         "| domain | isotonic gap | levels weighted by runs | null's isotonic offset | like-for-like | not crossing |",
         "|---|---|---|---|---|---|"]
    for d in ("real", "bool"):
        v = e[d]
        L.append(f"| {d} | {_f(v['mean_gap'], sign=True)} | {_f(v['mean_gap_per_run'], sign=True)} | "
                 f"{_f(v['mean_null_offset'], sign=True)} | {_f(v['mean_gap_like'], sign=True)} | "
                 f"{', '.join(v['not_bracketed']) or 'none'} |")
    r = i["restart16"]
    L += ["", "The earlier monotone reading in the notes (PAV over the level rates, levels weighted by runs) is the "
          "third column; with equal runs per target it equals the targets' reading. R-HYB's isotonic reach is at most "
          "its lowest level (its rate there is below 1/2). Restart arm at 16 × E₀ (32 × 100, P per target from all 64 "
          "runs) and the one-run arm (s2_ho_a16):", "",
          "| set | restarts, all levels: gap (null-read) | restarts, shared five | one run (five) | gain all / shared |",
          "|---|---|---|---|---|"]
    for s in r["one_run"]:
        if not isinstance(r["one_run"][s], dict):
            continue
        a, sh, o = r["all"][s], r["shared"][s], r["one_run"][s]
        fl = {"at_least_top": "^", "at_most_lowest": "v"}
        L.append(f"| {s} | {_f(a['gap'], sign=True)}{fl.get(a['flag'], '')} ({_f(a['gap_like'], sign=True)}) | "
                 f"{_f(sh['gap'], sign=True)}{fl.get(sh['flag'], '')} | {_f(o['gap'], sign=True)} | "
                 f"{_f(a['gain'], sign=True)} / {_f(sh['gain'], sign=True)}{fl.get(sh['flag'], '')} |")
    L += ["", f"Means: restarts on all levels {_f(r['all']['mean_gap'], sign=True)} (null-read "
          f"{_f(r['all']['mean_gap_like'], sign=True)}); on the shared five {_f(r['shared']['mean_gap'], sign=True)}; "
          f"one run {_f(r['one_run']['mean_gap'], sign=True)}. `^` at least the top level; the null covers all five "
          "shared levels within 320,000 evaluations, so its null-read gap there is not informative.", ""]
    return L


def _render_ert(e):
    L = ["## 10. Expected evaluations to success [ert] (REQ-ANA-113)", "",
         "ERT = evaluations spent by all runs (a success to its first exact hit, (gen_exact + 1) × 100 for tree GP, "
         "evals_exact for CGP; a failure, CPU stop or kill at the budget E) / successes: the expected cost of "
         "restarting E-evaluation runs until one succeeds. Per set and proven level on s2_cal_a0 + _a16 (tree GP "
         "base) and s2_cal_cgp (one CGP configuration per set; its gaps reproduce remedy.json). The null: "
         "U(<K) + (U(K) + 1)/2 evaluations on average for a target drawn from level K (expected), or U(≤K) to cover the "
         "level (cover). Gap = log₂(null) − log₂ ERT; median over cells with a success (bits).", "",
         "| system | budget | domain | cells | no success | median gap (expected) | median gap (cover) |",
         "|---|---|---|---|---|---|---|"]
    for sysname, by in e["summary"].items():
        for B, dd in by.items():
            for d, v in dd.items():
                L.append(f"| {sysname} | {B} × E₀ | {d} | {v['n_cells']} | {v['n_no_success']} | "
                         f"{_f(v['median_gap_expected'], sign=True)} | {_f(v['median_gap_cover'], sign=True)} |")
    L += ["", "At 16 × E₀ only the five budget levels per set have runs. A level's gap is GP against blind search at "
          "that level, not at the reach: it falls with size, and the median mixes easy and hard levels; at the 50 "
          "percent point an ERT gap sits about a bit below a K₅₀-style gap. Per-level values in the JSON.", ""]
    return L


def _render_bench(b):
    L = ["## 11. Benchmark cells [bench]", "",
         "s2_ho_bench: 55 cells (one proven target each), 30 tree GP runs at E₀ (the row count and successes equal "
         "stage2.json's). Per cell GP's success share at E₀ against the null's (1 if U(≤K) ≤ 20,000, else its share "
         "of level K); gap = log₂(null's expected evaluations, U(<K) + (U(K) + 1)/2) − log₂(GP's ERT at E₀). Both are "
         "expected evaluations to the first success on the same target: GP's over seeds (restarts of E₀ runs), the "
         "null's over the target's position within its level, the uniform position its registered success share "
         "assumes. The audit's d10 used U(≤K), the null's cost to cover the whole level: up to 1 bit in GP's favour "
         "(column cover). Floor: the null's expected evaluations ≤ 100, one GP generation.", "",
         "| domain | cells | GP below / above / tie | with a success | median gap | median (cover) | median, no floor |",
         "|---|---|---|---|---|---|---|"]
    for d in ("real", "bool"):
        v = b[d]
        L.append(f"| {d} | {v['n_cells']} | {v['below']} / {v['above']} / {v['tie']} | {v['n_with_success']} | "
                 f"{_f(v['median_gap_expected'], sign=True)} | {_f(v['median_gap_cover'], sign=True)} | "
                 f"{_f(v['median_gap_expected_nonfloor'], sign=True)} |")
    return L + [""]


def _render_operon(o):
    f, u = o["facts"], o["budget_use"]
    py = ("the installed pyoperon's SymbolicRegressor fits on every row (training range 0 to the row count; optimizer "
          "batch size 0 = all samples)" if f.get("pyoperon_all_rows") and f.get("pyoperon_batch_default_all")
          else "pyoperon not checked")
    L = ["## 12. Operon [operon]", "",
         f"Training data ({f['source']}): {f['training_points']} points drawn once per run from its seed on (2, 3) "
         f"and passed to one fit ({'checked' if f['draws_once'] and f['one_fit'] else 'NOT as stated'}); {py}: a "
         "fixed training set for the whole run. Budget use (s2_cal_operon + s2_ho_operon, medians of count / budget):"]
    L.append("; ".join(f"{c}: evaluation_count {_f(v['eval_share'], 3)}, residual {_f(v['residual_share'], 3)}, "
                       f"Jacobian {_f(v['jacobian_share'], 3)} ({v['n']} runs)" for c, v in u.items()) +
             f". Real gap at 16 × E₀ (`remedy.null_gaps`, equal to remedy.json): symbolic "
             f"{_f(o['gap_16']['symbolic']['mean_gap'], sign=True)}, default {_f(o['gap_16']['default']['mean_gap'], sign=True)} "
             f"over {o['gap_16']['symbolic']['n_sets']} real sets.")
    return L


def main(argv=None, root=None):
    from reach.paths import ROOT
    root = Path(root or ROOT)
    res = build(root)
    (root / "data/analysis/revision_exploratory.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (root / "docs/status/revision_exploratory.md").write_text(render(res))
    print(f"written data/analysis/revision_exploratory.json and docs/status/revision_exploratory.md "
          f"({res['cpu_seconds']} CPU s)")


if __name__ == "__main__":
    main()

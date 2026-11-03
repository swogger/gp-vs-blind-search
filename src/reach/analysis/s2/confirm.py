"""Confirmatory calibration analysis (Phase 6; Stage 1, OSF a3spg) and its DRYRUN on pilot data (REQ-ANA-09).

    .venv/bin/python -m reach.analysis.s2.confirm        -> data/analysis/calibration.json, docs/status/calibration_report.md

One code path for both: `build(PHASE_MAPS["calibration"])` on the s2_cal_* phases with the registered constants, and
`build(PHASE_MAPS["pilot"])` for the dry run (`reach.analysis.s2.dryrun`). Constants are those filed in Stage 1, typed
here once and never recomputed from data (REGISTERED).

Which runs enter which test (docs/design/analysis.md): H1, H2 and H4 read the reach phases only (calibration: s2_cal_a0
and s2_cal_a16, the 5 main seeds on every level); the allocation seeds enter only the best split. H2 uses every set of
the domain except the 6-input size checks, which are tested by H2s against a law fitted on the 5-input sets only; the
redundancy controls stay in H2 (whether alphabet-based units fail on them is their purpose).
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from reach.analysis.k50 import TALLY, counting, k50
from reach.analysis.s2 import (alloc, cfgint, contrasts, families, invariance, law, nullgap, predictor, rand, runtime,
                               theory, within)
from reach.analysis.s2.load import E0, outcome, read_phase, tidy
from reach.analysis.s2.termination import IncompleteError, exclusions, render_termination, termination  # noqa: F401

UNITS = ("nodes", "bits_uniform", "bits_syn", "bits_sem")
BUDGETS = (0.25, 0.5, 1, 2, 4, 8, 16)
H4_BUDGETS = (1, 2, 4, 8, 16)
REGISTERED = dict(delta=dict(real=0.0098, bool=0.0059), margin_nodes=1.0, deviating_factor=4.0, h3_share=0.6,
                  h4c_margin=dict(real=0.5, bool=1.0), h4e_share=0.75, boot=2000, alpha=0.05,
                  h5_margin=1.0)
SIZE_CHECKS = ("BOOL-NAND-n6", "BOOL-AON-n6")
SET_FEATURES = ("n_vars", "bits_per_node", "alphabet", "redundancy", "undefined_share_initial", "neutral_share_mutation")


def set_features(root, sets):
    """Set-level candidates for H7 (design 9.7): number of variables and the grammar properties of T11
    (data/analysis/properties.json), all computable without runs, so they exist for held-out sets too."""
    from reach import grammar as G
    props = json.loads((Path(root) / "data/analysis/properties.json").read_text())
    out = {}
    for g in sets:
        p = props.get(g, {})
        out[g] = dict({k: p[k] for k in SET_FEATURES if k in p and p[k] is not None}, n_vars=G.get(g).n_inputs)
    return out


def usable_features(feats):
    """Features present for every set and not constant across them (a constant one cannot separate sets)."""
    names = [k for k in SET_FEATURES if all(k in f for f in feats.values())]
    return tuple(k for k in names if len({f[k] for f in feats.values()}) > 1)
CONTROLS = {"F-EML-R": "F-EML", "BOOL-IMP0R-n5": "BOOL-IMP0-n5"}
PHASE_MAPS = {
    "calibration": dict(label="CALIBRATION (confirmatory; Stage 1, OSF a3spg)", targets="calibration", h1_family="declared",
                        tasks="data/registration/stage1_tasks.json",
                        levels="data/registration/levels.json", reach=("s2_cal_a0", "s2_cal_a16"),
                        mech=("s2_cal_mech",), robust=("s2_cal_robust",), alloc=("s2_cal_alloc",),
                        pool=("s2_cal_pool",), rand=("s2_cal_rand",), operon=("s2_cal_operon",)),
    "pilot": dict(label="DRYRUN (pilot data; not confirmatory)", targets="pilot", levels=None, h1_family="observed",
                  reach=("s2_pilot_reach_a", "s2_pilot_reach_b",
                  "s2_pilot_long"), mech=("s2_pilot_mech",), robust=("s2_pilot_robust",), alloc=(), pool=(),
                  rand=("s2_pilot_rand_a", "s2_pilot_rand_b"), operon=()),
}


def declared_h1_cells():
    """The H1 Holm family declared in Stage 1 ("per domain, H1 cells"; cells: set x algorithm A, B or headless chicken x
    budget), as this module builds them: every calibration set x A x the 7 budgets (s2_cal_a0 and s2_cal_a16, read by
    prefix) and every full set x B and HC at E0 (s2_cal_mech). Real 61 and Boolean 75 cells (REQ-ANA-28)."""
    from reach.runner.phases import CAL_SETS, FULL_SETS
    return ([(g, "A", "base", b) for g in CAL_SETS for b in BUDGETS]
            + [(g, a, "base", 1) for g in FULL_SETS for a in ("B", "HC")])


def _rows(root, phases):
    return [r for ph in phases for r in read_phase(ph, root) if "gens_reached" in r or "terminal_state" in r]


_UNITS_MAP = {}
_TARGETS = {}                     # (set, target_id) -> target record (expression, scale) for the H7 comparator


_NO_UNITS = []                    # records left out for want of size units (listed among the exclusions; row 102)


def _with_units(recs):
    """Units present on every record, with nodes from K (targets carry nodes as a float; K is the integer). A run killed
    by the queue has no units in its row: they come from its target file (`_UNITS_MAP`), so it still enters as a failure
    (primary rule; review finding 4). A record whose units cannot be found is left out and listed (`_NO_UNITS`)."""
    for r in recs:
        u = r["units"] or _UNITS_MAP.get((r["set"], r["target_id"]), {})
        r["units"] = dict(u, nodes=float(r["K"]))
    keep = [r for r in recs if all(u in r["units"] for u in UNITS)]
    _NO_UNITS.extend(r for r in recs if not all(u in r["units"] for u in UNITS))
    return keep


# The target files each Stage 2 phase reads (guardrail row 136): a report loads all of a phase's files in one
# `_load_units` call, so killed rows of every file keep their units (checked against the registered tasks' --targets).
STAGE2_TARGET_FILES = {"s2_cal_cgp": ("calibration",), "s2_ho_a0": ("heldout",), "s2_ho_a16": ("heldout",),
                       "s2_ho_bench": ("bench",), "s2_ho_operon": ("heldout", "bench"), "s2_ho_cgp": ("heldout",)}


def _load_units(root, sets, name):
    """Units (and target records) of the sets' target files `name`: one file name, or several (a phase that reads
    more than one file, STAGE2_TARGET_FILES). Each call replaces the maps, so every file a group of records needs is
    loaded in the same call; a target id found in two files with other units is refused (guardrail row 136)."""
    names = (name,) if isinstance(name, str) else tuple(name)
    _UNITS_MAP.clear()
    _TARGETS.clear()
    for g in sets:
        for n in names:
            p = Path(root) / f"data/targets_s2/{g}/{n}.json"
            if p.exists():
                for t in json.loads(p.read_text()):
                    k = (g, t["target_id"])
                    if k in _UNITS_MAP and _UNITS_MAP[k] != t["units"]:
                        raise ValueError(f"{g} {t['target_id']}: in two target files ({names}) with other units")
                    _UNITS_MAP[k] = t["units"]
                    _TARGETS[k] = t


def lower_bound_sizes(recs, constructed, ceilings, parities, units_fn):
    """Sensitivity (exploratory; not registered): a constructed target's true minimum size lies in (ceiling, K]. Return
    copies with K and units at the lower bound, the smallest size above the ceiling with the set's parity; proven
    targets are unchanged."""
    out = []
    for r in recs:
        r = dict(r)
        if (r["set"], r["target_id"]) in constructed:
            k = ceilings[r["set"]] + 1
            while k % 2 not in parities[r["set"]]:
                k += 1
            r["K"], r["units"] = k, dict(units_fn[r["set"]](k), nodes=float(k))
        out.append(r)
    return out


def _sensitivity_inputs(root, sets):
    """Constructed targets (from the target files), ceilings and parities (enumeration counts), size units."""
    from reach import grammar as G
    from reach.analysis.s2.phase1 import enum_dir
    from reach.units import SizeUnits
    cons, ceil, par, ufn = set(), {}, {}, {}
    for g in sets:
        for name in ("calibration", "pilot"):
            p = Path(root) / f"data/targets_s2/{g}/{name}.json"
            if p.exists():
                cons |= {(g, t["target_id"]) for t in json.loads(p.read_text()) if t.get("constructed")}
        levels = json.loads((enum_dir(g) / "counts.json").read_text())["levels"]
        ks = [st["K"] for st in levels if st["U"] > 0 and st["K"] >= 3]
        ceil[g], par[g] = max(ks), {k % 2 for k in ks}
        ufn[g] = SizeUnits(G.get(g), levels).all
    return cons, ceil, par, ufn


def _k50_signed(size, y, cluster):
    """K50 = -b0/b1 of the logistic fit (the gap midpoint when separated, D088); NaN when degenerate or when the slope
    is not negative (design 9.1, as `contrasts._k50` and `law._k50`; guardrail row 100)."""
    est, _, _, b1 = k50(size, y, cluster, boot=0)
    return float(est) if not np.isnan(b1) and b1 < 0 else float("nan")


def ranking_bits_sem(recs, sets):
    """H2 rule 6 ("report the ranking of sets in bits_sem"): the sets ordered by K50 in bits_sem at E0, largest reach
    first; a set without a K50 (degenerate or rising) goes last."""
    out = []
    for g in sets:
        rr = [r for r in recs if r["set"] == g]
        out.append(dict(set=g, k50=_k50_signed([r["units"]["bits_sem"] for r in rr], [r["y"] for r in rr],
                                               [r["target_id"] for r in rr]) if rr else float("nan")))
    return sorted(out, key=lambda x: -x["k50"] if np.isfinite(x["k50"]) else float("inf"))


def k50_difference(a, b, unit="nodes", boot=2000, seed=20261020, margin=None):
    """K50(a) - K50(b) for two independent samples (different targets), resampling targets within each (H2c). A K50
    with a slope that is not negative is NaN; replicates without a finite difference are dropped and counted. Stage 1,
    Missing data: "if more than 10% are dropped, the CI is not estimable and the hypothesis is not supported" (row 100).
    holds: estimable, finite, and the 95% CI within +-margin (REGISTERED margin_nodes by default)."""
    def by(recs):
        d = defaultdict(list)
        for r in recs:
            d[r["target_id"]].append(r)
        return d
    ba, bb = by(a), by(b)

    def k(d, ids):
        rr = [r for t in ids for r in d[t]]
        return _k50_signed([r["units"][unit] for r in rr], [r["y"] for r in rr], [r["target_id"] for r in rr])
    m = REGISTERED["margin_nodes"] if margin is None else margin
    rng = np.random.default_rng(seed)
    est = k(ba, list(ba)) - k(bb, list(bb))
    reps = [k(ba, list(rng.choice(list(ba), len(ba)))) - k(bb, list(rng.choice(list(bb), len(bb)))) for _ in range(boot)]
    reps = [x for x in reps if np.isfinite(x)]
    lo, hi = np.percentile(reps, [2.5, 97.5]) if reps else (float("nan"),) * 2
    estimable = len(reps) >= 0.9 * boot
    return dict(diff=float(est), lo=float(lo), hi=float(hi), dropped=boot - len(reps), estimable=bool(estimable),
                holds=bool(estimable and np.isfinite(est) and -m <= lo and hi <= m))


PHASE_KEYS = ("reach", "mech", "robust", "alloc", "pool", "rand", "operon")
SENSITIVITY_KEYS = ("h1", "h1_holm", "h2", "h2c", "h2s", "h2r", "h3", "h4_model", "h4", "h5", "h5_mcnemar", "h6", "h6b",
                    "h6c", "h8", "h8b", "h8c", "h9a", "holm", "alloc")


def check_complete(phmap, root, allow_incomplete=False):
    """Termination states of every phase of `phmap`; on the confirmatory phases (phmap["tasks"]: the registered task
    lists) refuse (IncompleteError) unless every registered task has exactly one row, or allow_incomplete is set."""
    root = Path(root)
    reg = root / phmap["tasks"] if phmap.get("tasks") else None
    t = termination(root, [ph for k in PHASE_KEYS for ph in phmap.get(k, ())], registered=reg)
    t["confirmatory_check"] = reg is not None
    if reg is not None and not t["registered"]:
        t["complete"] = False
        t["note"] = f"registered task lists not found: {phmap['tasks']}"
    if reg is not None and not t["complete"] and not allow_incomplete:
        bad = [f"{ph} ({v['rows']} rows of {v['tasks']} tasks, {v['missing']} missing)"
               for ph, v in t["phases"].items() if not v["complete"]]
        raise IncompleteError("confirmatory results refused: registered tasks without exactly one row: "
                              + ("; ".join(bad) or t.get("note", "")) + ". Run with --allow-incomplete to report "
                              "them labelled not confirmatory.")
    return t


def h3(pool_rows, domain_of, declared, censored_as_failure=True):
    """H3 per domain on the pool (s2_cal_pool; nodes): the function fixed-effects coefficient on size with its
    function-clustered 95% CI below 0, and the share of functions with a negative Kendall tau at least the registered
    0.6 (undefined tau: not negative, in the denominator; D079 b). Both parts must hold. A domain without pool runs is
    'missing: no data' in the declared (confirmatory) mode."""
    out = {}
    for d in ("real", "bool"):
        rows = [r for r in pool_rows if r.get("set") and domain_of(r["set"]) == d]
        if not rows:
            if declared:
                out[d] = dict(status="missing: no data", holds=False)
            continue
        recs = tidy(rows, (1,), censored_as_failure)
        rk = within.rank_share(recs)
        try:
            fe = within.fixed_effects(recs)
        except (ValueError, IndexError, np.linalg.LinAlgError) as e:
            out[d] = dict(status=f"not estimable: {e}", rank_share=rk, holds=False)
            continue
        fs = bool(np.isfinite(fe["hi"]) and fe["hi"] < 0)
        ss = bool(np.isfinite(rk["share_negative"]) and rk["share_negative"] >= REGISTERED["h3_share"])
        out[d] = dict(status="tested", unit="nodes", fixed_effects=fe, rank_share=rk, fe_supported=fs,
                      share_supported=ss, registered_share=REGISTERED["h3_share"], holds=fs and ss,
                      pooled_slope_no_fe=within.pooled_slope(recs))
    return out


def alloc_runs(a_rows, al_rows):
    """{set: (allocation targets, runs)}: per set with allocation runs, the allocation targets' runs from the reach
    phases (main seeds) and the allocation phase (the other 27 seeds); without an allocation phase (DRYRUN) every
    reach run of the set. One run per target and seed (the longest), K = K_construction."""
    src = al_rows if al_rows else a_rows
    out = {}
    for g in sorted({r.get("set") for r in src if r.get("set")}):
        tids = {r["target_id"] for r in src if r.get("set") == g}
        one = {}
        for r in [r for r in a_rows + al_rows if r.get("set") == g and r.get("target_id") in tids]:
            k = (r["target_id"], r.get("seed"))
            if k not in one or r.get("generations", 0) > one[k].get("generations", 0):
                one[k] = r
        out[g] = (tids, [dict(r, K=r.get("K_construction", r.get("K"))) for r in one.values()])
    return out


def best_split(a_rows, al_rows, boot, censored_as_failure=True):
    """Stage 1: "allocation (one run or a best split from 32 seeds)"; "the extra allocation seeds enter only the best
    split". Per set with allocation runs (`alloc_runs`): best split per budget 1-16 x E0 over lengths 25-200
    generations (`alloc.best_split_ci`, the best length re-selected in every replicate)."""
    out = {}
    for g, (tids, runs) in alloc_runs(a_rows, al_rows).items():
        per = defaultdict(set)
        for r in runs:
            per[r["target_id"]].add(r.get("seed"))
        budgets = {}
        for b in H4_BUDGETS:
            ci = alloc.best_split_ci(runs, b, boot=boot, censored_as_failure=censored_as_failure)
            ch = ci.pop("chosen_lengths")
            ci["chosen_share"] = {str(L): ch.count(L) / len(ch) for L in sorted(set(ch))} if ch else {}
            budgets[str(b)] = ci
        out[g] = dict(targets=len(tids), seeds_per_target=[min(map(len, per.values())), max(map(len, per.values()))],
                      runs=len(runs), lengths=[25, 50, 100, 200], budgets=budgets)
    return out


def sizes(recs):
    """Constructed-size labels per set (Stage 1: "Sizes above a set's enumeration ceiling are constructed sizes (upper
    bounds) and are labelled so"): levels with K_proven (proven minimum, = K_construction) and levels whose targets
    are constructed (K_proven unknown: constructed K, an upper bound on the minimum size)."""
    out = {}
    for g, rr in recs.items():
        cons = {r["K"] for r in rr if r.get("K_proven") is None or r.get("K_proven") != r["K"]}
        out[g] = dict(proven_levels=sorted({r["K"] for r in rr} - cons), constructed_levels=sorted(cons),
                      constructed_targets=len({r["target_id"] for r in rr if r["K"] in cons}),
                      label="constructed K: an upper bound on the minimum size (above the enumeration ceiling)")
    return out


def censored_runs(records):
    """The number of runs with a censored record (tidy records: one per run and budget), each run counted once
    however many budgets it is censored at (guardrail row 116). A run: set, target, seed, algorithm and configuration
    (and budget for Operon, which has one run per budget)."""
    return len({(r.get("set"), r.get("target_id"), r.get("seed"), r.get("algorithm"), r.get("config"),
                 r.get("budget") if r.get("algorithm") == "operon" else None) for r in records if r.get("censored")})


def _budgets_of(rr):
    return tuple(b for b in H4_BUDGETS if any(r["budget"] == b for r in rr))


def _null_levels(sets):
    """The enumeration counts of each set (the primary null, H6b)."""
    from reach.analysis.s2.phase1 import enum_dir
    return {g: json.loads((enum_dir(g) / "counts.json").read_text())["levels"] for g in sets}


def robust_parts(res, rob, recs, deltas, boot):
    """H8b and H8c (D081 d, e) from the robustness records `rob` {set: records at E0 and 4 x E0} and the base runs
    `recs` (regime A, base configuration) on the same targets and seeds. H8c in the domain's H2 report unit (the
    preferred unit; bits_sem under rule 6), against the domain's Delta, over the sets with the configuration."""
    from reach import grammar as G
    by_cfg = defaultdict(dict)
    for g, rr in rob.items():
        for cfg in sorted({r["config"] for r in rr}):
            cr = [r for r in rr if r["config"] == cfg]
            pairs = {(r["target_id"], r["seed"]) for r in cr}
            base = [r for r in recs.get(g, []) if r["budget"] in nullgap.H8_BUDGETS
                    and (r["target_id"], r["seed"]) in pairs]
            with counting("H8b"):
                res["h8b"][f"{g}|{cfg}"] = nullgap.h8b(base, cr, boot=boot)
            by_cfg[(G.get(g).kind, cfg)][g] = ([r for r in base if r["budget"] == 1], [r for r in cr if r["budget"] == 1])
    for (d, cfg), per in sorted(by_cfg.items()):
        h2 = res["h2"].get(d, {})
        unit = h2.get("report_unit", "bits_sem")
        with counting("H8c"):
            out = cfgint.h8c({g: v[0] for g, v in per.items()}, {g: v[1] for g, v in per.items()}, unit, deltas[d],
                             boot=boot)
        for g, v in out.items():
            res["h8c"][f"{g}|{cfg}"] = dict(v, preferred=h2.get("preferred"))


def _drop_reps(res):
    """Bootstrap replicates kept for the Holm step (keys starting with '_') stay out of the results."""
    for k in ("h6b", "h8c"):
        for v in res.get(k, {}).values():
            v.pop("_reps", None)


RUNTIME_MODEL_REGISTERED = True     # the run-time model's tests (H4d-f) and comparison are part of Stage 1
RUNTIME_MODEL = RUNTIME_MODEL_REGISTERED   # tests switch the default off for speed (tests/conftest.py); main passes True


def runtime_tests(a_rows, recs, in_band, main, budgets_of, al_runs, boot, censored_as_failure, d):
    """H4d-f and the Weibull-lognormal comparison for one domain (D080 d-h): per main set, the Weibull fit on regime
    A's runs of the budget levels with its target bootstrap (`runtime.set_model`; allocation runs where the set has
    them), the held-out comparison (`theory.compare_families`), and the domain's three Holm members."""
    models, comp, counts, failed = {}, {}, {}, {}
    for g in main:
        rows = [dict(r, units=_UNITS_MAP.get((g, r["target_id"])) or r.get("units") or {}) for r in a_rows
                if r.get("set") == g and r.get("algorithm", "A") == "A" and in_band(g, r.get("K_construction",
                                                                                              r.get("K")))]
        rows = [r for r in rows if "bits_sem" in r["units"]]
        runs, counts[g] = runtime.fit_runs(rows, censored_as_failure)
        band = [r for r in recs[g] if in_band(g, r["K"]) and r["budget"] in H4_BUDGETS]
        if not runs or not any(r["gen_exact"] is not None for r in runs):
            failed[g] = "no runs with a hit"
            continue
        al = al_runs.get(g, (None, None))[1]
        models[g] = runtime.set_model(runs, band, al, budgets_of(g), boot, censored_as_failure=censored_as_failure)
        comp[g] = theory.compare_families(runs)
    tests = runtime.domain_tests(models, REGISTERED["h4c_margin"][d], REGISTERED["h4e_share"])
    for g, why in failed.items():                                          # a set without a fit: not estimable
        tests["H4d"]["sets"][g] = dict(estimable=False, p=float("nan"), holds=False, note=why)
        tests["H4d"].update(p=None, untestable=f"not estimable in {g} ({why})", holds=False)
    tot = {f: sum(c["heldout_loglik"][f] for c in comp.values()) for f in ("weibull", "lognormal")}
    sets = {g: dict(point=m["point"], cells=m["cells"], weibull=m["weibull"], lstar_ci=runtime.lstar_ci(m)[1],
                    comparison=comp[g], fit_counts=counts[g]) for g, m in models.items()}
    return dict(sets=sets, failed=failed, tests=tests, boot=boot,
                comparison=dict(heldout_loglik=tot, preferred=max(tot, key=tot.get) if comp else None,
                                sets_preferring_weibull=sum(c["preferred"] == "weibull" for c in comp.values()),
                                sets=len(comp)))


def _guarded(fn, *args, **kw):
    """H7 is not a Stage 1 test: a failure in its selection or a comparator is recorded, never aborts the Stage 1
    report (review of 8e66047..4ac972b; guardrail row 119). Returns (value, None) or (None, "Type: message")."""
    try:
        return fn(*args, **kw), None
    except Exception as e:                                         # noqa: BLE001 (any numerical failure)
        return None, f"{type(e).__name__}: {e}"


def _comparator(mod, recs, vec, d, left):
    fit, err = _guarded(lambda: mod.summary(mod.fit_domain(recs, vec, d)))
    return dict(fit, left_out=left) if err is None else dict(error=err, left_out=left)


def _comp1_text(c1):
    """Comparator 1 (Graff, Poli and Flores 2013; D085) in the H7 report line: its form and 5-fold RSE per budget."""
    if not c1:
        return "Graff, Poli and Flores (2013) comparator not fitted (no target expressions)"
    if c1.get("error"):
        return f"Graff, Poli and Flores (2013) comparator failed ({c1['error']})"
    per = ", ".join(f"{b}x {mb['cv_rse']:.3f} (m {mb['m']})" if mb.get("cv_rse") is not None else f"{b}x n/a"
                    for b, mb in sorted(c1["budgets"].items(), key=lambda kv: float(kv[0])))
    return f"Graff, Poli and Flores (2013) comparator, {c1['form']} indicators, 5-fold RSE {per}"


def _comp2_text(c2):
    """Comparator 2 (Graff and Poli 2010; D084) in the H7 report line: its measure and 5-fold RSE per budget."""
    if not c2:
        return "Graff and Poli (2010) comparator not fitted (no target expressions)"
    if c2.get("error"):
        return f"Graff and Poli (2010) comparator failed ({c2['error']})"
    per = ", ".join(f"{b}x {mb['cv_rse']:.3f} (m {mb['m']})" if mb.get("cv_rse") is not None else f"{b}x n/a"
                    for b, mb in sorted(c2["budgets"].items(), key=lambda kv: float(kv[0])))
    return f"Graff and Poli (2010) comparator, d {c2['measure']}, 5-fold RSE {per}"


def h7_unit_source(rule6):
    """H7's unit label (D082): the H2-preferred unit (D079 a), or bits_sem when no unit is preferred (rule 6)."""
    return "no preferred unit, rule 6: bits_sem; D082" if rule6 else "H2-preferred unit, D079 a; D082"


def build(phmap, root=None, boot=None, delta=None, allow_incomplete=False, censored_as_failure=True, _sub=False,
          runtime_model=None):
    """The confirmatory analysis. allow_incomplete: report even when registered tasks lack rows (labelled not
    confirmatory). censored_as_failure=False: the Stage 1 sensitivity analysis (CPU-limited runs excluded); `_sub`
    marks that inner build (no H7, no lower-bound sensitivity, no completeness or nested sensitivity). runtime_model:
    compute H4d-f and the Weibull-lognormal comparison (default RUNTIME_MODEL; both entry points pass True)."""
    use_rt = RUNTIME_MODEL if runtime_model is None else runtime_model
    from reach import grammar as G
    from reach.paths import ROOT
    root = Path(root or ROOT)
    boot = boot or REGISTERED["boot"]
    deltas = dict(REGISTERED["delta"], **(delta or {}))
    caf = censored_as_failure
    if not _sub:
        TALLY.clear()                                                      # D088's counts (`separation_summary`)
    comp = None if _sub else check_complete(phmap, root, allow_incomplete)
    _NO_UNITS.clear()
    a_rows, m_rows, r_rows = (_rows(root, phmap[k]) for k in ("reach", "mech", "robust"))
    sets = sorted({r.get("set") for r in a_rows if r.get("set")})
    dom = {g: G.get(g).kind for g in sets}
    _load_units(root, sets, phmap["targets"])
    lv = json.loads((root / phmap["levels"]).read_text()) if phmap.get("levels") else None
    rs = rand.records(_rows(root, phmap.get("rand", ())), _TARGETS, caf)       # random sampling (D080 a)
    al_rows = _rows(root, phmap.get("alloc", ()))                         # best split from the allocation seeds
    al_runs = alloc_runs(a_rows, al_rows)
    in_band = (lambda g, K: K in lv[g]["budget"]) if lv else (lambda g, K: True)  # noqa: E731 H4: budget levels only
    res = dict(label=phmap["label"], registered=REGISTERED, h1=[], h2={}, h2c={}, h2s={}, h2r={}, h4={}, h5={}, h6={},
               h7={}, h4_model={} if use_rt else dict(note="not run in this build (runtime_model=False)"),
               h8=[], h9a=[], h1_holm={}, holm={}, h6b={}, h6c={}, h8b={}, h8c={},
               h3={}, h5_mcnemar={}, alloc={}, sizes={}, not_run=[])
    if comp is not None:
        res["completeness"] = comp
        if comp["confirmatory_check"] and not comp["complete"]:
            res["label"] += " — INCOMPLETE: registered tasks without a row (allow_incomplete); not confirmatory"

    recs = {g: _with_units(tidy([r for r in a_rows if r.get("set") == g], BUDGETS, caf)) for g in sets}
    res["sizes"] = sizes(recs)
    for g in sets:                                                         # H1: every cell of regime A
        for b in sorted({r["budget"] for r in recs[g]}):
            cell = [r for r in recs[g] if r["budget"] == b]
            res["h1"].append(dict(set=g, algorithm="A", config="base", budget=b, **law.slope_ci(cell, boot=boot)))
    mech = defaultdict(list)
    for g in sorted({r.get("set") for r in m_rows}):
        mrecs = _with_units(tidy([r for r in m_rows if r.get("set") == g], (1,), caf))
        for algo in ("B", "HC"):
            cell = [r for r in mrecs if r["algorithm"] == algo]
            if cell:
                res["h1"].append(dict(set=g, algorithm=algo, config="base", budget=1, **law.slope_ci(cell, boot=boot)))
        mech[g] = mrecs
    res["h1_holm"] = law.holm_cells(res["h1"], lambda g: G.get(g).kind, REGISTERED["alpha"],      # Stage 1: Holm per
                                    declared=declared_h1_cells() if phmap.get("h1_family") == "declared" else None)

    gaps = {}                                                              # H6c: the gap in reach per domain
    for d in ("real", "bool"):                                             # H2, H2s, H4, H6b per domain
        main = [g for g in sets if dom[g] == d and g not in SIZE_CHECKS]
        e0 = [r for g in main for r in recs[g] if r["budget"] == 1]
        if main:                                                           # H6b and H6c (D081 b, c)
            e0_by = {g: [r for r in recs[g] if r["budget"] == 1] for g in main}
            with counting("H6b, H6c"):
                res["h6b"][d] = nullgap.h6b(e0_by, _null_levels(main), E0, boot)
                gaps[d] = nullgap.domain_gap(e0_by, E0, boot)
        if len(main) < 3:
            continue
        inv = invariance.invariance_test(e0, list(UNITS), deltas[d], boot=boot)
        pref = invariance.preferred_unit(inv, REGISTERED["alpha"])          # rules 4 and 6 at 3b8aa5e (D079 a)
        best = pref["unit"]
        per = invariance.per_set_test(e0, best, boot=boot)
        lim = REGISTERED["deviating_factor"] * deltas[d]
        res["h2"][d] = dict(delta=deltas[d], sets=main, units=inv, preferred=pref["preferred"], report_unit=best,
                            preferred_rule=pref["rule"], invariant_holm=pref["invariant"], per_set=per,
                            deviating=[s for s, v in per.items() if v["upper95"] > lim], deviating_limit=lim)
        if pref["preferred"] is None:
            with counting("H2 rule 6 ranking"):
                res["h2"][d]["ranking_bits_sem"] = ranking_bits_sem(e0, main)
        res["h2r"][d] = rand.h2r(rs["by_set"], main, deltas[d], UNITS, boot)  # H2r (D080 b)
        checks = [g for g in sets if dom[g] == d and g in SIZE_CHECKS]
        if checks:
            both = e0 + [r for g in checks for r in recs[g] if r["budget"] == 1]
            ps = invariance.per_set_test(both, best, boot=boot, train=main)
            res["h2s"][d] = dict(unit=best, preferred=pref["preferred"], delta=deltas[d],
                                 sets={g: ps[g] for g in checks},
                                 holds=all(ps[g]["upper95"] <= deltas[d] for g in checks))
        h4 = [r for g in main for r in recs[g] if r["budget"] in H4_BUDGETS and in_band(g, r["K"])]
        with counting("H4 per set, H4c"):
            res["h4"][d] = dict(law_nodes=law.fit_law(h4, "nodes", set_specific=True),     # set intercepts (D079 c)
                                law_bits_sem=law.fit_law(h4, "bits_sem", set_specific=True),
                                curvature_nodes=law.curvature(h4, "nodes", set_specific=True), per_set={},
                                levels="budget levels only" if lv else "all levels",
                                rate_bits_sem=law.rate_ci(h4, "bits_sem", boot=boot, set_specific=True),  # H4b, H6a
                                law_nodes_one_intercept=law.fit_law(h4, "nodes"),       # reported beside it (D079 c)
                                law_bits_sem_one_intercept=law.fit_law(h4, "bits_sem"),
                                curvature_nodes_one_intercept=law.curvature(h4, "nodes"),
                                rate_bits_sem_one_intercept=law.rate_ci(h4, "bits_sem", boot=boot),
                                h4c=law.slope_equality(                                  # H4c (D081 a)
                                    {g: [r for r in recs[g] if in_band(g, r["K"])] for g in main},
                                    {g: _budgets_of(recs[g]) for g in main}, REGISTERED["h4c_margin"][d], "bits_sem",
                                    boot))
        if use_rt:                                                         # H4d-f; Weibull against lognormal (D080)
            with counting("H4d-f"):
                res["h4_model"][d] = runtime_tests(a_rows, recs, in_band, main, lambda g: _budgets_of(recs[g]),
                                                   al_runs, boot, caf, d)
        if _sub:
            continue                                                       # H7 is not a Stage 1 test
        h7 = [r for g in main for r in recs[g] if r["budget"] in H4_BUDGETS]          # several budgets: g estimable
        feats = set_features(root, main) if (root / "data/analysis/properties.json").exists() else {}
        cand = usable_features(feats) if feats else ()
        for r in h7:
            r["features"] = feats.get(r["set"], {})
        rule6 = pref["preferred"] is None                            # D082: the H2-preferred unit; bits_sem (rule 6)
        sel, sel_err = _guarded(predictor.select, h7, [best], features=cand)
        sel = sel or dict(unit=None, budget_term=None, feature=None, loso_logloss=float("nan"), params=None)
        from reach.analysis.s2 import gp2010, gpf2013      # comparators 1 (D085) and 2 (D084): fitted here, frozen in 3c
        vec, left = gp2010.vectors_for(h7, _TARGETS)
        h7v = [r for r in h7 if (r["set"], r["target_id"]) in vec]
        comp = _comparator(gpf2013, h7v, vec, d, left) if h7v else None
        comp2 = _comparator(gp2010, h7v, vec, d, left) if h7v else None
        res["h7"][d] = dict(selected_unit=sel["unit"], rule6=rule6, unit_source=h7_unit_source(rule6),
                            budget_term=sel["budget_term"], feature=sel["feature"],
                            candidates=list(cand), budgets=list(H4_BUDGETS), loso_logloss=sel["loso_logloss"],
                            params=sel["params"], comparator=comp, comparator2=comp2, note="frozen in step 3c")
        if sel_err:
            res["h7"][d]["error"] = sel_err
    res["h6c"] = nullgap.h6c(gaps)
    cons, ceil, par, ufn = _sensitivity_inputs(root, sets)               # sensitivity: constructed sizes at lower bound
    res["sensitivity_lower_bound"] = {}
    for d in (() if _sub else ("real", "bool")):
        main = [g for g in sets if dom[g] == d and g not in SIZE_CHECKS]
        if len(main) < 3:
            continue
        e0 = lower_bound_sizes([r for g in main for r in recs[g] if r["budget"] == 1], cons, ceil, par, ufn)
        inv = invariance.invariance_test(e0, list(UNITS), deltas[d], boot=boot)
        h1 = [law.slope_ci([r for r in e0 if r["set"] == g], boot=boot) for g in main]
        res["sensitivity_lower_bound"][d] = dict(
            constructed_runs=sum(1 for r in e0 if (r["set"], r["target_id"]) in cons), runs=len(e0), units=inv,
            preferred=invariance.preferred_unit(inv, REGISTERED["alpha"])["preferred"],
            h1_cells_below_zero=sum(c["hi"] < 0 for c in h1),
            h1_cells=len(h1), note="exploratory: constructed sizes at their lower bound (registered: upper bound)")
    for c, partner in CONTROLS.items():                                   # H2c: control against partner at E0
        if c in recs and partner in recs:
            e = lambda g: [r for r in recs[g] if r["budget"] == 1]        # noqa: E731
            with counting("H2c"):
                dd = k50_difference(e(c), e(partner), boot=boot)            # >10% dropped: not supported
            res["h2c"][c] = dict(partner=partner, **dd, margin=REGISTERED["margin_nodes"])
    for g in sets:                                                         # H4 per set; H6
        d = dom[g]
        bud = tuple(b for b in H4_BUDGETS if any(r["budget"] == b for r in recs[g]))
        band = [r for r in recs[g] if in_band(g, r["K"])]
        with counting("H4 per set, H4c"):
            per = {u: law.reach_per_doubling(band, u, bud, boot=boot) for u in ("nodes", "bits_sem")}
        if d in res["h4"]:
            res["h4"][d]["per_set"][g] = dict(budgets=list(bud), **{u: per[u] for u in per})
        e0 = [r for r in recs[g] if r["budget"] == 1]
        with counting("H6"):
            est, lo, hi, _ = k50([r["units"]["bits_sem"] for r in e0], [r["y"] for r in e0],
                                 [r["target_id"] for r in e0], boot=boot)
        null = math.log2(E0)                       # the primary null's reach at E0 is log2 E (REQ-ANA-13)
        res["h6"][g] = dict(domain=d, gp_reach_bits_sem=est, lo=lo, hi=hi, null_bits_sem=null, gap=est - null,
                            gap_lo=lo - null, gap_hi=hi - null, bits_per_doubling=per["bits_sem"])
    for g, mrecs in mech.items():                                          # H5: paired on the same targets and seeds
        seeds = {r["seed"] for r in mrecs}
        a = [r for r in recs.get(g, []) if r["budget"] == 1 and r["seed"] in seeds]
        both = a + mrecs
        res["h5"][g] = {}
        for x, y in (("A", "B"), ("HC", "A")):                            # A > B; HC - A within +-1 node or positive
            if any(r["algorithm"] == x for r in both) and any(r["algorithm"] == y for r in both):
                with counting("H5"):
                    c = contrasts.paired_k50_difference(both, x, y, boot=boot, keep_reps=True)
                null = 0.0 if x == "A" else -REGISTERED["h5_margin"]
                c.update(p=law.p_bound(c.pop("reps"), null, "above", 2), test=families.TESTS[f"{x}-{y}"])
                res["h5"][g][f"{x}-{y}"] = c
                res["h5_mcnemar"].setdefault(g, {})[f"{x}-{y}"] = contrasts.mcnemar_per_level(both, x, y)
        b_recs = [r for r in mrecs if r["algorithm"] == "B"]
        if b_recs and rs["by_set"].get(g):                                 # B - RS within +-1 node or positive
            with counting("H5"):
                c = rand.b_minus_rs(b_recs, rs["by_set"][g], boot, REGISTERED["h5_margin"])
            res["h5"][g][rand.CONTRAST] = dict(c, test=families.TESTS[rand.CONTRAST])
    rob = {}
    for g in sorted({r.get("set") for r in r_rows}):                       # H8: H1 under each configuration
        rr = rob[g] = _with_units(tidy([r for r in r_rows if r.get("set") == g], (1, 4), caf))
        for cfg in sorted({r["config"] for r in rr}):
            for b in (1, 4):
                cell = [r for r in rr if r["config"] == cfg and r["budget"] == b]
                if cell:
                    res["h8"].append(dict(set=g, config=cfg, budget=b, **law.slope_ci(cell, boot=boot)))
    robust_parts(res, rob, recs, deltas, boot)                            # H8b and H8c (D081 d, e)
    o_rows = _rows(root, phmap["operon"])                                  # H9a: Operon at 16 x E0, as H1 per cell
    for g in sorted({r.get("set") for r in o_rows if r.get("set")}):
        orecs = _with_units(tidy([r for r in o_rows if r.get("set") == g], (16,), caf))
        for cfg in sorted({r["config"] for r in orecs}):
            cell = [r for r in orecs if r["config"] == cfg]
            res["h9a"].append(dict(set=g, algorithm="operon", config=cfg, budget=16, **law.slope_ci(cell, boot=boot)))
    declared = phmap.get("h1_family") == "declared"
    p_rows = _rows(root, phmap.get("pool", ()))                           # H3 on the shared-function pool
    res["h3"] = h3(p_rows, lambda g: G.get(g).kind, declared, caf)
    with counting("best split"):
        res["alloc"] = best_split(a_rows, al_rows, boot, caf)
    res["holm"] = families.build(res, lambda g: G.get(g).kind, declared,   # Stage 1 Holm
                                 REGISTERED["alpha"], REGISTERED["boot"])
    _drop_reps(res)
    if _sub:
        return res
    res["separation"] = separation_summary(res, boot, TALLY)              # D088: how often a fit was separated
    res["exclusions"] = exclusions(root, [ph for k in PHASE_KEYS for ph in phmap.get(k, ())]) + [
        dict(phase="(analysis)", key=f"{r['set']}|{r['target_id']}|{r.get('algorithm')}|{r.get('seed')}|{r['budget']}",
             set=r["set"], target_id=r["target_id"], seed=r.get("seed"), algorithm=r.get("algorithm"),
             reason="no size units for its target (not in the target file): left out", detail="")
        for r in _NO_UNITS]
    cens = dict(reach=censored_runs([r for g in recs for r in recs[g]]),          # runs, not records (row 116)
                mech=censored_runs([r for g in mech for r in mech[g]]),
                robust=censored_runs(tidy(r_rows, (1, 4))),
                operon=censored_runs(tidy(o_rows, (16,))),
                pool=censored_runs(tidy(p_rows, (1,))), rand=rs["censored_parts"],
                alloc=sum(outcome(r, 200) == "censored" for r in al_rows if r.get("terminal_state") != "error"))
    n = sum(cens.values())
    if n:                                                                  # Stage 1: CPU-limited runs excluded
        sub = build(phmap, root, boot, delta, True, False, _sub=True, runtime_model=use_rt)
        res["sensitivity_cpu_excluded"] = dict(censored_runs=n, by_phase_group=cens, recomputed=True,
                                               note="CPU-limited runs excluded where censored (Stage 1 sensitivity)",
                                               **{k: sub[k] for k in SENSITIVITY_KEYS})
    else:
        res["sensitivity_cpu_excluded"] = dict(censored_runs=0, by_phase_group=cens, recomputed=False,
                                               note="identical to the primary analysis: no CPU-limited run")
    return res


def _f(x, d=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def _fs(x, d=2):
    """A slope or its CI end: ±∞ for a separated fit (D088), else as `_f`."""
    return ("−∞" if x < 0 else "+∞") if isinstance(x, float) and np.isinf(x) else _f(x, d)


def render_h1_cells(res, title, source):
    """The H1 table, one row per cell, in its own file (the main report would pass the 200-line limit; D077)."""
    f = _f
    L = [f"# {title}", "", f"> Index: `docs/README.md`. Generated by `{source}`. **{res['label']}.** Do not edit by hand.",
         "", "## H1: logistic slope on size per cell (nodes; 95% target-bootstrap CI)", "",
         "| Set | algorithm | config | budget | slope [CI] | CI below 0 | dropped / estimable | p (CI inversion) / Holm |",
         "|---|---|---|---|---|---|---|---|"]
    for c in res["h1"]:
        L.append(f"| {c['set']} | {c['algorithm']} | {c['config']} | {c['budget']} | {_fs(c['slope'])} "
                 f"[{_fs(c['lo'])}, {_fs(c['hi'])}] | {'yes' if c['hi'] < 0 else 'no'} | {c['dropped']} / "
                 f"{'yes' if c['estimable'] else 'no'} | {f(c.get('p_below0'), 4)} / {f(c.get('p_holm'), 4)}: "
                 f"{c.get('holm_status', '–')} |")
    return "\n".join(L) + "\n"


FAMILY_ORDER = ("H2", "H4", "H5", "H6", "H8", "H9a")


def render_holm_members(res, title, source, h8_file=None):
    """The Holm families other than H1, one row per member (REQ-ANA-29), in their own file (the main report would pass
    the 200-line limit; D077, D078). The H8 parts per cell and member are in `render_h8` (REQ-ANA-55)."""
    f, fp = _f, families.fmt_p
    L = [f"# {title}", "", f"> Index: `docs/README.md`. Generated by `{source}`. **{res['label']}.** Do not edit by hand.",
         "", "Stage 1, Inference criteria: Holm within each declared family per domain, untestable members counted; p by "
         "inverting each member's registered bound (D078). Untestable members are never supported.", ""]
    for fam in FAMILY_ORDER:
        L += [f"## {fam}", "", "| Domain | member | test | p | Holm p | decision |", "|---|---|---|---|---|---|"]
        for d, h in res["holm"].get(fam, {}).items():
            for mm in h["members"]:
                L.append(f"| {d} | {' '.join(str(x) for x in mm['id'])} | {mm.get('test') or '–'} | {fp(mm['p'])} | "
                         f"{fp(mm['p_holm'])} | {mm['status']} |")
        L.append("")
    L.append(f"H8's parts (H8a cells, H8b and H8c per member): `{h8_file or 'render_h8'}`.")
    return "\n".join(L) + "\n"


def render_h8(res, title, source):
    """H8's parts (D078 reading 5, D081 d-e): the H8a cells (H1 under each robustness configuration at E0 and 4 x E0)
    and, per member, H8b (the sign of GP - null against the base on the same targets and seeds) and H8c (the set x
    configuration interaction against Delta)."""
    f, fp = _f, families.fmt_p
    L = [f"# {title}", "", f"> Index: `docs/README.md`. Generated by `{source}`. **{res['label']}.** Do not edit by hand.",
         "", "## H8a cells: H1 under each robustness configuration (nodes; 95% target-bootstrap CI)", "",
         "| Set | config | budget | slope [CI] | p (CI inversion) |", "|---|---|---|---|---|"]
    for c in res["h8"]:
        L.append(f"| {c['set']} | {c['config']} | {c['budget']} | {_fs(c['slope'])} [{_fs(c['lo'])}, {_fs(c['hi'])}] | "
                 f"{fp(c.get('p_below0'))} |")
    L += ["", "## H8b and H8c per member", "",
          "H8b: GP's K₅₀ gap to the null (bits_sem) for the base and the configuration on the same targets and seeds, "
          "at E₀ and 4 × E₀; holds when both CIs lie on the same side of 0 (D081 d). H8c: excess log-loss of the "
          "other sets' reach shift over the set's own, 95% CI within ±Δ (D081 e).", "",
          "**Power (D081 (e), D082):** H8c is underpowered at the registered size: simulated with no interaction, its "
          "upper CI ends reached about 10 x Δ, recorded before any calibration outcome was seen. Almost every H8c part, "
          "and so most H8 members (p = the largest part's), are \"not supported\" whatever the data; a \"not "
          "supported\" H8 member is not evidence of an interaction.", "",
          "| Set | config | H8b E₀: base / config gap [CI] | H8b 4×E₀: base / config gap [CI] | H8b p | H8c unit: excess "
          "[CI] (shift own / others) | H8c p |", "|---|---|---|---|---|---|---|"]

    def gap(v, b):
        x = v.get("budgets", {}).get(b)
        if x is None:
            return v.get("untestable") or "–"
        return (f"{f(x['base_gap'])} [{f(x['base_lo'])}, {f(x['base_hi'])}] / {f(x['config_gap'])} "
                f"[{f(x['config_lo'])}, {f(x['config_hi'])}]")
    for k in sorted(set(res.get("h8b", {})) | set(res.get("h8c", {}))):
        g, cfg = k.split("|")
        b, c = res.get("h8b", {}).get(k, {}), res.get("h8c", {}).get(k, {})
        L.append(f"| {g} | {cfg} | {gap(b, 1)} | {gap(b, 4)} | {fp(b.get('p'))} | {c.get('unit', '–')}: "
                 f"{f(c.get('excess'), 4)} [{f(c.get('lo'), 4)}, {f(c.get('hi'), 4)}] ({f(c.get('shift_own'))} / "
                 f"{f(c.get('shift_logo'))}){'' if c.get('untestable') is None else ' ' + c['untestable']} | "
                 f"{fp(c.get('p'))} |")
    return "\n".join(L) + "\n"


def render_slopes_gaps(res, title, source):
    """H4's per-set slopes and H4c's pairs; H6b's gaps per level and set (D081 a-c)."""
    f, fp = _f, families.fmt_p
    L = [f"# {title}", "", f"> Index: `docs/README.md`. Generated by `{source}`. **{res['label']}.** Do not edit by hand.",
         "", "## H4: reach per budget doubling per set", "",
         "Beside H4b (exploratory, not a registered test): whether each real set's bits_sem rate has its two-sided 95% "
         "CI below 1; H4b itself is the pooled real rate's one-sided upper 95% bound below 1"
         + "".join(f"; pooled {d} rate {f(h['rate_bits_sem']['rate'])} bits_sem per doubling, upper 95% bound "
                   f"{f(h['rate_bits_sem'].get('upper95'))}" for d, h in res["h4"].items()
                   if d == "real" and "rate_bits_sem" in h) + ".", "",
         "| Set | budgets | nodes per doubling [CI] | bits_sem per doubling [CI] | CI below 1 (exploratory, beside H4b) |",
         "|---|---|---|---|---|"]
    for d, h in res["h4"].items():
        for g, p in h["per_set"].items():
            b = p["bits_sem"]
            below = ("yes" if b["hi"] < 1 else "no") if d == "real" and np.isfinite(b["hi"]) else "–"
            L.append(f"| {g} | {p['budgets']} | {f(p['nodes']['rate'])} [{f(p['nodes']['lo'])}, {f(p['nodes']['hi'])}]"
                     f" | {f(b['rate'])} [{f(b['lo'])}, {f(b['hi'])}] | {below} |")
    L += ["", "## H4c: pairs of per-set slopes (bits_sem per doubling; 95% CI within ±margin; D081 a)", "",
          "| Domain | pair | difference [CI] | margin | within | p |", "|---|---|---|---|---|---|"]
    for d, h in res["h4"].items():
        c = h.get("h4c") or {}
        for pr in c.get("pairs", []):
            L.append(f"| {d} | {pr['sets'][0]} − {pr['sets'][1]} | {f(pr['diff'])} [{f(pr['lo'])}, {f(pr['hi'])}] | "
                     f"±{c['margin']} | {'yes' if pr['holds'] else 'no'} | {fp(pr['p'])} |")
    L += ["", "## H6b: GP − null recovery gap per level at E₀ (mean over targets; D081 b)", "",
          "Gap = GP's share of exact runs minus the primary null's hit probability. Sign changes: consecutive levels "
          "(nodes) between which the gap's sign changes (levels with a gap of 0 skipped).", "",
          "| Domain | set | gap per level (K: gap) | sign changes |", "|---|---|---|---|"]
    for d, h in res.get("h6b", {}).items():
        for g, v in h["per_set"].items():
            L.append(f"| {d} | {g} | " + ", ".join(f"{K}: {m:+.2f}" for K, m, _ in v["levels"]) + " | "
                     + (", ".join(f"{a}-{b}" for a, b in v["sign_changes"]) or "none") + " |")
    return "\n".join(L) + "\n"


def _holm_lines(res, members_file):
    L = ["", "## Holm families (Stage 1, Inference criteria; D078)", "",
         f"Members, tests, p-values and decisions: `{members_file or 'render_holm_members'}`. Per domain, untestable "
         "members (no data, not estimable, saturated, test not built) count in m and are never supported. Resolution: "
         f"replicates beyond the bound at Holm's smallest threshold, at the registered B = {REGISTERED['boot']}.", ""]
    for fam in FAMILY_ORDER:
        for d, h in res.get("holm", {}).get(fam, {}).items():
            r = h["resolution"]
            unt = h.get("untestable", h["m"] - h["supported"] - h.get("not_supported", 0))
            why = "; ".join(f"{v} x {k}" for k, v in h.get("untestable_reasons", {}).items())
            res_txt = ("analytic p" if r["sided"] is None else
                       f"{r['replicates_beyond']:.2f} replicates beyond the bound ({'resolved' if r['resolvable'] else 'NOT resolved'})")
            L.append(f"- **{fam}, {d}:** {h['family']} family of m = {h['m']}, alpha {h['alpha']}: {h['supported']} "
                     f"supported, {h['not_supported']} not supported, {unt} untestable{f' ({why})' if why else ''}; "
                     f"smallest threshold {r['threshold']:.5f}: {res_txt}.")
    return L


def _completeness_lines(res, termination_file):
    t = res.get("completeness")
    if not t:
        return []
    bad = [ph for ph, v in t["phases"].items() if not v["complete"]]
    ex = res.get("exclusions", [])
    return ["## Completeness and exclusions", "",
            f"Checked against {'the registered task lists' if t.get('registered') else 'the phase manifests'}: "
            + ("every task has exactly one row." if t["complete"] else
               f"INCOMPLETE ({', '.join(bad) or t.get('note', '')}): not confirmatory.")
            + f" {len(ex)} rows excluded (each listed with its reason), "
            f"{sum(v['states']['timeout'] for v in t['phases'].values())} CPU-limited runs (failures; excluded in the "
            f"sensitivity analysis). Table per phase and the excluded rows: `{termination_file or 'render_termination'}`.",
            ""]


def _decisions(res):
    """{(family, domain, member id): supported} over H1 and the other Holm families (for the sensitivity summary)."""
    out = {("H1",) + tuple(map(str, (c["set"], c["algorithm"], c["config"], c["budget"]))): c.get("holm_supported")
           for c in res.get("h1", [])}
    for fam, by in res.get("holm", {}).items():
        for d, h in by.items():
            for mm in h["members"]:
                out[(fam, d) + tuple(map(str, mm["id"]))] = mm["supported"]
    return out


def _sensitivity_lines(res):
    s = res.get("sensitivity_cpu_excluded")
    if not s:
        return []
    f = _f
    L = ["", "## Sensitivity: CPU-limited runs excluded (Stage 1)", "",
         f"CPU-limited runs in the primary analysis (counted as failures where censored): {s['censored_runs']} ("
         + ", ".join(f"{k} {v}" for k, v in s["by_phase_group"].items()) + f"; rand: parts of 250,000 draws). "
         f"{s['note']}."]
    if not s["recomputed"]:
        return L
    a, b = _decisions(res), _decisions(s)
    diff = sorted(k for k in a if k in b and a[k] != b[k])
    L.append(f"Holm decisions (H1 and the other families) that differ from the primary analysis: {len(diff)}"
             + (": " + "; ".join(" ".join(k) for k in diff[:12]) + ("; …" if len(diff) > 12 else "") if diff else "")
             + ".")
    for d in res.get("h2", {}):
        p, q = res["h2"][d], s["h2"].get(d, {})
        L.append(f"- **{d}:** preferred unit {p['preferred']} -> {q.get('preferred')}; bits_sem per doubling "
                 f"{f(res['h4'][d]['rate_bits_sem']['rate'])} -> {f(s['h4'].get(d, {}).get('rate_bits_sem', {}).get('rate'))}"
                 f"; H3 holds {res['h3'].get(d, {}).get('holds')} -> {s['h3'].get(d, {}).get('holds')}.")
    for c, h in res.get("h2c", {}).items():
        L.append(f"- **H2c {c}:** holds {h['holds']} -> {s['h2c'].get(c, {}).get('holds')}.")
    for d, h in res.get("h2r", {}).items():
        L.append(f"- **H2r {d}:** holds {h['holds']} -> {s.get('h2r', {}).get(d, {}).get('holds')}.")
    return L


def _h3_lines(res, supplement_file):
    L = ["", "## H3: same function, different sets (pool; nodes)", ""]
    for d, h in res.get("h3", {}).items():
        if h.get("status") != "tested":
            L.append(f"- **{d}:** {h.get('status')}; not supported.")
            continue
        fe, rk = h["fixed_effects"], h["rank_share"]
        L.append(f"- **{d}:** fixed-effects coefficient on size {_f(fe['coef'], 3)} [{_f(fe['lo'], 3)}, {_f(fe['hi'], 3)}]"
                 f" (function-clustered; below 0: {'yes' if h['fe_supported'] else 'no'}); share of functions with a "
                 f"negative tau {rk['negative']}/{rk['functions']} = {_f(rk['share_negative'], 3)} (at least "
                 f"{h['registered_share']}: {'yes' if h['share_supported'] else 'no'}; exploratory, spread only "
                 f"{_f(rk['share_negative_spread_only'], 3)}); H3 holds: {'yes' if h['holds'] else 'no'}.")
    if not res.get("h3"):
        L.append("- No pool phase in this analysis.")
    L.append(f"- Details, McNemar per level, best split and constructed-size labels: "
             f"`{supplement_file or 'render_supplement'}`.")
    return L


def render_supplement(res, title, source):
    """H3 details, exact McNemar per level (H5), the best split per set and budget, and the constructed-size labels."""
    f = _f
    L = [f"# {title}", "", f"> Index: `docs/README.md`. Generated by `{source}`. **{res['label']}.** Do not edit by hand.",
         "", "## H3: counter-instances (functions with a positive tau)", ""]
    for d, h in res.get("h3", {}).items():
        rk = h.get("rank_share") or {}
        L.append(f"- **{d}:** {', '.join(rk.get('counter_instances', [])) or 'none'}; functions without a size spread "
                 f"{rk.get('no_spread', '–')} (undefined tau: not negative, in the denominator).")
    if not res.get("h3"):
        L.append("- No pool phase in this analysis.")
    L += ["", "## H5: exact McNemar per level (seed-matched pairs at E₀)", "",
          "Per level: b (first algorithm exact, second not) / c (the reverse), exact two-sided p; ties carry no "
          "information; '–' without discordant pairs.", "", "| Set | contrast | per level: K b/c p |", "|---|---|---|"]
    for g, cs in res.get("h5_mcnemar", {}).items():
        for k, lv in cs.items():
            L.append(f"| {g} | {k} | " + "; ".join(
                f"K{K} {v['b']}/{v['c']} {families.fmt_p(v['p'])}" for K, v in lv.items()) + " |")
    L += ["", "## Best split (allocation seeds; K₅₀ in nodes, lengths 25-200 generations)", "",
          "| Set | budget × E₀ | targets | seeds | best length | K₅₀ [CI] | dropped / estimable | chosen in replicates |",
          "|---|---|---|---|---|---|---|---|"]
    for g, a in res.get("alloc", {}).items():
        for b, c in a["budgets"].items():
            L.append(f"| {g} | {b} | {a['targets']} | {a['seeds_per_target'][0]}-{a['seeds_per_target'][1]} | "
                     f"{c['best_length'] or '–'} | {f(c['k50'])} [{f(c['lo'])}, {f(c['hi'])}] | {c['dropped']} / "
                     f"{'yes' if c['estimable'] else 'no'} | "
                     + (", ".join(f"{L_}: {v:.2f}" for L_, v in c["chosen_share"].items()) or "–") + " |")
    L += ["", "## Sizes: K_proven and K_construction", "",
          "K_proven: the proven minimum size (enumerated), equal to K_construction. Above a set's enumeration ceiling "
          "the size is a constructed K (K_construction, an upper bound on the minimum size); results there are "
          "labelled constructed K.", "", "| Set | proven levels (K_proven) | constructed K (upper bounds) | constructed targets |",
          "|---|---|---|---|"]
    for g, z in res.get("sizes", {}).items():
        L.append(f"| {g} | {', '.join(map(str, z['proven_levels'])) or '–'} | "
                 f"{(', '.join(map(str, z['constructed_levels'])) + ' (constructed K, upper bound)') if z['constructed_levels'] else '–'}"
                 f" | {z['constructed_targets']} |")
    return "\n".join(L) + "\n"


def _inf(x, d=1):
    return "∞" if x is not None and np.isfinite(x) and x >= runtime.LSTAR_CAP else _f(x, d)


def render_runtime(res, title, source):
    """The run-time model per set (Weibull fit, H4d, L*, held-out comparison) and the H4e/H4f cells (D080 d-h)."""
    f, fp = _f, families.fmt_p
    L = [f"# {title}", "", f"> Index: `docs/README.md`. Generated by `{source}`. **{res['label']}.** Do not edit by hand.",
         "", "Weibull run-time model per set on regime A's runs of the budget levels (target effect integrated out; "
         "generation-0 hits left out). H4d: κ/α against the set's measured slope (bits_sem per doubling), equivalence "
         "within the margin; L* = t₀/(1 − κ) with its target-bootstrap CI (κ ≥ 1: ∞; reported, not tested); held-out "
         "log-likelihood from 5-fold cross-validation over targets (D080). H4f (D080 h, amended): the model's best "
         "length on the cell's measured lengths, argmax of (B/L)(L − t₀)^κ, per replicate; the measured best length "
         "must be among the lengths chosen in the central 95% of replicates.", ""]
    if "note" in res.get("h4_model", {}):
        return "\n".join(L + [f"Run-time model: {res['h4_model']['note']}."]) + "\n"
    L += ["| Set | κ | α | t₀ | σ | κ/α | slope | difference [CI] | H4d p | L* [CI] | held-out ll W / LN |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d, h in res.get("h4_model", {}).items():
        for g, m in h["sets"].items():
            p, t, c = m["point"], h["tests"]["H4d"]["sets"][g], m["comparison"]["heldout_loglik"]
            L.append(f"| {g} | {f(p['kappa'])} | {f(p['alpha'])} | {f(p['t0'], 1)} | {f(p['sigma_target'])} | "
                     f"{f(p['ratio'])} | {f(p['slope'])} | {f(t['diff'])} [{f(t['lo'])}, {f(t['hi'])}] | {fp(t['p'])} | "
                     f"{_inf(p['lstar'])} [{_inf(m['lstar_ci']['lo'])}, {_inf(m['lstar_ci']['hi'])}] | "
                     f"{f(c['weibull'], 1)} / {f(c['lognormal'], 1)} |")
        for g, why in h.get("failed", {}).items():
            L.append(f"| {g} | – | – | – | – | – | – | not estimable ({why}) | – | – | – |")
    L += ["", "## H4e and H4f cells (best split against one run of the whole budget; K₅₀ in nodes)", "",
          "| Set | budget × E₀ | best split K₅₀ (length) | one run K₅₀ | restarts help | κ < 1 | agree | measured best "
          "length | model's length (central 95%) | among | H4f p |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    yn = lambda x: "–" if x is None else ("yes" if x else "no")          # noqa: E731
    for d, h in res.get("h4_model", {}).items():
        fc = {(c["set"], c["budget"]): c for c in h["tests"]["H4f"]["cells"]}
        for c in h["tests"]["H4e"]["cells"]:
            m = h["sets"][c["set"]]["cells"][c["budget"]]
            x = fc.get((c["set"], c["budget"]), {})
            L.append(f"| {c['set']} | {c['budget']} | {f(m['best_split_k50'])} ({m['best_split_length'] or '–'}) | "
                     f"{f(m['one_run_k50'])} | {yn(c['help'])} | {yn(c['kappa'] < 1 if np.isfinite(c['kappa']) else None)}"
                     f" | {yn(c['agree'])} | {m['length'] or '–'} | {x.get('model_length') or '–'} "
                     f"({', '.join(map(str, x.get('central') or [])) or '–'}) | {yn(x.get('among'))} | {fp(x.get('p'))} |")
    return "\n".join(L) + "\n"


def _holm_member(res, fam, d, mid):
    """One member of a Holm family (its unadjusted p, Holm p and status), or {} when the family was not built."""
    for mm in res.get("holm", {}).get(fam, {}).get(d, {}).get("members", []):
        if mm["id"] and mm["id"][0] == mid:
            return mm
    return {}


def _runtime_lines(res, runtime_file):
    L = ["", "## H4d-f: run-time model (D080 f-h); Weibull against lognormal (D080 d)", ""]
    if "note" in res.get("h4_model", {}):
        return L + [f"- {res['h4_model']['note']}."]
    f, fp = _f, families.fmt_p
    for d, h in res.get("h4_model", {}).items():
        t, cmp = h["tests"], h["comparison"]
        e, ff = t["H4e"], t["H4f"]
        among = sum(bool(c.get("among")) for c in ff["cells"])
        mf = _holm_member(res, "H4", d, "H4f")
        L.append(f"- **{d}:** H4d (margin ±{t['H4d']['margin']}): every set within: "
                 f"{'yes' if t['H4d']['holds'] else 'no'}, p {fp(t['H4d']['p'])}"
                 f"{'' if t['H4d']['untestable'] is None else ' (' + t['H4d']['untestable'] + ')'}; H4e: agreement "
                 f"{f(e['share'])} [{f(e['lo'])}, {f(e['hi'])}] of {len(e['cells'])} cells (at least "
                 f"{e['registered_share']}: {'yes' if e['holds_point'] else 'no'}), p {fp(e['p'])}"
                 f"{'' if e['untestable'] is None else ' (' + e['untestable'] + ')'}; H4f (retain): {among} of "
                 f"{len(ff['cells'])} measured best lengths among the model's (central 95%), p {fp(ff['p'])} "
                 f"(unadjusted), Holm p {fp(mf.get('p_holm'))}: {mf.get('status', '–')}"
                 f"{'' if ff['untestable'] is None else ' (' + ff['untestable'] + ')'}; held-out log-likelihood "
                 f"Weibull {f(cmp['heldout_loglik']['weibull'], 1)}, lognormal {f(cmp['heldout_loglik']['lognormal'], 1)}"
                 f": {cmp['preferred'] or '–'} preferred ({cmp['sets_preferring_weibull']} of {cmp['sets']} sets "
                 f"prefer Weibull).")
    L.append(f"- Per set and per cell: `{runtime_file or 'render_runtime'}`.")
    return L


def _h6bc_lines(res, slopes_file):
    fp = families.fmt_p
    L = []
    for d, h in res.get("h6b", {}).items():
        L.append(f"- **H6b, {d}** (D081 b): the GP − null recovery gap changes by {_f(h['theta'], 4)} "
                 f"[{_f(h['lo'], 4)}, {_f(h['hi'], 4)}] per node (set intercepts; {h['targets']} targets, "
                 f"{h['left_out']} without a null probability); p {fp(h['p'])}"
                 f"{'' if h['untestable'] is None else ' (' + h['untestable'] + ')'}. Gaps per level: "
                 f"`{slopes_file or 'render_slopes_gaps'}`.")
    c = res.get("h6c") or {}
    if c:
        gp = "; ".join(f"{d} {_f(g['gap'])} [{_f(g['lo'])}, {_f(g['hi'])}]" for d, g in (c.get("gaps") or {}).items())
        L.append(f"- **H6c** (secondary; D081 c): gap in reach at E₀, mean over the main sets (bits_sem): "
                 f"{gp or '–'}; opposite signs: p {fp(c.get('p'))}"
                 + (f" ({c['orientation']})" if c.get("orientation") else "")
                 + ("" if c.get("untestable") is None else f" ({c['untestable']})") + ".")
    return L


SLOPE_FAMILIES = (("H1", "h1"), ("H8a", "h8"), ("H9a", "h9a"))
K50_FAMILIES = ("H2 rule 6 ranking", "H2c", "H4 per set, H4c", "H4d-f", "H5", "H6", "H6b, H6c", "H8b", "H8c",
                "best split")
ROW118_LABELS = ("H8c",)            # one-predictor fits that keep row 118 (D088): a separated fit is dropped


def separation_summary(res, boot, tally):
    """D088's report: how often a fit was separated, per family. Slope families (H1, H8a, H9a) from their cells: cells
    whose point fit is separated (falling, rising) and separated replicates among cells x boot. K50 families from
    `k50.TALLY` (`counting` labels in `build`): one-predictor fits, point estimates and replicates together."""
    out = {}
    for name, key in SLOPE_FAMILIES:
        out[name] = law.separation_count(res.get(key, []), boot)
    for label in K50_FAMILIES:
        if label in tally:
            t = tally[label]
            out[label] = dict(fits=t["fits"], separated=t["falling"] + t["rising"], falling=t["falling"],
                              rising=t["rising"])
    return out


def separation_lines(res):
    """One short line per family (D088)."""
    s = res.get("separation")
    if not s:
        return []
    L = ["", "## Separated fits (D088)", "",
         "One-predictor slopes keep their sign (±∞, kept in the CI and the p); a K₅₀ separated with successes at the "
         "smaller sizes is the midpoint of the gap, one separated the other way is treated as a rising curve; truly "
         "degenerate fits are dropped. K₅₀ counts include bootstrap replicates.", ""]
    for name, _ in SLOPE_FAMILIES:
        v = s[name]
        L.append(f"- **{name}:** {v['separated_cells']} of {v['cells']} cells separated ({v['falling']} falling, "
                 f"{v['rising']} rising); {v['separated_replicates']} of {v['replicates']} replicates separated, kept "
                 "as ±∞.")
    for label in K50_FAMILIES:
        if label in s:
            v = s[label]
            how = "dropped (row 118 kept)" if label in ROW118_LABELS else "falling at the gap midpoint"
            L.append(f"- **{label}:** {v['separated']} of {v['fits']} one-predictor K₅₀ fits separated ({v['falling']} "
                     f"falling, {v['rising']} rising; {how}).")
    h2 = res.get("h2", {})
    drop = {d: sum(u.get("dropped", 0) for u in h["units"].values()) for d, h in h2.items()}
    L.append("- **Multi-predictor and log-loss fits (row 118 kept):** H2 replicates dropped for a separated fit: "
             + (", ".join(f"{d} {n}" for d, n in drop.items()) or "none")
             + "; H3, the H4 law, H4a and H7 say 'not estimable' where their fit is separated.")
    return L


def render(res, title, source, h1_cells_file=None, members_file=None, termination_file=None, supplement_file=None,
           runtime_file=None, slopes_file=None, h8_file=None):
    f = _f
    L = [f"# {title}", "", f"> Index: `docs/README.md`. Generated by `{source}`. **{res['label']}.** Do not edit by hand.",
         ""] + _completeness_lines(res, termination_file) + ["## H1: logistic slope on size per cell, with Holm per domain", "",
         f"Per-cell slopes, CIs and p-values: `{h1_cells_file or 'render_h1_cells'}`. Stage 1: family = H1 cells per "
         "domain, untestable members counted; p by inverting the 95% target-bootstrap CI below 0; a cell is supported "
         "when its Holm-adjusted p < alpha.", ""]
    for d, h in res.get("h1_holm", {}).items():
        L.append(f"- **{d}:** {h['family']} family of m = {h['m']} cells, alpha {h['alpha']}: {h['supported']} supported, "
                 f"{h['not_supported']} not supported, {h['saturated']} saturated (untestable), {h['not_estimable']} not "
                 f"estimable, {len(h['missing'])} missing; every testable cell supported: "
                 f"{'yes' if h['all_testable_supported'] else 'no'}.")
    L += _holm_lines(res, members_file)
    L += ["", "## H2: excess LOSO log-loss per unit at E₀ (upper 95% bound against Δ)", "",
          "| Domain | Δ | " + " | ".join(UNITS) + " | preferred | deviating sets (> 4Δ) |", "|---|---|" + "---|" * (len(UNITS) + 2)]
    for d, h in res["h2"].items():
        L.append(f"| {d} | {f(h['delta'], 4)} | " + " | ".join(
            f"{f(h['units'][u]['excess'], 4)} (≤ {f(h['units'][u]['upper95'], 4)}){' ✓' if h['units'][u]['invariant'] else ''}"
            for u in UNITS) + f" | {h['preferred'] or 'none (rule 6: bits_sem)'} | "
            f"{', '.join(h['deviating']) or 'none'} |")
    L += ["", "Preferred unit (D079 a, hypotheses.md rules 4 and 6 at 3b8aa5e): among the units invariant after Holm "
          "(✓ marks the unadjusted bound), the lowest mean LOGO log-loss; deviating sets in the preferred unit."]
    for d, h in res["h2"].items():
        L.append(f"- **{d}:** {h.get('preferred_rule', '–')}; invariant (Holm): "
                 f"{', '.join(h.get('invariant_holm', [])) or 'none'}; LOGO log-loss "
                 + ", ".join(f"{u} {f(h['units'][u].get('logo'), 4)}" for u in UNITS) + "."
                 + (" Ranking of sets in bits_sem (K50 at E0, rule 6): " + ", ".join(
                     f"{x['set']} {f(x['k50'])}" for x in h["ranking_bits_sem"]) + "." if "ranking_bits_sem" in h else ""))
    L += ["", "## Sensitivity (exploratory): constructed sizes at their lower bound", ""]
    for d, h in res.get("sensitivity_lower_bound", {}).items():
        L.append(f"- **{d}:** {h['constructed_runs']} of {h['runs']} E₀ runs on constructed targets; preferred unit "
                 f"{h['preferred'] or 'none (rule 6)'}; " + "; ".join(f"{u} {f(h['units'][u]['excess'], 4)} (≤ {f(h['units'][u]['upper95'], 4)})"
                                                   for u in UNITS)
                 + f"; H1 slope below 0 in {h['h1_cells_below_zero']} of {h['h1_cells']} sets.")
    L += ["", "## H2c (controls) and H2s (6-input sets)", ""]
    for c, h in res["h2c"].items():
        L.append(f"- **{c}** − {h['partner']}: {f(h['diff'])} [{f(h['lo'])}, {f(h['hi'])}] nodes; {h['dropped']} replicates "
                 f"dropped{'' if h.get('estimable', True) else ' (over 10%: not estimable)'}; within ±{h['margin']}: "
                 f"{'yes' if h['holds'] else 'no'}.")
    for d, h in res["h2s"].items():
        L.append(f"- **{d}, {h['unit']}{'' if h.get('preferred', h['unit']) else ' (no preferred unit, rule 6)'}:** " + "; ".join(f"{g} excess {f(v['excess'], 4)} (≤ {f(v['upper95'], 4)})"
                                                       for g, v in h["sets"].items())
                 + f"; within Δ = {f(h['delta'], 4)}: {'yes' if h['holds'] else 'no'}.")
    L += ["", "## H2r: random sampling's hit probability at E₀ (bits_uniform decides; D080 b)", ""]
    for d, h in res.get("h2r", {}).items():
        if h["status"] != "tested":
            L.append(f"- **{d}:** {h['status']}; not supported.")
            continue
        L.append(f"- **{d}:** {len(h['sets'])} sets, {h['targets']} targets; bits_uniform excess {f(h['excess'], 4)} "
                 f"(≤ {f(h['upper95'], 4)}) against Δ = {f(h['delta'], 4)}: H2r holds {'yes' if h['holds'] else 'no'}. "
                 "Other units: " + ", ".join(f"{u} {f(v['excess'], 4)} (≤ {f(v['upper95'], 4)})"
                                             for u, v in h["units"].items() if u != h["unit"]) + ".")
    L += ["", "## H4: pooled law, curvature, reach per doubling", "",
          "Registered model (D079 c): set intercepts with a common slope, so γ/β is the slope of K₅₀ on log₂ budget; "
          "the one-intercept fit is reported beside it.", ""]
    for d, h in res["h4"].items():
        ln, lb, cv, rc = h["law_nodes"], h["law_bits_sem"], h["curvature_nodes"], h["rate_bits_sem"]
        L.append(f"- **{d}:** {f(ln['rate'])} nodes per doubling (β {f(ln['beta'], 3)}, γ {f(ln['gamma'], 3)}); "
                 f"{f(lb['rate'])} bits_sem per doubling [{f(rc['lo'])}, {f(rc['hi'])}], upper 95% bound "
                 f"{f(rc['upper95'])}; quadratic term {f(cv['quad'], 3)} [{f(cv['lo'], 3)}, {f(cv['hi'], 3)}]; H4a "
                 f"(retain): p {families.fmt_p(cv.get('p_wald'))} (unadjusted), Holm p "
                 f"{families.fmt_p(_holm_member(res, 'H4', d, 'H4a').get('p_holm'))}: "
                 f"{_holm_member(res, 'H4', d, 'H4a').get('status', '–')}.")
        if "rate_bits_sem_one_intercept" in h:
            o, oc = h["rate_bits_sem_one_intercept"], h["curvature_nodes_one_intercept"]
            L.append(f"  - one intercept (reported beside it): {f(h['law_nodes_one_intercept']['rate'])} nodes and "
                     f"{f(o['rate'])} bits_sem [{f(o['lo'])}, {f(o['hi'])}] per doubling; quadratic term "
                     f"{f(oc['quad'], 3)} [{f(oc['lo'], 3)}, {f(oc['hi'], 3)}].")
    for d, h in res["h4"].items():
        c = h.get("h4c")
        if c:
            L.append(f"- **H4c, {d}** (margin ±{c['margin']} bits_sem per doubling, D081 a): "
                     f"{sum(pr['holds'] for pr in c['pairs'])} of {len(c['pairs'])} pairs of sets within; every pair: "
                     f"{'yes' if c['holds'] else 'no'}; p {families.fmt_p(c['p'])}"
                     f"{'' if c['untestable'] is None else ' (' + c['untestable'] + ')'}.")
    L.append(f"- Per-set slopes and H4c's pairs: `{slopes_file or 'render_slopes_gaps'}`.")
    L += _runtime_lines(res, runtime_file)
    L += ["", "## H5: paired K₅₀ differences at E₀ (nodes)", "", "| Set | contrast | difference [CI] | targets |",
          "|---|---|---|---|"]
    for g, cs in res["h5"].items():
        for k, c in cs.items():
            L.append(f"| {g} | {k} | {f(c['diff'])} [{f(c['lo'])}, {f(c['hi'])}] | {c['targets']} |")
    L += ["", "## H6: GP reach against the primary null at E₀ (bits_sem)", "",
          "| Set | domain | GP K₅₀ [CI] | null | gap [CI] | GP bits per doubling [CI] (null: 1) |", "|---|---|---|---|---|---|"]
    for g, h in res["h6"].items():
        bp = h["bits_per_doubling"]
        L.append(f"| {g} | {h['domain']} | {f(h['gp_reach_bits_sem'])} [{f(h['lo'])}, {f(h['hi'])}] | "
                 f"{f(h['null_bits_sem'])} | {f(h['gap'])} [{f(h['gap_lo'])}, {f(h['gap_hi'])}] | "
                 f"{f(bp['rate'])} [{f(bp['lo'])}, {f(bp['hi'])}] |")
    L += [""] + _h6bc_lines(res, slopes_file)
    L += ["", "## H7: leave-one-set-out selection", ""]
    for d, h in res["h7"].items():
        if h.get("error"):
            L.append(f"- **{d}:** H7 selection failed ({h['error']}); no model to freeze; "
                     f"{_comp1_text(h.get('comparator'))}; {_comp2_text(h.get('comparator2'))}.")
            continue
        L.append(f"- **{d}:** unit {h['selected_unit']} ({h.get('unit_source', '?')}), budget term "
                 f"{h['budget_term']}, set feature {h.get('feature') or 'none'} (of {', '.join(h.get('candidates', [])) or 'none'}), LOSO log-loss "
                 f"{f(h['loso_logloss'], 4)}; {_comp1_text(h.get('comparator'))}; {_comp2_text(h.get('comparator2'))}.")
    L += _h3_lines(res, supplement_file)
    L += _sensitivity_lines(res)
    hb, hc = res.get("h8b", {}), res.get("h8c", {})
    L += ["", "## H8 and H9a", "", f"H8a cells (H1 under each robustness configuration): {len(res['h8'])}; H8b "
          f"(sign of GP − null unchanged, D081 d): {sum(v['p'] is not None and v['p'] < 0.05 for v in hb.values())} of "
          f"{len(hb)} members with both CIs on the base's side, unadjusted; H8c (interaction within ±Δ, D081 e): "
          f"{sum(bool(v.get('holds')) for v in hc.values())} of {len(hc)} within, unadjusted; H9a cells (Operon at 16 x "
          f"E0): {len(res.get('h9a', []))}. Holm decisions: `{members_file or 'render_holm_members'}`; H8's parts: "
          f"`{h8_file or 'render_h8'}`."]
    L += separation_lines(res)
    L += ["", "## Not run yet", ""] + [f"- {x}" for x in res["not_run"] or ["none"]]
    return "\n".join(L) + "\n"


def main(argv=None):
    """Writes the calibration analysis; refuses (IncompleteError) while registered tasks lack rows, unless
    --allow-incomplete is given (the report is then labelled INCOMPLETE, not confirmatory)."""
    import sys
    from reach.paths import ROOT
    argv = sys.argv[1:] if argv is None else argv
    res = build(PHASE_MAPS["calibration"], allow_incomplete="--allow-incomplete" in argv, runtime_model=True)
    src, st = "reach.analysis.s2.confirm", "docs/status/calibration"
    files = {f"{st}_h1_cells.md": render_h1_cells(res, "Calibration analysis (confirmatory): H1 per cell", src),
             f"{st}_holm_members.md": render_holm_members(res, "Calibration analysis (confirmatory): Holm families", src),
             f"{st}_termination.md": render_termination(res, "Calibration analysis: termination and exclusions", src),
             f"{st}_supplement.md": render_supplement(res, "Calibration analysis (confirmatory): supplement", src),
             f"{st}_runtime.md": render_runtime(res, "Calibration analysis (confirmatory): run-time model", src),
             f"{st}_slopes_gaps.md": render_slopes_gaps(res, "Calibration analysis (confirmatory): H4c and H6b", src),
             f"{st}_h8.md": render_h8(res, "Calibration analysis (confirmatory): H8's parts", src)}
    files[f"{st}_holm_members.md"] = render_holm_members(res, "Calibration analysis (confirmatory): Holm families", src,
                                                         f"{st}_h8.md")
    files[f"{st}_report.md"] = render(res, "Calibration analysis (confirmatory)", src, f"{st}_h1_cells.md",
                                      f"{st}_holm_members.md", f"{st}_termination.md", f"{st}_supplement.md",
                                      f"{st}_runtime.md", f"{st}_slopes_gaps.md", f"{st}_h8.md")
    (ROOT / "data/analysis").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/analysis/calibration.json").write_text(json.dumps(res, indent=1, default=float))
    for k, v in files.items():
        (ROOT / k).write_text(v)
    print("written data/analysis/calibration.json and " + ", ".join(files) + " (list them in docs/README.md)")


if __name__ == "__main__":
    main()

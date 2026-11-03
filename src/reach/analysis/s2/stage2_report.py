"""Stage 2 report driver (D095; REQ-ANA-81 to 86): the held-out prediction tests (H7, H9b), the Cartesian GP tests
(H10a, H10b), the comparators and benchmarks, and the two registered sensitivity analyses, from the six Stage 2 phases.

    .venv/bin/python -m reach.analysis.s2.stage2_report [--allow-incomplete] [--force REASON]
        -> data/analysis/stage2.json, docs/status/stage2_report.md
    .venv/bin/python -m reach.analysis.s2.stage2_report transfer
        -> data/registration/cgp_transfer.json with its .sha256, once (the H10b transfer model; an OSF addendum
           records its checksum before s2_ho_cgp starts)

The Stage 2 form says the report "is produced by a driver that calls exactly the entry points named in this form".
This is that driver (guardrail row 139). It computes nothing of its own: every test is one call of the registered
entry point with the registered constants (`predictor.score_model("H7")`, `gp2010.score_beside_registered`,
`predictor.describe_model`, `h9b.h9b`, `h9b.describe_bench`, `cgp_confirm.h10a`, `h10a_sixteenth`, `h10b`,
`fit_transfer`) on records tidied by `load.tidy` at the registered budgets, with units from exactly the target files
each phase reads (`confirm.STAGE2_TARGET_FILES`, one `_load_units` call per phase, so a run killed without a result
counts as a failure) and each held-out set's frozen feature value (the prediction file's, checked against
properties.json). Before anything is read: every frozen input must match its .sha256; every target file its .sha256
and the hash the prediction file recorded; each phase's rows must be exactly the registered task keys
(stage2_tasks.json, `termination`); records of a set or budget not registered for a test are refused. A phase without
rows is missing and its tests pending, never fabricated. Sensitivity analyses (form, Other planned analysis 4), on H7,
H9b and H10b, labelled: (a) CPU-limit stops excluded (`tidy(censored_as_failure=False)`); (b) constructed held-out
targets at their lower-bound size (`confirm.lower_bound_sizes`, with units built as the target generator builds them
and checked to reproduce every target's recorded units). The JSON holds every result, the SHA-256 of every input, the
code hash and the commit; a result built from other inputs is never replaced silently (--force REASON, recorded).
"""
import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from reach.analysis.s2 import cgp_confirm, gp2010, gpf2013, h9b, predictor
from reach.analysis.s2 import confirm as C
from reach.analysis.s2 import stage2_freeze as SF
from reach.analysis.s2 import termination as T
from reach.analysis.s2.load import read_phase, tidy

FROZEN = dict(h7=SF.PATHS["h7"], operon=h9b.TRANSFER_FILE, gpf2013=gpf2013.FROZEN_FILE, gp2010=gp2010.FROZEN_FILE,
              cgp_settings="data/registration/cgp_settings.json", levels=SF.PATHS["levels"],
              predictions=SF.PATHS["predictions"])
TRANSFER = cgp_confirm.TRANSFER_FILE
TASKS = "data/registration/stage2_tasks.json"
PROPERTIES = "data/analysis/properties.json"
OUT_JSON, OUT_MD = "data/analysis/stage2.json", "docs/status/stage2_report.md"
PHASES = tuple(C.STAGE2_TARGET_FILES)                  # the six Stage 2 phases
TESTS = {"H7": ("s2_ho_a0", "s2_ho_a16"), "benchmarks": ("s2_ho_bench",), "H9b": ("s2_ho_operon",),
         "H10a": ("s2_cal_cgp",), "H10b": ("s2_cal_cgp", "s2_ho_cgp")}
SCORED_TESTS = ("H7", "H9b", "H10b")                   # the tests both sensitivity analyses recompute
LABEL = "STAGE 2 (held-out prediction H7 and H9b; Cartesian GP H10a and H10b)"
BOOT = C.REGISTERED["boot"]                            # the registered bootstrap size (Stage 1), as H10a and H10b use it


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- inputs
def inputs(root):
    """Every frozen and registration input, read only if it matches its .sha256 (`stage2_freeze._checked_targets`),
    with the SHA-256 of each under `sha`; the target files the prediction file was made for must still have the hash
    it recorded, and its model must be h7_model.json (REQ-ANA-82). cgp_transfer.json is read when present."""
    root = Path(root)
    obj, sha = {}, {}
    for k, rel in FROZEN.items():
        obj[k], sha[rel] = SF._checked_targets(root, rel)
    if (root / TRANSFER).exists():
        obj["transfer"], sha[TRANSFER] = SF._checked_targets(root, TRANSFER)
        cgp_confirm.check_transfer(obj["transfer"])
    if not (root / TASKS).exists():
        raise FileNotFoundError(f"{TASKS} is missing: the registered task lists are needed to check each phase's rows")
    sha[TASKS], sha[PROPERTIES] = _sha(root / TASKS), _sha(root / PROPERTIES)
    pred = obj["predictions"]
    if pred.get("model_sha256") != sha[FROZEN["h7"]]:
        raise ValueError("the prediction file was made with another model file than h7_model.json")
    for rel, h in sorted(pred["sources"].items()):
        _, got = SF._checked_targets(root, rel)
        if got != h:
            raise ValueError(f"{rel}: SHA-256 {got[:12]}... differs from the one the prediction file recorded "
                             f"({h[:12]}...): the predictions were made for other targets")
        sha[rel] = got
    obj["h7"] = {d: dict(m, sha256=sha[FROZEN["h7"]]) for d, m in obj["h7"].items()}    # as stage2_freeze.load_h7
    return dict(obj, sha=sha)


def phases(root, allow_incomplete=False):
    """The six phases' rows (the queue's last row per key) and their termination summary against the registered task
    keys. Status per phase: complete (exactly the registered keys), missing (no rows) or incomplete. A row whose key
    is not in the phase's own task list is never analysed (refused when incomplete phases are reported)."""
    root = Path(root)
    term = T.termination(root, list(PHASES), registered=root / TASKS)
    rows, status = {}, {}
    for ph in PHASES:
        rows[ph] = read_phase(ph, root)
        v = term["phases"][ph]
        status[ph] = "complete" if v["complete"] else ("missing" if v["rows"] == 0 else "incomplete")
        if status[ph] == "incomplete" and allow_incomplete and v["extra"]:
            raise ValueError(f"{ph}: {v['extra']} row(s) whose key is not in the phase's task list")
    return rows, status, term


def _pending(test, status, allow_incomplete):
    """None when the test can be computed, else why it is pending."""
    for ph in TESTS[test]:
        if status[ph] == "missing":
            return f"pending: {ph} has no rows (not run)"
        if status[ph] == "incomplete" and not allow_incomplete:
            return f"pending: {ph} is incomplete (its rows are not the registered task list; --allow-incomplete " \
                   "reports it labelled not confirmatory)"
    return None


# ---------------------------------------------------------------- records
def _records(root, phase, rows, budgets, caf):
    """Tidy records of one phase at `budgets`, censored runs as failures (caf) or left out, units from exactly the
    target files the phase reads in one `_load_units` call (killed rows keep their units; guardrail row 136)."""
    sets = sorted({r["set"] for r in rows if r.get("set")})
    C._load_units(root, sets, C.STAGE2_TARGET_FILES[phase])
    for g in sets:                                                   # the files read, each against its .sha256
        for name in C.STAGE2_TARGET_FILES[phase]:
            rel = f"data/targets_s2/{g}/{name}.json"
            if (Path(root) / rel).exists():
                _USED[rel] = SF._checked_targets(root, rel)[1]
    return C._with_units(tidy(rows, budgets, caf))


_USED = {}                                     # target files read for the records -> SHA-256 (into the JSON)


def _features(root, inp, recs):
    """Each held-out set's frozen feature value from the prediction file (the value its predictions and levels were
    made with), required to equal properties.json's (`confirm.set_features`), on every record. A set the prediction
    file does not hold is not a held-out set of Stage 2: refused."""
    feats = inp["predictions"]["features"]
    sets = sorted({r["set"] for r in recs})
    bad = [g for g in sets if g not in feats]
    if bad:
        raise ValueError(f"set(s) {bad} are not in the registered prediction file: not held-out sets of Stage 2")
    props = C.set_features(root, sets)
    for g in sets:
        for f, z in feats[g].items():
            if props.get(g, {}).get(f) != z:
                raise ValueError(f"{g}: feature {f} is {z} in the prediction file but {props.get(g, {}).get(f)!r} "
                                 f"in {PROPERTIES}")
    for r in recs:
        r["features"] = dict(feats[r["set"]])
    return recs


def _check_sets(label, items, allowed):
    bad = sorted({r["set"] for r in items if r.get("set")} - set(allowed))
    if bad:
        raise ValueError(f"{label}: records from set(s) {bad}, not registered for it")


def _check_regime(phase, rows):
    bad = sorted({(r.get("algorithm"), r.get("config")) for r in rows} - {("A", "base")}, key=str)
    if bad:
        raise ValueError(f"{phase}: rows of {bad}; the registered tasks are regime A, configuration base")


def _check_operon(rows):
    bad = sorted({r.get("budget") for r in rows} - set(h9b.BUDGETS), key=str)
    if bad:
        raise ValueError(f"s2_ho_operon: rows at budget(s) {bad}; Operon runs at {list(h9b.BUDGETS)} x E0 only")


def _summary(recs):
    return dict(n=len(recs), sets=sorted({r["set"] for r in recs}), budgets=sorted({r["budget"] for r in recs}),
                censored=sum(1 for r in recs if r.get("censored")), censored_runs=C.censored_runs(recs))


# ---------------------------------------------------------------- lower-bound sensitivity (form, 4b)
def _same(a, b):
    """A recorded unit value reproduced: floats to `math.isclose`'s default relative tolerance (an extrapolated level's
    least-squares line differs in its last bits between platforms, guardrail row 141), everything else exactly."""
    if isinstance(a, float) and isinstance(b, (int, float)) and not isinstance(b, bool):
        return math.isclose(a, b)
    return a == b


def lower_bound_inputs(root, sets):
    """`confirm.lower_bound_sizes`'s inputs for held-out sets: the constructed targets (heldout.json's flag), each set's
    ceiling and size parities from its enumeration counts (as `confirm._sensitivity_inputs`), and a units function
    built as the target generator builds it (`SizeUnits` on counts.json's levels, with the D050 noise rows the set's
    bench_log.json records for real sets), required to reproduce the recorded units of every held-out and benchmark
    target. Also the lower-bound size per constructed level and set."""
    from reach import grammar as G
    from reach.analysis.s2.phase1 import enum_dir
    from reach.paths import ROOT
    from reach.units import SizeUnits
    root = Path(root)
    cons, ceil, par, ufn, changes = set(), {}, {}, {}, {}
    for g in sets:
        lv = json.loads((root / enum_dir(g).relative_to(ROOT) / "counts.json").read_text())["levels"]
        ks = [st["K"] for st in lv if st["U"] > 0 and st["K"] >= 3]
        ceil[g], par[g] = max(ks), {k % 2 for k in ks}
        log = root / f"data/targets_s2/{g}/bench_log.json"
        noise = json.loads(log.read_text()).get("noise_rows") if log.exists() else None
        ufn[g] = SizeUnits(G.get(g), lv, noise_rows=noise).all
        for name in ("heldout", "bench"):
            p = root / f"data/targets_s2/{g}/{name}.json"
            for t in (json.loads(p.read_text()) if p.exists() else []):
                u = ufn[g](t["K_construction"])
                if any(not _same(u.get(k), v) for k, v in t["units"].items()):
                    raise ValueError(f"{g} {t['target_id']}: the units function does not reproduce the target "
                                     f"file's units ({u} against {t['units']}); the lower-bound sensitivity needs "
                                     "the generator's units")
                if t.get("constructed"):
                    cons.add((g, t["target_id"]))
                    changes.setdefault(g, {})[str(t["K_construction"])] = None
    for g in changes:
        for K in changes[g]:
            probe = [dict(set=g, target_id="probe", K=int(K), units={})]
            changes[g][K] = C.lower_bound_sizes(probe, {(g, "probe")}, ceil, par, ufn)[0]["K"]
    return (cons, ceil, par, ufn), changes


# ---------------------------------------------------------------- one pass of the registered entry points
def _scored(root, inp, rows, status, allow_incomplete, caf=True, lower=None, primary=False):
    """H7, H9b and H10b (and, when primary, the comparators, benchmarks, H10a and H10a at 1/16 x E0) for one variant:
    censored runs as failures (caf) or left out; sizes as registered or, with `lower`, constructed held-out targets
    at their lower bound. Returns (results, pending reasons, record summaries)."""
    from reach.runner.phases import CAL_SETS, heldout_sets
    ho, h7 = heldout_sets(), inp["h7"]
    out, pend, recs_of = {}, {}, {}
    C._NO_UNITS.clear()

    def usable(test):
        why = _pending(test, status, allow_incomplete)
        if why:
            pend[test] = why
        return why is None

    def lb(recs, test):
        """The variant's records of a scored test, summarised; with `lower`, constructed targets at the lower bound."""
        if lower is not None:
            recs = C.lower_bound_sizes(recs, *lower)
        recs_of[test] = _summary(recs)
        if lower is not None:
            recs_of[test]["constructed"] = sum(1 for r in recs if (r["set"], r["target_id"]) in lower[0])
        return recs

    if usable("H7"):
        recs = []
        for ph in TESTS["H7"]:
            _check_regime(ph, rows[ph])
            recs += _records(root, ph, rows[ph], predictor.SCORED["H7"]["budgets"], caf)
        vec, left = gp2010.vectors_for(recs, C._TARGETS)            # both phases read heldout.json (the last call)
        _check_sets("H7", recs, ho)
        recs = lb(_features(root, inp, recs), "H7")
        out["H7"] = {}
        if primary:
            out["comparators"] = {}
        for d, rr in sorted(cgp_confirm.by_domain(recs).items()):
            out["H7"][d] = predictor.score_model("H7", h7[d], rr, h7[d]["baseline"])
            if primary:
                rv = [r for r in rr if (r["set"], r["target_id"]) in vec]
                res, err = C._guarded(gp2010.score_beside_registered, h7[d], inp["gpf2013"][d], inp["gp2010"][d],
                                      rv, vec, h7[d]["baseline"])
                sets_d = {r["set"] for r in rr}
                out["comparators"][d] = dict(res or {}, error=err, n=len(rv),
                                             left_out=[k for k in left if k[0] in sets_d])
    if primary and usable("benchmarks"):
        _check_regime("s2_ho_bench", rows["s2_ho_bench"])
        recs = _records(root, "s2_ho_bench", rows["s2_ho_bench"], predictor.SCORED["H7"]["budgets"], caf)
        _check_sets("benchmarks", recs, ho)
        if not all(h9b._is_bench(r) for r in recs):
            raise ValueError("s2_ho_bench: a record that is not a benchmark cell")
        recs = _features(root, inp, recs)
        recs_of["benchmarks"] = _summary(recs)
        out["benchmarks"] = {}
        for d, rr in sorted(cgp_confirm.by_domain(recs).items()):
            res, err = C._guarded(predictor.describe_model, h7[d], rr, h7[d]["baseline"])
            out["benchmarks"][d] = res if err is None else dict(error=err)
    if usable("H9b"):
        _check_operon(rows["s2_ho_operon"])
        recs = _features(root, inp, _records(root, "s2_ho_operon", rows["s2_ho_operon"], h9b.BUDGETS, caf))
        held = lb([r for r in recs if not h9b._is_bench(r)], "H9b")
        out["H9b"] = h9b.h9b(inp["operon"], held)
        if primary:
            bench = [r for r in recs if h9b._is_bench(r)]
            recs_of["H9b_benchmarks"] = _summary(bench)
            res, err = C._guarded(h9b.describe_bench, inp["operon"], bench)
            out["H9b_benchmarks"] = res if err is None else dict(error=err)
    if primary and usable("H10a"):
        _check_sets("H10a", rows["s2_cal_cgp"], CAL_SETS)
        recs_of["H10a"] = _summary(_records(root, "s2_cal_cgp", rows["s2_cal_cgp"], cgp_confirm.H10A_BUDGETS, caf))
        out["H10a"] = cgp_confirm.h10a(rows["s2_cal_cgp"], boot=BOOT)
        out["H10a_sixteenth"] = cgp_confirm.h10a_sixteenth(rows["s2_cal_cgp"], boot=BOOT)
    if usable("H10b"):
        if "transfer" not in inp:
            pend["H10b"] = (f"pending: {TRANSFER} is not frozen (python -m reach.analysis.s2.stage2_report transfer, "
                            "once s2_cal_cgp is complete and before s2_ho_cgp)")
        else:
            cal = _records(root, "s2_cal_cgp", rows["s2_cal_cgp"], cgp_confirm.FIT_BUDGETS, caf)
            recs = _features(root, inp, _records(root, "s2_ho_cgp", rows["s2_ho_cgp"], cgp_confirm.FIT_BUDGETS, caf))
            _check_sets("H10b", recs, ho)
            recs = lb(recs, "H10b")
            spec, cal_by, out["H10b"] = predictor.SCORED["H10b"], cgp_confirm.by_domain(cal), {}
            for d, rr in sorted(cgp_confirm.by_domain(recs).items()):
                n = len({r["set"] for r in rr})
                if n > spec["n_sets"]:
                    raise ValueError(f"H10b: {n} sets in the {d} records; {spec['n_sets']} registered per domain")
                out["H10b"][d] = dict(cgp_confirm.h10b(inp["transfer"][d], cal_by.get(d, []), rr, boot=BOOT),
                                      test="H10b", n_sets_registered=spec["n_sets"], sets_missing=spec["n_sets"] - n)
    return out, pend, recs_of


def _changed(a, b):
    """Decisions (passes) that differ between the primary results `a` and a sensitivity variant `b`."""
    out = []
    for t in SCORED_TESTS:
        for k, v in (a.get(t) or {}).items():
            w = (b.get(t) or {}).get(k)
            if w is not None and bool(v.get("passes")) != bool(w.get("passes")):
                out.append(f"{t} {k}: {v.get('passes')} -> {w.get('passes')}")
    return out


def build(root=None, allow_incomplete=False):
    """The whole Stage 2 result: checked inputs, phases, every test the registered code computes, both sensitivity
    analyses, exclusions, the inputs' SHA-256, code hash and commit."""
    from reach.paths import ROOT
    from reach.runner.queue import code_hash, git_version
    root = Path(root or ROOT)
    _USED.clear()
    inp = inputs(root)
    rows, status, term = phases(root, allow_incomplete)
    prim, pend, recs_of = _scored(root, inp, rows, status, allow_incomplete, primary=True)
    no_units = [dict(set=r["set"], target_id=r["target_id"], seed=r.get("seed"), algorithm=r.get("algorithm"),
                     budget=r["budget"]) for r in C._NO_UNITS]
    cens = {t: recs_of[t]["censored_runs"] for t in SCORED_TESTS if t in recs_of}
    if sum(cens.values()):
        a, _, a_recs = _scored(root, inp, rows, status, allow_incomplete, caf=False)
        sens_a = dict(recomputed=True, censored_runs=cens, records=a_recs, changed=_changed(prim, a),
                      note="CPU-limit stops excluded (form: Data inclusion; Other planned analysis 4a); primary: "
                           "counted as failures", **{t: a[t] for t in SCORED_TESTS if t in a})
    else:
        sens_a = dict(recomputed=False, censored_runs=cens, changed=[],
                      note="identical to the primary analysis: no run was stopped by its CPU limit")
    ho_sets = sorted({g for t in SCORED_TESTS if t in recs_of for g in recs_of[t]["sets"]})
    sens_b = dict(recomputed=False, changed=[], note="no held-out records yet")
    if ho_sets:
        lower, changes = lower_bound_inputs(root, ho_sets)
        b, _, b_recs = _scored(root, inp, rows, status, allow_incomplete, lower=lower)
        sens_b = dict(recomputed=True, changes=changes, records=b_recs, changed=_changed(prim, b),
                      constructed_records={t: b_recs[t]["constructed"] for t in SCORED_TESTS if t in b},
                      note="constructed held-out targets at their lower-bound size (the smallest size above the "
                           "ceiling with the set's parity; registered: the constructed size, an upper bound); "
                           "exploratory, labelled (form: Other planned analysis 4b)",
                      **{t: b[t] for t in SCORED_TESTS if t in b})
    computed = {t: ("computed on incomplete phases: NOT CONFIRMATORY" if any(status[ph] != "complete"
                                                                           for ph in TESTS[t]) else "computed")
                for t in TESTS if t not in pend}
    label = LABEL + ("" if all(s == "complete" for s in status.values()) else
                     " — INCOMPLETE: not every phase has its registered rows; pending tests are not computed")
    return dict(label=label, built=time.strftime("%Y-%m-%d %H:%M:%S"),
                code=dict(commit=git_version(), code_hash=code_hash()),
                registered=dict(h7_rule=dict(predictor.H7_RULE, slope=list(predictor.H7_RULE["slope"])),
                                scored={k: dict(v, budgets=list(v["budgets"])) for k, v in predictor.SCORED.items()},
                                boot=BOOT, alpha=C.REGISTERED["alpha"]),
                inputs=dict(registered=dict(sorted(dict(inp["sha"], **_USED).items())),
                            phases={ph: _sha(root / "data/runs" / ph / "results.jsonl")
                                    if (root / "data/runs" / ph / "results.jsonl").exists() else None for ph in PHASES}),
                phases={ph: dict(term["phases"][ph], status=status[ph]) for ph in PHASES},
                confirmatory=all(s == "complete" for s in status.values()) and not allow_incomplete,
                allow_incomplete=allow_incomplete, status=dict(sorted(dict(computed, **pend).items())),
                tests=prim, records=recs_of, sensitivity=dict(cpu_excluded=sens_a, lower_bound=sens_b),
                exclusions=T.exclusions(root, list(PHASES)), no_units=no_units)


# ---------------------------------------------------------------- the H10b transfer model (once)
def transfer(root=None):
    """Fit and freeze the H10b transfer model (REQ-ANA-84): `cgp_confirm.fit_transfer` per domain on every record of
    a complete s2_cal_cgp at the fit budgets, with the frozen H7 model, before any held-out CGP row exists;
    `cgp_confirm.freeze_transfer` writes data/registration/cgp_transfer.json with its .sha256 and never replaces other
    content. Returns the file's SHA-256."""
    from reach.paths import ROOT
    from reach.runner.phases import CAL_SETS
    root = Path(root or ROOT)
    inp = inputs(root)
    rows, status, _ = phases(root)
    if status["s2_cal_cgp"] != "complete":
        raise RuntimeError(f"s2_cal_cgp is {status['s2_cal_cgp']}: the transfer model is fitted on all its registered "
                           "records (form, H10b); nothing frozen")
    if rows["s2_ho_cgp"] and "transfer" not in inp:
        raise RuntimeError("s2_ho_cgp has rows but no transfer model is frozen: the model must be frozen before any "
                           "held-out CGP run (form, H10b); nothing frozen")
    _check_sets("H10b transfer", rows["s2_cal_cgp"], CAL_SETS)
    recs = _records(root, "s2_cal_cgp", rows["s2_cal_cgp"], cgp_confirm.FIT_BUDGETS, True)
    models = {}
    for d, rr in sorted(cgp_confirm.by_domain(recs).items()):
        cgp_confirm.attach_features(root, rr, inp["h7"][d])
        models[d] = cgp_confirm.fit_transfer(inp["h7"][d], rr)
    return cgp_confirm.freeze_transfer(models, root / TRANSFER)["sha256"]


# ---------------------------------------------------------------- output
def _differences(old, new):
    """What changed since `old` was built: a registered input, or the records of a phase `old` had complete."""
    out = []
    o_reg, n_reg = old.get("inputs", {}).get("registered", {}), new["inputs"]["registered"]
    out += [f"{rel}: {o_reg.get(rel)} -> {n_reg.get(rel)}" for rel in sorted(set(o_reg) | set(n_reg))
            if o_reg.get(rel) != n_reg.get(rel)]
    for ph in PHASES:
        if old.get("phases", {}).get(ph, {}).get("status") == "complete" and \
                old.get("inputs", {}).get("phases", {}).get(ph) != new["inputs"]["phases"].get(ph):
            out.append(f"{ph}: records of a complete phase changed")
    return out


def write(root, res, force=None):
    """stage2.json and the report. An existing result built from other registered inputs, or from other records of a
    phase it had complete, is replaced only with --force REASON; the reason and what changed are recorded (REQ-ANA-86)."""
    p = Path(root) / OUT_JSON
    if p.exists():
        try:
            old = json.loads(p.read_text())
        except json.JSONDecodeError:
            old = {}
        diff = _differences(old, res)
        if diff and not force:
            raise RuntimeError(f"{OUT_JSON} exists and was built from other inputs: " + "; ".join(diff)
                               + ". Rerun with --force REASON to replace it (the reason is recorded).")
        if diff:
            res["force"] = dict(reason=force, replaced=diff, previous_built=old.get("built"))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1, default=float))
    md = Path(root) / OUT_MD
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(render(res))


def _f(x, d=2):
    return C._f(x, d)


def _crit(r):
    """One line of the H7 criteria as the registered code computed them."""
    ci = r.get("intercept_ci") or (None, None)
    return (f"AUC {_f(r.get('auc'))}; slope {_f(r.get('slope'))}; intercept {_f(r.get('intercept'))} "
            f"[{_f(ci[0])}, {_f(ci[1])}]; Brier {_f(r.get('brier'), 4)} vs baseline {_f(r.get('brier_baseline'), 4)}; "
            f"sets within eps {r.get('sets_ok')} of {r.get('n_sets_registered', len(r.get('per_set_error', {})))}"
            + (f" ({r['sets_missing']} without records)" if r.get("sets_missing") else ""))


def _passes(r):
    return "**supported**" if r.get("passes") else "not supported"


def _scored_lines(L, res, test, by, title):
    r = res.get(test)
    if r is None:
        L.append(f"- {title}: {res.get('_status', {}).get(test, 'not computed')}")
        return
    for k, v in sorted(r.items()):
        L.append(f"- {title}, {by} {k}: {_passes(v)}; {_crit(v)}")


def render(res):
    s, t = res["status"], res["tests"]
    L = [f"# Stage 2 report: held-out prediction and Cartesian GP", "",
         f"> Index: `docs/README.md`. Generated by `reach.analysis.s2.stage2_report`. **{res['label']}.** "
         f"Do not edit by hand. Built {res['built']} at commit {res['code']['commit']} (code hash "
         f"{res['code']['code_hash']}). Every number comes from `data/analysis/stage2.json`; the registered criteria "
         f"and constants are those of `predictor.H7_RULE`, `predictor.SCORED` and `confirm.REGISTERED` (B = "
         f"{res['registered']['boot']}).", ""]
    if res.get("force"):
        L += [f"**Replaced an earlier result with --force:** {res['force']['reason']} "
              f"(changed: {'; '.join(res['force']['replaced'])}).", ""]
    L += ["## Phases (final row per registered task)", "",
          "| Phase | status | tasks | rows | converged | budget_exhausted | timeout (CPU limit) | killed | error | other |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for ph, v in res["phases"].items():
        st = v["states"]
        L.append(f"| {ph} | {v['status']} | {v['tasks']} | {v['rows']} | {st['converged']} | {st['budget_exhausted']} "
                 f"| {st['timeout']} | {st['killed']} | {st['error']} | {st['other']} |")
    L += ["", f"Excluded rows (errors; listed in the JSON): {len(res['exclusions'])}. Records without size units: "
              f"{len(res['no_units'])}. A run stopped by its CPU limit counts as a failure (primary).", ""]
    L += ["## Tests (status per test; a pending test has not been computed)", ""]
    L += [f"- {k}: {v}" for k, v in s.items()]
    L += ["", "## H7: the frozen model on the held-out sets (per domain; all four criteria must hold)", ""]
    res_s = dict(t, _status=s)
    _scored_lines(L, res_s, "H7", "domain", "H7")
    if "comparators" in t:
        for d, c in sorted(t["comparators"].items()):
            if c.get("error"):
                L.append(f"- Comparators, domain {d}: failed ({c['error']})")
                continue
            for name, lab in (("comparator1", "comparator 1 (Graff, Poli and Flores 2013)"),
                              ("comparator2", "comparator 2 (Graff and Poli 2010)")):
                v = c[name]
                rse = ", ".join(f"{b}: {_f(x)}" for b, x in sorted(v.get("rse", {}).items(), key=lambda kv: int(kv[0])))
                L.append(f"- {lab}, domain {d} (descriptive): {_crit(v)}; RSE per budget {rse}; clipped "
                         f"{_f(v.get('clipped_share'), 3)}; {len(c.get('left_out', []))} target(s) without a vector")
    L += ["", "## Benchmarks (descriptive, pooled per domain; no pass or fail)", ""]
    for key, lab in (("benchmarks", "Tree GP (s2_ho_bench)"), ("H9b_benchmarks", "Operon (s2_ho_operon)")):
        if key not in t:
            L.append(f"- {lab}: {s.get('benchmarks' if key == 'benchmarks' else 'H9b', 'not computed')}")
            continue
        if isinstance(t[key].get("error"), str):
            L.append(f"- {lab}: failed ({t[key]['error']})")
            continue
        for d, v in sorted(t[key].items()):
            L.append(f"- {lab}, {d}: " + (f"failed ({v['error']})" if v.get("error") else
                                          f"{_crit(v)}; {len(v.get('per_cell_error', {}))} cells; largest cell error "
                                          f"{_f(max(v['per_cell_error'].values()) if v.get('per_cell_error') else None)}"))
    L += ["", "## H9b: Operon (per configuration, held-out targets of R-ARITH and R-ELEM6; both sets must pass)", ""]
    _scored_lines(L, res_s, "H9b", "configuration", "H9b")
    L += ["", "## H10a: CGP manipulation check (Holm per domain over the declared cells)", ""]
    if "H10a" in t:
        for d, h in sorted(t["H10a"]["holm"].items()):
            L.append(f"- H10a, {d}: m = {h['m']}; supported {h['supported']}, not supported {h['not_supported']}, "
                     f"saturated {h['saturated']}, not estimable {h['not_estimable']}, missing {len(h['missing'])}; "
                     f"every testable cell supported: {'yes' if h['all_testable_supported'] else 'no'}")
        sep = t["H10a"].get("separation", {})
        L.append(f"- H10a separation (D088): {sep.get('separated_cells', 0)} of {sep.get('cells', 0)} cells, "
                 f"{sep.get('separated_replicates', 0)} replicates")
        for d, h in sorted(t["H10a_sixteenth"]["holm"].items()):
            L.append(f"- H10a at 1/16 x E0 (Boolean CGP, own family), {d}: m = {h['m']}; supported {h['supported']}, "
                     f"saturated {h['saturated']}, not estimable {h['not_estimable']}, missing {len(h['missing'])}")
    else:
        L.append(f"- H10a: {s.get('H10a')}")
    L += ["", "## H10b: transfer of the frozen law to CGP (per domain)", ""]
    _scored_lines(L, res_s, "H10b", "domain", "H10b")
    L += ["", "## Sensitivity analyses (labelled; the registered results are those above)", ""]
    for key, title in (("cpu_excluded", "CPU-limit stops excluded"), ("lower_bound", "Constructed levels at lower-bound sizes")):
        v = res["sensitivity"][key]
        L.append(f"### {title}")
        L.append(f"- {v['note']}.")
        if not v.get("recomputed"):
            continue
        if key == "cpu_excluded":
            L.append("- Censored runs per test: " + ", ".join(f"{k} {n}" for k, n in v["censored_runs"].items()) + ".")
        else:
            L.append("- Lower bound per constructed level (set: level -> size): " + "; ".join(
                f"{g}: " + ", ".join(f"{K} -> {k}" for K, k in ch.items()) for g, ch in sorted(v["changes"].items())) + ".")
            L.append("- Records of constructed targets per test: " + ", ".join(
                f"{k} {n}" for k, n in v["constructed_records"].items()) + ".")
        for test in SCORED_TESTS:
            for k, r in sorted((v.get(test) or {}).items()):
                L.append(f"- {test} {k}: {_passes(r)}; {_crit(r)}")
        L.append("- Decisions that change: " + ("; ".join(v["changed"]) if v["changed"] else "none") + ".")
    L += ["", "## Inputs (SHA-256; every frozen file checked against its .sha256 before use)", ""]
    L += [f"- `{rel}`: {sha}" for rel, sha in res["inputs"]["registered"].items() if "targets_s2" not in rel]
    n_t = sum(1 for rel in res["inputs"]["registered"] if "targets_s2" in rel)
    L.append(f"- {n_t} target file(s) read, each checked against its .sha256 (hashes in the JSON).")
    return "\n".join(L) + "\n"


def main(argv=None):
    from reach.paths import ROOT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", nargs="?", choices=("report", "transfer"), default="report")
    ap.add_argument("--root", default=None, help="repository root (tests)")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="compute tests on incomplete phases too, labelled not confirmatory")
    ap.add_argument("--force", metavar="REASON", default=None,
                    help="replace a stage2.json built from other inputs; the reason is recorded")
    a = ap.parse_args(argv)
    root = Path(a.root) if a.root else ROOT
    if a.command == "transfer":
        print(f"{transfer(root)}  {TRANSFER}")
        return
    res = build(root, a.allow_incomplete)
    write(root, res, a.force)
    print(f"written {OUT_JSON} and {OUT_MD}: " + "; ".join(f"{k}: {v}" for k, v in res["status"].items()))


if __name__ == "__main__":
    main()

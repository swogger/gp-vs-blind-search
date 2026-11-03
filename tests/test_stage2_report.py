"""The Stage 2 report driver (D095; REQ-ANA-81 to 86; guardrail row 139), written before the code.

The Stage 2 form promised "a driver that calls exactly the entry points named in this form" and two sensitivity
analyses; neither existed (targeted re-check of D094). Here a small synthetic Stage 2 data set is built in a temporary
root: the real frozen files (models, comparators, transfer model, CGP settings) and real held-out, benchmark and
calibration targets (a few per level of two sets per domain), with outcomes drawn from the frozen models, all six
phases, both Operon configurations, CGP rows with the claim and re-score fields, a run killed at its CPU limit without
a result, rows killed from outside and crashed rows. The driver runs end to end on it. No held-out run is read: none
exists."""
import copy
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from reach import grammar as G
from reach.analysis.s2 import cgp_confirm, h9b, law, predictor
from reach.analysis.s2 import confirm as C
from reach.analysis.s2 import stage2_freeze as SF
from reach.analysis.s2 import stage2_report as R
from reach.analysis.s2.phase1 import enum_dir
from reach.paths import ROOT
from reach.s2_lock import task_list_hash

HO = {"R-ARITH": "real", "R-ELEM6": "real", "BOOL-AN-n5": "bool", "BOOL-AX1-n5": "bool"}
CAL = {"R-ARITH4": "real", "R-ELEM4": "real", "BOOL-NAND-n5": "bool", "BOOL-AONX-n5": "bool"}
N_T = 3                                   # targets per level
SEEDS = (300, 301)
KILLED = ("s2_ho_a16", "R-ARITH")         # the CPU-killed run (no result): its target is the first at the lowest level
E0 = 20_000


def _sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1))
    p.with_suffix(".sha256").write_text(f"{_sha(p)}  {p.name}\n")


def _subset(targets, levels, n):
    out = []
    for K in levels:
        out += [t for t in targets if t["K_construction"] == K][:n]
    return out


def _prob(model, t, budget, feats):
    return float(predictor.predict(model, [dict(units=t["units"], budget=budget, features=feats)])[0])


def _targets(root):
    """Held-out and calibration target files (subsets of the real ones, re-hashed), the enumeration counts, bench
    logs (D050 rows) and properties; the reduced levels file; the prediction file written by the registered writer."""
    reg = root / "data/registration"
    reg.mkdir(parents=True)
    for name in ("h7_model", "operon_transfer", "gpf2013_comparator", "gp2010_comparator", "cgp_settings"):
        for ext in (".json", ".sha256"):
            shutil.copy(ROOT / f"data/registration/{name}{ext}", reg / f"{name}{ext}")
    lv = json.loads((ROOT / "data/registration/heldout_levels.json").read_text())
    _write(reg / "heldout_levels.json", {g: v for g, v in lv.items() if g in HO or g.startswith("_")})
    (root / "data/analysis").mkdir(parents=True)
    shutil.copy(ROOT / "data/analysis/properties.json", root / "data/analysis/properties.json")
    files = {}
    for g in HO:
        src = ROOT / f"data/targets_s2/{g}"
        full = json.loads((src / "heldout.json").read_text())
        levels = [lv[g]["main"][0], lv[g]["main"][-1], lv[g]["budget"][0], lv[g]["budget"][-1]]
        files[(g, "heldout")] = _subset(full, levels, N_T)
        files[(g, "bench")] = json.loads((src / "bench.json").read_text())[:2]
        _write(root / f"data/targets_s2/{g}/heldout.json", files[(g, "heldout")])
        _write(root / f"data/targets_s2/{g}/bench.json", files[(g, "bench")])
        shutil.copy(src / "bench_log.json", root / f"data/targets_s2/{g}/bench_log.json")
        d = root / enum_dir(g).relative_to(ROOT)
        d.mkdir(parents=True)
        shutil.copy(enum_dir(g) / "counts.json", d / "counts.json")
    cl = json.loads((ROOT / "data/registration/levels.json").read_text())
    for g in CAL:
        full = json.loads((ROOT / f"data/targets_s2/{g}/calibration.json").read_text())
        levels = [cl[g]["main"][0], cl[g]["main"][-1], cl[g]["budget"][0], cl[g]["budget"][-1]]
        files[(g, "calibration")] = _subset(full, levels, N_T + 1)
        _write(root / f"data/targets_s2/{g}/calibration.json", files[(g, "calibration")])
    SF.write_prediction_file(root)
    return files


def _queue(key, state, qstatus="ok", code=0, sig=0):
    return dict(key=key, terminal_state=state, queue_status=qstatus, exit=code, signal=sig, cpu_s=1.0, wall_s=1.0,
                cpu_limit=1000, finished="2026-10-28 00:00:00", code_hash="synth", git="synth")


def _meta(g, t, algo, cfg, seed, budget):
    return dict(set=g, target_id=t["target_id"], K=t["K_construction"], K_proven=t["K_proven"], algorithm=algo,
                config=cfg, seed=seed, budget=budget)


def _tree(block, g, t, seed, gens, p_at, rng):
    key = f"{block}|{g}|{t['target_id']}|A|base|{gens}|{seed}"
    row = dict(_meta(g, t, "A", "base", seed, gens // 200), generations=gens, pop_size=100)
    v = rng.random()
    hit = next((b for b in (1, 2, 4, 8, 16) if b * 200 <= gens and v < p_at(b)), None)
    ge = None if hit is None else hit * 200 - 5
    row.update(K_construction=t["K_construction"], units=t["units"], exact=hit is not None, gen_exact=ge,
               gens_reached=gens if ge is None else ge + 1)
    return dict(row, **_queue(key, "converged" if hit else "budget_exhausted"))


def _rows(root, files, h7, op, feats):
    """The six phases' rows from the frozen models; a CPU-killed run without a result in s2_ho_a16, a run killed from
    outside in s2_ho_a0, a crash in s2_ho_bench, an Operon error and a killed Operon run, a CGP claim the re-scoring
    rejected and a hard-killed CGP run in s2_cal_cgp."""
    rng = np.random.default_rng(20261028)
    out = {ph: [] for ph in R.PHASES}
    for g, d in HO.items():
        m = h7[d]
        for t in files[(g, "heldout")]:
            for s in SEEDS:
                p = lambda b, t=t: _prob(m, t, b, feats[g])                                   # noqa: E731
                out["s2_ho_a0"].append(_tree("ho_a0", g, t, s, 200, p, rng))
                out["s2_ho_a16"].append(_tree("ho_a16", g, t, s, 3200, p, rng))
        for t in files[(g, "bench")]:
            for s in (400, 401, 402):
                out["s2_ho_bench"].append(_tree("ho_bench", g, t, s, 200, lambda b, t=t: _prob(m, t, b, feats[g]), rng))
        if g in ("R-ARITH", "R-ELEM6"):
            for kind, seeds in (("heldout", SEEDS), ("bench", (400, 401, 402))):
                for t in files[(g, kind)]:
                    for cfg in h9b.CONFIGS:
                        for s in seeds:
                            key = f"ho_operon|{g}|{t['target_id']}|operon|{cfg}|16|{s}"
                            ex = bool(rng.random() < _prob(op[cfg], t, 16, feats[g]))
                            out["s2_ho_operon"].append(dict(_meta(g, t, "operon", cfg, s, 16), K_construction=t["K_construction"],
                                                            units=t["units"], exact=ex, model="(X1 + 1)",
                                                            **_queue(key, "converged" if ex else "budget_exhausted")))
    truth = {"real": (0.3, 0.8), "bool": (0.5, 0.6)}
    for ph, sets, kind, seeds in (("s2_cal_cgp", CAL, "calibration", (710, 711)), ("s2_ho_cgp", HO, "heldout", (715, 716))):
        for g, d in sets.items():
            n = 100 if d == "real" else 2000
            z = feats[g] if g in feats else C.set_features(root, [g])[g]
            for t in files[(g, kind)]:
                for s in seeds:
                    key = f"{ph[3:]}|{g}|{t['target_id']}|CGP|n{n}|16|{s}"
                    c0, c1 = truth[d]
                    v, hit = rng.random(), None
                    for b in (0.0625, 0.25, 1, 2, 4, 8, 16):
                        pb = _prob(h7[d], t, b, z)
                        if v < 1 / (1 + np.exp(-(c0 + c1 * np.log(pb / (1 - pb))))):
                            hit = b
                            break
                    ev = None if hit is None else int(hit * E0) - 10
                    out[ph].append(dict(_meta(g, t, "CGP", f"n{n}", s, 16), K_construction=t["K_construction"],
                                        units=t["units"], budget_E=16 * E0, exact=hit is not None, claim_exact=hit is not None,
                                        claim_rejected=False, evals_exact=ev, evals_reached=16 * E0 if ev is None else ev + 4,
                                        n_nodes=n, **_queue(key, "converged" if hit else "budget_exhausted")))
    # planted rows (each replaces a generated row of the same key)
    def plant(ph, i, **fields):
        r = out[ph][i]
        keep = {k: r[k] for k in ("key", "set", "target_id", "K", "K_proven", "algorithm", "config", "seed", "budget")}
        out[ph][i] = dict(keep, **fields)
    plant("s2_ho_a16", 0, generations=3200, pop_size=100, **_queue(out["s2_ho_a16"][0]["key"], "timeout", "cpu_limit", -1, 9))
    plant("s2_ho_a0", 1, generations=200, pop_size=100, error_tail="killed",
          **_queue(out["s2_ho_a0"][1]["key"], "error", "killed_signal_15", -1, 15))
    plant("s2_ho_bench", 0, generations=200, pop_size=100, error_tail="Traceback\nValueError: boom",
          **_queue(out["s2_ho_bench"][0]["key"], "error", "error", 1, 0))
    plant("s2_ho_operon", 0, error_tail="RuntimeError: operon", **_queue(out["s2_ho_operon"][0]["key"], "error", "error", 1, 0))
    plant("s2_ho_operon", 1, **_queue(out["s2_ho_operon"][1]["key"], "timeout", "cpu_limit", -1, 9))
    r = out["s2_cal_cgp"][0]
    out["s2_cal_cgp"][0] = dict(r, exact=False, claim_exact=True, claim_rejected=True, evals_exact=None, evals_reached=500,
                                terminal_state="early_stopped")
    plant("s2_cal_cgp", 1, budget_E=16 * E0, n_nodes=out["s2_cal_cgp"][1]["n_nodes"],
          **_queue(out["s2_cal_cgp"][1]["key"], "timeout", "cpu_limit", -1, 9))
    return out


def make_root(root):
    files = _targets(root)
    h7 = SF.load_h7(root / SF.PATHS["h7"])
    op = json.loads((root / h9b.TRANSFER_FILE).read_text())
    feats = json.loads((root / SF.PATHS["predictions"]).read_text())["features"]
    rows = _rows(root, files, h7, op, feats)
    reg = {}
    for ph, rr in rows.items():
        d = root / "data/runs" / ph
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps(dict(phase=ph, n_tasks=len(rr), tasks=[dict(key=r["key"]) for r in rr])))
        (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rr))
        reg[ph] = dict(n_tasks=len(rr), sha256=task_list_hash(rr))
    (root / "data/registration/stage2_tasks.json").write_text(json.dumps(reg, indent=1))
    return root


@pytest.fixture(scope="module")
def fast(request):
    """Speed only: the registered bootstrap sizes are passed by the driver (checked by `seen`) and cut here."""
    mp = pytest.MonkeyPatch()
    seen = dict(score_p=[], slope_ci=[], h10b=[])
    real_p, real_s, real_h = predictor.score_p, law.slope_ci, cgp_confirm.h10b
    mp.setattr(predictor, "score_p", lambda *a, **k: seen["score_p"].append(dict(k, eps_set=k["eps_set"] if "eps_set" in k else a[3]))
               or real_p(*a, **dict(k, boot=40)))                     # eps_set is positional in the comparators' calls
    mp.setattr(law, "slope_ci", lambda *a, **k: seen["slope_ci"].append(k) or real_s(*a, **dict(k, boot=20)))
    mp.setattr(cgp_confirm, "h10b", lambda *a, **k: seen["h10b"].append(k) or real_h(*a, **k))
    request.addfinalizer(mp.undo)
    return seen


@pytest.fixture(scope="module")
def built(tmp_path_factory, fast):
    root = make_root(tmp_path_factory.mktemp("stage2") / "root")
    res = R.build(root)
    R.write(root, res)
    return root, res


def _copy(built, tmp_path):
    root = tmp_path / "root"
    shutil.copytree(built[0], root)
    return root


def _rewrite(root, phase, fn):
    p = root / "data/runs" / phase / "results.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    p.write_text("".join(json.dumps(fn(r) or r) + "\n" for r in rows))


# ------------------------------------------------------------------ REQ-ANA-81, 83: end to end, registered code
def test_driver_runs_end_to_end_on_synthetic_phases(built):
    root, res = built
    assert all(res["phases"][ph]["status"] == "complete" for ph in R.PHASES) and res["confirmatory"]
    assert set(res["tests"]) >= {"H7", "comparators", "benchmarks", "H9b", "H9b_benchmarks", "H10a", "H10a_sixteenth"}
    assert set(res["tests"]["H7"]) == {"real", "bool"} and set(res["tests"]["H9b"]) == set(h9b.CONFIGS)
    for d in ("real", "bool"):
        h = res["tests"]["H7"][d]
        assert h["test"] == "H7" and h["n_sets_registered"] == 6 and h["sets_missing"] == 4 and "passes" in h
        assert set(h["per_set_error"]) == {g for g, dd in HO.items() if dd == d}
        assert {"comparator1", "comparator2", "frozen"} <= set(res["tests"]["comparators"][d])
        assert res["tests"]["benchmarks"][d]["descriptive"] and len(res["tests"]["benchmarks"][d]["per_cell_error"]) == 4
    assert res["tests"]["H10a"]["holm"]["real"]["m"] == 14 and res["tests"]["H10a"]["holm"]["bool"]["m"] == 18
    assert res["tests"]["H10a_sixteenth"]["holm"]["bool"]["m"] == 9
    assert res["status"]["H10b"].startswith("pending") and "H10b" not in res["tests"]
    # the records: every regime A run of both phases at the registered budgets, the killed run included as a failure
    n, runs = res["records"]["H7"], len(HO) * 4 * N_T * len(SEEDS)
    assert n["n"] == (runs - 1) + runs * 5 and n["budgets"] == [1, 2, 4, 8, 16]   # a0: 1 budget, one error row; a16: 5
    assert (root / R.OUT_JSON).exists() and (root / R.OUT_MD).exists()


def test_every_registered_test_appears_with_passes_from_the_registered_code(built):
    """Each `passes` equals a direct call of the registered entry point on the same records."""
    root, res = built
    h7 = SF.load_h7(root / SF.PATHS["h7"])
    feats = json.loads((root / SF.PATHS["predictions"]).read_text())["features"]
    recs = []
    for ph in ("s2_ho_a0", "s2_ho_a16"):
        rows = [json.loads(l) for l in (root / "data/runs" / ph / "results.jsonl").read_text().splitlines()]
        C._load_units(root, sorted(HO), ("heldout",))
        recs += C._with_units(C.tidy(rows, predictor.SCORED["H7"]["budgets"], True))
    for r in recs:
        r["features"] = feats[r["set"]]
    for d, rr in cgp_confirm.by_domain(recs).items():
        want = predictor.score_model("H7", h7[d], rr, h7[d]["baseline"])
        got = res["tests"]["H7"][d]
        assert got["passes"] == want["passes"] and got["auc"] == pytest.approx(want["auc"])
        assert got["brier"] == pytest.approx(want["brier"]) and got["sets_ok"] == want["sets_ok"]
    assert all("passes" in v and v["test"] == "H9b" for v in res["tests"]["H9b"].values())
    assert all(c["holm_status"] for c in res["tests"]["H10a"]["cells"])


def test_driver_passes_only_registered_constants(built, fast):
    for k in fast["score_p"]:
        assert k["boot"] == predictor.H7_RULE["boot"] == 2000 and k["eps_set"] == predictor.H7_RULE["eps_set"]
        assert k["min_sets_ok"] in {predictor.SCORED[t]["min_sets_ok"] for t in predictor.SCORED} | {0}
    assert fast["slope_ci"] and all(k["boot"] == C.REGISTERED["boot"] for k in fast["slope_ci"])
    src = Path(R.__file__).read_text()
    assert not re.findall(r"\b\d+\.\d+\b", src), "a numeric literal in the driver: thresholds come from the registered code"
    assert not re.findall(r"boot\s*=\s*\d", src) and "min_sets_ok=" not in src and "eps_set=" not in src


def test_form_entry_points_exist():
    """Guardrail row 139: every `reach.analysis.s2...` entry point the Stage 2 form names exists, the driver included."""
    import importlib
    text = (ROOT / "docs/registration/osf/stage2_form.md").read_text()
    names = {m.rstrip(".") for m in re.findall(r"reach\.analysis\.s2\.[\w.]+", text)}
    assert "reach.analysis.s2.stage2_report" in names
    for name in sorted(names):
        parts = name.split(".")
        mod, attrs = None, []
        for i in range(len(parts), 2, -1):
            try:
                mod, attrs = importlib.import_module(".".join(parts[:i])), parts[i:]
                break
            except ImportError:
                continue
        assert mod is not None, name
        for a in attrs:
            assert hasattr(mod, a), name
            mod = getattr(mod, a)


# ------------------------------------------------------------------ REQ-ANA-81, 85: killed rows, sensitivities
def test_killed_row_is_a_failure_in_the_primary_and_absent_in_the_cpu_sensitivity(built):
    root, res = built
    rows = [json.loads(l) for l in (root / "data/runs/s2_ho_a16/results.jsonl").read_text().splitlines()]
    killed = rows[0]
    assert killed["terminal_state"] == "timeout" and "gens_reached" not in killed and "units" not in killed
    a = res["sensitivity"]["cpu_excluded"]
    assert a["recomputed"] and a["censored_runs"]["H7"] == 1 and a["censored_runs"]["H9b"] == 1
    assert res["records"]["H7"]["censored"] == 5 and a["records"]["H7"]["censored"] == 0        # 5 budgets of one run
    assert a["records"]["H7"]["n"] == res["records"]["H7"]["n"] - 5
    assert set(a) >= {"H7", "H9b", "changed", "note"} and "excluded" in a["note"]
    assert res["no_units"] == []                                                               # units from the target file


def test_lower_bound_sensitivity_changes_only_constructed_levels(built):
    root, res = built
    b = res["sensitivity"]["lower_bound"]
    cons = {g: {t["K_construction"] for t in json.loads((root / f"data/targets_s2/{g}/heldout.json").read_text())
               if t.get("constructed")} for g in HO}
    assert {g: set(map(int, v)) for g, v in b["changes"].items()} == {g: ks for g, ks in cons.items() if ks}
    for g, ch in b["changes"].items():
        for K, k in ch.items():
            assert k <= int(K)
    assert b["constructed_records"]["H7"] > 0 and set(b) >= {"H7", "H9b", "changed", "note"}
    # directly: proven records keep their units, constructed ones move to the lower bound
    lower, _ = R.lower_bound_inputs(root, sorted(HO))
    recs = [dict(set=g, target_id=t["target_id"], K=t["K_construction"], units=t["units"])
            for g in HO for t in json.loads((root / f"data/targets_s2/{g}/heldout.json").read_text())]
    out = C.lower_bound_sizes(recs, *lower)
    for r, o in zip(recs, out):
        if (r["set"], r["target_id"]) in lower[0]:
            assert o["K"] < r["K"] or o["K"] == r["K"] == lower[1][r["set"]] + 1
            assert o["units"]["bits_sem"] != r["units"]["bits_sem"] or o["K"] == r["K"]
        else:
            assert o["K"] == r["K"] and o["units"] == r["units"]
    # the units function must be the generator's: a changed target file is refused, a last-bits difference (the
    # extrapolation's least-squares line on another platform's maths library; guardrail row 141) is not
    p = root / "data/targets_s2/BOOL-AN-n5/heldout.json"
    orig = p.read_text()
    try:
        t = json.loads(orig)
        top = max(t, key=lambda x: x["K_construction"])                      # an extrapolated (constructed) level
        assert top["units"]["bits_sem_extrapolated"]
        for x in t:
            x["units"]["bits_sem"] = float(np.nextafter(x["units"]["bits_sem"], np.inf))
        p.write_text(json.dumps(t))
        R.lower_bound_inputs(root, ["BOOL-AN-n5"])                          # old code: refused
        t[0]["units"]["bits_sem"] += 0.01
        p.write_text(json.dumps(t))
        with pytest.raises(ValueError, match="reproduce"):
            R.lower_bound_inputs(root, ["BOOL-AN-n5"])
    finally:
        p.write_text(orig)


def test_missing_phase_is_reported_not_fabricated(built, tmp_path):
    root = _copy(built, tmp_path)
    shutil.rmtree(root / "data/runs/s2_ho_operon")
    res = R.build(root)
    assert res["phases"]["s2_ho_operon"]["status"] == "missing" and not res["confirmatory"]
    assert res["status"]["H9b"].startswith("pending") and "H9b" not in res["tests"]
    assert "H7" in res["tests"] and res["status"]["H7"] == "computed"
    assert "H9b" not in res["sensitivity"]["cpu_excluded"]
    text = R.render(res)
    assert "missing" in text and "pending" in text


# ------------------------------------------------------------------ REQ-ANA-82: inputs, keys, sets, budgets
def test_tampered_frozen_or_target_file_is_refused(built, tmp_path):
    root = _copy(built, tmp_path)
    p = root / SF.PATHS["h7"]
    p.write_text(p.read_text().replace("3.85", "3.86", 1))
    with pytest.raises(ValueError, match="checksum"):
        R.build(root)
    root = _copy(built, tmp_path / "b")
    t = root / "data/targets_s2/R-ELEM6/heldout.json"
    _write(t, json.loads(t.read_text())[:-1])                    # re-hashed, so only the prediction file's record differs
    with pytest.raises(ValueError, match="prediction file"):
        R.build(root)
    root = _copy(built, tmp_path / "c")
    (root / "data/registration/stage2_tasks.json").unlink()
    with pytest.raises(FileNotFoundError, match="registered task"):
        R.build(root)


def test_unregistered_set_or_budget_is_refused(built, tmp_path):
    root = _copy(built, tmp_path)
    _rewrite(root, "s2_ho_a0", lambda r: dict(r, set="R-TRIG") if r["key"].endswith("|300") and "R-ARITH|" in r["key"] else r)
    with pytest.raises(ValueError, match="R-TRIG"):
        R.build(root)
    root = _copy(built, tmp_path / "b")
    _rewrite(root, "s2_ho_operon", lambda r: dict(r, budget=4) if r["algorithm"] == "operon" and r["seed"] == 301 else r)
    with pytest.raises(ValueError, match="budget"):
        R.build(root)
    root = _copy(built, tmp_path / "c")                        # a row whose key is not registered: the phase is incomplete
    p = root / "data/runs/s2_ho_bench/results.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    p.write_text(p.read_text() + json.dumps(dict(rows[-1], key="ho_bench|extra")) + "\n")
    res = R.build(root)
    assert res["phases"]["s2_ho_bench"]["status"] == "incomplete" and res["status"]["benchmarks"].startswith("pending")
    with pytest.raises(ValueError, match="not in the phase's task list"):
        R.build(root, allow_incomplete=True)


def test_outputs_record_every_input_and_refuse_a_silent_replacement(built, tmp_path):
    root, res = built
    reg = res["inputs"]["registered"]
    assert set(reg) >= {R.FROZEN[k] for k in R.FROZEN} | {R.TASKS, R.PROPERTIES}
    assert all(f"data/targets_s2/{g}/{n}.json" in reg for g in HO for n in ("heldout", "bench"))
    assert all(f"data/targets_s2/{g}/calibration.json" in reg for g in CAL)
    assert reg[R.FROZEN["h7"]] == _sha(root / R.FROZEN["h7"]) and set(res["inputs"]["phases"]) == set(R.PHASES)
    assert res["code"]["commit"] and len(res["code"]["code_hash"]) == 12
    root2 = _copy(built, tmp_path)
    R.write(root2, copy.deepcopy(res))                                                    # same inputs: replaced
    old = json.loads((root2 / R.OUT_JSON).read_text())
    old["inputs"]["registered"][R.FROZEN["h7"]] = "0" * 64
    (root2 / R.OUT_JSON).write_text(json.dumps(old))
    with pytest.raises(RuntimeError, match="force"):
        R.write(root2, copy.deepcopy(res))
    assert json.loads((root2 / R.OUT_JSON).read_text())["inputs"]["registered"][R.FROZEN["h7"]] == "0" * 64
    R.write(root2, copy.deepcopy(res), force="test: a changed frozen input")
    new = json.loads((root2 / R.OUT_JSON).read_text())
    assert new["force"]["reason"] == "test: a changed frozen input" and R.FROZEN["h7"] in " ".join(new["force"]["replaced"])
    old = json.loads((root2 / R.OUT_JSON).read_text())                                   # a rerun phase's records
    old["inputs"]["phases"]["s2_ho_a0"] = "1" * 64
    (root2 / R.OUT_JSON).write_text(json.dumps(old))
    with pytest.raises(RuntimeError, match="s2_ho_a0"):
        R.write(root2, copy.deepcopy(res))


# ------------------------------------------------------------------ REQ-ANA-84: H10b and the transfer model
def test_h10b_is_pending_without_the_transfer_model_and_scored_with_it(built, tmp_path, fast):
    root = _copy(built, tmp_path)
    ho = root / "data/runs/s2_ho_cgp/results.jsonl"
    keep = ho.read_text()
    ho.unlink()                                                   # before any held-out CGP run: fit and freeze
    sha = R.transfer(root)
    p = root / cgp_confirm.TRANSFER_FILE
    assert p.exists() and p.with_suffix(".sha256").read_text().split()[0] == sha == _sha(p)
    cgp_confirm.check_transfer(json.loads(p.read_text()))
    ho.write_text(keep)
    res = R.build(root)
    assert res["status"]["H10b"] == "computed" and set(res["tests"]["H10b"]) == {"real", "bool"}
    for d in ("real", "bool"):
        h = res["tests"]["H10b"][d]
        assert "passes" in h and h["n_sets_registered"] == 6 and h["sets_missing"] == 4 and h["n"] > 0
    assert fast["h10b"][-1]["boot"] == C.REGISTERED["boot"]
    assert "H10b" in res["sensitivity"]["cpu_excluded"] and "H10b" in res["sensitivity"]["lower_bound"]
    assert res["inputs"]["registered"][cgp_confirm.TRANSFER_FILE] == sha
    p.write_text(p.read_text().replace("c0", "c9", 1))           # a changed transfer file is refused
    with pytest.raises(ValueError, match="checksum"):
        R.build(root)


def test_transfer_fit_writes_once_and_refuses_to_overwrite(built, tmp_path):
    root = _copy(built, tmp_path)
    with pytest.raises(RuntimeError, match="held-out CGP"):     # rows of s2_ho_cgp exist: too late to freeze
        R.transfer(root)
    ho = root / "data/runs/s2_ho_cgp/results.jsonl"
    ho.unlink()
    sha = R.transfer(root)
    assert R.transfer(root) == sha                                # same content: no change
    _rewrite(root, "s2_cal_cgp", lambda r: dict(r, exact=not r["exact"], evals_exact=None if r["exact"] else 100)
             if r["seed"] == 710 and r["terminal_state"] in ("converged", "budget_exhausted") else r)
    with pytest.raises(RuntimeError, match="never replaced"):
        R.transfer(root)
    assert (root / cgp_confirm.TRANSFER_FILE).with_suffix(".sha256").read_text().split()[0] == sha
    p = root / "data/runs/s2_cal_cgp/results.jsonl"               # an incomplete phase: no fit
    (root / cgp_confirm.TRANSFER_FILE).unlink()
    p.write_text("".join(p.read_text().splitlines(True)[:-1]))
    with pytest.raises(RuntimeError, match="complete"):
        R.transfer(root)


# ------------------------------------------------------------------ REQ-ANA-86: the report
def test_report_is_short_and_placeholder_free(built):
    root, res = built
    text = (root / R.OUT_MD).read_text()
    lines = text.splitlines()
    assert len(lines) <= 150 and text == R.render(res)
    assert "{" not in text and "TODO" not in text and "never quote" not in text
    assert text.startswith("# ") and "Do not edit by hand" in text
    for word in ("H7", "H9b", "H10a", "H10b", "comparator", "enchmark", "ensitivity", "pending", "SHA-256"):
        assert word in text, word
    for d in ("real", "bool"):
        assert C._f(res["tests"]["H7"][d]["auc"]) in text
    assert "status/stage2_report.md" in (ROOT / "docs/README.md").read_text()

"""Stage 5 analysis driver (REQ-ANA-120 to REQ-ANA-128; docs/design/stage5_analysis.md): the refusals, each on a planted
case; Rule S1 in every contrast; the estimators; each test's statistic against an independent computation; Holm per
family and the stage-wide Holm; the instrument checks on the registered records; end to end on a synthetic root.
Written with the requirements, before the driver was run on any Stage 5 phase."""
import copy
import json
import math
import shutil

import numpy as np
import pytest

import _stage5_synth as Y
from reach import s2_lock as L
from reach.analysis.s2 import alloc, h6d, iso, stage5
from reach.analysis.s2 import stage5_report as D
from reach.analysis.s2 import stage5_stats as S


def _logit(x):
    return 1 / (1 + math.exp(-x))


def _exact(sizes, k50, slope=1.2, prefix="t"):
    """An arm whose probabilities are exactly logistic in bits with the given K50 (recovered exactly by the fit)."""
    return {f"{prefix}{i}": (float(b), _logit(slope * (k50 - b)), 5) for i, b in enumerate(sizes)}


SIZES = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0] * 2


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("s5root")
    cfg, levels, extra = Y.make_root(root)
    return root, cfg, levels, extra


def _copy(synth, tmp_path):
    root, cfg, levels, extra = synth
    dst = tmp_path / "root"
    shutil.copytree(root, dst)
    return dst, cfg, levels


def _rows(root, phase):
    return D.phase_rows(root, phase)


def _tables(root, sets):
    tables, shas = {}, {}
    for g in sets:
        tables[g], shas[g] = D.target_table(root, g, ("calibration", "low", "heldout", "restart"))
    return tables, shas


# ---------------------------------------------------------------- refusals (REQ-ANA-120 to 122)
def test_refuses_an_incomplete_or_extra_phase(tmp_path, synth):
    rows = [dict(key=f"s5_grid|R-X|t{i}|A|FG|200|1", set="R-X", target_id=f"t{i}", seed=1) for i in range(3)]
    tasks = Y.write_phase(tmp_path, "s5_grid", rows)
    assert len(D.phase_rows(tmp_path, "s5_grid")) == 3
    Y.write_phase(tmp_path, "s5_grid", rows[:2], tasks=tasks)                       # plant 1: a missing row
    with pytest.raises(RuntimeError, match="incomplete or extra"):
        D.phase_rows(tmp_path, "s5_grid")
    Y.write_phase(tmp_path, "s5_grid", rows + [dict(rows[0], key="s5_grid|R-X|other")], tasks=tasks)   # plant 2
    with pytest.raises(RuntimeError, match="incomplete or extra"):
        D.phase_rows(tmp_path, "s5_grid")
    with pytest.raises(RuntimeError, match="not planned"):                         # plant 3: no manifest
        D.phase_rows(tmp_path, "s5_boolsub")
    # a family is analysed only when its phases are complete; another family's incomplete phase does not matter
    root, cfg, levels = _copy(synth, tmp_path)
    p = root / "data/runs/s5_pop1000/results.jsonl"
    p.write_text("".join(p.read_text().splitlines(keepends=True)[1:]))
    with pytest.raises(RuntimeError, match="s5_pop1000"):
        D.build(root, cfg, boot=3, families=["H14"], levels_given=levels)
    assert "H16-2" in D.build(root, cfg, boot=3, families=["H16"], levels_given=levels)["tests"]


def test_refuses_a_task_list_off_the_registration(tmp_path):
    rows = [dict(key=f"k{i}", set="R-X", target_id=f"t{i}", seed=1) for i in range(3)]
    tasks = Y.write_phase(tmp_path, "s5_grid", rows)
    reg_dir = tmp_path / "data/registration"
    reg_dir.mkdir(parents=True)
    good = dict(n_tasks=3, sha256=L.task_list_hash(tasks), sha256_full=L.task_list_full_hash(tasks))
    (reg_dir / "stage5_tasks.json").write_text(json.dumps({"s5_grid": good}))
    D.phase_rows(tmp_path, "s5_grid")
    (reg_dir / "stage5_tasks.json").write_text(json.dumps({"s5_grid": dict(good, sha256="0" * 64)}))
    with pytest.raises(RuntimeError, match="not the one registered"):              # plant 1: other keys registered
        D.phase_rows(tmp_path, "s5_grid")
    (reg_dir / "stage5_tasks.json").write_text(json.dumps({"s5_grid": good}))
    changed = copy.deepcopy(tasks)
    changed[0]["argv"].append("--timeout")                                         # plant 2: same keys, other argv
    Y.write_phase(tmp_path, "s5_grid", rows, tasks=changed)
    with pytest.raises(RuntimeError, match="full hash"):
        D.phase_rows(tmp_path, "s5_grid")
    Y.write_phase(tmp_path, "s5_wide", rows)                                       # plant 3: not registered
    with pytest.raises(RuntimeError, match="not in stage5_tasks.json"):
        D.phase_rows(tmp_path, "s5_wide")
    # an earlier stage's registration is checked by key hash (s4_restart against stage4_tasks.json)
    Y.write_phase(tmp_path, "s4_restart", rows)
    (reg_dir / "stage4_tasks.json").write_text(json.dumps({"s4_restart": dict(n_tasks=3, sha256="1" * 64)}))
    with pytest.raises(RuntimeError, match="stage4_tasks.json"):
        D.phase_rows(tmp_path, "s4_restart")
    # a pilot phase is never checked against a registration
    Y.write_phase(tmp_path, "s5_pilot_grid", rows)
    assert len(D.phase_rows(tmp_path, "s5_pilot_grid")) == 3


def test_refuses_an_unreproduced_reexecution_or_long_run_prefix(tmp_path):
    t = dict(target_id="R-X_K07_00", K_construction=7, K_proven=7, units=dict(bits_sem=9.6))
    src = [Y.run_row("s2_cal_a0", f"cal_a0|R-X|{t['target_id']}|A|base|200|{s}", "R-X", t, s, "base", 200, 100, 11.0)
           for s in (1, 2, 3)]
    Y.write_phase(tmp_path, "s2_cal_a0", src)
    again = Y.reexec(tmp_path, "s2_revisit_a0", "s2_cal_a0", src)
    assert D.check_reexec(tmp_path, "s2_revisit_a0") == dict(rows=3, identical=3)
    bad = [dict(again[0], final_tree_size=99)] + again[1:]                          # plant 1: a field differs
    Y.write_phase(tmp_path, "s2_revisit_a0", bad, json.loads((tmp_path / "data/runs/s2_revisit_a0/manifest.json")
                                                             .read_text())["tasks"])
    with pytest.raises(RuntimeError, match="do not reproduce"):
        D.check_reexec(tmp_path, "s2_revisit_a0")
    Y.write_phase(tmp_path, "s4_restart", src)
    rr = Y.reexec(tmp_path, "s5_revisit_restart", "s4_restart", src, keys=True)
    assert D.check_reexec(tmp_path, "s5_revisit_restart", keys=True)["identical"] == 3
    (tmp_path / rr[1]["keys"]["file"]).write_bytes(b"changed")                      # plant 2: a side file changed
    with pytest.raises(RuntimeError, match="do not reproduce"):
        D.check_reexec(tmp_path, "s5_revisit_restart", keys=True)
    # the long run's prefix (REQ-RUN-49): its first 200 generations must reproduce the s2_ho_a0 row
    short = [dict(r, key=f"ho_a0|R-X|{t['target_id']}|A|base|200|{r['seed']}", best_expr="x", final_tree_size=3)
             for r in src]
    Y.write_phase(tmp_path, "s2_ho_a0", short)
    long = [dict(r, key=f"s5_long_extra|{r['key']}", generations=3200, budget_E=320_000,
                 gens_reached=r["gens_reached"] if r["exact"] else 3200) for r in short]
    tasks = [dict(key=x["key"], argv=[], meta=dict(prefix_of="s2_ho_a0", prefix_key=s["key"]))
             for x, s in zip(long, short)]
    Y.write_phase(tmp_path, "s5_long_extra", long, tasks)
    assert D.check_long_extra(tmp_path, "s5_long_extra")["compared"] == 3
    hit = next(i for i, r in enumerate(long) if r["exact"])
    long[hit] = dict(long[hit], gen_exact=long[hit]["gen_exact"] + 1)               # plant 3: a moved hit
    Y.write_phase(tmp_path, "s5_long_extra", long, tasks)
    with pytest.raises(RuntimeError, match="prefix"):
        D.check_long_extra(tmp_path, "s5_long_extra")


def test_refuses_a_target_file_off_its_checksum(tmp_path):
    ts = Y.write_targets(tmp_path, "R-X", "calibration", [7, 9])
    Y.write_targets(tmp_path, "R-X", "low", [3])
    table, shas = D.target_table(tmp_path, "R-X", ("calibration", "low"))
    assert len(table) == 3 * Y.N_PER and set(shas) == {"calibration", "low"}
    row = dict(key="k", set="R-X", target_id=ts[0]["target_id"], targets_sha256=shas["calibration"])
    D.check_rows_targets([row], {"R-X": table}, {"R-X": shas})
    with pytest.raises(RuntimeError, match="not the registered"):                   # plant 1: run on another file
        D.check_rows_targets([dict(row, targets_sha256="f" * 64)], {"R-X": table}, {"R-X": shas})
    with pytest.raises(RuntimeError, match="not a target"):                         # plant 2: an unknown target
        D.check_rows_targets([dict(row, target_id="R-X_K99_00")], {"R-X": table}, {"R-X": shas})
    reg = {"data/targets_s2/R-X/low.json": "0" * 64}                               # plant 3: off the checksum list
    with pytest.raises(RuntimeError, match="checksum list"):
        D.target_table(tmp_path, "R-X", ("calibration", "low"), reg)
    p = tmp_path / "data/targets_s2/R-X/calibration.json"
    p.write_text(p.read_text().replace("9.5", "9.6", 1) + " ")                     # plant 4: a tampered file
    with pytest.raises(RuntimeError, match="differs from calibration.sha256"):
        D.target_table(tmp_path, "R-X", ("calibration", "low"))
    assert D.proven(dict(K_proven=7, K_construction=7)) and not D.proven(dict(K_proven=5, K_construction=7))
    assert not D.proven(dict(K_proven=7, K_construction=7, constructed=True))


def test_refuses_rows_of_another_lookup():
    rows = [dict(key=f"k{i}", set="R-X", rv=dict(lookup=dict(sha256="a"))) for i in range(2)]
    assert D.check_lookups(rows) == {"R-X": "a"}
    assert D.check_lookups(rows, {"R-X": "a"}) == {"R-X": "a"}
    with pytest.raises(RuntimeError, match="different lookups"):                    # plant 1: two lookups in a set
        D.check_lookups(rows + [dict(key="k9", set="R-X", rv=dict(lookup=dict(sha256="b")))])
    with pytest.raises(RuntimeError, match="registered"):                           # plant 2: not the registered one
        D.check_lookups(rows, {"R-X": "c"})
    assert D.check_lookups([dict(key="u", set="H-A", lookup=dict(sha256="z"))]) == {"H-A": "z"}   # union rows
    # every union histogram sums to U_K with n' + 1 cells (plant 3: a histogram from another lookup)
    U = {1: 4, 2: 6}
    good = dict(target_id="t", hist={"10000": dict(nohit={"1": [1, 2, 1], "2": [6, 0, 0]}, n_nohit=2)})
    assert set(D.union_twins([good], {"t": (8.0, 1)}, U, 2)) == {"t"}
    bad = copy.deepcopy(good)
    bad["hist"]["10000"]["nohit"]["2"] = [5, 0, 0]
    with pytest.raises(RuntimeError, match="summing"):
        D.union_twins([bad], {"t": (8.0, 1)}, U, 2)


# ---------------------------------------------------------------- Rule S1 in every contrast (REQ-ANA-123)
def test_every_contrast_refuses_arms_on_other_rows(synth):
    root, cfg, levels, _ = synth
    real, boolean, held = cfg["real"], cfg["boolean"], cfg["heldout"]
    tables, _ = _tables(root, list(real) + list(boolean) + list(held))
    base = _rows(root, "s2_cal_a0") + _rows(root, "s2_cal_a16")
    h12 = dict(grid=_rows(root, "s5_grid"), wide=_rows(root, "s5_wide"), boolsub=_rows(root, "s5_boolsub"), base=base)
    assert len(D.assemble_h12(h12, tables, real, boolean, cfg["wide_excluded"])) == 3
    t0 = h12["grid"][0]["target_id"]
    plants = {"one target fewer": [r for r in h12["grid"] if r["target_id"] != t0],
              "a swapped target": [dict(r, target_id=h12["grid"][-1]["target_id"]) if r["target_id"] == t0 else r
                                   for r in h12["grid"]],
              "another seed": [dict(h12["grid"][0], seed=999)] + h12["grid"][1:]}
    for what, grid in plants.items():
        with pytest.raises(ValueError, match="differ"):
            D.assemble_h12(dict(h12, grid=grid), tables, real, boolean, cfg["wide_excluded"])
    with pytest.raises(ValueError, match="differ"):                                 # H12-3: another exclusion
        D.assemble_h12(h12, tables, real, boolean, ("R-X_K11_01",))
    with pytest.raises(ValueError, match="differ"):                                 # H12-2: a Boolean row dropped
        D.assemble_h12(dict(h12, boolsub=h12["boolsub"][1:]), tables, real, boolean, cfg["wide_excluded"])
    # H13: SF on S_d (from the target files), the base arm on S_real, the twin's rows
    cal = json.loads((root / "data/registration/levels.json").read_text())
    U = D.levels(root, list(real) + list(boolean) + list(held), levels)
    h13 = dict(selfree=_rows(root, "s5_selfree"), low_base=_rows(root, "s5_low_base"), base=base,
               stage3=_rows(root, "s2_revisit_a0") + _rows(root, "s2_revisit_a16"))
    assert len(D.assemble_h13(h13, tables, cal, U, real, boolean)) == 2
    low_t = next(r["target_id"] for r in h13["selfree"] if "_K03_" in r["target_id"])
    with pytest.raises(ValueError, match="differ"):
        D.assemble_h13(dict(h13, selfree=[r for r in h13["selfree"] if r["target_id"] != low_t]), tables, cal, U, real,
                       boolean)
    bool_t = next(r["target_id"] for r in h13["selfree"] if r["set"] in boolean)   # only the S_d check sees a
    with pytest.raises(ValueError, match="S_d"):                                    # Boolean set (T and the twin are real)
        D.assemble_h13(dict(h13, selfree=[r for r in h13["selfree"] if r["target_id"] != bool_t]), tables, cal, U,
                       real, boolean)
    above = next(t for t, v in tables[boolean[0]].items() if v["units"]["bits_sem"] > D.CUT_BITS)
    extra = dict(h13["selfree"][0], key="extra", set=boolean[0], target_id=above)   # a run above the 18.29-bit cut
    with pytest.raises(ValueError, match="S_d"):
        D.assemble_h13(dict(h13, selfree=h13["selfree"] + [extra]), tables, cal, U, real, boolean)
    low_r = next(r["target_id"] for r in h13["low_base"] if r["set"] in real)
    with pytest.raises(ValueError, match="differ"):
        D.assemble_h13(dict(h13, low_base=[r for r in h13["low_base"] if r["target_id"] != low_r]), tables, cal, U,
                       real, boolean)
    # the held-out block (H14, H15, H11-2L) on the restart rows
    ho = dict(restart=_rows(root, "s4_restart"), pop1000=_rows(root, "s5_pop1000"), union=_rows(root, "s5_union"),
              long_extra=_rows(root, "s5_long_extra"), one16=_rows(root, "s2_ho_a16"))
    ho["long_counted"] = [r for r in ho["long_extra"] if r.get("rv")] + _rows(root, "s5_revisit_ho16")
    D.assemble_heldout(ho, tables, U, held)
    t1 = ho["pop1000"][0]["target_id"]
    for key, plant in (("pop1000", [r for r in ho["pop1000"] if r["target_id"] != t1]),
                       ("union", ho["union"][1:]),
                       ("long_extra", [r for r in ho["long_extra"] if r["target_id"] != ho["long_extra"][0]["target_id"]]),
                       ("long_counted", ho["long_counted"][1:])):
        with pytest.raises(ValueError, match="differ"):
            D.assemble_heldout(dict(ho, **{key: plant}), tables, U, held)
    with pytest.raises(ValueError, match="both"):                                   # a run in both one-run phases
        D.assemble_heldout(dict(ho, long_extra=ho["long_extra"] + [dict(ho["one16"][0], key="dup")]), tables, U, held)
    with pytest.raises(ValueError, match="more than one counted"):
        D.assemble_heldout(dict(ho, long_counted=ho["long_counted"] + ho["long_counted"][:1]), tables, U, held)
    # H16: the same (set, target, seed) runs at every population
    h16 = dict(revisit_robust=_rows(root, "s5_revisit_robust"), stage3_a16=_rows(root, "s2_revisit_a16"))
    D.assemble_h16(h16, tables, U, cfg["robust_real"], cfg["robust_bool"])
    r2 = [r for r in h16["revisit_robust"] if r["config"] == "R2"]
    with pytest.raises(ValueError, match="differs"):
        D.assemble_h16(dict(h16, revisit_robust=[r for r in h16["revisit_robust"] if r is not r2[0]]), tables, U,
                       cfg["robust_real"], cfg["robust_bool"])


# ---------------------------------------------------------------- estimators (REQ-ANA-124)
def test_bracketing_and_the_isotonic_reach_beside():
    sizes = [5.0, 7.0, 9.0, 11.0]
    r = S.reach(sizes, [1.0] * 4)                                                   # every target solved: +inf
    assert r["raw"] == float("inf") and r["k50"] == 11.0 and r["bracketed"]
    assert math.isnan(S.reach(sizes, [0.0] * 4)["k50"])                              # none solved: -inf, not estimable
    ps = [_logit(1.2 * (8.0 - x)) for x in sizes]
    r = S.reach(sizes, ps)
    assert r["k50"] == pytest.approx(8.0, abs=1e-6) and not r["bracketed"]
    assert r["iso"] == iso.reach(sizes, ps)["reach"] and r["iso_flag"] == "bracketed"
    r = S.reach(sizes, [_logit(0.8 * (14.0 - x)) for x in sizes])                   # above the top: the top, flagged
    assert r["raw"] == pytest.approx(14.0, abs=1e-6) and r["k50"] == 11.0 and r["bracketed"]
    assert math.isnan(S.reach(sizes, [0.1, 0.4, 0.6, 0.9])["k50"])                   # a rising curve: NaN
    assert math.isnan(S.reach([], [])["k50"])
    # one run: error rows out of n, a CPU stop a failure, a run shorter than L refused
    t = dict(target_id="t", K_construction=7, K_proven=7, units=dict(bits_sem=9.0))
    rows = [Y.run_row("b", f"k{s}", "R-X", t, s, "base", 200, 100, 30.0) for s in range(3)]
    rows[1] = dict(rows[1], terminal_state="error")
    rows[2] = dict(rows[2], exact=False, gen_exact=None, terminal_state="timeout", gens_reached=50)
    assert S.one_run_arm(rows, 200, {"t": 9.0}) == {"t": (9.0, 0.5, 2)}
    with pytest.raises(ValueError):
        S.one_run_arm(rows, 300, {"t": 9.0})


# ---------------------------------------------------------------- statistics (REQ-ANA-125 to 127)
def test_h12_statistics():
    base = {g: _exact(SIZES, 9.0 + i) for i, g in enumerate(("A", "B"))}
    new = {g: _exact(SIZES, 9.4 + i) for i, g in enumerate(("A", "B"))}
    ids = {g: sorted(v) for g, v in base.items()}
    stat = S.h12_contrast("H12-1", new, base)
    pt = stat(ids)
    assert pt["values"]["H12-1"]["primary"] == pytest.approx(0.4, abs=1e-6)
    manual = np.mean([iso.reach(*zip(*[(b, p) for b, p, _ in new[g].values()]))["reach"]
                      - iso.reach(*zip(*[(b, p) for b, p, _ in base[g].values()]))["reach"] for g in ids])
    assert pt["values"]["H12-1"]["iso"] == pytest.approx(manual)
    res, _ = S.run_block(["H12-1"], ids, stat, boot=40)
    t = res["H12-1"]["tests"]["primary"]
    assert all(x == pytest.approx(0.4, abs=1e-5) for x in res["H12-1"]["replicates"]["primary"])
    assert t["p"] == pytest.approx(1 / 41) and t["estimable"] and t["equivalent"]
    # H12-2: the mean gap on proven targets; a K50 above the top level is the top (a lower bound)
    arm = {"A": _exact(SIZES, S.NULL_E0 - 6.0), "B": _exact(SIZES, 20.0, slope=0.6)}
    v = S.h12_sign("H12-2", arm, base)(ids)
    assert v["per_set"]["B"]["arm"]["bracketed"] and v["per_set"]["B"]["gap"] == pytest.approx(12.0 - S.NULL_E0)
    assert v["values"]["H12-2"]["primary"] == pytest.approx((-6.0 + 12.0 - S.NULL_E0) / 2, abs=1e-6)
    assert v["per_set"]["A"]["d"] == pytest.approx(S.NULL_E0 - 6.0 - 9.0, abs=1e-6)


def _cov_runs(targets, frac, E=S.E0, hit=()):
    """Counted runs whose coverage of level k at prefix p is frac[k] * U_k * p / E (one run per target)."""
    out = {}
    for t, (bits, K) in targets.items():
        rv = dict(prefixes={str(p): dict(lv=[0] + [frac.get(k, 0.0) * Y.U[k] * p / E for k in range(1, Y.TOP + 1)])
                            for p in (5_000, 10_000, 20_000, 40_000, 80_000, 160_000, 320_000) if p <= E})
        out[t] = [dict(target_id=t, K=K, bits_sem=bits, y={E: int(t in hit)}, rv=rv)]
    return out


def test_h13_statistics():
    real, boolean = ["A", "B"], ["C", "D"]
    sf = {g: _exact(SIZES, 6.5 if g in real else 7.0) for g in real + boolean}
    ids = {g: sorted(v) for g, v in sf.items()}
    assert S.h13_domains("H13-1", sf, real, boolean)(ids)["values"]["H13-1"]["primary"] == pytest.approx(0.5, abs=1e-6)
    # H13-2: the twin from the counted runs of every picked target, read on S_real's rows
    rows = {g: {t: (b, int(b)) for t, (b, _p, _n) in sf[g].items()} for g in real}
    others = {g: {f"cov{i}": (20.0, 12) for i in range(3)} for g in real}          # targets above the cut
    frac = {k: max(0.0, 1.6 - 0.2 * k) for k in range(1, Y.TOP + 1)}
    cov = {g: {**_cov_runs(rows[g], frac), **_cov_runs(others[g], {k: 0.0 for k in frac})} for g in real}
    U = {g: dict(Y.U) for g in real}
    gp = {g: _exact(SIZES, 8.0) for g in real}
    pick = {g: sorted(set(rows[g]) | set(others[g])) for g in real}
    v = S.h13_twin("H13-2", {g: sf[g] for g in real}, rows, cov, U, gp=gp)(pick)
    for g in real:
        runs = [r for t in pick[g] for r in cov[g][t]]
        probs = h6d.twin_probs(h6d.cumulative(runs, S.E0, "lv", "stratified"), U[g])
        ts = sorted(rows[g])
        k_twin = S.reach([rows[g][t][0] for t in ts], [probs[rows[g][t][1]] for t in ts])["k50"]
        assert v["per_set"][g]["twin"]["k50"] == pytest.approx(k_twin)
        assert v["per_set"][g]["d"] == pytest.approx(6.5 - k_twin, abs=1e-6)
        assert v["per_set"][g]["T"] == pytest.approx(1.5, abs=1e-6)
        # plant: the twin from the S_real targets' runs alone differs (the high-level runs count in the coverage)
        alone = h6d.twin_probs(h6d.cumulative([r for t in ts for r in cov[g][t]], S.E0, "lv", "stratified"), U[g])
        assert S.reach([rows[g][t][0] for t in ts], [alone[rows[g][t][1]] for t in ts])["k50"] != pytest.approx(k_twin)
    assert v["T"]["primary"] == pytest.approx(1.5, abs=1e-6)


def test_restart_twin_fast_path_equals_the_library():
    rng = np.random.default_rng(20261108)
    U = {1: 5, 2: 9, 3: 14, 4: 20}
    targets = []
    for i in range(12):
        n = int(rng.integers(0, 7))
        hist = {}
        for K in (1, 2, 3):                                          # level 4 lies above the lookup: no histogram
            h = rng.multinomial(U[K], np.ones(n + 1) / (n + 1)).tolist()
            hist[K] = h
        targets.append(dict(id=f"t{i}", level=int(rng.integers(1, 4)), n_prime=n, hist=hist))
    ids = [t["id"] for t in targets] + ["t0", "t3", "t3"]
    by = {t["id"]: t for t in targets}
    for R in (1, 2, 3):
        entries = {t["id"]: (t["level"], S.twin_vector(t["hist"], t["n_prime"], U, R)) for t in targets}
        fast = S.restart_twin(entries, ids, U)
        lib = stage5.restart_twin_shares([by[t] for t in ids], R, U)["shares"]
        assert fast == pytest.approx(lib, abs=1e-12)
        assert fast[4] == 0.0
        # plant: levels weighted equally instead of by their targets gives other shares
        lv = {}
        for t in ids:
            lv.setdefault(by[t]["level"], []).append(entries[t][1])
        eq = np.mean([np.mean([v for v in vs if v is not None], axis=0) for vs in lv.values()
                      if any(v is not None for v in vs)], axis=0)
        assert not np.allclose(eq, [fast[K] for K in sorted(U)])


def test_heldout_block_statistics(synth):
    root, cfg, levels, _ = synth
    held = cfg["heldout"]
    tables, _ = _tables(root, held)
    U = D.levels(root, held, levels)
    ho = dict(restart=_rows(root, "s4_restart"), pop1000=_rows(root, "s5_pop1000"), union=_rows(root, "s5_union"),
              long_extra=_rows(root, "s5_long_extra"), one16=_rows(root, "s2_ho_a16"))
    ho["long_counted"] = [r for r in ho["long_extra"] if r.get("rv")] + _rows(root, "s5_revisit_ho16")
    blocks, arms = D.assemble_heldout(ho, tables, U, held)
    names, ids, stat = blocks[0]
    assert set(names) == {"H14-1", "H14-2", "H11-2L", "H15-1", "H15-2"}
    v = stat(ids)
    k = {a: {g: S.arm_reach(arms[a][g], ids[g])["k50"] for g in held} for a in ("restart", "r2", "long", "one")}
    assert v["values"]["H14-1"]["primary"] == pytest.approx(np.mean([k["restart"][g] - k["r2"][g] for g in held]))
    assert v["values"]["H14-2"]["primary"] == pytest.approx(np.mean([k["r2"][g] - S.NULL_16 for g in held]))
    assert v["values"]["H11-2L"]["primary"] == pytest.approx({g: k["restart"][g] - k["long"][g] for g in held})
    # H15-1 against the library path: the restart twins from `stage5.restart_twin_shares` on the union rows
    G = []
    for g in held:
        rows55 = arms["rows"][g]
        tw = []
        for R in (32, 1):
            ts = [dict(level=rows55[r["target_id"]][1], n_prime=r["hist"]["10000"]["n_nohit"],
                       hist={int(K): h for K, h in r["hist"]["10000"]["nohit"].items()})
                  for r in ho["union"] if r["set"] == g]
            sh = stage5.restart_twin_shares(ts, R, U[g])["shares"]
            t_ids = sorted(rows55)
            tw.append(S.reach([rows55[t][0] for t in t_ids], [sh[rows55[t][1]] for t in t_ids])["k50"])
        G.append((k["restart"][g] - tw[0]) - (k["one"][g] - tw[1]))
    assert v["values"]["H15-1"]["primary"] == pytest.approx(np.mean(G))
    # the one-run arm of H11-2L joins s2_ho_a16 and s5_long_extra: every restart target exactly once
    assert set(arms["long"]["H-A"]) == set(arms["restart"]["H-A"])


def _ho_arms(k, slope=1.2):
    """In-memory held-out arms whose every K50 is set exactly: k[g] = dict(restart, long, one, twin32, twin1, long)."""
    sizes = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
    U = {K: 100 for K in range(1, 8)}
    arms = dict(restart={}, long={}, one={}, rows={}, twin={}, long_cov={})
    for g, v in k.items():
        rows = {f"t{i}": (b, i + 1) for i, b in enumerate(sizes)}
        arms["rows"][g] = rows
        for a in ("restart", "long", "one"):
            arms[a][g] = {t: (b, _logit(slope * (v[a] - b)), 64) for t, (b, _K) in rows.items()}
        vec = {R: np.array([_logit(slope * (v[name] - sizes[K - 1])) if K <= len(sizes) else 0.0 for K in sorted(U)])
               for R, name in ((32, "twin32"), (1, "twin1"))}
        arms["twin"][g] = {t: (K, vec[32], vec[1]) for t, (_b, K) in rows.items()}
        lv = [0.0] + [100 * _logit(slope * (v["twin_long"] - sizes[K - 1])) if K <= len(sizes) else 0.0
                      for K in sorted(U)]
        arms["long_cov"][g] = {t: [dict(target_id=t, K=K, bits_sem=b, y={S.E16: 0},
                                        rv=dict(prefixes={str(S.E16): dict(lv=lv)}))] for t, (b, K) in rows.items()}
    return arms, {g: U for g in k}


def test_h15_2_is_a_ratio_of_means_and_drops_nonpositive_denominators():
    k = {"A": dict(restart=12.0, long=10.0, one=9.0, twin32=9.0, twin1=8.0, twin_long=8.5),
         "B": dict(restart=11.0, long=7.0, one=8.0, twin32=11.0, twin1=8.5, twin_long=8.0)}
    arms, U = _ho_arms(k)
    ids = {g: sorted(arms["rows"][g]) for g in k}
    v = S.heldout(arms, sorted(k), U)(ids)
    assert v["per_set"]["A"]["twin_long"]["k50"] == pytest.approx(8.5, abs=1e-6)
    mean_of_ratios = np.mean([(9.0 - 8.5) / 2.0, (11.0 - 8.0) / 4.0])
    ratio_of_means = np.mean([0.5, 3.0]) / np.mean([2.0, 4.0])
    assert v["values"]["H15-2"]["primary"] == pytest.approx(ratio_of_means, abs=1e-6)
    assert ratio_of_means != pytest.approx(mean_of_ratios)                          # the plant differs
    g15 = np.mean([(12.0 - 9.0) - (9.0 - 8.0), (11.0 - 11.0) - (8.0 - 8.5)])
    assert v["values"]["H15-1"]["primary"] == pytest.approx(g15, abs=1e-6)
    # the mean GP gain at or below 0: not kept (NaN), and the test counts kept replicates only
    k["B"]["long"] = 15.0                       # bracketed at the top (13): gains 2 and -2, a mean of 0 up to float noise
    arms, U = _ho_arms(k)
    assert math.isnan(S.heldout(arms, sorted(k), U)(ids)["values"]["H15-2"]["primary"])
    k["A"]["long"] = 13.0                                                           # gains -1 and -2
    arms, U = _ho_arms(k)
    assert math.isnan(S.heldout(arms, sorted(k), U)(ids)["values"]["H15-2"]["primary"])
    t = S.decide("H15-2", 0.8, [0.7] * 90 + [float("nan")] * 10, 100)
    assert t["kept"] == 90 and t["estimable"] and t["p"] == pytest.approx(1 / 91)
    assert not S.decide("H15-2", 0.8, [0.7] * 85 + [float("nan")] * 15, 100)["estimable"]


def test_long_run_twin_is_stage3s_never_hit_variant():
    rows = {f"t{i}": (8.0 + i, i + 1) for i in range(6)}
    U = {K: 100 for K in range(1, 8)}
    cov = {}
    for i, (t, (b, K)) in enumerate(rows.items()):
        hit = i % 2 == 0                                    # runs that hit saw more of the space before stopping
        frac = [0.0] + [(0.9 if hit else 0.4) * 100 * max(0.0, 1 - 0.15 * k) for k in range(1, 8)]
        rv = dict(prefixes={str(p): dict(lv=[x * p / S.E16 for x in frac]) for p in
                            (5_000, 10_000, 20_000, 40_000, 80_000, 160_000, 320_000) if not hit or p <= 40_000})
        cov[t] = [dict(target_id=t, K=K, bits_sem=b, y={S.E16: int(hit)}, rv=rv)]
    ids = sorted(rows)
    probs = S.coverage_probs(cov, ids, U, S.E16, "never_hit")
    want = h6d.twin_probs(h6d.cumulative([r for t in ids for r in cov[t]], S.E16, "lv", "never_hit"), U)
    assert probs == want
    strat = S.coverage_probs(cov, ids, U, S.E16, "stratified")
    assert strat != pytest.approx(want)                                             # the plant differs
    arms = dict(restart={"A": {t: (b, 0.5, 64) for t, (b, _K) in rows.items()}}, rows={"A": rows},
                long_cov={"A": cov})
    v = S.heldout(arms, ["A"], {"A": U})({"A": ids})
    assert v["per_set"]["A"]["twin_long"]["k50"] == pytest.approx(
        S.reach([rows[t][0] for t in ids], [want[rows[t][1]] for t in ids])["k50"])


def test_h16_statistics(synth):
    root, cfg, levels, _ = synth
    sets = list(cfg["robust_real"]) + list(cfg["robust_bool"])
    tables, _ = _tables(root, sets)
    U = D.levels(root, sets, levels)
    h16 = dict(revisit_robust=_rows(root, "s5_revisit_robust"), stage3_a16=_rows(root, "s2_revisit_a16"))
    (names, ids, stat), = D.assemble_h16(h16, tables, U, cfg["robust_real"], cfg["robust_bool"])
    v = stat(ids)
    by_pop = {25: [r for r in h16["revisit_robust"] if r["config"] == "R1"],
              1000: [r for r in h16["revisit_robust"] if r["config"] == "R2"]}
    keys = {(r["set"], r["target_id"], r["seed"]) for r in by_pop[25]}
    by_pop[100] = [r for r in h16["stage3_a16"] if (r["set"], r["target_id"], r["seed"]) in keys]
    reach = {}
    for p, rr in by_pop.items():
        for g in sets:
            runs = h6d.runs_from_rows([r for r in rr if r["set"] == g], (S.E0,))
            reach[p, g] = h6d.k50_interp(h6d.twin_probs(h6d.cumulative(runs, S.E0, "lv", "stratified"), U[g]), U[g])
    assert len({round(reach[p, sets[0]], 9) for p in (25, 100, 1000)}) == 3            # the twin moves with population
    r_, b_ = cfg["robust_real"], cfg["robust_bool"]
    for p in (25, 1000):
        want = np.mean([reach[p, g] for g in b_]) - np.mean([reach[p, g] for g in r_])
        assert v["values"]["H16-1"]["primary"][str(p)] == pytest.approx(want)
    means = [np.mean([reach[p, g] for g in sets]) for p in S.POPS]
    assert v["values"]["H16-2"]["primary"] == pytest.approx(np.polyfit(np.log2(S.POPS), means, 1)[0])
    # H16-2 is a TOST of the slope around 1
    t = S.decide("H16-2", 1.1, [1.0, 1.05, 1.1, 1.15, 1.2] * 20, 100)
    assert t["center"] == 1.0 and t["estimate"] == pytest.approx(0.1) and t["equivalent"]
    assert set(v["values"]["H16-2"]) == {"primary", "logistic", "iso"}


# ---------------------------------------------------------------- inference and output (REQ-ANA-128)
def _fake(p, p_iso=None):
    return dict(tests=dict(primary=dict(p=p, estimable=True), iso=dict(p=p if p_iso is None else p_iso,
                                                                       estimable=True)))


def test_families_holm_and_stage_wide_holm():
    res = {n: _fake(0.001) for n in S.TESTS}
    res["H12-1"], res["H12-2"], res["H12-3"] = _fake(0.010), _fake(0.020), _fake(0.040)
    res["H14-2"] = _fake(0.04, p_iso=0.06)                                          # the isotonic reach differs
    d = S.families(res)
    fam = d["families"]["H12"]["primary"]
    assert [round(fam[n]["threshold"], 4) for n in ("H12-1", "H12-2", "H12-3")] == [0.0167, 0.025, 0.05]
    assert all(fam[n]["supported"] for n in fam)
    assert d["families"]["H14"]["primary"]["H14-2"]["supported"] and d["flagged"] == ["H14-2"]
    assert d["families"]["H11-2L"]["primary"]["H11-2L"]["threshold"] == 0.05
    sw = d["stage_wide"]
    assert len(sw) == 12 and not sw["H12-3"]["supported"] and sw["H12-1"]["supported"]   # 0.040 > 0.05 / 2
    # a family's test that is not estimable counts as p = 1 and does not block the other (row 155)
    res["H13-1"] = dict(tests=dict(primary=dict(p=0.0001, estimable=False), iso=dict(p=0.0001, estimable=False)))
    d = S.families(res)
    assert not d["families"]["H13"]["primary"]["H13-1"]["supported"]
    assert d["families"]["H13"]["primary"]["H13-2"]["supported"]
    del res["H16-1"]
    assert S.families(res)["stage_wide"] is None                                    # not until all twelve exist


def test_set_level_intervals():
    from scipy.stats import t as tdist
    a, b = [1.0, 2.0, 4.0], [0.5, 0.7]
    w = S.welch(a, b)
    va, vb = np.var(a, ddof=1) / 3, np.var(b, ddof=1) / 2
    df = (va + vb) ** 2 / (va ** 2 / 2 + vb ** 2 / 1)
    q = tdist.ppf(0.95, df)
    assert w["mean"] == pytest.approx(np.mean(a) - np.mean(b))
    assert w["lo"] == pytest.approx(w["mean"] - q * math.sqrt(va + vb))
    t = S.set_level(dict(t={"x": 1.0, "y": 3.0}))
    assert t["mean"] == 2.0 and t["lo"] == stage5.set_level_interval([1.0, 3.0])["lo"]
    r = S.set_level(dict(ratio_of_means=dict(numerator={"x": 1.0, "y": 2.0}, denominator={"x": 3.0, "y": 5.0})))
    assert r["numerator"]["mean"] == 1.5 and r["denominator"]["mean"] == 4.0
    m = S.set_level(dict(members={"25": dict(welch=({"a": 1.0, "b": 2.0}, {"c": 0.0, "d": 1.0}))}))
    assert m["25"]["mean"] == pytest.approx(1.0)


def test_instrument_checks_reproduce_the_registered_values():
    from reach.analysis.s2.load import read_phase
    from reach.paths import ROOT
    cfg = D.CONF
    tables, _ = _tables(ROOT, list(cfg["real"]) + list(cfg["boolean"]))
    base = read_phase("s2_cal_a0", ROOT) + read_phase("s2_cal_a16", ROOT)
    assert len(D._instrument_h6c(ROOT, base, tables, cfg)) == 7
    assert len(D._instrument_bool(ROOT, base, tables, cfg)) == 7
    stage3 = read_phase("s2_revisit_a0", ROOT) + read_phase("s2_revisit_a16", ROOT)
    U = D.levels(ROOT, cfg["real"])
    assert len(D._instrument_twin(ROOT, stage3, U, cfg)) == 7
    # Stage 4's restart arm on its 55 levels (s4_restart is committed gzipped; checked against stage4_tasks.json)
    rows = D.phase_rows(ROOT, "s4_restart")
    ho, _ = _tables(ROOT, cfg["heldout"])
    sizes = {g: {t: v["units"]["bits_sem"] for t, v in ho[g].items() if D.proven(v)} for g in cfg["heldout"]}
    by = D._by(rows, "set", "target_id")
    arm = {g: S.restart_arm({t: rr for (h, t), rr in by.items() if h == g}, sizes[g], 32, 100) for g in cfg["heldout"]}
    assert len(D._instrument_restart(ROOT, arm)) == 6 and sum(len(a) for a in arm.values()) == 1650


def test_instrument_checks_fire_on_a_planted_change(tmp_path):
    want = {"A": 1.0, "B": 2.0}
    assert D.close_or_refuse("x", {"A": 1.0, "B": 2.0 + 1e-12}, want) == {"A": 1.0, "B": 2.0 + 1e-12}
    for got in ({"A": 1.0, "B": 2.001}, {"A": 1.0}, {"A": 1.0, "B": float("nan")}):
        with pytest.raises(RuntimeError, match="instrument check"):
            D.close_or_refuse("x", got, want)
    (tmp_path / "data/analysis").mkdir(parents=True)                                # a changed deposited value
    arm = {"H-A": _exact(SIZES, 9.0)}
    (tmp_path / "data/analysis/h11.json").write_text(json.dumps(dict(h11=dict(per_set={"H-A": dict(k50=9.0)}))))
    assert D._instrument_restart(tmp_path, arm)["H-A"] == pytest.approx(9.0, abs=1e-6)
    (tmp_path / "data/analysis/h11.json").write_text(json.dumps(dict(h11=dict(per_set={"H-A": dict(k50=9.1)}))))
    with pytest.raises(RuntimeError, match="instrument check"):
        D._instrument_restart(tmp_path, arm)
    with pytest.raises(RuntimeError, match="missing"):
        D._instrument_h6c(tmp_path / "nowhere", [], {}, D.CONF)


def test_end_to_end_on_a_synthetic_root(synth, tmp_path):
    root, cfg, levels = _copy(synth, tmp_path)
    res = D.build(root, cfg, boot=6, levels_given=levels)
    assert set(res["tests"]) == set(S.TESTS)
    assert res["decisions"]["stage_wide"] is not None and set(res["decisions"]["families"]) == set(S.FAMILIES)
    assert res["checks"]["reexec"]["s5_revisit_restart"]["identical"] == len(_rows(root, "s4_restart"))
    assert res["extra"]["wide_excluded"] == ["R-X_K09_00"]
    assert set(res["extra"]["long_twin_sensitivity"]["H-A"]) == {"never_hit", "stratified", "never_hit_by_level"}
    for n, v in res["tests"].items():
        assert len(v["replicates"]["primary"]) == 6 and {"primary", "iso"} <= set(v["tests"])
    D.write(tmp_path / "out", res)
    back = json.loads((tmp_path / "out/data/analysis/stage5.json").read_text())
    assert set(back["tests"]) == set(S.TESTS)
    md = (tmp_path / "out/docs/status/stage5_report.md").read_text()
    assert all(f"| {n} |" in md for n in S.TESTS) and len(md.splitlines()) <= 200
    # the same build twice gives the same tests (seeded bootstrap)
    again = D.build(root, cfg, boot=6, levels_given=levels)
    assert {n: v["tests"]["primary"]["p"] for n, v in again["tests"].items()} == \
        {n: v["tests"]["primary"]["p"] for n, v in res["tests"].items()}
    assert res["extra"]["frozen"] == "not planned" and "FB" not in res["detail"]
    # s5_frozen (descriptive): omitted while incomplete, read when complete, refused with a row of no planned task
    base = _rows(root, "s2_cal_a0") + _rows(root, "s2_cal_a16")
    fb = [dict(r, key=f"s5_frozen|{r['key']}", config="FB", pair_key=r["key"]) for r in base if r["set"] in cfg["real"]]
    tasks = Y.write_phase(root, "s5_frozen", fb)
    Y.write_phase(root, "s5_frozen", fb[1:], tasks)
    r12 = D.build(root, cfg, boot=3, families=["H12"], levels_given=levels)
    assert r12["extra"]["frozen"].startswith("incomplete") and "FB" not in r12["detail"]
    Y.write_phase(root, "s5_frozen", fb, tasks)
    r12 = D.build(root, cfg, boot=3, families=["H12"], levels_given=levels)
    assert r12["extra"]["frozen"] == "complete" and r12["detail"]["FB"]["values"]["FB"]["primary"] == pytest.approx(0.0)
    Y.write_phase(root, "s5_frozen", fb + [dict(fb[0], key="extra")], tasks)
    with pytest.raises(RuntimeError, match="no planned task"):
        D.build(root, cfg, boot=3, families=["H12"], levels_given=levels)
    (root / "data/runs/s5_frozen/manifest.json").unlink()
    # a missing union row stops H15, not H14
    p = root / "data/runs/s5_union/results.jsonl"
    p.write_text("".join(p.read_text().splitlines(keepends=True)[1:]))
    with pytest.raises(RuntimeError, match="s5_union"):
        D.build(root, cfg, boot=3, families=["H15"], levels_given=levels)
    assert set(D.build(root, cfg, boot=3, families=["H14"], levels_given=levels)["tests"]) == {"H14-1", "H14-2"}

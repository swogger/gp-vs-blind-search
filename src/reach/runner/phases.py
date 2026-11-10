"""Phase definitions. Each builder returns the task list that `plan` freezes into manifest.json.

A task is {key, argv, cwd, cpu_limit, weight, meta}: argv may use "{python}" for the queue's
interpreter, cwd is relative to the repository root, cpu_limit is in CPU seconds, and weight is
the relative expected cost (used only for the ETA).
"""
import importlib.util
import json
import math
from collections import defaultdict

from reach import regression
from reach.runner.queue import ROOT

PHASES = {}


def phase(name, description, verify=None):
    def deco(fn):
        PHASES[name] = dict(build=fn, description=description, verify=verify)
        return fn
    return deco


def legacy_seeds():
    spec = importlib.util.spec_from_file_location("legacy_seeds", ROOT / "legacy" / "research" / "seeds.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SEEDS


@phase("smoke", "Eight tiny self-test tasks covering every terminal state (about 10 s).")
def smoke():
    tasks = [dict(key=f"ok_{i}", argv=["{python}", "-m", "reach.runner.selftest_task", "--mode", "ok"],
                  cpu_limit=10) for i in range(5)]
    tasks += [dict(key="timeout", argv=["{python}", "-m", "reach.runner.selftest_task", "--mode", "timeout"],
                   cpu_limit=2, weight=10),
              dict(key="error", argv=["{python}", "-m", "reach.runner.selftest_task", "--mode", "error"],
                   cpu_limit=10),
              dict(key="noresult", argv=["{python}", "-m", "reach.runner.selftest_task", "--mode", "noresult"],
                   cpu_limit=10)]
    return tasks


def _verify_against_p1(ph):
    rows = ph.results()
    bad = regression.compare(rows, regression.load_p1())
    print(f"[{ph.name}] {len(rows)} runs compared with P1_synth_main.csv on {', '.join(regression.FIELDS)}")
    if not rows:
        print("no results yet")
        return 1
    for key, f, ours, ref in bad[:30]:
        print(f"  MISMATCH {key} {f}: ours {ours!r} vs stored {ref!r}")
    print("all identical" if not bad else f"{len(bad)} mismatches")
    return 0 if not bad else 1


@phase("legacy_check",
       "Legacy EML engine on every K = 7 and K = 13 target, both regimes, first P1 seed (122 runs, "
       "about 25 CPU-minutes). Checks the queue end to end against stored P1 rows.",
       verify=_verify_against_p1)
def legacy_check():
    seed = legacy_seeds()[1]
    targets = json.loads((ROOT / "legacy/results/synthetic_dense_k/targets.json").read_text())
    tasks = []
    for t in targets:
        if t["K_construction"] not in (7, 13):
            continue
        for reg in ("A", "B"):
            tasks.append(dict(
                key=f"{t['target_id']}|{reg}|{seed}", cwd="legacy",
                argv=["{python}", "-u", "research/rerun/synth_task.py",
                      "--targets", "results/synthetic_dense_k/targets.json",
                      "--target_id", t["target_id"], "--regime", reg, "--seed", str(seed),
                      "--generations", "200", "--timeout", "600", "--no_size_penalty", "--cpu_clock"],
                cpu_limit=700, weight=4.0 if reg == "A" else 1.0,
                meta=dict(target_id=t["target_id"], regime=reg, seed=seed)))
    return tasks


@phase("m1_regression",
       "M1 acceptance: the new engine in legacy mode on every K = 7 and K = 13 EML target, both "
       "regimes, all 10 P1 seeds (1,220 runs, about 4 CPU-hours). Must match P1 exactly.",
       verify=_verify_against_p1)
def m1_regression():
    seeds = legacy_seeds()[1:11]
    targets = json.loads((ROOT / "legacy/results/synthetic_dense_k/targets.json").read_text())
    tasks = []
    for t in targets:
        if t["K_construction"] not in (7, 13):
            continue
        for reg in ("A", "B"):
            for s in seeds:
                tasks.append(dict(
                    key=f"{t['target_id']}|{reg}|{s}",
                    argv=["{python}", "-m", "reach.tasks.treegp_task", "--grammar", "EML",
                          "--targets", "legacy/results/synthetic_dense_k/targets.json",
                          "--target_id", t["target_id"], "--regime", reg, "--seed", str(s),
                          "--generations", "200", "--timeout", "600", "--legacy"],
                    cpu_limit=700, weight=4.0 if reg == "A" else 1.0,
                    meta=dict(target_id=t["target_id"], regime=reg, seed=s)))
    return tasks


# ---------------------------------------------------------------- Study 2, Phase 0 (docs/plan/phases/phase0_restore.md)
# Fresh phase names for the two regressions, so the copied paper-1 records stay untouched (REQ-RULE-07).
phase("s2_legacy_check", "Study 2 Phase 0: legacy_check rerun in this repository (122 runs, ~30 CPU-minutes). "
      "Must equal P1_synth_main.csv.", verify=_verify_against_p1)(legacy_check)
phase("s2_legacy_check_t19", "Engine change check (T19 configuration options, 2026-10-20): legacy_check again; must "
      "equal P1_synth_main.csv.", verify=_verify_against_p1)(legacy_check)
phase("s2_m1_regression", "Study 2 Phase 0: m1_regression rerun in this repository (1,220 runs, ~5 CPU-hours; "
      "launched by the author). Must equal P1_synth_main.csv.", verify=_verify_against_p1)(m1_regression)



# ---------------------------------------------------------------- Study 2, Phase 1 (docs/plan/phases/phase1_enumeration.md)
S2_REAL = ["F-EML", "R-ARITH", "R-ARITH4", "R-ELEM6", "R-ELEM4", "R-HYB", "R-TRIG", "F-LDE0"]
S2_BOOL = [f"BOOL-{b}-n5" for b in ("NAND", "NOR", "AON", "AN", "AONX", "AX1", "RICH", "IMP0")]
S2_ENUM = "data/enumeration_s2"
MIN_FREE_GB = 2                    # free-disk floor for enumerations (author, 2026-10-22; was 15)


def _s2_enum(g, root, kmax, max_gb, cpu, extra=()):
    """Clean stop (--cpu_budget) 10 minutes before the queue's hard CPU limit; free-disk floor MIN_FREE_GB (author,
    2026-10-22: 2 GB; a disk stop fails the task and is retried, row 70)."""
    return dict(key=g, argv=["{python}", "-m", "reach.tasks.enumerate_task", "--grammar", g, "--root", root,
                             "--kmax", str(kmax), "--max_gb", str(max_gb), "--cpu_budget", str(cpu - 600),
                             "--min_free_gb", str(MIN_FREE_GB), *extra],
                cpu_limit=cpu, meta=dict(grammar=g, root=root))


@phase("s2_enum_real", "Study 2: enumerate the 8 real sets (guarded semantics, depth-aware) until the 8 GB memory "
       "cap or 4 CPU-hours. Run with --workers 2 (peak about 12 GB each).")
def s2_enum_real():
    return [_s2_enum(g, S2_ENUM, 40, 8, 4 * 3600) for g in S2_REAL]


@phase("s2_enum_ieee", "Study 2 (C4 comparison): ELEM6, HYB and EML under plain IEEE semantics, depth-aware, same caps. "
       "--workers 2.")
def s2_enum_ieee():
    return [_s2_enum(g, f"{S2_ENUM}/ieee", 40, 8, 4 * 3600) for g in ("ELEM6", "HYB", "EML")]


@phase("s2_enum_bool", "Study 2: enumerate the 8 Boolean sets (n = 5) until the 10 GB memory cap or 6 CPU-hours "
       "(clean stop). --workers 2. Launched by the author (long).")
def s2_enum_bool():
    return [_s2_enum(g, S2_ENUM, 60, 10, 6 * 3600) for g in S2_BOOL]


S2_EXTRA_REAL = ["F-EML-R", "R-SRB"]                       # D052
S2_EXTRA_BOOL = ["BOOL-IMP0R-n5", "BOOL-NAND-n6", "BOOL-AON-n6"]   # AON-n6: D053


@phase("s2_enum_extra", "Study 2 (D052): enumerate the added sets: F-EML-R and R-SRB (8 GB, 4 h) and IMP0-R (n = 5) and "
       "NAND and AON (n = 6) (10 GB, 6 h). --workers 2; launch only after s2_enum_bool (memory).")
def s2_enum_extra():
    return ([_s2_enum(g, S2_ENUM, 40, 8, 4 * 3600) for g in S2_EXTRA_REAL] +
            [_s2_enum(g, S2_ENUM, 60, 10, 6 * 3600) for g in S2_EXTRA_BOOL])


@phase("s2_enum_ceiling", "Study 2 (D053): raise the low ceilings of R-ELEM6 and R-TRIG with a 16 GB cap (fresh "
       "enumerations under data/enumeration_s2/cap16; they supersede the 8 GB ones). --workers 1 (peak about 20 GB); "
       "never alongside another enumeration.")
def s2_enum_ceiling():
    from reach.s2_sets import SETS
    return [_s2_enum(g, SETS[g]["root"], 40, 16, 4 * 3600) for g in ("R-ELEM6", "R-TRIG")]


@phase("s2_enum_random", "Study 2 (T26, D064): enumerate the random held-out candidates of data/registration/"
       "random_draw.json in draw order, alternating domains (first 6 per domain), with the listed sets' caps (real 8 GB "
       "and 4 CPU-h, Boolean 10 GB and 6 CPU-h). Not a search. --workers 1 next to GP runs (memory).")
def s2_enum_random():
    rec = json.loads((ROOT / "data/registration/random_draw.json").read_text())
    real = [_s2_enum(c["name"], S2_ENUM + "/random", 40, 8, 4 * 3600) for c in rec["real"]["candidates"][:6]]
    boo = [_s2_enum(c["name"], S2_ENUM + "/random", 60, 10, 6 * 3600) for c in rec["bool"]["candidates"][:6]]
    return [t for pair in zip(real, boo) for t in pair]


@phase("s2_enum_random_b", "Study 2 (D067): rerun the random-set enumerations of s2_enum_random that ended for disk "
       "(not a ceiling: swap from a memory spike filled the disk) or were killed from outside. Real 8 GB as before; "
       "Boolean 6 GB, because the 10 GB estimate let one Boolean set outgrow the 32 GB machine. --workers 1.")
def s2_enum_random_b():
    from reach.runner.queue import read_results
    rec = json.loads((ROOT / "data/registration/random_draw.json").read_text())
    first = read_results(ROOT / "data/runs/s2_enum_random/results.jsonl")
    bad = {k for k, r in first.items() if r.get("terminal_state") == "error" or r.get("stop_reason") == "disk"}
    real = [_s2_enum(c["name"], S2_ENUM + "/random", 40, 8, 4 * 3600) for c in rec["real"]["candidates"]
            if c["name"] in bad]
    boo = [_s2_enum(c["name"], S2_ENUM + "/random", 60, 6, 6 * 3600) for c in rec["bool"]["candidates"]
           if c["name"] in bad]
    return real + boo


AUDIT_PHASES = ("s2_cal_a0", "s2_cal_a16", "s2_cal_alloc")


@phase("s2_audit_repro", "QA (D068, not a test of any hypothesis): rerun 20 randomly chosen completed runs from each "
       "of s2_cal_a0, s2_cal_a16 and s2_cal_alloc, same commands and seeds, on the current code; "
       "reach.analysis.s2.audit compares the records field by field. --workers 1.")
def s2_audit_repro():
    from reach.analysis.s2.audit import sample_keys
    from reach.runner.queue import read_results
    tasks = []
    for ph in AUDIT_PHASES:
        man = {t["key"]: t for t in json.loads((ROOT / f"data/runs/{ph}/manifest.json").read_text())["tasks"]}
        rows = read_results(ROOT / f"data/runs/{ph}/results.jsonl")
        for k in sample_keys(rows, 20):
            t = json.loads(json.dumps(man[k]))
            t["key"] = f"audit|{k}"
            t["meta"] = dict(t.get("meta", {}), audit_of=ph)
            tasks.append(t)
    return tasks


BIGTREE_PHASES = ("s2_cal_a0", "s2_cal_a16", "s2_cal_alloc", "s2_cal_mech", "s2_cal_robust", "s2_cal_pool")


def bigtree_tasks(root=ROOT, phases=BIGTREE_PHASES, sample=None):
    """D091 (REQ-RUN-32): one task per exact run whose record has no expression (best tree over 200 nodes), the
    original manifest task unchanged except for its key, `--print_tree` and the meta naming the row it reproduces.
    sample=n: only the n cheapest per phase (original cpu_s) and the cheapest real-valued run of all phases."""
    from reach.runner.queue import read_results
    chosen = []
    for ph in phases:
        man = {t["key"]: t for t in json.loads((root / f"data/runs/{ph}/manifest.json").read_text())["tasks"]}
        rows = read_results(root / f"data/runs/{ph}/results.jsonl")
        chosen += [(ph, man[k], r) for k, r in sorted(rows.items()) if r.get("exact") is True and not r.get("best_expr")]
    if sample is not None:
        keep = set()
        for ph in phases:
            mine = sorted((r["cpu_s"], r["key"]) for p, _, r in chosen if p == ph)
            keep |= {(ph, k) for _, k in mine[:sample]}
        real = sorted((r["cpu_s"], ph, r["key"]) for ph, _, r in chosen if r.get("domain") == "real")
        keep |= {(ph, k) for _, ph, k in real[:1]}
        chosen = [c for c in chosen if (c[0], c[2]["key"]) in keep]
    tasks = []
    for ph, orig, r in chosen:
        if "reach.tasks.s2_run" not in orig["argv"]:
            raise ValueError(f"{ph} {r['key']}: not a tree-GP run task (reach.tasks.s2_run)")
        t = json.loads(json.dumps(orig))
        t["key"] = f"bigtree|{r['key']}"
        t["argv"] = t["argv"] + ["--print_tree"]
        t["meta"] = dict(t.get("meta", {}), audit_of=ph, audit_key=r["key"])
        tasks.append(t)
    return tasks


def _verify_bigtrees(ph):
    from reach.analysis.s2.audit import bigtree_report, print_bigtree_report
    rep = bigtree_report(ROOT, ph.name)
    print_bigtree_report(rep)
    return 0 if rep["ok"] else 1


@phase("s2_audit_bigtrees", "QA (D068, D091; not a test of any hypothesis): rerun every exact calibration run whose "
       "best tree had over 200 nodes (no expression recorded; 3,480 runs), same commands and seeds plus --print_tree; "
       "reach.analysis.s2.audit compares each record with its original and label-checks the full trees.",
       verify=_verify_bigtrees)
def s2_audit_bigtrees():
    return bigtree_tasks()


@phase("s2_audit_bigtrees_sample", "Short check of s2_audit_bigtrees (D091): the 3 cheapest of its runs per phase and "
       "the cheapest real-valued one.", verify=_verify_bigtrees)
def s2_audit_bigtrees_sample():
    return bigtree_tasks(sample=3)


# ---------------------------------------------------------------- Phase 7b: H6d re-runs (D098; REQ-RUN-39)
REVISIT_CPU_FACTOR = 2          # a re-run's hard CPU limit: the original's times this (room for the counter's time)
REVISIT = {"s2_revisit_a0": ("s2_cal_a0",), "s2_revisit_a16": ("s2_cal_a16",),
           "s2_revisit_pilot": ("s2_pilot_reach_a", "s2_pilot_reach_b")}


def revisit_tasks(sources, root=ROOT, resolve=False):
    """One task per task of each source manifest, unchanged except for its key (revisit|KEY), `--count_distinct
    --enum_dir DIR`, the hard CPU limit times REVISIT_CPU_FACTOR and meta `audit_of`, `audit_key`. DIR is the set's
    enumeration in this checkout (so a registered list hashes alike in every worktree) or, with resolve, the main
    worktree's when this checkout lacks the untracked level files (the pilot re-run from the dev worktree)."""
    from reach import revisit
    from reach.analysis.s2.phase1 import enum_dir
    tasks = []
    for src in sources:
        for o in json.loads((root / f"data/runs/{src}/manifest.json").read_text())["tasks"]:
            if "reach.tasks.s2_run" not in o["argv"]:
                raise ValueError(f"{src} {o['key']}: not a tree-GP run task (reach.tasks.s2_run)")
            t = json.loads(json.dumps(o))
            d = revisit.enum_root(o["meta"]["set"]) if resolve else enum_dir(o["meta"]["set"])
            t["key"] = f"revisit|{o['key']}"
            t["argv"] = o["argv"] + ["--count_distinct", "--enum_dir", str(d)]
            t["cpu_limit"] = o["cpu_limit"] * REVISIT_CPU_FACTOR
            t["meta"] = dict(o.get("meta", {}), audit_of=src, audit_key=o["key"])
            tasks.append(t)
    return tasks


def _spread(tasks):
    """Order so that the first tasks cover every (set, level) once, then a second task of each, and so on."""
    import collections
    seen, ranked = collections.Counter(), []
    for t in sorted(tasks, key=lambda t: (t["meta"]["set"], t["meta"]["K"], t["meta"]["target_id"], t["meta"]["seed"])):
        cell = (t["meta"]["set"], t["meta"]["K"])
        ranked.append(((seen[cell], cell, t["key"]), t))
        seen[cell] += 1
    return [t for _, t in sorted(ranked, key=lambda x: x[0])]


def _verify_revisit(ph):
    from reach.analysis.s2.audit import print_revisit_report, revisit_report
    rep = revisit_report(ROOT, ph.name)
    print_revisit_report(rep)
    return 0 if rep["ok"] else 1


@phase("s2_revisit_a0", "STAGE 3 (locked until Stage 3; H6d, D098): every s2_cal_a0 task again, same command and seed "
       "plus --count_distinct (distinct functions per budget prefix); every row must reproduce its original.",
       verify=_verify_revisit)
def s2_revisit_a0():
    from reach.s2_lock import check_stage3
    check_stage3()
    return revisit_tasks(REVISIT["s2_revisit_a0"])


@phase("s2_revisit_a16", "STAGE 3 (locked until Stage 3; H6d, D098): every s2_cal_a16 task again, same command and "
       "seed plus --count_distinct; every row must reproduce its original.", verify=_verify_revisit)
def s2_revisit_a16():
    from reach.s2_lock import check_stage3
    check_stage3()
    return revisit_tasks(REVISIT["s2_revisit_a16"])


@phase("s2_revisit_rand", "H6d negative control (D098; a check of the instrument, not a test): trees drawn as the "
       "random-sampling baseline draws them, counted by the H6d counter: every full set x 25 streams of E0 draws.")
def s2_revisit_rand():
    from reach import revisit
    return [dict(key=f"rand|{g}|{i}", argv=["{python}", "-m", "reach.tasks.s2_revisit_rand", "--set", g, "--enum_dir",
                                            str(revisit.enum_root(g)), "--seed", str(30_000_000 + 1_000 * j + i),
                                            "--n", "20000"],
                 cpu_limit=1800, meta=dict(set=g, stream=i)) for j, g in enumerate(FULL_SETS) for i in range(25)]


@phase("s2_revisit_pilot2", "Pilot for H6d with the version 2 counter (D098; never confirmatory): as s2_revisit_pilot, "
       "with lv_ok and target_first; sets and checks the margins of H6d before the Stage 3 registration.",
       verify=_verify_revisit)
def s2_revisit_pilot2():
    return _spread(revisit_tasks(REVISIT["s2_revisit_pilot"], resolve=True))


@phase("s2_revisit_pilot", "Pilot for H6d (D098; never confirmatory): every s2_pilot_reach_a and _b task again plus "
       "--count_distinct, ordered to cover every set and level first; checks reproduction and overhead and sets the "
       "margins of H6d before the Stage 3 registration.", verify=_verify_revisit)
def s2_revisit_pilot():
    return _spread(revisit_tasks(REVISIT["s2_revisit_pilot"], resolve=True))


def heldout_sets():
    """The 4 listed held-out sets (Stage 0 draw) and the 8 random ones (first 4 per domain, ceiling >= 10)."""
    from reach.tasks.s2_random_draw import heldout_from_files
    rec = json.loads((ROOT / "data/registration/heldout_draw.json").read_text())
    rnd = heldout_from_files()
    return [g for d in ("real", "bool") for g in rec[d]["held_out"] + rnd[d]]


@phase("s2_targets_bench", "Study 2 (T3, D069): benchmark targets on the 12 held-out sets, where each set's enumeration "
       "proves the benchmark's minimum size. Not a search. Real sets load their values: --workers 1.")
def s2_targets_bench():
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.s2_bench", "--grammar", g], cpu_limit=4 * 3600,
                 meta=dict(grammar=g)) for g in heldout_sets()]


@phase("s2_targets_bench2", "Study 2 (T3, D069, D091): the benchmark files of the 12 held-out sets regenerated through "
       "the queue with the SRBench and EML-GP additions; earlier entries must be reproduced byte for byte. Not a search. "
       "--workers 1.")
def s2_targets_bench2():
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.s2_bench", "--grammar", g, "--regenerate"],
                 cpu_limit=4 * 3600, meta=dict(grammar=g)) for g in heldout_sets()]


# ---------------------------------------------------------------- Study 2, Phase 7 held-out (locked until Stage 2)
PRINT_TREE = "--print_tree"     # s2_run option (REQ-ENG-22); on every Stage 2 tree-GP task (D092, REQ-RUN-33)


def tree_gp_without_full_tree(tasks):
    """Keys of tree-GP tasks (reach.tasks.s2_run) without `--print_tree` (D092: none in a Stage 2 phase)."""
    return [t["key"] for t in tasks if "reach.tasks.s2_run" in t["argv"] and PRINT_TREE not in t["argv"]]


HO_LEVELS = "data/registration/heldout_levels.json"     # written with its .sha256 by reach.analysis.s2.stage2_freeze


def _ho_levels():
    """The held-out levels from the frozen model, checked against their .sha256 (guardrail row 124)."""
    return _checked_json(HO_LEVELS, f"{HO_LEVELS} is missing (written by the freeze step, "
                         "`.venv/bin/python -m reach.analysis.s2.stage2_freeze`, from the frozen model)")


@phase("s2_ho_targets", "Study 2 held-out targets: 30 per level on the levels the frozen model gives each held-out set "
       "(data/registration/heldout_levels.json), seed TARGET_SEEDS['heldout']. Not a search. --workers 1.")
def s2_ho_targets():
    from reach.seeds import TARGET_SEEDS
    lv = _ho_levels()
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.s2_targets", "--grammar", g, "--name", "heldout",
                              "--n", "30", "--seed", str(TARGET_SEEDS["heldout"]),
                              "--levels", ",".join(str(k) for k in sorted(lv[g]["main"] + lv[g]["budget"]))],
                 cpu_limit=6 * 3600, meta=dict(grammar=g)) for g in heldout_sets() if g in lv]


def _ho(block, budget, which):
    from reach.seeds import s2_seeds
    from reach.s2_lock import check_stage2
    check_stage2()
    lv, tasks = _ho_levels(), []
    for g in heldout_sets():
        sizes = set(lv[g]["main"] if which == "main" else lv[g]["budget"])
        for t in json.loads((ROOT / f"data/targets_s2/{g}/heldout.json").read_text()):
            if t["K_construction"] in sizes:
                tasks += [_s2_task(block, g, "heldout", t, "A", "base", s, budget, print_tree=True)
                          for s in s2_seeds("heldout")]
    return tasks


@phase("s2_ho_a0", "HELD-OUT (locked until Stage 2): regime A at E0 on the main levels of the 12 held-out sets.")
def s2_ho_a0():
    return _ho("ho_a0", 1, "main")


@phase("s2_ho_a16", "HELD-OUT (locked until Stage 2): regime A at 16xE0 on the budget levels of the 12 held-out sets.")
def s2_ho_a16():
    return _ho("ho_a16", 16, "budget")


@phase("s2_ho_bench", "HELD-OUT (locked until Stage 2): regime A at E0 on every benchmark cell x 30 benchmark seeds.")
def s2_ho_bench():
    from reach.seeds import s2_seeds
    from reach.s2_lock import check_stage2
    check_stage2()
    tasks = []
    for g in heldout_sets():
        p = ROOT / f"data/targets_s2/{g}/bench.json"
        if p.exists():
            tasks += [_s2_task("ho_bench", g, "bench", t, "A", "base", s, 1, print_tree=True)
                      for t in json.loads(p.read_text()) for s in s2_seeds("benchmark")]
    return tasks


@phase("s2_ho_operon", "HELD-OUT (locked until Stage 2): Operon, both configurations, at 16xE0 on the held-out targets "
       "of the held-out standard-operator sets (H9b, 5 held-out seeds) and on their real benchmark cells (D049; the "
       "30 benchmark seeds, as s2_ho_bench).")
def s2_ho_operon():
    from reach.seeds import s2_seeds
    from reach.s2_lock import check_stage2
    check_stage2()
    tasks = []
    for g in [g for g in heldout_sets() if g in OPERON_SETS]:
        for t in json.loads((ROOT / f"data/targets_s2/{g}/heldout.json").read_text()):
            tasks += [_operon_task("ho_operon", g, "heldout", t, cfg, s, 16)
                      for cfg in ("symbolic", "default") for s in s2_seeds("heldout")]
        p = ROOT / f"data/targets_s2/{g}/bench.json"            # real benchmarks (D049; guardrail row 127)
        if p.exists():
            tasks += [_operon_task("ho_operon", g, "bench", t, cfg, s, 16)
                      for t in json.loads(p.read_text()) for cfg in ("symbolic", "default") for s in s2_seeds("benchmark")]
    return tasks


# ---------------------------------------------------------------- Stage 4: the remedy test (D100; docs/design/remedy.md)
REAL_HELDOUT = ("R-ARITH", "R-ELEM6", "RS-add_sqrt", "RS-eml_sin_sqrt_square", "RS-mul_div_eml_sin_square",
                "RS-mul_eml_sin")
RESTART_GENERATIONS = 100
RESTART_TIMEOUT = 900                     # soft CPU limit (s) of a 100-generation run; hard limit + 100
RESTART_TOP_BITS = math.log2(16 * 20_000) + 2   # the extra-level rule: reach at least 2 bits above 16 x E0's null


def restart_extra_levels(g):
    """The registered rule (REQ-RUN-41): the smallest proven enumeration levels above the set's top proven held-out
    level, until one lies at or above RESTART_TOP_BITS (bits = log2 of the enumeration's cumulative distinct count)."""
    from reach.analysis.s2.phase1 import enum_dir
    held = json.loads((ROOT / f"data/targets_s2/{g}/heldout.json").read_text())
    proven = [t for t in held if t["K_proven"] == t["K_construction"] and not t.get("constructed")]
    top_k = max(t["K_construction"] for t in proven)
    if max(t["units"]["bits_sem"] for t in proven) >= RESTART_TOP_BITS:
        return []
    levels = json.loads((enum_dir(g) / "counts.json").read_text())["levels"]
    out = []
    for st in sorted((st for st in levels if st["K"] > top_k and st["U"] > 0), key=lambda st: st["K"]):
        out.append(st["K"])
        if math.log2(st["U_cum"]) >= RESTART_TOP_BITS:
            break
    return out


@phase("s4_targets", "STAGE 4 targets (D100): RS-add_sqrt's extra proven levels by the registered rule, 30 targets "
       "each, seed TARGET_SEEDS['restart'], into data/targets_s2/<set>/restart.json. No search: drawn and checksummed "
       "before the Stage 4 registration, as the held-out targets before Stage 2. --workers 1.")
def s4_targets():
    from reach.seeds import TARGET_SEEDS
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.s2_targets", "--grammar", g, "--name", "restart",
                              "--n", "30", "--seed", str(TARGET_SEEDS["restart"]),
                              "--levels", ",".join(map(str, restart_extra_levels(g)))],
                 cpu_limit=6 * 3600, meta=dict(grammar=g)) for g in REAL_HELDOUT if restart_extra_levels(g)]


def _s4_task(g, tname, t, seed):
    """One restart run: as `_s2_task` (unchanged, registered) with 100 generations and its own CPU limit."""
    from reach.s2_lock import check_search_allowed
    from reach.tasks.s2_configs import CONFIGS
    check_search_allowed(g)
    rel = f"data/targets_s2/{g}/{tname}.json"
    sha = (ROOT / f"data/targets_s2/{g}/{tname}.sha256").read_text().split()[0]
    pop, gens = CONFIGS["base"]["pop_size"], RESTART_GENERATIONS
    return dict(key=f"s4_restart|{g}|{t['target_id']}|A|base|{gens}|{seed}",
                argv=["{python}", "-m", "reach.tasks.s2_run", "--set", g, "--targets", rel, "--target_id",
                      t["target_id"], "--targets_sha256", sha, "--algorithm", "A", "--config", "base", "--seed",
                      str(seed), "--generations", str(gens), "--pop_size", str(pop), "--penalty", "scaled",
                      "--stop_on", "exact", "--seed_leaves", "off", "--timeout", str(RESTART_TIMEOUT), PRINT_TREE],
                cpu_limit=RESTART_TIMEOUT + 100, weight=0.5,
                meta=dict(block="s4_restart", set=g, target_id=t["target_id"], K=t["K_construction"],
                          K_proven=t["K_proven"], algorithm="A", config="base", generations=gens, budget=0.5,
                          pop_size=pop, seed=seed, targets=tname))


@phase("s4_restart", "STAGE 4 (locked until Stage 4; D100): every proven level of the six real held-out sets (and "
       "RS-add_sqrt's restart levels) x 64 fresh seeds x 100 generations, regime A base (H11).")
def s4_restart():
    from reach.seeds import s2_seeds
    from reach.s2_lock import check_stage4
    check_stage4()
    tasks = []
    for g in REAL_HELDOUT:
        for tname in ("heldout", "restart"):
            p = ROOT / f"data/targets_s2/{g}/{tname}.json"
            if tname == "restart" and not restart_extra_levels(g):
                continue
            for t in json.loads(p.read_text()):
                if t["K_proven"] == t["K_construction"] and not t.get("constructed"):
                    tasks += [_s4_task(g, tname, t, s) for s in s2_seeds("restart")]
    return tasks


# ---------------------------------------------------------------- Study 2, Phase 3 pilot (calibration sets only; C1)
# Batch A: sets whose defining enumeration exists once s2_enum_bool ends; batch B: the rest (s2_enum_extra, cap16).
PILOT_SETS_A = ["F-EML", "R-ARITH4", "R-ELEM4", "R-HYB", "F-LDE0", "BOOL-NAND-n5", "BOOL-NOR-n5", "BOOL-IMP0-n5",
                "BOOL-AON-n5", "BOOL-AONX-n5", "BOOL-RICH-n5"]
PILOT_SETS_B = ["F-EML-R", "R-TRIG", "BOOL-IMP0R-n5", "BOOL-NAND-n6", "BOOL-AON-n6"]
DROPPED = {"R-SRB": "proven ceiling 9 < 10 at its defining (8 GB) enumeration: dropped by the D052 rule (D059)"}
PILOT_MECH_SETS = ["F-EML", "R-ELEM4", "BOOL-NAND-n5", "BOOL-AONX-n5"]       # 2 per domain: anchor and a rich set
S2_TIMEOUT = {1: 900, 4: 3000, 16: 7200}                                     # soft CPU limits by budget (xE0)


def _s2_task(block, g, tname, t, algo, config, seed, budget, penalty="scaled", print_tree=False):
    """One run of reach.tasks.s2_run with every scientific setting explicit (C6), at `budget` x E0; population and
    generations come from the configuration (T19). Refuses held-out sets (C1). print_tree: append `--print_tree`
    (REQ-ENG-22: the whole best tree after the search, nothing else changed); every Stage 2 tree-GP task has it (D092)."""
    from reach.s2_lock import check_search_allowed
    from reach.tasks.s2_configs import CONFIGS
    check_search_allowed(g)
    rel = f"data/targets_s2/{g}/{tname}.json"
    sha = (ROOT / f"data/targets_s2/{g}/{tname}.sha256").read_text().split()[0]
    pop, gens = CONFIGS[config]["pop_size"], budget * CONFIGS[config]["generations_per_E0"]
    timeout = S2_TIMEOUT[budget]
    return dict(key=f"{block}|{g}|{t['target_id']}|{algo}|{config}|{gens}|{seed}",
                argv=["{python}", "-m", "reach.tasks.s2_run", "--set", g, "--targets", rel, "--target_id", t["target_id"],
                      "--targets_sha256", sha, "--algorithm", algo, "--config", config, "--seed", str(seed),
                      "--generations", str(gens), "--pop_size", str(pop), "--penalty", penalty, "--stop_on", "exact",
                      "--seed_leaves", "off", "--timeout", str(timeout)] + ([PRINT_TREE] if print_tree else []),
                cpu_limit=timeout + 100, weight=budget,
                meta=dict(block=block, set=g, target_id=t["target_id"], K=t["K_construction"], K_proven=t["K_proven"],
                          algorithm=algo, config=config, generations=gens, budget=budget, pop_size=pop, seed=seed))


def _pilot_targets(g):
    """A set whose pilot targets are missing (its target task failed) is skipped with a warning, so one failure does
    not block a whole batch; the skip is visible in the console and the frozen manifest."""
    p = ROOT / f"data/targets_s2/{g}/pilot.json"
    if not p.exists():
        import sys
        print(f"WARNING: no pilot targets for {g}; skipped", file=sys.stderr)
        return []
    return json.loads(p.read_text())


def _targets_phase(sets):
    from reach.seeds import TARGET_SEEDS
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.s2_targets", "--grammar", g, "--name", "pilot",
                              "--n", "10", "--seed", str(TARGET_SEEDS["pilot"])],
                 cpu_limit=4 * 3600, meta=dict(grammar=g, out=f"data/targets_s2/{g}")) for g in sets]


@phase("s2_targets_pilot_a", "Study 2 pilot targets, batch A (sets enumerated by the end of s2_enum_bool). Real sets "
       "load their values (up to about 8 GB): --workers 2, never alongside an enumeration.")
def s2_targets_pilot_a():
    return _targets_phase(PILOT_SETS_A)


@phase("s2_targets_pilot_b", "Study 2 pilot targets, batch B (after s2_enum_extra and s2_enum_ceiling). --workers 2.")
def s2_targets_pilot_b():
    return _targets_phase(PILOT_SETS_B)


def _reach(sets):
    from reach.seeds import s2_seeds
    return [_s2_task("pilot_reach", g, "pilot", t, "A", "base", s, 4)
            for g in sets for t in _pilot_targets(g) for s in s2_seeds("pilot")[:3]]


@phase("s2_pilot_reach_a", "Pilot: regime A at 4xE0 (prefixes 1/4-4x) on every pilot target of batch A x 3 pilot seeds.")
def s2_pilot_reach_a():
    return _reach(PILOT_SETS_A)


@phase("s2_pilot_reach_b", "Pilot: regime A at 4xE0 on every pilot target of batch B x 3 pilot seeds.")
def s2_pilot_reach_b():
    return _reach(PILOT_SETS_B)


def _mid_levels(g, n=5):
    ks = sorted({t["K_construction"] for t in _pilot_targets(g) if t["K_proven"] is not None})
    idx = sorted({round(i * (len(ks) - 1) / (n - 1)) for i in range(n)})
    return {ks[i] for i in idx}


@phase("s2_pilot_mech", "Pilot: regimes B and headless chicken at E0 on 2 sets per domain x 5 levels x 10 targets x 3 "
       "pilot seeds (margins for H5).")
def s2_pilot_mech():
    from reach.seeds import s2_seeds
    tasks = []
    for g in PILOT_MECH_SETS:
        lv = _mid_levels(g)
        tasks += [_s2_task("pilot_mech", g, "pilot", t, algo, "base", s, 1)
                  for t in _pilot_targets(g) if t["K_construction"] in lv for algo in ("B", "HC")
                  for s in s2_seeds("pilot")[:3]]
    return tasks


@phase("s2_pilot_long", "Pilot: regime A at 16xE0 on 2 sets per domain x 3 levels x 10 targets x 2 further pilot seeds "
       "(cost and reach at 16x).")
def s2_pilot_long():
    from reach.seeds import s2_seeds
    tasks = []
    for g in PILOT_MECH_SETS:
        lv = sorted(_mid_levels(g, 5))[1:4]
        tasks += [_s2_task("pilot_long", g, "pilot", t, "A", "base", s, 16)
                  for t in _pilot_targets(g) if t["K_construction"] in lv for s in s2_seeds("pilot")[3:5]]
    return tasks


@phase("s2_pilot_robust", "Pilot: robustness configurations R2 (population 1,000) and R4 (lexicase) at 4xE0 on 2 sets "
       "per domain x 3 levels x 10 targets x 2 pilot seeds (cost and behaviour for H8).")
def s2_pilot_robust():
    from reach.seeds import s2_seeds
    tasks = []
    for g in PILOT_MECH_SETS:
        lv = sorted(_mid_levels(g, 5))[1:4]
        tasks += [_s2_task("pilot_robust", g, "pilot", t, "A", cfg, s, 4)
                  for t in _pilot_targets(g) if t["K_construction"] in lv for cfg in ("R2", "R4")
                  for s in s2_seeds("pilot")[:2]]
    return tasks


# ---------------------------------------------------------------- Cartesian GP: validation and pilot (D072 step 3)
# docs/design/cgp.md fixes both. Not confirmatory. CGP runs on calibration, pool or held-out targets wait for the
# Stage 2 registration (D072), so CGP tasks accept only the pilot and validation target files (REQ-RUN-26).
CGP_TARGET_FILES = ("pilot", "validation")
CGP_STAGE2_TARGET_FILES = ("calibration", "heldout")  # s2_cal_cgp, s2_ho_cgp: only once Stage 2 is recorded (D076)
CGP_PILOT_SETS = list(PILOT_MECH_SETS)              # F-EML, R-ELEM4, BOOL-NAND-n5, BOOL-AONX-n5
CGP_PILOT_LENGTHS = (100, 500, 2000)
CGP_PILOT_BUDGET = 16                               # x E0: one run gives every budget 1/4-16 x E0 by prefix
CGP_PILOT_TIMEOUT = 1800                            # soft CPU s; timing 2026-10-26: real failures 3-37 s, Boolean < 1 s
CGP_VALIDATION_LENGTHS = (500, 3000)                # Goldman and Punch 2015 Table II; 2013 Table 1
CGP_VALIDATION_BUDGET = 500                         # x E0 = 10^7 evaluations, their limit
CGP_VALIDATION_TIMEOUT = 600                        # soft CPU s; timing: about 0.02 s per run


def _cgp_binary():
    """REQ-RUN-27: the binary for the pinned Rust source must be in the store before a CGP phase is frozen; otherwise
    every task would end as an error row ("refused: no CGP binary")."""
    from reach import cgp
    try:
        return cgp.binary(store=cgp.STORE)
    except cgp.BinaryError as err:
        raise cgp.BinaryError(f"CGP phase not planned: {err}. Build the binary first, from the repository root of this "
                              "checkout: .venv/bin/python -c 'from reach import cgp; print(cgp.build())'") from None


def _cgp_task(block, g, tname, t, n_nodes, seed, budget, timeout, weight=1.0):
    """One run of reach.tasks.s2_cgp with every scientific setting explicit (C6), at `budget` x E0 evaluations.
    Refuses target files other than pilot and validation, and held-out sets (C1)."""
    from reach import cgp                                   # noqa: F401  (registers the validation set)
    from reach.s2_lock import check_search_allowed, check_stage2
    if tname in CGP_STAGE2_TARGET_FILES:                    # the confirmatory CGP phases (D076; REQ-RUN-31)
        try:
            check_stage2()
        except PermissionError:
            raise PermissionError(f"CGP on {tname} targets: CGP runs on calibration or held-out targets wait for the "
                                  "Stage 2 registration (D072)") from None
    elif tname not in CGP_TARGET_FILES:
        raise PermissionError(f"CGP on {tname} targets: CGP runs only on pilot and validation targets, and on "
                              "calibration and held-out targets after the Stage 2 registration (D072, D076)")
    check_search_allowed(g)
    rel = f"data/targets_s2/{g}/{tname}.json"
    sha = (ROOT / f"data/targets_s2/{g}/{tname}.sha256").read_text().split()[0]
    return dict(key=f"{block}|{g}|{t['target_id']}|CGP|n{n_nodes}|{budget}|{seed}",
                argv=["{python}", "-m", "reach.tasks.s2_cgp", "--set", g, "--targets", rel, "--target_id",
                      t["target_id"], "--targets_sha256", sha, "--seed", str(seed), "--n_nodes", str(n_nodes),
                      "--levels_back", "unrestricted", "--mutation", "single", "--strategy", "1+4",
                      "--budget", str(budget), "--penalty", "scaled", "--stop_on", "exact", "--timeout", str(timeout)],
                cpu_limit=timeout + 100, weight=weight,
                meta=dict(block=block, set=g, target_id=t["target_id"], K=t.get("K_construction"),
                          K_proven=t.get("K_proven"), algorithm="CGP", config=f"n{n_nodes}", n_nodes=n_nodes,
                          budget=budget, seed=seed))


@phase("s2_cgp_validate", "CGP validation (not confirmatory; docs/design/cgp.md): Goldman and Punch (2015) Table II "
       "Single-Normal, even-3 parity with {AND, OR, NAND, NOR}, (1+4)-ES, single active mutation, unrestricted "
       "levels-back, up to 10^7 evaluations, 500 nodes (and 3,000 nodes, their 2013 Table 1) x 200 seeds. "
       "Analysis: python -m reach.analysis.s2.cgp_validate.")
def s2_cgp_validate():
    from reach import cgp
    from reach.seeds import s2_seeds
    _cgp_binary()
    g = cgp.VALIDATION_SET
    (t,) = json.loads((ROOT / f"data/targets_s2/{g}/validation.json").read_text())
    return [_cgp_task("cgp_validate", g, "validation", t, n, s, CGP_VALIDATION_BUDGET, CGP_VALIDATION_TIMEOUT, 0.02)
            for n in CGP_VALIDATION_LENGTHS for s in s2_seeds("cgp_validation")]


@phase("s2_pilot_cgp", "CGP pilot (not confirmatory; docs/design/cgp.md): every pilot target of F-EML, R-ELEM4, "
       "BOOL-NAND-n5, BOOL-AONX-n5 x genome length 100, 500, 2,000 x 3 CGP pilot seeds, to 16xE0. "
       "Analysis: python -m reach.analysis.s2.cgp_pilot (genome-length rule fixed in advance).")
def s2_pilot_cgp():
    from reach import grammar as G
    from reach.seeds import s2_seeds
    _cgp_binary()
    tasks = []
    for g in CGP_PILOT_SETS:
        weight = 1.0 if G.get(g).kind == "real" else 0.03
        targets = json.loads((ROOT / f"data/targets_s2/{g}/pilot.json").read_text())    # all four must exist
        tasks += [_cgp_task("pilot_cgp", g, "pilot", t, n, s, CGP_PILOT_BUDGET, CGP_PILOT_TIMEOUT, weight)
                  for t in targets for n in CGP_PILOT_LENGTHS for s in s2_seeds("cgp_pilot")[:3]]
    return tasks


# ---------------------------------------------------------------- Cartesian GP: confirmatory phases (Stage 2; D076)
# Registered in Stage 2 (H10a, H10b); locked by check_stage2 and checked against stage2_tasks.json (s2_lock.STAGE2_PHASES).
# The pilot-chosen settings come ONLY from the settings file that reach.analysis.s2.cgp_pilot writes from the complete
# pilot (REQ-ANA-25, REQ-RUN-28). One run per target and seed to 16xE0 gives every budget 1/4-16xE0 by prefix.
CGP_SETTINGS = "data/registration/cgp_settings.json"
CGP_TRANSFER = "data/registration/cgp_transfer.json"   # H10b transfer model, frozen before any held-out CGP run
CGP_CONF_BUDGET = 16


def _checked_json(rel, what):
    """A registration file and its .sha256 (written together); the body must match the checksum."""
    import hashlib
    p = ROOT / rel
    if not p.exists():
        raise FileNotFoundError(what)
    sha = p.with_suffix(".sha256")
    if not sha.exists() or sha.read_text().split()[0] != hashlib.sha256(p.read_bytes()).hexdigest():
        raise ValueError(f"{rel} does not match its checksum ({sha.name}): it was changed after it was written")
    return json.loads(p.read_text())


def cgp_settings():
    """The confirmatory CGP settings (REQ-RUN-28): genome length and soft CPU limit per domain, and the levels."""
    st = _checked_json(CGP_SETTINGS, f"CGP confirmatory phase not planned: {CGP_SETTINGS} is missing. It is written by "
                       "`.venv/bin/python -m reach.analysis.s2.cgp_pilot` once s2_pilot_cgp is complete (genome length "
                       "by the rule fixed in docs/design/cgp.md)")
    for dom in ("real", "bool"):
        n = st.get("genome_length", {}).get(dom)
        if n not in CGP_PILOT_LENGTHS:
            raise ValueError(f"{CGP_SETTINGS}: genome length {n!r} for {dom} is not one the pilot compared "
                             f"{CGP_PILOT_LENGTHS}")
        lim = st.get("soft_cpu_limit_s", {}).get(dom)
        if not isinstance(lim, (int, float)) or lim < 600:
            raise ValueError(f"{CGP_SETTINGS}: soft CPU limit {lim!r} for {dom} is not a number of at least 600 s")
    if st.get("levels") not in ("all", "budget"):
        raise ValueError(f"{CGP_SETTINGS}: levels {st.get('levels')!r} is neither 'all' nor 'budget'")
    return st


def cgp_transfer_frozen():
    """H10b: the CGP transfer model must be frozen (file and .sha256) before any held-out CGP run (REQ-RUN-30)."""
    from reach.analysis.s2.cgp_confirm import check_transfer
    try:
        models = _checked_json(CGP_TRANSFER, f"{CGP_TRANSFER} is missing")
        check_transfer(models)                       # content, not only the checksum (row 91)
        return models
    except (FileNotFoundError, ValueError) as err:
        raise PermissionError(f"s2_ho_cgp not planned: held-out CGP runs wait for the H10b transfer model, fitted on "
                              f"s2_cal_cgp and frozen with its SHA-256 ({err}); see reach.analysis.s2.cgp_confirm") from None


def _cgp_levels(lv_g, which):
    return set(lv_g["main"] + lv_g["budget"]) if which == "all" else set(lv_g["budget"])


def _cgp_conf_tasks(block, sets, levels, tname, seeds, st, n_targets=None):
    from reach import grammar as G
    tasks = []
    for g in sets:
        dom = G.get(g).kind
        n, timeout = st["genome_length"][dom], st["soft_cpu_limit_s"][dom]
        sizes = _cgp_levels(levels[g], st["levels"])
        by = defaultdict(list)
        for t in json.loads((ROOT / f"data/targets_s2/{g}/{tname}.json").read_text()):
            if t["K_construction"] in sizes:
                by[t["K_construction"]].append(t)
        for K in sorted(by):
            tasks += [_cgp_task(block, g, tname, t, n, s, CGP_CONF_BUDGET, timeout, 1.0 if dom == "real" else 0.03)
                      for t in by[K][:n_targets] for s in seeds]
    return tasks


@phase("s2_cal_cgp", "CONFIRMATORY (locked until Stage 2; H10a, H10b): CGP on the 16 calibration sets, registered levels "
       "(all 10, or the 5 budget levels, as cgp_settings.json says) x 30 calibration targets x 5 CGP calibration seeds, "
       "one run to 16xE0 each; genome length per domain from data/registration/cgp_settings.json.")
def s2_cal_cgp():
    from reach.s2_lock import check_stage2
    from reach.seeds import s2_seeds
    check_stage2()
    st = cgp_settings()
    _cgp_binary()
    lv = _cal_levels()
    return _cgp_conf_tasks("cal_cgp", [g for g in CAL_SETS if g in lv], lv, "calibration", s2_seeds("cgp_calibration"),
                           st, n_targets=30)


def _ho_cgp_tasks():
    """s2_ho_cgp's task list; registration builds it before the transfer model exists (registration_build)."""
    from reach.s2_lock import check_stage2
    from reach.seeds import s2_seeds
    check_stage2()
    st = cgp_settings()
    _cgp_binary()
    if not (ROOT / "data/registration/heldout_levels.json").exists():
        raise FileNotFoundError("s2_ho_cgp not planned: data/registration/heldout_levels.json is missing (written by "
                                "the freeze step, reach.analysis.s2.freeze, from the frozen model)")
    lv = _ho_levels()
    missing = [g for g in heldout_sets() if g not in lv]
    if missing:
        raise ValueError(f"s2_ho_cgp not planned: no held-out levels for {missing} in heldout_levels.json")
    return _cgp_conf_tasks("ho_cgp", heldout_sets(), lv, "heldout", s2_seeds("cgp_heldout"), st)


@phase("s2_ho_cgp", "HELD-OUT (locked until Stage 2 and until the H10b transfer model is frozen): CGP on the 12 held-out "
       "sets, held-out levels from the frozen model (as many as s2_cal_cgp: cgp_settings.json) x their held-out "
       "targets x 5 CGP held-out seeds, one run to 16xE0 each.")
def s2_ho_cgp():
    from reach.s2_lock import check_stage2
    check_stage2()
    cgp_transfer_frozen()
    return _ho_cgp_tasks()


REGISTRATION_BUILD = {"s2_ho_cgp": _ho_cgp_tasks}      # task lists hashed for Stage 2 before the transfer model exists


def registration_build(name):
    """The builder whose task list is registered for `name` (scripts/s2_task_lists.py, guardrail tests)."""
    return REGISTRATION_BUILD.get(name, PHASES[name]["build"])


# ---------------------------------------------------------------- Study 2, gate G2 -> calibration targets (no search)
@phase("s2_targets_cal", "Study 2 calibration targets (gate G2, D062): 30 per level on the 10 levels of "
       "data/registration/levels.json, seed TARGET_SEEDS['calibration']. Target generation is not a search, so it runs "
       "before Stage 1; the files and checksums are attached to it. Real sets load their values: --workers 2.")
def s2_targets_cal():
    from reach.seeds import TARGET_SEEDS
    lv = _cal_levels()
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.s2_targets", "--grammar", g, "--name", "calibration",
                              "--n", "30", "--seed", str(TARGET_SEEDS["calibration"]),
                              "--levels", ",".join(str(k) for k in sorted(lv[g]["main"] + lv[g]["budget"]))],
                 cpu_limit=6 * 3600, meta=dict(grammar=g, out=f"data/targets_s2/{g}"))
            for g in CAL_SETS if g in lv]


@phase("s2_targets_pool", "Study 2 shared-function pool for H3 (T12): 40 functions per domain over the full sets that "
       "s2_cal_pool runs, seed TARGET_SEEDS['pool']. Loads every real set's values: --workers 1.")
def s2_targets_pool():
    from reach.seeds import TARGET_SEEDS
    return [dict(key=d, argv=["{python}", "-m", "reach.tasks.s2_pool", "--domain", d, "--n", "40",
                              "--seed", str(TARGET_SEEDS["pool"])],
                 cpu_limit=6 * 3600, meta=dict(domain=d)) for d in ("real", "bool")]


# ---------------------------------------------------------------- Study 2, Phase 5 calibration (confirmatory; locked)
# Locked until the Stage 1 registration is recorded (REQ-RUN-18). Levels come from data/registration/levels.json,
# written at gate G2 ({set: {"main": 5 sizes at E0 only, "budget": 5 sizes run to 16xE0}}); targets from
# data/targets_s2/<set>/calibration.json (30 per level). Seeds: "main" (5) and "calibration" (32; the first 5 = main).
CAL_SETS = [g for g in PILOT_SETS_A + PILOT_SETS_B]
REDUCED_SETS = ["F-EML-R", "BOOL-IMP0R-n5", "BOOL-NAND-n6", "BOOL-AON-n6"]      # controls and size checks: A only (D053)
FULL_SETS = [g for g in CAL_SETS if g not in REDUCED_SETS]


def _cal_levels():
    return json.loads((ROOT / "data/registration/levels.json").read_text())


def _cal(block, algos, budget, which, seeds, n_targets=30, sets=None, configs=("base",)):
    from reach.s2_lock import check_stage1
    check_stage1()
    lv = _cal_levels()
    tasks = []
    for g in (CAL_SETS if sets is None else sets):
        if g not in lv:
            continue
        sizes = set(lv[g]["main"] if which == "main" else lv[g]["budget"] if which == "budget"
                    else lv[g]["main"] + lv[g]["budget"])
        by = defaultdict(list)
        for t in json.loads((ROOT / f"data/targets_s2/{g}/calibration.json").read_text()):
            if t["K_construction"] in sizes:
                by[t["K_construction"]].append(t)
        for K in sorted(by):
            tasks += [_s2_task(block, g, "calibration", t, a, c, s, budget)
                      for t in by[K][:n_targets] for a in algos for c in configs for s in seeds]
    return tasks


@phase("s2_cal_a16", "CONFIRMATORY (locked until Stage 1): regime A at 16xE0 on the 5 budget levels x 30 targets x 5 seeds.")
def s2_cal_a16():
    from reach.seeds import s2_seeds
    return _cal("cal_a16", ["A"], 16, "budget", s2_seeds("main"))


@phase("s2_cal_a0", "CONFIRMATORY (locked until Stage 1): regime A at E0 on the 5 main levels x 30 targets x 5 seeds.")
def s2_cal_a0():
    from reach.seeds import s2_seeds
    return _cal("cal_a0", ["A"], 1, "main", s2_seeds("main"))


@phase("s2_cal_mech", "CONFIRMATORY (locked until Stage 1): regimes B and HC at E0 on all 10 levels x 30 targets x 5 seeds.")
def s2_cal_mech():
    from reach.seeds import s2_seeds
    return _cal("cal_mech", ["B", "HC"], 1, "all", s2_seeds("main"), sets=FULL_SETS)


@phase("s2_cal_alloc", "CONFIRMATORY (locked until Stage 1): regime A at E0 on the budget levels x 20 targets x seeds 6-32.")
def s2_cal_alloc():
    from reach.seeds import s2_seeds
    return _cal("cal_alloc", ["A"], 1, "budget", s2_seeds("calibration")[5:], n_targets=20, sets=FULL_SETS)


OPERON_SETS = ("R-ARITH", "R-ARITH4", "R-ELEM6", "R-ELEM4", "R-TRIG")     # standard-operator real sets (T20)


def _operon_task(block, g, tname, t, cfg, seed, budget):
    """One Operon run (reach.tasks.s2_operon) in .venv-ext, with its library path; refuses held-out sets (C1)."""
    from reach.s2_lock import check_search_allowed
    check_search_allowed(g)
    rel = f"data/targets_s2/{g}/{tname}.json"
    sha = (ROOT / f"data/targets_s2/{g}/{tname}.sha256").read_text().split()[0]
    return dict(key=f"{block}|{g}|{t['target_id']}|operon|{cfg}|{budget}|{seed}",
                argv=[str(ROOT / ".venv-ext/bin/python"), "-m", "reach.tasks.s2_operon", "--set", g, "--targets", rel,
                      "--target_id", t["target_id"], "--targets_sha256", sha, "--config", cfg,
                      "--budget", str(budget), "--seed", str(seed)],
                env=dict(DYLD_LIBRARY_PATH=str(ROOT / ".venv-ext/operon-libs")), cpu_limit=600, weight=0.05,
                meta=dict(block=block, set=g, target_id=t["target_id"], K=t["K_construction"], K_proven=t["K_proven"],
                          algorithm="operon", config=cfg, budget=budget, seed=seed))


@phase("s2_cal_operon", "CONFIRMATORY (locked until Stage 1): Operon, configurations symbolic and default, at 16xE0 on "
       "the standard real calibration sets x 10 levels x 30 targets x 5 seeds (H9a). Runs in .venv-ext.")
def s2_cal_operon():
    from reach.seeds import s2_seeds
    from reach.s2_lock import check_stage1
    check_stage1()
    lv = _cal_levels()
    tasks = []
    for g in [g for g in CAL_SETS if g in OPERON_SETS and g in lv]:
        sizes = set(lv[g]["main"] + lv[g]["budget"])
        for t in json.loads((ROOT / f"data/targets_s2/{g}/calibration.json").read_text()):
            if t["K_construction"] in sizes:
                tasks += [_operon_task("cal_operon", g, "calibration", t, cfg, s, 16)
                          for cfg in ("symbolic", "default") for s in s2_seeds("main")]
    return tasks


@phase("s2_smoke_cal", "Pre-launch smoke test (not confirmatory): every calibration task type once (A at 16x, B and HC "
       "at E0, R1-R6 at 4x, Operon both configurations at 16x) on pilot targets with an unused pilot seed, through the "
       "queue. Its records are read by the analysis to check the whole path before the confirmatory runs.")
def s2_smoke_cal():
    from reach.seeds import s2_seeds
    seed = s2_seeds("pilot")[19]
    tasks = []
    for g in ("F-EML", "BOOL-NAND-n5", "R-ARITH4"):
        ts = _pilot_targets(g)
        t = sorted(ts, key=lambda x: x["K_construction"])[len(ts) // 3]
        tasks += [_s2_task("smoke", g, "pilot", t, "A", "base", seed, 16)]
        tasks += [_s2_task("smoke", g, "pilot", t, a, "base", seed, 1) for a in ("B", "HC")]
        tasks += [_s2_task("smoke", g, "pilot", t, "A", c, seed, 4) for c in ("R1", "R2", "R3", "R4", "R5", "R6")]
    t = sorted(_pilot_targets("R-ARITH4"), key=lambda x: x["K_construction"])[len(_pilot_targets("R-ARITH4")) // 3]
    tasks += [_operon_task("smoke", "R-ARITH4", "pilot", t, cfg, seed, 16) for cfg in ("symbolic", "default")]
    return tasks


@phase("s2_cal_robust", "CONFIRMATORY (locked until Stage 1): robustness R1-R6 at 4xE0 on the G2 robustness sets x the "
       "budget levels x 20 targets x 5 seeds (H8).")
def s2_cal_robust():
    from reach.seeds import s2_seeds
    from reach.s2_lock import check_stage1
    check_stage1()
    sets = [g for d in ("real", "bool") for g in _cal_levels().get("_robust", {}).get(d, [])]
    return _cal("cal_robust", ["A"], 4, "budget", s2_seeds("main"), n_targets=20, sets=sets,
                configs=("R1", "R2", "R3", "R4", "R5", "R6"))


@phase("s2_cal_pool", "CONFIRMATORY (locked until Stage 1): regime A at E0 on the shared-function pool x 30 pool seeds (H3).")
def s2_cal_pool():
    from reach.seeds import s2_seeds
    from reach.s2_lock import check_stage1
    check_stage1()
    tasks = []
    for g in FULL_SETS:
        p = ROOT / f"data/targets_s2/{g}/pool.json"
        if p.exists():
            tasks += [_s2_task("cal_pool", g, "pool", t, "A", "base", s, 1)
                      for t in json.loads(p.read_text()) for s in s2_seeds("pool")]
    return tasks


@phase("s2_cal_rand", "CONFIRMATORY (locked until Stage 1): random sampling, 4 x 250,000 trees per full set, hits on the "
       "calibration targets (H5c, H2r).")
def s2_cal_rand():
    from reach.s2_lock import check_stage1
    check_stage1()
    tasks = []
    for g in FULL_SETS:
        sha = (ROOT / f"data/targets_s2/{g}/calibration.sha256").read_text().split()[0]
        tasks += [dict(key=f"{g}|{i}", argv=["{python}", "-m", "reach.tasks.s2_random", "--set", g, "--targets",
                                              f"data/targets_s2/{g}/calibration.json", "--targets_sha256", sha,
                                              "--n", "250000", "--seed", str(20261030 + 10 * FULL_SETS.index(g) + i)],
                       cpu_limit=3 * 3600, meta=dict(set=g, part=i)) for i in range(4)]
    return tasks


def _rand(sets):
    sets = [g for g in sets if (ROOT / f"data/targets_s2/{g}/pilot.json").exists()]
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.s2_random", "--set", g, "--targets",
                              f"data/targets_s2/{g}/pilot.json", "--targets_sha256",
                              (ROOT / f"data/targets_s2/{g}/pilot.sha256").read_text().split()[0],
                              "--n", "250000", "--seed", str(20261025 + i)],
                 cpu_limit=3 * 3600, meta=dict(set=g)) for i, g in enumerate(sets)]


@phase("s2_pilot_rand_a", "Pilot: random sampling, 2.5e5 trees per set of batch A, hits on the pilot targets.")
def s2_pilot_rand_a():
    return _rand(PILOT_SETS_A)


@phase("s2_pilot_rand_b", "Pilot: random sampling, 2.5e5 trees per set of batch B.")
def s2_pilot_rand_b():
    return _rand(PILOT_SETS_B)


@phase("s2_precision_extra", "Study 2 (D052): precision check (7 and 11 decimals, K <= 13) for the added real sets.")
def s2_precision_extra():
    return [dict(_s2_enum(g, f"{S2_ENUM}/precision{p}", 13, 6, 3600, ("--precision", str(p))), key=f"{g}|p{p}")
            for p in (7, 11) for g in S2_EXTRA_REAL]


@phase("s2_precision", "Study 2 measurement check (brief 7.2): the 8 real sets re-enumerated to K = 13 with keys "
       "rounded to 7 and to 11 decimals. --workers 2.")
def s2_precision():
    return [dict(_s2_enum(g, f"{S2_ENUM}/precision{p}", 13, 6, 3600, ("--precision", str(p))), key=f"{g}|p{p}")
            for p in (7, 11) for g in S2_REAL]

ENUM_GRAMMARS = ["EML", "ARITH", "ELEM6", "HYB"] + [
    f"BOOL-{b}-n{n}" for n in (3, 4) for b in ("NAND", "NOR", "AON", "AN", "AONX", "AX1", "RICH")]


@phase("enum_pilot",
       "M4: enumerate every pilot grammar by size until the memory ceiling (8 GB of stored values for "
       "real grammars; peak about 12 GB) or 4 CPU-hours (18 tasks; checkpointed per level). Run with "
       "--workers 2 so peak memory stays within 32 GB. Gives lambda_G and the null.")
def enum_pilot():
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.enumerate_task", "--grammar", g,
                              "--kmax", "40", "--max_gb", "8" if not g.startswith("BOOL") else "2"],
                 cpu_limit=4 * 3600, weight=1.0, meta=dict(grammar=g)) for g in ENUM_GRAMMARS]


PILOT_GRAMMARS = ["EML", "ARITH", "ELEM6", "HYB"] + [
    f"BOOL-{b}-n4" for b in ("NAND", "NOR", "AON", "AN", "AONX", "AX1", "RICH")]


def _gp_task(exp, g, t, regime, seed, gens, extra=(), weight=1.0, tname="targets"):
    timeout = {200: 600, 800: 2400, 3200: 6000, 12800: 3600 * 6}[gens]
    path = f"data/targets/{g}/{tname}.json"
    tag = "+".join([exp] + [e.strip("-") for e in extra])
    return dict(key=f"{tag}|{t['target_id']}|{regime}|{gens}|{seed}",
                argv=["{python}", "-m", "reach.tasks.treegp_task", "--grammar", g, "--targets", path,
                      "--target_id", t["target_id"], "--regime", regime, "--seed", str(seed),
                      "--generations", str(gens), "--timeout", str(timeout), *extra],
                cpu_limit=timeout + 100, weight=weight,
                meta=dict(exp=exp, grammar=g, target_id=t["target_id"], K=t["K_construction"],
                          K_proven=t["K_proven"], regime=regime, generations=gens, seed=seed,
                          seed_leaves="on" if "--seed_leaves" in extra else "off"))


@phase("pilot_p1",
       "Pilot scout (clean engine, Regime A): every pilot grammar's targets x 3 seeds at E0 = 2e4; "
       "plus probes: E1 = 8e4 (EML, ELEM6; 3 levels), seeded leaf on (EML K11, ELEM6 K10), and a "
       "3,200-generation cost probe. About 3,250 runs, about 20 CPU-hours.")
def pilot_p1():
    from reach.seeds import SEEDS
    tasks = []
    T = {g: json.loads((ROOT / f"data/targets/{g}/targets.json").read_text()) for g in PILOT_GRAMMARS}
    for g in PILOT_GRAMMARS:
        w = 1.0 if g.startswith("BOOL") else 4.0
        for t in T[g]:
            for s in SEEDS[:3]:
                tasks.append(_gp_task("scout", g, t, "A", s, 200, weight=w))
    for g, levels in (("EML", (11, 13, 15)), ("ELEM6", (8, 10, 12))):
        for t in T[g]:
            if t["K_construction"] in levels:
                for s in SEEDS[:3]:
                    tasks.append(_gp_task("e1", g, t, "A", s, 800, weight=16.0))
    for g, k in (("EML", 11), ("ELEM6", 10)):
        for t in T[g]:
            if t["K_construction"] == k:
                for s in SEEDS[:3]:
                    tasks.append(_gp_task("seedleaf", g, t, "A", s, 200, ("--seed_leaves", "on"), weight=4.0))
                tasks.append(_gp_task("e2", g, t, "A", SEEDS[0], 3200, weight=64.0))
    return tasks


@phase("pilot_p1b",
       "Pilot P1 rerun for the real grammars with targets_v2 (dead-subtree filter on every level, D020): "
       "scout x 3 seeds at E0, E1 probe, seeded leaf on, 3,200-generation cost probe. About 1,500 runs.")
def pilot_p1b():
    from reach.seeds import SEEDS
    tn = "targets_v2"
    T = {g: json.loads((ROOT / f"data/targets/{g}/{tn}.json").read_text()) for g in PILOT_GRAMMARS[:4]}
    tasks = [_gp_task("scout", g, t, "A", s, 200, weight=4.0, tname=tn)
             for g in T for t in T[g] for s in SEEDS[:3]]
    for g, levels in (("EML", (11, 13, 15)), ("ELEM6", (8, 10, 12))):
        tasks += [_gp_task("e1", g, t, "A", s, 800, weight=16.0, tname=tn)
                  for t in T[g] if t["K_construction"] in levels for s in SEEDS[:3]]
    for g, k in (("EML", 11), ("ELEM6", 10)):
        for t in T[g]:
            if t["K_construction"] == k:
                tasks += [_gp_task("seedleaf", g, t, "A", s, 200, ("--seed_leaves", "on"), weight=4.0, tname=tn)
                          for s in SEEDS[:3]]
                tasks.append(_gp_task("e2", g, t, "A", SEEDS[0], 3200, weight=64.0, tname=tn))
    return tasks


BOOL5 = [f"BOOL-{b}-n5" for b in ("NAND", "NOR", "AON", "AN", "AONX", "AX1", "RICH")]


@phase("enum_bool5", "Enumerate the 7 Boolean grammars with n = 5 inputs (1 CPU-hour and 3 GB each). "
       "n = 4 saturated at E0 (pilot P1). Run with --workers 2.")
def enum_bool5():
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.enumerate_task", "--grammar", g,
                              "--kmax", "60", "--max_gb", "3"],
                 cpu_limit=3600, meta=dict(grammar=g)) for g in BOOL5]


@phase("scout_bool5", "Scout of the n = 5 Boolean grammars at E0 (clean engine, Regime A, 3 seeds).")
def scout_bool5():
    from reach.seeds import SEEDS
    return [_gp_task("scout", g, t, "A", s, 200)
            for g in BOOL5 if (ROOT / f"data/targets/{g}/targets.json").exists()
            for t in json.loads((ROOT / f"data/targets/{g}/targets.json").read_text()) for s in SEEDS[:3]]


SWEEP_LEVELS = {"EML": (9, 11, 13, 15, 17), "ARITH": (9, 11, 13, 15, 17),
                "HYB": (5, 7, 9, 11, 13), "ELEM6": (4, 5, 6, 7, 8)}


@phase("budget_sweep", "Scaling pilot: real grammars at 800 and 3,200 generations (4x and 16x E0) on the "
       "5 levels around each grammar's E0 reach; targets_v2, same 3 seeds as pilot_p1b (paired).")
def budget_sweep():
    from reach.seeds import SEEDS
    tasks = []
    for g, levels in SWEEP_LEVELS.items():
        for t in json.loads((ROOT / f"data/targets/{g}/targets_v2.json").read_text()):
            if t["K_construction"] in levels:
                for gens, w in ((800, 4.0), (3200, 16.0)):
                    tasks += [_gp_task("sweep", g, t, "A", s, gens, weight=w, tname="targets_v2") for s in SEEDS[:3]]
    return tasks


@phase("budget_sweep_bool5", "Scaling pilot for the n = 5 Boolean grammars: 800 and 3,200 generations on the "
       "levels where E0 convergence (scout_bool5) is between 0.1 and 0.9 (at most 5 per grammar); same seeds.")
def budget_sweep_bool5():
    from collections import defaultdict
    from reach.runner.queue import read_results
    from reach.seeds import SEEDS
    rate = defaultdict(list)
    for r in read_results(ROOT / "data/runs/scout_bool5/results.jsonl").values():
        rate[(r["grammar"], r["K"])].append(r["converged"] == "true")
    tasks = []
    for g in BOOL5:
        ks = sorted(k for (gg, k), v in rate.items() if gg == g and 0.1 <= sum(v) / len(v) <= 0.9)
        mid = sorted(ks, key=lambda k: abs(sum(rate[(g, k)]) / len(rate[(g, k)]) - 0.5))[:5]
        for t in json.loads((ROOT / f"data/targets/{g}/targets.json").read_text()):
            if t["K_construction"] in mid:
                for gens, w in ((800, 4.0), (3200, 16.0)):
                    tasks += [_gp_task("sweep", g, t, "A", s, gens, weight=w) for s in SEEDS[:3]]
    return tasks


# ---------------------------------------------------------------- EML-family feasibility pilot (D021)
def _family():
    from reach.family import FAMILY
    return FAMILY


@phase("enum_family", "EML-family pilot step 1: enumerate the 7 family grammars (faithful semantics) until "
       "the 8 GB memory ceiling or 1 CPU-hour each. Run with --workers 2.")
def enum_family():
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.enumerate_task", "--grammar", g,
                              "--kmax", "40", "--max_gb", "8"], cpu_limit=3600, meta=dict(grammar=g))
            for g in _family()]


@phase("family_prepare", "EML-family pilot step 2: minimal-construction tables, scout targets (targets_fam) "
       "and calculator targets (calc) per operator.")
def family_prepare():
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.family_prepare", "--grammar", g],
                 cpu_limit=3600, meta=dict(grammar=g)) for g in _family()]


@phase("scout_family", "EML-family pilot step 3: GP at E0 (clean engine, Regime A): scout targets x 3 seeds "
       "and calculator targets x 20 seeds, per operator.")
def scout_family():
    from reach.seeds import SEEDS
    tasks = []
    for g in _family():
        for t in json.loads((ROOT / f"data/targets/{g}/targets_fam.json").read_text()):
            tasks += [_gp_task("scout", g, t, "A", s, 200, tname="targets_fam") for s in SEEDS[:3]]
        for t in json.loads((ROOT / f"data/targets/{g}/calc.json").read_text()):
            tasks += [_gp_task("calc", g, t, "A", s, 200, tname="calc") for s in SEEDS[:20]]
    return tasks


@phase("family_predictors", "EML-family: target-level searchability predictors from the enumeration alone "
       "(nearest smaller function in nMSE, near-duplicate counts, subtree viability); targets with K <= 19. "
       "Run with --workers 2 (memory).")
def family_predictors():
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.family_predictors", "--grammar", g, "--kmax", "19"],
                 cpu_limit=7200, meta=dict(grammar=g)) for g in _family()]


# ---------------------------------------------------------------- EML-family pilot, round 2 (D022)
@phase("family_prepare2", "EML-family round 2: targets_fam2 (sizes 3..19, near-duplicate filter). --workers 2.")
def family_prepare2():
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.family_prepare2", "--grammar", g],
                 cpu_limit=7200, meta=dict(grammar=g)) for g in _family()]


@phase("scout_family2", "EML-family round 2: GP at E0 with stop-on-exact: targets_fam2 x 3 seeds and calculator "
       "targets x 20 seeds, per operator.")
def scout_family2():
    from reach.seeds import SEEDS
    tasks, x = [], ("--stop_on", "exact")
    for g in _family():
        for t in json.loads((ROOT / f"data/targets/{g}/targets_fam2.json").read_text()):
            tasks += [_gp_task("scout", g, t, "A", s, 200, x, tname="targets_fam2") for s in SEEDS[:3]]
        for t in json.loads((ROOT / f"data/targets/{g}/calc.json").read_text()):
            tasks += [_gp_task("calc", g, t, "A", s, 200, x, tname="calc") for s in SEEDS[:20]]
    return tasks


@phase("family_predictors2", "EML-family round 2: enumeration-only predictors (incl. locality) for targets_fam2.")
def family_predictors2():
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.family_predictors", "--grammar", g,
                              "--targets", "targets_fam2", "--suffix", "_v2", "--kmax", "19"],
                 cpu_limit=7200, meta=dict(grammar=g)) for g in _family()]


# ---------------------------------------------------------------- full EML-family study (D023; confirmatory)
FAM2 = ["F2-EML", "F2-EDL", "F2-NEML", "F2-LDE0", "F2-LDE1", "F2-PLI", "F2-PLM"]
FULL_SEEDS = slice(20, 50)          # fresh seeds, disjoint from the pilots (SEEDS[0:20])


def _conf(g):
    return json.loads((ROOT / f"data/targets/{g}/targets_conf.json").read_text())


def _calc_sets(g1):
    g2 = "F2-" + g1[2:]
    out = [(g1, "calc", t) for t in json.loads((ROOT / f"data/targets/{g1}/calc.json").read_text())]
    p2 = ROOT / f"data/targets/{g2}/calc2.json"
    out += [(g2, "calc2", t) for t in json.loads(p2.read_text())] if p2.exists() else []
    return out


@phase("enum_family2v", "Full study step 1: enumerate the 7 two-variable family grammars (leaves x, y, c), 8 GB cap.")
def enum_family2v():
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.enumerate_task", "--grammar", g, "--kmax", "40",
                              "--max_gb", "8"], cpu_limit=3600, meta=dict(grammar=g)) for g in FAM2]


@phase("full_prepare", "Full study step 2: confirmatory targets (targets_conf) and two-variable tables + calc2.")
def full_prepare():
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.full_prepare", "--grammar", g], cpu_limit=7200,
                 meta=dict(grammar=g)) for g in _family() + FAM2]


@phase("full_gp", "Full study H1-H3 (confirmatory): GP-A at E0, stop on exact: targets_conf x 5 seeds; calculator "
       "targets (one and two variables) x 30 seeds. Fresh seeds.")
def full_gp():
    from reach.seeds import SEEDS
    s = SEEDS[FULL_SEEDS]
    x = ("--stop_on", "exact")
    tasks = []
    for g in _family():
        tasks += [_gp_task("conf", g, t, "A", sd, 200, x, tname="targets_conf") for t in _conf(g) for sd in s[:5]]
        tasks += [_gp_task(exp, gg, t, "A", sd, 200, x, tname=exp) for gg, exp, t in _calc_sets(g) for sd in s[:30]]
    return tasks


@phase("full_es", "Full study H5: (1+4)-ES at E0, stop on exact: targets_conf sizes 3..15, 15 per size x 3 seeds; "
       "calculator targets x 10 seeds.")
def full_es():
    from reach.seeds import SEEDS
    s = SEEDS[FULL_SEEDS]
    x = ("--stop_on", "exact", "--algorithm", "es")
    tasks = []
    for g in _family():
        by = {}
        for t in _conf(g):
            by.setdefault(t["K_construction"], []).append(t)
        tasks += [_gp_task("es", g, t, "A", sd, 200, x, weight=4.0, tname="targets_conf")
                  for K, ts in by.items() if K <= 15 for t in ts[:15] for sd in s[:3]]
        tasks += [_gp_task("es_" + exp, gg, t, "A", sd, 200, x, weight=4.0, tname=exp)
                  for gg, exp, t in _calc_sets(g) for sd in s[:10]]
    return tasks


@phase("full_budget", "Full study H4: GP-A at 4x and 16x E0, stop on exact: targets_conf sizes 9..17, 20 per size x 3 "
       "seeds; calculator targets x 10 seeds at 16x.")
def full_budget():
    from reach.seeds import SEEDS
    s = SEEDS[FULL_SEEDS]
    x = ("--stop_on", "exact")
    tasks = []
    for g in _family():
        by = {}
        for t in _conf(g):
            by.setdefault(t["K_construction"], []).append(t)
        for gens, w in ((800, 4.0), (3200, 16.0)):
            tasks += [_gp_task("budget", g, t, "A", sd, gens, x, weight=w, tname="targets_conf")
                      for K, ts in by.items() if 9 <= K <= 17 for t in ts[:20] for sd in s[:3]]
        tasks += [_gp_task("budget_" + exp, gg, t, "A", sd, 3200, x, weight=16.0, tname=exp)
                  for gg, exp, t in _calc_sets(g) for sd in s[:10]]
    return tasks


@phase("full_budget64", "Full study H4b: GP-A at 64x E0 (12,800 generations), stop on exact, on a subset: EML, LDE0, "
       "PLM; sizes 11..17; 10 targets per size x 2 seeds.")
def full_budget64():
    from reach.seeds import SEEDS
    s = SEEDS[FULL_SEEDS]
    tasks = []
    for g in ("F-EML", "F-LDE0", "F-PLM"):
        by = {}
        for t in _conf(g):
            by.setdefault(t["K_construction"], []).append(t)
        tasks += [_gp_task("budget64", g, t, "A", sd, 12800, ("--stop_on", "exact"), weight=64.0, tname="targets_conf")
                  for K, ts in by.items() if 11 <= K <= 17 for t in ts[:10] for sd in s[:2]]
    return tasks


@phase("full_pysr", "Full study, exploratory: PySR 2.7 (Julia; .venv-pysr) on the calculator targets (one and two "
       "variables) of 6 operators (-EML excluded: PySR rejects a -inf input column); 10 seeds at E0 and 16x E0.")
def full_pysr():
    return [dict(key=g, cwd=".", argv=[".venv-pysr/bin/python", "-m", "reach.tasks.pysr_check", "--grammar", g],
                 cpu_limit=6 * 3600, weight=30.0, meta=dict(grammar=g))
            for g in ("F-EML", "F-EDL", "F-LDE0", "F-LDE1", "F-PLI", "F-PLM")]


# ---------------------------------------------------------------- held-out extension (D027)
@phase("screen_ops", "Held-out extension A1: universality screen of 30 candidate forms x 4 constants (D027).")
def screen_ops():
    from reach.grammar import CAND_CONSTS, cand_forms
    return [dict(key=f"S-{f}-{c}", argv=["{python}", "-m", "reach.tasks.screen_op", "--form", f, "--const", c],
                 cpu_limit=1800, meta=dict(grammar=f"S-{f}-{c}")) for f in cand_forms() for c in CAND_CONSTS]


@phase("verify_fix", "Held-out extension A3: rerun 500 sampled non-error full_gp runs (must be bit-identical after the "
       "generation-0 elitism fix) and every error run of full_gp and full_budget (exploratory EDL sensitivity).")
def verify_fix():
    import random
    tasks = []
    for ph in ("full_gp", "full_budget"):
        man = json.loads((ROOT / f"data/runs/{ph}/manifest.json").read_text())["tasks"]
        res = {json.loads(l)["key"]: json.loads(l) for l in open(ROOT / f"data/runs/{ph}/results.jsonl")}
        err = [t for t in man if res[t["key"]]["terminal_state"] == "error"]
        pick = err + (random.Random(20261015).sample([t for t in man if res[t["key"]]["terminal_state"] != "error"], 500)
                      if ph == "full_gp" else [])
        tasks += [dict(t, key=f"{ph}|{t['key']}", meta=dict(t["meta"], source=ph)) for t in pick]
    return tasks


@phase("screen_ops2", "Held-out extension A1 (amended): screen with K1 <= 21, K2 <= 17 and the x - y / x / y criterion; "
       "the five known A(x) o B(y) universal operators included as positive controls (D027 amendment).")
def screen_ops2():
    from reach.grammar import CAND_CONSTS, cand_forms
    ctrl = [("esubl", "1"), ("edivl", "e"), ("lsube", "minf"), ("ldive", "0"), ("ldive", "1")]
    return [dict(key=f"S-{f}-{c}", argv=["{python}", "-m", "reach.tasks.screen_op", "--form", f, "--const", c,
                                         "--k1", "21", "--k2", "17", "--max_gb", "6"],
                 cpu_limit=3 * 3600, meta=dict(grammar=f"S-{f}-{c}", control=(f, c) in ctrl))
            for f, c in ctrl + [(f, c) for f in cand_forms() for c in CAND_CONSTS]]


@phase("screen_ops3", "Held-out extension A1 fallback pools (D027): Stachowiak's cos x - arccos y with the 4 constants, "
       "and the four known A(x) o B(y) forms with the constants not yet studied; same screen as screen_ops2.")
def screen_ops3():
    from reach.grammar import CAND_CONSTS
    used = {("esubl", "1"), ("edivl", "e"), ("lsube", "minf"), ("ldive", "0"), ("ldive", "1")}
    pairs = [("cosacos", c) for c in CAND_CONSTS] + [(f, c) for f in ("esubl", "edivl", "lsube", "ldive")
                                                       for c in CAND_CONSTS if (f, c) not in used]
    return [dict(key=f"S-{f}-{c}", argv=["{python}", "-m", "reach.tasks.screen_op", "--form", f, "--const", c,
                                         "--k1", "21", "--k2", "17", "--max_gb", "6"],
                 cpu_limit=3 * 3600, meta=dict(grammar=f"S-{f}-{c}")) for f, c in pairs]


HELDOUT = ["lsube-0", "esubl-e", "lsube-e", "lpowx-m1"]      # D027, chosen by the author after the screen


@phase("enum_heldout", "Held-out extension A2: enumerate the 4 held-out grammars, one and two variables (8 GB cap, "
       "as enum_family / enum_family2v). Run with --workers 2.")
def enum_heldout():
    return [dict(key=f"{p}-{h}", argv=["{python}", "-m", "reach.tasks.enumerate_task", "--grammar", f"{p}-{h}",
                                       "--kmax", "40", "--max_gb", "8"], cpu_limit=3600, meta=dict(grammar=f"{p}-{h}"))
            for h in HELDOUT for p in ("S", "S2")]


@phase("heldout_prepare", "Held-out extension A2: minimal tables, calculator targets and random minimal targets "
       "(fresh seed 20261115) for the held-out grammars.")
def heldout_prepare():
    return [dict(key=f"{p}-{h}", argv=["{python}", "-m", "reach.tasks.heldout_prepare", "--grammar", f"{p}-{h}"],
                 cpu_limit=7200, meta=dict(grammar=f"{p}-{h}")) for h in HELDOUT for p in ("S", "S2")]


HELD_SEEDS = slice(50, 80)           # fresh seeds for the held-out extension


def _held_calc_sets(h):
    out = [(f"S-{h}", "calc", t) for t in json.loads((ROOT / f"data/targets/S-{h}/calc.json").read_text())]
    p2 = ROOT / f"data/targets/S2-{h}/calc2.json"
    return out + ([(f"S2-{h}", "calc2", t) for t in json.loads(p2.read_text())] if p2.exists() else [])


@phase("heldout_gp", "Held-out extension H6/H7 (confirmatory after the OSF addendum): GP-A at E0, stop on exact; "
       "targets_conf x 5 seeds; calculator targets (one and two variables) x 30 seeds. Mirrors full_gp.")
def heldout_gp():
    from reach.seeds import SEEDS
    s, x, tasks = SEEDS[HELD_SEEDS], ("--stop_on", "exact"), []
    for h in HELDOUT:
        tasks += [_gp_task("conf", f"S-{h}", t, "A", sd, 200, x, tname="targets_conf") for t in _conf(f"S-{h}") for sd in s[:5]]
        tasks += [_gp_task(exp, g, t, "A", sd, 200, x, tname=exp) for g, exp, t in _held_calc_sets(h) for sd in s[:30]]
    return tasks


@phase("nox_gp", "Held-out extension H8 (confirmatory after the OSF addendum): GP without crossover (regime B: same "
       "population, selection and budget; every offspring gets subtree mutation) at E0, stop on exact, on the same "
       "targets_conf and seeds as full_gp's scout runs (paired with H1).")
def nox_gp():
    from reach.seeds import SEEDS
    s, x = SEEDS[FULL_SEEDS], ("--stop_on", "exact")
    return [_gp_task("confB", g, t, "B", sd, 200, x, tname="targets_conf") for g in _family() for t in _conf(g)
            for sd in s[:5]]


# D034 (2026-10-16), post hoc sensitivity analyses, specified and committed before any run (not preregistered):
# (1) the penalty for undefined values scaled to the target (pen_*): same targets and seeds as full_gp, nox_gp and
#     full_es (random targets), every operator, so EDL is compared with the pooled six under a fitness that never
#     ranks an undefined tree above a constant; (2) headless-chicken crossover (hc_gp), the standard control for
#     the crossover effect, on full_gp's random targets and first two seeds, with EML-GP's penalty as in full_gp.
@phase("pen_gp", "D034 sensitivity: GP-A at E0, stop on exact, penalty scaled to the target; full_gp's targets_conf "
       "and seeds (5 per target).")
def pen_gp():
    from reach.seeds import SEEDS
    s, x = SEEDS[FULL_SEEDS], ("--stop_on", "exact", "--penalty", "scaled")
    return [_gp_task("confP", g, t, "A", sd, 200, x, tname="targets_conf") for g in _family() for t in _conf(g)
            for sd in s[:5]]


@phase("pen_nox", "D034 sensitivity: GP without crossover (regime B) at E0, stop on exact, penalty scaled to the "
       "target; nox_gp's targets and seeds.")
def pen_nox():
    from reach.seeds import SEEDS
    s, x = SEEDS[FULL_SEEDS], ("--stop_on", "exact", "--penalty", "scaled")
    return [_gp_task("confBP", g, t, "B", sd, 200, x, tname="targets_conf") for g in _family() for t in _conf(g)
            for sd in s[:5]]


@phase("pen_es", "D034 sensitivity: (1+4)-ES at E0, stop on exact, penalty scaled to the target; full_es's random "
       "targets and seeds (sizes 3..15, 15 per size x 3 seeds).")
def pen_es():
    from reach.seeds import SEEDS
    s, x = SEEDS[FULL_SEEDS], ("--stop_on", "exact", "--algorithm", "es", "--penalty", "scaled")
    tasks = []
    for g in _family():
        by = {}
        for t in _conf(g):
            by.setdefault(t["K_construction"], []).append(t)
        tasks += [_gp_task("esP", g, t, "A", sd, 200, x, weight=4.0, tname="targets_conf")
                  for K, ts in by.items() if K <= 15 for t in ts[:15] for sd in s[:3]]
    return tasks


@phase("hc_gp", "D034 control: GP-A with headless-chicken crossover (each parent crossed with a fresh random tree) at "
       "E0, stop on exact, EML-GP's penalty; full_gp's targets_conf and first two seeds.")
def hc_gp():
    from reach.seeds import SEEDS
    s, x = SEEDS[FULL_SEEDS], ("--stop_on", "exact", "--crossover", "headless")
    return [_gp_task("confH", g, t, "A", sd, 200, x, tname="targets_conf") for g in _family() for t in _conf(g)
            for sd in s[:2]]


# D038 (2026-10-16), post hoc, specified and committed before any run (not preregistered), answering the critical
# referee report: (1) more seeds at E0 on the budget runs' set A targets, so that independent restarts can be
# compared with single long runs at equal total budget up to 16 E0; (2) a random-sampling baseline.
@phase("restart_seeds", "D038: GP-A at E0, stop on exact, on full_budget's set A targets (sizes 9..17, 20 per size) "
       "with 15 further seeds (the 6th to 20th), for restarts against single long runs.")
def restart_seeds():
    from reach.seeds import SEEDS
    s, x = SEEDS[FULL_SEEDS], ("--stop_on", "exact")
    tasks = []
    for g in _family():
        by = {}
        for t in _conf(g):
            by.setdefault(t["K_construction"], []).append(t)
        tasks += [_gp_task("conf", g, t, "A", sd, 200, x, tname="targets_conf")
                  for K, ts in by.items() if 9 <= K <= 17 for t in ts[:20] for sd in s[5:20]]
    return tasks


@phase("rand_sample", "D038: random-sampling baseline: 4 x 250,000 trees per operator drawn as GP's initial "
       "population; exact hits of every random target counted.")
def rand_sample():
    return [dict(key=f"rand|{g}|{i}", argv=["{python}", "-m", "reach.tasks.random_sample", "--grammar", g,
                                             "--targets", f"data/targets/{g}/targets_conf.json", "--n", "250000",
                                             "--seed", str(20261016 + i)],
                 cpu_limit=3600, meta=dict(exp="rand", grammar=g, part=i)) for g in _family() for i in range(4)]


# ---------------------------------------------------------------- Stage 5 (D125; docs/design/stage5_*.md; REQ-RUN-42 to 49)
# Every phase below except s5_targets and the s5_pilot_* phases is locked until the Stage 5 registration is recorded
# (s2_lock.check_stage5) and checked against data/registration/stage5_tasks.json (key and full hash). The registered
# builders of Stages 1 to 4 are not changed; the Stage 5 builders reuse them only by reading their task lists.
S5_MAIN_SETS = [g for g in CAL_SETS if g not in ("BOOL-NAND-n6", "BOOL-AON-n6")]   # H6c's 14 main sets
S5_CUT_BITS = math.log2(16 * 20_000)      # 18.29: calibration levels kept for selection-free GP (rule in the design)
LOW_MIN_U, LOW_MIN_BITS, LOW_N = 5, 3.0, 30
S5_KEYS_DIR = "data/keys_s5"              # side files of key sets (untracked, like the level files)
S5_LONG_TIMEOUT = S2_TIMEOUT[16]
STAGE5_ORDER = (("s5_targets",),
                ("s5_pilot_grid", "s5_pilot_wide", "s5_pilot_boolsub", "s5_pilot_selfree", "s5_pilot_pop1000",
                 "s5_pilot_keys", "s5_pilot_robust", "s5_pilot_restart10"),
                ("s5_grid", "s5_boolsub", "s5_wide"),
                ("s5_selfree", "s5_low_base"),
                ("s5_pop1000",),
                ("s5_revisit_robust",),
                ("s5_revisit_restart", "s5_union"),
                ("s5_long_extra", "s5_revisit_ho16"),
                ("s5_frozen",))                # priority order (design memo Section 10); IPOP is not run (D125)


def _domain_sets(domain):
    from reach import grammar as G
    return [g for g in S5_MAIN_SETS if G.get(g).kind == domain]


def _s5_task(block, g, tname, t, config, seed, generations, timeout, enum=None, keys_out=None, weight=1.0, **meta):
    """One Stage 5 run of reach.tasks.s2_run, every scientific setting explicit (C6), the whole best tree printed
    (REQ-RUN-33); enum: the set's enumeration for the counter (`--count_distinct`, hard CPU limit x
    REVISIT_CPU_FACTOR, the soft limit unchanged as REQ-ENG-23); keys_out: the side file of the run's key set."""
    from reach.s2_lock import check_search_allowed
    from reach.tasks.s2_configs import CONFIGS
    check_search_allowed(g)
    rel = f"data/targets_s2/{g}/{tname}.json"
    sha = (ROOT / f"data/targets_s2/{g}/{tname}.sha256").read_text().split()[0]
    pop = CONFIGS[config]["pop_size"]
    argv = ["{python}", "-m", "reach.tasks.s2_run", "--set", g, "--targets", rel, "--target_id", t["target_id"],
            "--targets_sha256", sha, "--algorithm", "A", "--config", config, "--seed", str(seed),
            "--generations", str(generations), "--pop_size", str(pop), "--penalty", "scaled", "--stop_on", "exact",
            "--seed_leaves", "off", "--timeout", str(timeout), PRINT_TREE]
    cpu = timeout + 100
    if enum is not None:
        argv += ["--count_distinct", "--enum_dir", str(enum)]
        cpu *= REVISIT_CPU_FACTOR
    if keys_out is not None:
        argv += ["--keys_out", keys_out]
    return dict(key=f"{block}|{g}|{t['target_id']}|A|{config}|{generations}|{seed}", argv=argv, cpu_limit=cpu,
                weight=weight, meta=dict(block=block, set=g, target_id=t["target_id"], K=t["K_construction"],
                                         K_proven=t["K_proven"], algorithm="A", config=config,
                                         generations=generations, budget=pop * generations / 20_000, pop_size=pop,
                                         seed=seed, targets=tname, **meta))


def _enum(g, resolve=False):
    """The set's enumeration: this checkout's (a registered list hashes alike in every worktree) or, with resolve
    (pilots), the main worktree's when this checkout lacks the untracked level files."""
    from reach import revisit
    from reach.analysis.s2.phase1 import enum_dir
    return revisit.enum_root(g) if resolve else enum_dir(g)


def _phase_file(phase, name, root=None):
    """A phase's manifest.json or results.jsonl, or its gzipped copy (files over 50 MB are committed gzipped, D097;
    guardrail row 176); FileNotFoundError when neither exists."""
    import gzip
    from pathlib import Path
    d = Path(root or ROOT) / f"data/runs/{phase}"
    if (d / name).exists():
        return (d / name).read_text()
    return gzip.decompress((d / f"{name}.gz").read_bytes()).decode()


def _manifest_tasks(phase, root=None):
    return json.loads(_phase_file(phase, "manifest.json", root))["tasks"]


def phase_rows(phase, root=None):
    """Last row per key of a phase (read_results' rule: a truncated line is ignored), from results.jsonl or
    results.jsonl.gz; {} when the phase has neither."""
    try:
        text = _phase_file(phase, "results.jsonl", root)
    except FileNotFoundError:
        return {}
    rows = {}
    for line in text.splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[r["key"]] = r
    return rows


def low_levels(g):
    """The low-level rule (REQ-RUN-43): every proven level below the set's lowest calibration level with at least
    LOW_MIN_U functions and log2 U(<=K) of at least LOW_MIN_BITS."""
    from reach.analysis.s2.phase1 import enum_dir
    lv = _cal_levels()[g]
    lowest = min(lv["main"] + lv["budget"])
    stats = json.loads((enum_dir(g) / "counts.json").read_text())["levels"]
    return [st["K"] for st in stats if st["K"] < lowest and st["U"] >= LOW_MIN_U and math.log2(st["U_cum"]) >= LOW_MIN_BITS]


@phase("s5_targets", "STAGE 5 targets (D125): the low levels of the 14 main sets by the registered rule (every "
       "proven level below the lowest calibration level with U >= 5 and log2 U(<=K) >= 3), up to 30 targets each, "
       "seed TARGET_SEEDS['low'], into data/targets_s2/<set>/low.json. No search: drawn and checksummed before the "
       "Stage 5 registration. --workers 2.")
def s5_targets():
    from reach.seeds import TARGET_SEEDS
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.s2_targets", "--grammar", g, "--name", "low",
                              "--n", str(LOW_N), "--seed", str(TARGET_SEEDS["low"]),
                              "--levels", ",".join(map(str, low_levels(g)))],
                 cpu_limit=6 * 3600, meta=dict(grammar=g)) for g in S5_MAIN_SETS if low_levels(g)]


def _cal_targets(g, which, n=30):
    """The first n calibration targets of each of the set's main or budget levels, as `_cal` takes them."""
    lv = _cal_levels()[g]
    sizes = set(lv["main"] if which == "main" else lv["budget"])
    by = defaultdict(list)
    for t in json.loads((ROOT / f"data/targets_s2/{g}/calibration.json").read_text()):
        if t["K_construction"] in sizes:
            by[t["K_construction"]].append(t)
    return [t for K in sorted(by) for t in by[K][:n]]


def _wide_ok(g, t):
    """Review F1 (guardrail row 173): the wider-interval arm takes a target only if its normaliser on the wide
    interval's 100-point grid is finite and positive in double precision (`s2_configs.wide_ok`)."""
    from reach import grammar as G
    from reach.tasks.s2_configs import wide_ok
    gr = G.get(g)
    return wide_ok(gr, G.parse_expr(t["expression"], gr))


def wide_excluded():
    """The calibration targets of the real main sets that the wider-interval arm leaves out (sorted ids): H12-3 drops
    them from both arms (paired exclusion)."""
    return sorted(t["target_id"] for g in _domain_sets("real") for which in ("main", "budget")
                  for t in _cal_targets(g, which) if not _wide_ok(g, t))


def _h6c_e0(block, config, domain, count, keep=None):
    """H6c's E0 design (REQ-RUN-44): every s2_cal_a0 task (main levels) and every s2_cal_a16 task cut to its first
    200 generations (budget levels) on the domain's main sets, same targets and seeds, under `config`; meta names the
    registered base run each new run is paired with. keep(set, target): a filter (the wider interval's)."""
    from reach.s2_lock import check_stage1, check_stage5
    from reach.seeds import s2_seeds
    check_stage1()
    check_stage5()
    tasks = []
    for g in _domain_sets(domain):
        for which, src, gens in (("main", "cal_a0", 200), ("budget", "cal_a16", 3200)):
            for t in _cal_targets(g, which):
                if keep is not None and not keep(g, t):
                    continue
                tasks += [_s5_task(block, g, "calibration", t, config, s, 200, S2_TIMEOUT[1],
                                   enum=_enum(g) if count else None,
                                   pair_of=f"s2_{src}", pair_key=f"{src}|{g}|{t['target_id']}|A|base|{gens}|{s}")
                          for s in s2_seeds("main")]
    return tasks


@phase("s5_grid", "STAGE 5 (locked until Stage 5; arm 1a, H12-1): real GP with the error on the 100-point identity "
       "grid (FG) on H6c's E0 design of the seven real main sets, with the counter.")
def s5_grid():
    return _h6c_e0("s5_grid", "FG", "real", True)


@phase("s5_wide", "STAGE 5 (locked until Stage 5; wider interval, H12-3): real GP with a fresh 75-point batch on "
       "(1.5, 3.5) (WIDE; identity, targets and exactness on (2, 3)) on H6c's E0 design of the seven real main sets, "
       "counter; targets whose wide normaliser is not finite are left out (guardrail row 173).")
def s5_wide():
    return _h6c_e0("s5_wide", "WIDE", "real", True, keep=_wide_ok)


@phase("s5_boolsub", "STAGE 5 (locked until Stage 5; arm 2, H12-2): Boolean GP with the error on 24 of 32 rows drawn "
       "afresh each generation (BS24) on H6c's E0 design of the seven Boolean main sets, with the counter.")
def s5_boolsub():
    return _h6c_e0("s5_boolsub", "BS24", "bool", True)


@phase("s5_frozen", "STAGE 5 (locked until Stage 5; arm 1b, descriptive): real GP on generation 0's batch kept (FB) "
       "on H6c's E0 design of the seven real main sets, no counter.")
def s5_frozen():
    return _h6c_e0("s5_frozen", "FB", "real", False)


def selfree_levels(g):
    """S_d's calibration part (REQ-RUN-45): the set's calibration levels (main and budget) whose bits_sem is at most
    S5_CUT_BITS."""
    out = set()
    for t in json.loads((ROOT / f"data/targets_s2/{g}/calibration.json").read_text()):
        lv = _cal_levels()[g]
        if t["K_construction"] in lv["main"] + lv["budget"] and t["units"]["bits_sem"] <= S5_CUT_BITS:
            out.add(t["K_construction"])
    return sorted(out)


def _low_targets(g):
    p = ROOT / f"data/targets_s2/{g}/low.json"
    if low_levels(g) and not p.exists():
        raise FileNotFoundError(f"{p}: the low-level targets are not drawn yet (phase s5_targets)")
    return json.loads(p.read_text()) if p.exists() else []


def _selfree(block, config, with_cal):
    from reach.s2_lock import check_stage1, check_stage5
    from reach.seeds import s2_seeds
    check_stage1()
    check_stage5()
    tasks = []
    for g in S5_MAIN_SETS:
        if with_cal:
            keep = set(selfree_levels(g))
            for which in ("main", "budget"):
                tasks += [_s5_task(block, g, "calibration", t, config, s, 200, S2_TIMEOUT[1], enum=_enum(g))
                          for t in _cal_targets(g, which) if t["K_construction"] in keep for s in s2_seeds("main")]
        tasks += [_s5_task(block, g, "low", t, config, s, 200, S2_TIMEOUT[1], enum=_enum(g))
                  for t in _low_targets(g) for s in s2_seeds("main")]
    return tasks


@phase("s5_selfree", "STAGE 5 (locked until Stage 5; arm 3, H13): selection-free GP (SF) at E0 on S_d (calibration "
       "levels up to 18.29 bits and the low levels) of the 14 main sets, main seeds, with the counter.")
def s5_selfree():
    return _selfree("s5_selfree", "SF", True)


@phase("s5_low_base", "STAGE 5 (locked until Stage 5; arm 3c): base GP at E0 on the low levels of the 14 main sets, "
       "main seeds, with the counter.")
def s5_low_base():
    return _selfree("s5_low_base", "base", False)


def restart_targets():
    """The Stage 4 restart arm's targets (s4_restart's rule, unchanged): every proven level of the six real held-out
    sets' heldout.json and the restart.json levels; [(set, file, target)]."""
    out = []
    for g in REAL_HELDOUT:
        for tname in ("heldout", "restart"):
            if tname == "restart" and not restart_extra_levels(g):
                continue
            for t in json.loads((ROOT / f"data/targets_s2/{g}/{tname}.json").read_text()):
                if t["K_proven"] == t["K_construction"] and not t.get("constructed"):
                    out.append((g, tname, t))
    return out


@phase("s5_pop1000", "STAGE 5 (locked until Stage 5; arm 4a, H14): population 1,000 (R2) for 320 generations "
       "(16 x E0) on the restart arm's 55 levels x 30 targets x 5 new seeds (slice s5_pop1000).")
def s5_pop1000():
    from reach.s2_lock import check_stage5
    from reach.seeds import s2_seeds
    check_stage5()
    return [_s5_task("s5_pop1000", g, tname, t, "R2", s, 320, S5_LONG_TIMEOUT, weight=16)
            for g, tname, t in restart_targets() for s in s2_seeds("s5_pop1000")]


def reexec_tasks(src, prefix="revisit", select=lambda t: True, keys=False, resolve=False,
                 keys_phase="s5_revisit_restart"):
    """Re-execution (REQ-RUN-47, 48): one task per selected task of `src`'s manifest, unchanged except for its key
    (prefix|KEY), `--count_distinct --enum_dir DIR` (and `--keys_out` under S5_KEYS_DIR with keys), the hard CPU
    limit x REVISIT_CPU_FACTOR and meta `audit_of`, `audit_key` (as Stage 3's revisit_tasks)."""
    tasks = []
    for o in _manifest_tasks(src):
        if not select(o):
            continue
        if "reach.tasks.s2_run" not in o["argv"] or "--count_distinct" in o["argv"]:
            raise ValueError(f"{src} {o['key']}: not a plain tree-GP run task (reach.tasks.s2_run)")
        m = o["meta"]
        t = json.loads(json.dumps(o))
        t["key"] = f"{prefix}|{o['key']}"
        t["argv"] = o["argv"] + ["--count_distinct", "--enum_dir", str(_enum(m["set"], resolve))]
        if keys:
            t["argv"] += ["--keys_out", keys_path(m["target_id"], m["seed"], keys_phase)]
        t["cpu_limit"] = o["cpu_limit"] * REVISIT_CPU_FACTOR
        t["meta"] = dict(m, audit_of=src, audit_key=o["key"])
        tasks.append(t)
    return tasks


def keys_path(target_id, seed, phase="s5_revisit_restart"):
    return f"{S5_KEYS_DIR}/{phase}/{target_id}/{seed}.npz"


def _verify_reexec(ph):
    from reach.analysis.s2.audit import print_revisit_report, revisit_report
    rep = revisit_report(ROOT, ph.name, extra_ignore=("keys",), check_keys=True)
    print_revisit_report(rep)
    return 0 if rep["ok"] else 1


@phase("s5_revisit_robust", "STAGE 5 (locked until Stage 5; arm 6, H16): every s2_cal_robust task of R1 "
       "(population 25) and R2 (population 1,000) again plus the counter; every row must reproduce its original.",
       verify=_verify_reexec)
def s5_revisit_robust():
    from reach.s2_lock import check_stage5
    check_stage5()
    return reexec_tasks("s2_cal_robust", select=lambda t: t["meta"]["config"] in ("R1", "R2"))


@phase("s5_revisit_restart", "STAGE 5 (locked until Stage 5; arm 5a, H15): every s4_restart task again plus the "
       "counter and its key set (side files under data/keys_s5); every row must reproduce its original.",
       verify=_verify_reexec)
def s5_revisit_restart():
    from reach.s2_lock import check_stage5
    check_stage5()
    return reexec_tasks("s4_restart", keys=True)


@phase("s5_union", "STAGE 5 (locked until Stage 5; arm 5a): per restart target, the union histograms of its 64 key "
       "sets (functions covered by exactly m runs, per level, at 50 and 100 generations; all runs and runs without "
       "a hit). Run after s5_revisit_restart is complete. No search.")
def s5_union():
    from reach.s2_lock import check_stage5
    check_stage5()
    from reach.analysis.s2.phase1 import enum_dir
    return [dict(key=f"union|{g}|{t['target_id']}",
                 argv=["{python}", "-m", "reach.tasks.s5_union", "--set", g, "--target_id", t["target_id"],
                       "--rows", "data/runs/s5_revisit_restart/results.jsonl", "--enum_dir", str(enum_dir(g)),
                       "--n_runs", "64"],
                 cpu_limit=1800, weight=0.05, meta=dict(set=g, target_id=t["target_id"], K=t["K_construction"]))
            for g, _, t in restart_targets()]


def long_extra_levels(g):
    """Arm 7's levels (REQ-RUN-49): the restart arm's proven levels of the set without a one-run 16 x E0 comparator
    (not among the set's Stage 2 budget levels)."""
    budget = set(_ho_levels()[g]["budget"])
    return sorted({t["K_construction"] for h, _, t in restart_targets() if h == g} - budget)


def _verify_long_extra(ph):
    from reach.analysis.s2.audit import long_extra_report, print_long_extra_report
    rep = long_extra_report(ROOT, ph.name)
    print_long_extra_report(rep)
    return 0 if rep["ok"] else 1


@phase("s5_long_extra", "STAGE 5 (locked until Stage 5; arm 7, H11-2L): one base run of 3,200 generations "
       "(16 x E0) on the 25 restart levels without one, 30 targets x the 5 held-out seeds; no counter (D126). Each "
       "run's first 200 generations must reproduce its s2_ho_a0 row where one exists.",
       verify=_verify_long_extra)
def s5_long_extra():
    from reach.s2_lock import check_stage5
    from reach.seeds import s2_seeds
    check_stage5()
    seeds, lv, tasks = s2_seeds("heldout"), _ho_levels(), []
    for g, tname, t in restart_targets():
        if t["K_construction"] not in long_extra_levels(g):
            continue
        main = t["K_construction"] in lv[g]["main"] and tname == "heldout"
        for s in seeds:
            extra = dict(prefix_of="s2_ho_a0", prefix_key=f"ho_a0|{g}|{t['target_id']}|A|base|200|{s}") if main else {}
            tasks.append(_s5_task("s5_long_extra", g, tname, t, "base", s, 3200, S5_LONG_TIMEOUT,
                                  weight=16, **extra))       # no counter (D126: H11-2L alone needs no counts)
    return tasks


@phase("s5_revisit_ho16", "STAGE 5 (locked until Stage 5; 7c): every s2_ho_a16 task of a real held-out set with the "
       "first held-out seed again plus the counter (900); every row must reproduce its original.",
       verify=_verify_reexec)
def s5_revisit_ho16():
    from reach.s2_lock import check_stage5
    from reach.seeds import s2_seeds
    check_stage5()
    first = s2_seeds("heldout")[0]
    return reexec_tasks("s2_ho_a16", select=lambda t: t["meta"]["set"] in REAL_HELDOUT and t["meta"]["seed"] == first)


# ---------------------------------------------------------------- Stage 5 pilots (never confirmatory; not locked)
def _pilot_s5(block, sets, configs, seeds, generations, timeout, count=True, levels=None):
    tasks = []
    for g in sets:
        for t in _pilot_targets(g):
            if levels is not None and not levels(g, t):
                continue
            tasks += [_s5_task(block, g, "pilot", t, c, s, generations, timeout,
                               enum=_enum(g, resolve=True) if count else None) for c in configs for s in seeds]
    return _spread(tasks)


@phase("s5_pilot_grid", "Stage 5 pilot (never confirmatory): FG at E0 on every pilot target of the seven real main "
       "sets x pilot seeds 0-1, with the counter (paired with s2_revisit_pilot2).")
def s5_pilot_grid():
    from reach.seeds import s2_seeds
    return _pilot_s5("s5_pilot_grid", _domain_sets("real"), ("FG",), s2_seeds("pilot")[:2], 200, S2_TIMEOUT[1])


@phase("s5_pilot_wide", "Stage 5 pilot (never confirmatory): WIDE at E0 on every pilot target of the seven real main "
       "sets x pilot seeds 0-1, with the counter.")
def s5_pilot_wide():
    from reach.seeds import s2_seeds
    return _pilot_s5("s5_pilot_wide", _domain_sets("real"), ("WIDE",), s2_seeds("pilot")[:2], 200, S2_TIMEOUT[1],
                     levels=_wide_ok)


@phase("s5_pilot_boolsub", "Stage 5 pilot (never confirmatory): BS24 at E0 on every pilot target of the seven Boolean "
       "main sets x pilot seeds 0-1, with the counter.")
def s5_pilot_boolsub():
    from reach.seeds import s2_seeds
    return _pilot_s5("s5_pilot_boolsub", _domain_sets("bool"), ("BS24",), s2_seeds("pilot")[:2], 200,
                     S2_TIMEOUT[1])


@phase("s5_pilot_selfree", "Stage 5 pilot (never confirmatory): SF and base at E0 on the pilot targets of the 14 main "
       "sets up to 18.29 bits x pilot seeds 0-1, with the counter.")
def s5_pilot_selfree():
    from reach.seeds import s2_seeds
    return _pilot_s5("s5_pilot_selfree", S5_MAIN_SETS, ("SF", "base"), s2_seeds("pilot")[:2], 200, S2_TIMEOUT[1],
                     levels=lambda g, t: t["units"]["bits_sem"] <= S5_CUT_BITS)


@phase("s5_pilot_pop1000", "Stage 5 pilot (never confirmatory): R2 for 320 generations (16 x E0) on the pilot targets "
       "of F-EML and R-ELEM4 x pilot seeds 8-9 (cost and estimability).")
def s5_pilot_pop1000():
    from reach.seeds import s2_seeds
    return _pilot_s5("s5_pilot_pop1000", ["F-EML", "R-ELEM4"], ("R2",), s2_seeds("pilot")[8:10], 320,
                     S5_LONG_TIMEOUT, count=False)


@phase("s5_pilot_keys", "Stage 5 pilot (never confirmatory): 100-generation base runs with the counter and key sets "
       "on the pilot targets of F-EML and R-ARITH4 x 16 seeds of the slice s5_pilot (union check, cost).")
def s5_pilot_keys():
    from reach.seeds import s2_seeds
    tasks = []
    for g in ("F-EML", "R-ARITH4"):
        for t in _pilot_targets(g):
            tasks += [_s5_task("s5_pilot_keys", g, "pilot", t, "base", s, RESTART_GENERATIONS, RESTART_TIMEOUT,
                               enum=_enum(g, resolve=True), keys_out=keys_path(t["target_id"], s, "s5_pilot_keys"))
                      for s in s2_seeds("s5_pilot")[:16]]
    return _spread(tasks)


@phase("s5_pilot_robust", "Stage 5 pilot (never confirmatory): s2_pilot_robust's R2 tasks again with the counter, and "
       "R1 at 4 x E0 on the same pilot targets and seeds (pilot seeds 0-1), with the counter.")
def s5_pilot_robust():
    from reach.seeds import s2_seeds
    again = reexec_tasks("s2_pilot_robust", select=lambda t: t["meta"]["config"] == "R2", resolve=True)
    r1 = [_s5_task("s5_pilot_robust", t["meta"]["set"], "pilot", dict(target_id=t["meta"]["target_id"],
                                                                      K_construction=t["meta"]["K"],
                                                                      K_proven=t["meta"]["K_proven"]),
                   "R1", t["meta"]["seed"], 3200, S2_TIMEOUT[4], enum=_enum(t["meta"]["set"], resolve=True))
          for t in again]
    assert {x["meta"]["seed"] for x in r1} <= set(s2_seeds("pilot")[:2])
    return _spread(again + r1)


# ---------------------------------------------------------------- QA: re-scoring of the held-out and restart runs
# D068's check 2 (independent evaluators) covered the calibration phases only (data/analysis/audit_labels_
# calibration_all.json, D069; the big trees by s2_audit_bigtrees, D091), while the supplement says every exact run
# entering a confirmatory test was re-scored (guardrail row 171). Every Stage 2 and Stage 4 tree-GP task printed the
# whole tree (REQ-RUN-33), so no rerun is needed: the records are re-scored directly. Not Stage 5; not locked.
RESCORE_PHASES = ("s2_ho_a0", "s2_ho_a16", "s2_ho_bench", "s2_ho_operon", "s4_restart")   # bench: row 177
RESCORE_FAILURES = 300          # sampled failures per phase, as the calibration check, split over its sets


def _verify_labels_ho(ph):
    from reach.analysis.s2.audit import heldout_labels_report, print_heldout_labels_report
    rep = heldout_labels_report(ROOT, ph.name)
    print_heldout_labels_report(rep)
    return 0 if rep["ok"] else 1


@phase("s2_audit_labels_ho", "QA (D068 check 2 extended, D125; not a test of any hypothesis): every exact run (and "
       "300 sampled failures per phase) of s2_ho_a0, s2_ho_a16, s2_ho_operon and s4_restart re-scored by the "
       "independent evaluators of reach.analysis.s2.audit_labels; one task per phase and set. No search.",
       verify=_verify_labels_ho)
def s2_audit_labels_ho():
    tasks = []
    for ph in RESCORE_PHASES:
        sets = sorted({t["meta"]["set"] for t in _manifest_tasks(ph)})
        n_fail = RESCORE_FAILURES // len(sets)
        tasks += [dict(key=f"labels|{ph}|{g}", argv=["{python}", "-m", "reach.tasks.s2_rescore", "--phase", ph, "--set",
                                                     g, "--n_fail", str(n_fail)],
                       cpu_limit=6 * 3600, weight=10 if ph == "s4_restart" else 1, meta=dict(phase=ph, set=g))
                  for g in sets]
    return tasks


@phase("s2_audit_labels_s5", "QA (REQ-RUN-51, guardrail row 209; not a test of any hypothesis): every exact run (and "
       "300 sampled failures) of Stage 5's s5_long_extra re-scored by the independent evaluators, as "
       "s2_audit_labels_ho; one task per set; run before the H11-2L analysis. No search.", verify=_verify_labels_ho)
def s2_audit_labels_s5():
    sets = sorted({t["meta"]["set"] for t in _manifest_tasks("s5_long_extra")})
    return [dict(key=f"labels|s5_long_extra|{g}", argv=["{python}", "-m", "reach.tasks.s2_rescore", "--phase",
                                                         "s5_long_extra", "--set", g, "--n_fail",
                                                         str(RESCORE_FAILURES // len(sets))],
                 cpu_limit=6 * 3600, weight=1, meta=dict(phase="s5_long_extra", set=g)) for g in sets]


@phase("s5_pilot_restart10","Stage 5 pilot (QA, never confirmatory; the outcomes are known): ten s4_restart tasks, "
       "drawn as D068's reproduction sample (seed 20261022), again with the counter and key sets (side files under "
       "data/keys_s5/s5_pilot_restart10); every row must reproduce its original.", verify=_verify_reexec)
def s5_pilot_restart10():
    from reach.analysis.s2.audit import sample_keys
    chosen = set(sample_keys(phase_rows("s4_restart"), 10))
    return reexec_tasks("s4_restart", select=lambda t: t["key"] in chosen, keys=True, resolve=True,
                        keys_phase="s5_pilot_restart10")

"""A small synthetic study root for the Stage 5 driver's tests (tests/test_stage5_driver.py, test_stage5_pilot.py):
target files with checksums, the manifests and rows of every phase the driver reads, counted rows (rv), key-set side
files and union histograms. Deterministic; every probability is a logistic in bits so that each K50 is finite."""
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np

PREFIXES = (5_000, 10_000, 20_000, 40_000, 80_000, 160_000, 320_000)
U = {K: 3 * 2 ** K for K in range(1, 19)}            # U_K
TOP = 10                                              # the lookup bound
REAL, BOOL, HELD = ("R-X", "R-Y"), ("BOOL-X", "BOOL-Y"), ("H-A", "H-B")
ROB_REAL, ROB_BOOL = ("R-X",), ("BOOL-X",)
LEVELS = {"main": [9, 11], "budget": [7, 16]}         # 16: above the 18.29-bit cut (left out of S_d)
LOW = [3, 5]
HELD_BUDGET, HELD_EXTRA = [6, 8], [7, 9]
SEEDS = [101, 102, 103]
N_PER = 4
RESTART_N = 64


def bits(K):
    return math.log2(sum(U[k] for k in range(1, K + 1)))


def _u(*parts):
    return random.Random("|".join(map(str, parts))).random()


def _logit(x):
    return 1 / (1 + math.exp(-x))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_targets(root, g, name, levels, n=N_PER, constructed=()):
    d = Path(root) / "data/targets_s2" / g
    d.mkdir(parents=True, exist_ok=True)
    ts = [dict(target_id=f"{g}_K{K:02d}_{i:02d}", K_construction=K, K_proven=K - 2 if K in constructed else K,
               constructed=K in constructed, expression="x", units=dict(bits_sem=bits(K), nodes=float(K)))
          for K in levels for i in range(n)]
    p = d / f"{name}.json"
    p.write_text(json.dumps(ts))
    (d / f"{name}.sha256").write_text(f"{sha(p)}  {p.name}\n")
    return ts


def snap(n, scale=600.0):
    lv = [0] + [min(U[k], int(U[k] * min(1.0, n / (U[k] * scale)))) for k in range(1, TOP + 1)]
    return dict(n=n, d=sum(lv) + n // 50, undef=0, lv=lv, lv_ok=list(lv))


def rv_for(row, g, scale=600.0):
    n_end = row["pop_size"] * (row["gens_reached"] + (row["terminal_state"] == "converged"))
    scale = scale * (row["pop_size"] / 100) ** 0.3          # coverage per evaluation falls with the population
    return dict(prefixes={str(p): snap(p, scale) for p in PREFIXES if p <= n_end and p % row["pop_size"] == 0},
                stop=snap(n_end, scale), cpu_seconds=0.01, target_first=n_end if row.get("exact") else None,
                target_level=row["K"], lookup=dict(sha256=f"lookup-{g}", top=TOP))


def run_row(block, key, g, t, seed, config, gens, pop, k50, slope=1.5, tsha=None, extra=None, domain="real"):
    """A run whose success within `gens` generations has probability logistic(slope (k50 - bits)); its generation of
    first exact hit is spread over the run."""
    b = t["units"]["bits_sem"]
    p = _logit(slope * (k50 - b))
    u = _u(block, g, t["target_id"], seed, config)
    hit = u < p
    ge = int(min(gens, 200) * u / p * 0.99) if hit else None     # every hit within the first 200 generations
    row = dict(key=key, block=block, set=g, target_id=t["target_id"], seed=seed, K=t["K_construction"],
               K_construction=t["K_construction"], K_proven=t["K_proven"], algorithm="A", config=config,
               generations=gens, pop_size=pop, budget_E=gens * pop, gens_reached=ge if hit else gens,
               gen_exact=ge, exact=bool(hit), terminal_state="converged" if hit else "budget_exhausted",
               units=dict(t["units"]), targets_sha256=tsha, domain=domain, cpu_s=0.5 + u, cpu_seconds=0.4 + u)
    if extra:
        row.update(extra)
    return row


def write_phase(root, phase, rows, tasks=None):
    d = Path(root) / "data/runs" / phase
    d.mkdir(parents=True, exist_ok=True)
    if tasks is None:
        tasks = [dict(key=r["key"], argv=["{python}", "-m", "reach.tasks.s2_run", "--seed", str(r.get("seed"))],
                      cpu_limit=1000, meta={k: r[k] for k in ("set", "target_id", "seed") if k in r}) for r in rows]
    (d / "manifest.json").write_text(json.dumps(dict(phase=phase, tasks=tasks)))
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tasks


def reexec(root, phase, src_phase, src_rows, keys=False, scale=600.0, prefix="revisit"):
    """A re-execution phase: every source row again with rv (and a side file with keys), tasks naming audit_of."""
    rows, tasks = [], []
    for o in src_rows:
        r = dict(o, key=f"{prefix}|{o['key']}", rv=rv_for(o, o["set"], scale))
        argv = ["{python}", "-m", "reach.tasks.s2_run", "--count_distinct"]
        if keys:
            f = f"data/keys_s5/{phase}/{o['target_id']}/{o['seed']}.npz"
            path = Path(root) / f
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"keys {o['key']}".encode())
            r["keys"] = dict(file=f, n=1, sha256=sha(path))
            argv += ["--keys_out", f]
        rows.append(r)
        tasks.append(dict(key=r["key"], argv=argv, cpu_limit=2000,
                          meta=dict(set=o["set"], target_id=o["target_id"], seed=o["seed"], audit_of=src_phase,
                                    audit_key=o["key"])))
    write_phase(root, phase, rows, tasks)
    return rows


def lookup_levels():
    """The synthetic lookup: the level of every index (levels 1..TOP, U_K indices each)."""
    return np.concatenate([np.full(U[k], k) for k in range(1, TOP + 1)]).astype(np.int64)


def key_sets(g, t, n_runs, rows, scale=600.0):
    """Per run (idx, first, hit): a random subset of each level, about the share `snap` gives at the run's end."""
    from reach.tasks.s5_union import histograms
    lv = lookup_levels()
    start = np.concatenate([[0], np.cumsum([U[k] for k in range(1, TOP + 1)])])
    out = []
    for i, r in enumerate(rows):
        rng = np.random.default_rng(int(hashlib.sha256(f"{g}|{t['target_id']}|{i}".encode()).hexdigest()[:8], 16))
        n_end = r["pop_size"] * (r["gens_reached"] + (r["terminal_state"] == "converged"))
        idx, first = [], []
        for k in range(1, TOP + 1):
            m = int(U[k] * min(1.0, n_end / (U[k] * scale)) * rng.uniform(0.5, 1.0))
            pick = rng.choice(U[k], m, replace=False) + start[k - 1]
            idx.append(pick)
            first.append(rng.integers(1, n_end + 1, m))
        hit = (r["gen_exact"] + 1) * r["pop_size"] if r["exact"] else None
        out.append((np.concatenate(idx).astype(np.int64), np.concatenate(first).astype(np.int64), hit))
    return out, histograms(out, lv, TOP)


def make_root(root, scale=600.0, wide_excluded=("R-X_K09_00",)):
    """Every phase the confirmatory driver reads, for REAL, BOOL (H12, H13), HELD (H14, H15, H11-2L) and the robust
    sets (H16). Returns the driver configuration and the level counts."""
    root = Path(root)
    (root / "data/registration").mkdir(parents=True, exist_ok=True)
    lv = {g: dict(LEVELS) for g in REAL + BOOL}
    (root / "data/registration/levels.json").write_text(json.dumps(lv))
    tg, tsha = {}, {}
    for g in REAL + BOOL:
        tg[g] = write_targets(root, g, "calibration", LEVELS["main"] + LEVELS["budget"],
                              constructed=(16,) if g in BOOL else ())
        tg[g + "|low"] = write_targets(root, g, "low", LOW)
        tsha[g] = sha(root / f"data/targets_s2/{g}/calibration.json")
        tsha[g + "|low"] = sha(root / f"data/targets_s2/{g}/low.json")
    for g in HELD:
        tg[g] = write_targets(root, g, "heldout", sorted(HELD_BUDGET + HELD_EXTRA))
        tsha[g] = sha(root / f"data/targets_s2/{g}/heldout.json")
    dom = {g: ("bool" if g in BOOL else "real") for g in REAL + BOOL + HELD}
    k0 = {g: (12.0 if g in BOOL else 10.0) for g in REAL + BOOL}

    def cal_rows(block, prefix, config, which, gens, k50, extra=lambda g, t, s: None, pop=100):
        rows = []
        for g in REAL + BOOL:
            for t in tg[g]:
                if t["K_construction"] not in LEVELS[which]:
                    continue
                for s in SEEDS:
                    rows.append(run_row(block, f"{prefix}|{g}|{t['target_id']}|A|{config}|{gens}|{s}", g, t, s, config,
                                        gens, pop, k50(g), tsha=tsha[g], extra=extra(g, t, s), domain=dom[g]))
        return rows
    a0 = cal_rows("s2_cal_a0", "cal_a0", "base", "main", 200, lambda g: k0[g])
    a16 = cal_rows("s2_cal_a16", "cal_a16", "base", "budget", 3200, lambda g: k0[g] + 1)
    write_phase(root, "s2_cal_a0", a0)
    write_phase(root, "s2_cal_a16", a16)
    reexec(root, "s2_revisit_a0", "s2_cal_a0", a0, scale=scale)
    rev16 = reexec(root, "s2_revisit_a16", "s2_cal_a16", a16, scale=scale)

    def paired(block, config, shift, sets, keep=lambda t: True):
        rows = []
        for src, gens in ((a0, 200), (a16, 3200)):
            for o in src:
                if o["set"] not in sets or not keep(o["target_id"]):
                    continue
                t = next(x for x in tg[o["set"]] if x["target_id"] == o["target_id"])
                rows.append(run_row(block, f"{block}|{o['set']}|{o['target_id']}|A|{config}|200|{o['seed']}",
                                    o["set"], t, o["seed"], config, 200, 100, k0[o["set"]] + shift, tsha=tsha[o["set"]],
                                    extra=dict(pair_of=o["block"], pair_key=o["key"]), domain=dom[o["set"]]))
        return rows
    write_phase(root, "s5_grid", paired("s5_grid", "FG", 0.3, REAL))
    write_phase(root, "s5_wide", paired("s5_wide", "WIDE", -0.2, REAL, lambda t: t not in wide_excluded))
    write_phase(root, "s5_boolsub", paired("s5_boolsub", "BS24", -0.5, BOOL))
    # H13: SF on S_d (calibration levels up to the cut, and the low levels); base on the low levels
    sf, lb = [], []
    for g in REAL + BOOL:
        for t in tg[g] + tg[g + "|low"]:
            low = t["K_construction"] in LOW
            if not low and t["units"]["bits_sem"] > math.log2(320_000):
                continue
            for s in SEEDS:
                sf.append(run_row("s5_selfree", f"s5_selfree|{g}|{t['target_id']}|A|SF|200|{s}", g, t, s, "SF", 200,
                                  100, 8.0 if g in REAL else 8.3, tsha=tsha[g + "|low" if low else g], domain=dom[g]))
                if low:
                    lb.append(run_row("s5_low_base", f"s5_low_base|{g}|{t['target_id']}|A|base|200|{s}", g, t, s,
                                      "base", 200, 100, k0[g], tsha=tsha[g + "|low"], domain=dom[g]))
    write_phase(root, "s5_selfree", sf)
    write_phase(root, "s5_low_base", lb)
    # held-out block: 64 restart runs per target, union histograms, long runs, population 1,000
    rs, unions, ho16, extra, r2 = [], [], [], [], []
    for g in HELD:
        for t in tg[g]:
            runs = [run_row("s4_restart", f"s4_restart|{g}|{t['target_id']}|A|base|100|{s}", g, t, s, "base", 100, 100,
                            9.5, slope=2.0, tsha=tsha[g]) for s in range(1000, 1000 + RESTART_N)]
            rs += runs
            _, hist = key_sets(g, t, RESTART_N, runs, scale)
            unions.append(dict(key=f"union|{g}|{t['target_id']}", set=g, target_id=t["target_id"], n_runs=RESTART_N,
                               terminal_state="completed", lookup=dict(sha256=f"lookup-{g}", top=TOP), hist=hist,
                               cpu_s=0.2))
            for s in SEEDS:
                if t["K_construction"] in HELD_BUDGET:
                    ho16.append(run_row("s2_ho_a16", f"ho_a16|{g}|{t['target_id']}|A|base|3200|{s}", g, t, s, "base",
                                        3200, 100, 9.0, tsha=tsha[g]))
                else:
                    r = run_row("s5_long_extra", f"s5_long_extra|{g}|{t['target_id']}|A|base|3200|{s}", g, t, s,
                                "base", 3200, 100, 9.0, tsha=tsha[g])
                    if s == SEEDS[0]:
                        r["rv"] = rv_for(r, g, scale)
                    extra.append(r)
                r2.append(run_row("s5_pop1000", f"s5_pop1000|{g}|{t['target_id']}|A|R2|320|{s}", g, t, s, "R2", 320,
                                  1000, 10.0, tsha=tsha[g]))
    write_phase(root, "s4_restart", rs)
    reexec(root, "s5_revisit_restart", "s4_restart", rs, keys=True, scale=scale)
    write_phase(root, "s5_union", unions)
    write_phase(root, "s2_ho_a16", ho16)
    write_phase(root, "s5_long_extra", extra)
    reexec(root, "s5_revisit_ho16", "s2_ho_a16", [r for r in ho16 if r["seed"] == SEEDS[0]], scale=scale)
    write_phase(root, "s5_pop1000", r2)
    # H16: s2_cal_robust R1 (population 25) and R2 (1,000) on the robust sets' budget levels, re-executed
    rob = []
    for g in ROB_REAL + ROB_BOOL:
        for t in tg[g]:
            if t["K_construction"] not in LEVELS["budget"]:
                continue
            for s in SEEDS:
                for c, gens, pop in (("R1", 3200, 25), ("R2", 80, 1000)):
                    rob.append(run_row("s2_cal_robust", f"cal_robust|{g}|{t['target_id']}|A|{c}|{gens}|{s}", g, t, s,
                                       c, gens, pop, k0[g], tsha=tsha[g], domain=dom[g]))
    write_phase(root, "s2_cal_robust", rob)
    reexec(root, "s5_revisit_robust", "s2_cal_robust", rob, scale=scale)
    cfg = dict(label="SYNTHETIC", out="stage5", confirmatory=True, instrument=False,
               phases=dict(grid="s5_grid", wide="s5_wide", boolsub="s5_boolsub", frozen="s5_frozen",
                           selfree="s5_selfree", low_base="s5_low_base", pop1000="s5_pop1000",
                           revisit_restart="s5_revisit_restart", union="s5_union", revisit_robust="s5_revisit_robust",
                           long_extra="s5_long_extra", revisit_ho16="s5_revisit_ho16", base_a0="s2_cal_a0",
                           base_a16="s2_cal_a16", stage3_a0="s2_revisit_a0", stage3_a16="s2_revisit_a16",
                           restart="s4_restart", one16="s2_ho_a16", robust="s2_cal_robust"),
               real=REAL, boolean=BOOL, heldout=HELD, robust_real=ROB_REAL, robust_bool=ROB_BOOL,
               wide_excluded=tuple(wide_excluded))
    levels = {g: dict(U) for g in REAL + BOOL + HELD}
    return cfg, levels, dict(rev16=rev16)


# ---------------------------------------------------------------- the pilot phases (tests/test_stage5_pilot.py)
P_REAL, P_BOOL, P_HELD = ("F-EML", "R-ARITH4", "R-ELEM4"), ("BOOL-AONX-n5", "BOOL-NAND-n5"), ("R-ELEM6",)
P_LEVELS = [3, 5, 7, 9, 11]
P_SEEDS = [1, 2, 3]


def keyed_runs(root, phase, g, t, seeds, gens=100, k50=8.0, tsha=None, scale=600.0):
    """Runs with real side files (idx, first) whose counter counts are the side file's own per-level counts."""
    lv = lookup_levels()
    start = np.concatenate([[0], np.cumsum([U[k] for k in range(1, TOP + 1)])])
    rows = []
    for s in seeds:
        r = run_row(phase, f"{phase}|{g}|{t['target_id']}|A|base|{gens}|{s}", g, t, s, "base", gens, 100, k50,
                    tsha=tsha)
        n_end = r["pop_size"] * (r["gens_reached"] + (r["terminal_state"] == "converged"))
        rng = np.random.default_rng(int(hashlib.sha256(r["key"].encode()).hexdigest()[:8], 16))
        idx, first = [], []
        for k in range(1, TOP + 1):
            m = int(U[k] * min(1.0, n_end / (U[k] * scale)) * rng.uniform(0.5, 1.0))
            idx.append(rng.choice(U[k], m, replace=False) + start[k - 1])
            first.append(rng.integers(1, n_end + 1, m))
        idx, first = np.concatenate(idx).astype(np.uint32), np.concatenate(first).astype(np.uint32)

        def counts(sel):
            return [0] + np.bincount(lv[sel.astype(np.int64)], minlength=TOP + 1)[1:TOP + 1].tolist()
        pre = {str(p): dict(n=p, d=0, undef=0, lv=counts(idx[first <= p]), lv_ok=counts(idx[first <= p]))
               for p in PREFIXES if p <= n_end}
        r["rv"] = dict(prefixes=pre, stop=dict(n=n_end, d=0, undef=0, lv=counts(idx), lv_ok=counts(idx)),
                       cpu_seconds=0.2, target_first=None, target_level=t["K_construction"],
                       lookup=dict(sha256=f"lookup-{g}", top=TOP))
        f = f"data/keys_s5/{phase}/{t['target_id']}/{s}.npz"
        p = Path(root) / f
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, idx=idx, first=first)
        r["keys"] = dict(file=f, n=int(len(idx)), sha256=sha(p))
        rows.append(r)
    return rows


def make_pilot_root(root, drop=5):
    """The eight pilot phases (s5_pilot_grid with `drop` rows missing: partial), their paired pilot runs and the
    registered Stage 4 values the approximations read."""
    root = Path(root)
    tg, tsha = {}, {}
    for g in P_REAL + P_BOOL:
        tg[g] = write_targets(root, g, "pilot", P_LEVELS, constructed=(11,) if g in P_BOOL else ())
        tsha[g] = sha(root / f"data/targets_s2/{g}/pilot.json")
    for g in P_HELD:
        tg[g] = write_targets(root, g, "heldout", [6, 7, 8])
        tsha[g] = sha(root / f"data/targets_s2/{g}/heldout.json")
    dom = {g: ("bool" if g in P_BOOL else "real") for g in P_REAL + P_BOOL + P_HELD}
    k0 = {g: (9.0 if g in P_BOOL else 8.0) for g in P_REAL + P_BOOL}

    def rows_for(block, config, sets, seeds, gens, pop, shift, rv=False, extra=lambda r: None):
        out = []
        for g in sets:
            for t in tg[g]:
                for s in seeds:
                    r = run_row(block, f"{block}|{g}|{t['target_id']}|A|{config}|{gens}|{s}", g, t, s, config, gens,
                                pop, k0.get(g, 8.0) + shift, tsha=tsha[g], domain=dom[g])
                    if rv:
                        r["rv"] = rv_for(r, g)
                    extra(r)
                    out.append(r)
        return out
    base = rows_for("s2_revisit_pilot2", "base", P_REAL + P_BOOL, P_SEEDS, 800, 100, 0.0, rv=True)
    write_phase(root, "s2_revisit_pilot2", base)
    grid = rows_for("s5_pilot_grid", "FG", P_REAL, P_SEEDS[:2], 200, 100, 0.3, rv=True)
    tasks = write_phase(root, "s5_pilot_grid", grid)
    write_phase(root, "s5_pilot_grid", grid[:-drop] if drop else grid, tasks)
    write_phase(root, "s5_pilot_wide", rows_for("s5_pilot_wide", "WIDE", P_REAL, P_SEEDS[:2], 200, 100, -0.2, rv=True))
    write_phase(root, "s5_pilot_boolsub", rows_for("s5_pilot_boolsub", "BS24", P_BOOL, P_SEEDS[:2], 200, 100, -0.4,
                                                   rv=True))

    def target_bias(r):                               # with selection, the search depends on the target (the plant)
        for snap_ in list(r["rv"]["prefixes"].values()) + [r["rv"]["stop"]]:
            snap_["d"] += int(r["target_id"][-2:]) + 1
    sf = rows_for("s5_pilot_selfree", "SF", P_REAL + P_BOOL, P_SEEDS[:2], 200, 100, -1.5, rv=True)
    sb = rows_for("s5_pilot_selfree", "base", P_REAL + P_BOOL, P_SEEDS[:2], 200, 100, 0.0, rv=True, extra=target_bias)
    write_phase(root, "s5_pilot_selfree", sf + sb)
    write_phase(root, "s5_pilot_pop1000", rows_for("s5_pilot_pop1000", "R2", ("F-EML", "R-ELEM4"), [8, 9], 320, 1000,
                                                   1.0))
    keys = [r for g in ("F-EML", "R-ARITH4") for t in tg[g]
            for r in keyed_runs(root, "s5_pilot_keys", g, t, range(500, 516), tsha=tsha[g])]
    write_phase(root, "s5_pilot_keys", keys)
    rob_sets = ("F-EML", "R-ELEM4", "BOOL-AONX-n5", "BOOL-NAND-n5")
    orig = rows_for("s2_pilot_robust", "R2", rob_sets, P_SEEDS[:2], 80, 1000, 0.0)
    orig4 = rows_for("s2_pilot_robust", "R4", rob_sets, P_SEEDS[:2], 800, 100, 0.0)
    write_phase(root, "s2_pilot_robust", orig + orig4)
    again = reexec(root, "_tmp_robust", "s2_pilot_robust", orig)
    r1 = rows_for("s5_pilot_robust", "R1", rob_sets, P_SEEDS[:2], 3200, 25, 0.0, rv=True)
    tasks = json.loads((Path(root) / "data/runs/_tmp_robust/manifest.json").read_text())["tasks"]
    tasks += [dict(key=r["key"], argv=[], meta=dict(set=r["set"], target_id=r["target_id"], seed=r["seed"]))
              for r in r1]
    write_phase(root, "s5_pilot_robust", again + r1, tasks)
    rs = [run_row("s4_restart", f"s4_restart|R-ELEM6|{t['target_id']}|A|base|100|{s}", "R-ELEM6", t, s, "base", 100,
                  100, 9.0, tsha=tsha["R-ELEM6"]) for t in tg["R-ELEM6"] for s in range(700, 704)]
    write_phase(root, "s4_restart", rs)
    reexec(root, "s5_pilot_restart10", "s4_restart", rs[:10], keys=True)
    (root / "data/analysis").mkdir(parents=True, exist_ok=True)
    sets = ("R-ARITH", "R-ELEM6")
    (root / "data/analysis/h11.json").write_text(json.dumps(dict(h11=dict(
        tests={"H11-1": dict(estimate=-0.43, lo90=-0.66, hi90=-0.18)},
        per_set={g: dict(gain=g_) for g, g_ in zip(sets, (6.5, 5.4))},
        per_set_gain_tests={g: dict(lo90=g_ - 1.0, hi90=g_ + 1.0) for g, g_ in zip(sets, (6.5, 5.4))}))))
    (root / "data/analysis/revision_exploratory.json").write_text(json.dumps(dict(h11_common=dict(per_set={
        g: dict(gain=g_) for g, g_ in zip(sets, (2.4, 4.6))}))))
    levels = {g: dict(U) for g in P_REAL + P_BOOL + P_HELD}

    def lookup(g):
        return lookup_levels(), TOP, f"lookup-{g}"
    return levels, lookup

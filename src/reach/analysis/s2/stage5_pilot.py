"""Stage 5 pilot report (REQ-ANA-129, REQ-ANA-130; docs/design/stage5_analysis.md). NEVER CONFIRMATORY: pilot targets
and pilot seeds only, except ten restart re-executions whose outcomes are known.

    python -m reach.analysis.s2.stage5_report --pilot [--boot B] [--root DIR] [--out DIR]

The same statistics as the registered tests (`stage5_stats`) on the s5_pilot_* phases against their paired pilot runs:
s2_revisit_pilot2 read at E0 (the counted base pilot runs, seeds 0 to 2) for the fitness arms and the twin;
s5_pilot_selfree's own base runs for T; s2_pilot_robust's originals for the re-executions. The phases may be running:
each contrast is restricted to the (set, target, seed) runs present in both arms, an incomplete phase is reported and
the report labelled PARTIAL, nothing is refused for incompleteness (a target file off its checksum or a key set
against another lookup is still refused). Power: the pilot's paired bootstrap SD scaled by sqrt(n_pilot /
n_registered), Stage 3's rule (`h6d.power`; the one-sided analogue for non-inferiority and sign tests), the pilot's
estimate taken as true. Writes data/analysis/stage5_pilot.json and docs/status/stage5_pilot.md.
"""
import hashlib
import itertools
import math
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from reach.analysis.s2 import h6d, stage5
from reach.analysis.s2 import stage5_report as D
from reach.analysis.s2 import stage5_stats as S
from reach.analysis.s2.load import read_phase

PHASES = ("s5_pilot_grid", "s5_pilot_wide", "s5_pilot_boolsub", "s5_pilot_selfree", "s5_pilot_pop1000",
          "s5_pilot_keys", "s5_pilot_robust", "s5_pilot_restart10")
BASE = "s2_revisit_pilot2"            # counted base pilot runs, 800 generations, pilot seeds 0 to 2 (read at E0)
ROBUST_SRC = "s2_pilot_robust"
KEYS_SETS, POP_SETS = ("F-EML", "R-ARITH4"), ("F-EML", "R-ELEM4")
ROB_REAL, ROB_BOOL = ("F-EML", "R-ELEM4"), ("BOOL-AONX-n5", "BOOL-NAND-n5")
KEYS_R = (1, 2, 4, 8, 16)
# The matching confirmatory phase and its registered runs (the form's Sample size), for the CPU projection.
PROJECT = {("s5_pilot_grid", None): ("s5_grid", 10_500), ("s5_pilot_wide", None): ("s5_wide", 10_465),
           ("s5_pilot_boolsub", None): ("s5_boolsub", 10_500), ("s5_pilot_selfree", "SF"): ("s5_selfree", 11_130),
           ("s5_pilot_selfree", "base"): ("s5_low_base", 2_280), ("s5_pilot_pop1000", None): ("s5_pop1000", 8_250),
           ("s5_pilot_keys", None): ("s5_revisit_restart", 105_600),
           ("s5_pilot_robust", None): ("s5_revisit_robust", 6_000),
           ("s5_pilot_restart10", None): ("s5_revisit_restart", 105_600)}
# Runs entering each registered statistic (every arm; the form's Sample size): the denominator of the SE scaling.
N_REGISTERED = {"H12-1": 21_000, "H12-3": 20_930, "H12-2": 6_600, "H13-1": 11_130, "H13-2": 6_505 + 10_500,
                "H14-2": 8_250, "H15-1": 105_600, "H16-1": 6_000 + 3_000, "H16-2": 6_000 + 3_000}
Z95, Z80 = 1.6448536269514722, 0.8416212335729143


# ---------------------------------------------------------------- power (REQ-ANA-129)
def power(name, point, reps, n_pilot, n_registered):
    """Approximate power of a registered test at its margin and the registered size, the pilot's estimate taken as
    true: SE = SD(pilot replicates) x sqrt(n_pilot / n_registered); a TOST by `h6d.power` (the nearer bound), a
    one-sided test above the margin Phi((estimate - margin) / SE - z_0.95), with the margin giving 80% power; a
    conjunction of TOSTs per member (its power at most the smallest member's)."""
    from scipy.stats import norm
    spec = S.TESTS[name]
    if spec["kind"] == "conj_tost":
        mem = {m: power_member(spec, point[m], [r[m] for r in reps], n_pilot, n_registered) for m in sorted(point)}
        pw = [v["power"] for v in mem.values()]
        return dict(members=mem, power_at_most=min(pw) if pw and all(np.isfinite(pw)) else float("nan"),
                    n_pilot=n_pilot, n_registered=n_registered)
    out = power_member(spec, point, reps, n_pilot, n_registered)
    return dict(out, n_pilot=n_pilot, n_registered=n_registered)


def power_member(spec, point, reps, n_pilot, n_registered):
    from scipy.stats import norm
    kept = np.asarray([x for x in reps if np.isfinite(x)], float)
    sd = float(np.std(kept, ddof=1)) if len(kept) > 1 else float("nan")
    if not np.isfinite(point) or not np.isfinite(sd):
        return dict(estimate=float(point), sd=sd, se=float("nan"), power=float("nan"), margin_80=float("nan"))
    if "delta" in spec:
        c = spec.get("center", 0.0)
        p = h6d.power(point - c, [x - c for x in kept], spec["delta"], n_pilot, n_registered)
        return dict(estimate=float(point), sd=sd, se=p["se"], power=p["power"], margin_80=p["delta_80"],
                    margin=f"{c:g} ± {spec['delta']:g}")
    se = sd * math.sqrt(n_pilot / n_registered)
    pw = float(norm.cdf((point - spec["margin"]) / se - Z95)) if se > 0 else float("nan")
    return dict(estimate=float(point), sd=sd, se=se, power=pw, margin_80=float(point - (Z95 + Z80) * se),
                margin=f"> {spec['margin']:+g}")


# ---------------------------------------------------------------- phases: completeness, CPU, success
def phase_summary(root, phase, rows):
    """Planned tasks, rows, terminal states, success share and queue CPU (per run, total) of a pilot phase."""
    from reach.runner.phases import _manifest_tasks
    try:
        planned = len(_manifest_tasks(phase, root))
    except FileNotFoundError:
        planned = 0
    st = defaultdict(int)
    for r in rows:
        st[r.get("terminal_state", "killed")] += 1
    cpu = [r["cpu_s"] for r in rows if r.get("cpu_s") is not None]
    ok = [r for r in rows if r.get("terminal_state") != "error"]
    return dict(planned=planned, rows=len(rows), complete=bool(planned) and len(rows) == planned, states=dict(st),
                errors=st.get("error", 0), cpu_stops=st.get("timeout", 0),
                success=float(np.mean([bool(r.get("exact")) for r in ok])) if ok else float("nan"),
                cpu_mean=float(np.mean(cpu)) if cpu else float("nan"), cpu_total_h=float(sum(cpu) / 3600))


def projection(summary, phase, config=None):
    target, n = PROJECT[(phase, config)]
    return dict(phase=target, runs=n, cpu_h=summary["cpu_mean"] * n / 3600 if np.isfinite(summary["cpu_mean"])
                else float("nan"))


# ---------------------------------------------------------------- pairing (partial phases)
def pair(new, base):
    """Both arms restricted to the (set, target, seed) runs present in both."""
    k = lambda r: (r["set"], r["target_id"], r["seed"])
    both = {k(r) for r in new} & {k(r) for r in base}
    return [r for r in new if k(r) in both], [r for r in base if k(r) in both]


def _sets_with(rows, sets):
    have = {r["set"] for r in rows}
    return [g for g in sets if g in have]


def _blocks(names, ids, stat, boot, n_pilot):
    """One bootstrap for the tests of a block: per test the point, the test (as it would be decided) and the power at
    the registered size."""
    ids = {g: v for g, v in ids.items() if v}
    if not ids:
        return {n: dict(status="no rows") for n in names}
    res, point = S.run_block(names, ids, stat, boot)
    out = {}
    for name in names:
        if name not in res:
            out[name] = dict(status="not computable")
            continue
        r = res[name]
        out[name] = dict(status="ok", n_pilot=n_pilot, point=r["point"], test=r["tests"]["primary"],
                         test_iso=r["tests"].get("iso"),
                         power=power(name, r["point"]["primary"], r["replicates"]["primary"], n_pilot,
                                     N_REGISTERED[name]) if name in N_REGISTERED else None,
                         per_set={g: _brief(v) for g, v in point["per_set"].items()}, sets=sorted(ids))
    return out


def _block(name, ids, stat, boot, n_pilot):
    return _blocks([name], ids, stat, boot, n_pilot)[name]


def _brief(v):
    """Per-set reaches without the bulky fields."""
    if isinstance(v, dict) and "k50" in v:
        return {k: v[k] for k in ("k50", "raw", "bracketed", "top", "iso", "iso_flag")}
    if isinstance(v, dict):
        return {k: _brief(x) for k, x in v.items()}
    return v


# ---------------------------------------------------------------- the fitness arms (H12) and selection-free (H13)
def fitness(root, rows, tables, base, boot):
    out = {}
    for name, phase, sets in (("H12-1", "s5_pilot_grid", D.REAL_MAIN), ("H12-3", "s5_pilot_wide", D.REAL_MAIN)):
        new, old = pair([r for r in rows[phase] if r["set"] in sets], [r for r in base if r["set"] in sets])
        have = _sets_with(new, sets)
        arms = _common({k: {g: S.one_run_arm([r for r in rr if r["set"] == g], 200, D._sizes(tables[g]))
                            for g in have} for k, rr in (("new", new), ("base", old))})
        S.guard(name, arms["new"], arms["base"])
        out[name] = _block(name, {g: sorted(arms["new"][g]) for g in have},
                           S.h12_contrast(name, arms["new"], arms["base"]), boot, len(new) + len(old))
    pv = {g: {t for t, v in tables[g].items() if D.proven(v)} for g in D.BOOL_MAIN if g in tables}
    new, old = pair([r for r in rows["s5_pilot_boolsub"] if r["target_id"] in pv.get(r["set"], ())],
                    [r for r in base if r["target_id"] in pv.get(r["set"], ())])
    have = _sets_with(new, D.BOOL_MAIN)
    both = _common(dict(new={g: S.one_run_arm([r for r in new if r["set"] == g], 200, D._sizes(tables[g])) for g in have},
                        base={g: S.one_run_arm([r for r in old if r["set"] == g], 200, D._sizes(tables[g]))
                              for g in have}))
    arm, barm = both["new"], both["base"]
    S.guard("H12-2", arm, barm)
    out["H12-2"] = _block("H12-2", {g: sorted(arm[g]) for g in have}, S.h12_sign("H12-2", arm, barm), boot, len(new))
    out["H12-2"]["detection_delay"] = detection_delay(new, old)
    out["H12-3"]["undefined_share"] = _undef_share(rows["s5_pilot_wide"])
    return out


def _common(arms):
    """Every arm of a pilot contrast restricted to the targets all of them hold (a target whose runs all ended in
    error leaves every arm; the confirmatory driver refuses instead)."""
    names = list(arms)
    sets = set.intersection(*(set(arms[n]) for n in names)) if names else set()
    out = {n: {} for n in names}
    for g in sets:
        keep = set.intersection(*(set(arms[n][g]) for n in names))
        for n in names:
            out[n][g] = {t: v for t, v in arms[n][g].items() if t in keep}
    return out


def detection_delay(new, base):
    """Evaluations between the first evaluation of a tree with the target's key and the exact hit (BS24 against the
    base), and the runs that evaluated the target but never detected it within E0."""
    out = {}
    for name, rr in (("BS24", new), ("base", base)):
        d, missed = [], 0
        for r in rr:
            tf = (r.get("rv") or {}).get("target_first")
            if tf is None or tf > S.E0:
                continue
            hit = (r["gen_exact"] + 1) * r["pop_size"] if r.get("exact") and r["gen_exact"] < 200 else None
            if hit is None:
                missed += 1
            else:
                d.append(hit - tf)
        out[name] = dict(runs=len(d), mean=float(np.mean(d)) if d else float("nan"),
                         median=float(np.median(d)) if d else float("nan"), seen_not_detected=missed)
    return out


def _undef_share(rows):
    """The counter's undefined-or-constant share of evaluations (on the identity grid) per run, at its stop."""
    v = [r["rv"]["stop"]["undef"] / r["rv"]["stop"]["n"] for r in rows if r.get("rv") and r["rv"]["stop"]["n"]]
    return dict(runs=len(v), mean=float(np.mean(v)) if v else float("nan"),
                note="batch errors that were undefined are not recorded by the runs")


def selfree(root, rows, tables, base, U, boot):
    sfr = [r for r in rows["s5_pilot_selfree"] if r.get("config") == "SF"]
    bsr = [r for r in rows["s5_pilot_selfree"] if r.get("config") == "base"]
    sf_sets = _sets_with(sfr, D.REAL_MAIN + D.BOOL_MAIN)
    sf = {g: S.one_run_arm([r for r in sfr if r["set"] == g], 200, D._sizes(tables[g])) for g in sf_sets}
    real = [g for g in D.REAL_MAIN if g in sf]
    boolean = [g for g in D.BOOL_MAIN if g in sf]
    out = {}
    if real and boolean:
        out["H13-1"] = _block("H13-1", {g: sorted(sf[g]) for g in sf}, S.h13_domains("H13-1", sf, real, boolean),
                              boot, len(sfr))
    else:
        out["H13-1"] = dict(status="needs SF runs in both domains")
    if real:
        new, old = pair([r for r in sfr if r["set"] in real], [r for r in bsr if r["set"] in real])
        gp = {g: S.one_run_arm([r for r in old if r["set"] == g], 200, D._sizes(tables[g])) for g in real}
        sfp = {g: S.one_run_arm([r for r in new if r["set"] == g], 200, D._sizes(tables[g])) for g in real}
        twin_rows = {g: {t: (tables[g][t]["units"]["bits_sem"], tables[g][t]["K_construction"]) for t in sf[g]}
                     for g in real}
        cov = {g: D._cov([r for r in base if r["set"] == g], g, (S.E0,)) for g in real}
        ids = {g: sorted(set(sf[g]) | set(cov[g])) for g in real}
        n = len([r for r in sfr if r["set"] in real]) + sum(len(v) for g in real for v in cov[g].values())
        out["H13-2"] = _block("H13-2", ids, S.h13_twin("H13-2", {g: sf[g] for g in real}, twin_rows, cov, U), boot, n)
        out["T"] = {g: dict(gp=S.arm_reach(gp[g], sorted(gp[g]))["k50"], sf=S.arm_reach(sfp[g], sorted(sfp[g]))["k50"])
                    for g in real if gp[g]}
    out["blindness"] = target_blindness(sfr, bsr)
    out["per_set"] = {}
    for g in sf_sets:
        r = S.arm_reach(sf[g], sorted(sf[g]))
        low = min(v[0] for v in sf[g].values())
        lows = [v[1] for v in sf[g].values() if v[0] == low]
        bsl = [r_["exact"] for r_ in bsr if r_["set"] == g and tables[g][r_["target_id"]]["units"]["bits_sem"] == low]
        out["per_set"][g] = dict(k50=r["k50"], raw=r["raw"], within=bool(np.isfinite(r["raw"]) and low <= r["raw"] <=
                                                                       r["top"]),
                                 lowest_bits=low, success_lowest_sf=float(np.mean(lows)),
                                 success_lowest_base=float(np.mean(bsl)) if bsl else float("nan"))
    cpu = {c: [r["cpu_s"] for r in rr if r.get("cpu_s") is not None] for c, rr in (("SF", sfr), ("base", bsr))}
    size = {c: [r["final_tree_size"] for r in rr if r.get("final_tree_size") is not None] for c, rr in
            (("SF", sfr), ("base", bsr))}
    out["cpu_and_size"] = {c: dict(cpu_mean=float(np.mean(cpu[c])) if cpu[c] else float("nan"),
                                   final_tree_size=float(np.mean(size[c])) if size[c] else float("nan"))
                           for c in cpu}
    return out


def target_blindness(sf, base):
    """Without selection a run's search does not depend on its target until it stops: two runs of one set and seed
    on different targets evaluate the same keys, so their counter's distinct and undefined counts agree at every
    prefix both reached. Checked on every such pair of SF runs; the same comparison on base runs (with selection)
    is the plant that must disagree."""
    out = {}
    for name, rr in (("SF", sf), ("base", base)):
        by = defaultdict(list)
        for r in rr:
            if r.get("rv"):
                by[r["set"], r["seed"]].append(r)
        pairs = agree = 0
        for runs in by.values():
            ref = runs[0]
            for r in runs[1:]:
                common = set(ref["rv"]["prefixes"]) & set(r["rv"]["prefixes"])
                if not common:
                    continue
                pairs += 1
                agree += all(ref["rv"]["prefixes"][p]["d"] == r["rv"]["prefixes"][p]["d"] and
                             ref["rv"]["prefixes"][p]["undef"] == r["rv"]["prefixes"][p]["undef"] for p in common)
        out[name] = dict(pairs=pairs, agree=agree)
    out["ok"] = out["SF"]["pairs"] == out["SF"]["agree"]
    return out


# ---------------------------------------------------------------- population 1,000 (H14)
def pop1000(root, rows, tables, boot):
    rr = rows["s5_pilot_pop1000"]
    have = _sets_with(rr, POP_SETS)
    arm = {g: S.one_run_arm([r for r in rr if r["set"] == g], 320, D._sizes(tables[g])) for g in have}

    def stat(pick):
        per = {g: S.arm_reach(arm[g], pick[g]) for g in have}
        return dict(values={"H14-2": dict(primary=S._mean(v["k50"] - S.NULL_16 for v in per.values()),
                                          iso=S._mean(v["iso"] - S.NULL_16 for v in per.values()))},
                    per_set=per, set_level={})
    out = dict(H14_2=_block("H14-2", {g: sorted(arm[g]) for g in have}, stat, boot, len(rr)) if have else
               dict(status="no rows"))
    ev = [r["pop_size"] * (r["gens_reached"] + (r["terminal_state"] == "converged")) for r in rr if "gens_reached" in r]
    out["evaluations_mean"] = float(np.mean(ev)) if ev else float("nan")
    return out


def h14_1_approx(root, h14_2):
    """H14-1 has no pilot on held-out sets: the restart arm's mean gap and SE from Stage 4's registered H11-1
    (h11.json: estimate, 90% interval / (2 x 1.645)), the R2 arm's from the pilot (scaled to 8,250 runs); the arms
    taken as independent (conservative: in the registered test they share targets)."""
    from scipy.stats import norm
    p = Path(root) / "data/analysis/h11.json"
    if not p.exists() or h14_2.get("status") != "ok":
        return dict(status="not computable")
    import json
    h = json.loads(p.read_text())["h11"]["tests"]["H11-1"]
    se_r = (h["hi90"] - h["lo90"]) / (2 * Z95)
    pw = h14_2["power"]
    est = h["estimate"] - h14_2["point"]["primary"]
    se = math.sqrt(se_r ** 2 + pw["se"] ** 2)
    return dict(status="approximate", estimate=est, se=se, restart_se=se_r, r2_se=pw["se"],
                power=float(norm.cdf((est + 1.0) / se - Z95)), margin_80=float(est - (Z95 + Z80) * se))


def h11_2l_approx(root):
    """H11-2L and H15-2 have no pilot: per-set gain SEs from Stage 4's registered H11-2 intervals (h11.json), power at
    +2 bits per set and for the conjunction (sets independent: the product), taking as true (a) the registered gains
    and (b) the like-for-like refit on the five shared levels (revision_exploratory.json, h11_common; lower bounds in
    four sets)."""
    import json
    from scipy.stats import norm
    p, q = Path(root) / "data/analysis/h11.json", Path(root) / "data/analysis/revision_exploratory.json"
    if not p.exists():
        return dict(status="not computable")
    h = json.loads(p.read_text())["h11"]
    common = json.loads(q.read_text())["h11_common"]["per_set"] if q.exists() else {}
    out = {}
    for reading, gains in (("registered", {g: v["gain"] for g, v in h["per_set"].items()}),
                           ("shared five levels", {g: v["gain"] for g, v in common.items()})):
        per = {}
        for g, gain in gains.items():
            t = h["per_set_gain_tests"][g]
            se = (t["hi90"] - t["lo90"]) / (2 * Z95)
            per[g] = dict(gain=gain, se=se, power=float(norm.cdf((gain - 2.0) / se - Z95)))
        out[reading] = dict(per_set=per, conjunction=float(np.prod([v["power"] for v in per.values()])) if per
                            else float("nan"))
    return dict(status="approximate", **out)


# ---------------------------------------------------------------- key sets and union coverage (H15; REQ-ANA-130)
def _lookup_levels(g):
    from reach import grammar as G
    from reach.revisit import Lookup, enum_root
    lk = Lookup(G.get(g), enum_root(g))
    return lk.lv.astype(np.int64), int(lk.top), lk.sha256


def load_keys(root, r):
    """(idx, first) of a row's side file, after checking it matches the row's SHA-256."""
    k = r.get("keys") or {}
    p = Path(root) / k.get("file", "")
    if not k.get("file") or not p.exists():
        return None, "no side file"
    if hashlib.sha256(p.read_bytes()).hexdigest() != k.get("sha256"):
        return None, "side file differs from its row"
    with np.load(p) as z:
        return (z["idx"].astype(np.int64), z["first"].astype(np.int64)), None


def key_checks(root, rows, levels_of, U, sample=2, rs=KEYS_R, prefixes=(5_000, 10_000)):
    """Per target of s5_pilot_keys: every side file present and matching its row; its per-level counts equal to the
    row's counter counts at every prefix and at the stop; the union histograms of the registered task code
    (`tasks.s5_union.histograms`) equal to an independent count (`stage5.histogram`) for both groups and prefixes;
    and, for `sample` targets per set, `stage5.union_expected` equal to brute force over every R-subset at every
    level. Returns (checks, per-target histograms in s5_union's format)."""
    from reach.tasks.s5_union import histograms
    problems, n_files, n_bytes, brute = [], 0, 0, 0
    hists = {}
    for g in sorted({r["set"] for r in rows}):
        lv, top, sha = levels_of(g)
        for r in rows:
            if r["set"] == g and (r.get("rv") or {}).get("lookup", {}).get("sha256") not in (None, sha):
                raise RuntimeError(f"{r['key']}: its key set was written against another lookup (guardrail row 175)")
        by = defaultdict(list)
        for r in rows:
            if r["set"] == g:
                by[r["target_id"]].append(r)
        for i, (t, rr) in enumerate(sorted(by.items())):
            runs = []
            for r in sorted(rr, key=lambda x: x["key"]):
                ks, why = load_keys(root, r)
                if ks is None:
                    problems.append(f"{r['key']}: {why}")
                    continue
                n_files += 1
                n_bytes += (Path(root) / r["keys"]["file"]).stat().st_size
                idx, first = ks
                rv = r["rv"]
                for p, snap in list(rv["prefixes"].items()) + [("stop", rv["stop"])]:
                    sel = idx if p == "stop" else idx[first <= int(p)]
                    got = np.bincount(lv[sel], minlength=top + 1)[1:top + 1].tolist()
                    if got != list(snap["lv"][1:top + 1]):
                        problems.append(f"{r['key']}: side file counts differ from the counter at {p}")
                        break
                hit = (r["gen_exact"] + 1) * r["pop_size"] if r.get("exact") else None
                runs.append((idx, first, hit))
            if len(runs) != len(rr):
                continue
            h = histograms(runs, lv, top, prefixes)
            for p in prefixes:
                for grp in ("all", "nohit"):
                    group = runs if grp == "all" else [x for x in runs if x[2] is None or x[2] > p]
                    ind = stage5.histogram([set(x[0][x[1] <= p].tolist()) for x in group], lambda k: int(lv[k]))
                    for K, cells in h[str(p)][grp].items():
                        mine = ind.get(int(K), [0] * (len(group) + 1))
                        if cells[1:] != mine[1:] or sum(cells) != U.get(g, {}).get(int(K), sum(cells)):
                            problems.append(f"{t} {p} {grp} level {K}: histograms differ")
            if i < sample:
                brute += _brute_force(runs, lv, rs, 10_000, problems, t)
            hists[t] = dict(set=g, target_id=t, hist=h)
    return dict(side_files=n_files, mean_bytes=n_bytes / n_files if n_files else float("nan"), brute_force=brute,
                problems=problems[:50], n_problems=len(problems), ok=not problems), hists


def _brute_force(runs, lv, rs, p, problems, t):
    """union_expected against the mean over every R-subset of the runs' union, per level (exact Fractions)."""
    n, checked = len(runs), 0
    keys = sorted({int(k) for x in runs for k in x[0][x[1] <= p]})
    if not keys:
        return 0
    pos = {k: i for i, k in enumerate(keys)}
    mask = np.zeros(len(keys), np.int64)
    for j, x in enumerate(runs):
        for k in x[0][x[1] <= p]:
            mask[pos[int(k)]] |= 1 << j
    level = np.array([lv[k] for k in keys])
    hist = {}
    for K in np.unique(level):
        m = np.array([bin(int(v)).count("1") for v in mask[level == K]])
        hist[int(K)] = np.bincount(m, minlength=n + 1).tolist()
    for R in (r for r in rs if r <= n):
        subs = np.array([sum(1 << j for j in c) for c in itertools.combinations(range(n), R)], np.int64)
        for K, h in hist.items():
            mk = mask[level == K]
            tot = int(((mk[None, :] & subs[:, None]) != 0).sum())
            if stage5.union_expected(h, R) != Fraction(tot, len(subs)):
                problems.append(f"{t}: union_expected differs from brute force at R = {R}, level {K}")
            checked += 1
    return checked


def keys_twins(rows, hists, tables, U, boot):
    """The pilot analogue of H15-1 at 16 restarts, G(16 x 100) - G(1 x 100), and the restart-twin rule check: per set
    and R the twin's K50 under the primary rule (runs without a hit, n' < R pooled at the level's mean), all runs, the
    union of all runs, and independence (1 - (1 - c1)^R), with the targets pooled."""
    sets = _sets_with(rows, KEYS_SETS)
    sizes = {g: {t: v["units"]["bits_sem"] for t, v in tables[g].items()} for g in sets}
    by = D._by(rows, "set", "target_id")
    out = dict(rule={})
    have = {g: sorted(t for (h, t) in by if h == g and t in hists) for g in sets}
    rows55 = {g: {t: (sizes[g][t], tables[g][t]["K_construction"]) for t in have[g]} for g in sets}
    n = max((len(v) for v in by.values()), default=0)
    for g in sets:
        ur = [hists[t] for t in have[g]]
        rule = {}
        for R in (x for x in KEYS_R if x <= n):
            ent = {}
            for x in ur:
                h = x["hist"][D.UNION_PREFIX]
                K_t = rows55[g][x["target_id"]][1]
                nohit = {int(K): v for K, v in h["nohit"].items()}
                alls = {int(K): v for K, v in h["all"].items()}
                v1 = S.twin_vector(nohit, h["n_nohit"], U[g], 1)
                ent[x["target_id"]] = dict(
                    primary=(K_t, S.twin_vector(nohit, h["n_nohit"], U[g], R)),
                    all_runs=(K_t, S.twin_vector(alls, h["n_all"], U[g], R)),
                    union_all=(K_t, S.twin_vector(alls, h["n_all"], U[g], h["n_all"])),
                    independence=(K_t, None if v1 is None else 1 - (1 - v1) ** R))
            ts = sorted(ent)
            rr = {}
            for name in ("primary", "all_runs", "union_all", "independence"):
                probs = S.restart_twin({t: ent[t][name] for t in ts}, ts, U[g])
                rr[name] = S.arm_reach(S._twin_arm(rows55[g], probs, ts), ts)["k50"]
            rr["pooled_targets"] = sum(1 for t in ts if ent[t]["primary"][1] is None)
            rule[str(R)] = rr
        out["rule"][g] = rule
    R = min(16, n)
    if not sets or R < 2:
        out["H15-1"] = dict(status="no rows")
        return out
    arms = dict(restart={}, one={}, rows=rows55, twin={})
    for g in sets:
        rb = {t: by[g, t] for t in have[g]}
        arms["restart"][g] = S.restart_arm(rb, sizes[g], R, 100)
        arms["one"][g] = S.restart_arm(rb, sizes[g], 1, 100)
        arms["twin"][g] = D.union_twins([hists[t] for t in have[g]], rows55[g], U[g], R)
    out["R"] = R
    out["H15-1"] = _block("H15-1", {g: have[g] for g in sets}, S.heldout(arms, sets, U), boot,
                          sum(len(by[g, t]) for g in sets for t in have[g]))
    return out


# ---------------------------------------------------------------- re-executions and population (H16)
def reexec_check(root, phase, src):
    """The re-executions of a phase (tasks naming audit_of; the others, new runs, set aside: guardrail row 200)
    against their originals (`audit.revisit_compare`; side files where the task names one)."""
    from reach.analysis.s2.audit import revisit_compare
    from reach.runner.phases import _manifest_tasks, phase_rows
    try:
        tasks = _manifest_tasks(phase, root)
    except FileNotFoundError:
        return dict(status="not planned")
    again, orig = phase_rows(phase, root), phase_rows(src, root)
    out = dict(planned=0, rerun=0, identical=0, differing={}, missing=0, missing_original=0, not_reexec=0)
    cpu = dict(rerun=0.0, original=0.0)
    for t in tasks:
        if "audit_of" not in t.get("meta", {}):
            out["not_reexec"] += 1
            continue
        out["planned"] += 1
        r, o = again.get(t["key"]), orig.get(t["meta"]["audit_key"])
        if r is None:
            out["missing"] += 1
            continue
        if o is None:
            out["missing_original"] += 1
            continue
        out["rerun"] += 1
        d = revisit_compare(o, r, ("keys",), "--keys_out" in t.get("argv", ()), root)
        if d:
            out["differing"][t["meta"]["audit_key"]] = d
        else:
            out["identical"] += 1
        cpu["rerun"] += r.get("cpu_s") or 0.0
        cpu["original"] += o.get("cpu_s") or 0.0
    out["cpu_ratio"] = cpu["rerun"] / cpu["original"] if cpu["original"] else float("nan")
    out["ok"] = not out["differing"] and not out["missing_original"]
    return out


def robust(root, rows, tables, base, U, boot):
    rr = rows["s5_pilot_robust"]
    pops = {25: [r for r in rr if r.get("config") == "R1"], 1000: [r for r in rr if r.get("config") == "R2"]}
    k = lambda r: (r["set"], r["target_id"], r["seed"])
    keys = {k(r) for r in pops[25]} & {k(r) for r in pops[1000]} & {k(r) for r in base}
    pops = {p: [r for r in v if k(r) in keys] for p, v in pops.items()}
    pops[100] = [r for r in base if k(r) in keys]
    real = [g for g in ROB_REAL if any(r["set"] == g for r in pops[25])]
    boolean = [g for g in ROB_BOOL if any(r["set"] == g for r in pops[25])]
    if not real or not boolean:
        return dict(status="needs runs at every population in both domains")
    sets = real + boolean
    rows_t = {g: {t: (tables[g][t]["units"]["bits_sem"], tables[g][t]["K_construction"])
                  for t in {r["target_id"] for r in pops[25] if r["set"] == g}} for g in sets}
    cov = {p: {g: D._cov(pops[p], g, (S.E0,)) for g in sets} for p in S.POPS}
    stat = S.h16(cov, rows_t, U, real, boolean)
    n = sum(len(v) for v in pops.values())
    ids = {g: sorted(rows_t[g]) for g in sets}
    out = _blocks(["H16-1", "H16-2"], ids, stat, boot, n)
    out["twin_reach"] = {str(p): {g: v["primary"] for g, v in out["H16-2"].get("per_set", {}).get(str(p), {}).items()}
                         for p in S.POPS} if out["H16-2"].get("status") == "ok" else None
    return out


# ---------------------------------------------------------------- build
def build(root, boot=stage5.BOOT, levels_given=None, lookup_levels=None):
    root = Path(root)
    t0 = time.process_time()
    rows = {ph: read_phase(ph, root) for ph in PHASES}
    base = read_phase(BASE, root)
    summary = {ph: phase_summary(root, ph, rows[ph]) for ph in PHASES}
    for ph in PHASES:
        if ph == "s5_pilot_selfree":
            for c in ("SF", "base"):
                rr = [r for r in rows[ph] if r.get("config") == c]
                summary[ph][f"projection_{c}"] = projection(phase_summary(root, ph, rr), ph, c)
        else:
            summary[ph]["projection"] = projection(summary[ph], ph)
    partial = not all(s["complete"] for s in summary.values())
    sets = sorted({r["set"] for ph in PHASES for r in rows[ph]} | {r["set"] for r in base})
    tables, shas = {}, {}
    for g in sets:
        tables[g], shas[g] = D.target_table(root, g, ("pilot", "heldout", "restart"))
    for ph in PHASES:
        D.check_rows_targets([r for r in rows[ph] if r["set"] in tables], tables, shas)
    D.check_rows_targets(base, tables, shas)
    U = D.levels(root, sets, levels_given)
    res = dict(label="PILOT (pilot targets and seeds; never confirmatory)" + (" - PARTIAL: phases still running"
                                                                             if partial else ""),
               partial=partial, boot=boot, seed=stage5.SEED, phases=summary)
    res["fitness"] = fitness(root, rows, tables, base, boot)
    res["selfree"] = selfree(root, rows, tables, base, U, boot)
    res["pop1000"] = pop1000(root, rows, tables, boot)
    res["pop1000"]["H14_1"] = h14_1_approx(root, res["pop1000"]["H14_2"])
    res["h11_2l"] = h11_2l_approx(root)
    kr = rows["s5_pilot_keys"]
    if kr:
        lk = lookup_levels or _lookup_levels
        res["keys"], hists = key_checks(root, kr, lk, U)
        res["keys"]["cpu_counter_share"] = _counter_share(kr)
        res["keys_twins"] = keys_twins(kr, hists, tables, U, boot)
    else:
        res["keys"], res["keys_twins"] = dict(status="no rows"), dict(status="no rows")
    res["reexec"] = dict(restart10=reexec_check(root, "s5_pilot_restart10", "s4_restart"),
                         robust=reexec_check(root, "s5_pilot_robust", ROBUST_SRC))
    r10 = res["reexec"]["restart10"]
    if r10.get("rerun"):
        orig_cpu = sum(r.get("cpu_s") or 0.0 for r in read_phase("s4_restart", root))
        res["reexec"]["restart10"]["projected_s5_revisit_restart_cpu_h"] = r10["cpu_ratio"] * orig_cpu / 3600
    rb = res["reexec"]["robust"]
    if rb.get("rerun"):
        orig = [r for r in read_phase("s2_cal_robust", root) if r.get("config") in ("R1", "R2")]
        res["reexec"]["robust"]["projected_s5_revisit_robust_cpu_h"] = rb["cpu_ratio"] * sum(
            r.get("cpu_s") or 0.0 for r in orig) / 3600
    res["robust"] = robust(root, rows, tables, base, U, boot)
    res["built"] = time.strftime("%Y-%m-%d %H:%M:%S")
    res["cpu_seconds"] = round(time.process_time() - t0, 1)
    return res


def _counter_share(rows):
    c = [(r["rv"].get("cpu_seconds") or 0.0, r.get("cpu_seconds") or 0.0) for r in rows if r.get("rv")]
    tot = sum(b for _, b in c)
    return sum(a for a, _ in c) / tot if tot else float("nan")


# ---------------------------------------------------------------- report
def _f(x, d=2, sign=True):
    return D._f(x, d, sign)


def _test_line(name, b):
    if b.get("status") != "ok":
        return f"| {name} | {b.get('status')} | | | | | | |"
    t, pw = b["test"], b.get("power") or {}
    est = b["point"]["primary"]
    est_s = ", ".join(f"{k} {_f(v)}" for k, v in sorted(est.items())) if isinstance(est, dict) else _f(est)
    if "members" in pw:
        sd = "; ".join(f"{m} {_f(v['sd'], 3, False)}" for m, v in pw["members"].items())
        se = "; ".join(f"{m} {_f(v['se'], 3, False)}" for m, v in pw["members"].items())
        p = f"at most {_f(pw['power_at_most'], 2, False)}"
        m80 = "; ".join(f"{m} {_f(v['margin_80'], 2, False)}" for m, v in pw["members"].items())
    else:
        sd, se, p, m80 = (_f(pw.get("sd"), 3, False), _f(pw.get("se"), 3, False), _f(pw.get("power"), 2, False),
                          _f(pw.get("margin_80"), 2))
    return (f"| {name} | {est_s} | {', '.join(b['sets'])} | {b['n_pilot']} | {sd} | {se} | {p} | {m80} |")


def render(res):
    L = [f"# Stage 5 pilot report: {res['label']}", "",
         f"> Generated by `reach.analysis.s2.stage5_report --pilot` ({res['built']}); every number from "
         "data/analysis/stage5_pilot.json. Do not edit by hand. Pilot targets and seeds only (s5_pilot_restart10: ten "
         f"restart runs with known outcomes): not confirmatory. Bootstrap {res['boot']} replicates, seed {res['seed']}.",
         "", "## Phases (queue CPU seconds)", "",
         "| Phase | planned | rows | errors | CPU stops | success share | CPU per run (s) | CPU total (h) | projected "
         "for (h) |", "|---|---|---|---|---|---|---|---|---|"]
    for ph, s in res["phases"].items():
        pr = [s[k] for k in ("projection", "projection_SF", "projection_base") if k in s]
        prs = "; ".join(f"{p['phase']} ({p['runs']:,} runs) {_f(p['cpu_h'], 1, False)}" for p in pr)
        L.append(f"| {ph} | {s['planned']} | {s['rows']} | {s['errors']} | {s['cpu_stops']} | "
                 f"{_f(s['success'], 3, False)} | {_f(s['cpu_mean'], 2, False)} | {_f(s['cpu_total_h'], 2, False)} | "
                 f"{prs} |")
    L += ["", "Projection: the pilot's mean CPU per run times the registered runs of the matching phase (pilot targets "
          "differ in level mix from the registered ones, so this is a rough guide); for re-executions also the "
          "re-run/original CPU ratio times the original phase's CPU (below).", "",
          "## Statistics, paired SD and approximate power at the registered margin", "",
          "SE at the registered size = the pilot's bootstrap SD x sqrt(n_pilot / n_registered) (Stage 3's rule), the "
          "pilot's estimate taken as true; one-sided tests Phi((estimate - margin)/SE - 1.645); TOSTs `h6d.power`. "
          "'margin 80%' is the margin (or, for TOSTs, the half-width) that would give 80% power; no margin is changed.",
          "", "| Test (pilot analogue) | estimate | sets | n pilot | SD | SE registered | power | margin 80% |",
          "|---|---|---|---|---|---|---|---|"]
    fit, sf, kt, rb = res["fitness"], res["selfree"], res["keys_twins"], res["robust"]
    for name, b in (("H12-1", fit["H12-1"]), ("H12-2", fit["H12-2"]), ("H12-3", fit["H12-3"]), ("H13-1", sf["H13-1"]),
                    ("H13-2", sf.get("H13-2", dict(status="no real SF rows"))), ("H14-2", res["pop1000"]["H14_2"]),
                    ("H15-1 (R = 16)", kt.get("H15-1", dict(status="no rows"))),
                    ("H16-1", rb.get("H16-1", rb)), ("H16-2", rb.get("H16-2", rb))):
        L.append(_test_line(name, b))
    a = res["pop1000"]["H14_1"]
    L += ["", f"- H14-1 (no pilot on held-out sets): {a.get('status')}" + (
        f"; estimate {_f(a['estimate'])} (Stage 4's restart gap minus the pilot R2 gap), SE {_f(a['se'], 3, False)}, "
        f"power at -1 {_f(a['power'], 2, False)}, margin for 80% {_f(a['margin_80'])}" if "se" in a else "")]
    h = res["h11_2l"]
    if h.get("status") == "approximate":
        for reading in ("registered", "shared five levels"):
            v = h[reading]
            L.append(f"- H11-2L at +2 bits, gains taken as the {reading} values: per set " + ", ".join(
                f"{g} {_f(x['power'], 2, False)}" for g, x in v["per_set"].items())
                + f"; the conjunction {_f(v['conjunction'], 2, False)}")
    L.append("- H15-2: not computable from the pilots (no counted long run in any pilot).")
    L += ["", "## Checks", ""]
    k = res["keys"]
    if "side_files" in k:
        L += [f"- s5_pilot_keys: {k['side_files']} side files (mean {_f(k['mean_bytes'], 0, False)} bytes), counts equal "
              f"to the counter's and histograms equal to an independent count: {'yes' if k['ok'] else 'NO'} "
              f"({k['n_problems']} problems); union_expected against brute force over every R-subset: "
              f"{k['brute_force']} (target, level, R) cases; the counter's share of the runs' CPU "
              f"{_f(k['cpu_counter_share'], 3, False)}."]
        for p in k["problems"][:5]:
            L.append(f"  - {p}")
    else:
        L.append(f"- s5_pilot_keys: {k.get('status')}")
    for name, src in (("restart10", "s4_restart"), ("robust", "s2_pilot_robust")):
        r = res["reexec"][name]
        if "planned" in r:
            proj = r.get("projected_s5_revisit_restart_cpu_h", r.get("projected_s5_revisit_robust_cpu_h"))
            L.append(f"- s5_pilot_{name} against {src}: {r['identical']} of {r['planned']} re-executions equal in every "
                     f"registered field ({r['rerun']} run, {len(r['differing'])} differ, {r['missing']} missing"
                     f"{', ' + str(r['not_reexec']) + ' new R1 tasks set aside' if r['not_reexec'] else ''}); CPU ratio "
                     f"re-run/original {_f(r['cpu_ratio'], 2, False)}" + (f", projected {_f(proj, 1, False)} CPU-h"
                                                                          if proj is not None else "") + ".")
        else:
            L.append(f"- s5_pilot_{name}: {r.get('status')}")
    bl = sf["blindness"]
    L.append(f"- Target blindness (SF runs of one set and seed agree in distinct and undefined counts at every shared "
             f"prefix): {bl['SF']['agree']} of {bl['SF']['pairs']} pairs; base runs (selection, the plant): "
             f"{bl['base']['agree']} of {bl['base']['pairs']}.")
    dd = fit["H12-2"].get("detection_delay", {})
    if dd:
        L.append("- BS24 detection delay (evaluations from the first evaluation of the target's key to the hit): "
                 + "; ".join(f"{n} mean {_f(v['mean'], 0, False)}, median {_f(v['median'], 0, False)} over "
                             f"{v['runs']} runs, seen but not detected {v['seen_not_detected']}" for n, v in dd.items()))
    us = fit["H12-3"].get("undefined_share", {})
    L.append(f"- WIDE: the counter's undefined-or-constant share {_f(us.get('mean'), 3, False)} over {us.get('runs')} "
             f"runs ({us.get('note')}).")
    L.append(f"- Population 1,000: mean evaluations per run {_f(res['pop1000']['evaluations_mean'], 0, False)}.")
    L += ["", "## Selection-free GP per set (pilot)", "", "| Set | K₅₀(SF) | within the tested sizes | lowest level (bits) | "
          "success there SF / base |", "|---|---|---|---|---|"]
    for g, v in sf["per_set"].items():
        L.append(f"| {g} | {_f(v['k50'])} | {'yes' if v['within'] else 'no'} | {_f(v['lowest_bits'], 2, False)} | "
                 f"{_f(v['success_lowest_sf'], 2, False)} / {_f(v['success_lowest_base'], 2, False)} |")
    cs = sf["cpu_and_size"]
    L.append(f"\nCPU per run SF {_f(cs['SF']['cpu_mean'], 2, False)} s, base {_f(cs['base']['cpu_mean'], 2, False)} s; "
             f"mean final tree size SF {_f(cs['SF']['final_tree_size'], 1, False)}, base "
             f"{_f(cs['base']['final_tree_size'], 1, False)} (the runs record no size by generation).")
    L += ["", "## The restart-twin rule (s5_pilot_keys; twin K₅₀ per R)", "",
          "| Set | R | primary (no-hit runs, pooled) | all runs | union of all runs | independence | targets pooled |",
          "|---|---|---|---|---|---|---|"]
    for g, rule in kt.get("rule", {}).items():
        for R, v in rule.items():
            L.append(f"| {g} | {R} | {_f(v['primary'])} | {_f(v['all_runs'])} | {_f(v['union_all'])} | "
                     f"{_f(v['independence'])} | {v['pooled_targets']} |")
    if rb.get("twin_reach"):
        L += ["", "## Twin reach at E₀ by population (s5_pilot_robust; interpolated)", "",
              "| Set | 25 | 100 | 1,000 |", "|---|---|---|---|"]
        for g in rb["twin_reach"]["25"]:
            L.append(f"| {g} | " + " | ".join(_f(rb["twin_reach"][str(p)][g]) for p in S.POPS) + " |")
    return "\n".join(L) + "\n"

"""Stage 5 statistics library (stage5_design.md Sections 1 and 6; REQ-ANA-109 to REQ-ANA-111): the shared-level guard
(Rule S1, guardrail row 170's analysis-side test), the union-coverage estimator checked against brute force, and the
inference functions reproducing Stages 3 and 4's registered tests."""
import itertools
import json
import math
from collections import defaultdict
from fractions import Fraction

import numpy as np
import pytest

from reach.analysis.s2 import law, remedy, stage5
from reach.paths import ROOT


def _arm(levels, targets=3, g="R-X"):
    return [dict(set=g, target_id=f"L{lv}t{t}", units=dict(bits_sem=float(lv)), y=1) for lv in levels
            for t in range(targets)]


def test_contrast_refuses_mismatched_levels():
    a, b = _arm([7, 9, 11, 13, 15]), _arm([7, 9, 11, 13, 15])
    assert len(stage5.shared_levels(a, b)) == 15
    assert stage5.contrast(a, b, lambda rr: len(rr)) == 0
    # float noise in a level's size is the same level
    noisy = [dict(r, units=dict(bits_sem=r["units"]["bits_sem"] + 1e-12)) for r in b]
    stage5.shared_levels(a, noisy)
    # plant 1: one arm fitted on more levels (H11-2's registered arms: 55 against 30 rows; row 170)
    with pytest.raises(ValueError, match="differ in"):
        stage5.contrast(_arm([7, 9, 11, 13, 15, 17, 19]), b, lambda rr: len(rr))
    # plant 2: the same levels, one target swapped: a level-only guard would miss it
    swapped = b[:-1] + [dict(b[-1], target_id="other")]
    assert {r["units"]["bits_sem"] for r in swapped} == {r["units"]["bits_sem"] for r in a}
    with pytest.raises(ValueError, match="differ in"):
        stage5.shared_levels(a, swapped)
    # plant 3: one row of another set
    with pytest.raises(ValueError):
        stage5.shared_levels(a, b[:-1] + [dict(b[-1], set="R-Y")])
    # plant 4: a level moved (a target read at another size)
    with pytest.raises(ValueError):
        stage5.shared_levels(a, b[:-1] + [dict(b[-1], units=dict(bits_sem=16.0))])
    # triples are accepted as rows
    assert stage5.shared_levels([("R-X", 7.0, "t")], [("R-X", 7.0, "t")]) == [("R-X", 7.0, "t")]


def _brute(key_sets, R, level_of, K):
    subs = list(itertools.combinations(range(len(key_sets)), R))
    tot = sum(len({k for i in s for k in key_sets[i] if level_of(k) == K}) for s in subs)
    return Fraction(tot, len(subs))


def test_union_coverage_equals_brute_force_over_all_subsets():
    rng = np.random.default_rng(20261108)
    def level_of(k):
        return k % 3
    independence_differs = 0
    for _ in range(40):
        n = int(rng.integers(2, 8))
        key_sets = [set(rng.choice(30, int(rng.integers(0, 14)), replace=False).tolist()) for _ in range(n)]
        hist = stage5.histogram(key_sets, level_of)
        for R in range(1, n + 1):
            for K in range(3):
                h = hist.get(K, [0] * (n + 1))
                exact = _brute(key_sets, R, level_of, K)
                assert stage5.union_expected(h, R) == exact
                # plant: Figure 4's independence estimate, U (1 - (1 - p)^R) with p the mean per-run share
                U = 10
                p = sum(len({k for k in ks if level_of(k) == K}) for ks in key_sets) / n / U
                independence_differs += abs(U * (1 - (1 - p) ** R) - float(exact)) > 1e-9
    assert independence_differs > 0                       # the brute-force check separates the two readings


def test_union_coverage_limits():
    key_sets = [{1, 2, 3}, {3, 4}, {4, 5, 6, 7}]
    h = stage5.histogram(key_sets, lambda k: 0)[0]
    assert h == [0, 5, 2, 0]                               # 5 functions in one run, 2 (3 and 4) in two
    assert stage5.union_expected(h, 3) == 7                # all runs: every covered function
    assert stage5.union_expected(h, 1) == Fraction(9, 3)   # one run: the mean run's count
    assert stage5.union_share(h, 3, 14) == 0.5
    for R in (0, 4):
        with pytest.raises(ValueError):
            stage5.union_expected(h, R)


def test_restart_twin_pools_targets_with_too_few_unhit_runs():
    U = {3: 10, 4: 10}
    t_ok = dict(level=5, n_prime=4, hist={3: [0, 0, 0, 0, 5]})          # 5 functions in every run: share 0.5
    t_few = dict(level=5, n_prime=1, hist={3: [0, 10]})                 # fewer unhit runs than R: pooled
    t_other = dict(level=7, n_prime=3, hist={3: [0, 3, 0, 0]})          # 3 functions, one run each: R = 2 covers 2
    t_lonely = dict(level=9, n_prime=1, hist={})                        # its level has no eligible target
    r = stage5.restart_twin_shares([t_ok, t_few, t_other, t_lonely], 2, U)
    assert math.isclose(r["shares"][3], (2 * 0.5 + 1 * 0.2) / 3)
    assert r["shares"][4] == 0.0
    assert (r["eligible"], r["pooled"], r["unpooled_levels"]) == (2, 1, [9])


@pytest.mark.parametrize("name, delta", [("H6d-1", 1.0), ("H6d-2", 0.15)])
def test_tost_reproduces_the_registered_stage3_tests(name, delta):
    h = json.loads((ROOT / "data/analysis/h6d.json").read_text())
    t = stage5.tost(h[name]["estimate"], h["replicates"][name], delta)
    for k in ("p", "lo90", "hi90", "estimable", "equivalent"):
        assert t[k] == h[name][k], k


def test_noninferiority_conjunction_and_holm():
    reps = [-0.5] * 95 + [-1.5] * 5
    t = stage5.noninferiority(-0.4, reps, -1.0, "above")
    assert t["p"] == 6 / 101 and t["estimable"]
    assert not stage5.noninferiority(float("nan"), reps, -1.0)["estimable"]
    assert not stage5.noninferiority(0.0, reps[:80] + [float("nan")] * 20, -1.0)["estimable"]   # 80% kept
    below = stage5.noninferiority(0.0, [0.5] * 99 + [1.5], 1.0, "below")
    assert below["p"] == 2 / 101
    c = stage5.conjunction({"a": dict(p=0.01, estimable=True), "b": dict(p=0.03, estimable=True)})
    assert c["p"] == 0.03 and c["estimable"]
    # guardrail row 155's plant: a non-estimable test counts as p = 1 and does not block the other
    out = stage5.holm({"x": dict(p=0.001, estimable=False), "y": dict(p=0.02, estimable=True)})
    assert out["y"]["supported"] and not out["x"]["supported"]
    iv = stage5.set_level_interval([1.0, 2.0, 3.0])
    assert math.isclose(iv["mean"], 2.0) and iv["lo"] < 2.0 < iv["hi"]


def _h11_stat(restart, one):
    """Stage 4's statistic (`remedy.h11`'s inner `stats`) on resampled ids."""
    def stat(pick):
        gaps, gains = {}, {}
        for g, ids in pick.items():
            runs, size = restart[g]
            rb = {f"{t}#{i}": runs[t] for i, t in enumerate(ids) if t in runs}
            k = remedy.restart_k50(rb, {f"{t}#{i}": size[t] for i, t in enumerate(ids) if t in runs})
            by = defaultdict(list)
            for r in one[g]:
                by[r["target_id"]].append(r)
            k1 = law._k50([r for t in ids for r in by.get(t, [])], "bits_sem")
            gaps[g], gains[g] = k["k50"] - math.log2(320_000), k["k50"] - k1
        return dict(D=float(np.mean(list(gaps.values()))) if all(np.isfinite(list(gaps.values()))) else float("nan"),
                    gains=gains)
    return stat


def _h11_by_library(restart, one, boot):
    ids = {g: set(restart[g][0]) | {r["target_id"] for r in one[g]} for g in restart}
    stat = _h11_stat(restart, one)
    point = stat({g: sorted(v) for g, v in ids.items()})
    reps = stage5.target_bootstrap(ids, stat, boot=boot)
    t1 = stage5.noninferiority(point["D"], [r["D"] for r in reps], remedy.H11_MARGIN, "above", boot)
    per = {g: stage5.noninferiority(point["gains"][g], [r["gains"][g] for r in reps], remedy.H11_GAIN, "above", boot)
           for g in sorted(restart)}
    t2 = stage5.conjunction(per)
    return stage5.holm({"H11-1": t1, "H11-2": t2}), per


def _synthetic_h11(rng):
    restart, one = {}, {}
    for g in ("S-A", "S-B"):
        runs, size, recs = {}, {}, []
        for lv in range(6):
            for t in range(8):
                tid = f"{g}{lv}{t}"
                size[tid] = 6.0 + 2 * lv
                p = 1 / (1 + math.exp(lv - 3))
                runs[tid] = [dict(terminal_state="converged", gens_reached=200, generations=200,
                                  gen_exact=10 if rng.random() < p / 3 else None) for _ in range(16)]
                if lv < 4:
                    recs += [dict(target_id=tid, y=int(rng.random() < p / 2), units=dict(bits_sem=size[tid]))
                             for _ in range(3)]
        restart[g], one[g] = (runs, size), recs
    return restart, one


def test_target_bootstrap_reproduces_stage4s_replicates():
    restart, one = _synthetic_h11(np.random.default_rng(7))
    ref = remedy.h11(restart, one, boot=25)
    got, per = _h11_by_library(restart, one, 25)
    for k in ("H11-1", "H11-2"):
        assert got[k]["p"] == ref["tests"][k]["p"] and got[k]["supported"] == ref["tests"][k]["supported"], k
    for g, v in per.items():
        assert v["p"] == ref["per_set_gain_tests"][g]["p"]
    # plant: the same statistic with the sets drawn in another order gives other replicates (the check fires)
    ids = {g: sorted(set(restart[g][0]) | {r["target_id"] for r in one[g]}) for g in restart}
    rng, stat, reps = np.random.default_rng(stage5.SEED), _h11_stat(restart, one), []
    for _ in range(25):
        pick = {}
        for g in sorted(ids, reverse=True):
            pick[g] = [ids[g][j] for j in rng.integers(0, len(ids[g]), len(ids[g]))]
        reps.append(stat(pick)["D"])
    assert stage5.noninferiority(0.0, reps, remedy.H11_MARGIN)["p"] != ref["tests"]["H11-1"]["p"] or \
        reps != [r["D"] for r in stage5.target_bootstrap(ids, stat, boot=25)]


@pytest.mark.study_machine
def test_h11_through_the_library_reproduces_the_registered_code_path():
    from reach.analysis.s2 import revision_report as R
    res, one, sizes, _ = R.restart_data(ROOT)
    restart = {g: (res[g], {t: sizes[g][t] for t in res[g]}) for g in res}
    ref = remedy.h11(restart, one, boot=12)
    got, _ = _h11_by_library(restart, one, 12)
    assert all(got[k]["p"] == ref["tests"][k]["p"] for k in ("H11-1", "H11-2"))
    h = json.loads((ROOT / "data/analysis/h11.json").read_text())["h11"]
    for g, v in h["per_set"].items():
        assert math.isclose(ref["per_set"][g]["k50"], v["k50"], abs_tol=1e-9)
    assert math.isclose(ref["tests"]["H11-1"]["estimate"], h["tests"]["H11-1"]["estimate"], abs_tol=1e-9)

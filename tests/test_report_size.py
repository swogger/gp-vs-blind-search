"""Every generated report stays within the 200-line limit at calibration size (REQ-ANA-55; guardrail row 110), written
before the split. Synthetic data only.

The calibration analysis writes nine files (main report, H1 cells, Holm members, termination, supplement, run-time
model, slopes and gaps, H8's parts; and the JSON). The DRYRUN is smaller than the calibration (fewer cells, sets and
members), so its reports passing the limit says nothing about the calibration's. Here a small declared build is padded
to the registered sizes (Stage 1: 16 calibration sets, 12 full sets, 6 robustness sets x 6 configurations at E0 and
4 x E0, 10 levels per set, 5 budget levels with allocation runs, the declared Holm families) and every report is
rendered.
"""
import copy
import json

from reach.analysis.s2 import confirm as C
from reach.analysis.s2 import families as F
from reach.paths import ROOT


def _small(tmp_path):
    from test_holm_families import _calibration
    _calibration(tmp_path)
    return C.build(dict(C.PHASE_MAPS["calibration"], levels=None), tmp_path, boot=20, allow_incomplete=True)


def _runtime(sets, budgets, alloc_sets):
    """A run-time model result of the shape `confirm.runtime_tests` returns (values are placeholders); cells only in
    the sets with allocation runs (the full sets)."""
    pt = dict(kappa=0.8, alpha=0.5, t0=3.0, sigma_target=1.0, ratio=1.6, slope=1.2, lstar=15.0)
    cell = dict(best_split_k50=12.0, best_split_length=50, one_run_k50=11.0, length=50)
    out = {}
    for d, ss in sets.items():
        out[d] = dict(sets={g: dict(point=pt, cells={b: cell for b in budgets} if g in alloc_sets else {},
                                    lstar_ci=dict(lo=10.0, hi=20.0),
                                    comparison=dict(heldout_loglik=dict(weibull=-10.0, lognormal=-11.0)))
                            for g in ss}, failed={},
                      tests=dict(H4d=dict(sets={g: dict(diff=0.1, lo=-0.1, hi=0.3, p=0.01) for g in ss}, holds=True,
                                          p=0.01, untestable=None, margin=0.5),
                                 H4e=dict(cells=[dict(set=g, budget=b, help=True, kappa=0.8, agree=True) for g in ss
                                                 if g in alloc_sets for b in budgets], share=1.0, lo=0.9, hi=1.0,
                                          registered_share=0.75,
                                          holds_point=True, p=0.0, untestable=None),
                                 H4f=dict(cells=[dict(set=g, budget=b, among=True, p=0.5, model_length=50,
                                                      central=[25, 50, 100]) for g in ss if g in alloc_sets
                                                 for b in budgets],
                                          p=0.5, untestable=None)),
                      comparison=dict(heldout_loglik=dict(weibull=-100.0, lognormal=-110.0), preferred="weibull",
                                      sets_preferring_weibull=len(ss), sets=len(ss)))
    return out


def pad(res):
    """`res` padded to the registered calibration sizes."""
    from reach import grammar as G
    from reach.runner.phases import CAL_SETS, FULL_SETS
    r = copy.deepcopy(res)
    dom = {g: G.get(g).kind for g in CAL_SETS}
    main = {d: [g for g in CAL_SETS if dom[g] == d and g not in C.SIZE_CHECKS] for d in ("real", "bool")}
    robust = json.loads((ROOT / "data/registration/levels.json").read_text())["_robust"]
    levels = list(range(5, 25, 2))
    c0 = r["h1"][0]
    r["h1"] = [dict(c0, set=g, algorithm=a, config=c, budget=b) for g, a, c, b in C.declared_h1_cells()]
    h80 = r["h8"][0]
    r["h8"] = [dict(h80, set=g, config=c, budget=b) for d in robust for g in robust[d] for c in F.H8_CONFIGS
               for b in (1, 4)]
    b0, c0_ = next(iter(r["h8b"].values())), next(iter(r["h8c"].values()))
    r["h8b"] = {f"{g}|{c}": b0 for d in robust for g in robust[d] for c in F.H8_CONFIGS}
    r["h8c"] = {f"{g}|{c}": c0_ for d in robust for g in robust[d] for c in F.H8_CONFIGS}
    h5 = next(iter(r["h5"].values()))["A-B"]
    r["h5"] = {g: {k: h5 for k in F.H5_CONTRASTS} for g in FULL_SETS}
    lv = {K: dict(b=3, c=1, p=0.6) for K in levels}
    r["h5_mcnemar"] = {g: {"A-B": lv, "HC-A": lv} for g in FULL_SETS}
    h6 = next(iter(r["h6"].values()))
    r["h6"] = {g: dict(h6, domain=dom[g]) for g in CAL_SETS}
    ps = next(iter(r["h4"]["bool"]["per_set"].values()))
    h4 = r["h4"]["bool"]
    pair = h4["h4c"]["pairs"][0]
    r["h4"] = {d: dict(h4, per_set={g: ps for g in CAL_SETS if dom[g] == d},
                       h4c=dict(h4["h4c"], pairs=[dict(pair, sets=[a, b]) for i, a in enumerate(main[d])
                                                  for b in main[d][i + 1:]]))
               for d in ("real", "bool")}
    hb = r["h6b"]["bool"]
    r["h6b"] = {d: dict(hb, per_set={g: dict(levels=[[K, -0.1, 30] for K in levels], sign_changes=[[9, 11]])
                                     for g in main[d]}) for d in ("real", "bool")}
    a0 = next(iter(r["alloc"].values()))
    r["alloc"] = {g: a0 for g in FULL_SETS}
    z0 = next(iter(r["sizes"].values()))
    r["sizes"] = {g: z0 for g in CAL_SETS}
    r["h4_model"] = _runtime(main, C.H4_BUDGETS, FULL_SETS)
    ex = dict(phase="s2_cal_a16", key="x|" * 6, set="F-EML", algorithm="A", reason="error row", detail="Traceback …")
    r["exclusions"] = [ex] * 500
    t = r["completeness"]
    ph = next(iter(t["phases"].values()))
    r["completeness"] = dict(t, phases={p: dict(ph, missing_keys=["k"] * 30, extra_keys=["e"] * 5)
                                        for p in ("s2_cal_a16", "s2_cal_a0", "s2_cal_mech", "s2_cal_alloc",
                                                  "s2_cal_robust", "s2_cal_pool", "s2_cal_rand", "s2_cal_operon")})
    r["h2c"] = {c: next(iter(res["h2c"].values())) for c in C.CONTROLS} if res["h2c"] else {}
    for k in ("h2", "h2r", "h7", "sensitivity_lower_bound", "h3"):             # both domains, as in the calibration
        if "bool" in r.get(k, {}):
            r[k]["real"] = r[k]["bool"]
    r["h3"] = {d: dict(status="missing: no data", holds=False) for d in ("real", "bool")}
    sub = {k: copy.deepcopy(r[k]) for k in C.SENSITIVITY_KEYS}
    r["sensitivity_cpu_excluded"] = dict(r["sensitivity_cpu_excluded"], recomputed=True, censored_runs=40, **sub)
    return r


def test_reports_stay_within_200_lines_at_calibration_size(tmp_path):
    res = pad(_small(tmp_path))
    assert len(res["h1"]) == 136 and len(res["h8"]) == 72 and len(res["h8b"]) == 36
    assert sum(len(res["holm"][f][d]["members"]) for f in res["holm"] for d in res["holm"][f]) == 103
    st = "docs/status/calibration"
    files = {"report": C.render(res, "t", "s", f"{st}_h1_cells.md", f"{st}_holm_members.md", f"{st}_termination.md",
                                f"{st}_supplement.md", f"{st}_runtime.md", f"{st}_slopes_gaps.md", f"{st}_h8.md"),
             "h1_cells": C.render_h1_cells(res, "t", "s"),
             "holm_members": C.render_holm_members(res, "t", "s", f"{st}_h8.md"),
             "termination": C.render_termination(res, "t", "s"),
             "supplement": C.render_supplement(res, "t", "s"),
             "runtime": C.render_runtime(res, "t", "s"),
             "slopes_gaps": C.render_slopes_gaps(res, "t", "s"),
             "h8": C.render_h8(res, "t", "s")}
    sizes = {k: len(v.splitlines()) for k, v in files.items()}
    assert all(n <= 200 for n in sizes.values()), sizes
    # the padded content is really there (a renderer that dropped rows would pass the limit trivially)
    assert files["h8"].count("\n| ") == 72 + 36 + 2 and files["h1_cells"].count("\n| ") == 136 + 1
    assert files["holm_members"].count("\n| ") == 103 + len(C.FAMILY_ORDER)
    assert files["slopes_gaps"].count("\n| ") == 16 + 1 + 42 + 1 + 14 + 1
    assert files["runtime"].count("\n| ") == 14 + 1 + 60 + 1
    assert files["supplement"].count("\n| ") >= 24 + 60 + 16
    assert "| BOOL-NAND-n5 | B-RS |" in files["report"]

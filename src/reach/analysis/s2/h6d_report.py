"""H6d report driver (D098; design version 2; REQ-ANA-88).

    python -m reach.analysis.s2.h6d_report [--pilot] [--boot B]

Refuses unless every re-run row reproduces its original (`audit.revisit_report`) and carries `rv`, and every set's rows
share one lookup SHA-256 (equal to the registered one once `data/registration/stage3_lookups.json` exists). Computes
the per-set terms at E0 and the H6d tests (`h6d.domain_tests`), the sensitivities, and the negative control (a blind
sampler: its hit probabilities on the calibration targets from s2_cal_rand's hits against its twin from
s2_revisit_rand), then writes data/analysis/h6d.json (pilot: h6d_pilot.json) and docs/status/h6d_report.md
(h6d_pilot_report.md).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from reach.analysis.s2 import h6d as H
from reach.analysis.s2 import law

CONF = dict(e0_phases=("s2_revisit_a0", "s2_revisit_a16"), slope_phases=("s2_revisit_a16",),
            budgets=(20_000, 40_000, 80_000, 160_000, 320_000), out="h6d", confirmatory=True,
            label="STAGE 3 (H6d; confirmatory once registered)")
PILOT = dict(e0_phases=("s2_revisit_pilot2",), slope_phases=("s2_revisit_pilot2",), budgets=(20_000, 40_000, 80_000),
             out="h6d_pilot", confirmatory=False, label="PILOT (pilot targets; not confirmatory)")
LOOKUPS = "data/registration/stage3_lookups.json"
N_REGISTERED = {"H6d-1": 21_000, "H6d-2": 5_250}   # runs entering each test: s2_cal_a0 + _a16 on the 14 main
                                                  # sets (2 x 10,500); s2_cal_a16 on the 7 real main sets (7 x 750)
SECONDARY = {"H6d-1": 0.09, "H6d-2": 0.14}   # secondary family: margins for 80% power at the full pilot
                                             # (0.082 and 0.131, rounded up; D098, before the Stage 3 registration)
CONTROL_MARGIN = 0.5      # bits: the control passes when each domain's mean blind-sampler G lies within this of 0


def _levels(sets, levels):
    if levels is not None:
        return {g: levels[g] for g in sets}
    from reach.revisit import enum_root
    return {g: H.levels_U(enum_root(g)) for g in sets}


def checks(root, phases, confirmatory):
    """Reproduction, rv present, one lookup per set (registered once Stage 3 is filed). Raises on any failure."""
    from reach.analysis.s2.audit import revisit_report
    rows = reproduced = 0
    sha = {}
    for ph in phases:
        rep = revisit_report(root, ph)
        bad = {k: v for e in rep["sources"].values() for k, v in e["differing"].items()}
        missing = sum(len(e["missing"]) for e in rep["sources"].values())
        if bad or missing:
            raise RuntimeError(f"{ph}: {len(bad)} rows do not reproduce their originals, {missing} missing "
                               f"(first: {list(bad.items())[:3]})")
        rows += sum(e["rerun"] for e in rep["sources"].values())
        reproduced += sum(e["identical"] for e in rep["sources"].values())
        for g, rr in H.rows_by_set(root, (ph,)).items():
            sha.setdefault(g, set()).update(r["rv"]["lookup"]["sha256"] for r in rr)
    if any(len(v) != 1 for v in sha.values()):
        raise RuntimeError(f"rows of one set used different lookups: {sorted(g for g, v in sha.items() if len(v) != 1)}")
    reg = Path(root) / LOOKUPS
    if confirmatory and reg.exists():
        want = json.loads(reg.read_text())
        off = sorted(g for g, v in sha.items() if want.get(g) not in v)
        if off:
            raise RuntimeError(f"lookup SHA-256 differs from the registered one for {off}")
    return dict(rows=rows, reproduced=reproduced, lookups={g: next(iter(v)) for g, v in sorted(sha.items())})


def control_set(rows, targets, hits, N, U, U_ok, E=H.E0):
    """The negative control for one set: G of a blind sampler = K50 of its own hit probabilities on the calibration
    targets, 1 - (1 - H/N)^E, minus K50 of its twin (its coverage per level), both by H6's estimator."""
    runs = [dict(target_id=f"s{i}", K=0, bits_sem=0.0, y={}, rv=r["rv"]) for i, r in enumerate(rows)]
    lv, lv_ok = H.cumulative(runs, E, "lv", how="pooled"), H.cumulative(runs, E, "lv_ok", how="pooled")
    if np.ndim(lv) == 0:                                                       # no stream reached every prefix
        return dict(k50_sampler=float("nan"), k50_twin=float("nan"), G=float("nan"), G_eligible=float("nan"),
                    streams=len(rows))
    tg = [dict(K=t.get("K_proven"), bits_sem=t["units"]["bits_sem"], target_id=t["target_id"]) for t in targets]
    own = law._k50([dict(units=dict(bits_sem=t["bits_sem"]), y=1 - (1 - hits.get(t["target_id"], 0) / N) ** E)
                    for t in tg], "bits_sem")
    twin = H.k50_rates(tg, H.twin_probs(lv, U))
    twin_ok = H.k50_rates(tg, H.twin_probs(lv_ok, U_ok))
    return dict(k50_sampler=own, k50_twin=twin, G=own - twin, G_eligible=own - twin_ok, streams=len(rows))


def control_summary(per_set, ctl):
    """Domain means of the blind sampler's G (the control passes when each lies within CONTROL_MARGIN of 0), and G net
    of the control for every set it covers (per-set control values carry the target-sampling bias of uneven hit
    probabilities within a level, which the target bootstrap does not show)."""
    dom = {d: [ctl[g]["G"] for g in s if g in ctl and np.isfinite(ctl[g]["G"])] for d, s in per_set.items()}
    means = {d: float(np.mean(v)) if v else float("nan") for d, v in dom.items()}
    net = {g: per_set[d][g]["G"] - ctl[g]["G"] for d, s in per_set.items() for g in s if g in ctl}
    return dict(domain=means, net=net, margin=CONTROL_MARGIN,
                **{"pass": bool(all(np.isfinite(m) and abs(m) <= CONTROL_MARGIN for m in means.values()))})


def control(root, levels=None):
    """The negative control on every full set with s2_revisit_rand rows; None when the phase has not run."""
    from reach.runner.queue import read_results
    p = Path(root) / "data/runs/s2_revisit_rand/results.jsonl"
    if not p.exists():
        return None
    rr = list(read_results(p).values())
    rand = list(read_results(Path(root) / "data/runs/s2_cal_rand/results.jsonl").values())
    sets = sorted({r["set"] for r in rr})
    lv = _levels(sets, levels)
    out = {}
    for g in sets:
        hits, N = {}, 0
        for r in (x for x in rand if x["set"] == g):
            N += r["n"]
            for t, h in r["hits"].items():
                hits[t] = hits.get(t, 0) + h
        targets = json.loads((Path(root) / f"data/targets_s2/{g}/calibration.json").read_text())
        out[g] = control_set([r for r in rr if r["set"] == g], targets, hits, N, *lv[g])
    return out


def build(root, cfg, boot=H.BOOT, levels=None):
    root = Path(root)
    phases = tuple(dict.fromkeys(cfg["e0_phases"] + cfg["slope_phases"]))
    chk = checks(root, phases, cfg["confirmatory"])
    by = H.rows_by_set(root, cfg["e0_phases"])
    lv = _levels(sorted(by), levels)
    e0 = {d: {g: (H.runs_from_rows(by[g], cfg["budgets"]), lv[g][0]) for g in H.main_sets(by, d)}
          for d in ("real", "bool")}
    sl_by = H.rows_by_set(root, cfg["slope_phases"])
    slope = {g: (H.runs_from_rows(sl_by[g], cfg["budgets"]), lv[g][0]) for g in H.main_sets(sl_by, "real")}
    res = H.domain_tests(e0, cfg["budgets"], boot=boot, slope_sets=slope,
                         secondary=SECONDARY if cfg["confirmatory"] else None)
    if not cfg["confirmatory"]:                       # the pilot: power of each test at the registered sample size
        n_pilot = {"H6d-1": sum(len(rr) for d in e0 for rr, _ in e0[d].values()),
                   "H6d-2": sum(len(rr) for rr, _ in slope.values())}
        res["power"] = {k: H.power(res[k]["estimate"], res["replicates"][k], H.DELTA[k], n_pilot[k], N_REGISTERED[k])
                        for k in ("H6d-1", "H6d-2") if k in res}
    res["sensitivities"] = {g: H.sensitivities(runs, H.E0, U, lv[g][1]) for d in e0 for g, (runs, U) in e0[d].items()}
    res["domain"] = {d: {k: float(np.mean([t[k] for t in s.values()])) for k in ("gap", "R", "S", "G", "O", "k50_twin")}
                     for d, s in res["per_set"].items()}
    res["control"] = control(root, levels) if cfg["confirmatory"] or (root / "data/runs/s2_revisit_rand").exists() else None
    res["control_summary"] = control_summary(res["per_set"], res["control"]) if res["control"] else None
    res.update(checks=chk, label=cfg["label"], phases=list(phases), budgets=list(cfg["budgets"]), boot=boot,
               built=time.strftime("%Y-%m-%d %H:%M:%S"))
    return res


def _f(x, d=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:+.{d}f}"


def render(res):
    L = [f"# H6d report: {res['label']}", "",
         f"> Generated by `reach.analysis.s2.h6d_report` ({res['built']}); every number from the JSON beside it. "
         f"{'Pilot data on pilot targets: not confirmatory.' if 'PILOT' in res['label'] else ''} Do not edit by hand.",
         "", f"Rows {res['checks']['rows']}, reproducing their originals {res['checks']['reproduced']}; "
         f"bootstrap {res['boot']} replicates.", "", "## Tests (equivalence, Holm over the two)", "",
         "| Test | estimate | 90% interval | margin | p | Holm threshold | estimable | supported |", "|---|---|---|---|---|---|---|---|"]
    for k in ("H6d-1", "H6d-2"):
        if k in res:
            t = res[k]
            L.append(f"| {k} | {_f(t['estimate'])} | [{_f(t['lo90'])}, {_f(t['hi90'])}] | ±{t['delta']} | {t['p']:.4f} | "
                     f"{t['threshold']:.4f} | {'yes' if t['estimable'] else 'no'} | {'yes' if t['supported'] else 'no'} |")
    if res.get("secondary"):
        L += ["", "Secondary family (other margins, the same replicates; Holm over the secondary family only):"]
        for k, s in res["secondary"].items():
            L.append(f"- {k}: margin ±{s['delta']}, 90% interval [{_f(s['lo90'])}, {_f(s['hi90'])}], p {s['p']:.4f}, "
                     f"supported {'yes' if s['supported'] else 'no'}")
    if res.get("power"):
        L += ["", "Power at the registered sample size (the pilot's estimate taken as true; SE scaled by sqrt(n_pilot / "
              "n_registered)):"]
        for k, s in res["power"].items():
            L.append(f"- {k}: estimate {_f(s['estimate'])}, registered SE {s['se']:.3f}, power at ±{s['delta']} "
                     f"{s['power']:.2f}; margin for 80% power {s['delta_80']:.2f}")
    L += ["", "## Terms at E₀ (bits; gap = G − S − R + O)", "", "| Domain | Set | gap | G | S | R | O | K₅₀ twin |",
          "|---|---|---|---|---|---|---|---|"]
    for d, s in res["per_set"].items():
        for g, t in sorted(s.items()):
            L.append(f"| {d} | {g} | {_f(t['gap'])} | {_f(t['G'])} | {_f(t['S'])} | {_f(t['R'])} | {_f(t['O'])} | "
                     f"{_f(t['k50_twin'])} |")
        m = res["domain"][d]
        L.append(f"| {d} | mean | {_f(m['gap'])} | {_f(m['G'])} | {_f(m['S'])} | {_f(m['R'])} | {_f(m['O'])} | "
                 f"{_f(m['k50_twin'])} |")
    L += ["", "## Sensitivities at E₀ (G under other choices; descriptive)", "",
          "| Set | pooled | carry | never hit | eligible | interpolated | undefined share | key/hit disagree |",
          "|---|---|---|---|---|---|---|---|"]
    for g, s in sorted(res["sensitivities"].items()):
        L.append(f"| {g} | {_f(s['G_pooled'])} | {_f(s['G_carry'])} | {_f(s['G_never_hit'])} | {_f(s['G_eligible'])} | "
                 f"{_f(s['G_interp'])} | {s['undef_share']:.3f} | {s['key_hit_disagree']} |")
    if res.get("control"):
        L += ["", "## Negative control: a blind sampler (G must lie near 0)", "",
              "| Set | streams | K₅₀ sampler | K₅₀ twin | G | G eligible |", "|---|---|---|---|---|---|"]
        for g, c in sorted(res["control"].items()):
            L.append(f"| {g} | {c['streams']} | {_f(c['k50_sampler'])} | {_f(c['k50_twin'])} | {_f(c['G'])} | "
                     f"{_f(c['G_eligible'])} |")
        s = res.get("control_summary")
        if s:
            L += ["", f"Domain means of the control's G: " + ", ".join(f"{d} {_f(m)}" for d, m in s["domain"].items())
                  + f" (margin ±{s['margin']}): {'passes' if s['pass'] else 'FAILS'}. G net of the control: "
                  + ", ".join(f"{g} {_f(v)}" for g, v in sorted(s["net"].items())) + "."]
    return "\n".join(L) + "\n"


def write(root, res, cfg):
    root = Path(root)
    (root / "data/analysis").mkdir(parents=True, exist_ok=True)
    (root / f"data/analysis/{cfg['out']}.json").write_text(json.dumps(res, indent=1, default=float))
    (root / "docs/status").mkdir(parents=True, exist_ok=True)
    (root / f"docs/status/{cfg['out']}_report.md").write_text(render(res))


def main(argv=None):
    from reach.paths import ROOT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--boot", type=int, default=H.BOOT)
    a = ap.parse_args(argv)
    cfg = PILOT if a.pilot else CONF
    res = build(ROOT, cfg, a.boot)
    write(ROOT, res, cfg)
    print(f"written data/analysis/{cfg['out']}.json; H6d-1 supported: {res['H6d-1']['supported']}"
          + (f"; H6d-2 supported: {res['H6d-2']['supported']}" if "H6d-2" in res else ""))


if __name__ == "__main__":
    main()

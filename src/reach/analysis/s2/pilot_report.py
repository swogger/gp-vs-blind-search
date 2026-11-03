"""Pilot report (Phase 3; gate G2 inputs). Every number is computed here from the run records; nothing is typed in.

    .venv/bin/python -m reach.analysis.s2.pilot_report        -> docs/status/pilot_report.md, data/analysis/pilot.json

Sections: phase status and costs; reach per set at 1/4-4xE0 (nodes, bits_sem; target-bootstrap CIs); budget slope;
model fit (Weibull, lognormal); mechanisms (A, B, headless chicken at E0); random sampling; proposed levels and margins
with the power check (docs/plan/phases/phase3_pilot.md). Pilot data are not confirmatory.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from reach.analysis.k50 import k50, logit
from reach.analysis.s2 import theory
from reach.analysis.s2.levels import level_rates, suggest_levels
from reach.analysis.s2.load import E0, read_phase, tidy

BUDGETS = (0.25, 0.5, 1, 2, 4)
PHASES = ("s2_pilot_reach_a", "s2_pilot_reach_b", "s2_pilot_mech", "s2_pilot_long", "s2_pilot_rand_a",
          "s2_pilot_rand_b")
SHIFT = {"real": 1.0, "bool": 2.0}             # margin rule: curve shift in nodes (docs/plan/phases/phase3_pilot.md)
BOOT = 300


def _status(root):
    out = {}
    for ph in PHASES:
        p = Path(root) / "data/runs" / ph
        if not (p / "manifest.json").exists():
            continue
        n = json.loads((p / "manifest.json").read_text())["n_tasks"]
        rows = read_phase(ph, root)
        st = defaultdict(int)
        for r in rows:
            st[r.get("terminal_state")] += 1
        cpu = [r.get("cpu_s", 0) for r in rows]
        out[ph] = dict(tasks=n, done=len(rows), states=dict(st), cpu_mean=float(np.mean(cpu)) if cpu else None,
                       cpu_total_h=float(np.sum(cpu) / 3600) if cpu else 0.0)
    return out


def _k50_table(recs, unit):
    rows = {}
    for b in BUDGETS:
        rr = [r for r in recs if r["budget"] == b and unit in (r["units"] or {}) or (unit == "nodes" and r["budget"] == b)]
        if not rr:
            continue
        size = [r["K"] if unit == "nodes" else r["units"][unit] for r in rr]
        est, lo, hi, slope = k50(size, [r["y"] for r in rr], [r["target_id"] for r in rr], boot=BOOT)
        rows[b] = dict(k50=est, lo=lo, hi=hi, slope=slope, n=len(rr), rate=float(np.mean([r["y"] for r in rr])))
    return rows


def _per_doubling(tab):
    pts = [(math.log2(b), v["k50"]) for b, v in tab.items() if np.isfinite(v["k50"])]
    if len(pts) < 3:
        return float("nan")
    x, y = np.array(pts).T
    return float(np.polyfit(x, y, 1)[0])


def _shift_margin(recs, shift):
    """Mean log-loss increase when the pooled logistic curve (nodes, E0) is shifted by `shift` nodes."""
    K = np.array([r["K"] for r in recs], float)
    y = np.array([r["y"] for r in recs], float)
    if len(set(y)) < 2:
        return float("nan")
    b0, b1 = logit(K, y)
    p = 1 / (1 + np.exp(-(b0 + b1 * K)))
    q = 1 / (1 + np.exp(-(b0 + b1 * (K - shift))))
    ll = lambda pp: -(y * np.log(np.clip(pp, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - pp, 1e-12, 1))).mean()   # noqa
    return float(ll(q) - ll(p))


def build(root=None):
    from reach.paths import ROOT
    from reach import grammar as G
    root = Path(root or ROOT)
    res = dict(status=_status(root), sets={}, mech={}, rand={}, proposals={})
    reach_rows = read_phase("s2_pilot_reach_a", root) + read_phase("s2_pilot_reach_b", root)
    by_set = defaultdict(list)
    for r in reach_rows:
        by_set[r.get("set") or r.get("meta", {}).get("set")].append(r)
    for g, rows in sorted(by_set.items()):
        recs = tidy(rows, BUDGETS)
        dom = G.get(g).kind
        ent = dict(domain=dom, runs=len(rows), nodes=_k50_table(recs, "nodes"), bits_sem=_k50_table(recs, "bits_sem"))
        ent["nodes_per_doubling"] = _per_doubling(ent["nodes"])
        ent["bits_per_doubling"] = _per_doubling(ent["bits_sem"])
        runs = [dict(set=g, target_id=r.get("target_id"), b=(r.get("units") or {}).get("bits_sem"), gen_exact=r.get("gen_exact"),
                     gens_reached=r.get("gens_reached"), generations=r.get("generations"))
                for r in rows if "gens_reached" in r and (r.get("units") or {}).get("bits_sem") is not None]
        if sum(1 for r in runs if r["gen_exact"] is not None) >= 20:
            fw, fl = theory.fit(runs, "weibull"), theory.fit(runs, "lognormal")
            ent["model"] = dict(kappa=fw["kappa"], alpha=fw["alpha"], t0=fw["t0"], bits_per_doubling=fw["bits_per_doubling"],
                                sigma_target=fw["sigma_target"], gen0_hits=fw["gen0_hits"],
                                best_run_length=theory.best_run_length(fw), loglik_weibull=fw["loglik"],
                                loglik_lognormal=fl["loglik"])
        e0 = [r for r in recs if r["budget"] == 1]
        ent["levels_E0"] = level_rates(e0)
        if len({r["y"] for r in e0}) == 2:
            b0, b1 = logit([r["K"] for r in e0], [r["y"] for r in e0])
            if b1 < 0:
                ent["suggested_main_levels"] = suggest_levels(sorted({r["K"] for r in e0}), b0, b1, n=10)
        res["sets"][g] = ent
    for dom in ("real", "bool"):
        e0 = [r for g, rows in by_set.items() if G.get(g).kind == dom for r in tidy(rows, (1,))]
        if e0:
            res["proposals"][dom] = dict(delta_H2=_shift_margin(e0, SHIFT[dom]), shift_nodes=SHIFT[dom],
                                         delta_hc=1.0, delta_rs=1.0, epsilon_set=0.10)
    mech = read_phase("s2_pilot_mech", root)
    for g in sorted({r.get("set") for r in mech}):
        out = {}
        for algo in ("B", "HC"):
            rr = tidy([r for r in mech if r.get("set") == g and r.get("algorithm") == algo], (1,))
            if rr:
                out[algo] = k50([r["K"] for r in rr], [r["y"] for r in rr], [r["target_id"] for r in rr], boot=BOOT)[:3]
        a = tidy([r for r in reach_rows if r.get("set") == g], (1,))
        if a:
            out["A"] = k50([r["K"] for r in a], [r["y"] for r in a], [r["target_id"] for r in a], boot=BOOT)[:3]
        res["mech"][g] = out
    for ph in ("s2_pilot_rand_a", "s2_pilot_rand_b"):
        for r in read_phase(ph, root):
            hits = r.get("hits") or {}
            res["rand"][r.get("set")] = dict(n=r.get("n"), targets_hit=sum(1 for v in hits.values() if v),
                                             p_hit_E0_mean=float(np.mean([1 - (1 - v / r["n"]) ** E0 for v in hits.values()]))
                                             if hits else None, undefined_share=r.get("undefined_share"))
    return res


def render(res):
    f = lambda x, d=1: "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"   # noqa
    L = ["# Pilot report (Phase 3)", "", "> Index: `docs/README.md`. Generated by `reach.analysis.s2.pilot_report`; "
         "values in `data/analysis/pilot.json`. **Pilot data: not confirmatory.** Do not edit by hand.", "",
         "## Phases", "", "| Phase | done / tasks | terminal states | mean CPU s | CPU-h |", "|---|---|---|---|---|"]
    for ph, s in res["status"].items():
        L.append(f"| {ph} | {s['done']} / {s['tasks']} | {s['states']} | {f(s['cpu_mean'])} | {f(s['cpu_total_h'])} |")
    L += ["", "## Reach K₅₀ by budget (regime A; target-bootstrap 95% CI)", "",
          "| Set | unit | " + " | ".join(f"{b}×E₀" for b in BUDGETS) + " | per doubling |", "|---|---|" + "---|" * (len(BUDGETS) + 1)]
    for g, e in res["sets"].items():
        for unit, key in (("nodes", "nodes_per_doubling"), ("bits_sem", "bits_per_doubling")):
            cells = [f"{f(e[unit][b]['k50'])} [{f(e[unit][b]['lo'])}, {f(e[unit][b]['hi'])}]" if b in e[unit] else "–"
                     for b in BUDGETS]
            L.append(f"| {g} | {unit} | " + " | ".join(cells) + f" | {f(e[key], 2)} |")
    L += ["", "## Analytic model (Weibull form; lognormal alternative)", "",
          "| Set | κ | α | t₀ | σ target | gen-0 hits | bits/doubling (κ/α) | best run length L* (gen) | loglik Weibull | "
          "loglik lognormal |", "|---|---|---|---|---|---|---|---|---|---|"]
    for g, e in res["sets"].items():
        m = e.get("model")
        if m:
            L.append(f"| {g} | {f(m['kappa'], 2)} | {f(m['alpha'], 2)} | {f(m['t0'])} | {f(m.get('sigma_target'), 2)} | "
                     f"{m.get('gen0_hits', '–')} | {f(m['bits_per_doubling'], 2)} | "
                     f"{f(m['best_run_length'])} | {f(m['loglik_weibull'])} | {f(m['loglik_lognormal'])} |")
    L += ["", "## Mechanisms at E₀ (K₅₀ in nodes, [CI])", "", "| Set | A | B | HC |", "|---|---|---|---|"]
    for g, m in res["mech"].items():
        c = lambda k: f"{f(m[k][0])} [{f(m[k][1])}, {f(m[k][2])}]" if k in m else "–"              # noqa
        L.append(f"| {g} | {c('A')} | {c('B')} | {c('HC')} |")
    L += ["", "## Random sampling", "", "| Set | trees | targets hit | mean P(hit within E₀) | undefined share |",
          "|---|---|---|---|---|"]
    for g, r in res["rand"].items():
        L.append(f"| {g} | {r['n']} | {r['targets_hit']} | {f(r['p_hit_E0_mean'], 3)} | {f(r['undefined_share'], 3)} |")
    L += ["", "## Proposals for gate G2", ""]
    for dom, p in res["proposals"].items():
        L.append(f"- **{dom}:** Δ (H2) = {f(p['delta_H2'], 4)} (mean log-loss increase from a {p['shift_nodes']}-node "
                 f"shift); δ_hc = δ_rs = {p['delta_hc']} node; ε_set = {p['epsilon_set']}.")
    for g, e in res["sets"].items():
        if e.get("suggested_main_levels"):
            L.append(f"- {g}: main levels {e['suggested_main_levels']}")
    return "\n".join(L) + "\n"


def main():
    from reach.paths import ROOT
    res = build()
    (ROOT / "data/analysis").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/analysis/pilot.json").write_text(json.dumps(res, indent=1, default=float))
    (ROOT / "docs/status/pilot_report.md").write_text(render(res))
    print(render(res))


if __name__ == "__main__":
    main()

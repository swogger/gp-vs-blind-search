"""Pilot P1 report: reach per grammar in nodes, syntactic bits and semantic bits, against the
size-ordered enumeration null (D009), plus the budget, seeded-leaf and cost probes.

    .venv/bin/python -m reach.analysis.pilot_report [--phases pilot_p1b,pilot_p1] [--out docs/results/pilot_p1_report.md]

Pilot data are not confirmatory (docs/03). Sizes above a grammar's enumeration ceiling are
constructed (upper bounds), and their semantic bits are extrapolated; both are marked.
"""
import argparse
import json
import math
import time
from collections import defaultdict

import numpy as np

from reach import grammar as G
from reach.analysis.k50 import k50
from reach.paths import ROOT
from reach.runner.queue import Phase, fmt_dur


class Sizes:
    """Size units for one grammar from its enumeration counts."""

    def __init__(self, name):
        self.g = G.get(name)
        lv = json.loads((ROOT / f"data/enumeration/{name}/counts.json").read_text())["levels"]
        self.ceiling = max(l["K"] for l in lv)
        self.U = {l["K"]: l["U"] for l in lv}
        self.Ucum = {l["K"]: l["U_cum"] for l in lv}
        nz = [l for l in lv if l["U"] > 0]
        self.cap = float("inf")
        if self.g.kind == "bool":                 # saturates at 2^(2^n) functions: fit the growth region
            n_fun = 2 ** (2 ** self.g.n_inputs)
            self.cap = math.log2(n_fun)
            nz = [l for l in nz if 10 <= l["U_cum"] <= n_fun / 2] or nz
        nz = nz[-4:]
        ks, ys = np.array([l["K"] for l in nz], float), np.log2([l["U_cum"] for l in nz])
        self.bits_per_node, self.icept = np.polyfit(ks, ys, 1)
        self.syn = np.cumsum(G.syntactic_counts(self.g, 80))

    def bits_sem(self, K):
        if K <= self.ceiling:
            return math.log2(self.Ucum[K])
        return float(min(self.cap, self.icept + self.bits_per_node * K))

    def bits_syn(self, K):
        return math.log2(self.syn[K])

    def null_p(self, K, E):
        """Size-ordered enumeration over distinct functions, random order within a level."""
        if K > self.ceiling or not self.U.get(K):
            return float("nan")
        below = self.Ucum[K] - self.U[K]
        return float(min(1.0, max(0.0, (E - below) / self.U[K])))

    def null_k50_nodes(self, E):
        """Size at which log2 U(<=K) reaches log2 E (linear interpolation over non-empty levels)."""
        pts = [(k, math.log2(self.Ucum[k])) for k in sorted(self.Ucum) if self.U[k] > 0]
        target = math.log2(E)
        for (k0, y0), (k1, y1) in zip(pts, pts[1:]):
            if y0 <= target <= y1:
                return k0 + (target - y0) / (y1 - y0) * (k1 - k0)
        return float("nan") if target < pts[0][1] else self.ceiling + (target - self.bits_sem(self.ceiling)) / self.bits_per_node


def fmt_ci(t):
    est, lo, hi = t[:3]
    return "–" if math.isnan(est) else f"{est:.1f} [{lo:.1f}, {hi:.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="pilot_p1b,scout_bool5,pilot_p1",
                    help="comma list; each grammar is taken from the first phase that has it")
    ap.add_argument("--out", default=str(ROOT / "docs/results/pilot_p1_report.md"))
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    rows, owner, n_total = [], {}, 0
    for name in a.phases.split(","):
        ph = Phase(name)
        if not ph.manifest_path.exists():
            continue
        res = list(ph.results().values())
        n_total += ph.manifest()["n_tasks"]
        for g in {r["grammar"] for r in res}:
            owner.setdefault(g, name)
        rows += [dict(r, phase=name) for r in res if owner[r["grammar"]] == name]
    conv = lambda r: float(r.get("converged") == "true")
    exact = lambda r: float(r.get("exact_hash_match") == "true")
    by = defaultdict(list)
    for r in rows:
        by[(r["exp"], r["grammar"])].append(r)
    grammars = sorted({r["grammar"] for r in rows}, key=lambda g: (g.startswith("BOOL"), g))
    sizes = {g: Sizes(g) for g in grammars}
    E0 = 100 * 200
    L = [f"# Pilot P1 report ({a.phases})", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `reach.analysis.pilot_report` from "
         f"phases {a.phases} ({len(rows)} runs used). **Pilot data: not confirmatory.** Grammar → phase: "
         + ", ".join(f"{g}: {p}" for g, p in sorted(owner.items())) + ".",
         "", "Clean engine (D010, D017): Regime A, population 100, no seeded leaf, symmetric depth limit, no size "
         "penalty. E₀ = 2×10⁴ evaluations. Censored (CPU-limit) runs count as failures. K₅₀ CIs: 1,000 target "
         "resamples. Semantic bits = log₂ U(≤K); above a grammar's enumeration ceiling, K is constructed and "
         "its bits are extrapolated. The null is size-ordered enumeration over distinct functions (D009): "
         "its reach in semantic bits is log₂ E₀ = 14.3.", "",
         "## Overview (scout, E₀)", "",
         "| Grammar | Ceiling K | Semantic bits/node | GP K₅₀ nodes | GP K₅₀ syntactic bits | GP K₅₀ semantic bits "
         "| Null K₅₀ nodes | GP − null (semantic bits) | Exact: K₅₀ nodes | Exact: GP − null (bits) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    detail = []
    for g in grammars:
        rr = by.get(("scout", g), [])
        if not rr:
            continue
        s = sizes[g]
        y = np.array([conv(r) for r in rr])
        cl = [r["target_id"] for r in rr]
        Ks = np.array([r["K"] for r in rr], float)
        kn = k50(Ks, y, cl, a.boot)
        ksyn = k50([s.bits_syn(int(k)) for k in Ks], y, cl, a.boot)
        ksem = k50([s.bits_sem(int(k)) for k in Ks], y, cl, a.boot)
        nk = s.null_k50_nodes(E0)
        gap = ksem[0] - math.log2(E0) if not math.isnan(ksem[0]) else float("nan")
        ye = np.array([exact(r) for r in rr])
        ken = k50(Ks, ye, cl, a.boot)
        kes = k50([s.bits_sem(int(k)) for k in Ks], ye, cl, a.boot)
        L.append(f"| {g} | {s.ceiling} | {s.bits_per_node:.2f} | {fmt_ci(kn)} | {fmt_ci(ksyn)} | {fmt_ci(ksem)} "
                 f"| {nk:.1f} | {gap:+.1f} | {fmt_ci(ken)} | {kes[0] - math.log2(E0):+.1f} |")
        detail += [f"### {g}", "", "| K | Size | Targets | Runs | GP conv | GP exact | Null P(success) | Censored | Mean CPU s |",
                   "|---|---|---|---|---|---|---|---|---|"]
        for K in sorted({r["K"] for r in rr}):
            kr = [r for r in rr if r["K"] == K]
            cen = sum(r["terminal_state"] == "timeout" for r in kr)
            err = sum(r["terminal_state"] == "error" for r in kr)
            npv = s.null_p(K, E0)
            detail.append(f"| {K} | {'proven' if K <= s.ceiling else 'constructed'} | {len({r['target_id'] for r in kr})} "
                          f"| {len(kr)} | {np.mean([conv(r) for r in kr]):.3f} | {np.mean([exact(r) for r in kr]):.3f} "
                          f"| {'–' if math.isnan(npv) else f'{npv:.3f}'} | {cen}{f' (+{err} errors)' if err else ''} "
                          f"| {np.mean([r['cpu_s'] for r in kr]):.1f} |")
        detail.append("")
    L += ["", "GP − null > 0 means GP's reach exceeds enumeration's at the same budget.", "", "## Per grammar (scout)", ""] + detail

    scout_idx = {(r["target_id"], r["seed"]): r for r in rows if r["exp"] == "scout"}
    L += ["## Budget probe: E₁ = 8×10⁴ against E₀ (same targets and seeds)", "",
          "| Grammar | K | Runs | Conv E₀ | Conv E₁ | Censored E₁ | Mean CPU s E₁ |", "|---|---|---|---|---|---|---|"]
    for g in grammars:
        rr = by.get(("e1", g), [])
        for K in sorted({r["K"] for r in rr}):
            kr = [r for r in rr if r["K"] == K]
            base = [scout_idx[(r["target_id"], r["seed"])] for r in kr if (r["target_id"], r["seed"]) in scout_idx]
            L.append(f"| {g} | {K} | {len(kr)} | {np.mean([conv(r) for r in base]) if base else float('nan'):.3f} "
                     f"| {np.mean([conv(r) for r in kr]):.3f} | {sum(r['terminal_state'] == 'timeout' for r in kr)} "
                     f"| {np.mean([r['cpu_s'] for r in kr]):.1f} |")
    L += ["", "## Seeded single-variable leaf: on against off (paired by target and seed, E₀)", "",
          "| Grammar | K | Pairs | Conv off | Conv on | Only off / only on |", "|---|---|---|---|---|---|"]
    for g in grammars:
        rr = by.get(("seedleaf", g), [])
        if rr:
            pairs = [(scout_idx[(r["target_id"], r["seed"])], r) for r in rr if (r["target_id"], r["seed"]) in scout_idx]
            off, on = [conv(p[0]) for p in pairs], [conv(p[1]) for p in pairs]
            L.append(f"| {g} | {rr[0]['K']} | {len(pairs)} | {np.mean(off):.3f} | {np.mean(on):.3f} "
                     f"| {sum(o > n for o, n in zip(off, on))} / {sum(n > o for o, n in zip(off, on))} |")
    L += ["", "## Cost probe: 3,200 generations (E₂ = 3.2×10⁵), one seed", "",
          "| Grammar | K | Runs | Conv | Timeouts | Mean CPU s | Max CPU s |", "|---|---|---|---|---|---|---|"]
    for g in grammars:
        rr = by.get(("e2", g), [])
        if rr:
            L.append(f"| {g} | {rr[0]['K']} | {len(rr)} | {np.mean([conv(r) for r in rr]):.3f} "
                     f"| {sum(r['terminal_state'] == 'timeout' for r in rr)} | {np.mean([r['cpu_s'] for r in rr]):.0f} "
                     f"| {max(r['cpu_s'] for r in rr):.0f} |")
    srows = []
    for name in ("budget_sweep", "budget_sweep_bool5"):
        sw = Phase(name)
        if sw.manifest_path.exists():
            srows += list(sw.results().values())
    if srows:
        L += ["", "## Scaling (budget_sweep, budget_sweep_bool5): reach at E₀, 4×E₀ and 16×E₀ on the same targets, levels and seeds", "",
              "Fitted on the sweep levels only (scout rows for E₀). Slope = semantic bits of reach gained per "
              "doubling of E (least squares over the three budgets); blind enumeration has slope 1. "
              "† = midpoint outside the measured sizes (extrapolated; the slope is then unreliable).", "",
              "| Grammar | Levels | K₅₀ nodes E₀ / 4× / 16× | K₅₀ semantic bits E₀ / 4× / 16× | Slope (bits per doubling) | Slope (nodes per doubling) | Runs done |",
              "|---|---|---|---|---|---|---|"]
        for g in sorted({r["grammar"] for r in srows}):
            s = sizes.get(g) or Sizes(g)
            lv = sorted({r["K"] for r in srows if r["grammar"] == g})
            per = {200: [r for r in by.get(("scout", g), []) if r["K"] in lv]}
            for gens in (800, 3200):
                per[gens] = [r for r in srows if r["grammar"] == g and r["generations"] == gens]
            kn, kb = [], []
            for gens in (200, 800, 3200):
                rr = per[gens]
                if not rr:
                    kn.append((float("nan"),) * 3); kb.append((float("nan"),) * 3); continue
                y, cl = np.array([conv(r) for r in rr]), [r["target_id"] for r in rr]
                kn.append(k50([r["K"] for r in rr], y, cl, a.boot))
                kb.append(k50([s.bits_sem(r["K"]) for r in rr], y, cl, a.boot))
            xs = [math.log2(100 * gn) for gn, k in zip((200, 800, 3200), kb) if not math.isnan(k[0])]
            ys = [k[0] for k in kb if not math.isnan(k[0])]
            slope = np.polyfit(xs, ys, 1)[0] if len(xs) >= 2 else float("nan")
            xn = [math.log2(100 * gn) for gn, k in zip((200, 800, 3200), kn) if not math.isnan(k[0])]
            slope_n = np.polyfit(xn, [k[0] for k in kn if not math.isnan(k[0])], 1)[0] if len(xn) >= 2 else float("nan")
            mark = lambda v: f"{v:.1f}" + ("†" if not (min(lv) <= v <= max(lv)) else "")
            L.append(f"| {g} | {','.join(map(str, lv))} | {' / '.join(mark(k[0]) for k in kn)} "
                     f"| {' / '.join(fmt_ci(k) for k in kb)} | {slope:.2f} | {slope_n:.2f} | {len(per[800]) + len(per[3200])} |")
    states = defaultdict(lambda: defaultdict(int))
    cpu = defaultdict(float)
    for r in rows:
        states[r["exp"]][r["terminal_state"]] += 1
        cpu[r["exp"]] += r["cpu_s"]
    L += ["", "## Terminations and cost", "", "| Experiment | Runs | Terminal states | Total CPU |", "|---|---|---|---|"]
    for e in sorted(states):
        L.append(f"| {e} | {sum(states[e].values())} | {', '.join(f'{k} {v}' for k, v in sorted(states[e].items()))} "
                 f"| {fmt_dur(cpu[e])} |")
    L += ["", "Mean CPU seconds per scout run at E₀, by grammar: " + ", ".join(
        f"{g} {np.mean([r['cpu_s'] for r in by[('scout', g)]]):.1f}" for g in grammars if by.get(("scout", g)))]
    (ROOT / "docs/results").mkdir(exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"report -> {a.out} ({len(rows)} runs)")


if __name__ == "__main__":
    main()

"""Synthetic dense-K analysis, applied identically to the original runs and the rerun.

    python3 research/rerun/analyze_p1.py > results_rerun/analysis_P1.txt
"""
import csv, math, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SYN = os.path.join(ROOT, "results", "synthetic_dense_k")
sys.path.insert(0, os.path.join(HERE, "..", "synthetic"))
import analysis_ncrit_vs_k as M  # noqa: E402  (same model code as Table 12)

OLD = (os.path.join(SYN, "synthetic_runs.csv"), os.path.join(SYN, "synthetic_runs_lowk.csv"))
NEW = (os.path.join(ROOT, "results_rerun", "P1_synth_main.csv"),
       os.path.join(ROOT, "results_rerun", "P1b_synth_lowk.csv"))
TG = {t["target_id"]: t for t in csv.DictReader(open(os.path.join(SYN, "synthetic_targets.csv")))}
RNG = np.random.default_rng(20260926)


def read(path):
    return list(csv.DictReader(open(path)))


def logit(K, y, iters=300):
    X = np.column_stack([np.ones(len(K)), K.astype(float)]); b = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b)); W = np.clip(p * (1 - p), 1e-9, None)
        nb = np.linalg.solve((X * W[:, None]).T @ X, (X * W[:, None]).T @ (X @ b + (y - p) / W))
        if np.max(np.abs(nb - b)) < 1e-12:
            return nb
        b = nb
    return b


def k50(runs, reg, metric, boot=1000):
    rr = [r for r in runs if r["regime"] == reg]
    y = lambda r: float(r["converged"] == "true" and (metric == "conv" or r["exact_hash_match"] == "true"))
    K = np.array([float(r["K_construction"]) for r in rr]); Y = np.array([y(r) for r in rr])
    b = logit(K, Y); est = -b[0] / b[1]
    by_t = defaultdict(list)
    for i, r in enumerate(rr):
        by_t[r["target_id"]].append(i)
    ids = list(by_t)
    bs = []
    for _ in range(boot):
        idx = np.concatenate([by_t[ids[j]] for j in RNG.integers(0, len(ids), len(ids))])
        bb = logit(K[idx], Y[idx]); bs.append(-bb[0] / bb[1])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi


def table11(old, new):
    print("TABLE 11  convergence (conv) and exact-hash recovery (exact): original -> rerun")
    print(" K  | A conv       A exact      | B conv       B exact")
    ks = sorted({int(r["K_construction"]) for r in new})
    for k in ks:
        cells = []
        for reg in "AB":
            o = [r for r in old if int(r["K_construction"]) == k and r["regime"] == reg]
            n = [r for r in new if int(r["K_construction"]) == k and r["regime"] == reg]
            f = lambda rs, ex: sum(r["converged"] == "true" and (not ex or r["exact_hash_match"] == "true") for r in rs) / max(1, len(rs))
            cells.append(f"{f(o,0):.3f}->{f(n,0):.3f}  {f(o,1):.3f}->{f(n,1):.3f}")
        print(f" {k:2d} | {cells[0]} | {cells[1]}")


def discordance(new):
    print("\nTABLE 13  paired A vs B on identical seeds (rerun): A-only, B-only, exact binomial p, Holm")
    by = defaultdict(dict)
    for r in new:
        by[(r["target_id"], r["seed"], int(r["K_construction"]))][r["regime"]] = r["converged"] == "true"
    rows = []
    for k in sorted({x[2] for x in by}):
        a = sum(v.get("A") and not v.get("B") for kk, v in by.items() if kk[2] == k)
        b = sum(v.get("B") and not v.get("A") for kk, v in by.items() if kk[2] == k)
        p = binomtest(a, a + b, 0.5).pvalue if a + b else 1.0
        rows.append([k, a, b, p])
    order = sorted(range(len(rows)), key=lambda i: rows[i][3]); m = len(rows); run = 0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * rows[i][3])); rows[i].append(run)
    for k, a, b, p, h in rows:
        print(f"  K={k:2d}  A-only {a:3d}  B-only {b:3d}  p={p:.2g}  Holm={h:.3g}")
    print(f"  pooled A-only {sum(r[1] for r in rows)}  B-only {sum(r[2] for r in rows)}")


def bands(new):
    print("\nTABLE 14  final nMSE, paired Wilcoxon A vs B (rerun)")
    by = defaultdict(dict)
    for r in new:
        by[(r["target_id"], r["seed"], int(r["K_construction"]))][r["regime"]] = float(r["final_nmse"])
    for lo, hi in ((7, 11), (13, 17), (19, 31)):
        pairs = [(v["A"], v["B"]) for k, v in by.items() if lo <= k[2] <= hi and "A" in v and "B" in v]
        a, b = np.array(pairs).T
        d = a - b
        p = wilcoxon(d[d != 0]).pvalue if np.any(d != 0) else 1.0
        na, nb = int(np.sum(d < 0)), int(np.sum(d > 0))
        print(f"  K={lo}-{hi}: median A {np.median(a):.3g}  median B {np.median(b):.3g}  p={p:.2g}  "
              f"(A lower in {na} pairs, B lower in {nb}, ties {int(np.sum(d == 0))})")


def models(new):
    rows = []
    for r in new:
        t = TG.get(r["target_id"])
        if t is None:
            continue
        rows.append({"target_id": r["target_id"], "regime": r["regime"],
                     "y": 1.0 if r["converged"] == "true" else 0.0,
                     "K": float(r["K_construction"]), "alpha": float(t["alpha_rigidity"]),
                     "n_crit": float(t["n_crit"]),
                     "g": 1.0 if t["gradient_absence_g"] == "true" else 0.0,
                     "rigid": t["rigid_subset"] == "true"})
    print("\nTABLE 12  run-level logistic models (rerun; same code as the original Table 12)")
    for regime in ("both", "A", "B"):
        rr = rows if regime == "both" else [r for r in rows if r["regime"] == regime]
        print(f" regime {regime}:")
        for cols in (["K"], ["n_crit"], ["K", "alpha"], ["K", "alpha", "g"]):
            M.show(" + ".join(cols), M.fit(rr, cols, groups=[r["target_id"] for r in rr]))


def misc(new):
    conv = [r for r in new if r["converged"] == "true"]
    loose = [r for r in conv if r["exact_hash_match"] != "true"]
    fail = [r for r in new if r["converged"] != "true"]
    print(f"\nMISC (rerun): runs {len(new)}, converged {len(conv)}, exact {len(conv)-len(loose)}, loose {len(loose)}")
    for reg in "AB":
        c = [r for r in conv if r["regime"] == reg]
        f = [r for r in fail if r["regime"] == reg]
        exact_min = sum(int(r["final_tree_size"]) == int(r["K_construction"]) for r in c)
        exc = [int(r["final_tree_size"]) - int(r["K_construction"]) for r in f]
        print(f"  Regime {reg}: converged at exactly K nodes {exact_min}/{len(c)}; failures above K "
              f"{sum(e > 0 for e in exc)/max(1,len(exc)):.3f}, median excess {np.median(exc):.0f}, "
              f"single-leaf {sum(int(r['final_tree_size']) == 1 for r in f)}")
    ref = [r for r in new if r["target_id"] == "reviewer_ce" and r["regime"] == "A"]
    print(f"  reference case eml(x, eml(eml(x,x),x)) Regime A: {sum(r['converged']=='true' for r in ref)}/{len(ref)}")
    st = defaultdict(int)
    for r in new:
        st[r.get("status", r["terminal_state"])] += 1
    cpu = [float(r.get("cpu_s") or 0) for r in new]
    print(f"  terminal states {dict(st)}; max CPU {max(cpu):.1f}s")


def main():
    old, oldlow = read(OLD[0]), read(OLD[1])
    new, newlow = read(NEW[0]), read(NEW[1])
    table11(old + oldlow, new + newlow)
    print("\nK50 (logistic on pre-specified K=7-31, 95% CI resampling targets)")
    for metric in ("conv", "exact"):
        for reg in "AB":
            o, n = k50(old, reg, metric), k50(new, reg, metric)
            print(f"  {metric:5s} Regime {reg}: original {o[0]:.2f} [{o[1]:.2f}, {o[2]:.2f}]  ->  "
                  f"rerun {n[0]:.2f} [{n[1]:.2f}, {n[2]:.2f}]")
    discordance(new)
    bands(new)
    models(new)
    misc(new)


if __name__ == "__main__":
    main()

"""Independent cross-check of the P1 synthetic numbers (different code path from analyze_p1.py).

A. pandas recomputation of Table 11 / Table 13 counts, and key-set equality with the original design.
B. K50 by direct maximum likelihood (scipy.optimize) with 2,000 target resamples and a different seed.
C. Determinism + meaning of `exact_hash_match`: re-run a sample of converged runs, recover the final
   tree, and test functional equivalence on a dense grid over (2, 3).

    python3 research/rerun/crosscheck_p1.py [n_sample]
"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SYN = os.path.join(ROOT, "results", "synthetic_dense_k")
sys.path.insert(0, os.path.join(HERE, "..", "synthetic"))
import eml_synth_core as C  # noqa: E402

new = pd.read_csv(os.path.join(ROOT, "results_rerun", "P1_synth_main.csv"), dtype={"seed": str, "final_nmse": str})
old = pd.read_csv(os.path.join(SYN, "synthetic_runs.csv"), dtype={"seed": str})
for df in (new, old):
    df["conv"] = df["converged"].astype(str).str.lower() == "true"
    df["exact"] = df["conv"] & (df["exact_hash_match"].astype(str).str.lower() == "true")

print("A. design integrity")
kn = set(zip(new.target_id, new.regime, new.seed)); ko = set(zip(old.target_id, old.regime, old.seed))
print(f"   rerun keys {len(kn)}, original keys {len(ko)}, identical sets: {kn == ko}")
t11 = new.groupby(["K_construction", "regime"])[["conv", "exact"]].mean().unstack().round(3)
print("   Table 11 (pandas):\n" + t11.to_string())
w = new.pivot_table(index=["target_id", "seed", "K_construction"], columns="regime", values="conv").reset_index()
disc = w.assign(Aonly=(w.A == 1) & (w.B == 0), Bonly=(w.B == 1) & (w.A == 0)).groupby("K_construction")[["Aonly", "Bonly"]].sum()
print("   Table 13 discordant (pandas):", disc.to_dict("index"), "pooled", int(disc.Aonly.sum()), int(disc.Bonly.sum()))


def mle_k50(K, y):
    nll = lambda b: np.sum(np.logaddexp(0, b[0] + b[1] * K) - y * (b[0] + b[1] * K))
    b = minimize(nll, x0=[3.0, -0.3], method="BFGS").x
    return -b[0] / b[1]


SKIP_B = False
print("\nB. K50 by direct MLE, 2,000 target resamples, seed 7")
rng = np.random.default_rng(7)
for metric in (() if SKIP_B else ("conv", "exact")):
    for reg in "AB":
        d = new[new.regime == reg]
        K, y = d.K_construction.to_numpy(float), d[metric].to_numpy(float)
        groups = d.groupby("target_id").indices; ids = list(groups)
        est = mle_k50(K, y)
        bs = []
        for _ in range(2000):
            idx = np.concatenate([groups[ids[j]] for j in rng.integers(0, len(ids), len(ids))])
            bs.append(mle_k50(K[idx], y[idx]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"   {metric:5s} Regime {reg}: {est:.2f} [{lo:.2f}, {hi:.2f}]")

n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 60
print(f"\nC. determinism and functional meaning of exact_hash_match ({n_sample} exact + {n_sample} loose converged runs)")
targets = {t["target_id"]: t for t in json.load(open(os.path.join(SYN, "targets.json")))}
conv = new[new.conv]
sample = pd.concat([conv[conv.exact].sample(n_sample, random_state=1),
                    conv[~conv.exact].sample(min(n_sample, int((~conv.exact).sum())), random_state=1)])
xs = np.array([complex(2 + (i + 0.6180339887) / 2000) for i in range(2000)])


def one(r):
    cmd = [sys.executable, os.path.join(HERE, "synth_task.py"), "--targets", os.path.join(SYN, "targets.json"),
           "--target_id", r.target_id, "--regime", r.regime, "--seed", str(r.seed),
           "--no_size_penalty", "--cpu_clock"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    d = json.loads([l for l in out.splitlines() if l.startswith("RESULT ")][-1][7:])
    same = float(d["final_nmse"]) == float(r.final_nmse)          # both parsed by Python float()
    if not same:
        print(f"   MISMATCH {r.target_id} {r.regime} {r.seed}: recorded {r.final_nmse} rerun {d['final_nmse']}", flush=True)
    if not d.get("best_expr"):
        return r.exact, same, None
    f = C.vec_eval(C.parse_expr(targets[r.target_id]["expression"]), xs)
    g = C.vec_eval(C.parse_expr(d["best_expr"]), xs)
    rel = np.max(np.abs(g - f) / np.maximum(1.0, np.abs(f)))
    return r.exact, same, bool(rel <= 1e-9)


with ThreadPoolExecutor(2) as ex:
    res = list(ex.map(one, [r for r in sample.itertuples()]))
det = sum(s for _, s, _ in res)
print(f"   deterministic reproduction: {det}/{len(res)} identical final nMSE")
for flag in (True, False):
    sub = [f for e, _, f in res if e == flag and f is not None]
    print(f"   hash {'exact' if flag else 'loose'}: dense-grid equivalent {sum(sub)}/{len(sub)}")

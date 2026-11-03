"""Calibrate the synthetic instrument against the main study's real targets.

Two comparisons, at the main study's pop=100 / G=200 cell:

 1. same-function: two real targets have an exact synthetic counterpart --
    exp(x) = eml(x, 1) = K03_01, and ln(x) = eml(x, eml(eml(x, x), 1)) = K07_07.
    Identical function, identical GP configuration; the only differences are the
    sampling domain and the convergence threshold's definition.  Each synthetic
    run is therefore ALSO re-scored under the main study's absolute
    MSE <= 1e-6 rule (absolute MSE = final_nmse * Var(target)) so the two
    criteria can be put on a common footing.

 2. band placement: where each real target's rate falls in the distribution of
    per-target rates at its K.  A real target is one target, not a K band, so
    the per-target distribution is the right reference -- not the band mean.
"""

from __future__ import annotations

import collections
import csv
import glob
import os

import numpy as np
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "results", "synthetic_dense_k")

# EML K from research/test_suite.EML_K: exp=3 and ln=7 are proven minimal,
# '+'=19 is proven minimal.
REAL = {"1.1": ("exp(x)", 3), "1.2": ("ln(x)", 7), "1.3": ("x1+x2", 19)}
SAME_FUNCTION = {"1.1": ("K03_01", "eml(x, 1)"),
                 "1.2": ("K07_07", "eml(x, eml(eml(x, x), 1))")}


def real_rates():
    out = {}
    for regime, d in (("A", "standard_gp"), ("B", "mutation_only")):
        agg = collections.defaultdict(lambda: [0, 0])
        for f in sorted(glob.glob(os.path.join(ROOT, "results", d, "results_200_100_*.csv"))):
            for row in csv.DictReader(open(f, encoding="utf-8")):
                if row["Test ID"] in REAL:
                    agg[row["Test ID"]][1] += 1
                    if "CONVERGED" in row["EML Status"]:
                        agg[row["Test ID"]][0] += 1
        for tid, (c, n) in agg.items():
            out[(tid, regime)] = (c, n)
    return out


def load_runs():
    rows = []
    for f in ("synthetic_runs.csv", "synthetic_runs_lowk.csv"):
        p = os.path.join(RES, f)
        if os.path.exists(p):
            rows += list(csv.DictReader(open(p, encoding="utf-8")))
    return rows


def main():
    real = real_rates()
    runs = load_runs()
    var = {}
    for f in ("synthetic_targets_extra.csv", "synthetic_targets_lowk_extra.csv"):
        p = os.path.join(RES, f)
        if os.path.exists(p):
            for t in csv.DictReader(open(p, encoding="utf-8")):
                var[t["target_id"]] = float(t["var_target"])

    # ---- 1. same-function -------------------------------------------------
    rows = []
    for tid_real, (tgt_id, expr) in SAME_FUNCTION.items():
        name, K = REAL[tid_real]
        v = var[tgt_id]
        for regime in ("A", "B"):
            rr = [r for r in runs if r["target_id"] == tgt_id and r["regime"] == regime]
            nm = [float(r["final_nmse"]) for r in rr]
            s_norm = sum(x <= 1e-6 for x in nm)
            s_abs = sum(x * v <= 1e-6 for x in nm)
            rc, rn = real[(tid_real, regime)]
            p_norm = float(fisher_exact([[rc, rn - rc], [s_norm, len(rr) - s_norm]])[1])
            p_abs = float(fisher_exact([[rc, rn - rc], [s_abs, len(rr) - s_abs]])[1])
            rows.append([name, tid_real, tgt_id, expr, K, regime, rc, rn, repr(rc / rn),
                         len(rr), s_norm, repr(s_norm / len(rr)), repr(p_norm),
                         s_abs, repr(s_abs / len(rr)), repr(p_abs), repr(v),
                         repr(1e-6 / v)])
    with open(os.path.join(RES, "calibration_same_function.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "main_study_test_id", "synthetic_target_id", "expression",
                    "K_eml", "regime", "real_converged", "real_runs", "real_conv_rate",
                    "synth_runs", "synth_converged_nmse_rule", "synth_conv_rate_nmse_rule",
                    "fisher_p_vs_real_nmse_rule", "synth_converged_abs_mse_rule",
                    "synth_conv_rate_abs_mse_rule", "fisher_p_vs_real_abs_rule",
                    "var_target", "nmse_equivalent_of_abs_1e-6"])
        w.writerows(rows)
    print("SAME-FUNCTION (identical target, pop=100 G=200)")
    print(f"  {'target':<8}{'reg':<5}{'real':<16}{'synth nMSE<=1e-6':<20}"
          f"{'synth absMSE<=1e-6':<20}{'p(norm)':<10}{'p(abs)':<8}")
    for r in rows:
        print(f"  {r[0]:<8}{r[5]:<5}{r[6]}/{r[7]} = {r[6]/r[7]:<8.2f}"
              f"{r[10]}/{r[9]} = {float(r[11]):<12.2f}"
              f"{r[13]}/{r[9]} = {float(r[14]):<12.2f}{float(r[12]):<10.3f}{float(r[15]):<8.3f}")

    # ---- 2. band placement ------------------------------------------------
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in runs:
        per[(int(r["K_construction"]), r["regime"])][r["target_id"]].append(
            r["converged"] == "true")
    brows = []
    for (tid, regime), (c, n) in sorted(real.items()):
        name, K = REAL[tid]
        d = per.get((K, regime))
        rv = c / n
        if d is None:
            brows.append([tid, name, K, regime, c, n, repr(rv), "", "", "", "", "", "",
                          "no synthetic level at this K"])
            continue
        rates = sorted(float(np.mean(v)) for v in d.values())
        allr = [x for v in d.values() for x in v]
        pct = 100.0 * sum(1 for x in rates if x < rv) / len(rates)
        brows.append([tid, name, K, regime, c, n, repr(rv), len(rates),
                      repr(float(np.mean(allr))), repr(rates[0]),
                      repr(rates[len(rates) // 2]), repr(rates[-1]), repr(pct),
                      "within synthetic per-target range"
                      if rates[0] <= rv <= rates[-1] else "OUTSIDE synthetic range"])
    with open(os.path.join(RES, "calibration_real_targets.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["test_id", "target", "K_eml", "regime", "real_converged", "real_runs",
                    "real_conv_rate", "synth_n_targets", "synth_band_conv_rate",
                    "synth_per_target_min", "synth_per_target_median",
                    "synth_per_target_max", "real_percentile_within_synth", "placement"])
        w.writerows(brows)
    print("\nBAND PLACEMENT")
    for r in brows:
        if r[7] == "":
            print(f"  {r[1]:<8} K={r[2]:<3}{r[3]}  real={float(r[6]):.3f}   {r[13]}")
        else:
            print(f"  {r[1]:<8} K={r[2]:<3}{r[3]}  real={float(r[6]):.3f}  "
                  f"synthetic per-target min/med/max = {float(r[9]):.2f}/"
                  f"{float(r[10]):.2f}/{float(r[11]):.2f}  "
                  f"{float(r[12]):.0f}th pct  {r[13]}")


if __name__ == "__main__":
    main()

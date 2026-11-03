"""Build synthetic_summary_by_K.csv (File 3) and run_metadata.json."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import eml_synth_core as C     # noqa: E402
import make_targets as M       # noqa: E402
from seeds import SEEDS        # noqa: E402

Z = 1.959963984540054   # 95%


def wilson(k, n, z=Z):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def sha1_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha1(fh.read()).hexdigest()


def summarize(res_dir, runs_name, targets_name, out_name):
    runs = list(csv.DictReader(open(os.path.join(res_dir, runs_name), encoding="utf-8")))
    targets = {t["target_id"]: t for t in
               csv.DictReader(open(os.path.join(res_dir, targets_name), encoding="utf-8"))}

    cells = {}
    for r in runs:
        key = (int(r["K_construction"]), r["regime"])
        c = cells.setdefault(key, {"runs": [], "targets": set()})
        c["runs"].append(r)
        c["targets"].add(r["target_id"])

    rows = []
    for (K, regime) in sorted(cells):
        c = cells[(K, regime)]
        n_runs = len(c["runs"])
        n_conv = sum(1 for r in c["runs"] if r["converged"] == "true")
        lo, hi = wilson(n_conv, n_runs)
        gens = [int(r["gens_to_converge"]) for r in c["runs"] if r["gens_to_converge"] != ""]
        alphas = [float(targets[t]["alpha_rigidity"]) for t in c["targets"]]
        gs = [targets[t]["gradient_absence_g"] == "true" for t in c["targets"]]
        rows.append([
            K, regime, len(c["targets"]), n_runs, repr(n_conv / n_runs),
            repr(lo), repr(hi), repr(statistics.mean(alphas)),
            repr(statistics.mean(1.0 if g else 0.0 for g in gs)),
            repr(statistics.median(gens)) if gens else "",
        ])

    path = os.path.join(res_dir, out_name)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["K_construction", "regime", "n_targets", "n_runs", "conv_rate",
                    "wilson_lo", "wilson_hi", "mean_alpha", "mean_gradient_absence",
                    "median_gens_to_converge"])
        w.writerows(rows)
    return rows, runs


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    res = os.path.join(root, "results", "synthetic_dense_k")

    rows, runs = summarize(res, "synthetic_runs.csv", "synthetic_targets.csv",
                           "synthetic_summary_by_K.csv")
    print(f"{len(rows)} (K, regime) cells over {len(runs)} runs")
    for r in rows:
        print(f"  K={r[0]:>2} {r[1]}  n={r[3]:>3}  conv={float(r[4]):.3f} "
              f"[{float(r[5]):.3f}, {float(r[6]):.3f}]  mean_alpha={float(r[7]):.3f}")

    if os.path.exists(os.path.join(res, "synthetic_runs_rigid_oversample.csv")):
        summarize(res, "synthetic_runs_rigid_oversample.csv",
                  "synthetic_targets_rigid_oversample.csv",
                  "synthetic_summary_by_K_rigid_oversample.csv")

    gen_log = json.load(open(os.path.join(res, "target_generation_log.json"), encoding="utf-8"))
    agreement = None
    ag_path = os.path.join(res, "evaluator_agreement.json")
    if os.path.exists(ag_path):
        agreement = json.load(open(ag_path, encoding="utf-8"))

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                           capture_output=True, text=True).stdout.strip()

    meta = {
        "experiment": "Dense-K synthetic-target experiment (R3.8 / R3.6 / R3.5)",
        "git_commit": commit,
        "git_worktree_dirty": bool(dirty),
        "eml_gp_version": {
            "note": "EML-GP has no version string; the search is ga-lib + the "
                    "reference-cell loop of research/deduce_formula.py, pinned by "
                    "these source hashes.",
            "ga-lib/node.py": sha1_file(os.path.join(root, "ga-lib", "node.py")),
            "ga-lib/generator.py": sha1_file(os.path.join(root, "ga-lib", "generator.py")),
            "ga-lib/evolution.py": sha1_file(os.path.join(root, "ga-lib", "evolution.py")),
            "research/deduce_formula.py": sha1_file(
                os.path.join(root, "research", "deduce_formula.py")),
            "research/synthetic/eml_synth_core.py": sha1_file(
                os.path.join(here, "eml_synth_core.py")),
            "research/synthetic/make_targets.py": sha1_file(
                os.path.join(here, "make_targets.py")),
        },
        "generation_rng_seed": M.GENERATION_RNG_SEED,
        "k_levels": gen_log["k_levels"],
        "targets_per_k": gen_log["targets_per_k"],
        "seeds_per_target_regime": 10,
        "seed_set": SEEDS[1:11],
        "seed_source": "research/seeds.py (the main study's seed list), SEEDS[1:11]; "
                       "the same 10 seeds are used for regime A and regime B, so runs "
                       "are paired within (target, seed).",
        "population": C.POP_SIZE,
        "generations": C.GENERATIONS,
        "node_cap": C.MAX_NODES,
        "max_depth": C.MAX_DEPTH,
        "timeout_seconds": C.TIMEOUT_SECONDS,
        "batch_size": C.BATCH_SIZE,
        "plateau_patience_generations": C.PLATEAU_GENS,
        "plateau_note": "eml_plateau_mult for the Phase-1 EML operator set is 12, so "
                        "patience = max(50, 200//10) * 12 = 600 generations > the "
                        "200-generation budget: plateau stopping never fires, exactly "
                        "as at the main study's reference cell. terminal_state "
                        "'early_stopped' is therefore expected to be absent.",
        "domain": {"lo": C.DOMAIN_LO, "hi": C.DOMAIN_HI, "variable": "x",
                   "note": "open interval (2, 3)"},
        "n_points": C.N_POINTS,
        "definition_sample": "100 evenly spaced points on (2, 3), endpoints excluded "
                             "(x_i = 2 + (i+1)/101). Used for target viability, the "
                             "novelty hash, all structural measurements "
                             "(alpha, g, best_proper_subtree_nmse) and exact_hash_match.",
        "training_sample": "Per generation, BATCH_SIZE=75 points drawn uniformly at "
                           "random from (2, 3) by the run's evolution RNG -- the "
                           "reference implementation's stochastic mini-batch, with the "
                           "domain changed to (2, 3). The target is an exact expression, "
                           "so it is evaluated pointwise on each batch.",
        "validation_sample": "One held-out sample of N=100 points drawn uniformly at "
                             "random from (2, 3) by random.Random(generation_rng_seed+1), "
                             "i.e. seed 20260722. It is identical for every target, "
                             "regime and seed, is disjoint from the definition grid, and "
                             "is never used for selection -- only for the convergence "
                             "test and final_nmse. (The reference implementation redraws "
                             "validation per master seed; a single fixed held-out sample "
                             "was used instead so final_nmse is directly comparable "
                             "across all runs. The per-run validation seed is still drawn "
                             "and discarded so the pop/evo RNG streams match the "
                             "reference derivation exactly.)",
        "convergence_metric": "nMSE = MSE / Var(target), where MSE is "
                              "mean |prediction - target|^2 over the held-out validation "
                              "sample with non-finite residuals charged 1000 each "
                              "(deduce_formula._mse semantics), and Var(target) = "
                              "mean |t - mean(t)|^2 on the definition grid.",
        "convergence_threshold": C.CONV_THRESHOLD,
        "fitness_normalisation": "Selection fitness is the same nMSE. Normalising by a "
                                 "per-target constant does not change selection order; "
                                 "it makes the reference cell's absolute thresholds "
                                 "(parsimony bonus at 1e-5, convergence at 1e-6) mean "
                                 "the same thing for targets whose magnitudes span many "
                                 "orders of magnitude.",
        "exact_hash_match": "sha1 of the candidate's output on the definition grid, "
                            "real and imaginary parts each divided by the target scale "
                            "max(1, max|target|) and rounded to 9 decimals, compared "
                            "with the same hash of the target. Relative (not absolute) "
                            "rounding, so it means 'same function to ~9 significant "
                            "digits' regardless of target magnitude.",
        "audit_minimality_max_k": M.AUDIT_MINIMALITY_MAX_K,
        "audit_minimality_tool": "eml-skill/.../eml_core/minimality.audit_minimality, "
                                 "leaves ('1','x'), precision 9, exhaustive over every "
                                 "EML tree with fewer operators. K_proven is populated "
                                 "only for K <= 11; above that K_construction is an "
                                 "upper bound.",
        "target_generation": {
            "sampling": "Uniform over EML trees with exactly m = (K-1)/2 operators, "
                        "leaves i.i.d. uniform on {x, 1}, subject to ga-lib depth <= 10. "
                        "The depth bound is the GP's own limit (mutate enforces "
                        "absolute_max_depth = MAX_DEPTH = 10), so it guarantees every "
                        "target lies inside the reachable search space. For m <= 6 the "
                        "space is enumerated exhaustively and shuffled; above that it is "
                        "sampled with Catalan-weighted, depth-bounded uniform sampling.",
            "acceptance_filters": [
                "viability: <=5% non-finite on the definition grid, Var >= 1e-12, "
                "max|v|/min|v| <= 1e6",
                "self-consistency (strengthening of viability): the ground-truth tree "
                "must be finite on every definition and validation point, so the target "
                "itself scores nMSE = 0. Without this a target could be unconvergeable "
                "by construction and a 0% result would not be attributable to search. "
                "All accepted targets therefore have overflow_fraction = 0.",
                "novelty: sha1 of the round-9 output vector on the definition grid, "
                "deduplicated globally across all K levels",
                "irreducibility (dead-subtree): for every proper subtree, substitute "
                "each probe in {x, 1, 2.5, 0.5, eml(x,x)} whose own output differs from "
                "the subtree's; if any substitution leaves nMSE <= 1e-6 the subtree is "
                "dead at the experiment's own convergence tolerance and the target is "
                "rejected",
                "minimality (K <= 11 only): exhaustive audit over all smaller EML trees; "
                "targets provably expressible with fewer operators are rejected and "
                "resampled, so every accepted K <= 11 target has K_proven == "
                "K_construction",
            ],
            "log": gen_log,
        },
        "rigid_oversample": {
            "note": "The pre-registered 30-targets-per-K random sample yields only 2-6 "
                    "rigid targets (alpha >= 0.5 and g) at several K. Extra rigid-only "
                    "targets were collected after each K's quota was filled, to a floor "
                    "of 8 rigid targets per K, and written to SEPARATE files so Files 1-2 "
                    "remain exactly the pre-registered design.",
            "files": ["synthetic_targets_rigid_oversample.csv",
                      "synthetic_runs_rigid_oversample.csv",
                      "synthetic_summary_by_K_rigid_oversample.csv"],
            "warning": "Do NOT pool these with File 1 / File 2 when computing the "
                       "P(converge) vs K curve -- they are selected on rigidity and "
                       "would bias it. Use them only to strengthen the rigid-subset arm.",
        },
        "alpha_rigidity_definition": "For each internal (operator) node, apply every "
                                     "applicable single local edit -- swap the node's two "
                                     "children, collapse the node to its left child, "
                                     "collapse to its right child, flip a leaf child "
                                     "1<->x -- discarding edits whose subtree output is "
                                     "unchanged, and take the MEDIAN resulting whole-tree "
                                     "nMSE. The node is rigid if that median > 10, i.e. "
                                     "more than one order of magnitude worse than the "
                                     "mean-predictor baseline nMSE = 1 (nMSE(T*) = 0 "
                                     "exactly, so the baseline must be the trivial "
                                     "predictor, not T*). alpha = rigid nodes / m. "
                                     "synthetic_targets_extra.csv also carries the "
                                     "min-over-edits and swap-only variants.",
        "gradient_absence_g_definition": "g = 1 iff best_proper_subtree_nmse >= "
                                         "trivial_leaf_nmse, where the minimum runs over "
                                         "proper subtrees rooted at operator nodes and "
                                         "trivial_leaf_nmse = min(nMSE(x), nMSE(1)). "
                                         "i.e. no partial structure beats the trivial "
                                         "leaves.",
        "n_crit_definition": "alpha_rigidity * m_operators.",
        "evaluator": {
            "note": "Search operators are ga-lib verbatim. Fitness evaluation is "
                    "vectorised with numpy over the sample instead of looping cmath per "
                    "point; this keeps the 200-generation budget (not wall clock) the "
                    "binding constraint, so no run is truncated by the 600 s timeout.",
            "agreement": agreement,
        },
        "confound_direction": "Construction-K over-estimates true K for K > 11 (no "
                              "minimality proof is available there), and over-estimated "
                              "targets are easier than their nominal K implies. Any "
                              "observed drop in convergence with K is therefore a "
                              "conservative estimate of the boundary: the true K50 can "
                              "only be lower, not higher.",
        "reviewer_ce": {
            "expression": M.REVIEWER_CE,
            "target_id": "reviewer_ce",
            "K_construction": 7,
            "K_proven": 7,
            "forced_inclusion": True,
            "measured": {"alpha_rigidity": 1.0, "alpha_min_rule": 0.6666666666666666,
                         "alpha_swap_rule": 0.6666666666666666,
                         "gradient_absence_g": True, "rigid_subset": True,
                         "conv_rate_A": 0.7, "conv_rate_B": 0.3,
                         "percentile_within_K7_A": 45, "percentile_within_K7_B": 32},
            "finding": "It is an unremarkable member of the K = 7 band: 7/10 in regime A "
                       "and 3/10 in regime B, ranking 17th and 19th of 31 targets, below "
                       "the band median in both. It is rigid (alpha >= 0.5 under all "
                       "three rules) and gradient-free (g = true), so conditions (i)-(iii) "
                       "hold for it empirically. It is NOT an outlier and should not be "
                       "presented as easier or harder than a generic K = 7 target.",
            "footnote": "One neutral caveat: it is the one target exempted from the "
                        "dead-subtree filter. Substituting the constant 2.5 for the x at "
                        "path RR leaves nMSE = 2.7e-7, below the 1e-6 threshold, because "
                        "that leaf contributes -ln(x) ~ 1 to a quantity of magnitude "
                        "1e3-1e8 that is then passed through a logarithm. It is exactly "
                        "minimal at K = 7 under the 1e-9 audit -- this is a property of "
                        "the working tolerance, not a defect in the counterexample.",
        },
        "rigidity_does_not_predict_beyond_scale": {
            "warning": "The alpha_rigidity, gradient_absence_g and n_crit columns are "
                       "reported as measured, but they do NOT carry predictive weight "
                       "beyond K in this data. Do not write condition (ii)/(iii) as if "
                       "they do.",
            "conv_K_auc": 0.9136, "conv_n_crit_auc": 0.7470,
            "conv_K_deviance": 3075.9, "conv_n_crit_deviance": 4426.0,
            "alpha_coefficient_given_K": "+0.69 (cluster-robust p = 0.022) -- POSITIVE, "
                                         "i.e. the opposite sign from what condition (ii) "
                                         "predicts",
            "g_coefficient_given_K": "~0 (p = 0.79 / 0.69 / 0.99)",
            "K_by_rigid_interaction": "coef -0.072, se 0.138, p = 0.604 -- there is no "
                                      "evidence the rigid subset declines faster with K. "
                                      "A separately fitted rigid-subset slope looks "
                                      "steeper only because the rigid subset is 39-40% of "
                                      "targets at K = 7/9 but 7-17% at higher K.",
            "high_K_floor": "At K >= 21, rigid targets converge at 0.000 and non-rigid at "
                            "0.005: non-rigid targets fail at ~0% too.",
            "robustness": "Holds under all three alpha rules (n_crit AUC 0.747 / 0.574 / "
                          "0.802 for median / min / swap-only, vs 0.914 for K alone).",
            "caveat": "alpha is measured only on accepted targets, and the dead-subtree "
                      "filter removes the most reducible ones, restricting alpha's range. "
                      "That is an attenuation argument; it does not explain the sign.",
            "detail": "analysis_scale_vs_rigidity.txt",
        },
        "low_k_extension": {
            "note": "POST-HOC, outside the pre-registration. K = 3 and K = 5 were added "
                    "after the main run to anchor the upper asymptote of the logistic fit "
                    "and to give the main study's K = 3 flagship a synthetic counterpart. "
                    "Both spaces are tiny (4 and 16 trees), enumerated exhaustively with "
                    "no quota and no sampling, so there is no RNG dependence.",
            "files": ["synthetic_targets_lowk.csv", "synthetic_runs_lowk.csv",
                      "synthetic_summary_by_K_lowk.csv"],
            "results": {"K3_A": 1.0, "K3_B": 0.833, "K5_A": 0.94, "K5_B": 0.53},
            "warning": "Keep separate from Files 1-2 when reporting the pre-registered "
                       "curve; label as post-hoc wherever plotted.",
        },
        "calibration_against_real_targets": {
            "note": "Two of the main study's real targets have an EXACT synthetic "
                    "counterpart -- exp(x) = eml(x, 1) = K03_01 and ln(x) = "
                    "eml(x, eml(eml(x, x), 1)) = K07_07 -- so they can be compared "
                    "function-for-function at the same reference cell, not just by band "
                    "placement. x1+x2 has no counterpart: it is two-variable, while every "
                    "synthetic target is univariate over {x, 1}.",
            "result": "exp(x) agrees (A 1.00 vs 1.00; B 1.00 vs 0.90, p = 0.25). For "
                      "ln(x) the synthetic criterion is 75x stricter in absolute terms "
                      "(Var = 0.0133, so nMSE <= 1e-6 means absolute MSE <= 1.3e-8 where "
                      "the main study allows 1e-6). Re-scored under the main study's own "
                      "absolute rule, synthetic ln(x) is 10/10 in both regimes: regime A "
                      "then agrees with the real 23/30 (p = 0.16), regime B does not "
                      "(real 6/30, p < 0.001).",
            "interpretation": "The instrument reproduces the phenomenon and the location "
                              "of the transition, but per-target rates are domain- and "
                              "threshold-sensitive. The real ln(x) is sampled on "
                              "(0.1, 5.0), which includes the steep near-zero region, "
                              "versus the benign (2, 3) here. Present the overlay as "
                              "'same phenomenon, same transition location', NOT as "
                              "'identical rates'.",
            "files": ["calibration_same_function.csv", "calibration_real_targets.csv"],
        },
        "terminal_states": ["converged", "early_stopped", "timeout", "budget_exhausted"],
        "files": {
            "synthetic_runs.csv": "File 1 (contract) -- one row per run",
            "synthetic_targets.csv": "File 2 (contract) -- one row per target",
            "synthetic_summary_by_K.csv": "File 3 (contract) -- one row per (K, regime)",
            "synthetic_targets_extra.csv": "supplementary per-target diagnostics "
                                           "(alpha variants, Var(target), depth, hash)",
        },
    }
    path = os.path.join(res, "run_metadata.json")
    json.dump(meta, open(path, "w", encoding="utf-8"), indent=1)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()

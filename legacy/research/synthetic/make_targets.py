"""Dense-K synthetic target generation, filtering and structural measurement.

Every target is an EML tree over {x, 1} with exactly m = (K-1)/2 operators, so
expressibility is guaranteed by construction and K is known.  Acceptance
filters run in the protocol's order: viability -> novelty -> irreducibility
(dead-subtree, plus exhaustive minimality audit for K <= 11).

Writes:
    synthetic_targets.csv        (output contract, File 2)
    synthetic_targets_extra.csv  (supplementary diagnostics, not in contract)
    targets.json                 (machine-readable input for run_synthetic.py)
    target_generation_log.json   (rejection accounting for run_metadata.json)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eml_synth_core as C  # noqa: E402

_SKILL_SHARED = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..",
                 "eml-skill", "eml-skill", "skills", "_shared")
)
if _SKILL_SHARED not in sys.path:
    sys.path.insert(0, _SKILL_SHARED)
from eml_core.minimality import audit_minimality  # noqa: E402

GENERATION_RNG_SEED = 20260721
K_LEVELS_FULL = [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
K_LEVELS_MINIMAL = [7, 11, 15, 19, 23, 27, 31]
TARGETS_PER_K_FULL = 30
TARGETS_PER_K_MINIMAL = 20
AUDIT_MINIMALITY_MAX_K = 11
REVIEWER_CE = "eml(x, eml(eml(x, x), x))"

# Viability thresholds (protocol section 2)
MAX_NONFINITE_FRACTION = 0.05
MIN_VARIANCE = 1e-12
MAX_MAGNITUDE_RATIO = 1e6

# Rigidity: nMSE = 1 is the mean-predictor baseline, so "degrades by more than
# one order of magnitude" == the perturbed tree scores worse than 10.
RIGIDITY_THRESHOLD = 10.0
DEAD_SUBTREE_NMSE = C.CONV_THRESHOLD   # a substitution that still "converges"
ENUMERATION_LIMIT_M = 6                # exhaustive below this, sampled above

# Pitfall 9 ("oversample at high K for rigid quota"): the pre-registered 30/K
# random sample yields only 2-6 rigid targets at some K, which under-powers the
# rigid-subset arm of R3.8.  Extra rigid-only targets are therefore collected
# *after* each K's quota is filled and written to SEPARATE files, so Files 1-2
# remain exactly the pre-registered design and the oversample is opt-in.
RIGID_OVERSAMPLE_MIN = 8


# --------------------------------------------------------------------------
# structural helpers
# --------------------------------------------------------------------------
def node_paths(tree, prefix=()):
    """Yield (path, node) for every node; path () is the root."""
    yield prefix, tree
    if tree.is_operator:
        if tree.left is not None:
            yield from node_paths(tree.left, prefix + ("L",))
        if tree.right is not None:
            yield from node_paths(tree.right, prefix + ("R",))


def replace_at(tree, path, new_sub):
    if not path:
        return new_sub
    out = tree.copy()
    cur = out
    for step in path[:-1]:
        cur = cur.left if step == "L" else cur.right
    if path[-1] == "L":
        cur.left = new_sub
    else:
        cur.right = new_sub
    return out


def flip_leaf(node):
    return C.leaf(1.0) if isinstance(node.value, str) else C.leaf("x")


def enumerate_trees_m(m, max_h):
    """All EML trees with exactly m operators, leaves in {x, 1}, depth <= max_h."""
    if m == 0:
        if max_h >= 1:
            yield C.leaf("x")
            yield C.leaf(1.0)
        return
    if max_h < 2:
        return
    for i in range(m):
        for a in enumerate_trees_m(i, max_h - 1):
            for b in enumerate_trees_m(m - 1 - i, max_h - 1):
                yield C.eml_node(a.copy(), b.copy())


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------
def viability(vals):
    """Returns (ok, reason, overflow_fraction, variance)."""
    finite = np.isfinite(vals.real) & np.isfinite(vals.imag)
    overflow_fraction = float(1.0 - finite.mean())
    if overflow_fraction > MAX_NONFINITE_FRACTION:
        return False, "overflow", overflow_fraction, float("nan")
    good = vals[finite]
    if good.size == 0:
        return False, "overflow", overflow_fraction, float("nan")
    var = C.target_variance(good)
    if not np.isfinite(var) or var < MIN_VARIANCE:
        return False, "variance", overflow_fraction, var
    mag = np.abs(good)
    if mag.min() <= 0.0 or (mag.max() / mag.min()) > MAX_MAGNITUDE_RATIO:
        return False, "magnitude_ratio", overflow_fraction, var
    return True, "", overflow_fraction, var


def dead_subtree(tree, grid, target_vals, var):
    """True if some proper subtree can be substituted without moving the
    sampled output off the convergence criterion -- i.e. the target's effective
    K is lower than its construction K."""
    probes = [C.leaf("x"), C.leaf(1.0), C.leaf(2.5), C.leaf(0.5),
              C.eml_node(C.leaf("x"), C.leaf("x"))]
    probe_vals = [C.vec_eval(p, grid) for p in probes]
    probe_hashes = [C.func_hash(v) for v in probe_vals]
    for path, node in node_paths(tree):
        if not path:
            continue  # root is not a proper subtree
        own = C.func_hash(C.vec_eval(node, grid))
        for probe, ph in zip(probes, probe_hashes):
            if ph == own:
                continue
            cand = replace_at(tree, path, probe.copy())
            nm = C.mse_against(C.vec_eval(cand, grid), target_vals) / var
            if nm <= DEAD_SUBTREE_NMSE:
                return True
    return False


def proven_minimal(target_vals, grid, K):
    """Exhaustive audit over every EML tree with fewer operators.
    Returns (is_minimal, found_at_k)."""
    xs = [complex(v) for v in grid]
    res = audit_minimality(
        tuple(complex(v) for v in target_vals),
        xs=xs, ys=xs, max_k=K - 2, precision=C.HASH_PRECISION,
        binary=False, leaves=("1", "x"),
    )
    return res["found_at_k"] is None, res["found_at_k"]


# --------------------------------------------------------------------------
# structural measurements
# --------------------------------------------------------------------------
def measure(tree, grid, target_vals, var):
    def nm(t):
        return C.mse_against(C.vec_eval(t, grid), target_vals) / var

    ops = [(p, n) for p, n in node_paths(tree) if n.is_operator]
    m = len(ops)

    n_rigid_median = n_rigid_min = n_rigid_swap = 0
    per_node = []
    for path, node in ops:
        a, b = node.left, node.right
        va, vb = C.vec_eval(a, grid), C.vec_eval(b, grid)
        own_hash = C.func_hash(C.vec_eval(node, grid))

        edits = []
        if C.func_hash(va) != C.func_hash(vb):
            edits.append(("swap", C.eml_node(b.copy(), a.copy())))
        edits.append(("collapse_left", a.copy()))
        edits.append(("collapse_right", b.copy()))
        if not a.is_operator:
            edits.append(("flip_left", C.eml_node(flip_leaf(a), b.copy())))
        if not b.is_operator:
            edits.append(("flip_right", C.eml_node(a.copy(), flip_leaf(b))))

        scores = {}
        for name, sub in edits:
            if C.func_hash(C.vec_eval(sub, grid)) == own_hash:
                continue  # local no-op: not a perturbation
            scores[name] = nm(replace_at(tree, path, sub))

        if not scores:
            per_node.append({"path": "".join(path), "scores": {}})
            continue
        vals = list(scores.values())
        if statistics.median(vals) > RIGIDITY_THRESHOLD:
            n_rigid_median += 1
        if min(vals) > RIGIDITY_THRESHOLD:
            n_rigid_min += 1
        if "swap" in scores and scores["swap"] > RIGIDITY_THRESHOLD:
            n_rigid_swap += 1
        per_node.append({"path": "".join(path), "scores": scores})

    alpha_median = n_rigid_median / m
    alpha_min = n_rigid_min / m
    alpha_swap = n_rigid_swap / m

    f_trivial = min(nm(C.leaf("x")), nm(C.leaf(1.0)))
    sub_scores = [nm(n) for p, n in ops if p]      # proper, operator-rooted
    best_proper = min(sub_scores) if sub_scores else float("inf")
    g = best_proper >= f_trivial

    return {
        "alpha_rigidity": alpha_median,
        "alpha_rigidity_min_rule": alpha_min,
        "alpha_rigidity_swap_rule": alpha_swap,
        "gradient_absence_g": g,
        "best_proper_subtree_nmse": best_proper,
        "trivial_leaf_nmse": f_trivial,
        "n_crit": alpha_median * m,
        "rigid_subset": bool(alpha_median >= 0.5 and g),
        "per_node": per_node,
    }


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------
def screen(tree, grid, val_xs, seen, K, stats, force=False):
    """Run one candidate through the acceptance filters.  Returns a target dict
    or None.  `stats` accumulates rejection reasons."""
    vals = C.vec_eval(tree, grid)
    ok, reason, overflow_fraction, var = viability(vals)
    if not ok and not force:
        stats["viability_" + reason] += 1
        return None

    val_vals = C.vec_eval(tree, val_xs)
    fully_finite = bool(np.isfinite(vals).all() and np.isfinite(val_vals).all())
    if not fully_finite and not force:
        # Strengthened viability: a target that is non-finite anywhere cannot
        # score nMSE = 0 even for the ground-truth tree, so 0% convergence
        # would not be attributable to search.
        stats["not_self_consistent"] += 1
        return None
    if not np.isfinite(var):
        var = C.target_variance(vals[np.isfinite(vals)])

    h = C.func_hash(vals)
    if h in seen and not force:
        stats["duplicate"] += 1
        return None

    if dead_subtree(tree, grid, vals, var) and not force:
        stats["dead_subtree"] += 1
        return None

    K_proven = ""
    if K <= AUDIT_MINIMALITY_MAX_K:
        is_min, found_at = proven_minimal(vals, grid, K)
        if not is_min and not force:
            stats["not_minimal"] += 1
            return None
        K_proven = K if is_min else found_at

    meas = measure(tree, grid, vals, var)
    seen.add(h)
    return {
        "expression": C.expr_str(tree),
        "K_construction": K,
        "K_proven": K_proven,
        "m_operators": C.n_operators(tree),
        "depth": C.tree_depth(tree),
        "overflow_fraction": overflow_fraction,
        "var_target": var,
        "target_scale": float(max(1.0, np.max(np.abs(vals)))),
        "grid_hash": h,
        "n_points": C.N_POINTS,
        **meas,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=["full", "minimal"], default="full")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..",
        "results", "synthetic_dense_k"))
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    k_levels = K_LEVELS_FULL if args.design == "full" else K_LEVELS_MINIMAL
    quota = TARGETS_PER_K_FULL if args.design == "full" else TARGETS_PER_K_MINIMAL

    grid = C.grid_sample()
    val_xs = C.validation_sample(GENERATION_RNG_SEED + 1)
    rng = random.Random(GENERATION_RNG_SEED)

    seen = set()
    targets = []
    oversample = []
    log = {"design": args.design, "k_levels": k_levels, "targets_per_k": quota,
           "generation_rng_seed": GENERATION_RNG_SEED,
           "rigid_oversample_min": RIGID_OVERSAMPLE_MIN, "per_k": {}}

    # ---- forced inclusion: the reviewer counterexample --------------------
    ce_stats = {k: 0 for k in ("viability_overflow", "viability_variance",
                               "viability_magnitude_ratio", "not_self_consistent",
                               "duplicate", "dead_subtree", "not_minimal")}
    ce_tree = C.parse_expr(REVIEWER_CE)
    ce = screen(ce_tree, grid, val_xs, seen, 7, ce_stats)
    forced = False
    if ce is None:
        forced = True
        ce = screen(ce_tree, grid, val_xs, seen, 7, ce_stats, force=True)
    ce["target_id"] = "reviewer_ce"
    targets.append(ce)
    log["reviewer_ce"] = {
        "expression": REVIEWER_CE,
        "passed_pipeline_unforced": not forced,
        "filter_rejections": {k: v for k, v in ce_stats.items() if v},
    }
    print(f"[reviewer_ce] accepted (forced={forced})  "
          f"alpha={ce['alpha_rigidity']:.3f} g={ce['gradient_absence_g']} "
          f"K_proven={ce['K_proven']}")

    # ---- per-K generation -------------------------------------------------
    for K in k_levels:
        m = (K - 1) // 2
        stats = {k: 0 for k in ("viability_overflow", "viability_variance",
                                "viability_magnitude_ratio", "not_self_consistent",
                                "duplicate", "dead_subtree", "not_minimal")}
        accepted = []
        extra_rigid = []
        n_candidates = 0

        # `accepted` is filled before any oversampling begins at this K, so the
        # oversample never displaces a pre-registered target.  (It does consume
        # RNG draws and novelty hashes, so *later* K levels see a different
        # candidate stream than they would without it -- the whole pipeline is
        # one deterministic pass from generation_rng_seed, reproducible as a
        # unit rather than level-by-level.)
        def take(tree):
            nonlocal n_candidates
            n_candidates += 1
            t = screen(tree, grid, val_xs, seen, K, stats)
            if t is None:
                return
            if len(accepted) < quota:
                accepted.append(t)
            elif t["rigid_subset"]:
                extra_rigid.append(t)

        def enough():
            n_rigid = sum(t["rigid_subset"] for t in accepted) + len(extra_rigid)
            return len(accepted) >= quota and n_rigid >= RIGID_OVERSAMPLE_MIN

        if m <= ENUMERATION_LIMIT_M:
            pool = list(enumerate_trees_m(m, C.MAX_DEPTH))
            rng.shuffle(pool)
            source = "exhaustive_enumeration"
            for tree in pool:
                if enough():
                    break
                take(tree)
        else:
            source = "uniform_random_sampling"
            max_attempts = quota * 400 + 8000
            while not enough() and n_candidates < max_attempts:
                take(C.sample_tree(m, C.MAX_DEPTH, rng))

        for i, t in enumerate(accepted):
            t["target_id"] = f"K{K:02d}_{i:02d}"
        for i, t in enumerate(extra_rigid):
            t["target_id"] = f"K{K:02d}_RS{i:02d}"
        targets.extend(accepted)
        oversample.extend(extra_rigid)
        log["per_k"][str(K)] = {
            "source": source,
            "candidates_screened": n_candidates,
            "accepted": len(accepted),
            "quota": quota,
            "rigid_in_sample": sum(t["rigid_subset"] for t in accepted),
            "rigid_oversample_added": len(extra_rigid),
            "rejections": stats,
        }
        alphas = [t["alpha_rigidity"] for t in accepted]
        print(f"[K={K:2d}] accepted {len(accepted):2d}/{quota} from "
              f"{n_candidates} candidates ({source})  "
              f"alpha mean={np.mean(alphas):.3f} "
              f"rigid_subset={sum(t['rigid_subset'] for t in accepted)}"
              f"(+{len(extra_rigid)} oversampled)  "
              f"rejects={ {k: v for k, v in stats.items() if v} }")

    # ---- write ------------------------------------------------------------
    write_targets(out_dir, targets, "synthetic_targets.csv",
                  "synthetic_targets_extra.csv", "targets.json")
    if oversample:
        write_targets(out_dir, oversample,
                      "synthetic_targets_rigid_oversample.csv",
                      "synthetic_targets_rigid_oversample_extra.csv",
                      "targets_rigid_oversample.json")
    with open(os.path.join(out_dir, "target_generation_log.json"), "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=1)

    print(f"\n{len(targets)} pre-registered targets + {len(oversample)} "
          f"rigid-oversample targets -> {out_dir}")


def write_targets(out_dir, targets, contract_name, extra_name, json_name):
    contract_cols = ["target_id", "K_construction", "K_proven", "m_operators",
                     "expression", "alpha_rigidity", "gradient_absence_g",
                     "best_proper_subtree_nmse", "n_crit", "rigid_subset",
                     "overflow_fraction", "domain_lo", "domain_hi", "n_points"]
    with open(os.path.join(out_dir, contract_name), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(contract_cols)
        for t in targets:
            w.writerow([
                t["target_id"], t["K_construction"], t["K_proven"], t["m_operators"],
                t["expression"], repr(t["alpha_rigidity"]),
                "true" if t["gradient_absence_g"] else "false",
                repr(t["best_proper_subtree_nmse"]), repr(t["n_crit"]),
                "true" if t["rigid_subset"] else "false",
                repr(t["overflow_fraction"]), repr(C.DOMAIN_LO), repr(C.DOMAIN_HI),
                t["n_points"],
            ])

    extra_cols = ["target_id", "K_construction", "depth", "var_target", "target_scale",
                  "alpha_rigidity_median_rule", "alpha_rigidity_min_rule",
                  "alpha_rigidity_swap_rule", "trivial_leaf_nmse", "grid_hash"]
    with open(os.path.join(out_dir, extra_name), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(extra_cols)
        for t in targets:
            w.writerow([
                t["target_id"], t["K_construction"], t["depth"], repr(t["var_target"]),
                repr(t["target_scale"]), repr(t["alpha_rigidity"]),
                repr(t["alpha_rigidity_min_rule"]), repr(t["alpha_rigidity_swap_rule"]),
                repr(t["trivial_leaf_nmse"]), t["grid_hash"],
            ])

    with open(os.path.join(out_dir, json_name), "w", encoding="utf-8") as fh:
        json.dump([{k: v for k, v in t.items() if k != "per_node"} for t in targets],
                  fh, indent=1)


if __name__ == "__main__":
    main()

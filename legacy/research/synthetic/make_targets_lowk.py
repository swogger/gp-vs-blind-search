"""POST-HOC extension of the Dense-K design to K = 3 and K = 5.

Outside the pre-registration.  Added after the main run to (a) anchor the upper
asymptote of the logistic fit and (b) give exp(x) -- the main study's K = 3
flagship, 30/30 in both regimes -- a synthetic level to be calibrated against.

Written to separate files.  Do NOT merge these rows into Files 1-2: they were
chosen after seeing the K >= 7 results, and the K levels were not pre-registered.

The pipeline is identical to make_targets.py (same filters, same structural
measurements, same grid and held-out sample).  The spaces are tiny -- 4 trees at
K = 3 and 16 at K = 5 -- so both levels are enumerated exhaustively and every
target that passes the filters is kept; there is no quota and no sampling, hence
no RNG dependence at all.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eml_synth_core as C      # noqa: E402
import make_targets as M        # noqa: E402

K_LEVELS_LOW = [3, 5]


def main():
    out_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "results", "synthetic_dense_k"))

    grid = C.grid_sample()
    val_xs = C.validation_sample(M.GENERATION_RNG_SEED + 1)

    # Seed the novelty set with the pre-registered targets so a low-K target
    # cannot duplicate a function already in Files 1-2.
    seen = set()
    with open(os.path.join(out_dir, "targets.json"), encoding="utf-8") as fh:
        for t in json.load(fh):
            seen.add(t["grid_hash"])

    targets = []
    log = {"note": "post-hoc, outside pre-registration", "per_k": {}}
    for K in K_LEVELS_LOW:
        m = (K - 1) // 2
        stats = {k: 0 for k in ("viability_overflow", "viability_variance",
                                "viability_magnitude_ratio", "not_self_consistent",
                                "duplicate", "dead_subtree", "not_minimal")}
        accepted = []
        pool = list(M.enumerate_trees_m(m, C.MAX_DEPTH))
        for tree in pool:
            t = M.screen(tree, grid, val_xs, seen, K, stats)
            if t is not None:
                accepted.append(t)
        for i, t in enumerate(accepted):
            t["target_id"] = f"K{K:02d}_{i:02d}"
        targets.extend(accepted)
        log["per_k"][str(K)] = {"source": "exhaustive_enumeration",
                                "candidates_screened": len(pool),
                                "accepted": len(accepted),
                                "rejections": stats}
        alphas = [t["alpha_rigidity"] for t in accepted]
        print(f"[K={K}] accepted {len(accepted)}/{len(pool)} enumerated  "
              f"alpha mean={np.mean(alphas) if alphas else float('nan'):.3f}  "
              f"rejects={ {k: v for k, v in stats.items() if v} }")
        for t in accepted:
            print(f"        {t['target_id']}  {t['expression']:<34} "
                  f"alpha={t['alpha_rigidity']:.2f} g={t['gradient_absence_g']}")

    M.write_targets(out_dir, targets,
                    "synthetic_targets_lowk.csv",
                    "synthetic_targets_lowk_extra.csv",
                    "targets_lowk.json")
    with open(os.path.join(out_dir, "target_generation_log_lowk.json"), "w",
              encoding="utf-8") as fh:
        json.dump(log, fh, indent=1)
    print(f"\n{len(targets)} low-K targets -> {out_dir} "
          f"({len(targets) * 20} runs)")


if __name__ == "__main__":
    main()

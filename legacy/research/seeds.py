# Deterministic per-run seeds for the grid search.
# One seed per run index (1-indexed, so index 0 is unused).
# Generated once from a fixed master seed so every run is reproducible
# and independent of every other run.

import random as _random

_master = _random.Random(20260429)
SEEDS = [None] + [_master.randint(0, 2**31 - 1) for _ in range(30)]

# Usage in run_matrix.py:
#   from seeds import SEEDS
#   cmd += ["--seed", str(SEEDS[run_idx])]
#
# Usage in deduce_formula.py:
#   random.seed(args.seed)

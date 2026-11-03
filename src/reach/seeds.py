"""Run seeds for this project (fresh stream; EML-GP's SEEDS are used only for regression)."""
import random as _random

_master = _random.Random(20261012)
SEEDS = [_master.randint(0, 2 ** 31 - 1) for _ in range(100)]


# ---------------------------------------------------------------- Study 2 stream (tool T1; REQ-RUN-09)
# A new master seed; 2,000 seeds, none shared with paper 1's SEEDS or EML-GP's legacy seeds. Slices are disjoint;
# "main" (the 5 seeds of every main-grid cell) is the first 5 of "calibration" (the 32-seed allocation subset, C7).
S2_MASTER = 20261019
_LEGACY = _random.Random(20260429)
_LEGACY_SEEDS = {_LEGACY.randint(0, 2 ** 31 - 1) for _ in range(30)}     # legacy/research/seeds.py, reproduced


def _unique(master, n, exclude):
    r, out, seen = _random.Random(master), [], set(exclude)
    while len(out) < n:
        s = r.randint(0, 2 ** 31 - 1)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


S2_SEEDS = _unique(S2_MASTER, 2000, set(SEEDS) | _LEGACY_SEEDS)
S2_SLICES = {"pilot": (0, 20), "calibration": (100, 132), "pool": (200, 230), "heldout": (300, 305),
             "benchmark": (400, 430),
             "cgp_validation": (500, 700), "cgp_pilot": (700, 710),   # CGP (D072): validation, pilot (new slices)
             "cgp_calibration": (710, 715), "cgp_heldout": (715, 720),  # CGP confirmatory (D076): new slices
             "restart": (720, 784)}                                    # Stage 4 restarts (D100): 64 fresh seeds
TARGET_SEEDS = {"pilot": 20261020, "calibration": 20261021, "heldout": 20261022, "pool": 20261023,
                "restart": 20261102}                                   # Stage 4: RS-add_sqrt's extra levels (D100)


def s2_seeds(name):
    if name == "main":
        return s2_seeds("calibration")[:5]
    a, b = S2_SLICES[name]
    return S2_SEEDS[a:b]

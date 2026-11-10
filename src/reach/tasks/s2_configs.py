"""GP configurations for Study 2 (tool T19; H8; docs/design/algorithms.md, robustness arm R1-R6) and Stage 5's arms
(D125; docs/design/stage5_arms.md; REQ-ENG-30).

Each entry: run_gp keyword arguments, population size, and generations per E0 = 2x10^4 evaluations (pop x gens = E0).
"base" is regime A's configuration (and B's and HC's). A Stage 5 entry may name the one domain it applies to
(`domain`); s2_run refuses it on a set of the other domain."""
import math

import numpy as np

WIDE_INTERVAL = (1.5, 3.5)       # the wider batch interval (D125; guarded-semantics check in docs/design/stage5_wide.md)

CONFIGS = {
    "base": dict(kwargs={}, pop_size=100, generations_per_E0=200),
    "R1": dict(kwargs={}, pop_size=25, generations_per_E0=800),                       # small population, long run
    "R2": dict(kwargs={}, pop_size=1000, generations_per_E0=20),                      # large population, short run
    "R3": dict(kwargs=dict(tournament_size=2), pop_size=100, generations_per_E0=200),  # weak selection
    "R4": dict(kwargs=dict(selection="lexicase"), pop_size=100, generations_per_E0=200),
    "R5": dict(kwargs=dict(init="uniform"), pop_size=100, generations_per_E0=200),    # uniform-size initialisation
    "R6": dict(kwargs=dict(max_depth=17), pop_size=100, generations_per_E0=200),      # Koza's depth limit
    # Stage 5 (D125): one factor each, everything else as base
    "FG": dict(kwargs=dict(real_fitness="grid"), pop_size=100, generations_per_E0=200, domain="real"),   # arm 1a
    "FB": dict(kwargs=dict(real_fitness="frozen"), pop_size=100, generations_per_E0=200, domain="real"),  # arm 1b
    "BS24": dict(kwargs=dict(bool_rows=24), pop_size=100, generations_per_E0=200, domain="bool"),         # arm 2
    "SF": dict(kwargs=dict(selection="none"), pop_size=100, generations_per_E0=200),                      # arm 3
    "WIDE": dict(kwargs=dict(batch_domain=WIDE_INTERVAL), pop_size=100, generations_per_E0=200, domain="real",
                 batch_var="wide"),                                                    # wider batch interval
}


def interval_grid(lo, hi, n=100):
    """n interior points of (lo, hi), spaced as the identity grid on (2, 3) (endpoints excluded)."""
    step = (hi - lo) / (n + 1)
    return np.array([lo + step * (i + 1) for i in range(n)], dtype=np.complex128)


def wide_var_norm(g, tree, interval=WIDE_INTERVAL):
    """The batch normaliser of the wider-interval arm: the target's variance on the interval's 100-point grid, over
    the points where it is finite (as var_target on (2, 3)), so the scale-aware penalty, 1000 x this, keeps an
    undefined tree behind a constant on the batch (D034)."""
    from reach import grammar as G
    v = G.evaluate(g, tree, interval_grid(*interval))
    with np.errstate(all="ignore"):
        v = v[np.isfinite(v)]
        return float(np.mean(np.abs(v - v.mean()) ** 2))


def wide_ok(g, tree, interval=WIDE_INTERVAL):
    """Whether the target's wide normaliser is finite and positive in double precision. A target where it is not
    (values near 1e200 overflow the variance) is excluded from the wider-interval arm, in both arms of its test
    (coordinator, 2026-11-08; review F1; guardrail row 173)."""
    v = wide_var_norm(g, tree, interval)
    return math.isfinite(v) and v > 0

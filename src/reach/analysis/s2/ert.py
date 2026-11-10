"""Expected evaluations to success (ERT) beside the exact null (REQ-ANA-113; EXPLORATORY, descriptive on registered
records; no test, no decision).

ERT (Price 1997; Auger and Hansen 2005) of a group of runs with budget E: the evaluations all runs spent, a successful
run counted to its first exact hit and an unsuccessful one at E, divided by the number of successes. It is the
expected cost of reaching the target by independent restarts of E-evaluation runs; with no success it is infinite (the
runs bound it from below: more than n x E / 1 when the next run would succeed).
- Tree GP: a run succeeds within E iff `load.outcome(r, E / pop_size) == "success"`; its evaluations to the hit are
  (gen_exact + 1) x pop_size (generation 0 evaluates the initial population; `h6d.runs_from_rows`). A run stopped on
  its CPU limit or killed without a result is unsuccessful and charged E (the primary censoring rule); 'beyond' (E
  above the run's own budget) and error rows are left out.
- Cartesian GP: `load.outcome_cgp(r, E)`, evaluations to the hit `evals_exact`.
The exact null enumerates distinct functions in size order. It covers level K after U(<=K) evaluations (`cover`; the
registered reading, bits_sem = log2 U(<=K) up to the float-noise correction), and finds a target drawn uniformly from
level K after U(<K) + (U(K) + 1) / 2 evaluations on average (`expected`; the uniform position its success share at a
budget, `null.p_hit`, assumes). Gaps in bits: log2(null evaluations) - log2(ERT), positive when GP needs fewer.
"""
import math

from reach.analysis.s2.load import outcome, outcome_cgp


def spent(r, E, kind="tree"):
    """(success, evaluations charged) of one run at budget E, or None when the run is left out."""
    if kind == "tree":
        o = outcome(r, round(E / r.get("pop_size", 100)))
        hit = (r["gen_exact"] + 1) * r.get("pop_size", 100) if o == "success" else None
    else:
        o = outcome_cgp(r, E)
        hit = r["evals_exact"] if o == "success" else None
    if o in ("beyond", "error"):
        return None
    return (True, hit) if o == "success" else (False, E)


def ert(rows, E, kind="tree"):
    """ERT of the rows at budget E: dict(ert, successes, runs, evaluations)."""
    got = [x for x in (spent(r, E, kind) for r in rows) if x is not None]
    s = sum(ok for ok, _ in got)
    total = sum(e for _, e in got)
    return dict(ert=total / s if s else float("inf"), successes=s, runs=len(got), evaluations=total,
                lower_bound=None if s else total)


def null_evals(U, K):
    """The exact null at level K of a set with level counts U {K: U(K)}: evaluations to cover the level, the expected
    evaluations for a target drawn uniformly from it, and the counts."""
    below = sum(u for k, u in U.items() if k < K)
    return dict(cover=below + U[K], expected=below + (U[K] + 1) / 2, below=below, at=U[K])


def null_success(U, K, E):
    """The null's success probability within E evaluations on a uniform target of level K (`null.p_hit`)."""
    n = null_evals(U, K)
    return 1.0 if E >= n["cover"] else max(0.0, (E - n["below"]) / n["at"])


def gap_bits(null_evaluations, ert_value):
    if ert_value <= 0 or null_evaluations <= 0:
        return float("nan")
    return math.log2(null_evaluations) - math.log2(ert_value) if math.isfinite(ert_value) else float("-inf")

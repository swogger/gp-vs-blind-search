"""Study 2 run records -> outcomes (tool T16; REQ-ANA-05). The one place where "success at a budget" is defined.

A run of regime A with stop-on-exact records gen_exact, the generation of its first exact match (None if none), and
gens_reached. Because a run of L generations is the prefix of a longer run with the same seed (C2), the outcome at L:
- success  if gen_exact is not None and gen_exact < L;
- failure  if not, and the run reached L generations (gens_reached >= L, or it used its whole budget >= L);
- censored if it stopped early on its CPU limit before L without success (primary analysis: failure; sensitivity:
           excluded);
- beyond   if L exceeds the run's own budget, whether or not it succeeded (a failure there is unknown, so keeping
           its successes alone would bias the rate up);
- error    for rows that ended in the error state (reported, excluded);
- a row killed by the queue without a result line is censored at every L.
Budgets are multiples of E0 = 2x10^4 evaluations; with population P, L = budget * 2x10^4 / P generations.
Operon runs have no prefix property: their outcome is `exact`, at their own budget only.
CGP runs count evaluations, not generations (outcome_cgp): success at E evaluations iff exact with evals_exact <= E.
"""
import json
from pathlib import Path

E0 = 20_000


def outcome(r, L):
    if r.get("terminal_state") == "error":
        return "error"
    if "gens_reached" not in r:
        return "censored"
    if L > r.get("generations", L):
        return "beyond"                 # before success: keeping only the successes beyond a run's length biases up
    ge = r.get("gen_exact")
    if ge is not None and ge < L:
        return "success"
    if r.get("terminal_state") == "timeout" and r["gens_reached"] < L:
        return "censored"
    return "failure"


def outcome_cgp(r, E):
    """The outcome of a CGP run (REQ-ANA-22) at a budget of E evaluations, by the same rules as `outcome`. A rejected
    exact claim stops the run early (terminal state early_stopped): censored above the evaluations it reached."""
    if r.get("terminal_state") == "error":
        return "error"
    if "evals_reached" not in r:
        return "censored"                 # killed without a result (hard CPU limit or from outside)
    if E > r.get("budget_E", E):
        return "beyond"
    if r.get("exact") and r.get("evals_exact") is not None and r["evals_exact"] <= E:
        return "success"
    if r.get("terminal_state") in ("timeout", "early_stopped") and r["evals_reached"] < E:
        return "censored"
    return "failure"


def tidy(rows, budgets, censored_as_failure=True):
    """One record per (run, budget): y (1 success, 0 failure), censored flag; 'beyond' and 'error' are dropped. Rows
    that are already tidy records (they carry `y`) are refused: tidying them again reads no `evals_reached` or
    generation and makes every record censored, a silent failure (guardrail row 134)."""
    if any("y" in r for r in rows):
        raise ValueError("tidy() got records that are already tidy (they carry 'y', e.g. from cgp_confirm.load); "
                         "pass the raw phase rows")
    out = []
    for r in rows:
        if r.get("algorithm") == "operon":           # no prefix property: `exact` at its own budget only (row 53)
            killed = "exact" not in r                 # queue kill without a result: censored (row 56)
            if r.get("terminal_state") != "error" and r.get("budget") in budgets and (censored_as_failure or not killed):
                out.append(dict(set=r.get("set"), target_id=r.get("target_id"), seed=r.get("seed"), algorithm="operon",
                                config=r.get("config"), K=r.get("K_construction", r.get("K")), K_proven=r.get("K_proven"),
                                units=r.get("units") or {}, budget=r["budget"], y=int(bool(r.get("exact"))),
                                censored=killed, cpu_s=r.get("cpu_s")))
            continue
        pop = r.get("pop_size", 100)
        for b in budgets:
            if r.get("algorithm") == "CGP":
                o = outcome_cgp(r, round(b * E0))
            else:
                o = outcome(r, round(b * E0 / pop))
            if o in ("beyond", "error") or (o == "censored" and not censored_as_failure):
                continue
            out.append(dict(set=r.get("set"), target_id=r.get("target_id"), seed=r.get("seed"),
                            algorithm=r.get("algorithm"), config=r.get("config"), K=r.get("K_construction", r.get("K")),
                            K_proven=r.get("K_proven"), units=r.get("units") or {}, budget=b,
                            y=int(o == "success"), censored=o == "censored", cpu_s=r.get("cpu_s")))
    return out


def read_phase(name, root=None):
    """Last row per key of data/runs/<name>/results.jsonl (the queue's file), or of the committed results.jsonl.gz
    when the raw file is absent (D097: results over 50 MB are committed gzipped, the raw file stays local; guardrail
    row 179: a clean checkout read no row of those phases). A truncated final line is ignored, as `read_results`."""
    import gzip
    from reach.runner.queue import read_results
    from reach.paths import ROOT
    p = Path(root or ROOT) / "data/runs" / name / "results.jsonl"
    gz = p.with_name("results.jsonl.gz")
    if p.exists() or not gz.exists():
        return list(read_results(p).values())
    rows = {}
    with gzip.open(gz, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows[row["key"]] = row
    return list(rows.values())

"""CGP pilot analysis (D072 step 3; REQ-ANA-24). Not confirmatory. Written before any pilot data existed.

    .venv/bin/python -m reach.analysis.s2.cgp_pilot   -> docs/status/cgp_pilot_report.md, data/analysis/cgp_pilot.json

The genome-length rule, fixed in docs/design/cgp.md before the pilot, applied here exactly:
  "per domain, the length with the highest mean exact recovery, averaged over budgets 1/4, 1, 4, 16xE0 and over the
   two sets with equal weight; a difference below 0.02 goes to the shorter genome (cheaper). Also measured: CPU per
   run, the share of runs reaching the CPU limit, and Rust-Python agreement."
How it is computed (readings of the rule, stated before the data):
- exact recovery at budget B for a set and length = the share of its runs (pilot targets x seeds) with an exact match
  within B x E0 evaluations, from load.tidy: a run stopped by its CPU limit before B counts as a failure (the primary
  convention of every Study 2 analysis); error rows are excluded and reported;
- the set's score = the mean of its four recovery rates; the domain's score = the mean of its two sets' scores;
- "a difference below 0.02 goes to the shorter genome": the chosen length is the shortest one whose score is less than
  0.02 below the highest score (so a near tie never moves the choice to a longer genome);
- a missing cell (no runs) gives no choice for that domain; before the phase is complete the choice is provisional.

Settings file (D076; REQ-ANA-25): once the phase is complete and both domains have a choice, the choice is written to
data/registration/cgp_settings.json (with a .sha256), the only source the confirmatory CGP phases read (REQ-RUN-28):
genome length per domain, this rule's text, the SHA-256 of the pilot's results and manifest, the soft CPU limit per
domain (cgp.md: 3 x the pilot's 99th percentile at the chosen length, at least 600 s; rounded up to whole seconds) and
the levels (cgp.md: all 10 registered levels unless the projection exceeds 100 CPU-h, then the 5 budget levels). An
existing file with other content is never replaced. Error rows (runs killed from outside) are rerun once by the
registered rule (scripts/keep_running.sh, marker ERRORS_RETRIED): a pilot with error rows gives settings only after that
retry has run and finished (the marker exists and the queue is not running), so the retry can never change the results
hash the settings record (guardrail row 93).
"""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from reach.analysis.s2.load import E0, tidy

PHASE = "s2_pilot_cgp"
SETS = {"real": ["F-EML", "R-ELEM4"], "bool": ["BOOL-NAND-n5", "BOOL-AONX-n5"]}
LENGTHS = (100, 500, 2000)
BUDGETS = (0.25, 1, 4, 16)
MARGIN = 0.02
REPORT = "docs/status/cgp_pilot_report.md"
JSON_OUT = "data/analysis/cgp_pilot.json"
CAL_RUNS_PER_SET = 10 * 30 * 5          # calibration as proposed in cgp.md: 10 levels x 30 targets x 5 seeds
SETTINGS = "data/registration/cgp_settings.json"
CAL_CPU_H_LIMIT = 100.0                 # cgp.md: above 100 projected CPU-h, budget levels only
RULE = ("per domain, the genome length (100, 500 or 2,000 nodes) with the highest mean exact recovery, averaged over "
        "budgets 1/4, 1, 4, 16 x E0 and over the two pilot sets with equal weight; a difference below 0.02 goes to the "
        "shorter genome (the shortest length less than 0.02 below the best); CPU stops count as failures")
LEVELS_RULE = ("all 10 registered levels (main and budget) unless the projected CGP calibration cost exceeds 100 CPU-h, "
               "then the 5 budget levels; the same choice for the held-out levels")
NAN = float("nan")


def recovery(rows):
    """{(set, length): {budget: share of runs exact within the budget}} (censored as failure; errors excluded)."""
    acc = defaultdict(list)
    for t in tidy([r for r in rows if r.get("algorithm") == "CGP"], BUDGETS, censored_as_failure=True):
        acc[(t["set"], int(str(t["config"]).lstrip("n")), t["budget"])].append(t["y"])
    out = defaultdict(dict)
    for (g, n, b), ys in acc.items():
        out[(g, n)][b] = float(np.mean(ys))
    return dict(out)


def scores(rec):
    """{domain: {length: mean over the two sets (equal weight) of the mean over the four budgets}}."""
    out = {}
    for dom, sets in SETS.items():
        out[dom] = {}
        for n in LENGTHS:
            per_set = []
            for g in sets:
                r = rec.get((g, n), {})
                per_set.append(float(np.mean([r[b] for b in BUDGETS])) if all(b in r for b in BUDGETS) else NAN)
            out[dom][n] = float(np.mean(per_set))
    return out


def choose(score_by_length, margin=MARGIN):
    """The shortest length whose score is less than `margin` below the highest; None if a score is missing."""
    s = {n: v for n, v in score_by_length.items()}
    if any(v is None or math.isnan(v) for v in s.values()):
        return None
    best = max(s.values())
    return min(n for n, v in s.items() if best - v < margin - 1e-12)


def _cpu(r):
    return r.get("cpu_s", r.get("cpu_seconds"))


def costs(rows):
    """{(set, length): runs, errors, CPU s per run (mean, median, p99, max; error rows excluded), runs that reached the
    CPU limit (soft stop or hard kill), Rust-Python disagreements (claims rejected, missed exact matches)}."""
    by = defaultdict(list)
    for r in rows:
        if r.get("algorithm") == "CGP":
            by[(r.get("set"), int(r.get("n_nodes")))].append(r)
    out = {}
    for k, rs in by.items():
        cpu = [float(_cpu(r)) for r in rs if r.get("terminal_state") != "error" and _cpu(r) is not None]
        out[k] = dict(runs=len(rs), errors=sum(r.get("terminal_state") == "error" for r in rs),
                      cpu_limit=sum(r.get("terminal_state") == "timeout" for r in rs),
                      claim_rejected=sum(bool(r.get("claim_rejected")) for r in rs),
                      missed_exact=sum(bool(r.get("missed_exact")) for r in rs),
                      cpu_mean=float(np.mean(cpu)) if cpu else None, cpu_median=float(np.median(cpu)) if cpu else None,
                      cpu_p99=float(np.percentile(cpu, 99)) if cpu else None, cpu_max=max(cpu) if cpu else None)
    return out


def projection(rows, choice):
    """Rough calibration cost at the chosen lengths (cgp.md: budget levels only if above 100 CPU-h) and the proposed
    soft CPU limit (3 x the pilot's 99th percentile, at least 600 s)."""
    from reach import grammar as G
    from reach.runner.phases import CAL_SETS
    out = {}
    for dom, sets in SETS.items():
        n = choice.get(dom)
        cpu = [float(_cpu(r)) for r in rows if r.get("set") in sets and r.get("n_nodes") == n
               and r.get("terminal_state") != "error" and _cpu(r) is not None]
        if n is None or not cpu:
            continue
        n_sets = sum(G.get(g).kind == dom for g in CAL_SETS)
        out[dom] = dict(length=n, cal_sets=n_sets, cal_runs=n_sets * CAL_RUNS_PER_SET,
                        cpu_h=n_sets * CAL_RUNS_PER_SET * float(np.mean(cpu)) / 3600,
                        soft_limit_s=max(600.0, 3 * float(np.percentile(cpu, 99))))
    return out


def assess(rows, n_tasks=None):
    rec = recovery(rows)
    sc = scores(rec)
    choice = {dom: choose(sc[dom]) for dom in SETS}
    return dict(complete=n_tasks is not None and len(rows) == n_tasks, rows=len(rows), n_tasks=n_tasks,
                recovery={f"{g}|{n}": v for (g, n), v in sorted(rec.items())}, scores=sc, choice=choice,
                costs=[dict(set=g, length=n, **c) for (g, n), c in sorted(costs(rows).items())],
                projection=projection(rows, choice))


def settings(res, results_sha256, manifest_sha256, error_rows=0):
    """The confirmatory CGP settings from a COMPLETE pilot (REQ-ANA-25); None while the choice is provisional."""
    if not res["complete"] or any(res["choice"].get(d) is None for d in SETS) or set(res["projection"]) != set(SETS):
        return None
    total = sum(p["cpu_h"] for p in res["projection"].values())
    return dict(pilot_phase=PHASE, pilot_results_sha256=results_sha256, pilot_manifest_sha256=manifest_sha256,
                pilot_rows=res["rows"], pilot_error_rows=error_rows, rule=RULE, scores=res["scores"],
                genome_length={d: int(res["choice"][d]) for d in SETS},
                soft_cpu_limit_s={d: int(math.ceil(res["projection"][d]["soft_limit_s"])) for d in SETS},
                projected_cal_cpu_h=total, levels="all" if total <= CAL_CPU_H_LIMIT else "budget",
                levels_rule=LEVELS_RULE)


def _queue_running(d):
    """As keep_running.sh's `running`: data/runs/<phase>/queue.pid names a live process."""
    import os
    try:
        os.kill(int((d / "queue.pid").read_text().strip()), 0)
    except (FileNotFoundError, ValueError, ProcessLookupError):
        return False
    except PermissionError:          # alive, owned by another user
        return True
    return True


def errors_settled(root, error_rows):
    """None if the pilot's rows are final; otherwise why not (row 93): error rows wait for the once-only retry."""
    d = Path(root) / "data/runs" / PHASE
    if error_rows is None:
        return "pilot_error_rows is not stated"
    if _queue_running(d):
        return f"the {PHASE} queue is running (queue.pid)"
    if error_rows and not (d / "ERRORS_RETRIED").exists():
        return (f"{error_rows} error rows and no ERRORS_RETRIED marker: the once-only retry of error rows "
                "(scripts/keep_running.sh) has not run, and it would change the results hash")
    return None


def write_settings(root, st):
    """Write the settings file and its .sha256; an existing file with other content is never replaced; error rows only
    after the once-only retry has run and finished (row 93)."""
    why = errors_settled(root, st.get("pilot_error_rows"))
    if why:
        raise RuntimeError(f"{SETTINGS} not written: {why}")
    path = Path(root) / SETTINGS
    body = json.dumps(st, indent=1, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != body:
        raise RuntimeError(f"{SETTINGS} exists with other content: a registered input is never replaced. Compare it "
                           "with the pilot's results; if it must change, record a decision and remove it by hand.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.with_suffix(".sha256").write_text(f"{hashlib.sha256(body.encode()).hexdigest()}  {path.name}\n")
    return path


def _f(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.{nd}f}"


def report(res):
    head = "final" if res["complete"] else "PROVISIONAL (phase incomplete)"
    lines = ["# CGP pilot: genome length and cost", "",
             "> Index: `docs/README.md`. Generated by `python -m reach.analysis.s2.cgp_pilot` (REQ-ANA-24); do not "
             "edit. Not confirmatory. Rule fixed in `docs/design/cgp.md` before the pilot.", "",
             f"Runs recorded: {res['rows']} of {res['n_tasks'] if res['n_tasks'] is not None else '?'}. Choice: **{head}**.",
             "", "## Exact recovery by budget (x E0) and the rule", "",
             "| Set | Length | 1/4 | 1 | 4 | 16 | Mean |", "|---|---|---|---|---|---|---|"]
    for dom, sets in SETS.items():
        for g in sets:
            for n in LENGTHS:
                r = res["recovery"].get(f"{g}|{n}", {})
                vals = [r.get(b) for b in BUDGETS]
                mean = float(np.mean(vals)) if all(v is not None for v in vals) else None
                lines.append(f"| {g} | {n} | " + " | ".join(_f(v) for v in vals) + f" | {_f(mean, 3)} |")
    lines += ["", "| Domain | " + " | ".join(f"score {n}" for n in LENGTHS) + " | Chosen length |",
              "|---|" + "---|" * (len(LENGTHS) + 1)]
    for dom in SETS:
        lines.append(f"| {dom} | " + " | ".join(_f(res["scores"][dom][n], 3) for n in LENGTHS)
                     + f" | {res['choice'][dom] if res['choice'][dom] is not None else '–'} |")
    lines += ["", "## Cost and Rust–Python agreement per setting", "",
              "| Set | Length | Runs | Errors | CPU s mean | median | p99 | max | CPU limit | Claims rejected | "
              "Missed exact |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in res["costs"]:
        lines.append(f"| {c['set']} | {c['length']} | {c['runs']} | {c['errors']} | {_f(c['cpu_mean'])} | "
                     f"{_f(c['cpu_median'])} | {_f(c['cpu_p99'])} | {_f(c['cpu_max'])} | {c['cpu_limit']} | "
                     f"{c['claim_rejected']} | {c['missed_exact']} |")
    if res["projection"]:
        lines += ["", "## Projection for the calibration (rough: pilot targets stand in for calibration targets)", ""]
        for dom, p in res["projection"].items():
            lines.append(f"- {dom}: length {p['length']}, {p['cal_sets']} sets, {p['cal_runs']:,} runs, about "
                         f"{_f(p['cpu_h'], 1)} CPU-h; proposed soft CPU limit {_f(p['soft_limit_s'], 0)} s.")
        tot = sum(p["cpu_h"] for p in res["projection"].values())
        lines.append(f"- total about {_f(tot, 1)} CPU-h (cgp.md: above 100 CPU-h, budget levels only).")
    return "\n".join(lines) + "\n"


def main(root=None):
    from reach.analysis.s2.load import read_phase
    from reach.paths import ROOT
    root = Path(root or ROOT)
    d = root / "data/runs" / PHASE
    rows = read_phase(PHASE, root) if (d / "results.jsonl").exists() else []
    n_tasks = json.loads((d / "manifest.json").read_text())["n_tasks"] if (d / "manifest.json").exists() else None
    res = assess(rows, n_tasks)
    for rel, text in ((REPORT, report(res)), (JSON_OUT, json.dumps(res, indent=1) + "\n")):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    print(f"CGP pilot: {res['rows']} runs; choice {res['choice']} ({'final' if res['complete'] else 'provisional'})")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None    # noqa: E731
    n_err = sum(r.get("terminal_state") == "error" for r in rows)
    st = settings(res, sha(d / "results.jsonl"), sha(d / "manifest.json"), n_err)
    why = errors_settled(root, n_err) if st is not None else None
    if why:
        print(f"settings not written yet: {why}")
    elif st is not None:
        print(f"settings -> {write_settings(root, st)}: genome length {st['genome_length']}, levels {st['levels']}")
    return res


if __name__ == "__main__":
    main()

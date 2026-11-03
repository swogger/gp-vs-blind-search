"""CGP validation against Goldman and Punch (2015) (D072 step 3; REQ-ANA-23). Not confirmatory.

    .venv/bin/python -m reach.analysis.s2.cgp_validate   -> docs/status/cgp_validation_report.md, data/analysis/cgp_validation.json

Acceptance rule, fixed in docs/design/cgp.md before any validation run, applied here exactly:
  "200 seeds per setting; every run succeeds; our median lies inside 885-1,487 (500 nodes); a two-sample
   Kolmogorov-Smirnov test against their 50 raw values gives p >= 0.01; for 3,000 nodes, our median lies within
   +-25% of 1,358. Their count starts at 0 for the initial parent, so we compare our count minus 1 (offspring-level)."
Our count is `eval_index_exact`, the evaluation index of the offspring that became the exact parent (the initial parent
is evaluation 1). The 50 raw values are the rows parity/single/normal/500 of the authors' rdata.csv, stored in
data/reference/goldman_punch_2015/ with its SHA-256 and source URL (source.json). A phase with missing runs is
INCOMPLETE, never PASS; a run that did not succeed makes the verdict FAIL at once.
"""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

PHASE = "s2_cgp_validate"
REF_DIR = "data/reference/goldman_punch_2015"
REF_ROWS = dict(problem="parity", duplication="single", ordering="normal", genome_size="500")
REPORT = "docs/status/cgp_validation_report.md"
JSON_OUT = "data/analysis/cgp_validation.json"
N_SEEDS = 200                      # per setting
CI_500 = (885, 1487)               # their 95% bootstrap CI of the median, Table II Single-Normal (median 1,214)
REF_MEDIAN_500 = 1214
MEDIAN_3000, TOL_3000 = 1358, 0.25  # Goldman and Punch (2013) Table 1, Single, 3,000 nodes
KS_ALPHA = 0.01


def reference_values(root):
    """The 50 raw evaluations to success of Table II Single-Normal (3-bit parity, 500 nodes); refuses a changed file."""
    d = Path(root) / REF_DIR
    src = json.loads((d / "source.json").read_text())
    raw = (d / "rdata.csv").read_bytes()
    if hashlib.sha256(raw).hexdigest() != src["files"]["rdata.csv"]:
        raise ValueError(f"{d / 'rdata.csv'}: SHA-256 differs from source.json (downloaded from {src['url']})")
    rows = [r for r in csv.DictReader(io.StringIO(raw.decode())) if all(r[k] == v for k, v in REF_ROWS.items())]
    return [int(r["evaluations"]) for r in rows]


def offspring_count(r):
    """Our count on their scale: the initial parent is evaluation 0 for them, 1 for us."""
    return r["eval_index_exact"] - 1


def _succeeded(r):
    return r.get("terminal_state") != "error" and bool(r.get("exact")) and r.get("eval_index_exact") is not None


def assess_setting(rows, n_nodes, ref, n_expected=N_SEEDS):
    seeds = {r.get("seed") for r in rows}
    ok = [r for r in rows if _succeeded(r)]
    counts = sorted(offspring_count(r) for r in ok)
    s = dict(n_nodes=n_nodes, expected=n_expected, rows=len(rows), distinct_seeds=len(seeds), successes=len(ok),
             not_succeeded=len(rows) - len(ok), complete=len(seeds) >= n_expected, all_succeed=len(ok) == n_expected
             and len(rows) == n_expected, median=float(np.median(counts)) if counts else None,
             q25=float(np.percentile(counts, 25)) if counts else None,
             q75=float(np.percentile(counts, 75)) if counts else None,
             min=min(counts) if counts else None, max=max(counts) if counts else None,
             active_nodes_median=float(np.median([r["active_nodes"] for r in ok if r.get("active_nodes") is not None]))
             if any(r.get("active_nodes") is not None for r in ok) else None,
             claim_rejected=sum(bool(r.get("claim_rejected")) for r in rows),
             cpu_s_mean=float(np.mean([r["cpu_s"] for r in rows if r.get("cpu_s") is not None]))
             if any(r.get("cpu_s") is not None for r in rows) else None)
    m = s["median"]
    if n_nodes == 500:
        s["median_ok"] = m is not None and CI_500[0] <= m <= CI_500[1]
        if counts:
            ks = ks_2samp(counts, ref)
            s["ks_D"], s["ks_p"] = float(ks.statistic), float(ks.pvalue)
        else:
            s["ks_D"], s["ks_p"] = None, None
        s["ks_ok"] = s["ks_p"] is not None and s["ks_p"] >= KS_ALPHA
    else:
        s["median_ok"] = m is not None and abs(m - MEDIAN_3000) <= TOL_3000 * MEDIAN_3000
        s["ks_ok"] = True                       # no raw data published for 3,000 nodes: median rule only
    s["passed"] = s["complete"] and s["all_succeed"] and s["median_ok"] and s["ks_ok"]
    return s


def assess(rows, ref, n_expected=N_SEEDS):
    by = {500: [], 3000: []}
    for r in rows:
        if r.get("n_nodes") in by:
            by[r["n_nodes"]].append(r)
    st = {str(n): assess_setting(by[n], n, ref, n_expected) for n in by}
    failed_run = any(s["not_succeeded"] for s in st.values())
    if failed_run:
        verdict = "FAIL"
    elif not all(s["complete"] for s in st.values()):
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS" if all(s["passed"] for s in st.values()) else "FAIL"
    return dict(verdict=verdict, settings=st, reference=dict(n=len(ref), median=float(np.median(ref)), min=min(ref),
                                                            max=max(ref)))


def _f(x, nd=0):
    return "–" if x is None else f"{x:,.{nd}f}"


def _spread(s):
    if s["median"] is None:
        return "–"
    return f"{_f(s['q25'])}–{_f(s['q75'])}; {_f(s['min'])}–{_f(s['max'])}"


def report(res):
    st, ref = res["settings"], res["reference"]
    lines = ["# CGP validation against Goldman and Punch (2015)", "",
             "> Index: `docs/README.md`. Generated by `python -m reach.analysis.s2.cgp_validate` (REQ-ANA-23); do not "
             "edit. Not confirmatory. Rule fixed in `docs/design/cgp.md` before the runs.", "",
             f"**Verdict: {res['verdict']}**", "",
             f"Reference: their 50 raw values (Table II, Single-Normal, 3-bit parity, 500 nodes): median "
             f"{_f(ref['median'])}, range {_f(ref['min'])}–{_f(ref['max'])}. Our counts are offspring-level "
             "(`eval_index_exact - 1`).", "",
             "| Check | 500 nodes | 3,000 nodes |", "|---|---|---|"]
    a, b = st["500"], st["3000"]
    rows = [("runs recorded / expected", f"{a['rows']} / {a['expected']}", f"{b['rows']} / {b['expected']}"),
            ("runs succeeded (all must)", f"{a['successes']} {'ok' if a['all_succeed'] else 'NO'}",
             f"{b['successes']} {'ok' if b['all_succeed'] else 'NO'}"),
            ("median evaluations to success", f"{_f(a['median'], 1)} (885–1,487: {'ok' if a['median_ok'] else 'NO'})",
             f"{_f(b['median'], 1)} (1,358 ± 25%: {'ok' if b['median_ok'] else 'NO'})"),
            ("quartiles; range", _spread(a), _spread(b)),
            ("KS vs their 50 values", f"D = {_f(a.get('ks_D'), 3)}, p = {_f(a.get('ks_p'), 3)} "
             f"(p ≥ 0.01: {'ok' if a['ks_ok'] else 'NO'})", "no raw data: median rule only"),
            ("median active nodes (theirs: 39 at 500)", _f(a["active_nodes_median"], 1), _f(b["active_nodes_median"], 1)),
            ("claims rejected by Python", str(a["claim_rejected"]), str(b["claim_rejected"])),
            ("mean CPU s per run (queue)", _f(a["cpu_s_mean"], 3), _f(b["cpu_s_mean"], 3)),
            ("setting passes", "yes" if a["passed"] else "no", "yes" if b["passed"] else "no")]
    lines += [f"| {c} | {x} | {y} |" for c, x, y in rows]
    if res["verdict"] == "INCOMPLETE":
        lines += ["", "The phase has not finished (or not started): no conclusion yet."]
    return "\n".join(lines) + "\n"


def main(root=None):
    from reach.analysis.s2.load import read_phase
    from reach.paths import ROOT
    root = Path(root or ROOT)
    ref = reference_values(root)
    rows = read_phase(PHASE, root) if (root / "data/runs" / PHASE / "results.jsonl").exists() else []
    res = assess(rows, ref)
    for rel, text in ((REPORT, report(res)), (JSON_OUT, json.dumps(res, indent=1) + "\n")):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    print(f"CGP validation: {res['verdict']} -> {REPORT}")
    return res


if __name__ == "__main__":
    main()

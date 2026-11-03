"""Termination states, completeness and exclusions of the analysed phases (REQ-ANA-35; guardrail row 102).

Stage 1 (OSF a3spg), as filed: "Every run in the registered task lists is included; no exclusion after the fact." "A run
killed from outside (not by its CPU limit) is an error and is rerun once; if still missing it is reported and
excluded." Design 9.9: "Report the termination states of every phase."

- `termination`: per phase, the final row of every task (the queue's last row per key) counted by state: converged,
  budget_exhausted, timeout (the CPU limit, soft or hard), killed (an error row killed from outside by a signal), error
  (any other error row: a crash or no result line), other (completed, early_stopped); and missing tasks. With the
  registered file (data/registration/stage1_tasks.json), a phase is complete when the SHA-256 of its rows' keys and
  their number equal the registered ones (`s2_lock.task_list_hash`), so every registered task has exactly one row and
  no other row is present. Without it (the DRYRUN), tasks are counted from the phase's manifest.
- `exclusions`: every final error row (crash or killed from outside), with phase, key, reason and the last line of its
  error output; these are the rows the analysis leaves out (`load.tidy`).
- `IncompleteError`: raised by `confirm.build` on the confirmatory phases when a registered task has no row (or the
  registered list is missing), unless it is run with allow_incomplete (the result is then labelled not confirmatory).
"""
import json
from pathlib import Path

STATES = ("converged", "budget_exhausted", "timeout", "killed", "error", "other")
KEY_LIST_MAX = 200                       # missing or extra keys listed per phase (the counts are always complete)


class IncompleteError(RuntimeError):
    """Confirmatory phases with registered tasks that have no row: no confirmatory result is reported."""


def state(r):
    t = r.get("terminal_state")
    if t == "error":
        return "killed" if str(r.get("queue_status", "")).startswith("killed_signal") else "error"
    return t if t in ("converged", "budget_exhausted", "timeout") else "other"


def _read(d):
    """{key: final row} and {key: number of rows} of a phase directory (a truncated last line is ignored)."""
    rows, n = {}, {}
    p = d / "results.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[r["key"]] = r
            n[r["key"]] = n.get(r["key"], 0) + 1
    return rows, n


def termination(root, phases, registered=None):
    """{phases: {phase: summary}, registered: bool, complete: bool}. `registered`: the path of the registered task-list
    file; when given, its phases are all checked (and listed) even if not in `phases`."""
    from reach.s2_lock import task_list_hash
    root = Path(root)
    reg = json.loads(Path(registered).read_text()) if registered is not None and Path(registered).exists() else None
    names = list(dict.fromkeys(list(phases) + (list(reg) if reg else [])))
    out = {}
    for ph in names:
        d = root / "data/runs" / ph
        rows, nrows = _read(d)
        man = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else None
        man_keys = [t["key"] for t in (man or {}).get("tasks", []) if "key" in t]
        states = {s: 0 for s in STATES}
        for r in rows.values():
            states[state(r)] += 1
        missing_keys = sorted(set(man_keys) - set(rows))
        extra_keys = sorted(set(rows) - set(man_keys)) if man_keys else []
        if reg is not None and ph in reg:
            tasks, source = reg[ph]["n_tasks"], "registered"
            complete = len(rows) == tasks and task_list_hash([dict(key=k) for k in rows]) == reg[ph]["sha256"]
        else:
            tasks = (man or {}).get("n_tasks", len(man_keys)) if man else len(rows)
            source = "manifest" if man else "rows only"
            complete = not missing_keys and not extra_keys and len(rows) >= tasks
        missing = len(missing_keys) if man_keys else max(0, tasks - len(rows))
        out[ph] = dict(tasks=tasks, rows=len(rows), states=states, missing=missing,
                       missing_keys=missing_keys[:KEY_LIST_MAX], extra_keys=extra_keys[:KEY_LIST_MAX],
                       extra=len(extra_keys), rerun=sum(1 for v in nrows.values() if v > 1), complete=bool(complete),
                       source=source)
    return dict(phases=out, registered=reg is not None, complete=all(v["complete"] for v in out.values()))


def _detail(r):
    lines = [x for x in str(r.get("error_tail") or "").splitlines() if x.strip()]
    return lines[-1].strip()[:160] if lines else ""


def exclusions(root, phases):
    """Every final error row of the phases, with the reason it is left out (Stage 1, Missing data)."""
    out = []
    for ph in phases:
        rows, nrows = _read(Path(root) / "data/runs" / ph)
        for k, r in rows.items():
            st = state(r)
            if st not in ("killed", "error"):
                continue
            if st == "killed":
                sig = str(r.get("queue_status", "")).replace("killed_signal_", "")
                why = f"killed from outside (signal {sig}): an error; rerun once; still an error: excluded"
            else:
                why = "error (a crash or no result line): rerun once; still an error: excluded"
            out.append(dict(phase=ph, key=k, set=r.get("set"), target_id=r.get("target_id"), seed=r.get("seed"),
                            algorithm=r.get("algorithm"), config=r.get("config"), state=st,
                            queue_status=r.get("queue_status"), rows_for_key=nrows.get(k, 1), reason=why,
                            detail=_detail(r)))
    return out


def render_termination(res, title, source, max_rows=120):
    """Termination states per phase, completeness and every excluded row (up to max_rows here; all in the JSON)."""
    t = res.get("completeness", {})
    L = [f"# {title}", "", f"> Index: `docs/README.md`. Generated by `{source}`. **{res['label']}.** Do not edit by hand.",
         "", "## Termination states per phase (final row per task)", "",
         f"Checked against: {'the registered task lists (stage1_tasks.json)' if t.get('registered') else 'phase manifests'}"
         f". Complete: {'yes' if t.get('complete') else 'NO'}. timeout = CPU limit (counted as failure; excluded in "
         "the sensitivity analysis); killed = killed from outside (an error).", "",
         "| Phase | tasks | rows | " + " | ".join(STATES) + " | missing | rerun keys | complete |",
         "|---|---|---|" + "---|" * len(STATES) + "---|---|---|"]
    for ph, v in t.get("phases", {}).items():
        L.append(f"| {ph} | {v['tasks']} | {v['rows']} | " + " | ".join(str(v["states"][s]) for s in STATES)
                 + f" | {v['missing']} | {v['rerun']} | {'yes' if v['complete'] else 'NO'} |")
    miss = [(ph, k) for ph, v in t.get("phases", {}).items() for k in v["missing_keys"] + v["extra_keys"]]
    if miss:
        L += ["", "Tasks without a row, or rows not in the task list (first ones; counts above):", ""]
        L += [f"- {ph}: `{k}`" for ph, k in miss[:20]]
    ex = res.get("exclusions", [])
    L += ["", f"## Excluded rows ({len(ex)})", "", "Each row left out of the analysis, with its reason (Stage 1, Missing "
          "data: no exclusion after the fact).", "", "| Phase | key | set | algorithm | reason | last error line |",
          "|---|---|---|---|---|---|"]
    for e in ex[:max_rows]:
        L.append(f"| {e['phase']} | `{e['key']}` | {e.get('set') or '–'} | {e.get('algorithm') or '–'} | {e['reason']} | "
                 f"{(e.get('detail') or '–').replace('|', '/')} |")
    if len(ex) > max_rows:
        L.append(f"| … | {len(ex) - max_rows} more: `exclusions` in the analysis JSON | | | | |")
    return "\n".join(L) + "\n"

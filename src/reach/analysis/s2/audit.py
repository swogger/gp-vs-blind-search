"""Quality audits of the calibration runs (D068). They check records, not hypotheses: only counts of agreement are
reported, never success rates.

1. Reproducibility: rerun a random sample of completed calibration runs (phase s2_audit_repro, through the queue, on
   the current code) and require every recorded field except timing and bookkeeping to be identical.
2. Success labels: recompute every "converged" run's exactness, and a sample of failures', with an independent
   evaluator (`audit_labels`).
"""
import random

IGNORE = {"key", "cpu_seconds", "wall_seconds", "cpu_s", "wall_s", "finished", "code_hash", "git", "queue_status",
          "exit", "signal", "cpu_limit", "block", "audit_of",
          "child_cpu_seconds", "rust_cpu_seconds"}          # CGP's own timing fields (guardrail row 137)


def compare_rows(orig, again):
    """Names of the fields that differ (sorted), ignoring timing and bookkeeping."""
    keys = (set(orig) | set(again)) - IGNORE
    return sorted(k for k in keys if orig.get(k) != again.get(k))


def sample_keys(rows, n, seed=20261022):
    """n keys of runs that ended normally (converged or budget_exhausted), drawn without replacement, sorted."""
    ok = sorted(k for k, r in rows.items() if r.get("terminal_state") in ("converged", "budget_exhausted"))
    return sorted(random.Random(seed).sample(ok, min(n, len(ok))))


def repro_report(root=None):
    """Compare every s2_audit_repro row with its original; returns per-phase counts and the differing field names."""
    from pathlib import Path
    from reach.paths import ROOT
    from reach.runner.queue import read_results
    root = Path(root or ROOT)
    again = read_results(root / "data/runs/s2_audit_repro/results.jsonl")
    out = {}
    for key, r in sorted(again.items()):
        ph, orig_key = r["audit_of"], key[len("audit|"):]
        orig = read_results(root / f"data/runs/{ph}/results.jsonl")[orig_key]
        d = compare_rows(orig, r)
        e = out.setdefault(ph, dict(checked=0, identical=0, differing={}))
        e["checked"] += 1
        if d:
            e["differing"][orig_key] = d
        else:
            e["identical"] += 1
    return out


# ---- Big-tree reruns (D091; REQ-ANA-76): check 1 and check 2 on the exact runs that recorded no expression ----

NEW_FIELD = "best_expr_full"                       # the whole best tree, recorded only by the reruns (--print_tree)
BIGTREE_IGNORE = IGNORE | {"audit_key"}            # timing and bookkeeping, as check 1, plus the original row's key


def _nodes(expr):
    """Node count by audit_labels' own parser (shares no code with the engine's)."""
    from reach.analysis.s2.audit_labels import parse

    def n(t):
        return 1 if isinstance(t, str) else 1 + sum(n(x) for x in t[1])
    return n(parse(expr))


def bigtree_compare(orig, again):
    """Problems of one rerun: every differing field except BIGTREE_IGNORE and the new field (sorted), then a missing
    full expression or one whose size is not the original's final_tree_size."""
    keys = (set(orig) | set(again)) - BIGTREE_IGNORE - {NEW_FIELD}
    out = sorted(k for k in keys if orig.get(k) != again.get(k))
    if NEW_FIELD in orig:
        out.append("full expression in the original")
    full = again.get(NEW_FIELD)
    if not full:
        out.append("no full expression")
    elif _nodes(full) != orig.get("final_tree_size"):
        out.append(f"full expression has {_nodes(full)} nodes, not {orig.get('final_tree_size')}")
    return out


REVISIT_IGNORE = BIGTREE_IGNORE | {"audit_of", "rv"}   # bookkeeping, timing and the counter's new field


def revisit_compare(orig, again):
    """Problems of one H6d re-run (D098): every differing field except REVISIT_IGNORE (sorted), then a missing rv or
    one whose evaluations are not the run's (the population of every evaluated generation)."""
    keys = (set(orig) | set(again)) - REVISIT_IGNORE
    out = sorted(k for k in keys if orig.get(k) != again.get(k))
    rv = again.get("rv")
    if not rv:
        return out + ["no rv"]
    n = again.get("pop_size", 0) * (again.get("gens_reached", 0) + (again.get("terminal_state") == "converged"))
    if rv["stop"]["n"] != n:
        out.append(f"rv counts {rv['stop']['n']} evaluations, the run made {n}")
    return out


def revisit_report(root, phase):
    """Each planned task of an H6d re-run phase against its original row, per source phase: planned, rerun,
    identical, differing {original key: problems}, missing; plus CPU seconds of the re-runs, of their originals and of
    the counter. ok only if nothing differs and nothing is missing."""
    import json
    from pathlib import Path
    from reach.runner.queue import read_results
    root = Path(root)
    tasks = json.loads((root / f"data/runs/{phase}/manifest.json").read_text())["tasks"]
    again = read_results(root / f"data/runs/{phase}/results.jsonl")
    out, ok = {}, True
    cpu = dict(rerun=0.0, original=0.0, counter=0.0)
    for src in sorted({t["meta"]["audit_of"] for t in tasks}):
        orig = read_results(root / f"data/runs/{src}/results.jsonl")
        e = dict(planned=0, rerun=0, identical=0, differing={}, missing=[])
        for t in (t for t in tasks if t["meta"]["audit_of"] == src):
            e["planned"] += 1
            k, r = t["meta"]["audit_key"], again.get(t["key"])
            if r is None:
                e["missing"].append(k)
                continue
            e["rerun"] += 1
            d = revisit_compare(orig[k], r)
            if d:
                e["differing"][k] = d
            else:
                e["identical"] += 1
            cpu["rerun"] += r.get("cpu_seconds") or 0.0
            cpu["original"] += orig[k].get("cpu_seconds") or 0.0
            cpu["counter"] += (r.get("rv") or {}).get("cpu_seconds") or 0.0
        ok = ok and not e["differing"] and not e["missing"]
        out[src] = e
    return dict(ok=ok, phase=phase, sources=out, cpu_seconds=cpu)


def print_revisit_report(rep):
    for src, e in rep["sources"].items():
        print(f"{rep['phase']} <- {src}: planned {e['planned']}, rerun {e['rerun']}, identical {e['identical']}, "
              f"differing {len(e['differing'])}, missing {len(e['missing'])}")
        for k, d in list(e["differing"].items())[:20]:
            print(f"  {k}: {', '.join(d)}")
    c = rep["cpu_seconds"]
    if c["original"]:
        print(f"CPU: re-runs {c['rerun'] / 3600:.2f} h, originals {c['original'] / 3600:.2f} h "
              f"(x{c['rerun'] / c['original']:.2f}); counter {c['counter'] / 3600:.2f} h")
    print("OK" if rep["ok"] else "NOT OK")


def bigtree_report(root, phase):
    """Each planned task of `phase` (s2_audit_bigtrees or its sample; the manifest names the original row) against
    its original, per source phase: planned, rerun, identical, differing {original key: problems}, missing (no rerun
    row yet), and the label check of the full expressions (audit_labels.audit on every rerun, no sample). ok only if
    nothing differs, nothing is missing, nothing is disputed and nothing is unchecked."""
    import json
    from pathlib import Path
    from reach.analysis.s2 import audit_labels as L
    from reach.runner.queue import read_results
    root = Path(root)
    tasks = json.loads((root / f"data/runs/{phase}/manifest.json").read_text())["tasks"]
    again = read_results(root / f"data/runs/{phase}/results.jsonl")
    out, ok = {}, True
    for ph in sorted({t["meta"]["audit_of"] for t in tasks}):
        orig = read_results(root / f"data/runs/{ph}/results.jsonl")
        mine = [t for t in tasks if t["meta"]["audit_of"] == ph]
        e = dict(planned=len(mine), rerun=0, identical=0, differing={}, missing=[])
        rows, targets = [], {}
        for t in mine:
            k, r = t["meta"]["audit_key"], again.get(t["key"])
            if r is None:
                e["missing"].append(k)
                continue
            e["rerun"] += 1
            d = bigtree_compare(orig[k], r)
            if d:
                e["differing"][k] = d
            else:
                e["identical"] += 1
            if r.get(NEW_FIELD):
                rows.append(dict(r, best_expr=r[NEW_FIELD]))
        for g in sorted({r["set"] for r in rows}):
            f = root / f"data/targets_s2/{g}/{L.targets_file(ph, 'calibration')}.json"
            targets.update({(g, x["target_id"]): x for x in json.loads(f.read_text())})
        lab = L.audit(rows, targets, n_fail=len(rows))
        e["labels"] = {k: (v if isinstance(v, int) else sorted(v)) for k, v in lab.items()}
        ok &= not (e["differing"] or e["missing"] or lab["exact_disputed"] or lab["failures_disputed"]
                   or lab["unchecked"] or lab["exact_ieee_only"])
        out[ph] = e
    out["ok"] = bool(ok and out)
    return out


def print_bigtree_report(rep):
    for ph, e in sorted((k, v) for k, v in rep.items() if k != "ok"):
        lab = e.get("labels", {})
        print(f"{ph}: planned {e['planned']}, rerun {e['rerun']}, identical {e['identical']}, differing "
              f"{len(e['differing'])}, missing {len(e['missing'])}; labels: exact checked {lab.get('exact_checked', 0)},"
              f" near {len(lab.get('exact_near_miss', []))}, disputed {len(lab.get('exact_disputed', []))}, "
              f"failures checked {lab.get('failures_checked', 0)}, disputed {len(lab.get('failures_disputed', []))}, "
              f"unchecked {len(lab.get('unchecked', []))}")
        for k, d in list(e["differing"].items())[:20]:
            print(f"  DIFFERS {k}: {d}")
    print("ALL IDENTICAL AND AGREED" if rep["ok"] else "PROBLEMS FOUND")


def main():
    """python -m reach.analysis.s2.audit bigtrees PHASE: writes data/analysis/audit_bigtrees[_sample].json."""
    import argparse
    import json
    from reach.paths import ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["bigtrees"])
    ap.add_argument("phase", choices=["s2_audit_bigtrees", "s2_audit_bigtrees_sample"])
    a = ap.parse_args()
    rep = bigtree_report(ROOT, a.phase)
    print_bigtree_report(rep)
    out = ROOT / f"data/analysis/{a.phase.replace('s2_audit_', 'audit_')}.json"
    out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print("written", out.relative_to(ROOT))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

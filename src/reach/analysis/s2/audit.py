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


def revisit_compare(orig, again, extra_ignore=(), check_keys=False, root=None):
    """Problems of one H6d re-run (D098): every differing field except REVISIT_IGNORE (sorted), then a missing rv or
    one whose evaluations are not the run's (the population of every evaluated generation). Stage 5 (D125): fields in
    extra_ignore are new fields of the re-run (`keys`); with check_keys (the row's task asked for a side file) the side
    file must exist and have the SHA-256 the row records."""
    keys = (set(orig) | set(again)) - REVISIT_IGNORE - set(extra_ignore)
    out = sorted(k for k in keys if orig.get(k) != again.get(k))
    if check_keys:
        import hashlib
        from pathlib import Path
        k = again.get("keys") or {}
        f = Path(root or ".") / k["file"] if k.get("file") else None
        if f is None or not f.exists():
            out.append("no side file")
        elif hashlib.sha256(f.read_bytes()).hexdigest() != k.get("sha256"):
            out.append("side file differs from its row")
    rv = again.get("rv")
    if not rv:
        return out + ["no rv"]
    n = again.get("pop_size", 0) * (again.get("gens_reached", 0) + (again.get("terminal_state") == "converged"))
    if rv["stop"]["n"] != n:
        out.append(f"rv counts {rv['stop']['n']} evaluations, the run made {n}")
    return out


def revisit_report(root, phase, extra_ignore=(), check_keys=False):
    """Each planned task of an H6d re-run phase against its original row, per source phase: planned, rerun,
    identical, differing {original key: problems}, missing (no re-run row), missing_original (no original row); plus
    CPU seconds of the re-runs, of their originals and of the counter. ok only if nothing differs and nothing is
    missing. Files are read with their gzip fallback (guardrail row 176). With check_keys (Stage 5) a side file is
    checked exactly for the tasks whose argv has --keys_out, and a `keys` field on any other task is a problem
    (guardrail row 174)."""
    from pathlib import Path
    from reach.runner.phases import _manifest_tasks, phase_rows
    root = Path(root)
    tasks = _manifest_tasks(phase, root)
    again = phase_rows(phase, root)
    out, ok = {}, True
    cpu = dict(rerun=0.0, original=0.0, counter=0.0)
    for src in sorted({t["meta"]["audit_of"] for t in tasks}):
        orig = phase_rows(src, root)
        e = dict(planned=0, rerun=0, identical=0, differing={}, missing=[], missing_original=[])
        for t in (t for t in tasks if t["meta"]["audit_of"] == src):
            e["planned"] += 1
            k, r = t["meta"]["audit_key"], again.get(t["key"])
            if r is None:
                e["missing"].append(k)
                continue
            if k not in orig:
                e["missing_original"].append(k)
                continue
            e["rerun"] += 1
            wants = check_keys and "--keys_out" in t.get("argv", ())
            d = revisit_compare(orig[k], r, extra_ignore, wants, root)
            if check_keys and not wants and "keys" in r:
                d = d + ["keys without --keys_out"]
            if d:
                e["differing"][k] = d
            else:
                e["identical"] += 1
            cpu["rerun"] += r.get("cpu_seconds") or 0.0
            cpu["original"] += orig[k].get("cpu_seconds") or 0.0
            cpu["counter"] += (r.get("rv") or {}).get("cpu_seconds") or 0.0
        ok = ok and not e["differing"] and not e["missing"] and not e["missing_original"]
        out[src] = e
    return dict(ok=ok, phase=phase, sources=out, cpu_seconds=cpu)


def print_revisit_report(rep):
    for src, e in rep["sources"].items():
        print(f"{rep['phase']} <- {src}: planned {e['planned']}, rerun {e['rerun']}, identical {e['identical']}, "
              f"differing {len(e['differing'])}, missing {len(e['missing'])}, missing originals "
              f"{len(e.get('missing_original', []))}")
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


# ---- Stage 5 arm 7 (D125; REQ-RUN-49): a long run's first 200 generations against its registered s2_ho_a0 run ----

def prefix_mismatch(long, short):
    """Problems of a long run against the registered shorter run of the same task (prefix property, REQ-ENG-03), or
    None when they cannot be compared (either stopped before the shorter run's last generation by its CPU limit)."""
    n = short.get("generations")
    ok_states = ("converged", "budget_exhausted")
    if short.get("terminal_state") not in ok_states:
        return None
    if long.get("terminal_state") not in ok_states and (long.get("gens_reached") or 0) < n:
        return None
    if short.get("exact"):
        fields = ("exact", "gen_exact", "best_expr", "final_tree_size") + (("best_expr_full",) if "best_expr_full" in short
                                                                           else ())
        return [f for f in fields if long.get(f) != short.get(f)]
    if long.get("exact") and long.get("gen_exact") is not None and long["gen_exact"] < n:
        return [f"exact at generation {long['gen_exact']}, before the shorter run's {n}"]
    return []


def long_extra_report(root, phase):
    """Every planned task of the long-run phase that names a registered shorter run (meta prefix_of, prefix_key)
    against that run's row: compared, mismatch {key: problems}, not comparable, missing (no long or no short row),
    no_prefix (levels without a shorter run). ok only if nothing differs and nothing is missing."""
    from pathlib import Path
    from reach.runner.phases import _manifest_tasks, phase_rows
    root = Path(root)
    tasks = _manifest_tasks(phase, root)
    rows = phase_rows(phase, root)
    short_rows = {}
    out = dict(phase=phase, compared=0, mismatch={}, not_comparable=[], missing=[], no_prefix=0)
    for t in tasks:
        m = t.get("meta", {})
        if "prefix_key" not in m:
            out["no_prefix"] += 1
            continue
        src = m["prefix_of"]
        if src not in short_rows:
            short_rows[src] = phase_rows(src, root)
        long, short = rows.get(t["key"]), short_rows[src].get(m["prefix_key"])
        if long is None or short is None:
            out["missing"].append(t["key"])
            continue
        d = prefix_mismatch(long, short)
        if d is None:
            out["not_comparable"].append(t["key"])
            continue
        out["compared"] += 1
        if d:
            out["mismatch"][t["key"]] = d
    out["ok"] = not out["mismatch"] and not out["missing"]
    return out


def print_long_extra_report(rep):
    print(f"{rep['phase']}: compared {rep['compared']} with their s2_ho_a0 prefix, mismatches {len(rep['mismatch'])}, "
          f"not comparable {len(rep['not_comparable'])}, missing {len(rep['missing'])}, no prefix {rep['no_prefix']}")
    for k, d in list(rep["mismatch"].items())[:20]:
        print(f"  MISMATCH {k}: {', '.join(d)}")
    print("OK" if rep["ok"] else "NOT OK")


# ---- Re-scoring of the held-out and restart phases (D125; REQ-RUN-50; guardrail row 171) ----

def recheck_extended(phase, g, keys, root):
    """Guardrail row 180: the exact runs listed as disputed by a re-scoring row written before the 'extended_only'
    class existed, checked again with audit_labels.max_rel_diff_extended (Study 2's guarded extended reals) against
    their targets; returns (extended-only keys, still disputed keys)."""
    if not keys:
        return [], []
    from reach.analysis.s2 import audit_labels as L
    from reach.runner.phases import phase_rows
    from reach.tasks.s2_rescore import targets_of
    rows, (targets, _) = phase_rows(phase, root), targets_of(g, root)
    ext, still = [], []
    for k in keys:
        r = rows.get(k)
        expr = r and (r.get("best_expr") or r.get("best_expr_full"))
        t = r and targets.get((g, r["target_id"]))
        ok = bool(expr and t and r.get("algorithm") != "operon"
                  and L.classify(L.max_rel_diff_extended(expr, t["expression"], t["target_scale"])) != "different")
        (ext if ok else still).append(k)
    return ext, still


def heldout_labels_report(root, phase="s2_audit_labels_ho", exact_rows=None):
    """Aggregates the re-scoring tasks: per source phase the sums of the tasks' counts, and ok only if every task has
    a row, every exact run was checked (exact_checked equals the task's exact_rows, and with exact_rows given
    {(phase, set): n} also the count read from the source phase), nothing is disputed or unchecked, every full
    expression has its row's size and every row names its target file."""
    import json
    from pathlib import Path
    from reach.runner.queue import read_results
    root = Path(root)
    tasks = json.loads((root / f"data/runs/{phase}/manifest.json").read_text())["tasks"]
    rows = read_results(root / f"data/runs/{phase}/results.jsonl")
    out, problems = {}, []
    for t in tasks:
        ph, g = t["meta"]["phase"], t["meta"]["set"]
        r = rows.get(t["key"])
        e = out.setdefault(ph, dict(sets=0, exact_rows=0, exact_checked=0, exact_near_miss=0, exact_ieee_only=0,
                                    exact_extended_only=[], exact_disputed=[], failures_checked=0, failures_disputed=[], unchecked=0,
                                    full_size_mismatch=[], target_file_mismatch=[]))
        if r is None or r.get("terminal_state") != "completed":
            problems.append(f"{t['key']}: no completed row")
            continue
        e["sets"] += 1
        for k in ("exact_rows", "exact_checked", "failures_checked"):
            e[k] += r[k]
        for k in ("exact_near_miss", "exact_ieee_only"):
            e[k] += len(r[k])
        e["unchecked"] += len(r["unchecked"])
        ext, still = recheck_extended(ph, g, r["exact_disputed"], root)        # rows written before row 180's class
        r = dict(r, exact_disputed=still)
        e["exact_extended_only"] += r.get("exact_extended_only", []) + ext
        for k in ("exact_disputed", "failures_disputed", "full_size_mismatch", "target_file_mismatch"):
            e[k] += r[k]
        if r["exact_checked"] != r["exact_rows"]:
            problems.append(f"{t['key']}: {r['exact_checked']} of {r['exact_rows']} exact runs checked")
        if exact_rows is not None and exact_rows.get((ph, g)) != r["exact_rows"]:
            problems.append(f"{t['key']}: {r['exact_rows']} exact rows, the phase holds {exact_rows.get((ph, g))}")
        for k in ("exact_disputed", "failures_disputed", "full_size_mismatch", "target_file_mismatch", "unchecked"):
            if r[k]:
                problems.append(f"{t['key']}: {k} {len(r[k])}")
    return dict(ok=not problems, phase=phase, phases=out, problems=problems)


def print_heldout_labels_report(rep):
    for ph, e in sorted(rep["phases"].items()):
        print(f"{ph}: {e['sets']} sets, exact {e['exact_checked']} of {e['exact_rows']} checked, near misses "
              f"{e['exact_near_miss']}, IEEE-only {e['exact_ieee_only']}, extended-reals-only "
              f"{len(e['exact_extended_only'])}, disputed {len(e['exact_disputed'])}; "
              f"failures checked {e['failures_checked']}, disputed {len(e['failures_disputed'])}; unchecked "
              f"{e['unchecked']}")
    for p in rep["problems"][:30]:
        print("  PROBLEM", p)
    print("ALL EXACT RUNS CHECKED AND AGREED" if rep["ok"] else "PROBLEMS FOUND")


def main():
    """python -m reach.analysis.s2.audit bigtrees PHASE: writes data/analysis/audit_bigtrees[_sample].json.
    python -m reach.analysis.s2.audit labels_heldout: writes data/analysis/audit_labels_heldout.json (D125).
    python -m reach.analysis.s2.audit labels_s5: writes data/analysis/audit_labels_s5.json (REQ-RUN-51)."""
    import argparse
    import json
    from reach.paths import ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["bigtrees", "labels_heldout", "labels_s5"])
    ap.add_argument("phase", nargs="?", choices=["s2_audit_bigtrees", "s2_audit_bigtrees_sample"])
    a = ap.parse_args()
    if a.check in ("labels_heldout", "labels_s5"):
        from reach.runner.phases import RESCORE_PHASES, phase_rows
        sources, qa, name = ((RESCORE_PHASES, "s2_audit_labels_ho", "audit_labels_heldout") if a.check == "labels_heldout"
                             else (("s5_long_extra",), "s2_audit_labels_s5", "audit_labels_s5"))
        n = {}
        for ph in sources:
            for r in phase_rows(ph).values():
                n[(ph, r["set"])] = n.get((ph, r["set"]), 0) + bool(r.get("exact"))
        rep = heldout_labels_report(ROOT, qa, exact_rows=n)
        print_heldout_labels_report(rep)
        out = ROOT / f"data/analysis/{name}.json"
        out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
        print("written", out.relative_to(ROOT))
        return 0 if rep["ok"] else 1
    rep = bigtree_report(ROOT, a.phase)
    print_bigtree_report(rep)
    out = ROOT / f"data/analysis/{a.phase.replace('s2_audit_', 'audit_')}.json"
    out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print("written", out.relative_to(ROOT))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

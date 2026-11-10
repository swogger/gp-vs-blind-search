"""Union histograms of one restart target's key sets (Stage 5 arm 5a, D125; REQ-RUN-48) -> one RESULT line. No search.

    python -m reach.tasks.s5_union --set G --target_id T --rows data/runs/s5_revisit_restart/results.jsonl \
        --enum_dir DIR --n_runs 64

Reads the target's rows of the re-execution phase (each names its side file and the file's SHA-256), refuses a missing
row or file and a file whose SHA-256 is not its row's, and counts, per level K of the set's lookup and per prefix
(5,000 and 10,000 evaluations: 50 and 100 generations), H_K[m], the number of functions of level K covered by exactly
m of the runs (m = 0 .. n; H_K[0] = U_K minus the covered ones), over all runs ("all") and over the runs without an
exact hit by that prefix ("nohit"). A run's own target is never in its key set (REQ-ENG-28). The side files hold
indices into the lookup they were written against, so every row's `rv.lookup.sha256` must be the loaded lookup's, and
the registered one (data/registration/stage5_lookups.json) when it is recorded (guardrail row 175). Only the lines
naming the target are parsed (the rows file holds 105,600 rows)."""
import argparse
import hashlib
import json
import sys

import numpy as np

from reach import grammar as G
from reach.revisit import Lookup

PREFIXES = (5_000, 10_000)


def histograms(runs, lv, top, prefixes=PREFIXES):
    """runs: [(idx, first, hit_eval or None)]; lv: level of every lookup index. {prefix: {all, nohit, n_all, n_nohit}}
    with each histogram {K: [H_K[0], ..., H_K[n]]} over the levels 1..top (n = the number of runs in that group)."""
    u = np.bincount(lv, minlength=top + 1)
    out = {}
    for p in prefixes:
        entry = {}
        for name, group in (("all", runs), ("nohit", [r for r in runs if r[2] is None or r[2] > p])):
            n = len(group)
            h = np.zeros((top + 1, n + 1), np.int64)
            if n:
                cov = np.concatenate([idx[first <= p] for idx, first, _ in group]).astype(np.int64)
                keys, m = np.unique(cov, return_counts=True)
                np.add.at(h, (lv[keys], m), 1)
            h[:, 0] = u - h[:, 1:].sum(axis=1)
            entry[name] = {str(k): h[k].tolist() for k in range(1, top + 1) if u[k]}
            entry[f"n_{name}"] = n
        out[str(p)] = entry
    return out


def target_rows(path, set_name, target_id):
    """The last row per key among the rows of this set and target (read_results' rule; a truncated line is ignored),
    parsing only the lines that name the target."""
    rows, tag = {}, f'"{target_id}"'
    with open(path) as fh:
        for line in fh:
            if tag not in line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("target_id") == target_id and r.get("set") == set_name:
                rows[r["key"]] = r
    return list(rows.values())


def main(argv=None):
    from reach.paths import ROOT
    ap = argparse.ArgumentParser()
    for a in ("--set", "--target_id", "--rows", "--enum_dir"):
        ap.add_argument(a, required=True)
    ap.add_argument("--n_runs", type=int, required=True)
    a = ap.parse_args(argv)
    rows = target_rows(a.rows, a.set, a.target_id)
    if len(rows) != a.n_runs or any(r.get("terminal_state") == "error" for r in rows):
        sys.exit(f"refused: {len(rows)} usable rows for {a.target_id}, expected {a.n_runs}")
    g = G.get(a.set)
    lookup = Lookup(g, a.enum_dir)
    for r in rows:
        sha = ((r.get("rv") or {}).get("lookup") or {}).get("sha256")
        if sha != lookup.sha256:
            sys.exit(f"refused: {r['key']}: its key set was written against lookup {sha}, not the loaded {lookup.sha256}")
    reg = ROOT / "data/registration/stage5_lookups.json"
    if reg.exists():
        want = json.loads(reg.read_text()).get(a.set)
        if want is not None and want != lookup.sha256:
            sys.exit(f"refused: the loaded lookup {lookup.sha256} is not the one stage5_lookups.json registers ({want})")
    runs = []
    for r in sorted(rows, key=lambda r: r["key"]):
        k = r.get("keys") or {}
        try:
            raw = open(k["file"], "rb").read()
        except (KeyError, OSError):
            sys.exit(f"refused: {r['key']}: no side file")
        if hashlib.sha256(raw).hexdigest() != k.get("sha256"):
            sys.exit(f"refused: {r['key']}: side file {k['file']} differs from its row's SHA-256")
        with np.load(k["file"]) as z:
            idx, first = z["idx"].astype(np.int64), z["first"].astype(np.int64)
        hit = (r["gen_exact"] + 1) * r["pop_size"] if r.get("exact") else None
        runs.append((idx, first, hit))
    lv = lookup.lv.astype(np.int64)
    print("RESULT " + json.dumps(dict(set=a.set, target_id=a.target_id, n_runs=len(runs), terminal_state="completed",
                                      lookup=lookup.describe(), hist=histograms(runs, lv, int(lookup.top)))),
          flush=True)


if __name__ == "__main__":
    main()

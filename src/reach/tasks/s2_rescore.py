"""Independent re-scoring of one phase's runs on one set (QA; D068's check 2 extended to the held-out and restart
phases, D125; REQ-RUN-50) -> one RESULT line. No search.

    python -m reach.tasks.s2_rescore --phase s4_restart --set R-ARITH --n_fail 50

Every exact run of the phase on the set, and n_fail sampled failures, are recomputed by `audit_labels.audit`: code
that shares nothing with the engine (a small parser, 30-digit mpmath on the identity grid, explicit truth tables; for
Operon its printed model with a refitted scale and intercept), as the calibration phases were (D068, D069, D091). The
expression is the row's `best_expr`, or `best_expr_full` (every Stage 2 and Stage 4 tree-GP task prints the whole
tree, REQ-RUN-33) when the tree had more than 200 nodes; a full expression whose node count is not the row's
final_tree_size is listed. Targets come from the set's heldout, restart and bench files; a row whose targets_sha256 is
not the SHA-256 of the file holding its target is listed."""
import argparse
import hashlib
import json

from reach.analysis.s2 import audit_labels as L
from reach.analysis.s2.audit import _nodes
from reach.paths import ROOT
from reach.runner.phases import phase_rows

TARGET_FILES = ("heldout", "restart", "bench")


def targets_of(g, root=ROOT):
    """{(set, target_id): target} over the set's target files, and {target_id: file SHA-256}; ids must be unique."""
    out, sha = {}, {}
    for name in TARGET_FILES:
        p = root / f"data/targets_s2/{g}/{name}.json"
        if not p.exists():
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        for t in json.loads(p.read_text()):
            if (g, t["target_id"]) in out:
                raise ValueError(f"{g}: target id {t['target_id']} in two target files")
            out[(g, t["target_id"])], sha[t["target_id"]] = t, h
    return out, sha


def rescore(rows, targets, sha, n_fail):
    """audit_labels.audit on the rows, with full expressions in place of missing ones; plus the bookkeeping checks."""
    size_bad, file_bad, use = [], [], []
    for r in rows:
        if r.get("targets_sha256") and r["targets_sha256"] != sha.get(r["target_id"]):
            file_bad.append(r["key"])
        if r.get("algorithm") != "operon" and not r.get("best_expr") and r.get("best_expr_full"):
            if _nodes(r["best_expr_full"]) != r.get("final_tree_size"):
                size_bad.append(r["key"])
            r = dict(r, best_expr=r["best_expr_full"])
        use.append(r)
    lab = L.audit(use, targets, n_fail=n_fail)
    return dict({k: (v if isinstance(v, (int, dict)) else sorted(v)) for k, v in lab.items()},
                rows=len(rows), exact_rows=sum(1 for r in rows if r.get("exact")),
                full_size_mismatch=sorted(size_bad), target_file_mismatch=sorted(file_bad))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--n_fail", type=int, required=True)
    a = ap.parse_args(argv)
    rows = [r for r in phase_rows(a.phase).values() if r.get("set") == a.set]
    targets, sha = targets_of(a.set)
    out = rescore(rows, targets, sha, a.n_fail)
    print("RESULT " + json.dumps(dict(phase=a.phase, set=a.set, terminal_state="completed", **out)), flush=True)


if __name__ == "__main__":
    main()

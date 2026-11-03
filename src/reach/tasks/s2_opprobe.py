"""Operator probe (tool T13; lesson 20; REQ-RULE-08): does every variation operator do what it claims?

For each set, parents are drawn like GP's initial population and each operator is applied once per event. Recorded per
operator: the share of events whose child computes a different function from the parent (grid hash or truth table), the
mean size change, and the share rejected by the depth or node cap (the child is then a copy of the parent). An operator
whose changed share, among accepted events with a defined parent (two undefined trees hash alike), is below FLAG_BELOW
is flagged as a near no-op (EML-GP's semantic crossover changed nothing in 99%
of events). The changed share among events not blocked by a cap is reported too.

    python -m reach.tasks.s2_opprobe [--sets ...] [--n 2000]      -> RESULT line per set; docs/status gets the table
"""
import argparse
import json
import math
import random

import numpy as np

from reach import grammar as G
from reach.engines.treegp import MAX_DEPTH, _patch
from reach.galib.evolution import crossover, mutate
from reach.galib.generator import generate_random_tree

FLAG_BELOW = 0.10          # a near no-op (EML-GP's semantic crossover changed <= 3% of events); neutrality above
                           # this is a property of the set (small Boolean spaces are highly neutral), not a defect


def _random_tree(g, rng):
    return generate_random_tree(rng.randint(2, MAX_DEPTH), g.operators, list(g.constants), list(g.variables),
                                rng.choice(["full", "grow"]))


def operators(g):
    mut_depth = max(2, round(math.sqrt(MAX_DEPTH)))
    cons, vars_ = list(g.constants), list(g.variables)
    return {
        "subtree_mutation": lambda t, rng: mutate(t, g.operators, cons, vars_, mut_depth, absolute_max_depth=MAX_DEPTH),
        "subtree_crossover": lambda t, rng: crossover(t, _random_tree(g, rng), max_depth=MAX_DEPTH)[0],
        "headless_crossover": lambda t, rng: crossover(t, _random_tree(g, rng), max_depth=MAX_DEPTH)[0],
    }


def _sem(g, t):
    if G.count_nodes(t) > G.MAX_NODES:
        return None
    if g.kind == "bool":
        return G.evaluate(g, t)
    v = G.evaluate(g, t, G.sample_grid(g))
    return G.scaled_func_hash(v, max(1.0, float(np.nanmax(np.abs(np.where(np.isfinite(v), v, 0))))))


def _defined(g, t):
    """Real: finite everywhere on the grid; Boolean: always."""
    if g.kind == "bool":
        return True
    return G.count_nodes(t) <= G.MAX_NODES and bool(np.isfinite(G.evaluate(g, t, G.sample_grid(g))).all())


def probe(g, ops, n=2000, seed=1):
    rng = random.Random(seed)
    _patch(rng)                                     # ga-lib draws from the global random module
    out = {}
    for name, op in ops.items():
        changed = rejected = defined = changed_def = accepted_def = 0
        dsize = []
        for _ in range(n):
            parent = _random_tree(g, rng)
            child = op(parent, rng)
            same_tree = G.expr_str(child) == G.expr_str(parent)
            rejected += same_tree
            ch = _sem(g, child) != _sem(g, parent)
            changed += ch
            if _defined(g, parent):
                defined += 1
                if not same_tree:
                    accepted_def += 1
                    changed_def += ch
            dsize.append(G.count_nodes(child) - G.count_nodes(parent))
        out[name] = dict(events=n, changed_share=changed / n, rejected_share=rejected / n,
                         changed_share_accepted=changed / max(n - rejected, 1),
                         defined_parent_share=defined / n,
                         changed_share_defined=changed_def / max(accepted_def, 1),
                         size_change_mean=float(np.mean(dsize)), size_change_sd=float(np.std(dsize)),
                         flag=(accepted_def >= 50 and changed_def / accepted_def < FLAG_BELOW)
                         or (changed / n < FLAG_BELOW and rejected / n > 0.5),       # returns the parent: a no-op
                         few_defined=accepted_def < 50)
    return out


def merge_results(path, new):
    """Add or replace the given sets in the probe file, keeping every other set's values."""
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(new)
    path.write_text(json.dumps(old, indent=1))
    return old


def main():
    """Sets -> merged into data/analysis/opprobe.json; docs/status/opprobe_report.md lists every probed set."""
    from reach import s2_sets as S
    from reach.paths import ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=list(S.SETS))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261020)
    a = ap.parse_args()
    (ROOT / "data/analysis").mkdir(parents=True, exist_ok=True)
    res = merge_results(ROOT / "data/analysis/opprobe.json",
                        {s: probe(G.get(s), operators(G.get(s)), a.n, a.seed) for s in a.sets})
    L = ["# Operator probe (T13; lesson 20)", "", "> Index: `docs/README.md`. Generated by `reach.tasks.s2_opprobe`; values "
         "in `data/analysis/opprobe.json`. Parents drawn like GP's initial population; n events per operator.", "",
         "| Set | defined parents | changed (defined parents): mutation / crossover / headless | blocked by a cap | flags |",
         "|---|---|---|---|---|"]
    for s, r in res.items():
        m, c, h = r["subtree_mutation"], r["subtree_crossover"], r["headless_crossover"]
        flags = [k for k, v in r.items() if v["flag"]] or (["few defined parents"] if any(v["few_defined"] for v in r.values()) else [])
        L.append(f"| {s} | {m['defined_parent_share']:.2f} | {m['changed_share_defined']:.2f} / {c['changed_share_defined']:.2f} / "
                 f"{h['changed_share_defined']:.2f} | {m['rejected_share']:.2f} / {c['rejected_share']:.2f} / "
                 f"{h['rejected_share']:.2f} | {', '.join(flags) or 'none'} |")
    (ROOT / "docs/status/opprobe_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

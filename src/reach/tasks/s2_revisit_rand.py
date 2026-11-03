"""Negative control for H6d (D098; design 4.5; REQ-RUN-40): trees drawn exactly as the random-sampling baseline draws
them (`s2_random.draw_trees`, no other random draw), shown to the H6d counter in batches of 100 with no target, so the
unguided twin of a blind sampler can be set against the sampler's own hit rate (its registered hits, s2_cal_rand).

    python -m reach.tasks.s2_revisit_rand --set S --enum_dir DIR --seed 30000000 --n 20000   -> RESULT line
"""
import argparse
import json

from reach import grammar as G
from reach import revisit as V
from reach.tasks.s2_random import draw_trees

ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True)
ap.add_argument("--enum_dir", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--n", type=int, required=True)
a = ap.parse_args()
g = G.get(a.set)
c = V.Counter(g, None, V.Lookup(g, a.enum_dir), 100, [p for p in V.PREFIXES if p <= a.n])
batch = []
for tree in draw_trees(g, a.n, a.seed):
    batch.append(tree)
    if len(batch) == 100:
        c.generation(batch)
        batch = []
if batch:
    c.generation(batch)
print("RESULT " + json.dumps(dict(set=g.name, domain=g.kind, seed=a.seed, n=a.n, rv=c.record())), flush=True)

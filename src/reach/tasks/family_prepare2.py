"""Scout targets v2 for one EML-family operator (D022): proven sizes 3..19, uniform over minimal functions,
dead-subtree filter, and no smaller function within nMSE <= 1e-6 (near-duplicate filter). 10 per level,
drawn from a pool of 40. Writes data/targets/<op>/targets_fam2.json (+ _log) -> RESULT line."""
import argparse
import json

from reach import grammar as G
from reach.family import enum_dir
from reach.family_metrics import load_values, near_duplicates
from reach.paths import ROOT
from reach.targets import generate

ap = argparse.ArgumentParser()
ap.add_argument("--grammar", required=True)
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--pool", type=int, default=40)
a = ap.parse_args()
g = G.get(a.grammar)
out = ROOT / f"data/targets/{g.name}/targets_fam2.json"
if not out.exists():
    levels = list(range(3, 20, 2))
    pool, log = generate(g, enum_dir(g.name), levels, a.pool, seed=20261014)
    vals = load_values(g, 17)
    keep, nd = [], {}
    xs = G.sample_grid(g)
    for K in levels:
        got = 0
        for t in [t for t in pool if t["K_construction"] == K]:
            if got == a.n:
                break
            f = G.evaluate(g, G.parse_expr(t["expression"], g), xs)
            d_min, cnt = near_duplicates(vals, f, t["var_target"], K)
            if cnt[6] > 0:
                nd[K] = nd.get(K, 0) + 1
                continue
            t["target_id"] = f"{g.name}_v2_K{K:02d}_{got:02d}"
            keep.append(t)
            got += 1
        log["levels"][K]["near_duplicate_rejected"] = nd.get(K, 0)
        log["levels"][K]["kept"] = got
    out.write_text(json.dumps(keep, indent=1))
    (out.parent / "targets_fam2_log.json").write_text(json.dumps(log, indent=1))
n = len(json.loads(out.read_text()))
print("RESULT " + json.dumps(dict(grammar=g.name, terminal_state="completed", targets=n)), flush=True)

"""Full EML-family study (D023): prepare one grammar.
One variable (F-*): confirmatory targets `targets_conf` (fresh seed; sizes 3..19; uniform over minimal functions;
dead-subtree and near-duplicate filters; up to 30 per size from a pool of 100).
Two variables (F2-*): minimal-construction table (data/family/minimal_<op>.json) and calculator targets calc2.json.
-> RESULT line. Idempotent: existing outputs are kept (frozen)."""
import argparse
import json

from reach import grammar as G
from reach.family import enum_dir, minimal_table
from reach.family_metrics import load_values, near_duplicates
from reach.paths import ROOT
from reach.targets import _real_info, generate

ap = argparse.ArgumentParser()
ap.add_argument("--grammar", required=True)
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--pool", type=int, default=100)
ap.add_argument("--seed", type=int, default=20261101)
a = ap.parse_args()
g = G.get(a.grammar)
outdir = ROOT / f"data/targets/{g.name}"
outdir.mkdir(parents=True, exist_ok=True)
if len(g.variables) == 1:
    out = outdir / "targets_conf.json"
    if not out.exists():
        levels = list(range(3, 20, 2))
        pool, log = generate(g, enum_dir(g.name), levels, a.pool, seed=a.seed)
        vals, xs, keep = load_values(g, 17), G.sample_grid(g), []
        for K in levels:
            got = rej = 0
            for t in [t for t in pool if t["K_construction"] == K]:
                if got == a.n:
                    break
                f = G.evaluate(g, G.parse_expr(t["expression"], g), xs)
                if near_duplicates(vals, f, t["var_target"], K)[1][6] > 0:
                    rej += 1
                    continue
                t["target_id"] = f"{g.name}_c_K{K:02d}_{got:02d}"
                keep.append(t)
                got += 1
            log["levels"][K].update(near_duplicate_rejected=rej, kept=got)
        out.write_text(json.dumps(keep, indent=1))
        (outdir / "targets_conf_log.json").write_text(json.dumps(log, indent=1))
    n = len(json.loads(out.read_text()))
else:
    mt = minimal_table(g.name)
    (ROOT / "data/family").mkdir(exist_ok=True)
    (ROOT / f"data/family/minimal_{g.name}.json").write_text(json.dumps(mt, indent=1))
    out = outdir / "calc2.json"
    if not out.exists():
        calc = []
        for fname, hit in mt["table"].items():
            if hit is None or not hit["verified"]:
                continue
            tree = G.parse_expr(hit["expr"], g)
            info, _ = _real_info(g, tree)
            if info is None:
                continue
            info.pop("vals")
            calc.append(dict(target_id=f"{g.name}_calc_{fname}", function=fname, expression=hit["expr"],
                             K_construction=hit["K"], K_proven=hit["K"], depth=G.tree_depth(tree), **info))
        out.write_text(json.dumps(calc, indent=1))
    n = len(json.loads(out.read_text()))
print("RESULT " + json.dumps(dict(grammar=g.name, terminal_state="completed", targets=n)), flush=True)

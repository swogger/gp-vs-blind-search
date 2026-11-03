"""Generate and freeze Study 2 targets for one set (tool T4) -> data/targets_s2/<set>/<name>.json (+ log, .sha256).

    python -m reach.tasks.s2_targets --grammar R-ARITH4 --name pilot --n 10 --seed 20261020 [--levels 5,7,9]

Without --levels, the pilot levels (`s2_targets.pilot_levels`) are used. Real sets load the enumerated values for the
near-duplicate filter and compute the float-noise rows (D050) for bits_sem. Prints one RESULT line."""
import argparse
import json

from reach import grammar as G
from reach import s2_targets as T
from reach.analysis.s2.phase1 import enum_dir, inflation_from
from reach.enumerate import Enumerator
from reach.paths import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--grammar", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--levels", default="", help="comma-separated sizes; default: pilot levels")
a = ap.parse_args()
g = G.get(a.grammar)
d = enum_dir(g.name)
stats = json.loads((d / "counts.json").read_text())["levels"]
if a.levels:
    levels = [int(k) for k in a.levels.split(",")]
else:
    proven, constructed = T.pilot_levels(stats)
    levels = proven + constructed
vals = noise = None
if g.kind == "real":
    e = Enumerator(g, d, log=lambda *_: None)
    e.load()
    vals, noise = e.vals, inflation_from(e)
ts, log = T.generate(g, d, levels, a.n, a.seed, vals=vals, noise_rows=noise)
p = T.write_frozen(ROOT / "data/targets_s2" / g.name, a.name, ts, log)
found = {k: v["found"] for k, v in log["levels"].items()}
print("RESULT " + json.dumps(dict(grammar=g.name, terminal_state="completed", targets=len(ts), file=str(p.relative_to(ROOT)),
                                  found=found)), flush=True)

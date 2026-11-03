"""Generate and freeze targets for one grammar: data/targets/<grammar>/targets.json (+ log)."""
import argparse
import json

from reach import grammar as G
from reach.paths import ROOT
from reach.targets import generate

ap = argparse.ArgumentParser()
ap.add_argument("--grammar", required=True)
ap.add_argument("--levels", required=True, help="comma-separated sizes K")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--seed", type=int, default=20261013)
ap.add_argument("--name", default="targets")
a = ap.parse_args()
g = G.get(a.grammar)
ts, log = generate(g, ROOT / "data/enumeration" / g.name, [int(k) for k in a.levels.split(",")], a.n, a.seed)
out = ROOT / "data/targets" / g.name
out.mkdir(parents=True, exist_ok=True)
(out / f"{a.name}.json").write_text(json.dumps(ts, indent=1))
(out / f"{a.name}_log.json").write_text(json.dumps(log, indent=1))
print(f"{g.name}: {len(ts)} targets; " + ", ".join(f"K{k}:{v['found']}" for k, v in log["levels"].items()))

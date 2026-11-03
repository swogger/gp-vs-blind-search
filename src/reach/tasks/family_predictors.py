"""Target-level searchability predictors from the enumeration alone (no GP), one operator -> RESULT line.
Writes data/family/predictors_<op><suffix>.json for the targets in data/targets/<op>/<targets>.json (K <= kmax)."""
import argparse
import json
import time

from reach import grammar as G
from reach.family_metrics import load_values, target_metrics
from reach.paths import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--grammar", required=True)
ap.add_argument("--targets", default="targets_fam")
ap.add_argument("--suffix", default="")
ap.add_argument("--kmax", type=int, default=19)
a = ap.parse_args()
g, t0 = G.get(a.grammar), time.process_time()
vals = load_values(g, a.kmax - 2)
targets = [t for t in json.loads((ROOT / f"data/targets/{g.name}/{a.targets}.json").read_text())
           if t["K_construction"] <= a.kmax]
out = []
for n, t in enumerate(targets):
    out.append(dict(target_id=t["target_id"], **target_metrics(g, G.parse_expr(t["expression"], g), t["var_target"], vals)))
    if n % 25 == 0:
        print(f"  {n + 1}/{len(targets)}, cpu {time.process_time() - t0:.0f}s", flush=True)
(ROOT / f"data/family/predictors_{g.name}{a.suffix}.json").write_text(json.dumps(out, indent=1))
print("RESULT " + json.dumps(dict(grammar=g.name, terminal_state="completed", targets=len(out))), flush=True)

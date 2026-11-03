"""Held-out extension A4 (D027): per-target predictions of the frozen corrected predictor for every held-out
target, written BEFORE any held-out search run -> data/heldout/predictions.json (checksum goes into the
OSF addendum)."""
import json
import math

from reach import grammar as G
from reach.family_metrics import target_metrics
from reach.paths import ROOT
from reach.runner.phases import HELDOUT


def target_sets(h):
    for g, tname in ((f"S-{h}", "targets_conf"), (f"S-{h}", "calc"), (f"S2-{h}", "calc2")):
        p = ROOT / f"data/targets/{g}/{tname}.json"
        if p.exists():
            yield g, tname, json.loads(p.read_text())


def main():
    pred = json.loads((ROOT / "data/heldout/predictor.json").read_text())
    b = pred["params"]
    out = {}
    for h in HELDOUT:
        for g, tname, ts in target_sets(h):
            gr = G.get(g)
            for t in ts:
                m = target_metrics(gr, G.parse_expr(t["expression"], gr), t["var_target"], {})
                x = dict(K=t["K_construction"], nvar=int(g.startswith("S2-")), calc=int(tname != "targets_conf"),
                         nb_good=m["nb_good"])
                eta = b["Intercept"] + sum(b[k] * x[k] for k in b if k != "Intercept")
                out[t["target_id"]] = dict(grammar=g, target_set=tname, **x, p=1 / (1 + math.exp(-eta)))
    (ROOT / "data/heldout/predictions.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"{len(out)} predictions; model {pred['model']}")


if __name__ == "__main__":
    main()

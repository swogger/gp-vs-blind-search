"""Power check for H2 (Stage 1 sample-size rationale; D062). Not a test of any hypothesis.

    .venv/bin/python -m reach.analysis.s2.power [--sims 20]

Simulates calibration data in which bits_sem is exactly invariant: every set of a domain follows the same E0 curve
P = logistic(a - b * bits_sem + u_target), u_target ~ N(0, SIGMA_U), with a and b from the pooled pilot fit. Each set
gets its registered 10 levels (data/registration/levels.json) x 30 targets x 5 seeds, with the bits_sem of each level
from the set's enumeration. The invariance test (`invariance.invariance_test`) is run on every simulated data set;
power is the share with upper95 <= delta. The same is done for a shifted set (one set's curve moved by `shift` bits)
to show the test can fail.
"""
import argparse
import json
import math

import numpy as np

from reach.analysis.s2.invariance import invariance_test

SIGMA_U = 1.0
TARGETS, SEEDS = 30, 5


def simulate(set_bits, a, b, rng, shift=None):
    """set_bits: {set: [bits_sem per level]}; shift: (set, bits) to move one set's curve."""
    recs = []
    for g, bits in set_bits.items():
        off = shift[1] if shift and shift[0] == g else 0.0
        for li, s in enumerate(bits):
            for t in range(TARGETS):
                u = rng.normal(0, SIGMA_U)
                p = 1 / (1 + math.exp(-(a - b * (s - off) + u)))
                tid = f"{g}_L{li}_{t}"
                recs += [dict(set=g, target_id=tid, units=dict(bits_sem=float(s)), y=int(rng.random() < p))
                         for _ in range(SEEDS)]
    return recs


def power(set_bits, a, b, delta, sims=20, boot=200, seed=20261020, shift=None):
    rng = np.random.default_rng(seed)
    ups = []
    for i in range(sims):
        r = invariance_test(simulate(set_bits, a, b, rng, shift), ["bits_sem"], delta, boot=boot, seed=seed + i)
        ups.append(r["bits_sem"]["upper95"])
    ups = np.array(ups)
    return dict(power=float(np.mean(ups <= delta)), upper95_median=float(np.median(ups)), sims=sims, delta=delta)


def main():
    from reach import grammar as G
    from reach.analysis.k50 import logit
    from reach.analysis.s2.load import read_phase, tidy
    from reach.analysis.s2.phase1 import enum_dir
    from reach.paths import ROOT
    from reach.runner.phases import CAL_SETS
    from reach.units import SizeUnits
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20)
    a = ap.parse_args()
    lv = json.loads((ROOT / "data/registration/levels.json").read_text())
    props = json.loads((ROOT / "data/analysis/pilot.json").read_text())["proposals"]
    rows = read_phase("s2_pilot_reach_a") + read_phase("s2_pilot_reach_b")
    out = {}
    for dom in ("real", "bool"):
        sets = [g for g in CAL_SETS if G.get(g).kind == dom]
        e0 = [r for r in tidy([r for r in rows if r.get("set") in sets], (1,)) if "bits_sem" in r["units"]]
        b0, b1 = logit([r["units"]["bits_sem"] for r in e0], [r["y"] for r in e0])
        set_bits = {}
        for g in sets:
            su = SizeUnits(G.get(g), json.loads((enum_dir(g) / "counts.json").read_text())["levels"])
            set_bits[g] = [su.all(K)["bits_sem"] for K in sorted(lv[g]["main"] + lv[g]["budget"])]
        delta = props[dom]["delta_H2"]
        out[dom] = dict(a=float(b0), b=float(-b1), invariant=power(set_bits, b0, -b1, delta, a.sims),
                        shifted=power(set_bits, b0, -b1, delta, a.sims, shift=(sets[0], 2.0)))
        print(dom, json.dumps(out[dom]))
    (ROOT / "data/analysis/power.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

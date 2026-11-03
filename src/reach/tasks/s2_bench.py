"""Benchmark targets for one held-out set (T3; D069) -> data/targets_s2/<set>/bench.json (+ log, .sha256).

    python -m reach.tasks.s2_bench --grammar R-ARITH

Deterministic (no seed): every benchmark of the set's domain whose minimum size the set's enumeration proves. Real
sets load the enumerated values for the near-duplicate filter and compute the D050 float-noise rows for bits_sem, as
the target generator does (recorded in the log; guardrail row 126). Not a search; the files are frozen and
checksummed at the Stage 2 registration.

    python -m reach.tasks.s2_bench --grammar R-ARITH --regenerate

regenerates an existing file through the queue (D091; guardrail row 129; phase s2_targets_bench2): the earlier entries
must be reproduced byte for byte and only appended benchmarks may differ (`s2_benchmarks.regenerate`); refused once
Stage 2 is registered. The RESULT records the old and new SHA-256 of bench.json."""
import argparse
import json

from reach import grammar as G
from reach import s2_benchmarks as B
from reach.analysis.s2.phase1 import enum_dir, inflation_from
from reach.paths import ROOT
from reach.s2_targets import write_frozen
from reach.targets import load_structure

ap = argparse.ArgumentParser()
ap.add_argument("--grammar", required=True)
ap.add_argument("--regenerate", action="store_true")
a = ap.parse_args()
g = G.get(a.grammar)
d = enum_dir(g.name)
e = load_structure(g, d)
vals = noise = None
if g.kind == "real":
    from reach.enumerate import Enumerator
    en = Enumerator(g, d, log=lambda *_: None)
    en.load()
    vals, noise = en.vals, inflation_from(en)
ts, log = B.bench_targets(g, e, vals=vals, noise_rows=noise)
logd = dict(grammar=g.name, benchmarks=log, noise_rows=noise)
out = ROOT / "data/targets_s2" / g.name
cmp = B.regenerate(out, ts, logd) if a.regenerate else {}
p = out / "bench.json" if a.regenerate else write_frozen(out, "bench", ts, logd)
print("RESULT " + json.dumps(B.result_record(g.name, ts, log, p, cmp, ROOT)), flush=True)

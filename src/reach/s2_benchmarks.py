"""Benchmark targets for H7c (docs/design/targets.md, set T3; D069).

Real (one variable on the grid, constant-free): Nguyen N1-N8 and Koza K2, K3 (K1 equals N2 and is dropped); added by
D091 (appended, so the earlier entries keep their order): SRBench's one one-variable ground-truth equation (Feynman
I.6.2a; the other 115 Feynman and all 14 Strogatz equations have 2 to 9 variables) and EML-GP's ten one-variable
targets, both read from data/reference with their source commits and checksums (`candidates`). Boolean
(5 inputs, x1..x5): classic single-output functions: odd parity, majority, the three outputs of a two-bit adder with
carry-in (a = x1x2, b = x3x4, c = x5), a two-bit comparator (x1x2 > x3x4) and a multiplexer (x1 selects x3 or x2).
They are not from Kalkreuth et al.'s suite (FOGA 2023), which holds multi-output circuits and high-order parity
(checked against the full text, D092). White et al. (2013) blacklist low-order polynomials and simple one-output
Boolean functions as a sole benchmark set; here benchmarks are descriptive reference points only (D090, D092).
A benchmark enters a held-out set only where that set's enumeration proves its minimum size (its key is found within
the ceiling): every benchmark size is proven, none constructed. Each cell is run with 30 benchmark seeds.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from reach import grammar as G
from reach.s2_targets import DEPTH_LIMIT, boolean_properties, real_filters
from reach.tasks.s2_pool import _find

REAL = {
    "N1": lambda x: x ** 3 + x ** 2 + x,
    "N2": lambda x: x ** 4 + x ** 3 + x ** 2 + x,
    "N3": lambda x: x ** 5 + x ** 4 + x ** 3 + x ** 2 + x,
    "N4": lambda x: x ** 6 + x ** 5 + x ** 4 + x ** 3 + x ** 2 + x,
    "N5": lambda x: np.sin(x ** 2) * np.cos(x) - 1,
    "N6": lambda x: np.sin(x) + np.sin(x + x ** 2),
    "N7": lambda x: np.log(x + 1) + np.log(x ** 2 + 1),
    "N8": lambda x: np.sqrt(x),
    "K2": lambda x: x ** 5 - 2 * x ** 3 + x,
    "K3": lambda x: x ** 6 - 2 * x ** 4 + x ** 2,
    # D091: SRBench (PMLB metadata, data/reference/srbench_groundtruth) and EML-GP (data/reference/eml_gp_targets)
    "feynman_I_6_2a": lambda x: np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi),
    "eml_1_1": lambda x: np.exp(x),
    "eml_1_2": lambda x: np.log(x),
    "eml_1_4": lambda x: np.exp(x) * np.exp(-x ** 2),
    "eml_2_1": lambda x: np.sin(x) + np.cos(x),
    "eml_3_1": lambda x: np.sinh(x),
    "eml_3_2": lambda x: np.arctan(x),
    "eml_3_3": lambda x: x ** x,
    "eml_3_4": lambda x: 1 / (1 + np.exp(-x)),
    "eml_3_5": lambda x: x ** 3 - 2 * x ** 2 + x,
    "eml_4_1": lambda x: np.exp(np.sin(x)) + np.log(x ** 2 + 1),
}


def eml_name(key):
    """EML-GP's matrix key ('3.5') -> our benchmark name ('eml_3_5')."""
    return "eml_" + key.replace(".", "_")


def candidates():
    """Every SRBench ground-truth problem and every EML-GP target with its decision (D091): a one-variable item is a
    benchmark (D069's rule then decides per set, logged in bench_log.json); any other is excluded with its reason."""
    import csv
    import json
    from reach.paths import ROOT
    ref = ROOT / "data/reference"
    out = []
    for r in csv.DictReader(open(ref / "srbench_groundtruth/equations.csv")):
        n = int(r["n_variables"])
        out.append(dict(source="SRBench", item=r["dataset"], n_variables=n,
                        benchmark=r["dataset"] if n == 1 else None,
                        excluded=None if n == 1 else f"{n} variables (the grid has one)"))
    for t in json.loads((ref / "eml_gp_targets/source.json").read_text())["targets"]:
        n = len(t["variables"])
        out.append(dict(source="EML-GP", item=f"EML-GP {t['key']}", n_variables=n,
                        benchmark=eml_name(t["key"]) if n == 1 else None,
                        excluded=None if n == 1 else f"{n} variables (the grid has one)"))
    return out


def _maj(a, b, c):
    return (a & b) | (a & c) | (b & c)


BOOL = {
    "parity5": lambda x: x[0] ^ x[1] ^ x[2] ^ x[3] ^ x[4],
    "majority5": lambda x: int(sum(x) >= 3),
    "add2_s0": lambda x: x[1] ^ x[3] ^ x[4],
    "add2_s1": lambda x: x[0] ^ x[2] ^ _maj(x[1], x[3], x[4]),
    "add2_cout": lambda x: _maj(x[0], x[2], _maj(x[1], x[3], x[4])),
    "cmp2_gt": lambda x: int(2 * x[0] + x[1] > 2 * x[2] + x[3]),
    "mux": lambda x: x[2] if x[0] else x[1],
}


def truth_table(f, n):
    """Packed as the engine packs it: row r has x_i = bit (n-1-i) of r, and bit r of the result is f(row r)."""
    return sum(f([(r >> (n - 1 - i)) & 1 for i in range(n)]) << r for r in range(1 << n))


def bench_targets(g, e, specs=None, vals=None, noise_rows=None):
    """Targets for every benchmark whose function the set's enumeration proves (within the ceiling, depth <= 10).
    e: a loaded structure (`targets.load_structure`); returns (targets, log). noise_rows: the set's D050 float-noise
    rows (`phase1.inflation_from`), required for a real set, as the target generator uses them for bits_sem; a Boolean
    set has none (guardrail row 126)."""
    from reach.units import SizeUnits
    if g.kind == "real" and noise_rows is None:
        raise ValueError(f"{g.name}: real benchmark sizes need the set's D050 noise rows (phase1.inflation_from)")
    if g.kind != "real" and noise_rows is not None:
        raise ValueError(f"{g.name}: D050 noise rows apply to real sets only")
    su = SizeUnits(g, e.stats, noise_rows=noise_rows)
    specs = specs or (REAL if g.kind == "real" else BOOL)
    out, log = [], {}
    for name, f in specs.items():
        if g.kind == "real":
            v = np.asarray(f(G.sample_grid(g).real), complex)
            key = int(e._key(v[None, :])[0])
        else:
            key = int(e._key(np.array([truth_table(f, g.n_inputs)], dtype=np.uint64))[0])
        hit = _find(e, key)
        if hit is None:
            log[name] = "not within the ceiling"
            continue
        K, i = hit
        if e.depth[K][i] > DEPTH_LIMIT:
            log[name] = "minimal tree deeper than 10"
            continue
        tree = G.parse_expr(e.expr(K, i), g)
        rec = dict(target_id=f"{g.name}_bench_{name}", benchmark=name, expression=G.expr_str(tree), K_construction=K,
                   K_proven=K, constructed=False, depth=int(e.depth[K][i]), units=su.all(K))
        if g.kind == "real":
            fields, why = real_filters(g, tree, K, vals)        # vals: enumerated values by level (near duplicates)
            if fields is None:
                log[name] = f"filtered: {why}"
                continue
            rec.update(fields)
        else:
            rec["properties"] = boolean_properties(G.evaluate(g, tree), g.n_inputs)
        out.append(rec)
        log[name] = f"proven K={K}"
    return out, log


def _dump(targets):
    return json.dumps(targets, indent=1, default=float)          # as s2_targets.write_frozen writes them


def regenerate(out_dir, targets, log):
    """Regenerate bench.json through the queue (D091; guardrail row 129), before the Stage 2 registration only.
    An identical result leaves every file untouched; otherwise the earlier entries must be reproduced byte for byte
    (only appended benchmarks may differ), the D050 noise rows and the earlier benchmarks' log lines must be equal,
    and the files are rewritten with a new .sha256. Anything else is refused and nothing is written. Returns the
    comparison with the old and new SHA-256 (recorded in the run's RESULT)."""
    from reach import s2_lock
    if s2_lock.TASKS2_FILE.exists() or (s2_lock.STAGE2_FILE.exists() and s2_lock.STAGE2_FILE.read_text().strip()):
        raise PermissionError("benchmark files are frozen once Stage 2 is registered")
    out = Path(out_dir)
    p, lp, sp = out / "bench.json", out / "bench_log.json", out / "bench.sha256"
    new, new_log = _dump(targets).encode(), json.dumps(log, indent=1).encode()
    old = p.read_bytes() if p.exists() else None
    res = dict(old_bench_sha256=hashlib.sha256(old).hexdigest() if old is not None else None,
               bench_sha256=hashlib.sha256(new).hexdigest(), added=[])
    if old is not None:
        prev, prev_log = json.loads(old), json.loads(lp.read_text())
        n = len(prev)
        if len(targets) < n or _dump(targets[:n]).encode() != old:
            raise ValueError(f"{out.name}: the earlier entries are not reproduced byte for byte")
        if prev_log.get("noise_rows") != log.get("noise_rows"):
            raise ValueError(f"{out.name}: the D050 noise rows differ")
        changed = [k for k, v in prev_log.get("benchmarks", {}).items() if log.get("benchmarks", {}).get(k) != v]
        if changed:
            raise ValueError(f"{out.name}: the log of earlier benchmarks differs: {changed}")
        res["added"] = [t["benchmark"] for t in targets[n:]]
    res["reproduced"] = old == new and lp.exists() and lp.read_bytes() == new_log
    res["log_reproduced"] = lp.exists() and lp.read_bytes() == new_log
    if not res["reproduced"]:
        out.mkdir(parents=True, exist_ok=True)
        p.write_bytes(new)
        lp.write_bytes(new_log)
        sp.write_text(f"{res['bench_sha256']}  bench.json\n")
    return res


BENCH_PHASES = ("s2_targets_bench", "s2_targets_bench2")


def bench_record_problems(root):
    """Committed bench files without a completed queue record of their content (guardrail row 129): for each
    data/targets_s2/<set>/bench.json, some completed row of a bench phase for that set must carry the file's SHA-256
    (`bench_sha256`), and the .sha256 sidecar must agree."""
    from reach.runner.queue import read_results
    root = Path(root)
    shas = {}
    for ph in BENCH_PHASES:
        for r in read_results(root / f"data/runs/{ph}/results.jsonl").values():
            if r.get("terminal_state") == "completed" and r.get("queue_status") == "ok" and r.get("bench_sha256"):
                shas.setdefault(r.get("grammar", r["key"]), set()).add(r["bench_sha256"])
    problems = []
    for p in sorted((root / "data/targets_s2").glob("*/bench.json")):
        g, sha = p.parent.name, hashlib.sha256(p.read_bytes()).hexdigest()
        side = (p.parent / "bench.sha256")
        if not side.exists() or side.read_text().split()[0] != sha:
            problems.append(f"{g}: bench.sha256 does not match bench.json")
        if sha not in shas.get(g, set()):
            problems.append(f"{g}: no completed queue record of bench.json {sha[:12]}")
    return problems


def result_record(grammar, targets, log, path, cmp, root):
    """The task's RESULT: the comparison from `regenerate` (empty for a first generation) and the written file's
    SHA-256, each key once (guardrail row 130)."""
    rec = dict(cmp)
    rec.update(grammar=grammar, terminal_state="completed", targets=len(targets), log=log,
               file=str(Path(path).relative_to(root)), bench_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    if cmp and cmp.get("bench_sha256") != rec["bench_sha256"]:
        raise RuntimeError(f"{grammar}: bench.json changed after it was written")
    return rec

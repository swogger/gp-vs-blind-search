"""Shared-function pool for H3 (tool T12; docs/design/targets.md, set T2; REQ-MEA-15).

Join the enumeration keys of the calibration sets of one domain (real keys are comparable: same grid, same key function;
Boolean keys are truth tables). A function enters the pool if its minimum size is proven in at least `min_sets` sets
(Boolean 4, real 3). Functions are sampled stratified by their size spread across sets (max - min minimum size), so that
H3 sees functions whose shortest construction differs a lot between sets. Each pool function gets one pool_id and,
in every set where it is proven, that set's shallowest minimal tree (C5; depth <= 10 required). Real per-tree filters
(complex, floating point, dead subtree; near duplicates when values are given) are applied per set; a function that
drops below `min_sets` after filtering is removed. Held-out sets are never included (C1).

    python -m reach.tasks.s2_pool --domain bool --n 40 --seed 20261023     -> data/targets_s2/<set>/pool.json (+ sha256)
"""
import argparse
import json
import random
from collections import defaultdict

import numpy as np

from reach import grammar as G
from reach.enumerate import read_level
from reach.s2_targets import DEPTH_LIMIT, boolean_properties, real_filters, write_frozen
from reach.targets import load_structure


def _levels(d):
    stats = json.loads((d / "counts.json").read_text())["levels"]
    return {st["K"]: read_level(d / f"level_{st['K']:03d}.npz")[0] for st in stats if st["U"] > 0}


def join_sizes(dirs, min_sets=1, candidates=None):
    """{key: {set: minimum size}} for functions in at least `min_sets` sets. Without `candidates`: a vectorised join of
    all keys (real sets); with `candidates` (uint64 keys): look only those up in every set (Boolean sets at n = 5)."""
    out = defaultdict(dict)
    if candidates is not None:
        cand = np.unique(np.asarray(candidates, dtype=np.uint64))
        for s, d in dirs.items():
            for K, keys in _levels(d).items():
                for k in cand[np.isin(cand, keys)].tolist():
                    out[k][s] = K
        return {k: v for k, v in out.items() if len(v) >= min_sets}
    ks, ss, ls = [], [], []
    names = sorted(dirs)
    for si, s in enumerate(names):
        for K, keys in _levels(dirs[s]).items():
            ks.append(keys)
            ss.append(np.full(len(keys), si, np.int16))
            ls.append(np.full(len(keys), K, np.int16))
    k, s_, l_ = np.concatenate(ks), np.concatenate(ss), np.concatenate(ls)
    order = np.argsort(k, kind="stable")
    k, s_, l_ = k[order], s_[order], l_[order]
    uk, start, cnt = np.unique(k, return_index=True, return_counts=True)
    for key, a, c in zip(uk[cnt >= min_sets].tolist(), start[cnt >= min_sets], cnt[cnt >= min_sets]):
        for j in range(a, a + c):
            out[key][names[s_[j]]] = int(l_[j])
    return dict(out)


def _spread_bin(spread):
    return 0 if spread == 0 else 1 if spread <= 2 else 2 if spread <= 4 else 3 if spread <= 8 else 4


def _anchor_candidates(d, n_cand, rng):
    """Keys sampled from the anchor, spread evenly over its non-trivial levels (Boolean path)."""
    lv = {K: keys for K, keys in _levels(d).items() if K >= 2}
    per = max(1, n_cand // len(lv))
    out = []
    for K in sorted(lv):
        keys = lv[K]
        idx = rng.sample(range(len(keys)), min(per, len(keys)))
        out += [int(keys[i]) for i in idx]
    return out


def _find(e, key):
    for K in sorted(e.keys):
        hit = np.nonzero(e.keys[K] == np.uint64(key))[0]
        if len(hit):
            return K, int(hit[0])
    return None


N_BINS = 5


def _select(bins, evaluate, n):
    """Equal quotas per spread bin, binned AFTER the per-set filters (guardrail row 51: filters remove most real
    functions with a spread, so binning before them filled the pool with spread-0 functions, which carry no
    within-function information). bins: {pre-filter bin: keys in random order}; evaluate(key) -> (recs, spread) or
    None. Filters only drop sets, so a function's spread can only shrink: a pre-filter bin is skipped once every bin up
    to it is full. A shortfall is filled with the passing functions left over, largest spread first."""
    quota = {b: n // N_BINS + (1 if b >= N_BINS - n % N_BINS else 0) for b in range(N_BINS)}
    chosen, spare = {b: [] for b in range(N_BINS)}, []
    queues = {b: list(v) for b, v in bins.items()}
    full = lambda b: all(len(chosen[p]) >= quota[p] for p in range(b + 1))         # noqa: E731
    while sum(len(c) for c in chosen.values()) < n and any(queues[b] and not full(b) for b in queues):
        for b in sorted(queues):
            if not queues[b] or full(b):
                continue
            key = queues[b].pop(0)
            r = evaluate(key)
            if r is None:
                continue
            pb = _spread_bin(r[1])
            (chosen[pb] if len(chosen[pb]) < quota[pb] else spare).append((key, r[0], r[1]))
    need = n - sum(len(c) for c in chosen.values())
    for b in sorted(queues, reverse=True):                    # shortfall: draw on, largest spread bins first
        while need > len(spare) and queues[b]:
            key = queues[b].pop(0)
            r = evaluate(key)
            if r is not None:
                spare.append((key, r[0], r[1]))
    out = [x for b in range(N_BINS) for x in chosen[b]]
    out += sorted(spare, key=lambda x: -x[2])[:max(0, n - len(out))]
    return out[:n]


def build_pool(dirs, min_sets, n, seed, vals=None, anchor=None, n_cand=4000):
    rng = random.Random(seed)
    if anchor is not None:
        sizes = join_sizes(dirs, min_sets, candidates=_anchor_candidates(dirs[anchor], n_cand, rng))
    else:
        sizes = join_sizes(dirs, min_sets)
    sizes = {k: v for k, v in sizes.items() if min(v.values()) >= 2}
    bins = defaultdict(list)
    for k, v in sizes.items():
        bins[_spread_bin(max(v.values()) - min(v.values()))].append(k)
    for b in bins.values():
        b.sort()
        rng.shuffle(b)
    structs = {s: load_structure(G.get(s), d) for s, d in dirs.items()}

    def evaluate(key):
        recs = {}
        for s in sizes[key]:
            g, e = G.get(s), structs[s]
            K_, i = _find(e, key)
            if e.depth[K_][i] > DEPTH_LIMIT:
                continue
            tree = G.parse_expr(e.expr(K_, i), g)
            rec = dict(expression=G.expr_str(tree), K_construction=K_, K_proven=K_, constructed=False,
                       depth=int(e.depth[K_][i]))
            if g.kind == "real":
                fields, why = real_filters(g, tree, K_, (vals or {}).get(s))
                if fields is None:
                    continue
                rec.update(fields)
            else:
                rec["properties"] = boolean_properties(G.evaluate(g, tree), g.n_inputs)
            recs[s] = rec
        if len(recs) < min_sets:
            return None
        return recs, max(r["K_proven"] for r in recs.values()) - min(r["K_proven"] for r in recs.values())

    pool = defaultdict(list)
    for j, (key, recs, spread) in enumerate(_select(bins, evaluate, n)):
        pid = f"pool{j:03d}"
        for s, r in recs.items():
            r.update(pool_id=pid, target_id=f"{s}_{pid}", size_spread=spread, n_sets=len(recs))
            pool[s].append(r)
    return dict(pool)


def pool_sets(domain):
    """The full calibration sets of a domain: exactly the sets that s2_cal_pool runs (guardrail row 49)."""
    from reach.runner.phases import FULL_SETS
    return [g for g in FULL_SETS if G.get(g).kind == domain]


def main():
    from reach.analysis.s2.phase1 import enum_dir, inflation_from
    from reach.paths import ROOT
    from reach.units import SizeUnits
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["real", "bool"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    sets = pool_sets(a.domain)
    dirs = {g: enum_dir(g) for g in sets}
    vals, noise = None, {}
    if a.domain == "real":                                     # near duplicates need each set's values
        from reach.enumerate import Enumerator
        vals = {}
        for g in sets:
            e = Enumerator(G.get(g), dirs[g], log=lambda *_: None)
            e.load()
            vals[g], noise[g] = e.vals, inflation_from(e)     # D050 noise rows: bits_sem as in calibration targets
    pool = build_pool(dirs, 3 if a.domain == "real" else 4, a.n, a.seed, vals,
                      anchor="BOOL-NAND-n5" if a.domain == "bool" else None)
    for g, ts in pool.items():
        su = SizeUnits(G.get(g), json.loads((dirs[g] / "counts.json").read_text())["levels"], noise_rows=noise.get(g))
        for t in ts:
            t["units"] = su.all(t["K_proven"])
        write_frozen(ROOT / "data/targets_s2" / g, "pool", ts, dict(domain=a.domain, seed=a.seed, n=a.n, sets=sets))
    print("RESULT " + json.dumps(dict(domain=a.domain, terminal_state="completed", functions=len({t["pool_id"] for ts in
                                      pool.values() for t in ts}), sets={g: len(ts) for g, ts in pool.items()})), flush=True)


if __name__ == "__main__":
    main()

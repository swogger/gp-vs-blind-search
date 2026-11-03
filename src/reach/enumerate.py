"""Bottom-up enumeration of distinct functions by size (M4; decisions D009, D012; Study 2 v2: D048 C5, D049).

Level K holds every function whose smallest tree has exactly K nodes. A level is built from
smaller levels only (unary op on level K-1; binary op on levels i, j with i + j = K - 1), and
a candidate is kept only if its semantic key has not been seen at any smaller size, so the
first size at which a function appears is its minimal size (on the sample, for real grammars).

Real grammars: values on EML-GP's 100-point grid on (2, 3), rounded to 9 decimals (absolute,
as EML-GP's func_hash), -0.0 and NaN canonicalised, then hashed to 64 bits. Boolean grammars:
the key is the packed truth table itself (exact).

Per level the output records N_G(K) (syntactic), C(K) (candidates evaluated: the cost of
realisable bottom-up enumeration, D009), U_G(K) (new distinct functions) and cumulative counts.
Each level is checkpointed (keys plus parent pointers; values are rebuilt on resume), so an
interrupted or CPU-limited run resumes at the next level.

Study 2 (v2) additions:
- depth: for every function, the smallest depth among its minimal trees. Every subtree of a minimal tree is minimal
  for its own function, so depth(f) = min over the candidates that first produce f of 1 + max(children's depths).
  The stored representative is a minimal tree of that depth (C5).
- compact, atomic level files (`read_level` also reads the paper-1 format); counts.json written atomically.
- clean stops before a level that would exceed the CPU budget, the memory cap or the free-disk floor
  (`stop_reason`: cpu, memory, disk); the checkpoint stays valid (REQ-MEA-18, REQ-RUN-16).
- Boolean constants enumerated with their own value (constant 0 is the all-zeros table; review F14).
"""
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from reach import grammar as G

CHUNK = 200_000                 # candidates evaluated at once (real: 100 complex values each)
DEPTH_LIMIT = 10                # tree GP's symmetric depth limit (reach.engines.treegp.MAX_DEPTH)
BOOL_BYTES = 45                 # memory per Boolean function: value, key, parents (int32 x 5), depth, known key
BOOL_BYTES_KEYS_ONLY = 37       # the same without the stored value (last level)
_MIX = np.uint64(0x9E3779B97F4A7C15)


def _coeffs(n):
    return np.random.default_rng(20261012).integers(1, 2 ** 63, size=n, dtype=np.uint64) | np.uint64(1)


def real_keys(v, coeffs, precision=G.HASH_PRECISION):
    """64-bit semantic key per row of a complex (n, P) array (values rounded to `precision` decimals, absolute, as in
    paper 1). Above about 1e6 in magnitude, 9 absolute decimals are finer than double precision, so float noise can
    split one function into several keys; the size of that effect on U is measured in Phase 1 (REQ-MEA-21).
    A per-function scale is not a valid alternative: it makes c*g and g share a key (tried and rejected 2026-10-19)."""
    with np.errstate(all="ignore"):
        x = np.concatenate([np.round(v.real, precision), np.round(v.imag, precision)], axis=1) + 0.0
    x[np.isnan(x)] = np.nan
    u = x.view(np.uint64)
    u = (u ^ (u >> np.uint64(31))) * _MIX
    u ^= u >> np.uint64(29)
    return (u * coeffs).sum(axis=1, dtype=np.uint64)


def read_level(path):
    """(keys uint64, parents int32 (n, 5) = op, i, a, j, b, depth uint8 or None) from a level file of either
    format: v2 (compact fields) or paper 1's (int64 `parents`, no depth)."""
    with np.load(path) as z:
        keys = z["keys"].astype(np.uint64)
        if "parents" in z.files:
            return keys, z["parents"].astype(np.int32), None
        par = np.stack([z["op"], z["i"], z["a"], z["j"], z["b"]], axis=1).astype(np.int32)
        return keys, par, z["depth"].astype(np.uint8)


def _atomic_write_text(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class Enumerator:
    def __init__(self, g, out_dir, log=print, precision=G.HASH_PRECISION):
        self.g, self.out, self.log = g, Path(out_dir), log
        self.out.mkdir(parents=True, exist_ok=True)
        self.real = g.kind == "real"
        self.precision = precision
        self.xs = G.sample_grid(g) if self.real else None
        self.coeffs = _coeffs(2 * G.N_POINTS) if self.real else None
        self.mask = np.uint64(g.mask) if not self.real else None
        self.unary = [o for o in g.ops if o.arity == 1]
        self.binary = [o for o in g.ops if o.arity == 2]
        self.opi = {o.name: i for i, o in enumerate(g.ops)}
        self.vals, self.keys, self.parents, self.depth = {}, {}, {}, {}
        self.known = np.empty(0, dtype=np.uint64)          # sorted keys of all levels so far
        self.stats = []
        self.stop_reason = None
        self.syn = G.syntactic_counts(g, 200)

    # ------------------------------------------------------------ values
    def _apply(self, op, a, b=None):
        with np.errstate(all="ignore"):
            if self.real:
                return op.fn(a) if b is None else op.fn(a, b)
            return op.fn(self.mask, a) if b is None else op.fn(self.mask, a, b)

    def _key(self, v):
        return real_keys(v, self.coeffs, self.precision) if self.real else v.astype(np.uint64)

    def _leaves(self):
        names = list(self.g.variables) + list(self.g.constants)
        if self.real:
            xs = self.xs if isinstance(self.xs, dict) else {self.g.variables[0]: self.xs}
            shape = next(iter(xs.values())).shape
            leaves = [xs[v] for v in self.g.variables] + [np.full(shape, complex(c)) for c in self.g.constants]
            return np.stack(leaves), [("leaf", v) for v in names]
        vm = G.var_masks(self.g.n_inputs)
        vals = [vm[v] for v in self.g.variables] + [self.g.mask if c else 0 for c in self.g.constants]
        return np.array(vals, dtype=np.uint64), [("leaf", v) for v in names]

    # ------------------------------------------------------------ one level
    def build_level(self, k, progress_every=30.0, store_vals=True):
        """store_vals=False keeps keys and parents only: for the last level, which nothing builds on."""
        t0 = time.process_time()
        new_vals, new_keys, new_par, new_ok, new_dep = [], [], [], [], []
        n_cand = 0
        jobs = []                                            # (op, i, j): j = None for unary
        for op in self.unary:
            if k - 1 in self.vals:
                jobs.append((op, k - 1, None))
        for i in range(1, k - 1):
            j = k - 1 - i
            if i in self.vals and j in self.vals:
                for op in self.binary:
                    jobs.append((op, i, j))
        total = sum(len(self.vals[i]) * (1 if j is None else len(self.vals[j])) for _, i, j in jobs)
        last = time.time()
        for op, i, j in jobs:
            A = self.vals[i]
            if j is None:
                blocks = [(s, None) for s in range(0, len(A), CHUNK)]
            else:
                B = self.vals[j]
                cols = max(1, min(len(B), CHUNK))              # chunk both operands
                rows = max(1, CHUNK // max(cols, 1))
                blocks = [(s, c) for s in range(0, len(A), rows) for c in range(0, len(B), cols)]
            for s, c in blocks:
                if j is None:
                    a_idx = np.arange(s, min(s + CHUNK, len(A)))
                    v = self._apply(op, A[a_idx])
                    b_idx = np.full(len(a_idx), -1)
                    d = 1 + self.depth[i][a_idx]
                else:
                    a_rng = np.arange(s, min(s + rows, len(A)))
                    b_rng = np.arange(c, min(c + cols, len(B)))
                    if self.real:
                        v = self._apply(op, A[a_rng][:, None, :], B[b_rng][None, :, :]).reshape(-1, A.shape[1])
                    else:
                        v = self._apply(op, A[a_rng][:, None], B[b_rng][None, :]).reshape(-1)
                    a_idx = np.repeat(a_rng, len(b_rng))
                    b_idx = np.tile(b_rng, len(a_rng))
                    d = 1 + np.maximum(self.depth[i][a_idx], self.depth[j][b_idx])
                n_cand += len(a_idx)
                keys = self._key(v)
                order = np.lexsort((d, keys))                  # by key, then depth; stable: then position
                ks = keys[order]
                head = np.r_[True, ks[1:] != ks[:-1]] if len(ks) else np.zeros(0, bool)
                rep, keys_u = order[head], ks[head]            # per key: shallowest, earliest candidate
                pos = np.searchsorted(self.known, keys_u)
                pos[pos >= len(self.known)] = 0
                fresh = (self.known[pos] != keys_u) if len(self.known) else np.ones(len(keys_u), bool)
                sel = rep[fresh]
                if len(sel):
                    if store_vals:
                        new_vals.append(v[sel])
                    new_ok.append(self._viable(v[sel]))
                    new_keys.append(keys[sel])
                    new_dep.append(d[sel].astype(np.uint8))
                    new_par.append(np.stack([np.full(len(sel), self.opi[op.name]), np.full(len(sel), i),
                                             a_idx[sel], np.full(len(sel), -1 if j is None else j),
                                             b_idx[sel]], axis=1))
                if time.time() - last > progress_every:
                    el = time.process_time() - t0
                    frac = n_cand / max(total, 1)
                    self.log(f"  K={k}: {100 * frac:5.1f}% of {total:,} candidates, "
                             f"cpu {el:,.0f}s, level ETA {el / frac - el if frac else 0:,.0f}s")
                    last = time.time()
        if new_keys:
            keys, dep = np.concatenate(new_keys), np.concatenate(new_dep)
            order = np.lexsort((dep, keys))                    # across chunks: shallowest, then earliest
            ks = keys[order]
            head = np.r_[True, ks[1:] != ks[:-1]]
            rep = order[head]
            first = np.minimum.reduceat(order, np.flatnonzero(head))   # first appearance of each key
            pick = rep[np.argsort(first, kind="stable")]       # level order as in v1: by first appearance
            vals = np.concatenate(new_vals)[pick] if store_vals else None
            ok = np.concatenate(new_ok)[pick]
            par = np.concatenate(new_par)[pick].astype(np.int32)
            keys, dep = keys[pick], dep[pick]
        else:
            vals = np.empty((0, G.N_POINTS), np.complex128) if self.real else np.empty(0, np.uint64)
            keys, par = np.empty(0, np.uint64), np.empty((0, 5), np.int32)
            ok, dep = np.empty(0, bool), np.empty(0, np.uint8)
        self._add_level(k, vals, keys, par, dep)
        self.stats.append(self._level_stats(k, n_cand, keys, ok, dep, time.process_time() - t0, store_vals))
        self._save_level(k)

    def _n_candidates(self, k):
        n = len(self.unary) * len(self.vals.get(k - 1, ()))
        return n + len(self.binary) * sum(len(self.vals.get(i, ())) * len(self.vals.get(k - 1 - i, ()))
                                          for i in range(1, k - 1))

    def _add_level(self, k, vals, keys, par, depth):
        if vals is not None:
            self.vals[k] = vals
        self.keys[k], self.parents[k], self.depth[k] = keys, par, depth
        self.known = np.union1d(self.known, keys)

    def _viable(self, v):
        """Real: finite on the whole grid and non-constant. Bool: every function."""
        if not self.real:
            return np.ones(len(v), bool)
        with np.errstate(all="ignore"):
            return np.isfinite(v).all(axis=1) & (np.abs(v - v.mean(axis=1, keepdims=True)).max(axis=1) > 1e-6)

    def _level_stats(self, k, n_cand, keys, ok, depth, cpu, stored=True):
        cum = sum(len(self.keys[i]) for i in self.keys if i <= k)
        return dict(K=k, N_syntactic=self.syn[k], candidates=int(n_cand), U=int(len(keys)), U_cum=int(cum),
                    U_finite_nonconst=int(ok.sum()), U_depth_ok=int((depth <= DEPTH_LIMIT).sum()),
                    max_min_depth=int(depth.max()) if len(depth) else 0, cpu_s=round(cpu, 2), vals_stored=stored)

    # ------------------------------------------------------------ persistence
    def _save_level(self, k):
        keys = self.keys[k]
        if not self.real and int(self.g.mask) < 2 ** 32:
            keys = keys.astype(np.uint32)                     # n <= 5: truth tables fit in 32 bits
        p = self.parents[k]
        path = self.out / f"level_{k:03d}.npz"
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, keys=keys, op=p[:, 0].astype(np.int16), i=p[:, 1].astype(np.int16),
                     a=p[:, 2].astype(np.int32), j=p[:, 3].astype(np.int16), b=p[:, 4].astype(np.int32),
                     depth=self.depth[k].astype(np.uint8))
        os.replace(tmp, path)
        self._write_counts()

    def _write_counts(self):
        _atomic_write_text(self.out / "counts.json", json.dumps(dict(
            grammar=self.g.name, levels=self.stats, hash_precision=self.precision, format="v2",
            stop_reason=self.stop_reason,
            sample="grid (2,3) 100 points" if self.real else "full truth table"), indent=1))

    def load(self):
        """Rebuild saved levels from parent pointers; returns the last complete K (negative: keys-only last)."""
        cj = self.out / "counts.json"
        if not cj.exists():
            return 0
        self.stats = json.loads(cj.read_text())["levels"]
        last = 0
        for st in self.stats:
            k = st["K"]
            keys, par, depth = read_level(self.out / f"level_{k:03d}.npz")
            if depth is None:
                raise ValueError(f"{self.out}: level {k} is in the paper-1 format (no depth); "
                                 "enumerate into a fresh directory")
            if not st.get("vals_stored", True):
                self._add_level(k, None, keys, par, depth)
                return -k
            if k == 1:
                vals, _ = self._leaves()
            else:
                vals = np.empty((len(par), G.N_POINTS), np.complex128) if self.real else np.empty(len(par), np.uint64)
                for oi, i, j in np.unique(par[:, [0, 1, 3]], axis=0):
                    r = np.where((par[:, 0] == oi) & (par[:, 1] == i) & (par[:, 3] == j))[0]
                    op, a = self.g.ops[oi], self.vals[i][par[r, 2]]
                    vals[r] = self._apply(op, a) if j < 0 else self._apply(op, a, self.vals[j][par[r, 4]])
            self._add_level(k, vals, keys, par, depth)
            last = k
        return last

    # ------------------------------------------------------------ budgets (predictions for the next level)
    def _ref_level(self, field):
        ref = [st for st in self.stats if st["K"] > 1 and st["candidates"] > 0 and st[field] > 0]
        return ref[-1] if ref else None

    def _predict_new(self, k):
        """New functions at level k: candidates times the last level's new-per-candidate rate (that rate falls with
        K, so this over-predicts: a conservative estimate)."""
        cand = self._n_candidates(k)
        st = self._ref_level("U")
        return 0 if cand == 0 else (cand if st is None else int(cand * st["U"] / st["candidates"]) + 1)

    def _predict_cpu(self, k):
        st = self._ref_level("cpu_s")
        return 0.0 if st is None else self._n_candidates(k) * st["cpu_s"] / st["candidates"]

    def _stop(self, reason, k_done, msg):
        self.stop_reason = reason
        self._write_counts()
        self.log(f"{self.g.name}: stopped before K={k_done + 1} ({reason}): {msg}")
        return k_done, "budget_exhausted"

    def run(self, kmax, max_gb=12.0, cpu_budget=None, min_free_gb=0.0):
        """cpu_budget: process CPU seconds (the queue's limit covers the whole process); min_free_gb: free-disk
        floor; max_gb: memory cap. Each is checked against a prediction before every level."""
        start = self.load()
        if start < 0:
            self.log(f"{self.g.name}: already finished at K={-start} (memory ceiling)")
            return -start, "budget_exhausted"
        if start:
            self.log(f"{self.g.name}: resumed after K={start}")
        if start == 0:
            vals, _ = self._leaves()
            keys = self._key(vals)
            depth = np.ones(len(keys), np.uint8)
            self._add_level(1, vals, keys, np.full((len(keys), 5), -1, np.int32), depth)
            self.stats.append(self._level_stats(1, len(keys), keys, self._viable(vals), depth, 0.0))
            self._save_level(1)
            start = 1
        for k in range(start + 1, kmax + 1):
            if not self.real and len(self.known) == (1 << (1 << self.g.n_inputs)):
                self.log(f"{self.g.name}: all Boolean functions found by K={k - 1}")
                return k - 1, "completed"
            new = self._predict_new(k)
            if cpu_budget is not None:
                need = self._predict_cpu(k)
                if time.process_time() + need > cpu_budget:
                    return self._stop("cpu", k - 1, f"predicted {need:,.0f} CPU-s exceeds the budget")
            if min_free_gb:
                per = (4 if not self.real and int(self.g.mask) < 2 ** 32 else 8) + 15
                free_gb = (shutil.disk_usage(self.out).free - new * per) / 2 ** 30
                if free_gb < min_free_gb:
                    return self._stop("disk", k - 1, f"{free_gb:.1f} GB would remain (< {min_free_gb} GB)")
            keys_only = False
            if self.real:                                      # upper bound: every candidate is new
                size_gb = sum(v.nbytes for v in self.vals.values()) / 1e9
                size_gb += self._n_candidates(k) * 16 * G.N_POINTS / 1e9
                keys_only = size_gb > max_gb
            else:
                cur = sum(len(v) for v in self.keys.values())
                if (cur + new) * BOOL_BYTES_KEYS_ONLY / 1e9 > max_gb:
                    return self._stop("memory", k - 1, f"predicted {new:,} new functions exceed {max_gb} GB")
                keys_only = (cur + new) * BOOL_BYTES / 1e9 > max_gb
            if keys_only:
                self.stop_reason = "memory"
                self.log(f"{self.g.name}: memory ceiling ({max_gb} GB): K={k} is the last level, keys only")
                self.build_level(k, store_vals=False)
                self._log_level()
                return k, "budget_exhausted"
            self.build_level(k)
            self._log_level()
        return kmax, "completed"

    def _log_level(self):
        st = self.stats[-1]
        self.log(f"{self.g.name} K={st['K']}: N={st['N_syntactic']:,} candidates={st['candidates']:,} "
                 f"U={st['U']:,} U_cum={st['U_cum']:,} depth<=10 {st['U_depth_ok']:,} cpu={st['cpu_s']:,.1f}s")

    # ------------------------------------------------------------ lookup
    def min_size(self, vec):
        """Smallest K at which this function (value vector or truth table) appears, or None."""
        key = self._key(vec[None, :] if self.real else np.array([vec], dtype=np.uint64))[0]
        for k in sorted(self.keys):
            if np.any(self.keys[k] == key):
                return k
        return None

    def expr(self, k, idx):
        """Representative expression of function idx at level k (a minimal tree of the smallest depth)."""
        op_i, i, a, j, b = self.parents[k][idx]
        if k == 1:
            return G.expr_str(G.leaf(self._leaves()[1][idx][1]))
        name = self.g.ops[op_i].name
        if j < 0:
            return f"{name}({self.expr(i, a)})"
        return f"{name}({self.expr(i, a)}, {self.expr(j, b)})"

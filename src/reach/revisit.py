"""Re-run counter for H6d (T25, D098; docs/design/h6_explanation.md; REQ-ENG-23, REQ-ENG-24).

GP's evaluations are counted as the primary null counts functions: one per distinct enumeration key (the truth table;
for real sets `enumerate.real_keys` of the values on the enumeration grid), functions undefined somewhere on the grid
or constant included, because `units.bits_sem` and `null.p_hit` count U, not `U_finite_nonconst`. `Lookup` gives a
key's level (its minimal size) from the set's enumeration up to a bound. `Counter` is shown every evaluated generation
by `run_gp(counter=...)` and keeps cumulative counts, recorded at budget prefixes and at the stop. It draws no random
number; `run_gp` times it from outside and leaves that time out of the soft CPU limit.
"""
import functools
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from reach import grammar as G
from reach.enumerate import _coeffs, real_keys

BOUND_BITS = 23.0                                       # levels with log2 U(<=K) up to this are looked up
PREFIXES = tuple(5_000 * 2 ** i for i in range(7))      # 1/4 to 16 x E0 (E0 = 20,000 evaluations)


@functools.lru_cache(maxsize=None)
def enum_root(name):
    """The set's enumeration directory; in a second worktree, whose untracked level files are absent, the main
    worktree's (pilot re-runs from the dev worktree)."""
    import subprocess
    from reach.analysis.s2.phase1 import enum_dir
    from reach.paths import ROOT
    d = enum_dir(name)
    if (d / "level_001.npz").exists():
        return d
    r = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, capture_output=True, text=True)
    common = r.stdout.strip()
    if r.returncode or not common:                      # not a git checkout (a release archive): row 157
        return d
    return (ROOT / common).resolve().parent / d.relative_to(ROOT)


def key_function(g):
    """tree -> (key, defined): the enumerator's key of the tree's function, and whether the function is finite on the
    whole grid and not constant (the enumerator's own rule, `Enumerator._viable`; Boolean functions always are)."""
    if g.kind != "real":
        return lambda tree: (int(G.evaluate(g, tree)), True)
    xs, coeffs = G.sample_grid(g), _coeffs(2 * G.N_POINTS)

    def key(tree):
        with np.errstate(all="ignore"):
            v = G.evaluate(g, tree, xs)
            ok = bool(np.isfinite(v).all() and np.abs(v - v.mean()).max() > 1e-6)
            return int(real_keys(v[None, :], coeffs, G.HASH_PRECISION)[0]), ok
    return key


class Lookup:
    """Level of a key in a set's enumeration, for the levels with log2 U(<=K) <= bound_bits (the ceiling at most);
    0 for a key not found there."""

    def __init__(self, g, enum_dir, bound_bits=BOUND_BITS):
        d = Path(enum_dir)
        c = json.loads((d / "counts.json").read_text())
        if c.get("grammar", g.name) != g.name:
            raise ValueError(f"{d} enumerates {c['grammar']}, not {g.name}")
        if c.get("hash_precision", G.HASH_PRECISION) != G.HASH_PRECISION:
            raise ValueError(f"{d}: hash precision {c['hash_precision']}, not {G.HASH_PRECISION}")
        levels = [st for st in c["levels"] if math.log2(st["U_cum"]) <= bound_bits]
        keys, lv = [np.empty(0, np.uint64)], [np.empty(0, np.uint8)]
        for st in levels:
            with np.load(d / f"level_{st['K']:03d}.npz") as z:
                k = z["keys"].astype(np.uint64)
            if len(k) != st["U"]:
                raise ValueError(f"{d}: level {st['K']} holds {len(k)} keys, counts.json says {st['U']}")
            keys.append(k)
            lv.append(np.full(len(k), st["K"], np.uint8))
        keys, lv = np.concatenate(keys), np.concatenate(lv)
        order = np.argsort(keys, kind="stable")
        self.keys, self.lv = keys[order], lv[order]
        self.top = levels[-1]["K"] if levels else 0
        self.bound_bits = bound_bits
        self.sha256 = hashlib.sha256(self.keys.tobytes() + self.lv.tobytes()).hexdigest()

    def levels(self, keys):
        keys = np.asarray(keys, dtype=np.uint64)
        if not len(self.keys):
            return np.zeros(len(keys), int)
        i = np.minimum(np.searchsorted(self.keys, keys), len(self.keys) - 1)
        return np.where(self.keys[i] == keys, self.lv[i], 0).astype(int)

    def describe(self):
        return dict(top=int(self.top), n_keys=int(len(self.keys)), bound_bits=self.bound_bits, sha256=self.sha256)


class Counter:
    """Cumulative counts over the evaluations `run_gp` shows it: n evaluations, d distinct keys, undef evaluations of
    undefined or constant functions (real), lv distinct keys per level (index 0: not found below the bound) and lv_ok
    the same for keys whose function is finite on the grid and not constant, the run's own target left out of both;
    target_first, the evaluation at which a tree with the target's key was first evaluated (None if never). `cpu` is
    set by run_gp, which times every call."""

    def __init__(self, g, target_tree, lookup, pop_size, prefixes=PREFIXES):
        if any(p % pop_size for p in prefixes):
            raise ValueError(f"every prefix must be a whole number of generations of {pop_size}")
        self.key, self.lookup = key_function(g), lookup
        self.target = self.key(target_tree)[0] if target_tree is not None else None   # None: no target (control)
        self.prefixes = {int(p) for p in prefixes}
        self.seen, self.n, self.undef, self.cpu = set(), 0, 0, 0.0
        self.lv = np.zeros(lookup.top + 1, np.int64)
        self.lv_ok = np.zeros(lookup.top + 1, np.int64)
        self.target_first = None
        self.snaps = {}

    def generation(self, population):
        new, new_ok = [], []
        for tree in population:
            k, ok = self.key(tree)
            self.n += 1
            self.undef += not ok
            if k not in self.seen:
                self.seen.add(k)
                if k == self.target:
                    self.target_first = self.n
                else:
                    new.append(k)
                    new_ok.append(ok)
        if new:
            lv = self.lookup.levels(np.array(new, dtype=np.uint64))
            np.add.at(self.lv, lv, 1)
            np.add.at(self.lv_ok, lv[np.array(new_ok, bool)], 1)
        if self.n in self.prefixes:
            self.snaps[str(self.n)] = self._snap()

    def _snap(self):
        return dict(n=self.n, d=len(self.seen), undef=self.undef, lv=self.lv.tolist(), lv_ok=self.lv_ok.tolist())

    def record(self):
        return dict(prefixes=dict(self.snaps), stop=self._snap(), cpu_seconds=self.cpu, target_first=self.target_first,
                    target_level=int(self.lookup.levels([self.target])[0]) if self.target is not None else None,
                    lookup=self.lookup.describe())

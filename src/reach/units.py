"""Size units of a target (tool T5; brief §7.3; REQ-MEA-09, REQ-MEA-21).

nodes K; bits_uniform K·log2(r) with r symbols (terminals included); bits_syn log2 N_G(<=K) (syntactic count,
exact recurrence); bits_sem log2 U_G(<=K) (distinct functions from the enumeration), by default corrected for
float-noise duplicates (D050: per-level `noise` rows from the Phase 1 report; a level above the measured ones takes the
share of the highest measured level). Above the enumeration ceiling bits_sem is extrapolated from a least-squares line
of log2 U_cum on K over the last `fit_levels` non-empty levels, and flagged.

fit_levels = 3 was chosen on held-back levels of 12 enumerations (2026-10-19): mean error +0.006 bits, mean absolute
0.15, maximum 0.51 (5 levels: biased low, mean -0.05, maximum -0.79; quadratic fits biased high). Analyses that use
extrapolated values carry this error."""
import math

import numpy as np

from reach import grammar as G


def space_bits(g):
    """The largest possible size in bits_sem: a Boolean set with n inputs has 2^(2^n) functions, so 2^n bits; real
    sets have no bound (REQ-MEA-32). The registered unit below extrapolates above the enumeration ceiling without this
    bound (guardrail row 154)."""
    return float(1 << g.n_inputs) if g.kind == "bool" else math.inf


def capped(bits, g):
    """A bits_sem value capped at the set's space bound."""
    return min(bits, space_bits(g))


class SizeUnits:
    def __init__(self, g, levels, noise_rows=None, fit_levels=3):
        a0, a1, a2 = g.arity_counts()
        self.r = a0 + a1 + a2
        self.U = {st["K"]: st["U"] for st in levels}
        self.top = max(self.U)
        self.syn = G.syntactic_counts(g, max(400, self.top + 200))
        self.noise = {row["K"]: row["noise"] for row in (noise_rows or [])}
        self.noise_top = max(self.noise) if self.noise else None
        self._fits = {c: self._fit(c, fit_levels) for c in (True, False)}

    def _noise_at(self, k):
        if not self.noise:
            return 0
        if k in self.noise:
            return self.noise[k]
        if k > self.noise_top:
            ref = self.noise_top
            return int(round(self.U.get(k, 0) * self.noise[ref] / max(self.U[ref], 1)))
        return 0

    def _ucum(self, K, corrected):
        return sum(self.U[k] - (self._noise_at(k) if corrected else 0) for k in self.U if k <= K)

    def _fit(self, corrected, n):
        pts = [(k, math.log2(self._ucum(k, corrected))) for k in sorted(self.U) if k > 1 and self.U[k] > 0][-n:]
        if len(pts) < 2:
            return None
        k, y = np.array(pts).T
        slope, icpt = np.polyfit(k, y, 1)
        return float(slope), float(icpt)

    def nodes(self, K):
        return float(K)

    def bits_uniform(self, K):
        return K * math.log2(self.r)

    def bits_syn(self, K):
        return math.log2(sum(self.syn[1:K + 1]))

    def bits_sem(self, K, corrected=True, with_flag=False):
        if K <= self.top:
            v, flag = math.log2(self._ucum(K, corrected)), False
        else:
            slope, icpt = self._fits[corrected]
            v, flag = icpt + slope * K, True
        return (v, flag) if with_flag else v

    def all(self, K):
        sem, flag = self.bits_sem(K, with_flag=True)
        return dict(nodes=self.nodes(K), bits_uniform=self.bits_uniform(K), bits_syn=self.bits_syn(K),
                    bits_sem=sem, bits_sem_uncorrected=self.bits_sem(K, corrected=False), bits_sem_extrapolated=flag)

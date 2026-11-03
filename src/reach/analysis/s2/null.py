"""Blind-search nulls (tool T10; brief §4.4; H6). All computed from enumeration counts, no runs.

- "enumeration" (primary): size-ordered enumeration over distinct functions, in random order within a size. With E
  evaluations, a target of minimum size K is hit with probability clip((E - U(<=K-1)) / U(K), 0, 1). Its reach in
  bits_sem = log2 U(<=K) is exactly log2 E (one bit per budget doubling).
- "bottom_up": the realisable enumerator that evaluates every candidate (duplicates included): the same with the
  cumulative candidate counts C(<=K).
- "uniform": a searcher that knows the size draws uniformly, with replacement, from the U(<=K) functions:
  1 - (1 - 1/U(<=K))^E (Kronberger et al.'s idealised random search).
"""
import math


def _cum(levels, field):
    out, c = {}, 0
    for st in sorted(levels, key=lambda s: s["K"]):
        c += st[field]
        out[st["K"]] = (c - st[field], st[field])          # (cumulative below K, at K)
    return out


def p_hit(levels, K, E, kind="enumeration"):
    if kind == "uniform":
        below, at = _cum(levels, "U")[K]
        return 1 - (1 - 1 / (below + at)) ** E
    below, at = _cum(levels, "U" if kind == "enumeration" else "candidates")[K]
    if at == 0:
        return 1.0 if E >= below else 0.0
    return min(1.0, max(0.0, (E - below) / at))


def reach_bits_sem(levels, E, kind="enumeration"):
    """Reach of a null at budget E on the bits_sem scale: the point where its per-level hit probability crosses 1/2,
    interpolated linearly in bits_sem = log2 U(<=K) between the last level with P >= 1/2 and the first with P < 1/2.
    For the primary null this equals log2 E up to the granularity of one size level (brief §4.4)."""
    pts = [(math.log2(st["U_cum"]), p_hit(levels, st["K"], E, kind)) for st in sorted(levels, key=lambda s: s["K"])
           if st["U"] > 0]
    for (b0, p0), (b1, p1) in zip(pts, pts[1:]):
        if p0 >= 0.5 > p1:
            return b0 + (b1 - b0) * (p0 - 0.5) / (p0 - p1)
    return float("nan")

"""Phase 1 report (docs/plan/phases/phase1_enumeration.md): enumeration facts per function set, from the stored
enumerations under data/enumeration_s2/. Every number here is computed from files; nothing is typed in.

    .venv/bin/python -m reach.analysis.s2.phase1 [--sets ...] [--inflation]

1. Sets: ceiling (last complete level), stop reason, U(<= ceiling), lambda_G (bits per node: slope of log2 U_cum over
   the last five non-empty levels), share of functions without a minimal tree of depth <= 10 (C5), CPU.
2. Guarded vs IEEE (C4): functions present in both enumerations whose minimum size differs, and functions present in
   only one, per pair (R-ELEM6/ELEM6, R-HYB/HYB, F-EML/EML), up to the smaller ceiling.
3. Precision (brief 7.2): for a sample of functions per level (9 decimals), the level at which their representative
   tree first appears at 7 and at 11 decimals.
4. Float-noise inflation of U (REQ-MEA-21, --inflation; loads all values): per level, the share of functions that
   merge with another when values are quantised per point at about 1e-9 relative (an estimate: it can miss a noise pair
   that straddles a rounding boundary, and it also merges distinct functions closer than 1e-9 relative).
-> docs/status/phase1_report.md and data/analysis/phase1.json
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from reach import grammar as G
from reach.enumerate import _MIX, Enumerator, _coeffs, read_level, real_keys
from reach.paths import ROOT

ENUM = ROOT / "data/enumeration_s2"
from reach.s2_sets import SETS  # noqa: E402

REAL = [n for n, s in SETS.items() if s["domain"] == "real"]
BOOL = [n for n, s in SETS.items() if s["domain"] == "bool"]


def enum_dir(g):
    """Where a set's defining enumeration lives (s2_sets.SETS root; superseded 8 GB runs stay in data/enumeration_s2)."""
    if g.startswith("RS-") or g.startswith("BOOL-RS-"):                     # random held-out candidates (T26)
        return ENUM / "random" / g
    return ROOT / SETS[g]["root"] / g if g in SETS else ENUM / g
PAIRS = [("R-ELEM6", "ELEM6"), ("R-HYB", "HYB"), ("F-EML", "EML")]


def counts(d):
    p = d / "counts.json"
    return json.loads(p.read_text()) if p.exists() else None


def lam(levels, n=5):
    """Bits per node: least-squares slope of log2 U_cum on K over the last n levels that added functions."""
    pts = [(st["K"], np.log2(st["U_cum"])) for st in levels if st["U"] > 0 and st["K"] > 1][-n:]
    if len(pts) < 3:
        return float("nan")
    k, y = np.array(pts).T
    return float(np.polyfit(k, y, 1)[0])


def top_excluded(levels):
    """Share without a depth-<=10 minimal tree at the last level that has functions (NAND/NOR have empty even levels)."""
    top = next((st for st in reversed(levels) if st["U"] > 0), None)
    return 0.0 if top is None else round(1 - top.get("U_depth_ok", top["U"]) / top["U"], 4)


def set_rows(names):
    rows = []
    for g in names:
        c = counts(enum_dir(g))
        if c is None:
            rows.append(dict(set=g, status="not enumerated"))
            continue
        L = c["levels"]
        U = sum(st["U"] for st in L)
        bad = sum(st["U"] - st.get("U_depth_ok", st["U"]) for st in L)
        rows.append(dict(set=g, ceiling=L[-1]["K"], keys_only_last=not L[-1].get("vals_stored", True),
                         stop=c.get("stop_reason") or "completed", U=U, bits_per_node=round(lam(L), 3),
                         depth_excluded=round(bad / U, 4) if U else 0.0,
                         depth_excluded_top=top_excluded(L),
                         cpu_h=round(sum(st["cpu_s"] for st in L) / 3600, 2)))
    return rows


def level_keys(d):
    c = counts(d)
    return {st["K"]: read_level(d / f"level_{st['K']:03d}.npz")[0] for st in c["levels"]} if c else {}


def compare_pair(a_dir, b_dir):
    A, B = level_keys(a_dir), level_keys(b_dir)
    if not A or not B:
        return None
    kmax = min(max(A), max(B))
    ka = np.concatenate([A[k] for k in A if k <= kmax]); la = np.concatenate([np.full(len(A[k]), k) for k in A if k <= kmax])
    kb = np.concatenate([B[k] for k in B if k <= kmax]); lb = np.concatenate([np.full(len(B[k]), k) for k in B if k <= kmax])
    common, ia, ib = np.intersect1d(ka, kb, assume_unique=False, return_indices=True)
    diff = la[ia] != lb[ib]
    return dict(up_to_K=int(kmax), common=int(len(common)), size_differs=int(diff.sum()),
                guarded_smaller=int((la[ia] < lb[ib]).sum()), ieee_smaller=int((la[ia] > lb[ib]).sum()),
                only_guarded=int(len(ka) - len(common)), only_ieee=int(len(kb) - len(common)))


def precision_check(g, per_level=200, kmax=13, seed=20261019):
    """Sampled representative trees at 9 decimals: their first level at 7 and 11 decimals."""
    base = enum_dir(g)
    if counts(base) is None:
        return None
    e = Enumerator(G.get(g), base, log=lambda *_: None)
    for st in counts(base)["levels"]:
        e.keys[st["K"]], e.parents[st["K"]], e.depth[st["K"]] = read_level(base / f"level_{st['K']:03d}.npz")
    other = {p: level_keys(ENUM / f"precision{p}" / g) for p in (7, 11)}
    if not all(other.values()):
        return None
    gg, grid, rng = G.get(g), G.sample_grid(G.get(g)), random.Random(seed)
    coeffs = _coeffs(2 * G.N_POINTS)
    out = {p: dict(checked=0, changed=0, absent=0) for p in (7, 11)}
    top = min(kmax, *(max(o) for o in other.values()))
    for k in range(1, top + 1):
        idx = list(range(len(e.keys.get(k, []))))
        for i in rng.sample(idx, min(per_level, len(idx))):
            v = G.evaluate(gg, G.parse_expr(e.expr(k, i), gg), grid)
            if not np.isfinite(v).all():
                continue
            for p, levels in other.items():
                key = real_keys(v[None, :], coeffs, precision=p)[0]
                found = next((kk for kk in sorted(levels) if np.any(levels[kk] == key)), None)
                out[p]["checked"] += 1
                out[p]["absent"] += found is None
                out[p]["changed"] += found is not None and found != k
    return dict(up_to_K=top, **{f"p{p}": v for p, v in out.items()})


def relative_keys(v, coeffs3, digits=9):
    """Keys of values quantised per point to 9 decimals of the mantissa m = v / 10**e, e = floor(log10(max(1, |v|))),
    with e itself part of the key (so 5 and 7 stay distinct). Above 1 this is about 1e-9 relative; below 1, 1e-9
    absolute (as the enumeration key)."""
    with np.errstate(all="ignore"):
        a = np.where(np.isfinite(v), np.abs(v), 1.0)
        e = np.floor(np.log10(np.maximum(1.0, a)))
        m = v / 10.0 ** e
        x = np.concatenate([np.round(m.real, digits), np.round(m.imag, digits), e], axis=1) + 0.0
    x[np.isnan(x)] = np.nan
    u = x.view(np.uint64)
    u = (u ^ (u >> np.uint64(31))) * _MIX
    u ^= u >> np.uint64(29)
    return (u * coeffs3).sum(axis=1, dtype=np.uint64)


NOISE_REL = 1e-13      # float noise in our trees is about 1e-14 relative (exp(exp(exp(log x))) vs exp(exp x): 7e-15)


def max_rel_diff(u, v):
    """Largest difference between two value vectors, relative to max(1, |u|) per point (non-finite points must
    agree in being non-finite)."""
    fu, fv = np.isfinite(u), np.isfinite(v)
    if (fu != fv).any():
        return np.inf
    if not fu.any():
        return 0.0
    return float(np.max(np.abs(u[fu] - v[fu]) / np.maximum(1.0, np.abs(u[fu]))))


def inflation(g):
    """Float-noise duplicates (REQ-MEA-21). Candidates: functions whose values merge with an earlier function under the
    coarse relative key (`relative_keys`, 9 digits). Noise duplicates: candidates whose largest relative difference
    from that earlier function is at most NOISE_REL. Per level: U, candidates, noise duplicates."""
    e = Enumerator(G.get(g), enum_dir(g), log=lambda *_: None)
    if e.load() == 0:
        return None
    return inflation_from(e)


def inflation_from(e):
    """As `inflation`, from an Enumerator whose values are loaded."""
    coeffs3 = _coeffs(3 * G.N_POINTS)
    first, rows = {}, []                             # relative key -> (level, index) of its first function
    for k in sorted(e.vals):                         # a keys-only last level has no values and is skipped
        q = relative_keys(e.vals[k], coeffs3, 9)
        cand = noise = 0
        for i, key in enumerate(q.tolist()):
            if key in first:
                cand += 1
                k0, i0 = first[key]
                noise += max_rel_diff(e.vals[k0][i0], e.vals[k][i]) <= NOISE_REL
            else:
                first[key] = (k, i)
        rows.append(dict(K=k, U=len(q), candidates=cand, noise=int(noise), share=round(noise / max(len(q), 1), 5)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inflation", action="store_true", help="also measure REQ-MEA-21 (loads all values)")
    a = ap.parse_args()
    res = dict(sets=set_rows(REAL + BOOL),
               ieee=[dict(guarded=x, ieee=y, **(compare_pair(ENUM / x, ENUM / "ieee" / y) or {})) for x, y in PAIRS],
               precision={g: precision_check(g) for g in REAL})
    if a.inflation:
        res["inflation"] = {g: inflation(g) for g in REAL}
    out = ROOT / "data/analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase1.json").write_text(json.dumps(res, indent=1, default=float))
    L = ["# Phase 1 report: enumeration", "", "> Index: `docs/README.md`. Generated by `reach.analysis.s2.phase1` "
         "from `data/enumeration_s2/`; values in `data/analysis/phase1.json`. Do not edit by hand.", "",
         "## Sets", "", "| Set | ceiling K | stop | U(≤K) | bits/node | no depth-≤10 tree (all / top level) | CPU-h |",
         "|---|---|---|---|---|---|---|"]
    for r in res["sets"]:
        if "ceiling" not in r:
            L.append(f"| {r['set']} | – | {r['status']} | | | | |")
            continue
        L.append(f"| {r['set']} | {r['ceiling']}{' (keys only)' if r['keys_only_last'] else ''} | {r['stop']} | "
                 f"{r['U']:,} | {r['bits_per_node']} | {100 * r['depth_excluded']:.1f}% / "
                 f"{100 * r['depth_excluded_top']:.1f}% | {r['cpu_h']} |")
    L += ["", "## Guarded vs IEEE semantics (C4)", "",
          "| Guarded | IEEE | up to K | common | size differs | guarded smaller | IEEE smaller | only guarded | only IEEE |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in res["ieee"]:
        if "common" in r:
            L.append(f"| {r['guarded']} | {r['ieee']} | {r['up_to_K']} | {r['common']:,} | {r['size_differs']:,} | "
                     f"{r['guarded_smaller']:,} | {r['ieee_smaller']:,} | {r['only_guarded']:,} | {r['only_ieee']:,} |")
    L += ["", "## Precision sensitivity of minimum sizes (sampled representatives)", "",
          "| Set | up to K | checked | changed at 7 | absent at 7 | changed at 11 | absent at 11 |", "|---|---|---|---|---|---|---|"]
    for g, r in res["precision"].items():
        if r:
            L.append(f"| {g} | {r['up_to_K']} | {r['p7']['checked']:,} | {r['p7']['changed']} | {r['p7']['absent']} | "
                     f"{r['p11']['changed']} | {r['p11']['absent']} |")
    if a.inflation:
        L += ["", "## Float-noise inflation of U (REQ-MEA-21)", "",
              "Candidates: functions equal to an earlier one at 9 relative digits. Noise duplicates: candidates whose",
              "largest relative difference from it is at most 1e-13 (float noise is about 1e-14; distinct functions differ",
              "by more). Share = noise duplicates / U at that level.", "",
              "| Set | candidates (all levels) | noise duplicates | max share (level) |", "|---|---|---|---|"]
        for g, r in res["inflation"].items():
            if r:
                m = max(r, key=lambda x: x["share"])
                L.append(f"| {g} | {sum(x['candidates'] for x in r):,} | {sum(x['noise'] for x in r):,} | "
                         f"{100 * m['share']:.3f}% ({m['K']}) |")
    (ROOT / "docs/status/phase1_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

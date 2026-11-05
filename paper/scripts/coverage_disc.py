"""Figure 4 and the graphical abstract (D116; REQ-ANA-100): where 320,000 evaluations go on R-ARITH4 at 16 x E0.
Rings are size levels split into cells; the filled cells per ring are the measured coverage share times the cells
(at least one when the share is above zero); positions inside a ring are not data (blind search in order, GP
scattered). Every number is read from the instrumented re-runs (s2_revisit_a16) and the enumeration counts.
Layouts: "square" is the paper's Figure 4 (900 x 1020); "wide" is the graphical abstract (2000 x 800, horizontal,
large type, the bars in a right-hand column). Text sizes: REQ-ANA-105. Construction and limits: supplement S20."""
import json
import math
import random
import statistics as st
from pathlib import Path

SET, E, RESTARTS, E_RESTART = "R-ARITH4", 320000, 32, 10000
CELLS = [2, 8, 24, 48, 96, 160, 256, 384, 512]
BAND_CELLS, PER_BAND_CELL = 512, 200
COL = dict(bg="#ffffff", ink="#1d1d1b", mute="#6b6a66", empty="#ecebe6", bandc="#f4f3ef",
           bs="#666666", rs="#D55E00", lr="#0072B2")
CLS = ["bs", "rs", "lr"]
NAMES = ["Exact blind search", f"{RESTARTS} restarts of 100 generations", "One run of the whole budget"]
SHORT = ["blind search", f"{RESTARTS} restarts", "one run"]


def compute(root):
    """Coverage per search and level, distinct-function counts, repeat shares and run counts, from the records."""
    from reach.analysis.s2.load import read_phase
    root = Path(root)
    lv = [l for l in json.loads((root / f"data/enumeration_s2/{SET}/counts.json").read_text())["levels"] if l["U"] > 0]
    K = [l["K"] for l in lv]
    U = {l["K"]: l["U"] for l in lv}
    cells = dict(zip(K, CELLS))
    rows = [r for r in read_phase("s2_revisit_a16", root) if r.get("set") == SET and "rv" in r]

    def at(P):
        have = [r["rv"]["prefixes"][P] for r in rows if P in r["rv"]["prefixes"]]
        return (len(have), {k: st.mean(p["lv"][k] for p in have) for k in K}, st.mean(p["d"] for p in have),
                st.mean(p["lv"][0] for p in have), st.mean(p["n"] for p in have))
    one_runs, one, one_d, one_above, one_n = at(str(E))
    rs_runs, rs, rs_d, rs_above, rs_n = at(str(E_RESTART))
    left, blind = E, {}
    for k in K:
        blind[k] = min(U[k], left)
        left -= blind[k]
    cov = [{k: blind[k] / U[k] for k in K},
           {k: 1 - (1 - min(1, rs[k] / U[k])) ** RESTARTS for k in K},        # independence estimate (S20)
           {k: one[k] / U[k] for k in K}]
    capped = {k: min(1, RESTARTS * rs[k] / U[k]) for k in K}                  # the no-overlap ceiling
    above = [0, RESTARTS * rs_above, one_above]
    small = [sum(blind.values()), sum(U[k] * cov[1][k] for k in K), sum(one.values())]
    small_capped = sum(min(U[k], RESTARTS * rs[k]) for k in K)
    bars = [(1.0, 0.0, 0.0),
            (sum(rs.values()) / rs_n, rs_above / rs_n, 1 - rs_d / rs_n),
            (sum(one.values()) / one_n, one_above / one_n, 1 - one_d / one_n)]
    return dict(K=K, U=U, cells=cells, cov=cov, capped=capped, above=above, small=small, small_capped=small_capped,
                bars=bars, runs=(0, rs_runs, one_runs), one_restart_small=sum(rs.values()), one_restart_above=rs_above,
                one_run_evaluations=one_n, one_restart_evaluations=rs_n)


def n_fill(c, cells):
    return 0 if c <= 0 else max(1, round(c * cells))


def rings_svg(d, cx, cy, r0=30, w=27, bandgap=9, gap=20, text_px=17, numbers=True):
    """The disc: three slices (one per search), one ring per level and an aggregate ring above the ceiling."""
    K = d["K"]
    angles = [30, -90, 150]

    def ring_r(i):
        r1 = r0 + i * w + (bandgap if i >= len(K) else 0)
        return r1, r1 + w - 1.5

    def arc(r, a):
        return cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))

    def edge(r, a0, t):
        inset = math.degrees(math.asin(gap / 2 / r))
        return a0 + inset + t * (120 - 2 * inset)

    def cell(r1, r2, a0, t1, t2):
        (x1, y1), (x2, y2) = arc(r2, edge(r2, a0, t1)), arc(r2, edge(r2, a0, t2))
        (x3, y3), (x4, y4) = arc(r1, edge(r1, a0, t2)), arc(r1, edge(r1, a0, t1))
        return f"M{x1:.2f},{y1:.2f}A{r2},{r2} 0 0 1 {x2:.2f},{y2:.2f}L{x3:.2f},{y3:.2f}A{r1},{r1} 0 0 0 {x4:.2f},{y4:.2f}Z"

    rng = random.Random(11)
    out = []
    for s, a0 in enumerate(angles):
        rings = [(k, d["cells"][k], n_fill(d["cov"][s][k], d["cells"][k])) for k in K] + \
                [(">17", BAND_CELLS, round(d["above"][s] / PER_BAND_CELL))]
        for i, (k, n, f) in enumerate(rings):
            r1, r2 = ring_r(i)
            filled = set(range(f)) if s == 0 else set(rng.sample(range(n), f))
            for j in range(n):
                c = CLS[s] if j in filled else ("bandc" if k == ">17" else "empty")
                out.append(f'<path class="{c}" d="{cell(r1, r2, a0, j / n, (j + 1) / n)}"/>')
    for i, k in enumerate(K + [">17"] if numbers else []):
        x, y = arc(sum(ring_r(i)) / 2, -90)
        out.append(f'<text class="rn" font-size="{text_px}" x="{x:.0f}" y="{y:.0f}" text-anchor="middle" '
                   f'dominant-baseline="middle">{str(k).replace(">", "&gt;")}</text>')
    outer = ring_r(len(K))[1]
    return "\n".join(out), outer


def bar_svg(d, s, x, y, bw, bh):
    out = []
    for frac, kind in zip(d["bars"][s], ("small", "large", "rep")):
        if frac <= 0:
            continue
        out.append(f'<rect class="bar {CLS[s]}-{kind}" x="{x:.2f}" y="{y}" width="{max(bw * frac, 1):.2f}" height="{bh}"/>')
        x += bw * frac
    return "\n".join(out)


def subtitles(d):
    """Figure 4's lines under each search's name: above the bar and below it, short enough for 17 px type."""
    small, above, bars = d["small"], d["above"], d["bars"]
    return [([f"all {small[0]:,} distinct functions of up to 17 nodes; none larger"], ["no function evaluated twice"]),
            ([f"about {small[1]:,.0f} distinct functions of up to 17 nodes",
              f"(at most {d['small_capped']:,.0f}) and up to {above[1]:,.0f} larger ones"],
             [f"{bars[1][2]:.0%} of evaluations repeat a function", "within the same restart"]),
            ([f"{small[2]:,.0f} distinct functions of up to 17 nodes", f"and {above[2]:,.0f} larger ones"],
             [f"{bars[2][2]:.0%} of evaluations repeat a function", "already evaluated"])]


def style(lab, sub, key, head=0):
    """Text sizes in px per class: Figure 4 uses 21/17/17 (at least 7 pt at the 390 pt line width), the graphical
    abstract 46/38/38 with a 64 px head (at least 9 px when shown 491 px wide; REQ-ANA-105)."""
    return (f"path{{stroke:{COL['bg']};stroke-width:.5}} .empty{{fill:{COL['empty']}}} .bandc{{fill:{COL['bandc']}}} "
            f".bs{{fill:{COL['bs']}}} .rs{{fill:{COL['rs']}}} .lr{{fill:{COL['lr']}}} text{{font-family:Helvetica,Arial,sans-serif}} "
            f".lab{{font-weight:bold;font-size:{lab}px;fill:{COL['ink']}}} .sub{{font-size:{sub}px;fill:{COL['mute']}}} "
            f".rn{{fill:{COL['mute']}}} .key{{font-size:{key}px;fill:{COL['mute']}}} .bar{{stroke:{COL['bg']};stroke-width:1}} "
            f".bs-small{{fill:{COL['bs']}}} .rs-small{{fill:{COL['rs']}}} .lr-small{{fill:{COL['lr']}}} "
            f".rs-large{{fill:{COL['rs']};opacity:.4}} .lr-large{{fill:{COL['lr']};opacity:.4}} .rs-rep,.lr-rep{{fill:{COL['empty']}}} "
            f".head{{font-weight:bold;font-size:{head}px;fill:{COL['ink']}}}")


def key_items(x, y, items, px=17):
    out = []
    for fill, label, opacity in items:
        out.append(f'<rect x="{x:.0f}" y="{y - 0.75 * px:.0f}" width="{0.9 * px:.0f}" height="{0.9 * px:.0f}" fill="{fill}" opacity="{opacity}"/>'
                   f'<text class="key" x="{x + 1.35 * px:.0f}" y="{y}">{label}</text>')
        x += 1.35 * px + 0.55 * px * len(label) + 1.6 * px
    return "\n".join(out)


def svg_square(d):
    """The paper's Figure 4 (900 x 1020): two searches' labels and bars above the disc, blind search's below, then
    the key; 17 px text prints at 7.4 pt at the line width."""
    W, cx, cy = 900, 450, 462
    rings, outer = rings_svg(d, cx, cy)
    H = round(cy + outer + 152 + 81 + 16)
    subs = subtitles(d)
    out = [rings]
    bw, bh = 300, 11
    for s, name in enumerate(NAMES):
        lx, ly = [(cx, cy + outer + 50), (cx + 222, 24), (cx - 222, 24)][s]
        above, below = subs[s]
        out.append(f'<text class="lab" x="{lx}" y="{ly}" text-anchor="middle">{name}</text>')
        y = ly + 22
        for line in above:
            out.append(f'<text class="sub" x="{lx}" y="{y}" text-anchor="middle">{line}</text>')
            y += 20
        out.append(bar_svg(d, s, lx - bw / 2, y - 10, bw, bh))
        y += 21
        for line in below:
            out.append(f'<text class="sub" x="{lx}" y="{y}" text-anchor="middle">{line}</text>')
            y += 20
    counts = " · ".join(f"{k}: {d['U'][k]:,}" for k in d["K"])
    y0 = cy + outer + 152
    out.append(f'<text class="key" x="30" y="{y0}" font-weight="bold">Functions per level (nodes: count):</text>')
    out.append(f'<text class="key" x="30" y="{y0 + 23}">{counts}</text>')
    out.append(key_items(30, y0 + 52, [(COL["bs"], "blind search", 1), (COL["rs"], f"{RESTARTS} restarts (independence estimate)", 1),
                                       (COL["lr"], "one run", 1)]))
    out.append(f'<text class="key" x="30" y="{y0 + 81}" font-weight="bold">Bars, share of evaluations:</text>')
    out.append(key_items(265, y0 + 81, [(COL["lr"], "new, up to 17 nodes", 1), (COL["lr"], "new, larger", .4), (COL["empty"], "evaluated again", 1)]))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">\n<style>{style(21, 17, 17)}</style>\n'
            f'<rect width="{W}" height="{H}" fill="{COL["bg"]}"/>\n' + "\n".join(out) + "\n</svg>\n")


def svg_wide(d):
    """The graphical abstract (2000 x 800, the guide's 2.5 : 1): the disc on the left without ring numbers, one block
    per search on the right in large type, so the smallest text is 9 px at the guide's 13 cm (491 px) viewing width.
    No subscript glyphs (rsvg renders them badly)."""
    W, H = 2000, 800
    cx, cy = 400, 410
    rings, outer = rings_svg(d, cx, cy, numbers=False)
    small, bars = d["small"], d["bars"]
    out = [rings]
    for s, (short, x, y) in enumerate(zip(SHORT, (cx, cx + outer * 0.7, cx - outer * 0.7), (cy + outer + 56, 66, 66))):
        out.append(f'<text class="lab" x="{x:.0f}" y="{y:.0f}" text-anchor="middle" style="fill:{COL[CLS[s]]}">{short}</text>')
    x0, bw, bh = 830, 1120, 26
    out.append(f'<text class="head" x="{x0}" y="78">Where 320,000 evaluations go</text>')
    out.append(f'<text class="sub" x="{x0}" y="130">{SET}, real-valued symbolic regression</text>')
    out.append(f'<text class="sub" x="{x0}" y="176">Rings: size levels; filled: share of the level evaluated</text>')
    lines = [f"{small[0]:,} new functions of up to 17 nodes", f"about {small[1]:,.0f} new functions of up to 17 nodes",
             f"{small[2]:,.0f} new functions of up to 17 nodes"]
    for s, name in enumerate(NAMES):
        ly = 252 + s * 142
        out.append(f'<rect x="{x0}" y="{ly - 34}" width="36" height="36" fill="{COL[CLS[s]]}"/>')
        out.append(f'<text class="lab" x="{x0 + 52}" y="{ly}">{name}</text>')
        out.append(f'<text class="sub" x="{x0}" y="{ly + 46}">{lines[s]}</text>')
        out.append(bar_svg(d, s, x0, ly + 62, bw, bh))
    out.append(key_items(x0, 712, [(COL["lr"], "new, up to 17 nodes", 1), (COL["lr"], "new, larger", .4), (COL["empty"], "repeat", 1)], px=38))
    out.append(f'<text class="key" x="{x0}" y="770">GP: means over runs; restarts: independence estimate</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">\n<style>{style(46, 38, 38, 64)}</style>\n'
            f'<rect width="{W}" height="{H}" fill="{COL["bg"]}"/>\n' + "\n".join(out) + "\n</svg>\n")


def render(root, layout="square"):
    d = compute(root)
    return (svg_square if layout == "square" else svg_wide)(d), d

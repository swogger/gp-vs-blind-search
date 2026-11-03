"""Random held-out function sets (tool T26; D053; docs/design/function_sets.md, "Held-out sets (b)").

Universes: real {+, -, *, /, exp, log, sin, cos, sqrt, square, eml} with constant 1 (guarded semantics); Boolean the 10
non-trivial two-input gates, not, and the constants 0 and 1. Draw a size k uniformly (real 2-5, Boolean 2-4), then a
uniform subset of that size; reject (and log) a real set without a binary operator, a Boolean set that is not
functionally complete (Post's criterion, validated against enumeration in tests), and any set equal to a listed set.
The ceiling rule (>= 10) is applied after enumeration, and the next candidate replaces a failing one. The seed comes
from the Stage 1 registration (`draw_seed` of its OSF identifier, plus 0 for real and 1 for Boolean, as in Stage 0);
the draw runs only after that filing (D064).
"""
import random

from reach import grammar as G
from reach import s2_sets as S

REAL_UNIVERSE = ["+", "-", "*", "/", "exp", "log", "sin", "cos", "sqrt", "square", "eml"]
BOOL_GATES = ["and", "or", "nand", "nor", "xor", "xnor", "imp", "rimp", "nimp", "rnimp"]
BOOL_UNIVERSE = BOOL_GATES + ["not", "c0", "c1"]


def _table(sym):
    """Truth table of a symbol as a dict from argument tuples to 0/1 (constants are nullary)."""
    if sym in ("c0", "c1"):
        return {(): int(sym[1])}
    if sym == "not":
        return {(a,): 1 - a for a in (0, 1)}
    m = 1
    f = G.BOOL_OPS[sym].fn
    return {(a, b): f(m, a, b) & 1 for a in (0, 1) for b in (0, 1)}


def is_complete(ops, consts):
    """Post's functional completeness: for each of the five maximal clones, some symbol lies outside it."""
    syms = list(ops) + [f"c{c}" for c in consts]
    tabs = [_table(s) for s in syms]
    def out_t0(t): return t[tuple(0 for _ in next(iter(t)))] != 0
    def out_t1(t): return t[tuple(1 for _ in next(iter(t)))] != 1
    def out_mono(t): return any(t[x] > t[y] for x in t for y in t if all(a <= b for a, b in zip(x, y)))
    def out_selfdual(t): return any(t[x] == t[tuple(1 - a for a in x)] for x in t) if next(iter(t)) != () else True
    def out_affine(t):
        n = len(next(iter(t)))
        for c0 in (0, 1):
            for coef in range(1 << n):
                if all(t[x] == (c0 ^ (sum(((coef >> i) & 1) * x[i] for i in range(n)) & 1)) for x in t):
                    return False
        return True
    return all(any(test(t) for t in tabs) for test in (out_t0, out_t1, out_mono, out_selfdual, out_affine))


def listed_symbol_sets(domain):
    out = []
    for name, s in S.SETS.items():
        if s["domain"] != domain:
            continue
        g = G.get(name)
        syms = list(g.operators)
        if domain == "bool":
            syms += [f"c{c}" for c in g.constants]
        out.append(syms)
    return out


SAFE = {"+": "add", "-": "sub", "*": "mul", "/": "div"}          # path-safe names (an enumeration is a folder)
UNSAFE = {v: k for k, v in SAFE.items()}


def draw_seed(osf_id):
    """The first 8 bytes (big-endian) of the SHA-256 of the Stage 1 OSF identifier (lower case), as the Stage 0 draw
    used the set-list file: OSF assigns the identifier, so nobody can choose the seed."""
    import hashlib
    return int.from_bytes(hashlib.sha256(osf_id.lower().encode()).digest()[:8], "big")


def grammar_from_name(name):
    """Rebuild a random set from its name (G.get calls this in any process, e.g. the enumeration task)."""
    if name.startswith("BOOL-RS-"):
        body, n = name[len("BOOL-RS-"):].rsplit("-n", 1)
        parts = body.split("_")
        return make_grammar("bool", [p for p in parts if not p.startswith("c")],
                            [int(p[1]) for p in parts if p.startswith("c")], int(n))
    if name.startswith("RS-"):
        return make_grammar("real", [UNSAFE.get(p, p) for p in name[len("RS-"):].split("_")], [])
    raise KeyError(name)


def make_grammar(domain, ops, consts, n=5):
    if domain == "real":
        name = "RS-" + "_".join(SAFE.get(o, o) for o in ops)
        g = G.Grammar(name, "real", tuple(G.GUARDED_OPS[o] for o in ops), ("x",), (1.0,))
    else:
        name = f"BOOL-RS-{'_'.join(ops)}{''.join(f'_c{c}' for c in consts)}-n{n}"
        g = G.Grammar(name, "bool", tuple(G.BOOL_OPS[o] for o in ops), tuple(f"x{i + 1}" for i in range(n)),
                      tuple(consts), n)
    G.GRAMMARS[name] = g
    return g


def draw_candidates(domain, seed, n=4, max_tries=10_000):
    """The first n accepted candidates in draw order (a later ceiling failure takes the next one), with the log."""
    rng = random.Random(seed)
    listed = {frozenset(s) for s in listed_symbol_sets(domain)}
    out, seen = [], set()
    for _ in range(max_tries):
        if len(out) == n:
            break
        if domain == "real":
            k = rng.randint(2, 5)
            ops = sorted(rng.sample(REAL_UNIVERSE, k))
            consts, key = [], frozenset(ops)
            if not any(G.REAL_SYMBOL_ARITY[o] == 2 for o in ops) or key in listed or key in seen:
                continue
        else:
            k = rng.randint(2, 4)
            syms = sorted(rng.sample(BOOL_UNIVERSE, k))
            ops = [s for s in syms if not s.startswith("c")]
            consts = [int(s[1]) for s in syms if s.startswith("c")]
            key = frozenset(syms)
            if not ops or key in listed or key in seen or not is_complete(ops, consts):
                continue
        seen.add(key)
        out.append(dict(domain=domain, k=k, ops=ops, constants=consts))
    return out

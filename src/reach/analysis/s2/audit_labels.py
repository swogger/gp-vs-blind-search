"""Audit 2 (D068): recompute a run's exactness independently of the engine.

The engine calls a run exact when its best tree's values on the 100-point grid equal the target's after rounding to
9 decimals relative to the target's scale (Boolean: equal truth tables). Here the same question is answered with code
that shares nothing with the engine: a small parser, 30-digit mpmath arithmetic on the same grid points (real), and an
explicit row-by-row truth table (Boolean). Two real functions agree if every point differs by at most TOL relative to
the scale. TOL is far below the engine's 9-decimal rounding, so identical functions agree (they differ by about
1e-30 at 30 digits) while a near miss the engine might round into a match is exposed. Values stay real while their
imaginary part is zero, so ln 0 = -inf and exp(-inf) = 0 behave as in the engine's extended-real semantics.
Only counts of agreement are reported.
"""
import mpmath as mp

DPS = 30
TOL = 1e-12
_LO, _HI, _N = 2.0, 3.0, 100


def parse(s):
    """'op(a, b)' -> (op, [args]); leaves are names or numbers."""
    s = s.strip()
    i = s.find("(")
    if i < 0:
        return s
    op, body, depth, args, cur = s[:i].strip(), s[i + 1:-1], 0, [], ""
    for ch in body:
        if ch == "," and depth == 0:
            args.append(cur)
            cur = ""
            continue
        depth += (ch == "(") - (ch == ")")
        cur += ch
    args.append(cur)
    return op, [parse(a) for a in args]


EXP_LIMIT = 1e5                     # exp beyond this is +inf or 0, as in IEEE double (limit ~710); D091


def _exp(a):
    """mp.exp, except that an argument beyond +-EXP_LIMIT gives +inf or 0 (IEEE's overflow and underflow, which any
    double reaches far earlier): 30-digit mpmath never returned on exp(exp(exp(7))) (a robust-phase failure)."""
    re = a.real if isinstance(a, mp.mpc) else a
    if mp.isfinite(re) and abs(re) > EXP_LIMIT:
        if isinstance(a, mp.mpc) and a.imag != 0:
            raise OverflowError("complex exp beyond the double range")
        return mp.inf if re > 0 else mp.mpf(0)
    return mp.exp(a)


_REAL = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b, "/": lambda a, b: a / b,
         "exp": _exp, "log": mp.log, "sin": mp.sin, "cos": mp.cos, "sqrt": mp.sqrt, "square": lambda a: a * a,
         "eml": lambda a, b: _exp(a) - mp.log(b), "reml": lambda a, b: _exp(b) - mp.log(a),
         "lde": lambda a, b: mp.log(a) / _exp(b)}


def _real(t, x):
    if isinstance(t, str):
        return mp.mpf(x) if t == "x" else mp.mpf(t)
    op, args = t
    v = _REAL[op](*[_real(a, x) for a in args])
    return mp.mpf(v.real) if isinstance(v, mp.mpc) and v.imag == 0 else v


NEAR = 1e-9                         # the engine's 9-decimal rounding can accept differences below about this


def max_rel_diff_real(a, b, scale):
    """Largest |a - b| / scale over the grid at 30 digits; inf if either is undefined somewhere."""
    ta, tb = parse(a), parse(b)
    step = (mp.mpf(_HI) - _LO) / (_N + 1)
    worst = mp.mpf(0)
    with mp.workdps(DPS):
        for i in range(_N):
            x = _LO + step * (i + 1)
            try:
                va, vb = _real(ta, x), _real(tb, x)
            except (ZeroDivisionError, ValueError, OverflowError):
                return float("inf")
            if not (mp.isfinite(va) and mp.isfinite(vb)):
                if va == vb:                                   # both the same infinity: equal there
                    continue
                return float("inf")
            worst = max(worst, abs(va - vb) / scale)
    return float(worst)


def same_function_real(a, b, scale):
    return max_rel_diff_real(a, b, scale) <= TOL


def classify(diff):
    """identical (<= TOL), near (accepted by the registered rounding, < NEAR), different."""
    return "identical" if diff <= TOL else "near" if diff < NEAR else "different"


_BOOL = {"and": lambda a, b: a & b, "or": lambda a, b: a | b, "nand": lambda a, b: 1 - (a & b),
         "nor": lambda a, b: 1 - (a | b), "xor": lambda a, b: a ^ b, "xnor": lambda a, b: 1 - (a ^ b),
         "not": lambda a: 1 - a, "imp": lambda a, b: (1 - a) | b, "rimp": lambda a, b: a | (1 - b),
         "nimp": lambda a, b: a & (1 - b), "rnimp": lambda a, b: (1 - a) & b}


def _bool(t, row):
    if isinstance(t, str):
        return row[t] if t in row else int(float(t))
    op, args = t
    return _BOOL[op](*[_bool(a, row) for a in args])


def same_function_bool(a, b, n):
    ta, tb = parse(a), parse(b)
    for r in range(1 << n):
        row = {f"x{i + 1}": (r >> (n - 1 - i)) & 1 for i in range(n)}
        if _bool(ta, row) != _bool(tb, row):
            return False
    return True


# ---- Operon (D091): its printed infix model, real-valued, with a free scale and intercept (D056's rule) ----

_OPF = {"exp": _exp, "log": mp.log, "sin": mp.sin, "cos": mp.cos, "sqrt": mp.sqrt, "square": lambda a: a * a,
        "abs": abs}


def parse_infix(s):
    """Operon's infix string -> tree of tuples: ('num', v), ('X1',), (op, a, b), ('neg', a), ('f', name, a).
    Precedence: + - < * / < unary minus < ^ (right associative), as Operon prints and Python reads it."""
    toks, i = [], 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
        elif c.isdigit() or c == ".":
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] in ".eE" or (s[j] in "+-" and s[j - 1] in "eE")):
                j += 1
            toks.append(("num", s[i:j]))
            i = j
        elif c.isalpha():
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                j += 1
            toks.append(("name", s[i:j]))
            i = j
        else:
            toks.append(("op", c))
            i += 1
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else ("end", "")

    def take(kind=None, val=None):
        t = peek()
        if (kind and t[0] != kind) or (val and t[1] != val):
            raise ValueError(f"parse error at token {pos[0]}: {t}")
        pos[0] += 1
        return t

    def expr():
        a = term()
        while peek() in (("op", "+"), ("op", "-")):
            a = (take()[1], a, term())
        return a

    def term():
        a = unary()
        while peek() in (("op", "*"), ("op", "/")):
            a = (take()[1], a, unary())
        return a

    def unary():
        if peek() == ("op", "-"):
            take()
            return ("neg", unary())
        if peek() == ("op", "+"):
            take()
            return unary()
        return power()

    def power():
        a = atom()
        if peek() == ("op", "^"):
            take()
            return ("^", a, unary())
        return a

    def atom():
        t = take()
        if t[0] == "num":
            return ("num", t[1])
        if t == ("op", "("):
            a = expr()
            take("op", ")")
            return a
        if t[0] == "name" and t[1] == "X1":
            return ("X1",)
        if t[0] == "name" and t[1] in _OPF:
            take("op", "(")
            a = expr()
            take("op", ")")
            return ("f", t[1], a)
        raise ValueError(f"unknown token {t}")

    tree = expr()
    if peek()[0] != "end":
        raise ValueError("trailing tokens")
    return tree


class Undefined(Exception):
    """A value the real-valued model does not have."""


def operon_value(t, x):
    """Value at x (30 digits when called inside mp.workdps). Real only: a complex or non-finite intermediate whose
    real-valued semantics has no value raises Undefined; exp(-inf) = 0 is kept, as in IEEE arithmetic."""
    k = t[0]
    if k == "num":
        return mp.mpf(t[1])
    if k == "X1":
        return mp.mpf(x)
    if k == "neg":
        return -operon_value(t[1], x)
    if k == "f":
        a = operon_value(t[2], x)
        if t[1] in ("log", "sqrt") and a < 0:
            raise Undefined(t[1])
        v = _OPF[t[1]](a)
    else:
        a, b = operon_value(t[1], x), operon_value(t[2], x)
        if k == "/" and b == 0:
            raise Undefined("division by zero")
        v = {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b, "/": lambda: a / b,
             "^": lambda: a ** b}[k]()
    if isinstance(v, mp.mpc):
        if v.imag != 0:
            raise Undefined("complex")
        v = mp.mpf(v.real)
    if mp.isnan(v):
        raise Undefined("nan")
    return v


def operon_diff(model, target_expr, scale):
    """Largest |a + b h - t| / scale over the grid at 30 digits, a and b refitted by least squares (D056: Operon's
    scale and intercept are part of its representation); inf if the model or the target has no real value somewhere."""
    tm, tt = parse_infix(model), parse(target_expr)
    step = (mp.mpf(_HI) - _LO) / (_N + 1)
    with mp.workdps(DPS):
        hs, ts = [], []
        for i in range(_N):
            x = _LO + step * (i + 1)
            try:
                h, tv = operon_value(tm, x), _real(tt, x)
            except (Undefined, ZeroDivisionError, ValueError, OverflowError):
                return float("inf")
            if isinstance(tv, mp.mpc) or not (mp.isfinite(h) and mp.isfinite(tv)):
                return float("inf")
            hs.append(h)
            ts.append(tv)
        n = len(hs)
        mh, mt = sum(hs) / n, sum(ts) / n
        sxx = sum((h - mh) ** 2 for h in hs)
        b = sum((h - mh) * (tv - mt) for h, tv in zip(hs, ts)) / sxx if sxx > 0 else mp.mpf(0)
        a = mt - b * mh
        return float(max(abs(a + b * h - tv) for h, tv in zip(hs, ts)) / scale)


def _ieee(t, x):
    """IEEE double evaluation (numpy float64 throughout: x/0 = +-inf, 0/inf = 0, log 0 = -inf), D056's arithmetic;
    used only when the 30-digit real evaluation finds the model undefined."""
    import numpy as np
    k = t[0]
    if k == "num":
        return np.float64(t[1])
    if k == "X1":
        return np.float64(x)
    if k == "neg":
        return -_ieee(t[1], x)
    if k == "f":
        a = _ieee(t[2], x)
        return {"exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos, "sqrt": np.sqrt, "square": np.square,
                "abs": np.abs}[t[1]](a)
    a, b = _ieee(t[1], x), _ieee(t[2], x)
    return {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide, "^": np.power}[k](a, b)


def operon_diff_ieee(model, target_expr, scale):
    """As operon_diff, with the model's values in IEEE double arithmetic (the target still at 30 digits)."""
    import numpy as np
    tm, tt = parse_infix(model), parse(target_expr)
    step = (mp.mpf(_HI) - _LO) / (_N + 1)
    with np.errstate(all="ignore"), mp.workdps(DPS):
        hs, ts = [], []
        for i in range(_N):
            x = _LO + step * (i + 1)
            h = _ieee(tm, float(x))
            try:
                tv = _real(tt, x)
            except (ZeroDivisionError, ValueError, OverflowError):
                return float("inf")
            if not np.isfinite(h) or isinstance(tv, mp.mpc) or not mp.isfinite(tv):
                return float("inf")
            hs.append(mp.mpf(float(h)))
            ts.append(tv)
        n = len(hs)
        mh, mt = sum(hs) / n, sum(ts) / n
        sxx = sum((h - mh) ** 2 for h in hs)
        b = sum((h - mh) * (tv - mt) for h, tv in zip(hs, ts)) / sxx if sxx > 0 else mp.mpf(0)
        a = mt - b * mh
        return float(max(abs(a + b * h - tv) for h, tv in zip(hs, ts)) / scale)


def _classify_operon(model, target_expr, scale):
    """identical / near / different in real arithmetic; 'ieee_only' when the model is undefined in real arithmetic
    but equals the target (within the registered rounding) in IEEE double arithmetic."""
    d = operon_diff(model, target_expr, scale)
    if d == float("inf") and classify(operon_diff_ieee(model, target_expr, scale)) != "different":
        return "ieee_only"
    return classify(d)


def _expression(r):
    return r.get("model") if r.get("algorithm") == "operon" else r.get("best_expr")


def audit(rows, targets, n_fail=300, seed=20261022):
    """rows: run records of one phase; targets: {(set, target_id): target}. Every exact run and n_fail sampled failures
    are recomputed. Returns counts and the keys where the independent answer disagrees with the record. A completed
    row without an expression is never dropped silently (row 128): it is listed under `unchecked` with the reason."""
    import random
    from reach import grammar as G
    done = [r for r in rows if r.get("terminal_state") in ("converged", "budget_exhausted")]
    unchecked = {r["key"]: "no expression recorded" for r in done if not _expression(r)}
    ok = [r for r in done if _expression(r)]
    exact = [r for r in ok if r.get("exact")]
    fails = [r for r in ok if not r.get("exact")]
    fails = random.Random(seed).sample(fails, min(n_fail, len(fails)))
    out = dict(exact_checked=0, exact_near_miss=[], exact_disputed=[], failures_checked=0, failures_near=[],
               failures_disputed=[], unchecked=unchecked, exact_ieee_only=[])
    for r, label in [(r, True) for r in exact] + [(r, False) for r in fails]:
        t = targets[(r["set"], r["target_id"])]
        if r.get("algorithm") == "operon":
            cls = _classify_operon(r["model"], t["expression"], t["target_scale"])
        elif G.get(r["set"]).kind == "bool":
            cls = "identical" if same_function_bool(r["best_expr"], t["expression"], G.get(r["set"]).n_inputs) \
                else "different"
        else:
            cls = classify(max_rel_diff_real(r["best_expr"], t["expression"], t["target_scale"]))
        out["exact_checked" if label else "failures_checked"] += 1
        if cls == "ieee_only":                                  # Operon only (D091): exact by D056's IEEE rule
            out["exact_ieee_only" if label else "failures_disputed"].append(r["key"])
        elif label and cls == "near":
            out["exact_near_miss"].append(r["key"])            # exact by the registered rule, not mathematically
        elif label and cls == "different" or not label and cls == "identical":
            out["exact_disputed" if label else "failures_disputed"].append(r["key"])   # a real disagreement
        elif not label and cls == "near":
            out["failures_near"].append(r["key"])
    return out


TARGET_FILES = {"s2_cal_pool": "pool"}          # phases whose runs use another targets file than the default


def targets_file(phase, default):
    return TARGET_FILES.get(phase, default)


def run(phases, name, root):
    import json
    from reach.runner.queue import read_results
    res = {}
    for ph in phases:
        rows = list(read_results(root / f"data/runs/{ph}/results.jsonl").values())
        targets = {}
        for g in sorted({r["set"] for r in rows if r.get("set")}):
            for t in json.loads((root / f"data/targets_s2/{g}/{targets_file(ph, name)}.json").read_text()):
                targets[(g, t["target_id"])] = t
        res[ph] = audit(rows, targets)
        a = res[ph]
        print(ph, f"exact checked {a['exact_checked']}, near misses {len(a['exact_near_miss'])}, disputed "
                  f"{len(a['exact_disputed'])}; failures checked {a['failures_checked']}, disputed "
                  f"{len(a['failures_disputed'])}; unchecked {len(a['unchecked'])}", flush=True)
    return res


def main():
    """python -m reach.analysis.s2.audit_labels PHASE ... NAME [--out STEM]: NAME is the default targets file
    (calibration or pilot); the result goes to data/analysis/audit_labels_{STEM or NAME}.json."""
    import argparse
    import json
    from reach.paths import ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("args", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    phases, name = a.args[:-1], a.args[-1]
    res = run(phases, name, ROOT)
    out = ROOT / f"data/analysis/audit_labels_{a.out or name}.json"
    out.write_text(json.dumps(res, indent=1))
    print("written", out.relative_to(ROOT))


if __name__ == "__main__":
    main()

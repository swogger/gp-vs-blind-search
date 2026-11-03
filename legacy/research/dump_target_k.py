"""Compute EML K for every study target by compiling it with the witness library.

Replaces hand-written K estimates in Appendix C with values produced by the
project's own compiler (eml_core.compile), which lowers a formula into an actual
EML tree by substituting library witnesses and counts its tokens.

Every K reported here is an UPPER BOUND (a specific construction exists at this
size), except where the witness is annotated as proven minimal. K is always odd:
an EML tree with m operators has exactly 2m+1 tokens.

Usage:  python3 dump_target_k.py [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(
    HERE, "..", "eml-skill", "eml-skill", "skills", "_shared")))

import cmath                                          # noqa: E402
from eml_core.compile import compile_formula          # noqa: E402
from eml_core.eml import evaluate, k_tokens           # noqa: E402
from eml_core.witnesses import WITNESSES              # noqa: E402
from test_suite import TEST_MATRIX, EML_K, eml_plateau_mult  # noqa: E402

# sympy-parseable form of each target, transcribed from test_suite.target_* .
TARGET_FORMULAS = {
    "1.1": "exp(x)",
    "1.2": "log(x)",
    "1.3": "x + y",
    "1.4": "exp(x)*exp(-x**2)",
    "2.1": "sin(x) + cos(x)",
    "3.1": "sinh(x)",
    "3.2": "atan(x)",
    "3.3": "x**x",
    "3.4": "1/(1 + exp(-x))",
    "3.5": "x**3 - 2*x**2 + x",
    "4.1": "exp(sin(x)) + log(x**2 + 1)",
    # test_suite writes the offset as the decimal literal 0.1; sympy Floats are
    # not synthesizable over the leaf alphabet, so it is compiled as the exact
    # rational 1/10 it denotes.  Footnote this wherever 4.2's K is quoted.
    "4.2": "sin(x*exp(y)) - cos(log(x**2 + y**2 + Rational(1,10)))",
}

# Numerical check: does the compiled tree actually reproduce the target?
VERIFY = {
    "1.1": lambda x, y: cmath.exp(x),
    "1.2": lambda x, y: cmath.log(x),
    "1.3": lambda x, y: x + y,
    "1.4": lambda x, y: cmath.exp(x) * cmath.exp(-x ** 2),
    "2.1": lambda x, y: cmath.sin(x) + cmath.cos(x),
    "3.1": lambda x, y: cmath.sinh(x),
    "3.2": lambda x, y: cmath.atan(x),
    "3.3": lambda x, y: cmath.exp(x * cmath.log(x)),
    "3.4": lambda x, y: 1 / (1 + cmath.exp(-x)),
    "3.5": lambda x, y: x ** 3 - 2 * x ** 2 + x,
    "4.1": lambda x, y: cmath.exp(cmath.sin(x)) + cmath.log(x ** 2 + 1),
    "4.2": lambda x, y: (cmath.sin(x * cmath.exp(y))
                         - cmath.cos(cmath.log(x ** 2 + y ** 2 + 0.1))),
    "nguyen1": lambda x, y: x ** 3 + x ** 2 + x,
    "nguyen7": lambda x, y: cmath.log(x + 1) + cmath.log(x ** 2 + 1),
}

VERIFY_POINTS = [(0.3, 0.4), (0.7, 1.1), (1.3, 0.2), (2.0, 1.7), (0.45, 0.9)]

SR_FORMULAS = {
    "nguyen1": "x**3 + x**2 + x",
    "nguyen7": "log(x + 1) + log(x**2 + 1)",
    "keijzer6": None,   # harmonic partial sum: not an elementary closed form
    "vladislavleva4":
        "10/(5 + (x1-3)**2 + (x2-3)**2 + (x3-3)**2 + (x4-3)**2 + (x5-3)**2)",
}

SR_STD_OPS = ["+", "-", "*", "/", "exp", "log"]   # run_sr_benchmarks.STD_OPS


def entry(label, name, formula, std_ops):
    d = {"label": label, "name": name, "formula": formula,
         "patience_mult": eml_plateau_mult(std_ops) if std_ops else None,
         "std_ops": std_ops}
    if formula is None:
        d.update(K=None, status="not an elementary closed form; no compilation")
        return d
    try:
        r = compile_formula(formula)
    except Exception as exc:                       # noqa: BLE001
        d.update(K=None, status=f"compile error: {exc}")
        return d
    K = r.K if (r.K is not None and r.K > 0) else None   # compile uses -1 for "blocked"
    d.update(
        K=K,
        odd=(K % 2 == 1) if K else None,
        m_operators=((K - 1) // 2) if K else None,
        depth=r.depth,
        leaves=r.leaves,
        needs_tree=[{"primitive": n.primitive, "K_upper_bound": n.K_upper_bound}
                    for n in (r.needs_tree or [])],
        diagnostics=list(r.diagnostics or []),
        used_witnesses=sorted(r.used_witnesses or []),
        status="compiled" if K else "blocked",
    )
    if K and k_tokens(r.ast) != K:
        d["status"] = f"INCONSISTENT: k_tokens={k_tokens(r.ast)} != K={K}"
    fn = VERIFY.get(label)
    if K and fn is not None:
        worst = 0.0
        for xv, yv in VERIFY_POINTS:
            try:
                got = evaluate(r.ast, x=complex(xv), y=complex(yv))
                want = fn(complex(xv), complex(yv))
                worst = max(worst, abs(got - want) / max(1.0, abs(want)))
            except Exception as exc:                # noqa: BLE001
                worst = float("nan")
                d["verify_error"] = str(exc)
                break
        d["verify_max_rel_error"] = worst
        d["verified"] = bool(worst == worst and worst < 1e-9)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    out = {"primitives": {k: {"K": v, "odd": v % 2 == 1, "m_operators": (v - 1) // 2,
                              "minimal": bool(getattr(WITNESSES.get(
                                  {"+": "add", "-": "sub", "*": "mult", "/": "div",
                                   "exp": "exp", "log": "ln"}.get(k, k), None),
                                  "minimal", False))}
                          for k, v in EML_K.items()},
           "targets": {}, "sr_benchmarks": {}}

    for tid, t in TEST_MATRIX.items():
        out["targets"][tid] = entry(tid, t["name"], TARGET_FORMULAS.get(tid),
                                    t["std_ops"])
    for bid, f in SR_FORMULAS.items():
        out["sr_benchmarks"][bid] = entry(bid, bid, f, SR_STD_OPS)

    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return

    print("PRIMITIVES (test_suite.EML_K)")
    print(f"  {'op':<6}{'K':>6}  odd  {'m':>5}")
    for k, v in out["primitives"].items():
        print(f"  {k:<6}{v['K']:>6}  {str(v['odd']):<5}{v['m_operators']:>5}")

    print("\nSTUDY TARGETS (K computed by eml_core.compile; construction upper bounds)")
    print(f"  {'id':<5}{'K':>7} {'odd':<5}{'m':>5} {'mult':>5} {'verified':<9}"
          f"{'formula':<42}status")
    for tid, d in out["targets"].items():
        K = d["K"] if d["K"] is not None else "-"
        v = d.get("verified")
        vs = "yes" if v else ("NO" if v is False else "-")
        print(f"  {tid:<5}{str(K):>7} {str(d.get('odd','-')):<5}"
              f"{str(d.get('m_operators','-')):>5} {str(d['patience_mult']):>5} {vs:<9}"
              f"{str(d['formula'])[:40]:<42}{d['status']}")
        if d.get("needs_tree"):
            print(f"        blocked on: {d['needs_tree']}")
        if d.get("diagnostics"):
            print(f"        diagnostics: {d['diagnostics'][:2]}")

    print("\nSR BENCHMARKS (all share one STD_OPS, so all share one multiplier)")
    for bid, d in out["sr_benchmarks"].items():
        K = d["K"] if d["K"] is not None else "-"
        v = d.get("verified")
        vs = "yes" if v else ("NO" if v is False else "-")
        print(f"  {bid:<16}{str(K):>7}  verified={vs:<5} mult={d['patience_mult']}  "
              f"{d['status']}")
        if d.get("diagnostics"):
            print(f"        diagnostics: {d['diagnostics'][:2]}")

    add = WITNESSES["add"]
    print(f"\nADDITION WITNESS  K={add.K}  minimal={add.minimal}")
    print(f"  {add.tree}")
    print(f"  note: {add.note}")

    n1 = out["sr_benchmarks"]["nguyen1"]["patience_mult"]
    poly = out["targets"]["3.5"]["patience_mult"]
    print(f"\nR3.4 figure: Nguyen-1 mult={n1}, Polynomial (test 3.5) mult={poly}")
    print(f"  Nguyen-1 receives {100*(1-n1/poly):.1f}% less patience than Polynomial")


if __name__ == "__main__":
    main()

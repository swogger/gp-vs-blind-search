"""
EML tree → human-readable formula via sympy.

Converts a ga-lib Node tree (with 'eml' operators and numeric/variable leaves)
into a simplified sympy expression, then renders it as a clean string.

eml(a, b) = exp(a) - ln(b)

Falls back to the raw nested form if sympy simplification raises or times out.

Thread safety: uses multiprocessing.Process for the timeout so it works from
any calling context (main thread, ThreadPoolExecutor, or subprocess).
SIGALRM is NOT used — SIGALRM can only be set from the main thread and would
crash if called from a worker thread inside run_matrix.py.
"""

import multiprocessing
from typing import Dict, Optional
import sympy as sp

# ---------------------------------------------------------------------------
# Sympy symbol table — covers all variable names in test_suite.py.
# ---------------------------------------------------------------------------
_SYMS: Dict[str, sp.Symbol] = {
    "x":  sp.Symbol("x",  real=True),
    "x1": sp.Symbol("x1", real=True),
    "x2": sp.Symbol("x2", real=True),
    "y":  sp.Symbol("y",  real=True),
}

_TIMEOUT_S = 3.0


def _node_to_sympy(node) -> sp.Expr:
    """Recursively convert a ga-lib Node to a sympy expression."""
    if not node.is_operator:
        v = node.value
        if isinstance(v, str):
            if v not in _SYMS:
                _SYMS[v] = sp.Symbol(v, real=True)
            return _SYMS[v]
        f = float(v.real) if isinstance(v, complex) else float(v)
        return sp.Rational(f).limit_denominator(1000) if abs(f - round(f, 6)) < 1e-9 else sp.Float(f, 10)

    op    = node.value
    left  = _node_to_sympy(node.left)  if node.left  else sp.Integer(0)
    right = _node_to_sympy(node.right) if node.right else sp.Integer(0)

    if op == 'eml':  return sp.exp(left) - sp.log(right)
    if op == '+':    return left + right
    if op == '-':    return left - right
    if op == '*':    return left * right
    if op == '/':    return left / right
    if op == 'exp':  return sp.exp(left)
    if op == 'log':  return sp.log(left)
    if op == 'sin':  return sp.sin(left)
    if op == 'cos':  return sp.cos(left)
    if op == 'tan':  return sp.tan(left)
    if op == 'sinh': return sp.sinh(left)
    if op == 'cosh': return sp.cosh(left)
    if op == 'tanh': return sp.tanh(left)
    if op == 'atan': return sp.atan(left)
    if op == 'sqrt': return sp.sqrt(left)
    raise ValueError(f"Unknown operator: {op!r}")


def _worker(expr_str: str, result_queue):
    """Run in a child process: parse expr string, simplify, put result."""
    try:
        expr = sp.sympify(expr_str)
        simplified = sp.simplify(expr)
        simplified = sp.nsimplify(simplified, rational=False, tolerance=1e-8)
        result_queue.put(str(simplified))
    except Exception:
        # Fall back to cheap expand so we still return something.
        try:
            result_queue.put(str(sp.expand(sp.sympify(expr_str))))
        except Exception:
            result_queue.put(None)


def _sympy_simplify_with_timeout(expr: sp.Expr, timeout_s: float) -> Optional[str]:
    """
    Simplify a sympy expression in a child process within `timeout_s` seconds.
    Returns the simplified string, or None on timeout / error.
    Safe to call from any thread.
    """
    expr_str = str(expr)
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker, args=(expr_str, q), daemon=True)
    p.start()
    p.join(timeout=timeout_s)

    if p.is_alive():
        p.terminate()
        p.join()
        return None

    return q.get_nowait() if not q.empty() else None


def simplify_eml_tree(node, timeout_s: float = _TIMEOUT_S) -> str:
    """
    Convert a ga-lib Node tree to a simplified human-readable string.

    Returns a clean sympy string (e.g. "exp(x)", "x**2 + 2*x + 1").
    Falls back to node.to_formula() if conversion or simplification fails.
    Safe to call from any thread or process.
    """
    try:
        expr_raw = _node_to_sympy(node)
    except Exception:
        return node.to_formula()

    result = _sympy_simplify_with_timeout(expr_raw, timeout_s)
    if result is not None:
        return result

    # Timeout fallback: cheap local expand (no subprocess, no simplify).
    try:
        return str(sp.expand(expr_raw))
    except Exception:
        return node.to_formula()

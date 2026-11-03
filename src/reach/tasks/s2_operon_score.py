"""Operon scoring in double precision (D056; tool T20). Pure numpy: imported by the Operon task (.venv-ext) and by tests.

Operon predicts in float32 and appends a fitted scale and intercept, so its raw predictions never match a target to 9
digits. A model is scored exact when its expression (printed with 17 digits) re-evaluated in double precision, with
the scale and intercept refitted by least squares in double, hashes like the target with the target's scale."""
import hashlib

import numpy as np

_FUN = {"exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos, "sqrt": np.sqrt, "square": np.square,
        "abs": np.abs}


def evaluate_model(s, x):
    """Evaluate an Operon infix string on x (float64 array) with numpy; X1 is the variable; ^ is a power."""
    ns = dict(_FUN, X1=np.asarray(x, float), __builtins__={})
    with np.errstate(all="ignore"):
        try:
            return np.asarray(eval(s.replace("^", "**"), ns), float) * np.ones(len(x))    # noqa: S307 (trusted output)
        except (ZeroDivisionError, OverflowError, ValueError):
            return np.full(len(x), np.nan)        # plain-Python constant arithmetic failed: undefined, not exact (row 74)


def scaled_hash(vec, scale, precision=9):
    """Identical to reach.grammar.scaled_func_hash (tested)."""
    with np.errstate(all="ignore"):
        v = np.asarray(vec, dtype=np.complex128) / scale
        r, i = np.round(v.real, precision) + 0.0, np.round(v.imag, precision) + 0.0
        r[~np.isfinite(r)] = np.nan
        i[~np.isfinite(i)] = np.nan
    return hashlib.sha1(r.tobytes() + i.tobytes()).hexdigest()


def is_exact(model_str, target, scale, x=None):
    """target: complex values on the grid (x defaults to the 100-point grid on (2, 3))."""
    if x is None:
        x = np.array([2.0 + (i + 1) / 101 for i in range(100)])
    h = evaluate_model(model_str, x)
    t = np.asarray(target, complex)
    # complex only if the imaginary part survives the hash's rounding (float noise of 1e-16 does not; review 2, row 66)
    if not np.isfinite(h).all() or np.any(np.round(t.imag / scale, 9) != 0):
        return False
    A = np.column_stack([np.ones_like(h), h])
    coef, *_ = np.linalg.lstsq(A, t.real, rcond=None)
    return scaled_hash(A @ coef, scale) == scaled_hash(t, scale)

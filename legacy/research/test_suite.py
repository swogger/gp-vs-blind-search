import cmath
import math
import statistics as _stats

# ---------------------------------------------------------------------------
# EML K-complexity for each standard-GP operator token.
#
# Source: eml-skill witness library (arXiv:2603.21852, Table 4 + closure page).
# K is the number of nodes (operators + leaves) in the minimal EML tree that
# implements that operator.  Standard GP costs 1 node per operator by definition,
# so the ratio EML_K[op] / 1 is the exact structural penalty EML pays.
#
# Used by eml_plateau_mult() to derive a principled patience multiplier.
# ---------------------------------------------------------------------------
EML_K: dict = {
    '+':    19,   # add       (proven minimal)
    '-':    11,   # sub       (proven minimal)
    '*':    17,   # mult      (proven minimal)
    '/':    17,   # div
    'exp':   3,   # exp       (proven minimal, Axiom [2])
    'log':   7,   # ln        (proven minimal, Axiom [3])
    'sqrt': 59,   # sqrt      (upper bound; paper direct-search K=43)
    'sin':  351,  # sin       (upper bound, i-cascade optimised)
    'cos':  269,  # cos       (upper bound, i-cascade optimised)
    'tan':  651,  # tan       (upper bound)
    'sinh':  81,  # sinh      (upper bound)
    'cosh':  89,  # cosh      (upper bound)
    'tanh': 201,  # tanh      (upper bound)
    'atan': 355,  # arctan    (upper bound)
}


def eml_plateau_mult(std_ops: list) -> int:
    """
    Compute the EML patience multiplier for a given standard-GP operator set.

    Rationale
    ---------
    Standard GP pays 1 node per operator; EML pays EML_K[op] nodes for the
    same semantic primitive.  The mean EML K across the operator set is the
    average structural work EML must do per operator application — a natural
    measure of how much harder the search space is for EML relative to
    standard GP on the same target.

    We use the *mean* (not max) because:
      • Not every solution path requires the most expensive operator.
      • The max (e.g. tan=651) would give absurdly large patience values
        on tests that merely include tan as an option.

    The result is rounded to the nearest integer and floored at 1.

    Examples (rounded)
    ------------------
      Phase 1 (arith + exp/log) : mean K ≈ 12  → mult = 12
      Phase 2 (arith + trig)    : mean K ≈ 191 → mult = 191
      Phase 3 (arith + hyperbolic): mean K ≈ 49 → mult = 49
    """
    costs = [EML_K[op] for op in std_ops if op in EML_K]
    if not costs:
        return 1
    return max(1, round(_stats.mean(costs)))


# ---------------------------------------------------------------------------
# Safe wrappers — return complex inf on domain errors so fitness stays finite.
# ---------------------------------------------------------------------------

def safe_exp(x):
    try: return cmath.exp(x)
    except: return complex(float('inf'), float('inf'))

def safe_log(x):
    try: return cmath.log(x) if x != 0j else complex(float('-inf'), 0)
    except: return complex(float('inf'), float('inf'))

def safe_sin(x):
    try: return cmath.sin(x)
    except: return complex(float('inf'), float('inf'))

def safe_cos(x):
    try: return cmath.cos(x)
    except: return complex(float('inf'), float('inf'))

def safe_sinh(x):
    try: return cmath.sinh(x)
    except: return complex(float('inf'), float('inf'))

def safe_cosh(x):
    try: return cmath.cosh(x)
    except: return complex(float('inf'), float('inf'))

def safe_tanh(x):
    try: return cmath.tanh(x)
    except: return complex(float('inf'), float('inf'))

def safe_arctan(x):
    try: return cmath.atan(x)
    except: return complex(float('inf'), float('inf'))

def safe_sqrt(x):
    try: return cmath.sqrt(x)
    except: return complex(float('inf'), float('inf'))

# ---------------------------------------------------------------------------
# Target functions
# ---------------------------------------------------------------------------

def target_1_1(x):   return safe_exp(x)
def target_1_2(x):   return safe_log(x)
def target_1_3(x1, x2): return x1 + x2

def target_1_4(x):
    return safe_exp(x) * safe_exp(-x**2)

def target_2_1(x):   return (x**3) - (2*(x**2)) + x
def target_2_2(x):   return safe_sin(x) + safe_cos(x)

def target_3_1(x):   return safe_sinh(x)
def target_3_2(x):   return safe_arctan(x)
def target_3_3(x):
    try: return x**x
    except: return complex(float('inf'), float('inf'))
def target_3_4(x):
    try: return 1.0 / (1.0 + safe_exp(-x))
    except: return complex(float('inf'), float('inf'))

def target_4_1(x):
    return safe_exp(safe_sin(x)) + safe_log((x**2) + 1.0)
def target_4_2(x1, x2):
    return safe_sin(x1 * safe_exp(x2)) - safe_cos(safe_log((x1**2) + (x2**2) + 0.1))

# ---------------------------------------------------------------------------
# Standard-GP operator sets — full primitive set so standard GP is not
# artificially handicapped.  Each phase unlocks operators appropriate to
# its complexity tier, giving standard GP everything it could need while
# keeping the comparison honest (EML uses only 'eml').
#
# Operator tokens understood by Node.evaluate():
#   arithmetic : '+', '-', '*', '/'
#   exponential: 'exp', 'log'
#   trig       : 'sin', 'cos', 'tan'       (unary, left-child only)
#   hyperbolic : 'sinh', 'cosh', 'tanh'    (unary, left-child only)
#   inverse trig: 'atan'                   (unary)
#   power      : 'sqrt'                    (unary)
# ---------------------------------------------------------------------------

_ARITH  = ['+', '-', '*', '/']
_EXP    = ['exp', 'log']
_TRIG   = ['sin', 'cos', 'tan']
_HTRIG  = ['sinh', 'cosh', 'tanh']
_INV    = ['atan']
_SQRT   = ['sqrt']

TEST_MATRIX = {
    "1.1": {
        "name":    "Phase 1: f(x) = exp(x)",
        "func":    target_1_1,
        "vars":    ["x"],
        "range":   (-3.0, 3.0),
        "std_ops": _ARITH + _EXP,
    },
    "1.2": {
        "name":    "Phase 1: f(x) = ln(x)",
        "func":    target_1_2,
        "vars":    ["x"],
        "range":   (0.1, 5.0),
        "std_ops": _ARITH + _EXP,
    },
    "1.3": {
        "name":    "Phase 1: f(x1, x2) = x1 + x2",
        "func":    target_1_3,
        "vars":    ["x1", "x2"],
        "range":   (-5.0, 5.0),
        "std_ops": _ARITH,
    },
    "1.4": {
        "name":    "Phase 1: f(x) = exp(x) * exp(-x^2)  [EML home territory]",
        "func":    target_1_4,
        "vars":    ["x"],
        "range":   (-2.0, 2.0),
        "std_ops": _ARITH + _EXP,
    },
    "2.1": {
        "name":    "Phase 2: f(x) = sin(x) + cos(x)",
        "func":    target_2_2,
        "vars":    ["x"],
        "range":   (-3.14159, 3.14159),
        "std_ops": _ARITH + _TRIG,
    },
    "3.1": {
        "name":    "Phase 3: f(x) = sinh(x)",
        "func":    target_3_1,
        "vars":    ["x"],
        "range":   (-3.0, 3.0),
        "std_ops": _ARITH + _EXP + _HTRIG,
    },
    "3.2": {
        "name":    "Phase 3: f(x) = arctan(x)",
        "func":    target_3_2,
        "vars":    ["x"],
        "range":   (-10.0, 10.0),
        "std_ops": _ARITH + _EXP + _TRIG + _INV,
    },
    "3.3": {
        "name":    "Phase 3: f(x) = x^x",
        "func":    target_3_3,
        "vars":    ["x"],
        "range":   (0.1, 3.0),
        "std_ops": _ARITH + _EXP + _SQRT,
    },
    "3.4": {
        "name":    "Phase 3: f(x) = 1 / (1 + exp(-x))  [sigmoid]",
        "func":    target_3_4,
        "vars":    ["x"],
        "range":   (-5.0, 5.0),
        "std_ops": _ARITH + _EXP,
    },
    "3.5": {
        "name":    "Phase 3: f(x) = x^3 - 2x^2 + x  [EML stress: integer powers]",
        "func":    target_2_1,
        "vars":    ["x"],
        "range":   (-5.0, 5.0),
        "std_ops": _ARITH,
    },
    "4.1": {
        "name":    "Phase 4: f(x) = exp(sin(x)) + ln(x^2 + 1)",
        "func":    target_4_1,
        "vars":    ["x"],
        "range":   (-3.0, 3.0),
        "std_ops": _ARITH + _EXP + _TRIG,
    },
    "4.2": {
        "name":    "Phase 4: f(x1, x2) = sin(x1*exp(x2)) - cos(ln(x1^2+x2^2+0.1))",
        "func":    target_4_2,
        "vars":    ["x1", "x2"],
        "range":   (-2.0, 2.0),
        "std_ops": _ARITH + _EXP + _TRIG,
    },
}

"""Study 2 function sets: the single source of truth for names, domains, roles and enumeration roots
(docs/design/function_sets.md; decisions D048 C9, D052, D053).

Roles: anchor and control sets are always calibration; size_check sets test the law at 6 inputs (always calibration);
eligible sets enter the held-out draw (2 per domain). Random held-out sets from an operator universe (D053) are
generated later by tool T26 and are not listed here.
"""
ENUM = "data/enumeration_s2"
ROLES = ("anchor", "control", "size_check", "eligible")


def _s(domain, role, root=ENUM):
    return dict(domain=domain, role=role, root=root)


SETS = {
    # real-valued (one variable on (2, 3); guarded semantics)
    "F-EML": _s("real", "anchor"),
    "F-EML-R": _s("real", "control"),                 # redundancy control of F-EML (D052)
    "R-ARITH": _s("real", "eligible"),
    "R-ARITH4": _s("real", "eligible"),
    "R-ELEM6": _s("real", "eligible", f"{ENUM}/cap16"),   # ceiling raised with a 16 GB cap (D053)
    "R-ELEM4": _s("real", "eligible"),
    "R-HYB": _s("real", "eligible"),
    "R-TRIG": _s("real", "eligible", f"{ENUM}/cap16"),    # ceiling raised with a 16 GB cap (D053)
    "F-LDE0": _s("real", "eligible"),
    "R-SRB": _s("real", "eligible"),                  # kept only if its proven ceiling is >= 10 (D052)
    # Boolean (n = 5 unless stated)
    "BOOL-NAND-n5": _s("bool", "anchor"),
    "BOOL-NOR-n5": _s("bool", "control"),             # duality control
    "BOOL-IMP0-n5": _s("bool", "control"),            # partner of the redundancy control (D052)
    "BOOL-IMP0R-n5": _s("bool", "control"),           # redundancy control (D052)
    "BOOL-AON-n5": _s("bool", "eligible"),
    "BOOL-AN-n5": _s("bool", "eligible"),
    "BOOL-AONX-n5": _s("bool", "eligible"),
    "BOOL-AX1-n5": _s("bool", "eligible"),
    "BOOL-RICH-n5": _s("bool", "eligible"),
    "BOOL-NAND-n6": _s("bool", "size_check"),         # problem-size contrast (D052, D053)
    "BOOL-AON-n6": _s("bool", "size_check"),
}


def eligible(domain):
    return [n for n, s in SETS.items() if s["domain"] == domain and s["role"] == "eligible"]


def calibration_only(domain):
    return [n for n, s in SETS.items() if s["domain"] == domain and s["role"] != "eligible"]

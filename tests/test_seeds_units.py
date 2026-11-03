"""T1 seed stream and T5 size units (written before the code; REQ-RUN-09, REQ-MEA-09, REQ-MEA-21)."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from reach import grammar as G

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ T1
def test_s2_seed_slices_are_disjoint_and_new():
    from reach import seeds as S
    assert len(set(S.S2_SEEDS)) == len(S.S2_SEEDS) == 2000
    used = [set(S.s2_seeds(name)) for name in S.S2_SLICES]
    for i in range(len(used)):
        for j in range(i + 1, len(used)):
            assert not used[i] & used[j]
    assert not set(S.S2_SEEDS) & set(S.SEEDS)                                   # paper 1's stream
    import importlib.util
    spec = importlib.util.spec_from_file_location("legacy_seeds", ROOT / "legacy/research/seeds.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert not set(S.S2_SEEDS) & {s for s in m.SEEDS if s is not None}           # EML-GP's stream
    assert len(S.s2_seeds("calibration")) == 32 and S.s2_seeds("calibration")[:5] == S.s2_seeds("main")


# ------------------------------------------------------------------ T5
def _levels(us, start=1):
    cum, out = 0, []
    for k, u in enumerate(us, start):
        cum += u
        out.append(dict(K=k, U=u, U_cum=cum))
    return out


def test_bits_syn_and_uniform_match_definitions():
    from reach.units import SizeUnits
    g = G.get("EML")                                        # a0 = 2, binary 1: N = 2, 0, 2^2.., table C1: 2, 4, 16, 80
    su = SizeUnits(g, _levels([2, 0, 3, 0, 10]))
    n = G.syntactic_counts(g, 7)
    assert su.bits_syn(7) == pytest.approx(math.log2(sum(n[1:8])))
    assert su.bits_uniform(7) == pytest.approx(7 * math.log2(3))     # alphabet: x, 1, eml


def test_bits_sem_counts_and_noise_correction():
    from reach.units import SizeUnits
    lv = _levels([2, 0, 3, 0, 10])
    su = SizeUnits(G.get("EML"), lv)
    assert su.bits_sem(5) == pytest.approx(math.log2(15)) and su.bits_sem(4) == pytest.approx(math.log2(5))
    noise = [dict(K=5, noise=5, U=10)]
    su2 = SizeUnits(G.get("EML"), lv, noise_rows=noise)
    assert su2.bits_sem(5) == pytest.approx(math.log2(10))          # 15 - 5 duplicates
    assert su2.bits_sem(5, corrected=False) == pytest.approx(math.log2(15))


def test_bits_sem_extrapolates_above_the_ceiling_and_flags_it():
    from reach.units import SizeUnits
    us = [int(3 * 2 ** k) for k in range(1, 16)]                    # exact geometric growth, 1 bit per node
    su = SizeUnits(G.get("EML"), _levels(us))
    v, extrapolated = su.bits_sem(20, with_flag=True)
    assert extrapolated and not su.bits_sem(10, with_flag=True)[1]
    assert v == pytest.approx(math.log2(sum(us) + sum(3 * 2 ** k for k in range(16, 21))), abs=0.05)


def test_extrapolation_predicts_real_enumeration_levels():
    """Validation on real data: fit on F-EML up to ceiling - 4, predict the last 4 levels' log2 U_cum within 0.5 bit."""
    from reach.units import SizeUnits
    p = ROOT / "data/enumeration_s2/F-EML/counts.json"
    if not p.exists():
        pytest.skip("F-EML enumeration not present")
    levels = json.loads(p.read_text())["levels"]
    top = max(st["K"] for st in levels)
    su = SizeUnits(G.get("F-EML"), [st for st in levels if st["K"] <= top - 4])
    for st in levels:
        if st["K"] > top - 4 and st["U"] > 0:
            assert abs(su.bits_sem(st["K"], corrected=False) - math.log2(st["U_cum"])) < 0.5, st["K"]

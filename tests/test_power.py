"""Power check for H2 (Stage 1 rationale): a generous margin gives high power, a tiny one low power, and a shifted set
lowers it."""
import numpy as np

from reach.analysis.s2 import power as W

SETS = {f"S{i}": [2.0 * k for k in range(2, 12)] for i in range(4)}


def test_power_rises_with_the_margin_and_falls_with_a_shift(monkeypatch):
    monkeypatch.setattr(W, "TARGETS", 8)
    monkeypatch.setattr(W, "SEEDS", 2)
    lo = W.power(SETS, a=6.0, b=0.6, delta=1e-5, sims=4, boot=40)
    hi = W.power(SETS, a=6.0, b=0.6, delta=0.5, sims=4, boot=40)
    assert lo["power"] == 0.0 and hi["power"] == 1.0
    sh = W.power(SETS, a=6.0, b=0.6, delta=0.5, sims=4, boot=40, shift=("S0", 6.0))
    assert sh["upper95_median"] > hi["upper95_median"]

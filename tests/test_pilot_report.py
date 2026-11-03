"""Pilot report on simulated phase data with a known answer (T16; lesson 19: validate the instrument first)."""
import json
import math

import numpy as np
import pytest

from reach.analysis.s2 import pilot_report as R


def _fake_phase(root, kappa=0.6, alpha=0.5, c=2.0, t0=4.0, seed=1):
    """Weibull run times with a warm-up of t0 generations: evolution cannot hit before generation 1 (theory.fit)."""
    rng = np.random.default_rng(seed)
    rows = []
    for K in range(6, 28, 2):
        for t in range(10):
            for s in range(3):
                lam = math.exp(c - alpha * K * math.log(2))
                T = t0 + (-math.log(rng.random()) / lam) ** (1 / kappa)
                g = math.ceil(T) - 1
                ok = g < 800
                rows.append(dict(key=f"{K}|{t}|{s}", set="BOOL-NAND-n5", target_id=f"K{K}_{t}", seed=s, K_construction=K,
                                 K_proven=K, units=dict(nodes=float(K), bits_sem=float(K)), algorithm="A", config="base",
                                 pop_size=100, generations=800, gen_exact=g if ok else None,
                                 gens_reached=g if ok else 800, terminal_state="converged" if ok else "budget_exhausted",
                                 cpu_s=1.0))
    d = root / "data/runs/s2_pilot_reach_a"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(dict(n_tasks=len(rows), tasks=[])))
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_report_recovers_known_reach_and_slope(tmp_path):
    _fake_phase(tmp_path)
    res = R.build(tmp_path)
    e = res["sets"]["BOOL-NAND-n5"]
    truth = (2.0 + 0.6 * math.log(200 - 4.0) - math.log(math.log(2))) / (0.5 * math.log(2))   # 16.0 nodes at E0
    assert abs(e["nodes"][1]["k50"] - truth) < 1.5
    assert e["nodes_per_doubling"] == pytest.approx(0.6 / 0.5, abs=0.3)
    assert e["model"]["kappa"] == pytest.approx(0.6, abs=0.1) and e["model"]["alpha"] == pytest.approx(0.5, abs=0.1)
    assert res["status"]["s2_pilot_reach_a"]["done"] == 330
    text = R.render(res)
    assert "BOOL-NAND-n5" in text and "Proposals for gate G2" in text

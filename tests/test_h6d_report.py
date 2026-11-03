"""H6d report driver and negative control (D098; REQ-ANA-88, REQ-RUN-40). Written before the code."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from reach import grammar as G
from reach.analysis.s2 import h6d_report as D
from reach.enumerate import Enumerator

ROOT = Path(__file__).resolve().parents[1]


def test_negative_control_of_a_blind_sampler_has_no_guidance():
    """A blind sampler whose hit probability on every target equals its coverage of the target's level has G = 0;
    one that hits its targets more often than its coverage (a planted premium) does not."""
    U = {k: 2 ** k for k in range(1, 15)}
    cov = {k: 1 / (1 + math.exp(math.log2(2 ** (k + 1) - 2) - 9.0)) for k in U}
    lv = [0] + [cov[k] * U[k] for k in U]
    pre = {str(p): dict(n=p, d=5000 * p // 20000, undef=0, lv=[x * p / 20000 for x in lv], lv_ok=[x * p / 20000 for x in lv])
           for p in (5000, 10000, 20000)}                                      # every prefix, as the task records
    rows = [dict(rv=dict(prefixes=pre, stop=pre["20000"]))] * 3
    targets = [dict(target_id=f"t{k}_{i}", K_proven=k, units=dict(bits_sem=math.log2(2 ** (k + 1) - 2)))
               for k in range(3, 15) for i in range(10)]
    N = 10 ** 6
    hits = {t["target_id"]: N * (1 - (1 - cov[t["K_proven"]]) ** (1 / 20000)) for t in targets}
    r = D.control_set(rows, targets, hits, N, U, U, 20000)
    assert r["G"] == pytest.approx(0, abs=1e-6)
    boosted = {k: N * (1 - (1 - min(0.999, 3 * cov[int(k.split("_")[0][1:])])) ** (1 / 20000)) for k in hits}
    assert D.control_set(rows, targets, boosted, N, U, U, 20000)["G"] > 0.5


def _enum(tmp_path, name, kmax):
    g = G.get(name)
    d = tmp_path / name
    Enumerator(g, d, log=lambda *_: None).run(kmax)
    return g, d


def test_rand_task_counts_the_baseline_draws(tmp_path):
    """The task shows the counter exactly the trees `draw_trees` yields for the same seed: n draws, the same distinct
    keys (an independent count here), nothing left out as a target."""
    from reach import revisit as V
    from reach.tasks.s2_random import draw_trees
    g, d = _enum(tmp_path, "BOOL-NAND-n3", 9)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))                                     # guardrail row 88
    pr = subprocess.run([sys.executable, "-m", "reach.tasks.s2_revisit_rand", "--set", g.name, "--enum_dir", str(d),
                         "--seed", "11", "--n", "5000"], cwd=ROOT, capture_output=True, text=True, env=env)
    assert pr.returncode == 0, pr.stderr[-500:]
    row = json.loads(next(l for l in pr.stdout.splitlines() if l.startswith("RESULT "))[7:])
    key = V.key_function(g)
    keys = [key(t)[0] for t in draw_trees(g, 5000, 11)]
    assert row["rv"]["stop"]["n"] == 5000 and row["rv"]["stop"]["d"] == len(set(keys))
    assert sum(row["rv"]["stop"]["lv"]) == len(set(keys)) and row["rv"]["target_first"] is None
    assert set(row["rv"]["prefixes"]) == {"5000"}


def test_rand_phase_covers_every_full_set():
    from reach.runner import phases as P
    tasks = P.PHASES["s2_revisit_rand"]["build"]()
    sets = {t["meta"]["set"] for t in tasks}
    assert sets == set(P.FULL_SETS) and len(tasks) == 25 * len(sets)
    seeds = [int(t["argv"][t["argv"].index("--seed") + 1]) for t in tasks]
    assert len(set(seeds)) == len(seeds) and min(seeds) >= 30_000_000
    assert all(t["argv"][t["argv"].index("--n") + 1] == "20000" for t in tasks)


def _phase(root, name, tasks, rows):
    d = root / f"data/runs/{name}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(dict(phase=name, n_tasks=len(tasks), tasks=tasks)))
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _synthetic_root(tmp_path, differ=False):
    """Two sets per domain, each 20 targets x 3 seeds; originals in s2_pilot_reach_a, re-runs in s2_revisit_pilot2."""
    rng = np.random.default_rng(3)
    U = {k: 2 ** k for k in range(1, 16)}
    bits = {k: math.log2(sum(U[j] for j in U if j <= k)) for k in U}
    origs, again, tasks = [], [], []
    sets = {"R-ARITH4": ("real", 9.0, 7.0), "F-EML": ("real", 9.5, 7.2), "BOOL-AON-n5": ("bool", 13.0, 7.1),
            "BOOL-IMP0-n5": ("bool", 12.5, 6.9)}
    for g, (dom, k50, mid) in sets.items():
        for i in range(20):
            k = int(rng.integers(3, 16))
            for s in range(3):
                hit = rng.random() < 1 / (1 + math.exp(1.2 * (bits[k] - k50)))
                gen = int(rng.integers(0, 799)) if hit else None
                key = f"pilot_reach|{g}|{g}_K{k:02d}_{i:02d}|A|base|800|{s}"
                row = dict(key=key, set=g, domain=dom, target_id=f"{g}_K{k:02d}_{i:02d}", K=k, units=dict(bits_sem=bits[k]),
                           exact=bool(hit), gen_exact=gen, gens_reached=gen if hit else 800, pop_size=100,
                           budget_E=80000, terminal_state="converged" if hit else "budget_exhausted", seed=s)
                n = (gen + 1) * 100 if hit else 80000
                pre = {}
                for p in (5000, 10000, 20000, 40000, 80000):
                    if p <= n:
                        f = math.sqrt(p / 20000)
                        lv = [0] + [min(U[j], round(f * U[j] / (1 + math.exp(bits[j] - mid)))) for j in U]
                        pre[str(p)] = dict(n=p, d=sum(lv) + 5, undef=0, lv=lv, lv_ok=lv)
                stop = dict(pre[max(pre, key=int)] if pre else dict(n=n, d=5, undef=0, lv=[0] * 16, lv_ok=[0] * 16), n=n)
                rv = dict(prefixes=pre, stop=stop, target_first=n if hit else None, cpu_seconds=0.1,
                          lookup=dict(sha256=f"sha-{g}", top=15))
                origs.append(row)
                again.append(dict(row, key=f"revisit|{key}", rv=rv, gen_exact=(gen + 1 if differ and i == 0 and s == 0
                                                                                and hit else gen)))
                tasks.append(dict(key=f"revisit|{key}", meta=dict(audit_of="s2_pilot_reach_a", audit_key=key, set=g)))
    _phase(tmp_path, "s2_pilot_reach_a", [dict(key=r["key"]) for r in origs], origs)
    _phase(tmp_path, "s2_revisit_pilot2", tasks, again)
    return {g: (U, U) for g in sets}


def test_report_refuses_a_row_that_does_not_reproduce(tmp_path):
    levels = _synthetic_root(tmp_path, differ=True)
    with pytest.raises(RuntimeError, match="reproduce"):
        D.build(tmp_path, D.PILOT, boot=20, levels=levels)


def test_report_end_to_end_on_a_synthetic_root(tmp_path):
    levels = _synthetic_root(tmp_path)
    res = D.build(tmp_path, D.PILOT, boot=30, levels=levels)
    assert {"H6d-1", "H6d-2", "per_set", "sensitivities", "checks", "power"} <= set(res)
    assert all(0 <= res["power"][k]["power"] <= 1 and res["power"][k]["delta_80"] > 0 for k in ("H6d-1", "H6d-2"))
    for d in ("real", "bool"):
        for g, t in res["per_set"][d].items():
            assert t["G"] - t["S"] - t["R"] + t["O"] == pytest.approx(t["gap"], abs=1e-9)
    assert res["checks"]["reproduced"] == res["checks"]["rows"] == 240
    D.write(tmp_path, res, D.PILOT)
    j = json.loads((tmp_path / "data/analysis/h6d_pilot.json").read_text())
    assert j["H6d-1"]["delta"] == 1.0 and "not confirmatory" in (tmp_path / "docs/status/h6d_pilot_report.md").read_text()


def test_control_criterion_and_guidance_net_of_the_control():
    """The control passes when each domain's mean blind-sampler G lies within CONTROL_MARGIN of 0 (per-set values are
    reported: they carry the target-sampling bias of uneven hit probabilities within a level); G net of the control is
    reported for the sets the control covers."""
    per = {"real": {"A": dict(G=3.0), "B": dict(G=2.0)}, "bool": {"C": dict(G=20.0), "D": dict(G=18.0)}}
    ctl = {"A": dict(G=0.2), "C": dict(G=-1.7), "D": dict(G=0.3)}
    out = D.control_summary(per, ctl)
    assert out["net"] == {"A": pytest.approx(2.8), "C": pytest.approx(21.7), "D": pytest.approx(17.7)}
    assert out["domain"]["real"] == pytest.approx(0.2) and out["domain"]["bool"] == pytest.approx(-0.7)
    assert out["pass"] is (abs(-0.7) <= D.CONTROL_MARGIN and abs(0.2) <= D.CONTROL_MARGIN)
    assert not D.control_summary(per, {"A": dict(G=0.9), "C": dict(G=0.1)})["pass"]

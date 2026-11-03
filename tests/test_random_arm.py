"""The random-sampling arm (REQ-ANA-40 to 42; D080 a-c; guardrail row 105), written before the code. Synthetic data only.

Stage 1 (OSF a3spg), as filed: "s2_cal_rand 48 (4 x 250,000 random trees per full set)"; "H2r: random sampling's hit
probability is invariant in bits_uniform (same criterion as H2)"; "H5: ... B minus random sampling within +-1 node or
positive"; "Runs stopped by their CPU limit count as failures (primary) and are excluded in a sensitivity analysis."
"""
import json
import math

import numpy as np
import pytest

from reach.analysis.s2 import rand as R
from reach.analysis.s2.load import E0

ALPHABETS = {"BOOL-NAND-n5": 4, "BOOL-NOR-n5": 8, "BOOL-AON-n5": 16}


def _targets(g, sizes, per=10):
    lr = math.log2(ALPHABETS[g])
    return {(g, f"{g}_K{K}_{t}"): dict(target_id=f"{g}_K{K}_{t}", K_construction=K, K_proven=K,
                                       units=dict(nodes=float(K), bits_uniform=K * lr, bits_syn=1.5 * K,
                                                  bits_sem=float(K)))
            for K in sizes for t in range(per)}


def _rand_rows(g, targets, log2_e0p, seed, parts=4, n=250_000):
    """Parts of a random-sampling phase: hits ~ Binomial(n, p) per target with E0 * p = 2^log2_e0p(target)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(parts):
        hits = {}
        for (gg, tid), t in targets.items():
            if gg == g:
                p = min(1.0, 2.0 ** log2_e0p(t) / E0)
                hits[tid] = int(rng.binomial(n, p))
        rows.append(dict(key=f"{g}|{i}", set=g, terminal_state="completed", n=n, seed=seed + i, hits=hits))
    return rows


def _sizes(g):
    return {"BOOL-NAND-n5": range(5, 16), "BOOL-NOR-n5": range(4, 11), "BOOL-AON-n5": range(3, 9)}[g]


def test_rs_records_sum_parts_and_give_the_e0_probability():
    t = _targets("BOOL-NAND-n5", [5, 7], per=1)
    rows = [dict(key="g|0", set="BOOL-NAND-n5", terminal_state="completed", n=250_000,
                 hits={"BOOL-NAND-n5_K5_0": 30, "BOOL-NAND-n5_K7_0": 0}),
            dict(key="g|1", set="BOOL-NAND-n5", terminal_state="completed", n=750_000,
                 hits={"BOOL-NAND-n5_K5_0": 10, "BOOL-NAND-n5_K7_0": 1}),
            dict(key="g|2", set="BOOL-NAND-n5", terminal_state="error", queue_status="error")]   # excluded
    out = R.records(rows, t)
    recs = {r["target_id"]: r for r in out["by_set"]["BOOL-NAND-n5"]}
    a, b = recs["BOOL-NAND-n5_K5_0"], recs["BOOL-NAND-n5_K7_0"]
    assert a["hits"] == 40 and a["draws"] == 1_000_000 and b["hits"] == 1
    assert a["y"] == pytest.approx(1 - (1 - 40 / 1e6) ** E0, rel=1e-12)          # P(hit within E0 draws)
    assert b["y"] == pytest.approx(1 - (1 - 1 / 1e6) ** E0, rel=1e-12)
    assert a["K"] == 5 and a["units"]["nodes"] == 5.0 and a["units"]["bits_uniform"] == 10.0
    assert a["algorithm"] == "RS" and a["budget"] == 1 and not a["censored"]


def test_rs_part_stopped_by_cpu_limit_is_a_failure_and_left_out_in_sensitivity():
    t = _targets("BOOL-NAND-n5", [5], per=1)
    rows = [dict(key="g|0", set="BOOL-NAND-n5", terminal_state="completed", n=250_000, hits={"BOOL-NAND-n5_K5_0": 50}),
            dict(key="g|1", set="BOOL-NAND-n5", terminal_state="timeout", queue_status="cpu_limit")]
    prim = R.records(rows, t)
    r = prim["by_set"]["BOOL-NAND-n5"][0]
    assert r["draws"] == 500_000 and r["hits"] == 50 and r["censored"] and prim["censored_parts"] == 1
    sens = R.records(rows, t, censored_as_failure=False)
    s = sens["by_set"]["BOOL-NAND-n5"][0]
    assert s["draws"] == 250_000 and not s["censored"] and sens["left_out_parts"] == 1
    assert s["y"] > r["y"]                                       # failures lower the primary estimate


def _arm(law_in):
    targets, rows = {}, []
    for i, g in enumerate(ALPHABETS):
        t = _targets(g, _sizes(g))
        targets.update(t)
        rows += _rand_rows(g, t, law_in(g), 100 + 10 * i)
    return R.records(rows, targets)["by_set"]


def test_h2r_invariant_in_bits_uniform_only_for_kozik_like_data():
    """Positive control: the hit probability falls with bits_uniform on one curve for every set (Kozik-like), so
    bits_uniform is invariant and nodes is not; negative control: one curve in nodes, so bits_uniform is not."""
    kozik = _arm(lambda g: (lambda t: 20.0 - t["units"]["bits_uniform"]))
    res = R.h2r(kozik, list(ALPHABETS), 0.01, ("nodes", "bits_uniform"), boot=60)
    assert res["status"] == "tested" and res["unit"] == "bits_uniform"
    assert res["holds"] and res["upper95"] <= 0.01 and res["upper95"] == res["units"]["bits_uniform"]["upper95"]
    assert not res["units"]["nodes"]["invariant"]
    by_nodes = _arm(lambda g: (lambda t: 14.0 - 2.0 * t["units"]["nodes"]))
    neg = R.h2r(by_nodes, list(ALPHABETS), 0.01, ("nodes", "bits_uniform"), boot=60)
    assert not neg["holds"] and neg["units"]["nodes"]["invariant"]
    few = R.h2r(kozik, list(ALPHABETS)[:2], 0.01, ("bits_uniform",), boot=10)
    assert few["status"].startswith("not testable") and not few["holds"]


def _b_recs(g, targets, k50, seed, seeds=5):
    rng = np.random.default_rng(seed)
    out = []
    for (gg, tid), t in targets.items():
        p = 1 / (1 + math.exp(1.5 * (t["K_construction"] - k50)))
        out += [dict(set=g, target_id=tid, algorithm="B", seed=s, K=t["K_construction"], y=int(rng.random() < p),
                     budget=1) for s in range(seeds)]
    return out


def test_b_minus_rs_is_paired_by_target_and_inverts_the_minus_one_bound():
    g = "BOOL-NAND-n5"
    t = _targets(g, range(5, 16), per=12)
    rs = R.records(_rand_rows(g, t, lambda x: 20.0 - x["units"]["bits_uniform"], 7), t)["by_set"][g]
    k_rs = 20.0 / 2 - math.log2(math.log(2)) / 2                       # E0 p = 2^(20 - 2K): P = 1/2 at this K
    for shift, supported in ((3.0, True), (-0.5, True), (-3.0, False)):     # -0.5: within +-1 node, not above 0
        c = R.b_minus_rs(_b_recs(g, t, k_rs + shift, 3), rs, boot=300, margin=1.0)
        assert c["diff"] == pytest.approx(shift, abs=0.8) and c["targets"] == len(t)
        assert (c["p"] < 0.05) == (c["lo"] > -1.0) == supported
    one = R.b_minus_rs(_b_recs(g, dict(list(t.items())[:60]), k_rs, 3), rs, boot=50, margin=1.0)
    assert one["targets"] == 60                                          # only targets with both arms


def _write(root, phase, rows):
    d = root / "data/runs" / phase
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(dict(n_tasks=len(rows), tasks=[dict(key=r["key"]) for r in rows])))
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_confirm_reports_h2r_and_b_minus_rs_in_the_h5_family(tmp_path):
    from reach.analysis.s2 import confirm as C
    from test_dryrun import SETS, _rows
    a = [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0, i)]
    _write(tmp_path, "s2_cal_a16", a)
    _write(tmp_path, "s2_cal_mech", _rows("BOOL-NAND-n5", "B", 0.5, 9))
    rows = []
    for i, g in enumerate(SETS):
        tg = {}
        for r in a:
            if r["set"] == g:
                tg[(g, r["target_id"])] = dict(target_id=r["target_id"], K_construction=r["K_construction"],
                                               K_proven=r["K_proven"], units=r["units"])
        d = tmp_path / f"data/targets_s2/{g}"
        d.mkdir(parents=True)
        (d / "calibration.json").write_text(json.dumps(list(tg.values())))
        rows += _rand_rows(g, tg, lambda t: 16.0 - t["units"]["bits_uniform"] / 2, 50 + i)
    _write(tmp_path, "s2_cal_rand", rows)
    res = C.build(dict(C.PHASE_MAPS["calibration"], levels=None), tmp_path, boot=30, allow_incomplete=True)
    h = res["h2r"]["bool"]
    assert h["status"] == "tested" and h["unit"] == "bits_uniform" and set(h["sets"]) == set(SETS)
    assert h["holds"] == (h["upper95"] <= C.REGISTERED["delta"]["bool"]) and set(h["units"]) == set(C.UNITS)
    c = res["h5"]["BOOL-NAND-n5"]["B-RS"]
    assert c["test"] == "95% CI of K50(B) - K50(RS) above -1 node (paired bootstrap)" and "reps" not in c
    mem = {tuple(mm["id"]): mm for mm in res["holm"]["H5"]["bool"]["members"]}
    assert mem[("BOOL-NAND-n5", "B-RS")]["p"] == c["p"] and mem[("BOOL-NAND-n5", "B-RS")]["untestable"] is None
    assert mem[("BOOL-NOR-n5", "B-RS")]["untestable"] == "missing: no data"            # no B runs there
    text = C.render(res, "t", "s")
    assert "## H2r" in text and "| BOOL-NAND-n5 | B-RS |" in text and len(text.splitlines()) <= 200
    assert not any("H2r" in x or "B - random sampling" in x for x in res["not_run"])


def test_b_minus_rs_p_is_the_two_sided_inversion_at_minus_one(monkeypatch):
    """Known answer: replicates evenly spaced on [-2, 4] put the two-sided CI's lower end above -1 from alpha = 1/3 on
    (a one-sided bound would give 1/6, a bound against 0 would give 2/3)."""
    from reach.analysis.s2 import contrasts
    reps = list(np.linspace(-2.0, 4.0, 101))
    monkeypatch.setattr(contrasts, "paired_k50_difference",
                        lambda *a, **k: dict(diff=1.0, lo=-1.85, hi=3.85, targets=5, dropped=0, estimable=True, reps=reps))
    c = R.b_minus_rs([dict(target_id="t", K=5, y=1)], [], boot=101, margin=1.0)
    assert c["p"] == pytest.approx(1 / 3, abs=1e-9) and "reps" not in c

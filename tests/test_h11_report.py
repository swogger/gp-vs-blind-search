"""H11 driver (REQ-ANA-96; guardrail row 156): completeness, target checksums, error/killed/timeout rows, end to end on a
synthetic root. Written before the code."""
import hashlib
import json
import math

import pytest

from reach import s2_lock as L
from reach.analysis.s2 import h11_report as H
from reach.analysis.s2 import law, remedy


def _row(key, tid, gen_exact=None, state="budget_exhausted", gens=100, **kw):
    r = dict(key=key, set="R-ARITH", target_id=tid, generations=100, gens_reached=gens, terminal_state=state,
             gen_exact=gen_exact)
    r.update(kw)
    return r


def test_refuses_an_incomplete_or_extra_phase():
    rows = [_row(f"k{i}", "t") for i in range(3)]
    reg = dict(n_tasks=3, sha256=L.task_list_hash(rows))
    H.check_complete(rows, reg)
    with pytest.raises(RuntimeError):
        H.check_complete(rows[:2], reg)                                    # a missing row
    with pytest.raises(RuntimeError):
        H.check_complete(rows + [_row("k9", "t")], reg)                    # an extra row
    with pytest.raises(RuntimeError):
        H.check_complete(rows[:2] + [_row("other", "t")], reg)             # same count, other keys


def _write_targets(root, g, name, targets):
    d = root / "data/targets_s2" / g
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.json"
    p.write_text(json.dumps(targets))
    (d / f"{name}.sha256").write_text(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n")
    return p


def _targets(levels, per=4):
    return [dict(target_id=f"R-ARITH_K{K:02d}_{i:02d}", K_construction=K, K_proven=K, constructed=False,
                 units=dict(bits_sem=b, nodes=float(K)))
            for K, b in levels for i in range(per)]


def test_target_files_are_checked(tmp_path):
    _write_targets(tmp_path, "R-ARITH", "heldout", _targets([(9, 8.0), (11, 10.0)]) +
                   [dict(target_id="R-ARITH_K27_00", K_construction=27, K_proven=23, constructed=True,
                         units=dict(bits_sem=24.6))])
    sizes = H.load_sizes(tmp_path, ["R-ARITH"])
    assert set(sizes["R-ARITH"].values()) == {8.0, 10.0}                  # constructed level left out
    p = tmp_path / "data/targets_s2/R-ARITH/heldout.json"
    p.write_text(p.read_text().replace("8.0", "8.5"))                      # tampered file
    with pytest.raises(RuntimeError):
        H.load_sizes(tmp_path, ["R-ARITH"])


def test_error_killed_and_timeout_rows():
    runs = [_row("a", "t", gen_exact=10, state="converged", gens=10),     # exact at generation 10
            _row("b", "t", state="error"),                                 # error: left out of n
            dict(key="c", set="R-ARITH", target_id="t", terminal_state="killed"),   # no result: a failure
            _row("d", "t", state="timeout", gens=40)]                      # CPU stop before 100: a failure
    s, n = remedy.hits(runs, L=100)
    assert (s, n) == (1, 3)


def test_end_to_end_on_synthetic_runs(tmp_path):
    """A synthetic root with one set: the driver reproduces h11 on the same inputs, refuses an incomplete phase and
    checks the one-run arm against remedy.json."""
    levels = [(9, 8.0), (11, 10.0), (13, 12.0), (15, 14.0), (17, 16.0), (19, 18.0), (21, 20.0), (23, 22.0)]
    tg = _targets(levels)
    _write_targets(tmp_path, "R-ARITH", "heldout", tg)
    rows, one = [], []
    for t in tg:
        b = t["units"]["bits_sem"]
        k = max(0, round(64 * 0.25 * 2 ** (-(b - 10) / 2.0)))              # share of 100-generation runs exact
        for s in range(64):
            rows.append(_row(f"s4|{t['target_id']}|{s}", t["target_id"], gen_exact=5 if s < k else None,
                             state="converged" if s < k else "budget_exhausted", gens=5 if s < k else 100))
        if b <= 16.0:                                                       # one run: budget levels only, 5 seeds
            for s in range(5):
                one.append(dict(key=f"a16|{t['target_id']}|{s}", set="R-ARITH", target_id=t["target_id"],
                                algorithm="A", config="base", generations=3200, pop_size=100, gens_reached=3200,
                                terminal_state="budget_exhausted", K_construction=t["K_construction"],
                                gen_exact=100 if (b < 12 or (b == 12 and s < 2)) else None))
    for name, rr in (("s4_restart", rows), ("s2_ho_a16", one)):
        d = tmp_path / "data/runs" / name
        d.mkdir(parents=True)
        (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rr))
    reg = tmp_path / "data/registration"
    reg.mkdir(parents=True)
    (reg / "stage4_tasks.json").write_text(json.dumps({"s4_restart": dict(n_tasks=len(rows),
                                                                           sha256=L.task_list_hash(rows))}))
    one_k = law._k50([dict(units={"bits_sem": t["units"]["bits_sem"]}, y=int(r["gen_exact"] is not None))
                      for r in one for t in tg if t["target_id"] == r["target_id"]], "bits_sem")
    (tmp_path / "data/analysis").mkdir(parents=True)
    (tmp_path / "data/analysis/remedy.json").write_text(json.dumps(
        {"systems": {"tree GP, 16 x E0": {"R-ARITH": {"k50": one_k}}}}))
    res = H.build(tmp_path, sets=["R-ARITH"], boot=40)
    sizes = {t["target_id"]: t["units"]["bits_sem"] for t in tg}
    by = {}
    for r in rows:
        by.setdefault(r["target_id"], []).append(r)
    want = remedy.restart_k50(by, sizes)
    assert res["h11"]["per_set"]["R-ARITH"]["k50"] == pytest.approx(want["k50"])
    assert res["h11"]["per_set"]["R-ARITH"]["one_run"] == pytest.approx(one_k)
    assert res["h11"]["tests"]["H11-1"]["estimate"] == pytest.approx(want["k50"] - math.log2(320_000))
    assert set(res["descriptive"]["R-ARITH"]["restarts_L100"]) == {"2", "4", "8", "16", "32"}
    # instrument check: a one-run reference that differs is refused
    ref = tmp_path / "data/analysis/remedy.json"
    good = ref.read_text()
    ref.write_text(json.dumps({"systems": {"tree GP, 16 x E0": {"R-ARITH": {"k50": one_k + 0.5}}}}))
    with pytest.raises(RuntimeError):
        H.build(tmp_path, sets=["R-ARITH"], boot=40)
    ref.write_text(good)
    # incomplete phase refused
    (tmp_path / "data/runs/s4_restart/results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]))
    with pytest.raises(RuntimeError):
        H.build(tmp_path, sets=["R-ARITH"], boot=40)

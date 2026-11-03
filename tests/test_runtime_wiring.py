"""The run-time model wired into the confirmatory analysis (REQ-ANA-47; guardrail rows 106 and 107), written before
the code. Synthetic data only."""
import json

import pytest

from reach.analysis.s2 import confirm as C
from test_runtime_model import sim_set

SETS = ("BOOL-NAND-n5", "BOOL-NOR-n5", "BOOL-AON-n5")


class _Stop(Exception):
    pass


def test_runtime_model_is_on_in_both_entry_points(monkeypatch):
    """Tests switch the run-time model off for speed (conftest); the registered default is on and both entry points
    ask for it explicitly, so the confirmatory analysis and the DRYRUN never depend on the default."""
    from reach.analysis.s2 import dryrun as D
    assert C.RUNTIME_MODEL_REGISTERED is True
    seen = []

    def spy(*a, **k):
        seen.append(k.get("runtime_model"))
        raise _Stop
    monkeypatch.setattr(C, "build", spy)
    monkeypatch.setattr(D, "_build", spy)
    for main in (C.main, D.main):
        with pytest.raises(_Stop):
            main([]) if main is C.main else main()
    assert seen == [True, True]


def _write(root, phase, rows):
    d = root / "data/runs" / phase
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(dict(n_tasks=len(rows), tasks=[dict(key=r["key"]) for r in rows])))
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.mark.runtime_model
def test_confirm_reports_the_runtime_model(tmp_path, monkeypatch):
    a16, al = [], []
    for i, (g, kappa) in enumerate(zip(SETS, (0.5, 0.6, 0.7))):
        rows, extra = sim_set(g, kappa, 0.45, 1.0, 2.0, range(4, 14), targets=5, alloc_seeds=6, seed=10 + i)
        a16 += rows
        al += [dict(r, key=r["key"] + "|al") for r in extra]
    _write(tmp_path, "s2_cal_a16", a16)
    _write(tmp_path, "s2_cal_alloc", al)
    calls = []

    def fake_compare(runs, **k):                       # the comparison itself: test_runtime_model.py
        calls.append(sorted({r["set"] for r in runs}))
        return dict(heldout_loglik=dict(weibull=-10.0, lognormal=-12.5), preferred="weibull", scored_folds=5,
                    difference=2.5)
    monkeypatch.setattr(C.theory, "compare_families", fake_compare)
    res = C.build(dict(C.PHASE_MAPS["calibration"], levels=None), tmp_path, boot=12, allow_incomplete=True)
    h = res["h4_model"]["bool"]
    assert sorted(h["sets"]) == sorted(SETS) and calls == [[g] for g in sorted(SETS)]   # per set
    assert h["comparison"]["heldout_loglik"] == dict(weibull=-30.0, lognormal=-37.5)
    assert h["comparison"]["preferred"] == "weibull" and h["comparison"]["sets_preferring_weibull"] == 3
    for g in SETS:
        p = h["sets"][g]["point"]
        assert p["kappa"] < 1 and p["ratio"] == pytest.approx(p["kappa"] / p["alpha"])
        assert set(h["sets"][g]["cells"]) == set(C.H4_BUDGETS) and h["sets"][g]["fit_counts"]["error"] == 0
    t = h["tests"]
    mem = {mm["id"][0]: mm for mm in res["holm"]["H4"]["bool"]["members"]}
    assert mem["H4d"]["p"] == t["H4d"]["p"] and mem["H4d"]["kind"] == "reject"
    assert mem["H4e"]["p"] == t["H4e"]["p"] and mem["H4f"]["kind"] == "retain"
    assert mem["H4c"]["p"] == res["h4"]["bool"]["h4c"]["p"] and mem["H4c"]["kind"] == "reject"   # step C (D081 a)
    assert all(mm["test"] for mm in mem.values())
    assert mem["H4f"]["p"] == t["H4f"]["p"] and all("model_length" in c and "central" in c for c in t["H4f"]["cells"])
    text = C.render(res, "t", "s", runtime_file="r.md")
    assert "## H4d-f: run-time model" in text and len(text.splitlines()) <= 200
    fp = C.families.fmt_p                    # H4f and H4a: the unadjusted p beside the Holm-adjusted one (review)
    assert (f"p {fp(t['H4f']['p'])} (unadjusted), Holm p {fp(mem['H4f']['p_holm'])}: {mem['H4f']['status']}"
            in text)
    assert (f"H4a (retain): p {fp(mem['H4a']['p'])} (unadjusted), Holm p {fp(mem['H4a']['p_holm'])}: "
            f"{mem['H4a']['status']}" in text)
    rt = C.render_runtime(res, "t", "s")
    assert rt.count("\n| BOOL-") == len(SETS) + len(SETS) * len(C.H4_BUDGETS) and len(rt.splitlines()) <= 200
    assert not any("H4d" in x or "lognormal" in x for x in res["not_run"])
    json.dumps(res, default=float)


def test_runtime_model_off_is_labelled_and_untestable(tmp_path):
    from test_dryrun import _rows
    _write(tmp_path, "s2_cal_a16", [r for i, g in enumerate(SETS) for r in _rows(g, "A", 2.0, i)])
    res = C.build(dict(C.PHASE_MAPS["calibration"], levels=None), tmp_path, boot=10, allow_incomplete=True,
                  runtime_model=False)
    mem = {mm["id"][0]: mm for mm in res["holm"]["H4"]["bool"]["members"]}
    assert all(mem[k]["untestable"].startswith("not run in this build") and not mem[k]["supported"]
               for k in ("H4d", "H4e", "H4f"))
    assert "not run in this build" in C.render(res, "t", "s") and "not run" in C.render_runtime(res, "t", "s")

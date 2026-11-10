"""Stage 5 pilot mode (REQ-ANA-129, REQ-ANA-130; never confirmatory): power scaled from the pilot to the registered
size, the key-set checks and their plants, the restart-twin rule check, the re-execution check of a phase that mixes
re-executions and new runs (guardrail row 200), and the pilot report end to end on a synthetic root with a partial
phase. Written with the requirements, before the pilot phases finished."""
import json
import math

import numpy as np
import pytest
from scipy.stats import norm

import _stage5_synth as Y
from reach.analysis.s2 import h6d, stage5
from reach.analysis.s2 import stage5_pilot as P
from reach.analysis.s2 import stage5_report as D
from reach.analysis.s2 import stage5_stats as S


@pytest.fixture(scope="module")
def proot(tmp_path_factory):
    root = tmp_path_factory.mktemp("s5pilot")
    levels, lookup = Y.make_pilot_root(root)
    return root, levels, lookup


def test_pilot_power_is_scaled_from_the_pilot_to_the_registered_size():
    rng = np.random.default_rng(1)
    reps = list(0.3 + 0.4 * rng.standard_normal(500))
    sd = float(np.std(reps, ddof=1))
    t = P.power("H12-1", 0.3, reps, 1_000, 16_000)                                # TOST: Stage 3's `h6d.power`
    want = h6d.power(0.3, reps, 1.0, 1_000, 16_000)
    assert t["se"] == pytest.approx(sd / 4) and t["power"] == want["power"] and t["margin_80"] == want["delta_80"]
    one = P.power("H14-2", -0.2, reps, 1_000, 16_000)                             # one-sided, margin -1
    assert one["se"] == pytest.approx(sd / 4)
    assert one["power"] == pytest.approx(norm.cdf((-0.2 + 1.0) / (sd / 4) - 1.6448536269514722))
    assert one["margin_80"] == pytest.approx(-0.2 - (1.6448536269514722 + 0.8416212335729143) * sd / 4)
    unscaled = P.power("H14-2", -0.2, reps, 1, 1)                                 # plant: no scaling
    assert unscaled["power"] < one["power"]
    c = P.power("H16-1", {"25": 0.2, "1000": -0.4}, [{"25": x, "1000": x - 0.6} for x in reps], 100, 900)
    assert c["power_at_most"] == min(v["power"] for v in c["members"].values())
    s = P.power("H16-2", 1.05, [x + 0.75 for x in reps], 100, 900)                # a TOST around 1
    assert s["power"] == pytest.approx(h6d.power(0.05, [x - 0.25 for x in reps], 0.25, 100, 900)["power"])


def test_pilot_key_checks_fire_on_a_planted_mismatch(proot, tmp_path, monkeypatch):
    root, levels, lookup = proot
    rows = D.read_phase("s5_pilot_keys", root)
    chk, hists = P.key_checks(root, rows, lookup, levels)
    assert chk["ok"] and chk["side_files"] == len(rows) and chk["brute_force"] > 0
    assert len(hists) == len({r["target_id"] for r in rows})
    # plant 1: a side file changed after its row was written
    r0 = rows[0]
    p = root / r0["keys"]["file"]
    keep = p.read_bytes()
    p.write_bytes(keep[:-3] + b"xyz")
    try:
        bad, _ = P.key_checks(root, rows, lookup, levels)
        assert not bad["ok"] and any("differs from its row" in x for x in bad["problems"])
    finally:
        p.write_bytes(keep)
    # plant 2: the counter's counts are not the side file's
    moved = [dict(r) for r in rows]
    moved[0] = json.loads(json.dumps(rows[0]))
    moved[0]["rv"]["stop"]["lv"][3] += 1
    bad, _ = P.key_checks(root, moved, lookup, levels)
    assert any("differ from the counter" in x for x in bad["problems"])
    # plant 3: a key set written against another lookup is refused
    with pytest.raises(RuntimeError, match="another lookup"):
        P.key_checks(root, rows, lambda g: (Y.lookup_levels(), Y.TOP, "other"), levels)
    # plant 4: the registered histogram code drops a run: the independent count disagrees
    from reach.tasks import s5_union
    real = s5_union.histograms
    monkeypatch.setattr(s5_union, "histograms", lambda runs, lv, top, prefixes=s5_union.PREFIXES:
                        real(runs[1:], lv, top, prefixes))
    bad, _ = P.key_checks(root, rows, lookup, levels)
    assert any("histograms differ" in x for x in bad["problems"])
    monkeypatch.setattr(s5_union, "histograms", real)
    # plant 5: an independence estimate in place of the exact union: brute force disagrees
    monkeypatch.setattr(stage5, "union_expected", lambda h, R: sum(h[1:]) * R)
    bad, _ = P.key_checks(root, rows, lookup, levels)
    assert any("brute force" in x for x in bad["problems"])


def test_restart_twin_rule_check(proot):
    root, levels, lookup = proot
    rows = D.read_phase("s5_pilot_keys", root)
    _, hists = P.key_checks(root, rows, lookup, levels)
    tables = {g: D.target_table(root, g, ("pilot",))[0] for g in ("F-EML", "R-ARITH4")}
    kt = P.keys_twins(rows, hists, tables, levels, boot=5)
    assert set(kt["rule"]) == {"F-EML", "R-ARITH4"} and set(kt["rule"]["F-EML"]) == {str(R) for R in P.KEYS_R}
    for g, rule in kt["rule"].items():
        assert rule["1"]["pooled_targets"] <= rule["16"]["pooled_targets"]        # more targets pooled as R grows
        assert len({round(v["union_all"], 9) for v in rule.values() if np.isfinite(v["union_all"])}) <= 1
        # R = 16 of 16 runs: all runs and the union of all runs are the same reading
        assert rule["16"]["all_runs"] == pytest.approx(rule["16"]["union_all"])
    # the primary rule's twin at R = 1 equals the library's restart twin on the no-hit histograms
    g = "R-ARITH4"
    ts = sorted(t for t in hists if hists[t]["set"] == g)
    lib = stage5.restart_twin_shares([dict(level=tables[g][t]["K_construction"],
                                           n_prime=hists[t]["hist"]["10000"]["n_nohit"],
                                           hist={int(K): v for K, v in hists[t]["hist"]["10000"]["nohit"].items()})
                                      for t in ts], 1, levels[g])["shares"]
    k = S.reach([tables[g][t]["units"]["bits_sem"] for t in ts], [lib[tables[g][t]["K_construction"]] for t in ts])
    assert kt["rule"][g]["1"]["primary"] == pytest.approx(k["k50"])
    assert kt["R"] == 16 and kt["H15-1"]["status"] == "ok"


def test_reexecution_check_of_a_mixed_phase(proot, tmp_path):
    root, levels, lookup = proot
    from reach.analysis.s2.audit import revisit_report
    with pytest.raises(KeyError):                       # guardrail row 200: the registered report needs audit_of on all
        revisit_report(root, "s5_pilot_robust")
    r = P.reexec_check(root, "s5_pilot_robust", "s2_pilot_robust")
    n = len([x for x in D.read_phase("s2_pilot_robust", root) if x["config"] == "R2"])
    assert r["planned"] == r["identical"] == n and r["not_reexec"] == n and r["ok"]
    r10 = P.reexec_check(root, "s5_pilot_restart10", "s4_restart")
    assert r10["identical"] == 10 and r10["ok"]
    # plants: a re-executed row differs; a side file changed
    import shutil
    dst = tmp_path / "r"
    shutil.copytree(root, dst)
    p = dst / "data/runs/s5_pilot_robust/results.jsonl"
    lines = p.read_text().splitlines()
    row = json.loads(lines[0])
    lines[0] = json.dumps(dict(row, best_expr="changed"))
    p.write_text("\n".join(lines) + "\n")
    bad = P.reexec_check(dst, "s5_pilot_robust", "s2_pilot_robust")
    assert not bad["ok"] and len(bad["differing"]) == 1
    k = json.loads((dst / "data/runs/s5_pilot_restart10/results.jsonl").read_text().splitlines()[0])["keys"]["file"]
    (dst / k).write_bytes(b"changed")
    bad = P.reexec_check(dst, "s5_pilot_restart10", "s4_restart")
    assert any("side file" in x for d in bad["differing"].values() for x in d)
    assert P.reexec_check(dst, "s5_pilot_nothing", "s4_restart")["status"] == "not planned"


def test_pilot_end_to_end_on_a_synthetic_root(proot, tmp_path):
    root, levels, lookup = proot
    res = P.build(root, boot=8, levels_given=levels, lookup_levels=lookup)
    assert res["partial"] and "PARTIAL" in res["label"]
    g = res["phases"]["s5_pilot_grid"]
    assert g["planned"] == g["rows"] + 5 and not g["complete"]
    assert res["phases"]["s5_pilot_selfree"]["projection_SF"]["phase"] == "s5_selfree"
    assert res["phases"]["s5_pilot_keys"]["projection"]["runs"] == 105_600
    for name in ("H12-1", "H12-2", "H12-3"):
        b = res["fitness"][name]
        assert b["status"] == "ok" and np.isfinite(b["power"]["sd"]) and b["power"]["n_registered"] == \
            P.N_REGISTERED[name]
    # the grid's pairing keeps only the runs present in both arms (the five missing rows leave their base runs out)
    assert res["fitness"]["H12-1"]["n_pilot"] == 2 * res["phases"]["s5_pilot_grid"]["rows"]
    assert res["selfree"]["H13-1"]["status"] == "ok" and res["selfree"]["H13-2"]["status"] == "ok"
    bl = res["selfree"]["blindness"]
    assert bl["ok"] and bl["SF"]["pairs"] > 0 and bl["base"]["agree"] < bl["base"]["pairs"]   # the plant disagrees
    assert res["pop1000"]["H14_2"]["status"] == "ok" and res["pop1000"]["H14_1"]["status"] == "approximate"
    assert res["keys"]["ok"] and res["keys_twins"]["H15-1"]["status"] == "ok"
    assert res["reexec"]["restart10"]["identical"] == 10 and res["reexec"]["robust"]["not_reexec"] > 0
    assert res["robust"]["H16-1"]["status"] == "ok" and set(res["robust"]["twin_reach"]) == {"25", "100", "1000"}
    assert res["h11_2l"]["shared five levels"]["conjunction"] < res["h11_2l"]["registered"]["conjunction"]
    D.write(tmp_path, res, "stage5_pilot", P.render)
    md = (tmp_path / "docs/status/stage5_pilot.md").read_text()
    assert len(md.splitlines()) <= 200 and "PARTIAL" in md and "not confirmatory" in md
    for s in ("H12-1", "H13-2", "H14-1", "H15-1", "H16-2", "H11-2L", "H15-2", "restart-twin rule", "Target blindness"):
        assert s in md, s
    back = json.loads((tmp_path / "data/analysis/stage5_pilot.json").read_text())
    assert back["phases"]["s5_pilot_grid"]["rows"] == g["rows"]
    # a target whose runs all ended in error leaves both arms of a pilot contrast (never a refusal in the pilot)
    c = P._common(dict(new={"A": {"t1": (1.0, 0.5, 2), "t2": (2.0, 0.1, 2)}}, base={"A": {"t1": (1.0, 0.4, 2)}}))
    assert c == dict(new={"A": {"t1": (1.0, 0.5, 2)}}, base={"A": {"t1": (1.0, 0.4, 2)}})
    # complete phases: no longer partial
    full = tmp_path / "full"
    full.mkdir()
    lv2, lk2 = Y.make_pilot_root(full, drop=0)
    assert not P.build(full, boot=3, levels_given=lv2, lookup_levels=lk2)["partial"]

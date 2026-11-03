"""Confirmatory CGP settings file (D076; REQ-ANA-25, REQ-RUN-28), written before the code. Synthetic pilot records only:
never the real pilot's partial data."""
import hashlib
import json

import pytest

from reach.analysis.s2 import cgp_pilot as CP
from reach.runner import phases as P

E0 = 20_000


def _prow(g, n, tid, seed, evals_exact=None, cpu=1.0, **over):
    r = dict(algorithm="CGP", set=g, n_nodes=n, config=f"n{n}", target_id=tid, seed=seed, budget=16, budget_E=320_000,
             exact=evals_exact is not None, evals_exact=evals_exact, evals_reached=320_000, cpu_s=cpu,
             terminal_state="converged" if evals_exact is not None else "budget_exhausted",
             claim_rejected=False, missed_exact=False, K_construction=5, K_proven=5)
    r.update(over)
    return r


def _pilot(tmp_path, complete=True, real_cpu=5.0):
    """A synthetic pilot phase: real recovery best at 500 nodes, Boolean equal everywhere (-> 100)."""
    ph = tmp_path / "data/runs" / CP.PHASE
    ph.mkdir(parents=True, exist_ok=True)
    rows = []
    for g in CP.SETS["real"]:
        for n in CP.LENGTHS:
            rows += [_prow(g, n, f"t{i}", 1, E0 if (n == 500 or i < 2) else None, cpu=real_cpu + i % 3)
                     for i in range(6)]
    rows += [_prow(g, n, f"t{i}", 1, E0 // 4, cpu=0.15) for g in CP.SETS["bool"] for n in CP.LENGTHS for i in range(6)]
    for i, r in enumerate(rows):
        r["key"] = f"k{i}"
    (ph / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (ph / "manifest.json").write_text(json.dumps(dict(n_tasks=len(rows) + (0 if complete else 3), tasks=[])))
    return ph


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_settings_are_written_only_from_a_complete_pilot(tmp_path):
    ph = _pilot(tmp_path, complete=False)
    CP.main(tmp_path)
    out = tmp_path / CP.SETTINGS
    assert not out.exists() and not out.with_suffix(".sha256").exists()     # provisional choice: no settings file
    _pilot(tmp_path, complete=True)
    res = CP.main(tmp_path)
    st = json.loads(out.read_text())
    assert st["genome_length"] == res["choice"] == {"real": 500, "bool": 100}
    assert st["pilot_phase"] == "s2_pilot_cgp" and st["pilot_results_sha256"] == _sha(ph / "results.jsonl")
    assert st["pilot_manifest_sha256"] == _sha(ph / "manifest.json") and st["pilot_rows"] == res["rows"]
    assert "0.02" in st["rule"] and "shorter" in st["rule"]                 # the rule's text, as fixed in cgp.md
    for dom in ("real", "bool"):
        assert st["soft_cpu_limit_s"][dom] >= 600 and float(st["soft_cpu_limit_s"][dom]).is_integer()
    assert st["levels"] == "all"
    assert out.with_suffix(".sha256").read_text().split()[0] == _sha(out)
    # planted: a missing domain choice (no rows for one Boolean set) writes nothing
    tmp2 = tmp_path / "other"
    ph2 = _pilot(tmp2)
    rows = [json.loads(l) for l in (ph2 / "results.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["set"] != CP.SETS["bool"][0]]
    (ph2 / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (ph2 / "manifest.json").write_text(json.dumps(dict(n_tasks=len(rows), tasks=[])))
    CP.main(tmp2)
    assert not (tmp2 / CP.SETTINGS).exists()


def test_settings_are_never_silently_replaced(tmp_path):
    _pilot(tmp_path)
    CP.main(tmp_path)
    out = tmp_path / CP.SETTINGS
    first = out.read_text()
    CP.main(tmp_path)                                                       # same data: same file, no error
    assert out.read_text() == first
    out.write_text(first.replace('"real": 500', '"real": 2000'))           # planted: an edited file
    with pytest.raises(RuntimeError, match="never replaced"):
        CP.main(tmp_path)


def test_levels_follow_the_100_cpu_hour_rule(tmp_path):
    """cgp.md: 10 levels unless the pilot projects more than 100 CPU-h for them, then the 5 budget levels."""
    _pilot(tmp_path, real_cpu=5.0)                     # 7 real sets x 1,500 runs x ~6 s: well under 100 CPU-h
    res = CP.main(tmp_path)
    st = json.loads((tmp_path / CP.SETTINGS).read_text())
    assert st["levels"] == "all" and st["projected_cal_cpu_h"] == pytest.approx(
        sum(p["cpu_h"] for p in res["projection"].values())) and st["projected_cal_cpu_h"] <= 100
    tmp2 = tmp_path / "costly"
    _pilot(tmp2, real_cpu=60.0)                        # ~61 s per real run: about 180 CPU-h
    CP.main(tmp2)
    st2 = json.loads((tmp2 / CP.SETTINGS).read_text())
    assert st2["projected_cal_cpu_h"] > 100 and st2["levels"] == "budget"


def test_phases_read_the_file_the_pilot_analysis_writes(tmp_path, monkeypatch):
    _pilot(tmp_path)
    CP.main(tmp_path)
    monkeypatch.setattr(P, "ROOT", tmp_path)
    st = P.cgp_settings()
    assert st["genome_length"] == {"real": 500, "bool": 100} and st["levels"] == "all"


def test_settings_wait_for_the_once_only_error_retry(tmp_path):
    """Guardrail row 93: a pilot with error rows counted as complete (rows == tasks), so the settings would be written
    and the once-only retry (keep_running.sh, marker ERRORS_RETRIED) would then change the pilot's results hash.
    Error rows are accepted only after that retry has run and finished (the marker exists, the queue is not running)."""
    import os
    import subprocess
    ph = _pilot(tmp_path)
    rows = [json.loads(l) for l in (ph / "results.jsonl").read_text().splitlines()]
    rows[0] = dict(rows[0], terminal_state="error", exact=False, evals_exact=None)        # one killed run
    (ph / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / CP.SETTINGS
    res = CP.main(tmp_path)
    assert res["complete"] and not out.exists()                                         # planted: retry not yet run
    st = CP.settings(res, _sha(ph / "results.jsonl"), _sha(ph / "manifest.json"), error_rows=1)
    with pytest.raises(RuntimeError, match="ERRORS_RETRIED"):
        CP.write_settings(tmp_path, st)
    (ph / "ERRORS_RETRIED").touch()
    (ph / "queue.pid").write_text(str(os.getpid()))                                     # the retry is still running
    CP.main(tmp_path)
    assert not out.exists()
    with pytest.raises(RuntimeError, match="running"):
        CP.write_settings(tmp_path, st)
    dead = subprocess.Popen(["true"])
    dead.wait()
    (ph / "queue.pid").write_text(str(dead.pid))                                        # finished: a stale pid file
    CP.main(tmp_path)
    assert json.loads(out.read_text())["pilot_error_rows"] == 1
    with pytest.raises(RuntimeError, match="pilot_error_rows"):                         # the count must be stated
        CP.write_settings(tmp_path / "x", {k: v for k, v in st.items() if k != "pilot_error_rows"})

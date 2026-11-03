"""Boolean sizes cannot exceed the function space (REQ-MEA-32, REQ-ANA-92; guardrail row 154)."""
import json
import math
from pathlib import Path

import pytest

from reach import grammar as G
from reach import units

ROOT = Path(__file__).resolve().parents[1]


def test_space_bits_is_the_number_of_input_rows():
    assert units.space_bits(G.get("BOOL-AON-n5")) == 32
    assert units.space_bits(G.get("BOOL-NAND-n6")) == 64
    assert units.space_bits(G.get("R-TRIG")) == math.inf


def test_capped_never_exceeds_the_space():
    g = G.get("BOOL-AON-n5")
    assert units.capped(48.1, g) == 32
    assert units.capped(20.5, g) == 20.5
    assert units.capped(48.1, G.get("R-TRIG")) == 48.1          # real sets have no bound


def test_registered_unit_exceeds_the_space_only_when_extrapolated():
    """The registered target files: a Boolean bits_sem above the bound is always a constructed size whose value is
    extrapolated; no proven size exceeds it (a proven one would mean the enumeration counts are wrong)."""
    over, seen = 0, set()
    for f in sorted((ROOT / "data/targets_s2").glob("BOOL-*/*.json")):
        if f.name.endswith("_log.json"):
            continue
        try:
            g = G.get(f.parent.name)
        except KeyError:              # an older, non-Study-2 folder (e.g. BOOL-AONN-n3); coverage is asserted below
            continue
        seen.add(f.parent.name)
        for t in json.loads(f.read_text()):
            u = t.get("units") or {}
            if u.get("bits_sem", 0) > units.space_bits(g):
                over += 1
                assert t.get("constructed") and u.get("bits_sem_extrapolated"), (f.name, t["target_id"])
    model = json.loads((ROOT / "data/registration/h7_model.json").read_text())
    preds = json.loads((ROOT / "data/registration/heldout_predictions.json").read_text())
    study = set(model["bool"]["fit_records"]["sets"]) | {g for g in preds["features"] if g.startswith("BOOL")}
    assert study <= seen, study - seen     # every Study 2 Boolean set was checked
    assert over > 0          # the registered flaw exists (row 154); if this fails, the files changed


def test_size_variants_change_only_what_they_name():
    from reach.analysis.s2 import preempt
    g = "BOOL-AON-n5"
    recs = [dict(set=g, target_id="p", K=9, units={"bits_sem": 20.0}, y=1, budget=1),
            dict(set=g, target_id="c", K=41, units={"bits_sem": 45.0}, y=0, budget=1)]
    cons = {(g, "c")}
    lb = {g: lambda k: {"bits_sem": 33.5, "nodes": float(k)}}       # lower bound itself above the space: capped too
    v = preempt.bool_size_variants(recs, cons, {g: 30}, {g: {1}}, lb)
    assert [r["units"]["bits_sem"] for r in v["registered"]] == [20.0, 45.0]
    assert [r["units"]["bits_sem"] for r in v["capped"]] == [20.0, 32]
    assert [r["units"]["bits_sem"] for r in v["lower_bound"]] == [20.0, 32]
    assert [r["target_id"] for r in v["proven_only"]] == ["p"]
    assert recs[1]["units"]["bits_sem"] == 45.0                      # the input records are never changed

"""Held-out draw (tool T17; docs/design/function_sets.md; REQ-RUN-17)."""
import pytest

from reach import s2_sets as S
from reach.tasks.s2_draw import draw, seed_from


def test_seed_is_bound_to_the_file_content():
    assert seed_from(b"abc") == seed_from(b"abc")
    assert seed_from(b"abc") != seed_from(b"abd")
    assert 0 <= seed_from(b"abc") < 2 ** 64


def test_draw_is_deterministic_and_a_permutation():
    a, b = draw(12345), draw(12345)
    assert a == b
    for dom in ("real", "bool"):
        assert sorted(a[dom]["order"]) == sorted(S.eligible(dom))


def test_first_two_passing_sets_are_held_out():
    ceilings = {g: 20 for g in S.SETS}
    d = draw(12345, ceilings=ceilings)
    for dom in ("real", "bool"):
        assert d[dom]["held_out"] == d[dom]["order"][:2] and d[dom]["status"] == "final"
    order = d["real"]["order"]
    failing = dict(ceilings, **{order[0]: 9})                  # a set below the 10-node rule is skipped
    d2 = draw(12345, ceilings=failing)
    assert d2["real"]["held_out"] == order[1:3]


def test_unknown_ceiling_in_the_first_positions_leaves_the_draw_pending():
    order = draw(12345)["real"]["order"]
    ceilings = {g: 20 for g in S.SETS}
    ceilings.pop(order[0])
    d = draw(12345, ceilings=ceilings)
    assert d["real"]["status"] == "pending" and d["real"]["pending_on"] == order[0]


def test_held_out_sets_are_locked_until_stage2(tmp_path, monkeypatch):
    """REQ-RUN-10 (C1): no search run on a held-out set before the Stage 2 registration ID is recorded."""
    import json
    rec = tmp_path / "heldout_draw.json"
    rec.write_text(json.dumps({"real": {"held_out": ["R-ARITH"]}, "bool": {"held_out": ["BOOL-AN-n5"]}}))
    from reach import s2_lock as L
    monkeypatch.setattr(L, "DRAW_FILE", rec)
    monkeypatch.setattr(L, "STAGE2_FILE", tmp_path / "stage2_registration.txt")
    monkeypatch.setattr(L, "RANDOM_FILE", tmp_path / "no_random_draw.json")   # listed sets only (random: test_random_sets)
    assert L.held_out() == {"R-ARITH", "BOOL-AN-n5"}
    L.check_search_allowed("R-ARITH4")                       # calibration set: allowed
    with pytest.raises(PermissionError):
        L.check_search_allowed("R-ARITH")
    (tmp_path / "stage2_registration.txt").write_text("https://osf.io/xxxxx\n")
    L.check_search_allowed("R-ARITH")                       # unlocked once the Stage 2 ID is recorded


def test_recorded_draw_matches_the_committed_set_list():
    """The stored draw record reproduces from the set list (when present)."""
    import json
    from reach import s2_lock as L
    if not L.DRAW_FILE.exists():
        pytest.skip("no draw yet")
    rec = json.loads(L.DRAW_FILE.read_text())
    d = draw(rec["seed"], rec["ceilings_used"])
    for dom in ("real", "bool"):
        assert d[dom]["order"] == rec[dom]["order"] and d[dom]["held_out"] == rec[dom]["held_out"]


def test_set_list_frozen_after_draw():
    """The set list seeds the draw: after the draw its bytes must not change (a change is a registered deviation)."""
    import hashlib
    import json
    from reach import s2_lock as L
    if not L.DRAW_FILE.exists():
        pytest.skip("no draw yet")
    rec = json.loads(L.DRAW_FILE.read_text())
    assert hashlib.sha256((L._ROOT / rec["file"]).read_bytes()).hexdigest() == rec["sha256"]
    assert seed_from((L._ROOT / rec["file"]).read_bytes()) == rec["seed"]

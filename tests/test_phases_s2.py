"""Study 2 phase definitions (docs/plan/phases/)."""
from reach.runner.phases import PHASES


def test_phase0_aliases_match_paper1_regressions():
    """REQ-RULE-07: the Phase 0 regressions run exactly paper 1's task lists and verify step."""
    for new, old in (("s2_legacy_check", "legacy_check"), ("s2_m1_regression", "m1_regression")):
        assert PHASES[new]["build"]() == PHASES[old]["build"]()
        assert PHASES[new]["verify"] is PHASES[old]["verify"] is not None

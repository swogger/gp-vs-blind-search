"""Release test configuration. Tests marked study_machine need the study machine's untracked files (enumeration
level files, run-code snapshots) or its exact platform; the tests listed below read the authors' private working
documents or history, which are not released. Both are skipped here; every other test runs."""
import pytest

SKIPPED = {'tests/test_cgp_confirm_phases.py::test_docs_name_the_cgp_seed_slices': "reads the authors' private working documents", 'tests/test_checksums.py::test_form_check_fires_on_planted_violations': "reads the authors' private working documents", 'tests/test_checksums.py::test_stage2_checksums_cover_every_attachment_the_form_names': "reads the authors' private working documents", 'tests/test_guardrails.py::test_ci_wait_reports_the_commit_it_watched': 'checks a private process script', 'tests/test_guardrails.py::test_docs_index_complete': "reads the authors' private working documents", 'tests/test_guardrails.py::test_requirements_traceable': "reads the authors' private working documents", 'tests/test_guardrails.py::test_watchdog_agent_keeps_the_queues_it_starts_and_runs_at_normal_priority': 'checks a private process script', 'tests/test_s2_chain.py::test_order_is_the_registered_order': "reads the authors' private working documents", 'tests/test_s2_chain.py::test_registered_code_is_the_registered_commits': 'needs the private git history', 'tests/test_stage2_report.py::test_form_entry_points_exist': "reads the authors' private working documents", 'tests/test_stage2_report.py::test_report_is_short_and_placeholder_free': "reads the authors' private working documents"}


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.nodeid in SKIPPED:
            item.add_marker(pytest.mark.skip(reason=SKIPPED[item.nodeid]))
        elif "study_machine" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="needs the study machine's untracked files or exact platform"))

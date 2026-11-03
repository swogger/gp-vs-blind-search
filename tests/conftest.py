"""Shared test configuration.

`study_machine` marks tests that hold only on the study machine (exact Python patch version, exact float
reproduction of EML-GP's runs). They run locally and in the pre-commit hook, and are skipped in CI (the `CI`
environment variable), where the platform's maths library differs (docs/requirements/runs.md, REQ-RUN-14).
"""
import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "study_machine: holds only on the study machine; skipped in CI")
    config.addinivalue_line("markers", "runtime_model: keep confirm.build's run-time model on (slow)")


def pytest_collection_modifyitems(config, items):
    if not os.environ.get("CI"):
        return
    skip = pytest.mark.skip(reason="study-machine test: exact reproduction is checked locally, not in CI")
    for item in items:
        if "study_machine" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def cgp_exe():
    """The CGP binary for the Rust source in this tree, built once per session (cached by cargo) and installed in the
    store under its source hash (reach.cgp.build; REQ-RUN-23)."""
    from reach import cgp
    return cgp.build()


@pytest.fixture(autouse=True)
def _runtime_model_off_by_default(request, monkeypatch):
    """Speed only: `confirm.build` fits the run-time model with a target bootstrap per set (seconds per set and
    replicate), so tests that do not ask for it run without it. Tests marked `runtime_model` keep the registered
    default, and both entry points (confirm.main, dryrun.main) pass runtime_model=True explicitly
    (tests/test_runtime_wiring.py checks both)."""
    if "runtime_model" in request.keywords:
        return
    from reach.analysis.s2 import confirm
    monkeypatch.setattr(confirm, "RUNTIME_MODEL", False)

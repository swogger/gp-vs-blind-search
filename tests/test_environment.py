"""The environment must match the pins that reproduce EML-GP's stored results exactly."""
import platform
from importlib.metadata import version
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _pins():
    pins = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, ver = line.split("==")
            pins[name] = ver
    return pins


@pytest.mark.study_machine
def test_python_version():
    assert platform.python_version() == "3.9.6"


@pytest.mark.parametrize("name,pinned", sorted(_pins().items()))
def test_package_pinned(name, pinned):
    assert version(name) == pinned


def test_reach_importable():
    import reach  # noqa: F401

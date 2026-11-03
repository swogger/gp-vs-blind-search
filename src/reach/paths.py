"""Repository root. Tasks run from a code snapshot (data/runs/<phase>/code/<hash>/), where
__file__ does not point into the repository, so the queue passes the root in REACH_ROOT."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("REACH_ROOT") or Path(__file__).resolve().parents[2])

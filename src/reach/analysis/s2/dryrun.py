"""DRYRUN: the confirmatory analysis (`reach.analysis.s2.confirm`) run end to end on pilot data (REQ-ANA-09).

    .venv/bin/python -m reach.analysis.s2.dryrun      -> data/analysis/dryrun.json, docs/status/dryrun_report.md

The same code path as the calibration analysis, on the pilot phases, with small bootstraps. Every estimate is a pilot
estimate and must never be quoted as a result.
"""
import json

from reach.analysis.s2.confirm import PHASE_MAPS, UNITS, build as _build, render as _render   # noqa: F401 (UNITS)
from reach.analysis.s2.confirm import render_h1_cells as _render_h1
from reach.analysis.s2.confirm import render_h8 as _render_h8
from reach.analysis.s2.confirm import render_holm_members as _render_members
from reach.analysis.s2.confirm import render_runtime as _render_runtime
from reach.analysis.s2.confirm import render_slopes_gaps as _render_slopes
from reach.analysis.s2.confirm import render_supplement as _render_supp
from reach.analysis.s2.termination import render_termination as _render_term

BOOT = 200
PHASE_MAPS["pilot"]["label"] = "DRYRUN: pilot data, not confirmatory; never quote these numbers"


def build(root=None, delta=None, boot=BOOT, runtime_model=None):
    return _build(PHASE_MAPS["pilot"], root, boot, None if delta is None else {"real": delta, "bool": delta},
                  runtime_model=runtime_model)


H1_CELLS = "docs/status/dryrun_h1_cells.md"
MEMBERS = "docs/status/dryrun_holm_members.md"
TERMINATION = "docs/status/dryrun_termination.md"
SUPPLEMENT = "docs/status/dryrun_supplement.md"
RUNTIME = "docs/status/dryrun_runtime.md"
SLOPES = "docs/status/dryrun_slopes_gaps.md"
H8 = "docs/status/dryrun_h8.md"


def render(res):
    return _render(res, "DRYRUN: confirmatory analysis on pilot data", "reach.analysis.s2.dryrun", H1_CELLS, MEMBERS,
                   TERMINATION, SUPPLEMENT, RUNTIME, SLOPES, H8)


def render_slopes(res):
    return _render_slopes(res, "DRYRUN: per-set slopes, H4c pairs and H6b gaps on pilot data", "reach.analysis.s2.dryrun")


def render_h8(res):
    return _render_h8(res, "DRYRUN: H8's parts (H8a cells, H8b, H8c) on pilot data", "reach.analysis.s2.dryrun")


def render_runtime(res):
    return _render_runtime(res, "DRYRUN: run-time model (H4d-f; Weibull against lognormal) on pilot data",
                           "reach.analysis.s2.dryrun")


def render_termination(res):
    return _render_term(res, "DRYRUN: termination states and exclusions on pilot data", "reach.analysis.s2.dryrun")


def render_supplement(res):
    return _render_supp(res, "DRYRUN: supplement (H3, McNemar, best split, sizes) on pilot data",
                        "reach.analysis.s2.dryrun")


def render_members(res):
    return _render_members(res, "DRYRUN: Holm families on pilot data", "reach.analysis.s2.dryrun", H8)


def render_h1(res):
    return _render_h1(res, "DRYRUN: H1 per cell on pilot data", "reach.analysis.s2.dryrun")


def main():
    from reach.paths import ROOT
    res = build(runtime_model=True)
    (ROOT / "data/analysis").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/analysis/dryrun.json").write_text(json.dumps(res, indent=1, default=float))
    (ROOT / "docs/status/dryrun_report.md").write_text(render(res))
    (ROOT / H1_CELLS).write_text(render_h1(res))
    (ROOT / MEMBERS).write_text(render_members(res))
    (ROOT / TERMINATION).write_text(render_termination(res))
    (ROOT / SUPPLEMENT).write_text(render_supplement(res))
    (ROOT / RUNTIME).write_text(render_runtime(res))
    (ROOT / SLOPES).write_text(render_slopes(res))
    (ROOT / H8).write_text(render_h8(res))
    print(render(res))


if __name__ == "__main__":
    main()

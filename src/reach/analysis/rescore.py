"""Correction for the signed-zero exact-match bug (review 2026-10-15; deviations log).

The run-level exact check hashed -0.0 and +0.0 imaginary parts differently, so some final trees identical to the
target on the grid were scored "not exact". This script re-scores every non-error run of the confirmatory phases
from its stored final tree with the corrected hash (a correction can only turn a miss into a match), writes the
flips to data/rescored/<phase>.json, and reruns the registered analysis `reach.analysis.family_full` unchanged
(same code, same RNG) on the corrected outcomes -> docs/results/eml_family_full_report_rescored.md.

    .venv/bin/python -m reach.analysis.rescore
"""
import json
import sys

from reach import grammar as G
from reach.analysis import family_full
from reach.paths import ROOT
from reach.runner.queue import Phase

PHASES = ("full_gp", "full_es", "full_budget", "full_budget64")


def rescore(phase):
    man = {t["key"]: t for t in json.loads((ROOT / f"data/runs/{phase}/manifest.json").read_text())["tasks"]}
    cache, flips, n_noexpr = {}, [], 0
    for k, r in Phase(phase).results().items():
        if r["terminal_state"] == "error" or r.get("exact_hash_match") == "true":
            continue
        if not r.get("best_expr"):
            n_noexpr += 1
            continue
        argv = man[k]["argv"]
        path = argv[argv.index("--targets") + 1]
        if path not in cache:
            cache[path] = {t["target_id"]: t for t in json.loads((ROOT / path).read_text())}
        t, g = cache[path][r["target_id"]], G.get(r["grammar"])
        xs = G.sample_grid(g)
        tv = G.evaluate(g, G.parse_expr(t["expression"], g), xs)
        bv = G.evaluate(g, G.parse_expr(r["best_expr"], g), xs)
        if G.scaled_func_hash(bv, t["target_scale"]) == G.scaled_func_hash(tv, t["target_scale"]):
            flips.append(k)
    return flips, n_noexpr


class RescoredPhase(Phase):
    flips = {}

    def results(self):
        res = super().results()
        for k in self.flips.get(self.name, ()):
            res[k] = dict(res[k], exact_hash_match="true")
        return res


def main():
    out = ROOT / "data/rescored"
    out.mkdir(exist_ok=True)
    summary = {}
    for ph in PHASES:
        flips, n_noexpr = rescore(ph)
        (out / f"{ph}.json").write_text(json.dumps(sorted(flips), indent=1))
        RescoredPhase.flips[ph] = set(flips)
        summary[ph] = dict(flipped=len(flips), misses_without_stored_tree=n_noexpr)
        print(ph, summary[ph], flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    family_full.Phase = RescoredPhase
    sys.argv = ["family_full", "--out", "docs/results/eml_family_full_report_rescored.md"]
    family_full.main()


if __name__ == "__main__":
    main()

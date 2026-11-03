"""Compare runs with EML-GP's stored corrected runs (legacy/results_rerun/P1_synth_main.csv)."""
import csv
from pathlib import Path

from reach.paths import ROOT
P1 = ROOT / "legacy" / "results_rerun" / "P1_synth_main.csv"
FIELDS = ("converged", "exact_hash_match", "gens_to_converge", "final_nmse",
          "final_tree_size", "terminal_state", "gens_run")


def load_p1(path: Path = P1) -> dict:
    with open(path, newline="") as fh:
        return {r["key"]: r for r in csv.DictReader(fh)}


def compare(rows: dict, reference: dict, fields=FIELDS) -> list:
    """Exact comparison as strings: final_nmse is stored as repr(), so equal strings mean equal floats."""
    bad = []
    for key, row in rows.items():
        ref = reference.get(key)
        if ref is None:
            bad.append((key, "<missing in reference>", "", ""))
            continue
        for f in fields:
            ours = "" if row.get(f) is None else str(row.get(f))
            if ours != ref[f]:
                bad.append((key, f, ours, ref[f]))
    return bad

"""Draw the random held-out set candidates (T26; D064) -> data/registration/random_draw.json (written once).

    python -m reach.tasks.s2_random_draw --osf a3spg

Seed: `s2_random_sets.draw_seed` of the Stage 1 OSF identifier, plus 0 (real) or 1 (Boolean), as in the Stage 0 draw.
N_CANDIDATES accepted candidates per domain, in draw order; they are enumerated in that order, and the held-out sets
are the first 4 per domain whose proven ceiling is at least 10 nodes. Every candidate is locked (s2_lock.held_out)."""
import argparse
import hashlib
import json

from reach.paths import ROOT
from reach.s2_lock import RANDOM_FILE
from reach.tasks import s2_random_sets as R

N_CANDIDATES = 10


def draw(osf_id):
    seed = R.draw_seed(osf_id)
    out = dict(osf_id=osf_id.lower(), seed=seed, rule="first 4 per domain, in order, with proven ceiling >= 10",
               code_sha256=hashlib.sha256((ROOT / "src/reach/tasks/s2_random_sets.py").read_bytes()).hexdigest())
    for i, dom in enumerate(("real", "bool")):
        cands = R.draw_candidates(dom, seed + i, n=N_CANDIDATES)
        for c in cands:
            c["name"] = R.make_grammar(dom, c["ops"], c["constants"]).name
        out[dom] = dict(seed=seed + i, candidates=cands)
    return out


def heldout_sets(rec, ceilings, n=4):
    """The registered rule: per domain, the first n candidates in draw order whose proven ceiling is at least 10.
    A candidate before the n-th accepted one without an enumeration is an error (the order decides)."""
    out = {}
    for dom in ("real", "bool"):
        got = []
        for c in rec[dom]["candidates"]:
            if len(got) == n:
                break
            if c["name"] not in ceilings:
                raise ValueError(f"{c['name']} is not enumerated yet; the draw order decides")
            if ceilings[c["name"]] >= 10:
                got.append(c["name"])
        out[dom] = got
    return out


def heldout_from_files(root=None):
    from reach.analysis.s2.phase1 import enum_dir
    rec = json.loads(RANDOM_FILE.read_text())
    ceil = {}
    for dom in ("real", "bool"):
        for c in rec[dom]["candidates"]:
            p = enum_dir(c["name"]) / "counts.json"
            if p.exists():
                ks = [st["K"] for st in json.loads(p.read_text())["levels"] if st["U"] > 0 and st["K"] >= 3]
                if ks:
                    ceil[c["name"]] = max(ks)
    return heldout_sets(rec, ceil)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osf", required=True, help="the Stage 1 OSF identifier")
    a = ap.parse_args()
    if RANDOM_FILE.exists():
        raise SystemExit(f"{RANDOM_FILE} exists: the draw is done once")
    rec = draw(a.osf)
    RANDOM_FILE.write_text(json.dumps(rec, indent=1) + "\n")
    for dom in ("real", "bool"):
        print(dom, [c["name"] for c in rec[dom]["candidates"]])


if __name__ == "__main__":
    main()

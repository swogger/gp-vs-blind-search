"""Held-out draw of listed function sets (tool T17; D048 C1; docs/design/function_sets.md).

The seed is the first 8 bytes (big-endian) of the SHA-256 of `src/reach/s2_sets.py` as committed at HEAD, so anyone
can recompute it from the repository. Per domain, the eligible sets (sorted by name) are put in a random order with
random.Random(seed + domain index); the held-out sets are the first 2 in that order whose proven ceiling is at least
10 nodes. If a set in front of them has no known ceiling yet, the draw is pending on it (the order never changes).

    .venv/bin/python -m reach.tasks.s2_draw            # -> data/registration/heldout_draw.json
"""
import hashlib
import json
import random
import subprocess

from reach import s2_sets as S
from reach.paths import ROOT

DOMAINS = ("real", "bool")
N_HELD_OUT = 2
MIN_CEILING = 10


def seed_from(content: bytes) -> int:
    return int.from_bytes(hashlib.sha256(content).digest()[:8], "big")


def draw(seed, ceilings=None):
    ceilings = ceilings or {}
    out = {}
    for i, dom in enumerate(DOMAINS):
        order = sorted(S.eligible(dom))
        random.Random(seed + i).shuffle(order)
        held, status, pending = [], "final", None
        for g in order:
            if len(held) == N_HELD_OUT:
                break
            c = ceilings.get(g)
            if c is None:
                status, pending = "pending", g
                break
            if c >= MIN_CEILING:
                held.append(g)
        out[dom] = dict(order=order, held_out=held, status=status, pending_on=pending)
    return out


def known_ceilings(sources=None):
    """Proven ceilings (last level with functions), first found among: the defining enumeration; the Study 2
    enumeration at 8 GB (a lower bound for sets re-enumerated with a larger cap); for Boolean sets, the archived
    enumeration (exact semantics, so its ceiling is proven too). None if no enumeration exists yet."""
    out = {}
    for g, s in S.SETS.items():
        cands = [ROOT / s["root"] / g, ROOT / S.ENUM / g] + ([ROOT / "data/enumeration" / g] if s["domain"] == "bool" else [])
        for d in cands:
            p = d / "counts.json"
            if p.exists():
                levels = json.loads(p.read_text())["levels"]
                out[g] = max(st["K"] for st in levels if st["U"] > 0)
                if sources is not None:
                    sources[g] = str(d.relative_to(ROOT))
                break
    return out


def main():
    rel = "src/reach/s2_sets.py"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    content = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, check=True).stdout
    seed = seed_from(content)
    src = {}
    ceil = known_ceilings(src)
    res = dict(file=rel, commit=commit, sha256=hashlib.sha256(content).hexdigest(), seed=seed,
               rule=f"per domain: random.Random(seed + index), index real=0 bool=1, over sorted eligible sets; "
                    f"held out = first {N_HELD_OUT} with proven ceiling >= {MIN_CEILING}",
               ceilings_used={g: ceil.get(g) for d in DOMAINS for g in S.eligible(d)},
               ceiling_sources={g: src.get(g) for d in DOMAINS for g in S.eligible(d)}, **draw(seed, ceil))
    out = ROOT / "data/registration"
    out.mkdir(parents=True, exist_ok=True)
    (out / "heldout_draw.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({d: res[d] for d in DOMAINS}, indent=1))


if __name__ == "__main__":
    main()

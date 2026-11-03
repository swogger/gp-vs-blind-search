"""Step 3c driver: freeze the Stage 2 predictors and write the held-out levels (REQ-ANA-74; D090).

    .venv/bin/python -m reach.analysis.s2.stage2_freeze

From the confirmatory calibration result (data/analysis/calibration.json) and calibration data only (no held-out
data exist), each written once with its .sha256 and never replaced with other content:
- PATHS["h7"]: the H7 model per domain (`freeze.frozen_model`), with its Brier baseline (nodes + log2 budget, D090)
  fitted on the H7 selection records, which are rebuilt as confirm built them and must reproduce calibration.json's
  coefficients (`check_reproduces`);
- PATHS["gpf2013"], PATHS["gp2010"]: comparators 1 and 2 as fitted by confirm (comparator 2 with its reference vectors
  restored from the calibration target files);
- PATHS["operon"]: the H9b transfer model per Operon configuration (`h9b.fit_transfer` on s2_cal_operon);
- PATHS["levels"]: the held-out levels of the 12 held-out sets from the frozen model of their domain
  (`freeze.heldout_levels`: the G2 rule, each set's own size units with the D050 rows for real sets, its feature value
  from data/analysis/properties.json). Real sets load their enumerated values (level files) for the D050 rows.
Analysis only (seconds to minutes); no run. The held-out targets follow from the levels (queue phase s2_ho_targets,
launched by the author). Then, once:

    .venv/bin/python -m reach.analysis.s2.stage2_freeze predictions

writes PATHS["predictions"] (REQ-ANA-77): the frozen H7 model's P(exact) for every held-out target and benchmark cell
of the held-out sets at every scored budget (predictor.SCORED["H7"]), with the model's SHA-256 and the SHA-256 of
every target file read (`prediction_body`, `write_prediction_file`).
"""
import hashlib
import json
import math
from pathlib import Path

from reach.analysis.s2 import freeze, gp2010, gpf2013, h9b, predictor

PATHS = dict(h7="data/registration/h7_model.json", gpf2013=gpf2013.FROZEN_FILE, gp2010=gp2010.FROZEN_FILE,
             operon=h9b.TRANSFER_FILE, levels="data/registration/heldout_levels.json",
             predictions="data/registration/heldout_predictions.json")
CALIBRATION = "data/analysis/calibration.json"
KMAX = 200                     # sizes searched for the landmarks and the available sizes (as the G2 levels)


def h7_records(root, phmap=None):
    """The H7 selection records per domain, built as `confirm.build` builds them: regime A rows of the reach phases,
    censored runs as failures, units from the target files, the main sets (size checks left out), budgets 1-16 x E0,
    each record with its set's features."""
    from reach import grammar as G
    from reach.analysis.s2 import confirm as C
    from reach.analysis.s2.load import tidy
    phmap = phmap or C.PHASE_MAPS["calibration"]
    root = Path(root)
    a_rows = C._rows(root, phmap["reach"])
    sets = sorted({r.get("set") for r in a_rows if r.get("set")})
    C._load_units(root, sets, phmap["targets"])
    recs = {g: C._with_units(tidy([r for r in a_rows if r.get("set") == g], C.BUDGETS, True)) for g in sets}
    out = {}
    for d in ("real", "bool"):
        main = [g for g in sets if G.get(g).kind == d and g not in C.SIZE_CHECKS]
        if len(main) < 3:
            continue
        h7 = [r for g in main for r in recs[g] if r["budget"] in C.H4_BUDGETS]
        feats = C.set_features(root, main) if (root / "data/analysis/properties.json").exists() else {}
        for r in h7:
            r["features"] = feats.get(r["set"], {})
        out[d] = h7
    return out


def check_reproduces(h7, recs, rel=1e-8):
    """The selected model refitted on `recs` must give calibration.json's coefficients: the records are the ones the
    model was selected and fitted on, so the baseline fitted on them is fitted on the same data."""
    p = predictor._fit(recs, h7["selected_unit"], h7["budget_term"], h7["feature"])
    if len(p) != len(h7["params"]) or not all(math.isclose(a, b, rel_tol=rel, abs_tol=1e-12)
                                             for a, b in zip(p, h7["params"])):
        raise ValueError(f"the H7 records do not reproduce the calibration fit: {list(map(float, p))} against "
                         f"{h7['params']}")


def h7_models(cal, recs, cal_sha256):
    out = {}
    for d, rr in sorted(recs.items()):
        h = cal["h7"][d]
        check_reproduces(h, rr)
        m = freeze.frozen_model(h, d)
        m.update(baseline=predictor.fit_baseline(rr, budget_term=True), calibration_sha256=cal_sha256,
                 fit_records=dict(n=len(rr), sets=sorted({r["set"] for r in rr}), budgets=sorted({r["budget"] for r in rr}),
                                  phases=["s2_cal_a0", "s2_cal_a16"], algorithm="A", censored="failure"))
        out[d] = m
    return out


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_h7(models, path):
    sha = freeze.freeze_models(models, path)["sha256"]
    return {d: dict(m, sha256=sha) for d, m in models.items()}


def load_h7(path):
    """The frozen H7 models, each with the file's SHA-256, after checking the file against its .sha256."""
    path = Path(path)
    sha = _sha(path)
    if path.with_suffix(".sha256").read_text().split()[0] != sha:
        raise ValueError(f"{path.name} does not match its checksum: it was changed after it was frozen")
    return {d: dict(m, sha256=sha) for d, m in json.loads(path.read_text()).items()}


def comparator_models(cal, vectors):
    """Comparators 1 (gpf2013) and 2 (gp2010) per domain as confirm fitted them; refused if either failed or was not
    fitted. Comparator 2 gets its reference vectors back (`gp2010.attach_vectors`)."""
    c1, c2 = {}, {}
    for d in ("real", "bool"):
        h = cal["h7"][d]
        for name, c in (("comparator", h.get("comparator")), ("comparator2", h.get("comparator2"))):
            if not c or c.get("error"):
                raise ValueError(f"{d} {name}: {c.get('error') if c else 'not fitted'}; nothing to freeze")
        c1[d] = gpf2013.summary(h["comparator"])
        c2[d] = gp2010.attach_vectors(h["comparator2"], vectors)
    return c1, c2


def operon_records(root):
    """s2_cal_operon records at 16 x E0 (killed runs as failures), units from the calibration target files."""
    from reach.analysis.s2 import confirm as C
    from reach.analysis.s2.load import tidy
    rows = C._rows(Path(root), ("s2_cal_operon",))
    C._load_units(Path(root), sorted({r.get("set") for r in rows if r.get("set")}), "calibration")
    return C._with_units(tidy(rows, h9b.BUDGETS, True))


def heldout_levels_for(models, domains, units, available, features):
    """Per set: the G2 levels from its domain's frozen model, its units map and its features (`freeze.heldout_levels`)."""
    out = {}
    for g, d in sorted(domains.items()):
        out[g] = dict(freeze.heldout_levels(models[d], units[g], available[g], KMAX, features[g]), domain=d)
    out["_h7_model_sha256"] = {d: models[d]["sha256"] for d in sorted(set(domains.values()))}
    out["_source"] = ("reach.analysis.s2.stage2_freeze: the G2 rule on the frozen H7 model of each domain, mapped to "
                      "nodes through each set's size units (D050 rows for real sets); no held-out data")
    return out


def write_once(obj, path):
    """JSON with its .sha256 (predictor.freeze): an existing file with other content is never replaced."""
    return predictor.freeze(obj, path)["sha256"]


def set_units(root, g):
    """(units map K -> units, available sizes) for a held-out set, as the target generator builds its units."""
    from reach import grammar as G
    from reach.analysis.s2 import g2_levels
    from reach.analysis.s2.phase1 import enum_dir, inflation_from
    from reach.units import SizeUnits
    gr = G.get(g)
    d = enum_dir(g)
    levels = json.loads((d / "counts.json").read_text())["levels"]
    noise = None
    if gr.kind == "real":
        from reach.enumerate import Enumerator
        e = Enumerator(gr, d, log=lambda *_: None)
        e.load()
        noise = inflation_from(e)
    return SizeUnits(gr, levels, noise_rows=noise).all, g2_levels.available_sizes(levels, KMAX)


def main(root=None):
    from reach import grammar as G
    from reach.analysis.s2 import confirm as C
    from reach.paths import ROOT
    from reach.runner.phases import heldout_sets
    root = Path(root or ROOT)
    cal = json.loads((root / CALIBRATION).read_text())
    recs = h7_records(root)
    if set(recs) != {"real", "bool"}:
        raise ValueError(f"H7 records for {sorted(recs)}; both domains expected")
    vec, _ = gp2010.vectors_for([r for rr in recs.values() for r in rr], C._TARGETS)   # calibration target vectors
    frozen = freeze_h7(h7_models(cal, recs, _sha(root / CALIBRATION)), root / PATHS["h7"])
    c1, c2 = comparator_models(cal, vec)
    out = dict(h7=frozen["real"]["sha256"], gpf2013=write_once(c1, root / PATHS["gpf2013"]),
               gp2010=write_once(c2, root / PATHS["gp2010"]))
    op = h9b.fit_transfer(frozen["real"], operon_records(root))
    out["operon"] = h9b.freeze_transfer(op, root / PATHS["operon"])["sha256"]
    sets = heldout_sets()
    units, avail = {}, {}
    for g in sets:
        units[g], avail[g] = set_units(root, g)
    lv = heldout_levels_for(frozen, {g: G.get(g).kind for g in sets}, units, avail, C.set_features(root, sets))
    out["levels"] = write_once(lv, root / PATHS["levels"])
    for k, sha in out.items():
        print(f"{sha}  {PATHS[k]}")
    for g in sets:
        print(f"{g:32s} main {lv[g]['main']}  budget {lv[g]['budget']}")
    for c, m in sorted(op.items()):
        print(f"H9b {c}: c0 {m['transfer']['c0']:.4f}, c1 {m['transfer']['c1']:.4f}, n {m['n']}")
    return out


def _checked_targets(root, rel):
    """A target file read only if it matches its .sha256 (as the run tasks require)."""
    f = Path(root) / rel
    side = f.with_suffix(".sha256")
    if not side.exists() or side.read_text().split()[0] != _sha(f):
        raise ValueError(f"{rel} does not match its checksum file {side.name}: refused")
    return json.loads(f.read_text()), _sha(f)


def prediction_inputs(root):
    """The frozen H7 models (checked against their .sha256), the held-out sets of heldout_levels.json (checked) and
    every target of their heldout.json and bench.json files (each checked), with its domain and its set's value of the
    model's feature. The feature value must be the one the set's levels were placed with."""
    from reach import grammar as G
    from reach.analysis.s2 import confirm as C
    root = Path(root)
    models = load_h7(root / PATHS["h7"])
    lv, _ = _checked_targets(root, PATHS["levels"])
    sets = sorted(g for g in lv if not g.startswith("_"))
    feats = C.set_features(root, sets)
    targets, sources, used = [], {}, {}
    for g in sets:
        d = lv[g].get("domain") or G.get(g).kind
        z = freeze._features(models[d], feats.get(g))
        f = models[d].get("feature")
        if f and lv[g].get("rule", {}).get("feature_value") != z[f]:
            raise ValueError(f"{g}: feature {f} = {z[f]}, but its held-out levels were placed with "
                             f"{lv[g].get('rule', {}).get('feature_value')!r}")
        used[g] = z
        for kind in ("heldout", "bench"):
            rel = f"data/targets_s2/{g}/{kind}.json"
            if kind == "bench" and not (root / rel).exists():
                continue
            ts, sources[rel] = _checked_targets(root, rel)
            targets += [dict(set=g, target_id=t["target_id"], units=t["units"], domain=d, kind=kind, features=z)
                        for t in ts]
    return models, targets, dict(sources=sources, features=used)


def _prediction_args(root):
    models, targets, meta = prediction_inputs(root)
    budgets = list(predictor.SCORED["H7"]["budgets"])
    meta.update(budgets=budgets, model_file=PATHS["h7"], levels_file=PATHS["levels"],
                source=("reach.analysis.s2.stage2_freeze predictions: P(exact) of the frozen H7 model of each set's "
                        "domain for every held-out target and benchmark cell at every scored budget; no held-out run"))
    return models, targets, budgets, meta


def prediction_body(root):
    """The prediction file's text, built in memory from the committed inputs (for writing and for checking)."""
    return freeze.prediction_body(*_prediction_args(root))


def write_prediction_file(root=None):
    """Write PATHS["predictions"] with its .sha256 once; an existing file with other content is refused (row 132)."""
    from reach.paths import ROOT
    root = Path(root or ROOT)
    models, targets, budgets, meta = _prediction_args(root)
    return _sha(freeze.write_predictions(models, targets, budgets, root / PATHS["predictions"], meta=meta))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["predictions"]:
        print(f"{write_prediction_file()}  {PATHS['predictions']}")
    else:
        main()

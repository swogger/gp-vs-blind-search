"""Step 3c (Phase 6 -> Stage 2): freeze the H7 predictor, place held-out levels from it, write the prediction file.

The frozen model (`predictor.select` on calibration data, one per domain) is logit P = a + b * s_u + g * log2(B/E0),
in the H2-preferred unit u, or bits_sem when no unit is preferred (rule 6; D082), recorded in the model file.
Held-out levels come from this model alone (no held-out run exists; C1): its E0 curve and its K50 at 1/4 and 16 x E0
are mapped from u to nodes through each held-out set's own size units (enumeration counts) and placed by the G2 rule
(`g2_levels.choose_k`). The prediction file holds P(exact) for every held-out target and benchmark cell at every
registered budget, with the model's SHA-256; its own SHA-256 goes into the Stage 2 form.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from reach.analysis.s2 import g2_levels, predictor


def _features(model, features):
    """The set's value of the model's set-level feature (design 9.7), required when the model has one (row 122)."""
    f = model.get("feature")
    if not f:
        return {}
    z = (features or {}).get(f)
    if not isinstance(z, (int, float)) or isinstance(z, bool) or not math.isfinite(z):
        raise ValueError(f"the frozen model has the set-level feature {f!r}: the set's value is needed (got {z!r})")
    return {f: float(z)}


def _s_at(model, p, budget, features=None):
    """Size in the model's unit where the frozen model gives P = p at the budget, with the set's feature value: the
    model's own linear predictor (`predictor._X`, as `predictor.predict`) at size 0, so the levels come from exactly
    the model that is scored (guardrail row 122: the feature term was left out)."""
    rec = dict(units={model["unit"]: 0.0}, budget=budget, features=_features(model, features))
    eta0 = float((predictor._X([rec], model["unit"], model["budget_term"], model.get("feature"))
                  @ np.array(model["params"], float))[0])
    return (math.log(p / (1 - p)) - eta0) / model["params"][1]


def landmarks(model, units_fn, kmax=200, features=None):
    """Sizes in nodes where the model gives P = 0.95, 0.05 (E0), P = 1/2 at 1/4 and 16 x E0, and d (half the E0 curve's
    25-75% width), by inverting the set's monotone map K -> s_u(K). features: the set's set-level features (required
    when the model has one; data/analysis/properties.json via confirm.set_features)."""
    _features(model, features)
    ks = np.arange(1, kmax + 1, dtype=float)
    s = np.array([units_fn(int(k))[model["unit"]] for k in ks], float)
    keep = np.r_[True, s[1:] > np.maximum.accumulate(s)[:-1]]   # sizes where the unit rises (binary sets step; row 67)
    ks, s = ks[keep], s[keep]
    if len(s) < 2:
        raise ValueError("the size unit must increase with nodes")
    to_k = lambda sv: float(np.interp(sv, s, ks))                             # noqa: E731
    at = lambda p, b: to_k(_s_at(model, p, b, features))                     # noqa: E731
    k25, k75 = at(0.25, 1), at(0.75, 1)
    return dict(k95=at(0.95, 1), k05=at(0.05, 1), d=(k25 - k75) / 2, k50_quarter=at(0.5, 0.25), k50_16=at(0.5, 16))


def heldout_levels(model, units_fn, available, kmax=200, features=None):
    lm = landmarks(model, units_fn, kmax, features)
    out = g2_levels.choose_k(available, lm["k95"], lm["k05"], lm["d"], lm["k50_quarter"], lm["k50_16"])
    out["rule"]["source"] = "frozen model (no held-out data)"
    if model.get("feature"):
        out["rule"].update(feature=model["feature"], feature_value=_features(model, features)[model["feature"]])
    return out


def frozen_model(h7, domain):
    """The model to freeze from confirm's H7 result for one domain: unit, budget term, set-level feature, coefficients,
    and where the unit came from (D082: the H2-preferred unit, or bits_sem under rule 6, `rule6` True). Refused when
    the H7 selection failed (guardrail row 119)."""
    if h7.get("error") or h7.get("params") is None:
        raise ValueError(f"no H7 model to freeze for {domain}: {h7.get('error', 'no coefficients')}")
    return dict(domain=domain, unit=h7["selected_unit"], rule6=h7["rule6"], unit_source=h7["unit_source"],
                budget_term=h7["budget_term"], feature=h7["feature"], params=list(h7["params"]))


def _check_unit_source(model):
    """D082: the model records its unit and whether it came from rule 6, consistently with its unit and label."""
    if not isinstance(model.get("rule6"), bool) or not model.get("unit_source"):
        raise ValueError("the frozen H7 model must record its unit source and rule-6 flag (D082)")
    if model["rule6"] != ("rule 6" in model["unit_source"]) or (model["rule6"] and model["unit"] != "bits_sem"):
        raise ValueError(f"inconsistent H7 unit source: unit {model['unit']}, rule6 {model['rule6']}, "
                         f"{model['unit_source']!r} (D082: bits_sem under rule 6)")


def freeze_model(model, path):
    """Freeze with SHA-256 (predictor.freeze). D082: the model records its unit and whether it came from rule 6; a
    model without them, or with a rule-6 flag that disagrees with its unit or label, is refused (nothing written)."""
    _check_unit_source(model)
    return predictor.freeze(model, path)


def freeze_models(models, path):
    """Both domains' H7 models in one file with its .sha256 (step 3c): each passes the D082 check and carries its Brier
    baseline (nodes + log2 budget, D090); an existing file with other content is never replaced (row 123)."""
    from reach.analysis.s2 import transfer
    if not isinstance(models, dict) or not models or set(models) - {"real", "bool"}:
        raise ValueError(f"H7 models by domain expected, got {sorted(models) if isinstance(models, dict) else models!r}")
    for d, m in models.items():
        if m.get("domain") != d:
            raise ValueError(f"H7 model under {d} has domain {m.get('domain')!r}")
        _check_unit_source(m)
        transfer.check_baseline(m.get("baseline"), True, f"H7 model ({d})")
    return predictor.freeze(models, path)


def prediction_body(model, targets, budgets, meta=None):
    """The prediction file's text: one row per target and budget, sorted, each with the model's SHA-256. model: one
    frozen model, or frozen models by domain with each target's `domain` choosing its model (one model file)."""
    by_domain = "params" not in model
    shas = {m["sha256"] for m in (model.values() if by_domain else [model])}
    if len(shas) != 1:
        raise ValueError(f"the models come from {len(shas)} files; one frozen model file expected")
    rows = []
    for d in sorted({t.get("domain") for t in targets} if by_domain else [None]):
        m = model[d] if by_domain else model
        mine = [t for t in targets if not by_domain or t.get("domain") == d]
        recs = [dict(t, budget=b, features=t.get("features", {})) for t in mine for b in budgets]
        rows += [dict(set=r["set"], target_id=r["target_id"], budget=r["budget"], p=float(q), model_sha256=m["sha256"],
                      **({"kind": r["kind"]} if "kind" in r else {}))
                 for r, q in zip(recs, predictor.predict(m, recs))]
    rows.sort(key=lambda r: (r["set"], r["target_id"], r["budget"]))
    return json.dumps(dict(meta or {}, model_sha256=shas.pop(), predictions=rows), indent=1, sort_keys=True)


def write_predictions(model, targets, budgets, path, meta=None):
    """targets: dicts with set, target_id and units (and domain when `model` is by domain). File + .sha256. An existing
    file with other content is never replaced (guardrail row 132): the registered predictions change only by a recorded
    decision and an OSF update, and the old file is then removed by hand."""
    path = Path(path)
    body = prediction_body(model, targets, budgets, meta)
    if path.exists() and path.read_text() != body:
        raise RuntimeError(f"{path.name} exists with other content: a registered prediction file is never replaced. If "
                           "it must change, record a decision and an OSF update, and remove it by hand.")
    path.write_text(body)
    path.with_suffix(".sha256").write_text(f"{hashlib.sha256(body.encode()).hexdigest()}  {path.name}\n")
    return path

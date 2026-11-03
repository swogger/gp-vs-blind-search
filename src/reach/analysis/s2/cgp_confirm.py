"""H10a and H10b: Cartesian GP in the confirmatory analysis (Stage 2 draft; D072, D076; REQ-ANA-26, REQ-ANA-27).

H10a (manipulation check, as H1): for every calibration set, the logistic slope of exact recovery on minimum size in
nodes, with a 95% target-bootstrap CI (`law.slope_ci`, the H1 code), on CGP rows of s2_cal_cgp, at E0 and 16 x E0
(both read from the same 16 x E0 runs by prefix). A cell supports H10a when it is estimable and its CI lies below 0;
a saturated cell (all or no runs exact) is reported as untestable, never dropped. Multiplicity as for H1: Holm per
domain over the 32 registered cells (`law.holm_cells`, the H1 function; real 14, Boolean 18), untestable members counted;
the registered decision is `holm_supported` (`supports` is the cell's unadjusted CI, kept as before). Boolean CGP
saturates at E0, so H10a is also tested for the 9 Boolean sets at 1/16 x E0, read by prefix from the same runs, as a
separate Holm family (D083; `h10a_sixteenth`); real CGP gets none.

H10b (transfer of the frozen law): per domain, logit P = c0 + c1 * eta, with eta the frozen H7 model's linear
predictor, fitted by maximum likelihood on all CGP calibration records at 1, 2, 4, 8, 16 x E0 (registered rule; no other
choice). The fit is a model of the same form as the frozen one (c0 + c1 * a, c1 * b, c1 * g, ...), so
`predictor.predict` and `predictor.score` read it unchanged. It is frozen with SHA-256 (data/registration/
cgp_transfer.json) before any held-out CGP run (s2_ho_cgp refuses to plan without it) and scored on s2_ho_cgp at the same
budgets with the H7 criteria: AUC >= 0.80; calibration slope in [0.7, 1.3] with an intercept CI covering 0; Brier score
below a baseline of nodes and log2 budget fitted on the CGP calibration records and frozen with the model (D090);
per-set absolute error <= 0.10 in at least 5 of 6 sets; intercept CI with the registered number of target-bootstrap
replicates (confirm.REGISTERED, 2,000). The frozen tree model is not changed; the fit is `transfer.fit` (shared with
H9b). The transfer file is checked for content (both domains, a valid c0 + c1 *
eta model each) and is never replaced with other content (guardrail row 91).
"""
from collections import defaultdict

from reach.analysis.s2 import law, predictor, transfer
from reach.analysis.s2.confirm import REGISTERED
from reach.analysis.s2.load import tidy

CAL_PHASE, HO_PHASE = "s2_cal_cgp", "s2_ho_cgp"
H10A_BUDGETS = (1, 16)
FIT_BUDGETS = (1, 2, 4, 8, 16)            # = confirm.H4_BUDGETS (the H7 budgets)
EPS_SET, MIN_SETS_OK = predictor.H7_RULE["eps_set"], predictor.SCORED["H10b"]["min_sets_ok"]   # H7 criteria
TRANSFER_FILE = "data/registration/cgp_transfer.json"


def _only_cgp(items):
    other = {r.get("algorithm") for r in items} - {"CGP"}
    if other:
        raise ValueError(f"H10 reads CGP rows only; found algorithm(s) {sorted(map(str, other))}")


def records(rows, budgets):
    """CGP rows -> tidy records (CPU stops and kills count as failures, the primary convention)."""
    _only_cgp(rows)
    return tidy(rows, budgets, censored_as_failure=True)


def declared_h10a_cells():
    """The H10a Holm family (Stage 2 draft): every calibration set x CGP x E0 and 16 x E0 (32 cells)."""
    from reach.runner.phases import CAL_SETS
    return [(g, "CGP", None, b) for g in CAL_SETS for b in H10A_BUDGETS]


def h10a(rows, boot=2000, declared="registered"):
    """declared: "registered" (the 32 cells), a list of cell keys, or None (the cells present; DRYRUN only)."""
    from reach import grammar as G
    recs = records(rows, H10A_BUDGETS)
    cells, summary = [], defaultdict(lambda: dict(cells=0, supported=0, untestable=0))
    by = defaultdict(list)
    for r in recs:
        by[(r["set"], r["budget"])].append(r)
    for (g, b) in sorted(by):
        c = law.slope_ci(by[(g, b)], unit="nodes", boot=boot)
        dom = G.get(g).kind
        cell = dict(set=g, domain=dom, algorithm="CGP", budget=b, unit="nodes", **c)
        cell["untestable"] = not c["estimable"]
        cell["supports"] = bool(c["estimable"] and c["hi"] < 0)
        cells.append(cell)
        s = summary[dom]
        s["cells"] += 1
        s["supported"] += cell["supports"]
        s["untestable"] += cell["untestable"]
    fam = declared_h10a_cells() if declared == "registered" else declared
    holm = law.holm_cells(cells, lambda g: G.get(g).kind, REGISTERED["alpha"], declared=fam)
    return dict(cells=cells, summary=dict(summary), budgets=list(H10A_BUDGETS), holm=holm,
                separation=law.separation_count(cells, boot))                 # D088: separated cells, replicates


H10A_SIXTEENTH = 1 / 16                    # D083: Boolean CGP saturates at E0; 1/16 x E0 = 1,250 evaluations


def declared_h10a_sixteenth_cells():
    """The separate Holm family of D083 (Stage 2): every Boolean calibration set x CGP x 1/16 x E0 (9 cells)."""
    from reach import grammar as G
    from reach.runner.phases import CAL_SETS
    return [(g, "CGP", None, H10A_SIXTEENTH) for g in CAL_SETS if G.get(g).kind == "bool"]


def h10a_sixteenth(rows, boot=2000, declared="registered"):
    """H10a for Boolean CGP at 1/16 x E0 (D083): the H10a test (`law.slope_ci` on nodes, the H1 code) per Boolean
    calibration set, read by prefix from the s2_cal_cgp runs, with its own Holm family over the 9 Boolean sets
    (`law.holm_cells`, CI-inversion p-values; saturated, not-estimable and missing cells are untestable members). Real
    rows get no cells (real CGP is not saturated at E0). H10a's 32-cell families are not touched."""
    from reach import grammar as G
    _only_cgp(rows)
    recs = records([r for r in rows if G.get(r["set"]).kind == "bool"], (H10A_SIXTEENTH,))
    by = defaultdict(list)
    for r in recs:
        by[r["set"]].append(r)
    cells = []
    for g in sorted(by):
        c = law.slope_ci(by[g], unit="nodes", boot=boot)
        cells.append(dict(set=g, domain="bool", algorithm="CGP", budget=H10A_SIXTEENTH, unit="nodes", **c,
                          untestable=not c["estimable"], supports=bool(c["estimable"] and c["hi"] < 0)))
    fam = declared_h10a_sixteenth_cells() if declared == "registered" else declared
    holm = law.holm_cells(cells, lambda g: G.get(g).kind, REGISTERED["alpha"], declared=fam)
    return dict(cells=cells, budget=H10A_SIXTEENTH, holm=holm, separation=law.separation_count(cells, boot),
                rule="H10a at 1/16 x E0, Boolean CGP only; its own Holm family over the 9 Boolean sets (D083)")


def linear_predictor(model, recs):
    return transfer.linear_predictor(model, recs)


def fit_transfer(frozen, cal_recs):
    """The registered rule: logit P = c0 + c1 * eta by maximum likelihood on CGP records at 1-16 x E0, with its Brier
    baseline (nodes + log2 budget, D090) fitted on the same records and frozen with it. Separated data (`k50.separated`
    on eta) have no finite c0 and c1, only a 0/1 step on eta that is not a model of the frozen form: refused
    (ValueError; guardrail row 118, D088), never frozen with the large finite values an unchecked fit gives."""
    _only_cgp(cal_recs)
    return transfer.fit(frozen, cal_recs, FIT_BUDGETS, "H10b transfer, CGP", baseline_budget_term=True)


DOMAINS = ("real", "bool")


def check_transfer(models):
    """Raise ValueError unless `models` holds exactly both domains, each a transfer model of the frozen form: a known
    unit, the fit budgets, finite c0 and c1, params = [c0 + c1 * b0, c1 * b1, ...] of its base params (row 91), and
    its frozen baseline, nodes + log2 budget (D090)."""
    transfer.check(models, DOMAINS, FIT_BUDGETS, "H10b transfer model", baseline_budget_term=True)


def freeze_transfer(models, path):
    """Both domains in one file with its .sha256 (predictor.freeze); s2_ho_cgp checks it before planning. A model that
    fails `check_transfer` is refused; an existing file with other content is never replaced (row 91)."""
    return transfer.freeze(models, path, check_transfer)


def h10b(transfer, cal_recs, ho_recs, boot=None):
    """Score the frozen transfer model on held-out CGP records with the H7 criteria (boot: the registered 2,000),
    against the baseline frozen with it (D090; cal_recs are checked to be CGP rows only)."""
    boot = boot or REGISTERED["boot"]
    _only_cgp(cal_recs)
    _only_cgp(ho_recs)
    ho = [r for r in ho_recs if r["budget"] in FIT_BUDGETS]
    base = transfer["baseline"]                       # frozen with the transfer model (D090), never refitted here
    res = predictor.score(transfer, ho, base, EPS_SET, boot=boot, min_sets_ok=MIN_SETS_OK)
    return dict(res, baseline=base, criteria=dict(auc=0.80, slope=[0.7, 1.3], eps_set=EPS_SET, min_sets_ok=MIN_SETS_OK),
                n=len(ho), transfer=transfer.get("transfer"))


def load(root, phase, targets_name, budgets):
    """Records of a CGP phase with units from the target files (confirm's helpers; killed rows keep their units)."""
    from reach.analysis.s2 import confirm
    from reach.analysis.s2.load import read_phase
    rows = read_phase(phase, root)
    recs = records(rows, budgets)
    confirm._load_units(root, sorted({r["set"] for r in recs}), targets_name)
    return confirm._with_units(recs)


def attach_features(root, recs, model):
    """A frozen model with a set-level feature needs it on every record (confirm.set_features, no runs needed)."""
    if model.get("feature"):
        from reach.analysis.s2 import confirm
        feats = confirm.set_features(root, sorted({r["set"] for r in recs}))
        for r in recs:
            r["features"] = feats[r["set"]]
    return recs


def by_domain(recs):
    from reach import grammar as G
    out = defaultdict(list)
    for r in recs:
        out[G.get(r["set"]).kind].append(r)
    return out

"""Shared, self-tested instruments for the findings suite.

Every loader reads raw files with the csv module (strings), converts with Python float(),
and never mixes precisions: comparisons against the trajectory harness (6-decimal logs)
round the other side to 6 decimals first. `self_tests()` runs positive and negative controls
for every instrument; the suite refuses to report findings if any control fails.
"""
from __future__ import annotations

import csv, json, math, os, re, subprocess, sys, unicodedata, urllib.parse, urllib.request, zipfile
from collections import defaultdict
from functools import lru_cache

import numpy as np
from scipy.stats import wilcoxon, fisher_exact

csv.field_size_limit(sys.maxsize)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "results")
RERUN = os.path.join(ROOT, "results_rerun")
CACHE = os.path.join(HERE, "cache")
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "ga-lib"))
sys.path.insert(0, os.path.join(ROOT, "research", "synthetic"))
CELLS = [(g, p) for g in (100, 200, 500) for p in (100, 200, 500)]
TESTS = ["1.1", "1.2", "1.3", "1.4", "2.1", "3.1", "3.2", "3.3", "3.4", "3.5", "4.1", "4.2"]
BLOAT = ["1.4", "2.1", "3.2", "3.3", "3.4", "3.5", "4.1", "4.2"]


# ------------------------------------------------------------------ data loaders
@lru_cache(None)
def main_grid(regime):
    """(algo, G, P, test) -> list of (status, nodes, error-string) for runs 1..30 in order."""
    d = "standard_gp" if regime == "A" else "mutation_only"
    out = defaultdict(list)
    for G, P in CELLS:
        for r in range(1, 31):
            for row in csv.DictReader(open(os.path.join(RES, d, f"results_{G}_{P}_{r}.csv"))):
                for algo, pre in (("std", "Std"), ("eml", "EML")):
                    out[(algo, G, P, row["Test ID"])].append((row[f"{pre} Status"], row[f"{pre} Nodes"], row[f"{pre} Error"]))
    return out


def grid_rate(regime, algo, G, P, t):
    return sum(s == "CONVERGED" for s, _, _ in main_grid(regime)[(algo, G, P, t)]) / 30


@lru_cache(None)
def traj(regime, algo, t, G=200, P=100, r=1):
    """Trajectory-harness file -> list of (gen, best, median_size); values are 6-decimal logs."""
    d = "standard_gp" if regime == "A" else "mutation_only"
    f = os.path.join(RES, d, "convergence_data", f"convergence_{t}_{algo}_{G}_{P}_{r:02d}.csv")
    return tuple((int(x["generation"]), float(x["best_fitness"]), int(x["median_tree_size"])) for x in csv.DictReader(open(f)))


def traj_final(regime, algo, t, G=200, P=100, r=1):
    return traj(regime, algo, t, G, P, r)[-1][1]


@lru_cache(None)
def rows(path):
    return tuple(csv.DictReader(open(path)))


def synth_orig():
    return rows(os.path.join(RES, "synthetic_dense_k", "synthetic_runs.csv"))


def synth_rerun():
    return rows(os.path.join(RERUN, "P1_synth_main.csv"))


def control():
    return rows(os.path.join(RES, "fixed_set_control", "control_runs.csv"))


def read_code(rel):
    return open(os.path.join(ROOT, rel)).read()


# ------------------------------------------------------------------ statistics
def holm(ps):
    order = np.argsort(ps); m = len(ps); out = [0.0] * m; run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i])); out[i] = run
    return out


def wilcoxon_p(d):
    d = np.asarray(d, float); nz = d[d != 0]
    return wilcoxon(nz).pvalue if len(nz) else 1.0


def fisher_p(k1, n1, k2, n2):
    return fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]


def logit_k50(K, y, iters=300):
    X = np.column_stack([np.ones(len(K)), np.asarray(K, float)]); b = np.zeros(2); y = np.asarray(y, float)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b)); W = np.clip(p * (1 - p), 1e-9, None)
        nb = np.linalg.solve((X * W[:, None]).T @ X, (X * W[:, None]).T @ (X @ b + (y - p) / W))
        if np.max(np.abs(nb - b)) < 1e-12:
            b = nb; break
        b = nb
    return -b[0] / b[1]


# ------------------------------------------------------------------ functional equivalence
def _parse_formula(s):
    from node import Node
    s = s.strip(); pos = 0

    def expr():
        nonlocal pos
        if s[pos] == "(":
            pos += 1; left = expr(); pos += 1
            j = s.index(" ", pos); op = s[pos:j]; pos = j + 1
            right = expr(); assert s[pos] == ")"; pos += 1
            return Node(op, left, right, True)
        j = pos
        while j < len(s) and s[j] not in "() ":
            j += 1
        tok = s[pos:j]
        if j < len(s) and s[j] == "(" and tok in Node.UNARY_OPS:
            pos = j + 1; child = expr(); assert s[pos] == ")"; pos += 1
            return Node(tok, child, None, True)
        pos = j
        try:
            return Node(float(tok))
        except ValueError:
            return Node(tok)

    t = expr(); assert pos == len(s)
    return t


def equivalent(formula, test_id):
    """True iff |pred - f| <= 1e-9 max(1,|f|) on a dense grid offset off rational points."""
    from test_suite import TEST_MATRIX
    tm = TEST_MATRIX[test_id]; lo, hi = tm["range"]; vs = tm["vars"]; tree = _parse_formula(formula)
    if len(vs) == 1:
        pts = [{vs[0]: lo + (hi - lo) * (i + 0.6180339887) / 1000} for i in range(1000)]
    else:
        g = [lo + (hi - lo) * (i + 0.6180339887) / 40 for i in range(40)]
        pts = [{vs[0]: a, vs[1]: b} for a in g for b in g]
    for kw in pts:
        y = tm["func"](*[kw[v] for v in vs])
        try:
            e = abs(tree.evaluate(**kw) - y) / max(1.0, abs(y))
        except Exception:
            return False
        if not (e <= 1e-9):
            return False
    return True


# ------------------------------------------------------------------ text sources
def normalise(text, keep_line_hyphens=False):
    """Lower-case, NFKC, unified dashes and quotes, collapsed whitespace. Words broken across lines
    are re-joined without the hyphen (default) or keeping it (for compounds such as EML-GP)."""
    t = unicodedata.normalize("NFKC", text)
    for a_, b_ in ((chr(0x2212), "-"), (chr(0x2013), "-"), (chr(0x2014), "-"), (chr(0x00ad), ""),
                   (chr(0x2019), "'"), (chr(0x2018), "'"), (chr(0x201c), '"'), (chr(0x201d), '"')):
        t = t.replace(a_, b_)
    t = re.sub(r"-\s*\n\s*", "-" if keep_line_hyphens else "", t)
    t = re.sub(r"\s+", " ", t)
    return t.lower()


def read_document(path):
    if path.lower().endswith(".pdf"):
        return subprocess.run(["pdftotext", path, "-"], capture_output=True, text=True).stdout
    if path.lower().endswith(".docx"):
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf8")
        xml = re.sub(r"</w:p>", "\n", xml)
        return re.sub(r"<[^>]+>", "", xml)
    return open(path, encoding="utf8").read()


@lru_cache(None)
def odrz(part):
    return normalise(open(os.path.join(CACHE, f"odrzywolek_2603.21852v2_{part}.txt")).read())


def crossref(doi, online=False):
    cache_f = os.path.join(CACHE, "doi_metadata.json")
    cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
    if doi in cache and not online:
        return cache[doi]
    hdr = {"User-Agent": "findings-suite/1.0"}
    if doi.startswith(("10.48550", "10.5281")):
        url = "https://api.datacite.org/dois/" + urllib.parse.quote(doi, safe="")
        a = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=20))["data"]["attributes"]
        meta = {"title": a["titles"][0]["title"], "page": "", "year": a.get("publicationYear")}
    else:
        url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
        m = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=20))["message"]
        meta = {"title": (m.get("title") or [""])[0], "page": m.get("page", ""),
                "year": ((m.get("issued", {}).get("date-parts") or [[None]])[0][0])}
    cache[doi] = meta
    json.dump(cache, open(cache_f, "w"), indent=1)
    return meta


# ------------------------------------------------------------------ self-tests (positive and negative controls)
def self_tests():
    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    # 1. float parsing: string round-trip through csv + float() is exact
    r = synth_rerun()[0]
    check("I1 csv/float round-trip", float(repr(float(r["final_nmse"]))) == float(r["final_nmse"]))
    # 2. functional-equivalence evaluator: exact, removable-singularity and approximate controls
    check("I2a evaluator accepts exact tree", equivalent("(x1 + x2)", "1.3"))
    check("I2b evaluator accepts removable singularity off-grid",
          equivalent("((x1 + x2) + ((x1 - x1) / x1))", "1.3"))   # wrong only at x1 = 0, which the offset grid never hits
    check("I2c evaluator rejects 1e-6 perturbation", not equivalent("((x1 + x2) + 0.000001)", "1.3"))
    check("I2d evaluator accepts exact EML ln", equivalent("(1.0 eml ((1.0 eml x) eml 1.0))", "1.2"))
    # 3. Holm against statsmodels
    from statsmodels.stats.multitest import multipletests
    ps = list(np.random.default_rng(0).uniform(0, 0.2, 12))
    check("I3 Holm == statsmodels", np.allclose(holm(ps), multipletests(ps, method="holm")[1]))
    # 4. Wilcoxon: 10 positive differences, exact two-sided p = 2/1024
    check("I4 Wilcoxon exact control", abs(wilcoxon_p(np.arange(1, 11)) - 2 / 1024) < 1e-12)
    # 5. K50: recover a simulated midpoint, and reproduce the published original fit (8.46)
    rng = np.random.default_rng(1); K = rng.choice(np.arange(3, 32, 2), 20000)
    y = rng.random(20000) < 1 / (1 + np.exp(-(4.0 - 0.4 * K)))
    check("I5a K50 recovers simulated 10.0", abs(logit_k50(K, y) - 10.0) < 0.2)
    so = [x for x in synth_orig() if x["regime"] == "A"]
    k = logit_k50([float(x["K_construction"]) for x in so], [x["converged"] == "true" for x in so])
    check("I5b K50 reproduces published 8.46", abs(k - 8.46) < 0.01, f"{k:.3f}")
    # 6. main-grid and trajectory readers reproduce known published values
    check("I6 main-grid reader: Table 4 ExpComp G200P100 = 0.47", abs(grid_rate("A", "std", 200, 100, "1.4") - 14 / 30) < 1e-9)
    check("I7 trajectory reader: Table 10 arctan A median = 0.406",
          abs(np.median([traj_final("A", "eml", "3.2", r=i) for i in range(1, 31)]) - 0.4059) < 1e-3)
    # 8. text extraction of [9] finds strings we know are there
    t = odrz("main")
    check("I8 [9] text contains known strings", "36 primitives" in t and "448 attempts" in t)
    # 9. DOI metadata: a known-good DOI matches its title
    try:
        check("I9 DOI lookup control", "distilling free-form" in crossref("10.1126/science.1165893")["title"].lower())
    except Exception as e:
        check("I9 DOI lookup control", False, repr(e))
    # 10. manuscript normaliser joins hyphenated line breaks and unifies minus signs
    check("I10 normaliser", normalise("conver-\n gence " + chr(0x2212) + " x") == "convergence - x"
          and normalise("EML-\nGP doesn" + chr(0x2019) + "t", True) == "eml-gp doesn't")
    return results

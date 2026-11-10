"""Stage 5 analysis driver (D125; REQ-ANA-120 to REQ-ANA-128; docs/design/stage5_analysis.md; the draft form
docs/registration/osf/stage5_form.md). CONFIRMATORY once the Stage 5 registration is recorded.

    python -m reach.analysis.s2.stage5_report [--families H12,H13,...] [--boot B] [--root DIR] [--out DIR]
    python -m reach.analysis.s2.stage5_report --pilot [--boot B] [--root DIR] [--out DIR]     (stage5_pilot)

Refuses (RuntimeError) unless every phase a requested family reads holds exactly its manifest's tasks, and each Stage
5 phase's task list matches data/registration/stage5_tasks.json once it exists (REQ-ANA-120); every re-execution row
reproduces its original and every long run its s2_ho_a0 prefix (REQ-ANA-121); every target file matches its checksum,
every row its target file, every counted set one (registered) lookup (REQ-ANA-122); and every contrast's arms share
their (set, level, target) rows (Rule S1, REQ-ANA-123). Instrument checks: the base arms reproduce the registered
values they share (REQ-ANA-124). Then the twelve tests (`stage5_stats`), Holm per family, the stage-wide Holm and the
set-level intervals beside; writes data/analysis/stage5.json and docs/status/stage5_report.md.
"""
import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from reach import s2_lock as L
from reach.analysis.s2 import h6d, stage5
from reach.analysis.s2 import stage5_stats as S
from reach.analysis.s2.load import read_phase

REAL_MAIN = ("F-EML", "F-EML-R", "F-LDE0", "R-ARITH4", "R-ELEM4", "R-HYB", "R-TRIG")
BOOL_MAIN = ("BOOL-AON-n5", "BOOL-AONX-n5", "BOOL-IMP0-n5", "BOOL-IMP0R-n5", "BOOL-NAND-n5", "BOOL-NOR-n5",
             "BOOL-RICH-n5")
HELDOUT = ("R-ARITH", "R-ELEM6", "RS-add_sqrt", "RS-eml_sin_sqrt_square", "RS-mul_div_eml_sin_square",
           "RS-mul_eml_sin")
ROBUST_REAL, ROBUST_BOOL = ("F-EML", "F-LDE0", "R-TRIG"), ("BOOL-NAND-n5", "BOOL-NOR-n5", "BOOL-RICH-n5")
ROBUST_N = 20                     # the robustness targets: the first 20 of each budget level's 30
CUT_BITS = math.log2(16 * 20_000)  # S_d: calibration levels at or below 18.29 bits
UNION_PREFIX = "10000"            # the restart twin reads the histograms at 100 generations

PHASES = dict(grid="s5_grid", wide="s5_wide", boolsub="s5_boolsub", frozen="s5_frozen", selfree="s5_selfree",
              low_base="s5_low_base", pop1000="s5_pop1000", revisit_restart="s5_revisit_restart", union="s5_union",
              revisit_robust="s5_revisit_robust", long_extra="s5_long_extra", revisit_ho16="s5_revisit_ho16",
              base_a0="s2_cal_a0", base_a16="s2_cal_a16", stage3_a0="s2_revisit_a0", stage3_a16="s2_revisit_a16",
              restart="s4_restart", one16="s2_ho_a16", robust="s2_cal_robust")
# The phases each family reads (s5_frozen is read beside H12 when planned; a family is analysed once all are complete).
FAMILY_PHASES = dict(H12=("grid", "wide", "boolsub", "base_a0", "base_a16"),
                     H13=("selfree", "low_base", "base_a0", "base_a16", "stage3_a0", "stage3_a16"),
                     H14=("pop1000", "restart"),
                     H15=("restart", "revisit_restart", "union", "long_extra", "revisit_ho16", "one16"),
                     H16=("revisit_robust", "robust", "stage3_a16"),
                     **{"H11-2L": ("restart", "long_extra", "one16")})
REEXEC = dict(revisit_restart=True, revisit_robust=False, revisit_ho16=False, stage3_a0=False, stage3_a16=False)
REGISTRATION = (("s5_", "stage5_tasks.json"), ("s4_", "stage4_tasks.json"), ("s2_revisit_", "stage3_tasks.json"),
                ("s2_ho_", "stage2_tasks.json"), ("s2_cal_", "stage1_tasks.json"))
CONF = dict(label="STAGE 5 (H11-2L; confirmatory once registered)", out="stage5", confirmatory=True,
            registered=True, registered_families=("H11-2L",),       # D126: the other families are paused
            instrument=True, phases=PHASES, real=REAL_MAIN, boolean=BOOL_MAIN, heldout=HELDOUT,
            robust_real=ROBUST_REAL, robust_bool=ROBUST_BOOL, wide_excluded=None)


# ---------------------------------------------------------------- refusals (REQ-ANA-120 to 122)
def _registration(root, phase):
    for prefix, name in REGISTRATION:
        if phase.startswith(prefix) and not phase.startswith("s5_pilot_"):
            p = Path(root) / "data/registration" / name
            return (p if p.exists() else None), prefix == "s5_"
    return None, False


def check_complete(root, phase, rows, strict=False):
    """Refuse unless the rows' keys are exactly the manifest's (no missing, no extra row), and, once the phase's
    stage registration exists, the manifest is the registered task list (count and key hash; full hash for Stage 5).
    strict (the registered driver; guardrail row 204): the stage's task-list file must exist and name the phase."""
    from reach.runner.phases import _manifest_tasks
    try:
        tasks = _manifest_tasks(phase, root)
    except FileNotFoundError:
        raise RuntimeError(f"{phase}: not planned (no manifest)")
    want, have = {t["key"] for t in tasks}, {r["key"] for r in rows}
    if want != have:
        raise RuntimeError(f"{phase}: {len(want - have)} planned tasks without a row (e.g. {sorted(want - have)[:2]}), "
                           f"{len(have - want)} rows of no planned task (e.g. {sorted(have - want)[:2]}): incomplete "
                           "or extra phase")
    reg_file, stage5_phase = _registration(root, phase)
    if strict and reg_file is None and _registration_name(phase):
        raise RuntimeError(f"{phase}: data/registration/{_registration_name(phase)} is missing")
    if reg_file is not None:
        reg = json.loads(reg_file.read_text())
        if phase not in reg:
            if stage5_phase or strict:
                raise RuntimeError(f"{phase}: not in {reg_file.name}")
        else:
            e = reg[phase]
            if e["n_tasks"] != len(tasks) or e["sha256"] != L.task_list_hash(tasks):
                raise RuntimeError(f"{phase}: the task list is not the one registered in {reg_file.name}")
            if (stage5_phase or strict) and "sha256_full" in e and e["sha256_full"] != L.task_list_full_hash(tasks):
                raise RuntimeError(f"{phase}: the same keys, but settings differ from {reg_file.name} (full hash)")
    return tasks


def _registration_name(phase):
    for prefix, name in REGISTRATION:
        if phase.startswith(prefix) and not phase.startswith("s5_pilot_"):
            return name
    return None


def _prefix_sources(root, phase):
    """The registered shorter-run phases the long-run tasks name (meta prefix_of)."""
    from reach.runner.phases import _manifest_tasks
    return sorted({t["meta"]["prefix_of"] for t in _manifest_tasks(phase, root) if "prefix_of" in t.get("meta", {})})


def check_registration(root):
    """The registered driver (guardrail row 204): the Stage 5 registration is recorded, its task list and checksum
    list exist, and every registration file the checksum list names (stage5_tasks.json at least) matches it."""
    reg = Path(root) / "data/registration"
    f = reg / "stage5_registration.txt"
    if not (f.exists() and f.read_text().strip()):
        raise RuntimeError("the Stage 5 registration is not recorded in data/registration/stage5_registration.txt")
    for name in ("stage5_tasks.json", "stage5_checksums.txt"):
        if not (reg / name).exists():
            raise RuntimeError(f"data/registration/{name} is missing")
    sums = registered_checksums(root)
    if "data/registration/stage5_tasks.json" not in sums:
        raise RuntimeError("stage5_tasks.json is not listed in stage5_checksums.txt")
    for rel, want in sums.items():
        if rel.startswith("data/registration/") and (Path(root) / rel).exists() and _sha(Path(root) / rel) != want:
            raise RuntimeError(f"{rel}: SHA-256 differs from stage5_checksums.txt")


def phase_rows(root, phase, strict=False):
    rows = read_phase(phase, root)
    check_complete(root, phase, rows, strict)
    return rows


def frozen_rows(root, phase):
    """s5_frozen (FB, descriptive beside H12-1; last in the priority order): its rows when complete; (None, note) when
    not planned or still running (a missing row omits FB, it does not hold up the tests); a row of no planned task is
    refused."""
    from reach.runner.phases import _manifest_tasks
    try:
        want = {t["key"] for t in _manifest_tasks(phase, root)}
    except FileNotFoundError:
        return None, "not planned"
    rows = read_phase(phase, root)
    have = {r["key"] for r in rows}
    if have - want:
        raise RuntimeError(f"{phase}: {len(have - want)} rows of no planned task (extra phase)")
    if want - have:
        return None, f"incomplete ({len(have)} of {len(want)} rows): FB omitted"
    check_complete(root, phase, rows)
    return rows, "complete"


def check_reexec(root, phase, keys=False):
    """Every re-execution row reproduces its original (`audit.revisit_report`; side files where the task names one)."""
    from reach.analysis.s2.audit import revisit_report
    rep = revisit_report(root, phase, extra_ignore=("keys",), check_keys=True) if keys or phase.startswith("s5_") \
        else revisit_report(root, phase)
    if not rep["ok"]:
        bad = {k: v for e in rep["sources"].values() for k, v in e["differing"].items()}
        miss = sum(len(e["missing"]) + len(e.get("missing_original", [])) for e in rep["sources"].values())
        raise RuntimeError(f"{phase}: {len(bad)} re-executed rows do not reproduce their originals, {miss} missing "
                           f"(first: {list(bad.items())[:2]})")
    return dict(rows=sum(e["rerun"] for e in rep["sources"].values()),
                identical=sum(e["identical"] for e in rep["sources"].values()))


def check_long_extra(root, phase):
    from reach.analysis.s2.audit import long_extra_report
    rep = long_extra_report(root, phase)
    if not rep["ok"]:
        raise RuntimeError(f"{phase}: {len(rep['mismatch'])} long runs differ from their s2_ho_a0 prefix, "
                           f"{len(rep['missing'])} missing (first: {list(rep['mismatch'].items())[:2]})")
    return dict(compared=rep["compared"], not_comparable=len(rep["not_comparable"]), no_prefix=rep["no_prefix"])


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def target_table(root, g, names, checksums=None):
    """{target_id: target} over the set's target files `names` that exist, each checked against its .sha256 (and the
    registered checksum list when given, {path: sha}); also returns {name: sha}."""
    out, shas = {}, {}
    for name in names:
        p = Path(root) / f"data/targets_s2/{g}/{name}.json"
        if not p.exists():
            continue
        got = _sha(p)
        want = (p.parent / f"{name}.sha256").read_text().split()[0]
        if got != want:
            raise RuntimeError(f"{p}: SHA-256 {got[:12]} differs from {name}.sha256 ({want[:12]})")
        rel = f"data/targets_s2/{g}/{name}.json"
        if checksums and rel in checksums and checksums[rel] != got:
            raise RuntimeError(f"{rel}: SHA-256 differs from the registered checksum list")
        shas[name] = got
        for t in json.loads(p.read_text()):
            if t["target_id"] in out and out[t["target_id"]]["units"] != t["units"]:
                raise RuntimeError(f"{g} {t['target_id']}: in two target files with other units")
            out[t["target_id"]] = dict(t, _file=name)
    return out, shas


def registered_checksums(root):
    """{path: sha} of data/registration/stage5_checksums.txt once it exists (sha256sum format)."""
    p = Path(root) / "data/registration/stage5_checksums.txt"
    if not p.exists():
        return None
    out = {}
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            out[parts[1].lstrip("*")] = parts[0]
    return out


def proven(t):
    return t["K_proven"] == t["K_construction"] and not t.get("constructed")


def check_rows_targets(rows, tables, shas, require_sha=True):
    """Every row names a target of its set's registered files, and its targets_sha256 is that file's (required on
    every search row; s5_union's rows, from a task without search that reads no target file, have none)."""
    for r in rows:
        t = tables.get(r["set"], {}).get(r["target_id"])
        if t is None:
            raise RuntimeError(f"{r['key']}: {r['target_id']} is not a target of the registered files")
        want = shas[r["set"]].get(t["_file"])
        if not r.get("targets_sha256"):                          # guardrail row 205 (review m3)
            if require_sha:
                raise RuntimeError(f"{r['key']}: the row records no targets_sha256")
            continue
        if r["targets_sha256"] != want:
            raise RuntimeError(f"{r['key']}: run against target file {r['targets_sha256'][:12]}, not the registered "
                               f"{t['_file']}.json ({want[:12]})")


def check_lookups(rows, registered=None, what="rows"):
    """Every counted row of a set used one lookup, the registered one when given ({set: sha})."""
    sha = defaultdict(set)
    for r in rows:
        lk = (r.get("rv") or {}).get("lookup") or r.get("lookup") or {}
        if r.get("rv") and not lk:                               # a counted row (guardrail row 205, review m3)
            raise RuntimeError(f"{what}: counted row {r['key']} records no lookup")
        if lk:
            sha[r["set"]].add(lk.get("sha256"))
    many = sorted(g for g, v in sha.items() if len(v) != 1)
    if many:
        raise RuntimeError(f"{what}: rows of one set used different lookups: {many}")
    if registered:
        off = sorted(g for g, v in sha.items() if registered.get(g) not in v)
        if off:
            raise RuntimeError(f"{what}: lookup SHA-256 differs from the registered one for {off}")
    return {g: next(iter(v)) for g, v in sorted(sha.items())}


def _lookups_file(root, name):
    p = Path(root) / "data/registration" / name
    return json.loads(p.read_text()) if p.exists() else None


def levels(root, sets, given=None):
    """{set: {K: U_K}} from each set's counts.json (or `given`, for tests)."""
    if given is not None:
        return {g: {int(k): v for k, v in given[g].items()} for g in sets}
    from reach.analysis.s2.phase1 import enum_dir
    from reach.paths import ROOT
    out = {}
    for g in sets:
        d = enum_dir(g)
        d = Path(root) / d.relative_to(ROOT) if str(d).startswith(str(ROOT)) else d
        out[g] = h6d.levels_U(d)[0]
    return out


def close_or_refuse(what, got, want, tol=1e-9):
    """Instrument check: every value equal to its registered one within tol, else RuntimeError (REQ-ANA-124)."""
    bad = {g: (got.get(g), w) for g, w in want.items()
           if g not in got or not (math.isfinite(got[g]) and abs(got[g] - w) <= tol)}
    if bad:
        raise RuntimeError(f"instrument check ({what}): differs from the registered values: {bad}")
    return {g: got[g] for g in want}


# ---------------------------------------------------------------- the arms of each family (Rule S1, REQ-ANA-123)
def _by(rows, *keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys) if len(keys) > 1 else r[keys[0]]].append(r)
    return out


def _sizes(table):
    return {t: v["units"]["bits_sem"] for t, v in table.items()}


def paired(name, new, base):
    """Pairing by set, target and seed (H12; each new run names its base run, meta pair_key): refuse otherwise."""
    a = {(r["set"], r["target_id"], r["seed"]) for r in new}
    b = {(r["set"], r["target_id"], r["seed"]) for r in base}
    if a != b:
        raise ValueError(f"{name}: the arms differ in (set, target, seed) pairs: {len(a - b)} only in the new arm, "
                         f"{len(b - a)} only in the base arm")
    keys = {r["key"] for r in base}
    off = [r["key"] for r in new if r.get("pair_key") is not None and r["pair_key"] not in keys]
    if off:
        raise ValueError(f"{name}: {len(off)} new runs name a base run that is not in the base arm (e.g. {off[:1]})")


def assemble_h12(rows, tables, real, boolean, excluded=(), L=200):
    """Blocks of H12: H12-1 (grid), H12-3 (wide, without `excluded` in both arms), H12-2 (BS24, proven targets);
    `rows` {grid, wide, boolsub, base[, frozen]} each a list of rows. Returns [(names, ids, stat, detail)]."""
    excluded = set(excluded)
    base = rows["base"]
    blocks = []
    for name, key, sets, keep in (("H12-1", "grid", real, lambda t: True),
                                  ("H12-3", "wide", real, lambda t: t not in excluded)):
        new = [r for r in rows[key] if r["set"] in sets]
        old = [r for r in base if r["set"] in sets and keep(r["target_id"])]
        paired(name, new, old)
        arms = {k: {g: S.one_run_arm([r for r in rr if r["set"] == g], L, _sizes(tables[g])) for g in sets}
                for k, rr in (("new", new), ("base", old))}
        S.guard(name, arms["new"], arms["base"])
        blocks.append(([name], {g: sorted(arms["new"][g]) for g in sets}, S.h12_contrast(name, arms["new"], arms["base"])))
    if rows.get("frozen"):
        new = [r for r in rows["frozen"] if r["set"] in real]
        old = [r for r in base if r["set"] in real]
        paired("FB", new, old)
        arms = {k: {g: S.one_run_arm([r for r in rr if r["set"] == g], L, _sizes(tables[g])) for g in real}
                for k, rr in (("new", new), ("base", old))}
        S.guard("FB", arms["new"], arms["base"])
        blocks.append((["FB"], {g: sorted(arms["new"][g]) for g in real}, S.h12_contrast("FB", arms["new"], arms["base"])))
    pv = {g: {t for t, v in tables[g].items() if proven(v)} for g in boolean}
    new = [r for r in rows["boolsub"] if r["set"] in boolean and r["target_id"] in pv[r["set"]]]
    old = [r for r in base if r["set"] in boolean and r["target_id"] in pv[r["set"]]]
    paired("H12-2", new, old)
    arm = {g: S.one_run_arm([r for r in new if r["set"] == g], L, _sizes(tables[g])) for g in boolean}
    barm = {g: S.one_run_arm([r for r in old if r["set"] == g], L, _sizes(tables[g])) for g in boolean}
    S.guard("H12-2", arm, barm)
    blocks.append((["H12-2"], {g: sorted(arm[g]) for g in boolean}, S.h12_sign("H12-2", arm, barm)))
    return blocks


def s_d(tables, cal_levels, g, n=30, cut=CUT_BITS):
    """S_d of a set from the target files: the first n calibration targets of each main and budget level whose
    bits_sem is at most the cut, and every low-level target. {target: (bits, K)}."""
    sizes = set(cal_levels[g]["main"]) | set(cal_levels[g]["budget"])
    by = defaultdict(list)
    for t, v in tables[g].items():
        if v["_file"] == "calibration" and v["K_construction"] in sizes:
            by[v["K_construction"]].append(v)
    out = {}
    for K, ts in by.items():
        for v in ts[:n]:
            if v["units"]["bits_sem"] <= cut:
                out[v["target_id"]] = (v["units"]["bits_sem"], v["K_construction"])
    out.update({t: (v["units"]["bits_sem"], v["K_construction"]) for t, v in tables[g].items() if v["_file"] == "low"})
    return out


def assemble_h13(rows, tables, cal_levels, U, real, boolean, L=200):
    """Blocks of H13: H13-1 (SF on S_d, 14 sets) and H13-2 with T (real: SF, the twin from Stage 3's counted runs on
    S_real from the target files, and the base arm on S_real: s2_cal on its calibration levels, s5_low_base on its low
    levels). `rows` {selfree, low_base, base, stage3}."""
    sets = list(real) + list(boolean)
    sd = {g: s_d(tables, cal_levels, g) for g in sets}
    sf = {g: S.one_run_arm([r for r in rows["selfree"] if r["set"] == g], L, _sizes(tables[g])) for g in sets}
    want = {g: {t: (b, 0) for t, (b, _k) in sd[g].items()} for g in sets}
    S.guard("H13 (SF on S_d)", sf, want)
    gp_rows = [r for r in rows["base"] if r["set"] in real and r["target_id"] in sd[r["set"]]] + \
        [r for r in rows["low_base"] if r["set"] in real]
    gp = {g: S.one_run_arm([r for r in gp_rows if r["set"] == g], L, _sizes(tables[g])) for g in real}
    S.guard("T (GP on S_real)", gp, {g: sf[g] for g in real})
    twin_rows = {g: sd[g] for g in real}
    S.guard("H13-2 (twin on S_real)", {g: {t: (b, 0) for t, (b, _k) in twin_rows[g].items()} for g in real},
            {g: sf[g] for g in real})
    cov = {g: _cov(rows["stage3"], g, (S.E0,)) for g in real}
    ids13 = {g: sorted(set(sf[g]) | set(cov[g])) for g in real}
    return [(["H13-1"], {g: sorted(sf[g]) for g in sets}, S.h13_domains("H13-1", sf, real, boolean)),
            (["H13-2"], ids13, S.h13_twin("H13-2", {g: sf[g] for g in real}, twin_rows, cov, U, gp=gp))]


def _cov(rows, g, budgets):
    """{target: [h6d run dicts]} of a set's counted rows."""
    out = defaultdict(list)
    for r in h6d.runs_from_rows([r for r in rows if r["set"] == g], budgets):
        out[r["target_id"]].append(r)
    return dict(out)


def assemble_heldout(rows, tables, U, sets, want=("H14", "H15", "H11-2L"), R=32, L=100, L_long=3200, L_r2=320):
    """The held-out block (H14, H15, H11-2L) on the 55 restart levels. `rows` {restart, pop1000, union, long_extra,
    one16, long_counted}; only the arms of the requested families are built."""
    sizes = {g: {t: v["units"]["bits_sem"] for t, v in tables[g].items() if proven(v)} for g in sets}
    restart_rows = _by([r for r in rows["restart"] if r["set"] in sets], "set", "target_id")
    arms = dict(restart={g: S.restart_arm({t: rr for (h, t), rr in restart_rows.items() if h == g}, sizes[g], R, L)
                         for g in sets})
    rows55 = {g: {t: (sizes[g][t], tables[g][t]["K_construction"]) for t in arms["restart"][g]} for g in sets}
    names = []
    if "H14" in want:
        arms["r2"] = {g: S.one_run_arm([r for r in rows["pop1000"] if r["set"] == g], L_r2, sizes[g]) for g in sets}
        S.guard("H14-1", arms["restart"], arms["r2"])
        names += ["H14-1", "H14-2"]
    if "H11-2L" in want or "H15" in want:
        long_rows = [r for r in rows["one16"] + rows["long_extra"] if r["set"] in sets]
        dup = [k for k, v in _by(long_rows, "set", "target_id", "seed").items() if len(v) > 1]
        if dup:
            raise ValueError(f"H11-2L: {len(dup)} (set, target, seed) runs in both s2_ho_a16 and s5_long_extra")
        arms["long"] = {g: S.one_run_arm([r for r in long_rows if r["set"] == g], L_long, sizes[g]) for g in sets}
        S.guard("H11-2L", arms["restart"], arms["long"])
        names += ["H11-2L"]
    if "H15" in want:
        arms["one"] = {g: S.restart_arm({t: rr for (h, t), rr in restart_rows.items() if h == g}, sizes[g], 1, L)
                       for g in sets}
        arms["rows"] = rows55
        arms["twin"] = {g: union_twins([r for r in rows["union"] if r["set"] == g], rows55[g], U[g], R) for g in sets}
        S.guard("H15 (union rows)", arms["restart"], {g: _on_rows(arms["twin"][g], rows55[g]) for g in sets})
        counted = [r for r in rows["long_counted"] if r["set"] in sets]
        dup = [k for k, v in _by(counted, "set", "target_id").items() if len(v) != 1]
        if dup:
            raise ValueError(f"H15-2: {len(dup)} targets with more than one counted long run (e.g. {dup[:1]})")
        arms["long_cov"] = {g: _cov(counted, g, (S.E16,)) for g in sets}
        S.guard("H15-2 (counted long runs)", arms["restart"],
                {g: _on_rows(arms["long_cov"][g], rows55[g]) for g in sets})
        names += ["H15-1", "H15-2"]
    ids = {g: sorted(arms["restart"][g]) for g in sets}
    return [(names, ids, S.heldout(arms, sets, U))], arms


def _on_rows(by_target, rows55):
    """An arm's rows for Rule S1 from the targets it holds (a target outside the restart rows gets no level, so the
    check refuses it)."""
    return {t: (rows55[t][0] if t in rows55 else float("nan"), 0) for t in by_target}


def union_twins(rows, rows55, U, R, prefix=UNION_PREFIX):
    """{target: (own level K, vector for R restarts, vector for one run)} from the s5_union rows of one set, the
    histograms of the runs without a hit by `prefix` evaluations; every histogram must sum to U_K."""
    out = {}
    for r in rows:
        t = r["target_id"]
        h = r["hist"][prefix]
        n = h["n_nohit"]
        hist = {int(K): v for K, v in h["nohit"].items()}
        for K, v in hist.items():
            if len(v) != n + 1 or sum(v) != U.get(K, -1):
                raise RuntimeError(f"s5_union {t}: histogram of level {K} has {len(v)} cells summing to {sum(v)}, "
                                   f"against n' + 1 = {n + 1} and U_K = {U.get(K)}")
        K_t = rows55[t][1] if t in rows55 else None
        out[t] = (K_t, S.twin_vector(hist, n, U, R), S.twin_vector(hist, n, U, 1))
    return out


def assemble_h16(rows, tables, U, real, boolean, n_targets=ROBUST_N):
    """H16: the counted runs at populations 25 (R1) and 1,000 (R2) of s5_revisit_robust and 100 (s2_revisit_a16) on
    the robustness targets (the first n of each budget level), the same (set, target, seed) in every population."""
    sets = list(real) + list(boolean)
    rr = {25: [r for r in rows["revisit_robust"] if r.get("config") == "R1"],
          1000: [r for r in rows["revisit_robust"] if r.get("config") == "R2"]}
    keys = {(r["set"], r["target_id"], r["seed"]) for r in rr[25]}
    rr[100] = [r for r in rows["stage3_a16"] if (r["set"], r["target_id"], r["seed"]) in keys]
    for p in (100, 1000):
        k2 = {(r["set"], r["target_id"], r["seed"]) for r in rr[p]}
        if k2 != keys:
            raise ValueError(f"H16: population {p} differs from population 25 in {len(k2 ^ keys)} (set, target, seed) "
                             "runs")
    rows_t = {g: {t: (tables[g][t]["units"]["bits_sem"], tables[g][t]["K_construction"])
                  for t in {r["target_id"] for r in rr[25] if r["set"] == g}} for g in sets}
    cov = {p: {g: _cov(rr[p], g, (S.E0,)) for g in sets} for p in S.POPS}
    for p in (100, 1000):
        S.guard(f"H16 (population {p})", {g: {t: (rows_t[g][t][0], 0) for t in cov[25][g]} for g in sets},
                {g: {t: (rows_t[g][t][0], 0) for t in cov[p][g]} for g in sets})
    return [(["H16-1", "H16-2"], {g: sorted(rows_t[g]) for g in sets}, S.h16(cov, rows_t, U, real, boolean))]


# ---------------------------------------------------------------- build
def _sets(cfg, fam):
    return dict(H12=list(cfg["real"]) + list(cfg["boolean"]), H13=list(cfg["real"]) + list(cfg["boolean"]),
                H14=list(cfg["heldout"]), H15=list(cfg["heldout"]), **{"H11-2L": list(cfg["heldout"])},
                H16=list(cfg["robust_real"]) + list(cfg["robust_bool"]))[fam]


def build(root, cfg=CONF, boot=stage5.BOOT, families=None, levels_given=None):
    root = Path(root)
    strict = bool(cfg.get("registered"))
    fams = list(families or (cfg["registered_families"] if strict else S.FAMILIES))
    if strict:                                   # the registered driver (D126; guardrail row 204)
        off = [f for f in fams if f not in cfg["registered_families"]]
        if off:
            raise RuntimeError(f"families {off} are not registered (D126: only {cfg['registered_families']})")
        if boot != stage5.BOOT:
            raise RuntimeError(f"bootstrap of {boot} replicates: the registration fixes {stage5.BOOT}")
        check_registration(root)
    P = cfg["phases"]
    t0 = time.process_time()
    need = sorted({k for f in fams for k in FAMILY_PHASES[f]})
    rows = {k: phase_rows(root, P[k], strict) for k in need}
    if strict and "long_extra" in need:          # the shorter runs the prefix check reads are registered too (row 207)
        for src in _prefix_sources(root, P["long_extra"]):
            phase_rows(root, src, strict)
    checks = dict(phases={P[k]: len(v) for k, v in rows.items()}, reexec={}, long_extra=None, lookups={},
                  instrument={})
    for k in need:
        if k in REEXEC:
            checks["reexec"][P[k]] = check_reexec(root, P[k], keys=REEXEC[k])
    if "long_extra" in need:
        checks["long_extra"] = check_long_extra(root, P["long_extra"])
    reg5, reg3 = _lookups_file(root, "stage5_lookups.json"), _lookups_file(root, "stage3_lookups.json")
    for k in need:
        if k in ("stage3_a0", "stage3_a16"):
            checks["lookups"][P[k]] = check_lookups(rows[k], reg3, P[k])
        elif k in ("revisit_restart", "revisit_robust", "revisit_ho16", "union", "selfree", "low_base", "grid",
                   "wide", "boolsub", "long_extra"):
            checks["lookups"][P[k]] = check_lookups(rows[k], reg5, P[k])
    sums = registered_checksums(root)
    all_sets = sorted({g for f in fams for g in _sets(cfg, f)})
    tables, shas = {}, {}
    for g in all_sets:
        tables[g], shas[g] = target_table(root, g, ("calibration", "low", "heldout", "restart"), sums)
    for k in need:
        check_rows_targets([r for r in rows[k] if r["set"] in tables], tables, shas, require_sha=k != "union")
    U = levels(root, all_sets, levels_given)
    blocks, extra = [], {}
    if "H12" in fams:
        frozen, extra["frozen"] = frozen_rows(root, P["frozen"])
        excluded = cfg["wide_excluded"] if cfg["wide_excluded"] is not None else _wide_excluded()
        base = rows["base_a0"] + rows["base_a16"]
        blocks += assemble_h12(dict(grid=rows["grid"], wide=rows["wide"], boolsub=rows["boolsub"], base=base,
                                    frozen=frozen), tables, cfg["real"], cfg["boolean"], excluded)
        extra["wide_excluded"] = sorted(excluded)
        if cfg["instrument"]:
            checks["instrument"]["H6c real gaps"] = _instrument_h6c(root, base, tables, cfg)
            checks["instrument"]["proven-only Boolean"] = _instrument_bool(root, base, tables, cfg)
    if "H13" in fams:
        cal_levels = json.loads((root / "data/registration/levels.json").read_text())
        blocks += assemble_h13(dict(selfree=rows["selfree"], low_base=rows["low_base"],
                                    base=rows["base_a0"] + rows["base_a16"],
                                    stage3=rows["stage3_a0"] + rows["stage3_a16"]),
                               tables, cal_levels, U, cfg["real"], cfg["boolean"])
        if cfg["instrument"]:
            checks["instrument"]["Stage 3 twins"] = _instrument_twin(root, rows["stage3_a0"] + rows["stage3_a16"],
                                                                     U, cfg)
    ho = [f for f in fams if f in ("H14", "H15", "H11-2L")]
    if ho:
        get = {k: rows.get(k, []) for k in ("restart", "pop1000", "union", "long_extra", "one16")}
        get["long_counted"] = [r for r in rows.get("long_extra", []) if r.get("rv")] + rows.get("revisit_ho16", [])
        bl, arms = assemble_heldout(dict(get), tables, U, cfg["heldout"], want=ho)
        blocks += bl
        if cfg["instrument"]:
            checks["instrument"]["Stage 4 restart K50"] = _instrument_restart(root, arms["restart"])
        extra["long_twin_sensitivity"] = _long_twin_sensitivity(arms, cfg["heldout"], U) if "long_cov" in arms else None
    if "H16" in fams:
        blocks += assemble_h16(dict(revisit_robust=rows["revisit_robust"], stage3_a16=rows["stage3_a16"]), tables, U,
                               cfg["robust_real"], cfg["robust_bool"])
    results, detail = {}, {}
    for names, ids, stat in blocks:
        if any(n in S.TESTS for n in names):
            res, point = S.run_block(names, ids, stat, boot)
        else:                                            # descriptive (FB): the point only
            res, point = {}, stat({g: sorted(v) for g, v in ids.items()})
        results.update({n: v for n, v in res.items() if n in S.TESTS})
        key = "+".join(names)
        detail[key] = dict(per_set=point["per_set"], set_level={n: S.set_level(s) for n, s in
                                                                point["set_level"].items()},
                           values=point["values"], T=point.get("T"))
    out = dict(label=cfg["label"], families=fams, boot=boot, seed=stage5.SEED, checks=checks, tests=results,
               detail=detail, decisions=S.families(results), extra=extra,
               built=time.strftime("%Y-%m-%d %H:%M:%S"), cpu_seconds=round(time.process_time() - t0, 1))
    return out


def _wide_excluded():
    from reach.runner.phases import wide_excluded
    return wide_excluded()


def _registered(root, name):
    p = Path(root) / "data/analysis" / name
    if not p.exists():
        raise RuntimeError(f"instrument check: {p} is missing")
    return json.loads(p.read_text())


def _instrument_h6c(root, base, tables, cfg):
    """H6c's registered real per-set gaps (calibration.json) from the base arm at E0."""
    want = _registered(root, "calibration.json")["h6c"]["gaps"]["real"]["per_set"]
    got = {g: S.reach(*_sp(S.one_run_arm([r for r in base if r["set"] == g], 200, _sizes(tables[g]))))["k50"] - S.NULL_E0
           for g in cfg["real"]}
    return close_or_refuse("H6c real gaps", got, {g: want[g] for g in cfg["real"]})


def _sp(arm):
    ids = sorted(arm)
    return [arm[t][0] for t in ids], [arm[t][1] for t in ids]


def _instrument_bool(root, base, tables, cfg):
    """The deposited proven-only Boolean reading with bracketing (revision_exploratory.json, bool_capped)."""
    want = _registered(root, "revision_exploratory.json")["bool_capped"]["per_set"]
    got = {}
    for g in cfg["boolean"]:
        pv = {t for t, v in tables[g].items() if proven(v)}
        arm = S.one_run_arm([r for r in base if r["set"] == g and r["target_id"] in pv], 200, _sizes(tables[g]))
        got[g] = S.reach(*_sp(arm))["k50"]
    return close_or_refuse("proven-only Boolean K50", got, {g: want[g]["k50_used"] for g in cfg["boolean"]}, 1e-6)


def _instrument_twin(root, stage3, U, cfg):
    """Stage 3's registered twins (h6d.json per set) from the counted runs at E0 by `h6d.set_terms`."""
    want = _registered(root, "h6d.json")["per_set"]["real"]
    want = {g: want[g]["k50_twin"] for g in cfg["real"]}
    got = {g: h6d.set_terms(h6d.runs_from_rows([r for r in stage3 if r["set"] == g], (S.E0,)), S.E0, U[g])["k50_twin"]
           for g in cfg["real"]}
    close_or_refuse("Stage 3 twins", got, want)
    mine = {}                                    # the same twin through H13-2's path: per target, alloc._k50, bracketing
    for g in cfg["real"]:
        cov = _cov(stage3, g, (S.E0,))
        rows = {t: (rr[0]["bits_sem"], rr[0]["K"]) for t, rr in cov.items()}
        probs = S.coverage_probs(cov, sorted(cov), U[g], S.E0, "stratified")
        mine[g] = S.arm_reach(S._twin_arm(rows, probs, sorted(cov)), sorted(cov))["k50"]
    return close_or_refuse("Stage 3 twins through the H13-2 path", mine, want, 1e-6)


def _instrument_restart(root, restart):
    """Stage 4's registered restart K50 per set (h11.json) from the restart arm."""
    want = _registered(root, "h11.json")["h11"]["per_set"]
    got = {g: S.reach(*_sp(arm))["k50"] for g, arm in restart.items()}
    return close_or_refuse("Stage 4 restart K50", got, {g: want[g]["k50"] for g in restart})


def _long_twin_sensitivity(arms, sets, U):
    """Beside H15-2 (point only): the long run's twin K50 under Stage 3's never-hit variant (primary, coordinator
    2026-11-08), its stratified life table (Stage 3's registered primary) and the never-hit mean stratified by target
    level with targets weighted (the restart twin's own rule at R = 1)."""
    out = {}
    for g in sets:
        ids = sorted(arms["restart"][g])
        rows55, cov = arms["rows"][g], arms["long_cov"][g]
        v = {}
        for how in ("never_hit", "stratified"):
            probs = S.coverage_probs(cov, ids, U[g], S.E16, how)
            v[how] = S.arm_reach(S._twin_arm(rows55, probs, ids), ids)["k50"] if probs else math.nan
        by = defaultdict(list)
        for t in ids:
            for r in cov.get(t, ()):
                by[rows55[t][1]].append(r)
        acc, w = None, 0
        for K, runs in by.items():
            lv = h6d.cumulative(runs, S.E16, "lv", "never_hit")
            if hasattr(lv, "__len__"):
                n = sum(1 for t in ids if rows55[t][1] == K)
                acc = n * lv if acc is None else acc + n * lv
                w += n
        if acc is not None and w:
            probs = h6d.twin_probs(acc / w, U[g])
            v["never_hit_by_level"] = S.arm_reach(S._twin_arm(rows55, probs, ids), ids)["k50"]
        out[g] = v
    return out


# ---------------------------------------------------------------- report
def _f(x, d=2, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def _yn(b):
    return "n/a" if b is None else ("yes" if b else "no")


def _margin(name):
    s = S.TESTS[name]
    if "delta" in s:
        return f"{s.get('center', 0.0):g} ± {s['delta']:g}"
    return f"> {s['margin']:+g}"


def render(res):
    dec = res["decisions"]
    L_ = [f"# Stage 5 report: {res['label']}", "",
          f"> Generated by `reach.analysis.s2.stage5_report` ({res['built']}); every number from data/analysis/"
          f"stage5.json. Do not edit by hand. Families {', '.join(res['families'])}; bootstrap {res['boot']} "
          f"replicates, seed {res['seed']}.", "", "## Tests (Holm within each family; Rule S3)", "",
          "| Test | estimate | 90% interval | margin | p | Holm threshold | estimable | supported | isotonic p | "
          "isotonic supported | stage-wide Holm |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in S.TESTS:
        if name not in res["tests"]:
            continue
        fam = dec["families"][S.TESTS[name]["family"]]
        t, ti = fam["primary"][name], fam.get("iso", {}).get(name)
        est = t.get("statistic")
        if isinstance(est, dict):
            est_s = ", ".join(f"{k} {_f(v)}" for k, v in sorted(est.items()))
            iv = "per member"
        else:
            est_s, iv = _f(est), f"[{_f(t.get('lo90'))}, {_f(t.get('hi90'))}]"
            if "center" in t:
                iv = f"[{_f(t['lo90'] + t['center'])}, {_f(t['hi90'] + t['center'])}]" if t.get("lo90") is not None \
                    else iv
        sw = dec["stage_wide"][name]["supported"] if dec.get("stage_wide") else None
        tp, ts = (f"{ti['p']:.4f}", _yn(ti["supported"])) if ti else ("n/a", "n/a")
        L_.append(f"| {name} | {est_s} | {iv} | {_margin(name)} | {t['p']:.4f} | {t['threshold']:.4f} | "
                  f"{_yn(t['estimable'])} | {_yn(t['supported'])} | {tp} | {ts} | {_yn(sw)} |")
    L_ += ["", "Flagged (the isotonic reach would decide otherwise; the registered decision stands): "
           + (", ".join(dec["flagged"]) or "none") + ".",
           "Stage-wide Holm over the registered tests: " + ("reported in the last column." if dec.get("stage_wide")
                                                       else "not computed until every family is analysed."), "",
           "## Set-level intervals (90%, t over sets or Welch; descriptive, never deciding)", ""]
    for key, d in res["detail"].items():
        for name, sl in d["set_level"].items():
            if "mean" in sl:
                L_.append(f"- {name}: {_f(sl['mean'])} [{_f(sl.get('lo'))}, {_f(sl.get('hi'))}]")
            else:
                L_.append(f"- {name}: " + "; ".join(f"{m} {_f(v.get('mean'))} [{_f(v.get('lo'))}, {_f(v.get('hi'))}]"
                                                   for m, v in sl.items()))
    L_ += ["", "## Checks", "", f"- Phases and rows: " + ", ".join(f"{k} {v}" for k, v in res["checks"]["phases"].items()),
           "- Re-executions reproducing their originals: " + (", ".join(
               f"{k} {v['identical']} of {v['rows']}" for k, v in res["checks"]["reexec"].items()) or "none read"),
           f"- Long-run prefixes: {res['checks']['long_extra']}",
           "- Instrument checks passed: " + (", ".join(res["checks"]["instrument"]) or "none (not confirmatory)")]
    return "\n".join(L_) + "\n"


def write(out, res, name="stage5", render_fn=render):
    out = Path(out)
    (out / "data/analysis").mkdir(parents=True, exist_ok=True)
    (out / f"data/analysis/{name}.json").write_text(json.dumps(res, indent=1, default=_json_default) + "\n")
    (out / "docs/status").mkdir(parents=True, exist_ok=True)
    (out / f"docs/status/{name}_report.md" if name == "stage5" else out / f"docs/status/{name}.md").write_text(
        render_fn(res))


def _json_default(x):
    import numpy as np
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (set, tuple)):
        return list(x)
    return str(x)


def main(argv=None):
    from reach.paths import ROOT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--families", default=None, help="comma-separated, e.g. H12,H13 (default: all six)")
    ap.add_argument("--boot", type=int, default=stage5.BOOT)
    ap.add_argument("--root", default=None, help="where the run records are read (default: this checkout)")
    ap.add_argument("--out", default=None, help="where the JSON and the report are written (default: --root)")
    a = ap.parse_args(argv)
    root = Path(a.root or ROOT)
    out = Path(a.out or root)
    if a.pilot:
        from reach.analysis.s2 import stage5_pilot
        res = stage5_pilot.build(root, boot=a.boot)
        write(out, res, "stage5_pilot", stage5_pilot.render)
        print(f"written data/analysis/stage5_pilot.json and docs/status/stage5_pilot.md ({res['label']})")
        return
    fams = a.families.split(",") if a.families else None
    res = build(root, CONF, a.boot, fams)
    write(out, res)
    print("written data/analysis/stage5.json; supported: " + ", ".join(
        f"{n} {res['decisions']['families'][S.TESTS[n]['family']]['primary'][n]['supported']}" for n in res["tests"]))


if __name__ == "__main__":
    main()

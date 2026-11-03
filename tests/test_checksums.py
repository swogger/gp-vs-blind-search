"""Registration checksums and the targets zip (tool T17; REQ-RUN-34): every line verifies, the zip is deterministic
and holds only listed files, and every file and SHA-256 the Stage 2 form names is in the Stage 2 checksums file."""
import hashlib
import importlib.util
import re
import zipfile
from pathlib import Path

import pytest

from reach.paths import ROOT

spec = importlib.util.spec_from_file_location("s2_checksums", ROOT / "scripts/s2_checksums.py")
CK = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CK)

FORM = ROOT / "docs/registration/osf/stage2_form.md"
LIST = ROOT / "data/registration/stage2_checksums.txt"
ZIP = "data/registration/stage2_targets.zip"
NOT_LISTED = {"data/registration/stage2_checksums.txt", ZIP, "data/registration/stage2_registration.txt",
              "data/registration/cgp_transfer.json"}     # frozen after s2_cal_cgp; its checksum goes in the addendum


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _tree(tmp_path):
    for g in ("A", "B"):
        d = tmp_path / f"data/targets_s2/{g}"
        d.mkdir(parents=True)
        (d / "heldout.json").write_text(f'["{g}"]')
        (d / "bench.json").write_text(f'["{g}b"]')
    (tmp_path / "data/registration").mkdir(parents=True)
    (tmp_path / "data/registration/m.json").write_text("{}")
    return tmp_path


def test_checksum_lines_verify_and_the_zip_is_deterministic(tmp_path):
    root = _tree(tmp_path)
    pats = ["data/registration/m.json", "data/targets_s2/*/heldout.json", "data/targets_s2/*/bench.json"]
    CK.main("out.txt", pats, root=root, zip_out="t.zip", zip_patterns=["data/targets_s2/*/*.json"])
    lines = [l for l in (root / "out.txt").read_text().splitlines() if not l.startswith("#")]
    assert len(lines) == 5 and all(_sha(root / p) == h for h, p in (l.split("  ") for l in lines))
    first = (root / "t.zip").read_bytes()
    assert (root / "t.sha256").read_text().split()[0] == _sha(root / "t.zip")
    CK.main("out.txt", pats, root=root, zip_out="t.zip", zip_patterns=["data/targets_s2/*/*.json"])
    assert (root / "t.zip").read_bytes() == first                             # same inputs, same bytes
    with zipfile.ZipFile(root / "t.zip") as z:
        assert sorted(z.namelist()) == sorted(p for _, p in (l.split("  ") for l in lines) if "targets_s2" in p)
        assert all(z.read(n) == (root / n).read_bytes() for n in z.namelist())


def test_zip_refuses_a_file_missing_from_the_checksums(tmp_path):
    """Planted: a zip pattern matching a file the checksums file does not list is refused."""
    root = _tree(tmp_path)
    with pytest.raises(SystemExit, match="not in the checksums"):
        CK.main("out.txt", ["data/targets_s2/*/heldout.json"], root=root, zip_out="t.zip",
                zip_patterns=["data/targets_s2/*/*.json"])


def listed():
    return {p: h for h, p in (l.split("  ") for l in LIST.read_text().splitlines() if l and not l.startswith("#"))}


def _heldout_sets():
    """The 12 held-out sets: the registered levels file's set keys (its `_`-keys are metadata)."""
    import json
    sets = sorted(k for k in json.loads((ROOT / "data/registration/heldout_levels.json").read_text())
                  if not k.startswith("_"))
    assert len(sets) == 12, sets
    return sets


def form_problems(form_text, listed_map, root=ROOT):
    """Files the form names under data/ that are missing from the list, listed files whose hash is wrong, and 64-hex
    values in the form that are no listed file's SHA-256 (the zip's and the checksums file's own hashes allowed). The
    filed placeholder SET (OSF strips `<set>`) stands for each of the 12 held-out sets: every expansion must be listed
    (guardrail row 142)."""
    out = []
    for p, h in listed_map.items():
        if not (root / p).exists() or _sha(root / p) != h:
            out.append(f"wrong or missing: {p}")
    named = set(re.findall(r"data/(?:registration|targets_s2|reference|analysis)/[\w./*-]+\.(?:json|txt|csv|yaml)",
                           form_text))
    for p in sorted(named - NOT_LISTED):
        if "*" in p or "<" in p:
            continue
        for q in ([p.replace("/SET/", f"/{g}/") for g in _heldout_sets()] if "/SET/" in p else [p]):
            if q not in listed_map:
                out.append(f"named, not listed: {q}")
    allowed = set(listed_map.values()) | {_sha(root / "data/registration/stage2_checksums.txt")}
    side = (root / ZIP).with_suffix(".sha256")                               # the zip is untracked, its hash is not
    if side.exists():
        allowed.add(side.read_text().split()[0])
        if (root / ZIP).exists() and _sha(root / ZIP) != side.read_text().split()[0]:
            out.append("stage2_targets.zip does not match its .sha256")
    for h in sorted(set(re.findall(r"\b[0-9a-f]{64}\b", form_text)) - allowed):
        out.append(f"hash in the form matches no listed file: {h[:12]}")
    return out


def test_stage2_checksums_cover_every_attachment_the_form_names():
    if not LIST.exists():
        pytest.skip("Stage 2 checksums not written yet")
    assert form_problems(FORM.read_text(), listed()) == []
    for g in {p.split("/")[2] for p in listed() if p.startswith("data/targets_s2/")}:
        assert f"data/targets_s2/{g}/heldout.json" in listed()


def test_form_check_fires_on_planted_violations():
    if not LIST.exists():
        pytest.skip("Stage 2 checksums not written yet")
    text, ls = FORM.read_text(), listed()
    k = next(iter(ls))
    assert any("wrong or missing" in p for p in form_problems(text, dict(ls, **{k: "0" * 64})))
    assert any("named, not listed" in p for p in form_problems(text, {q: h for q, h in ls.items()
                                                                     if q != "data/registration/h7_model.json"}))
    assert any("matches no listed file" in p for p in form_problems(text + " " + "ab" * 32, ls))
    if "/SET/" in text:                                  # row 142: one held-out set's file missing from the list fires
        gone = "data/targets_s2/RS-add_sqrt/heldout.json"
        assert f"named, not listed: {gone}" in form_problems(text, {q: h for q, h in ls.items() if q != gone})
        assert "named, not listed: data/targets_s2/SET2/heldout.json" in form_problems(
            text + " data/targets_s2/SET2/heldout.json", ls)        # a literal path that only looks like it still fires
    if (ROOT / ZIP).exists():                                                # study machine: a changed zip fires
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d)
            (fake / "data/registration").mkdir(parents=True)
            for f in ("stage2_checksums.txt", "stage2_targets.sha256"):
                shutil.copy(ROOT / "data/registration" / f, fake / "data/registration" / f)
            (fake / ZIP).write_bytes(b"not the zip")
            assert "stage2_targets.zip does not match its .sha256" in form_problems(text, ls, root=fake)

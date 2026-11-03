"""SHA-256 of the files a registration freezes, and the targets zip (tool T17; REQ-RUN-34).

    .venv/bin/python scripts/s2_checksums.py OUT PATTERN [PATTERN ...] [--zip ZIP ZPATTERN [ZPATTERN ...]]

Glob patterns are relative to the repository. Writes `<sha256>  <path>` lines, sorted by path, and a final line with
the commit. Verify with `shasum -a 256 -c OUT` (the commit line is a comment). With --zip, also writes ZIP holding the
files matched by ZPATTERN, each of which must be listed in OUT, and its SHA-256 in ZIP's .sha256 (tracked; the zip is
not); the zip is deterministic (sorted names, fixed timestamps and permissions): the same files give the same bytes.

Stage 2 (D092):
    .venv/bin/python scripts/s2_checksums.py data/registration/stage2_checksums.txt \\
        data/registration/{h7_model,gpf2013_comparator,gp2010_comparator,operon_transfer,heldout_levels}.json \\
        data/registration/{cgp_settings,heldout_predictions,stage2_tasks,random_draw,heldout_draw}.json \\
        data/analysis/calibration.json data/analysis/properties.json \\
        'data/targets_s2/*/heldout.json' 'data/targets_s2/*/bench.json' 'data/targets_s2/*/bench_log.json' \\
        'data/reference/srbench_groundtruth/*' data/reference/eml_gp_targets/source.json \\
        --zip data/registration/stage2_targets.zip \\
        'data/targets_s2/*/heldout.json' 'data/targets_s2/*/bench.json' 'data/targets_s2/*/bench_log.json'
"""
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def _files(root, patterns):
    return sorted({p for pat in patterns for p in root.glob(pat) if p.is_file()})


def write_zip(path, root, files):
    """Deterministic zip: names relative to root, sorted; fixed timestamp and permissions; deflate."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            info = zipfile.ZipInfo(str(p.relative_to(root)), date_time=FIXED_TIME)
            info.compress_type, info.external_attr, info.create_system = zipfile.ZIP_DEFLATED, 0o644 << 16, 3
            z.writestr(info, p.read_bytes())


def main(out, patterns, root=ROOT, zip_out=None, zip_patterns=()):
    root = Path(root)
    files = _files(root, patterns)
    if not files:
        sys.exit("no files matched")
    zfiles = _files(root, zip_patterns) if zip_out else []
    missing = sorted(str(p.relative_to(root)) for p in set(zfiles) - set(files))
    if missing:
        sys.exit(f"zip files not in the checksums: {missing}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}" for p in files]
    (root / out).write_text("\n".join(lines) + f"\n# commit {commit}\n")
    print(f"{len(files)} files -> {out}")
    if zip_out:
        z = root / zip_out
        write_zip(z, root, zfiles)
        sha = hashlib.sha256(z.read_bytes()).hexdigest()
        z.with_suffix(".sha256").write_text(f"{sha}  {z.name}\n")         # tracked; the zip itself is not
        print(f"{len(zfiles)} files -> {zip_out}  sha256 {sha}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--zip" in args:
        i = args.index("--zip")
        main(args[0], args[1:i], zip_out=args[i + 1], zip_patterns=args[i + 2:])
    else:
        main(args[0], args[1:])

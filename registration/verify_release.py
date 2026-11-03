"""Check that the released code and registered files are the versions registered on OSF.

The registered versions are identified by git tree hashes of src/reach (registration/registered_trees.txt) at the
commits named in the OSF registrations of the authors' private working repository. A git tree hash depends only on
file names, modes and contents, so it can be recomputed from the released files without the private history. The
code changed after each registration; registration/src_reach_<commit>.patch holds every change from that commit to
this release and is reversed here before hashing. The registered data files are checked against the SHA-256 lists
attached to the registrations (data/registration/stage*_checksums.txt); a listed code file is checked against the
registered code it belongs to. Requires git. Run from the repository root:

    python3 registration/verify_release.py
"""
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "registration"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")


def git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def registered_code(patch, work):
    """src/reach as registered: the released src/reach with the patch reversed, in work."""
    shutil.copytree(ROOT / "src/reach", work, ignore=IGNORE)
    git(["init", "-q"], work)
    if patch.stat().st_size:
        git(["apply", "-R", "-p3", "--whitespace=nowarn", str(patch)], work)    # a/src/reach/...: strip three
    git(["add", "-A"], work)
    return git(["write-tree"], work)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = [[c.strip() for c in l.split("|")] for l in (REG / "registered_trees.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    ok, code = True, {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, (label, commit, tree, patch) in enumerate(rows):
            work = Path(tmp) / str(i)
            try:
                got = registered_code(REG / patch, work)
            except subprocess.CalledProcessError as e:
                got = f"patch does not apply ({e.stderr.strip()[:80]})"
            match = got == tree
            ok &= match
            print(f"{'ok  ' if match else 'FAIL'} {label}: src/reach at {commit[:7]} tree {tree[:12]}"
                  + ("" if match else f", got {got[:60]}"))
            for f in work.rglob("*"):
                if f.is_file() and ".git" not in f.parts:
                    code.setdefault(sha256(f), set()).add("src/reach/" + str(f.relative_to(work)))
        for listing in sorted((ROOT / "data/registration").glob("stage*_checksums.txt")):
            bad = []
            for line in listing.read_text().splitlines():
                if not line.strip() or line.startswith("#"):
                    continue
                digest, path = line.split(None, 1)
                path = path.strip()
                if path.startswith("src/reach/"):
                    good = path in code.get(digest, set())                     # the registered version of the code
                else:
                    good = (ROOT / path).is_file() and sha256(ROOT / path) == digest
                if not good:
                    bad.append(path)
            ok &= not bad
            print(f"{'ok  ' if not bad else 'FAIL'} {listing.name}" + (f": {bad[:5]}" if bad else ""))
    print("all registered versions verified" if ok else "verification FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""After a change to rust/cgp: set reach.cgp.RUST_SOURCE_SHA256 to the hash of the source in the tree and install the
binary built from it (REQ-RUN-23). The constant is Python code, so the change also changes the pinned code hash: a
phase pinned to the old code keeps running the old binary from the store."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reach import cgp  # noqa: E402

path = ROOT / "src/reach/cgp.py"
h = cgp.source_hash()
text = path.read_text()
new = re.sub(r'^RUST_SOURCE_SHA256 = "[0-9a-f]{64}"$', f'RUST_SOURCE_SHA256 = "{h}"', text, count=1, flags=re.M)
if new == text and h not in text:
    sys.exit("RUST_SOURCE_SHA256 line not found")
path.write_text(new)
cgp.drop_bytecode(path)               # same size, maybe the same second: a cached bytecode file would win (row 82)
print(f"RUST_SOURCE_SHA256 = {h}\ninstalled {cgp.build()}")

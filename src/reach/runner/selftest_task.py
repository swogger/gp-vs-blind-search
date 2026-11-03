"""A fake task for testing the queue: burns CPU, then reports, crashes or never finishes; or sleeps (wall time,
no CPU) and then reports, to show that limits are CPU time, not wall time."""
import argparse
import json
import time

from reach.paths import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["ok", "timeout", "error", "noresult", "sleep", "selfkill", "env"], default="ok")
ap.add_argument("--cpu", type=float, default=0.2, help="CPU seconds to burn first")
ap.add_argument("--sleep", type=float, default=0.0, help="wall seconds to sleep (mode sleep)")
a = ap.parse_args()

t0 = time.process_time()
x = 0
while a.mode == "timeout" or time.process_time() - t0 < a.cpu:
    x += 1
if a.mode == "selfkill":                     # killed from outside before its CPU limit (like a memory kill)
    import os
    import signal
    os.kill(os.getpid(), signal.SIGKILL)
if a.mode == "sleep":
    time.sleep(a.sleep)
if a.mode == "error":
    raise RuntimeError("deliberate failure")
if a.mode in ("ok", "sleep"):
    print("RESULT " + json.dumps({"terminal_state": "converged", "module_file": __file__, "root": str(ROOT)}), flush=True)
if a.mode == "env":
    import os
    print("RESULT " + json.dumps({"terminal_state": "converged", "env_value": os.environ.get("REACH_TEST_ENV"),
                                  "dyld_value": os.environ.get("DYLD_FALLBACK_LIBRARY_PATH")}), flush=True)

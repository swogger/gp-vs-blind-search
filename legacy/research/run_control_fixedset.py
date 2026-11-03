#!/usr/bin/env python3
"""
Fixed-primitive-set Std-GP control (Section 5.5).

Arm FIX : one fixed operator set {+,-,*,/,exp,log,sin,cos} applied uniformly to
          all 12 targets, replacing the per-target sets of Table 2. sin and cos
          are retained so that sin+cos and the Phase 4 composites stay exactly
          expressible and a zero rate measures search difficulty, not
          inexpressibility.
Arm ABL : {+,-,*,/,exp,log} on the two mixed targets (sinh, x^x), removing the
          matching primitive while adding none, so that ablation is separated
          from dilution.

Std-GP only: EML-GP is unaffected by the operator-set override, so the existing
main-study EML-GP runs at this cell are the comparison arm.

Reference cell (G=200, P=100), both regimes, the same 30 seeds as the main grid,
so every run is seed-matched to its main-study counterpart.

Results append to results/fixed_set_control/control_runs.csv and the script is
resumable: completed (arm, regime, test, seed) rows are skipped on restart.
"""
import argparse, csv, datetime, os, re, signal, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_suite import TEST_MATRIX
from seeds import SEEDS

HERE        = os.path.dirname(os.path.abspath(__file__))
OUT_DIR     = os.path.abspath(os.path.join(HERE, "..", "results", "fixed_set_control"))
OUT_CSV     = os.path.join(OUT_DIR, "control_runs.csv")
GENERATIONS = 200
POP_SIZE    = 100
CPU_LIMIT_S = 600      # hard CPU-time limit per trial (NOT wall clock)
WALL_BACKSTOP_S = 5400 # only fires if a run hangs without burning CPU

# One fixed operator set applied uniformly to all twelve targets, replacing the
# per-target sets of Table 2 (Section 5.5).
#
# sin and cos are retained so that eleven of the twelve targets remain EXACTLY
# expressible in the control grammar; without them sin+cos, arctan and both
# Nested composites become representationally impossible rather than merely
# harder, which would reintroduce the very confound Section 5.4 was built to
# remove.  arctan (Test 3.2) is the sole exception and is reported as
# approximation-only.
#
# Targets whose matching primitive is removed by this set: 3.1 sinh
# (sinh/cosh/tanh dropped, exp retained), 3.2 arctan (atan dropped),
# 3.3 x^x (sqrt dropped).  Targets 1.3 and 3.5 GAIN exp/log relative to their
# per-target arithmetic-only sets: uniformity cuts both ways and this is
# reported as such.
#
# ABL is the ablation-only arm: it removes each target's matching primitive and adds
# nothing, so that ablation is separated from the dilution effect that the uniform FIX
# grammar also introduces.  Applied only to the two mixed-case targets whose objective
# remains exactly expressible after the removal: sinh (sinh/cosh/tanh dropped, exp
# retained, sinh = (e^x - e^-x)/2) and x^x (sqrt dropped, x^x = exp(x ln x)).
ARMS = {
    "FIX": {"ops": "+,-,*,/,exp,log,sin,cos", "tests": list(TEST_MATRIX.keys())},
    "ABL": {"ops": "+,-,*,/,exp,log",         "tests": ["3.1", "3.3"]},
}

# Exact expressibility of each target in the FIX grammar over the reals.
EXPRESSIBLE = {t: (t != "3.2") for t in TEST_MATRIX}
FIELDS = ["arm", "ops", "regime", "test_id", "test_name", "seed",
          "final_error", "nodes", "best_gen", "status", "cpu_s", "wall_s", "formula"]


def run_one(arm, regime, test_id, seed):
    cmd = ["python3", "_cpu_limit_run.py", str(CPU_LIMIT_S),
           "python3", "deduce_formula.py",
           "--test_id", test_id,
           "--generations", str(GENERATIONS),
           "--pop_size", str(POP_SIZE),
           "--seed", str(seed),
           "--plateau_mult", "1",
           "--std_ops", ARMS[arm]["ops"]]
    if regime == "B":
        cmd.append("--mutation_only")

    t0 = datetime.datetime.now()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd=HERE)
    try:
        stdout, stderr = proc.communicate(timeout=WALL_BACKSTOP_S)
    except subprocess.TimeoutExpired:
        proc.kill(); stdout, stderr = proc.communicate()
        wall = (datetime.datetime.now() - t0).total_seconds()
        return dict(arm=arm, ops=ARMS[arm]["ops"], regime=regime, test_id=test_id,
                    test_name=TEST_MATRIX[test_id]["name"], seed=seed,
                    final_error="", nodes="", best_gen=GENERATIONS,
                    status="WALL_BACKSTOP", cpu_s="", wall_s=round(wall, 1), formula="")

    wall = (datetime.datetime.now() - t0).total_seconds()
    cpu, termsig = None, 0
    for ln in (stderr or "").split("\n"):
        if ln.startswith("__CPU_SECONDS__="): cpu = float(ln.split("=")[1])
        if ln.startswith("__TERMSIG__="):     termsig = int(ln.split("=")[1])
    cpu_timeout = (termsig == signal.SIGXCPU)
    final_error, nodes, best_gen = None, None, GENERATIONS
    lowest, status, formula = float("inf"), "BUDGET EXHAUSTED", "None"

    for line in stdout.split("\n"):
        if "Best Formula: Result =" in line:
            formula = line.split("Result = ")[1].strip()
        if line.startswith("Node count:"):
            m = re.search(r"Node count:\s*(\d+)", line)
            if m: nodes = int(m.group(1))
        if "Final MSE:" in line:
            m = re.search(r"Final MSE:\s*([0-9.eE+\-]+)", line)
            if m:
                try: final_error = float(m.group(1))
                except ValueError: pass
        if "Status:" in line and "CONVERGED" in line:
            status = "CONVERGED"
        if "Generation" in line and "Best Error" in line:
            m = re.search(r"Generation\s+(\d+)\s+-\s+Best Error:\s+([0-9.eE+\-inf]+)"
                          r"\s+-\s+Val Error:\s+([0-9.eE+\-inf]+)\s+-\s+Nodes:\s+(\d+)", line)
            if m:
                try: err = float(m.group(2))
                except ValueError: err = float("inf")
                nodes = int(m.group(4))
                if err < lowest:
                    lowest, best_gen = err, int(m.group(1))

    if cpu_timeout:
        status, final_error = "CPU_TIMEOUT", None
    return dict(arm=arm, ops=ARMS[arm]["ops"], regime=regime, test_id=test_id,
                test_name=TEST_MATRIX[test_id]["name"], seed=seed,
                final_error="" if final_error is None else final_error,
                nodes="" if nodes is None else nodes, best_gen=best_gen,
                status=status, cpu_s="" if cpu is None else round(cpu, 1),
                wall_s=round(wall, 1), formula=formula)


def done_keys():
    if not os.path.exists(OUT_CSV): return set()
    with open(OUT_CSV) as f:
        return {(r["arm"], r["regime"], r["test_id"], r["seed"]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="FIX")
    ap.add_argument("--n_seeds", type=int, default=30, help="seeds per cell (pilot: use 3)")
    ap.add_argument("--workers", type=int, default=6,
                    help="kept below core count to leave headroom for other work")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    have = done_keys()
    jobs = [(arm, regime, tid, SEEDS[i])
            for arm in args.arms.split(",")
            for regime in ("A", "B")
            for tid in ARMS[arm]["tests"]
            for i in range(1, args.n_seeds + 1)
            if (arm, regime, tid, str(SEEDS[i])) not in have]

    print(f"{len(jobs)} runs to do ({len(have)} already complete), {args.workers} workers")
    if not jobs: return

    new = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new: w.writeheader()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, *j): j for j in jobs}
            for n, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                w.writerow(row); fh.flush()
                print(f"[{n}/{len(jobs)}] {row['arm']}/{row['regime']} {row['test_id']} "
                      f"seed={row['seed']} {row['status']} cpu={row['cpu_s']}s wall={row['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()

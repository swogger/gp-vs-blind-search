"""
Standard SR Benchmark Experiment
=================================
Runs 4 standard symbolic regression benchmarks:
  - Nguyen-1         : f(x) = x³ + x² + x
  - Nguyen-7         : f(x) = ln(x+1) + ln(x²+1)
  - Keijzer-6        : f(x) = Σ(1/i, i=1..x)  [integer domain]
  - Vladislavleva-4  : f(x1..x5) = 10 / (5 + Σ(xi-3)²)

Each benchmark is run with 4 configurations:
  std_a : Std-GP  + crossover+mutation  (Regime A)
  eml_a : EML-GP  + crossover+mutation  (Regime A)
  std_b : Std-GP  + mutation only       (Regime B)
  eml_b : EML-GP  + mutation only       (Regime B)

Reference cell: Generations=200, Population=100, 30 seeded runs.

Usage:
    python run_sr_benchmarks.py [output_dir]

Output:
    <output_dir>/sr_benchmark_results_summary.csv
    <output_dir>/sr_benchmark_convergence_data.csv
    <output_dir>/sr_benchmark_nguyen1.png
    <output_dir>/sr_benchmark_nguyen7.png
    <output_dir>/sr_benchmark_keijzer6.png
    <output_dir>/sr_benchmark_vladislavleva4.png
"""

import sys
import os
import csv
import math
import cmath
import random
import signal
import time
import datetime
import resource
import subprocess
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
_GALIB_DIR    = os.path.abspath(os.path.join(_RESEARCH_DIR, "../ga-lib"))
sys.path.insert(0, _GALIB_DIR)

from generator  import generate_ramped_population
from evolution  import crossover, mutate, point_mutate
from node       import Node
from test_suite import eml_plateau_mult
from seeds      import SEEDS

# ---------------------------------------------------------------------------
# Budget — must match deduce_formula.py
# ---------------------------------------------------------------------------
MAX_NODES       = 2_000
MAX_DEPTH       = math.ceil(math.log2(MAX_NODES + 1)) - 1   # 10
BATCH_SIZE      = 75
VALIDATION_SIZE = 200
GENERATIONS     = 200
POP_SIZE        = 100
NUM_RUNS        = 30
CPU_TIMEOUT     = 600   # seconds of CPU time per run

STD_OPS = ['+', '-', '*', '/', 'exp', 'log']   # arith + exp for all benchmarks
EML_OPS = ['eml']

plt.rcParams["font.family"] = ["Arial"]

ALGO_STYLES = {
    "std_a": {"color": "#1f77b4", "ls": "-",  "lw": 1.5, "label": "Std-GP (crossover+mut)"},
    "eml_a": {"color": "#d62728", "ls": ":",  "lw": 1.5, "label": "EML-GP (crossover+mut)"},
    "std_b": {"color": "#1f77b4", "ls": "--", "lw": 1.5, "label": "Std-GP (mutation only)"},
    "eml_b": {"color": "#d62728", "ls": "--", "lw": 1.5, "label": "EML-GP (mutation only)"},
}


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------
def _safe(v):
    """Return complex inf for non-finite values."""
    if cmath.isnan(v) or cmath.isinf(v):
        return complex(float("inf"), float("inf"))
    return v

def _nguyen1(x):
    return _safe(x**3 + x**2 + x)

def _nguyen7(x):
    try:
        return _safe(cmath.log(x + 1) + cmath.log(x**2 + 1))
    except Exception:
        return complex(float("inf"), float("inf"))

def _keijzer6(x):
    # x is integer in [1, 50]; harmonic sum
    try:
        n = max(1, int(round(abs(x.real if isinstance(x, complex) else x))))
        return complex(sum(1.0 / i for i in range(1, n + 1)))
    except Exception:
        return complex(float("inf"), float("inf"))

def _vladislavleva4(*xs):
    try:
        denom = 5.0 + sum((xi - 3.0)**2 for xi in xs)
        return _safe(10.0 / denom)
    except Exception:
        return complex(float("inf"), float("inf"))


# Fixed fitness point grids per benchmark (deterministic, not seeded)
def _make_nguyen1_inputs():
    # 20 evenly-spaced points on [-1, 1] inclusive
    return [[round(-1.0 + i * (2.0 / 19), 10)] for i in range(20)]

def _make_nguyen7_inputs():
    # 20 evenly-spaced points on [0, 2] inclusive
    return [[round(i * (2.0 / 19), 10)] for i in range(20)]

def _make_keijzer6_inputs():
    return [[float(i)] for i in range(1, 51)]          # integers 1..50

def _vladislavleva4_inputs(seed=20260429):
    rng = random.Random(seed)
    pts = []
    for _ in range(1024):
        pts.append([rng.uniform(0.05, 6.05) for _ in range(5)])
    return pts


BENCHMARKS = {
    "nguyen1": {
        "name":        "Nguyen-1: f(x) = x³+x²+x",
        "vars":        ["x"],
        "range":       (-1.0, 1.0),
        "func":        _nguyen1,
        "fixed_inputs": _make_nguyen1_inputs(),
        "plot_name":   "sr_benchmark_nguyen1.png",
    },
    "nguyen7": {
        "name":        "Nguyen-7: f(x) = ln(x+1)+ln(x²+1)",
        "vars":        ["x"],
        "range":       (0.0, 2.0),
        "func":        _nguyen7,
        "fixed_inputs": _make_nguyen7_inputs(),
        "plot_name":   "sr_benchmark_nguyen7.png",
    },
    "keijzer6": {
        "name":        "Keijzer-6: f(x) = Σ(1/i, i=1..x)",
        "vars":        ["x"],
        "range":       (1.0, 50.0),
        "func":        _keijzer6,
        "fixed_inputs": _make_keijzer6_inputs(),
        "plot_name":   "sr_benchmark_keijzer6.png",
    },
    "vladislavleva4": {
        "name":        "Vladislavleva-4: f(x1..x5) = 10/(5+Σ(xi-3)²)",
        "vars":        ["x1", "x2", "x3", "x4", "x5"],
        "range":       (0.05, 6.05),
        "func":        _vladislavleva4,
        "fixed_inputs": _vladislavleva4_inputs(),
        "plot_name":   "sr_benchmark_vladislavleva4.png",
    },
}

BENCHMARK_ORDER = ["nguyen1", "nguyen7", "keijzer6", "vladislavleva4"]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def count_nodes(tree):
    if tree is None:
        return 0
    if not tree.is_operator:
        return 1
    return 1 + count_nodes(tree.left) + count_nodes(tree.right)


# ---------------------------------------------------------------------------
# Single GP run (mirrors deduce_formula.py exactly)
# ---------------------------------------------------------------------------
def run_gp(bench_id, use_eml, mutation_only, seed,
           generations=GENERATIONS, pop_size=POP_SIZE, size_penalty=True, trace=False):
    """
    Run one GP experiment on a benchmark and return per-generation records.
    Returns list of (generation, best_val_error) tuples.
    """
    bench       = BENCHMARKS[bench_id]
    variables   = bench["vars"]
    data_range  = bench["range"]
    target_func = bench["func"]
    fixed_pts   = bench["fixed_inputs"]

    operators = EML_OPS if use_eml else STD_OPS
    terminals = [1.0]

    # Three isolated RNG streams (same design as deduce_formula.py)
    _m       = random.Random(seed)
    val_seed = _m.randint(0, 2**31 - 1)
    pop_seed = _m.randint(0, 2**31 - 1)
    evo_seed = _m.randint(0, 2**31 - 1)

    val_rng = random.Random(val_seed)
    pop_rng = random.Random(pop_seed)
    evo_rng = random.Random(evo_seed)

    import random as _rm
    _rm.choice  = evo_rng.choice
    _rm.random  = evo_rng.random
    _rm.sample  = evo_rng.sample
    _rm.uniform = evo_rng.uniform
    _rm.seed    = lambda *a, **kw: None

    # Validation set: use the fixed benchmark fitness points if provided,
    # otherwise draw randomly from the domain (same strategy as main framework).
    # For Keijzer-6 the domain is integer-valued, so fixed points are essential.
    if fixed_pts:
        val_inputs = fixed_pts
    else:
        val_inputs = [
            [val_rng.uniform(data_range[0], data_range[1]) for _ in variables]
            for _ in range(VALIDATION_SIZE)
        ]

    # Plateau patience — same formula as deduce_formula.py line 113
    plateau_mult = eml_plateau_mult(STD_OPS) if use_eml else 1
    plateau_gens = max(50, generations // 10) * plateau_mult

    # Population init with pop_rng
    _rm.choice  = pop_rng.choice
    _rm.random  = pop_rng.random
    _rm.sample  = pop_rng.sample
    _rm.uniform = pop_rng.uniform

    population = generate_ramped_population(pop_size, MAX_DEPTH, operators, terminals, variables)

    _rm.choice  = evo_rng.choice
    _rm.random  = evo_rng.random
    _rm.sample  = evo_rng.sample
    _rm.uniform = evo_rng.uniform

    mut_depth        = max(2, round(math.sqrt(MAX_DEPTH)))
    use_point_mutate = len(operators) > 1 or not all(op == "eml" for op in operators)

    def _mse(tree, inputs):
        if count_nodes(tree) > MAX_NODES:
            return float("inf")
        error = 0.0
        valid = 0
        try:
            for inp in inputs:
                kwargs   = {var: val for var, val in zip(variables, inp)}
                pred     = tree.evaluate(**kwargs)
                expected = target_func(*inp)
                diff     = abs(pred - expected)
                if cmath.isnan(diff) or cmath.isinf(diff):
                    error += 1000.0
                else:
                    error += diff ** 2
                valid += 1
            return error / valid if valid > 0 else float("inf")
        except Exception:
            return float("inf")

    def training_fitness(tree, batch):
        mse = _mse(tree, batch)
        if size_penalty and mse < 1e-5:
            return mse + count_nodes(tree) * 0.0001
        return mse

    def validation_error(tree):
        return _mse(tree, val_inputs)

    best_tree       = None
    best_val_error  = float("inf")
    gens_no_improve = 0
    converged       = False
    records         = []

    for gen in range(generations):
        batch = [
            [evo_rng.uniform(data_range[0], data_range[1]) for _ in variables]
            for _ in range(BATCH_SIZE)
        ]

        fitness_scores = [(ind, training_fitness(ind, batch)) for ind in population]
        fitness_scores.sort(key=lambda x: x[1])

        gen_best = fitness_scores[0][0]
        gen_val  = validation_error(gen_best)

        if gen_val < best_val_error:
            best_val_error  = gen_val
            best_tree       = gen_best
            gens_no_improve = 0
        else:
            gens_no_improve += 1

        records.append((gen, best_val_error))
        if trace:
            print(f"GEN {gen} BEST {best_val_error:.9e}", flush=True)

        if best_val_error <= 1e-6:
            converged = True
            break
        if gens_no_improve >= plateau_gens:
            break

        # Reproduction (mirrors deduce_formula.py lines 239-278)
        new_pop = [best_tree.copy()]

        while len(new_pop) < pop_size:
            t1 = evo_rng.sample(fitness_scores, k=5)
            t1.sort(key=lambda x: x[1])
            parent1 = t1[0][0]

            if not mutation_only and evo_rng.random() < 0.75 and len(new_pop) < pop_size - 1:
                t2 = evo_rng.sample(fitness_scores, k=5)
                t2.sort(key=lambda x: x[1])
                parent2 = t2[0][0]
                child1, child2 = crossover(parent1, parent2, max_depth=None)
                if count_nodes(child1) > MAX_NODES:
                    child1 = parent1.copy()
                if count_nodes(child2) > MAX_NODES:
                    child2 = parent2.copy()
                new_pop.extend([child1, child2])
            else:
                new_pop.append(parent1.copy())

        for i in range(1, len(new_pop)):
            if mutation_only:
                if use_point_mutate and evo_rng.random() < 0.05:
                    new_pop[i] = point_mutate(new_pop[i], operators, terminals, variables)
                else:
                    cand = mutate(new_pop[i], operators, terminals, variables,
                                  mut_depth, absolute_max_depth=MAX_DEPTH)
                    new_pop[i] = cand if count_nodes(cand) <= MAX_NODES else new_pop[i]
            else:
                if evo_rng.random() < 0.20:
                    cand = mutate(new_pop[i], operators, terminals, variables,
                                  mut_depth, absolute_max_depth=MAX_DEPTH)
                    new_pop[i] = cand if count_nodes(cand) <= MAX_NODES else new_pop[i]
                elif use_point_mutate and evo_rng.random() < 0.05:
                    new_pop[i] = point_mutate(new_pop[i], operators, terminals, variables)

        population = new_pop[:pop_size]

    return records, converged


# ---------------------------------------------------------------------------
# Single-task entry point (invoked as a subprocess)
# ---------------------------------------------------------------------------
def _set_cpu_limit():
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_TIMEOUT, CPU_TIMEOUT))


def run_task(bench_id, algo_key, run_idx, seed, runs_dir):
    """Execute one GP run and write its CSV. Called inside a fresh subprocess."""
    use_eml       = algo_key.startswith("eml")
    mutation_only = algo_key.endswith("_b")

    filename = f"sr_{bench_id}_{algo_key}_{run_idx:02d}.csv"
    filepath = os.path.join(runs_dir, filename)

    records, converged = run_gp(bench_id, use_eml, mutation_only, seed)

    if not records:
        print(f"  WARNING: {algo_key} {bench_id} run {run_idx:02d} — no records")
        return

    tmp = filepath + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "best_fitness", "converged"])
        for i, (gen, fit) in enumerate(records):
            w.writerow([gen, fit, int(converged) if i == len(records) - 1 else 0])
    os.replace(tmp, filepath)

    status = "CONVERGED" if converged else "budget exhausted"
    print(f"  [{algo_key.upper()}] {bench_id} run {run_idx:02d} — {len(records)} gens, {status}")


# ---------------------------------------------------------------------------
# Load and aggregate
# ---------------------------------------------------------------------------
def load_runs(runs_dir, bench_id, algo_key):
    pattern = os.path.join(runs_dir, f"sr_{bench_id}_{algo_key}_*.csv")
    dfs = []
    for path in sorted(glob.glob(pattern)):
        try:
            dfs.append(pd.read_csv(path))
        except Exception:
            pass
    return dfs


def aggregate(dfs):
    if not dfs:
        return None
    max_gen = max(df["generation"].max() for df in dfs)
    aligned = pd.DataFrame(index=np.arange(max_gen + 1))
    for i, df in enumerate(dfs):
        s = df.set_index("generation")["best_fitness"]
        s = s.reindex(np.arange(max_gen + 1)).ffill()
        aligned[i] = s
    return {
        "generations": aligned.index.values,
        "median": aligned.median(axis=1).values,
        "q1":     aligned.quantile(0.25, axis=1).values,
        "q3":     aligned.quantile(0.75, axis=1).values,
        "n_runs": len(dfs),
    }


def convergence_rate(runs_dir, bench_id, algo_key):
    """Fraction of runs that converged (best_fitness <= 1e-6 at any point)."""
    pattern = os.path.join(runs_dir, f"sr_{bench_id}_{algo_key}_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return float("nan")
    converged = 0
    for path in files:
        try:
            df = pd.read_csv(path)
            if (df["best_fitness"] <= 1e-6).any():
                converged += 1
        except Exception:
            pass
    return converged / len(files)


def median_final_mse(data):
    if data is None:
        return float("nan")
    return float(np.nanmedian(data["median"][-1:]))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_benchmark(bench_id, series, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, data in series.items():
        if data is None:
            continue
        style = ALGO_STYLES[key]
        med = np.clip(data["median"], 1e-10, None)
        q1  = np.clip(data["q1"],    1e-10, None)
        q3  = np.clip(data["q3"],    1e-10, None)
        ax.plot(data["generations"], med,
                color=style["color"], ls=style["ls"], lw=style["lw"],
                label=f"{style['label']} (n={data['n_runs']})")
        ax.fill_between(data["generations"], q1, q3,
                        color=style["color"], alpha=0.10)

    ax.set_yscale("log")
    ax.set_title(BENCHMARKS[bench_id]["name"], fontsize=10)
    ax.set_xlabel("Generation", fontsize=9)
    ax.set_ylabel("MSE", fontsize=9)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------
def _regime_direction(mse_a, mse_b):
    if math.isnan(mse_a) or math.isnan(mse_b):
        return "N/A"
    if mse_a < mse_b * 0.99:
        return f"A better  ({mse_a:.4g} vs {mse_b:.4g})"
    if mse_b < mse_a * 0.99:
        return f"B better  ({mse_b:.4g} vs {mse_a:.4g})"
    return f"Equivalent ({mse_a:.4g})"


def write_summary(bench_ids, runs_dir, all_series, out_path):
    rows = []
    for bid in bench_ids:
        s = all_series[bid]
        eml_a_mse = median_final_mse(s.get("eml_a"))
        eml_b_mse = median_final_mse(s.get("eml_b"))
        rows.append({
            "benchmark":            bid,
            "name":                 BENCHMARKS[bid]["name"],
            "std_a_convergence":    f"{convergence_rate(runs_dir, bid, 'std_a'):.3f}",
            "eml_a_convergence":    f"{convergence_rate(runs_dir, bid, 'eml_a'):.3f}",
            "std_b_convergence":    f"{convergence_rate(runs_dir, bid, 'std_b'):.3f}",
            "eml_b_convergence":    f"{convergence_rate(runs_dir, bid, 'eml_b'):.3f}",
            "std_a_median_mse":     median_final_mse(s.get("std_a")),
            "eml_a_median_mse":     eml_a_mse,
            "std_b_median_mse":     median_final_mse(s.get("std_b")),
            "eml_b_median_mse":     eml_b_mse,
            "eml_regime_direction": _regime_direction(eml_a_mse, eml_b_mse),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved summary → {out_path}")
    return df


def write_convergence_csv(bench_ids, all_series, out_path):
    rows = []
    for bid in bench_ids:
        for algo_key, data in all_series[bid].items():
            if data is None:
                continue
            for g, med, q1, q3 in zip(
                data["generations"], data["median"], data["q1"], data["q3"]
            ):
                rows.append({
                    "test_id":         bid,
                    "generations_config": GENERATIONS,
                    "pop_size":        POP_SIZE,
                    "algorithm":       algo_key,
                    "generation":      g,
                    "median_fitness":  med,
                    "q1_fitness":      q1,
                    "q3_fitness":      q3,
                    "n_runs":          data["n_runs"],
                })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved convergence data → {out_path}")


# ---------------------------------------------------------------------------
# Subprocess orchestrator
# ---------------------------------------------------------------------------
def _run_tasks_parallel(tasks):
    """
    Spawn one subprocess per task.  Each process gets a fresh RLIMIT_CPU budget
    via preexec_fn, so timeouts are fully independent of one another.
    """
    script = os.path.abspath(__file__)
    max_workers = os.cpu_count() or 4
    pending = list(tasks)
    active  = {}   # proc -> (bench_id, algo_key, run_idx)

    while pending or active:
        # Fill up to max_workers
        while pending and len(active) < max_workers:
            bench_id, algo_key, run_idx, seed, runs_dir = pending.pop(0)
            cmd = [
                sys.executable, script,
                "--run-task", bench_id, algo_key, str(run_idx), str(seed), runs_dir,
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=_set_cpu_limit,
            )
            active[proc] = (bench_id, algo_key, run_idx)

        # Wait for any one to finish
        finished = []
        for proc in list(active):
            ret = proc.poll()
            if ret is not None:
                out, _ = proc.communicate()
                bench_id, algo_key, run_idx = active.pop(proc)
                for line in out.splitlines():
                    if line.strip():
                        print(f"  {line}")
                if ret in (-signal.SIGXCPU, -signal.SIGKILL):
                    print(f"  TIMEOUT {bench_id} {algo_key} run {run_idx:02d}: CPU limit exceeded")
                elif ret != 0:
                    print(f"  ERROR {bench_id} {algo_key} run {run_idx:02d}: exit code {ret}")
                finished.append(proc)

        if not finished:
            time.sleep(0.1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    output_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 \
                 else os.path.join(_RESEARCH_DIR, "sr_benchmark_results")
    runs_dir = os.path.join(output_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    algo_keys = ["std_a", "eml_a", "std_b", "eml_b"]
    total = len(BENCHMARK_ORDER) * len(algo_keys) * NUM_RUNS

    print(f"\n[{datetime.datetime.now()}] Standard SR Benchmark Experiment")
    print(f"Benchmarks: {BENCHMARK_ORDER}")
    print(f"Generations: {GENERATIONS}, Population: {POP_SIZE}, Runs: {NUM_RUNS}")
    print(f"Total runs: {total}, Output: {output_dir}\n")

    # ------------------------------------------------------------------
    # Phase 1: build task list (skip completed runs)
    # ------------------------------------------------------------------
    tasks = []
    for bench_id in BENCHMARK_ORDER:
        for algo_key in algo_keys:
            for run_idx in range(1, NUM_RUNS + 1):
                seed = SEEDS[run_idx]
                fp   = os.path.join(runs_dir, f"sr_{bench_id}_{algo_key}_{run_idx:02d}.csv")
                if not os.path.exists(fp):
                    tasks.append((bench_id, algo_key, run_idx, seed, runs_dir))

    if tasks:
        print(f"Launching {len(tasks)} runs (skipping {total - len(tasks)} already on disk)...\n")
        _run_tasks_parallel(tasks)
    else:
        print("All runs already on disk — skipping execution.\n")

    # ------------------------------------------------------------------
    # Phase 2: aggregate
    # ------------------------------------------------------------------
    print("\nAggregating results...")
    all_series = {}
    for bench_id in BENCHMARK_ORDER:
        all_series[bench_id] = {}
        for algo_key in algo_keys:
            dfs = load_runs(runs_dir, bench_id, algo_key)
            all_series[bench_id][algo_key] = aggregate(dfs)
            print(f"  {bench_id:20s} {algo_key}: {len(dfs)} runs")

    # ------------------------------------------------------------------
    # Phase 3: plots
    # ------------------------------------------------------------------
    print("\nGenerating plots...")
    for bench_id in BENCHMARK_ORDER:
        out_png = os.path.join(output_dir, BENCHMARKS[bench_id]["plot_name"])
        plot_benchmark(bench_id, all_series[bench_id], out_png)

    # ------------------------------------------------------------------
    # Phase 4: summary + convergence CSV
    # ------------------------------------------------------------------
    summary_df = write_summary(
        BENCHMARK_ORDER, runs_dir, all_series,
        os.path.join(output_dir, "sr_benchmark_results_summary.csv"),
    )
    write_convergence_csv(
        BENCHMARK_ORDER, all_series,
        os.path.join(output_dir, "sr_benchmark_convergence_data.csv"),
    )

    print("\n--- Results Summary ---")
    print(summary_df.to_string(index=False))
    print(f"\n[{datetime.datetime.now()}] Done.")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--run-task":
        # Called by the orchestrator: run one task and exit.
        # argv: script --run-task bench_id algo_key run_idx seed runs_dir
        _, _, bench_id, algo_key, run_idx, seed, runs_dir = sys.argv
        run_task(bench_id, algo_key, int(run_idx), int(seed), runs_dir)
    else:
        main()

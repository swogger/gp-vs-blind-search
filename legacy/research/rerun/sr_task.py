"""One Phase-5 benchmark run (reference cell) -> streamed GEN lines + one RESULT JSON line."""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import run_sr_benchmarks as B  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True, choices=B.BENCHMARK_ORDER)
ap.add_argument("--algo", choices=["std", "eml"], required=True)
ap.add_argument("--regime", choices=["A", "B"], required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--generations", type=int, default=B.GENERATIONS)
ap.add_argument("--no_size_penalty", action="store_true")
a = ap.parse_args()
rec, conv = B.run_gp(a.bench, a.algo == "eml", a.regime == "B", a.seed, generations=a.generations,
                     size_penalty=not a.no_size_penalty, trace=True)
last_gen, best = rec[-1]
stop = "converged" if conv else ("budget" if last_gen == a.generations - 1 else "plateau")
print("RESULT " + json.dumps({"final_err": repr(best), "last_gen": last_gen,
                               "converged": conv, "stop_reason": stop}), flush=True)

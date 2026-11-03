"""Write the registered task lists: per phase, its task count and the SHA-256 of its sorted task keys (REQ-RUN-19).
Builds the task lists only; nothing runs. The registration locks are lifted in memory for the build, and no lock file
is written. Run in the main worktree.

    .venv/bin/python scripts/s2_task_lists.py              # Stage 1: data/registration/stage1_tasks.json
    .venv/bin/python scripts/s2_task_lists.py --stage 2    # Stage 2: data/registration/stage2_tasks.json (D076)
    .venv/bin/python scripts/s2_task_lists.py --stage 3    # Stage 3: stage3_tasks.json and stage3_lookups.json (D098)
    .venv/bin/python scripts/s2_task_lists.py --stage 4    # Stage 4: stage4_tasks.json (D100; after s4_targets)

Stage 1 needs the calibration and pool targets. Stage 2 needs the held-out levels and targets, the benchmark targets,
the CGP settings file (data/registration/cgp_settings.json) and the CGP binary; s2_ho_cgp is built without the H10b
transfer model, which is frozen later (phases.registration_build)."""
import argparse
import json
import tempfile
from pathlib import Path

from reach import s2_lock as L
from reach.paths import ROOT
from reach.runner.phases import PHASES, registration_build

CONFIRMATORY = ["s2_cal_a16", "s2_cal_a0", "s2_cal_mech", "s2_cal_alloc", "s2_cal_robust", "s2_cal_pool",
                "s2_cal_rand", "s2_cal_operon"]
STAGE2 = list(L.STAGE2_PHASES)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=(1, 2, 3, 4), default=1)
    a = ap.parse_args(argv)
    stage1 = sorted(n for n in PHASES if n.startswith("s2_cal_") and n not in L.STAGE2_PHASES)
    assert sorted(CONFIRMATORY) == stage1, "phase list out of date"
    names, out_name = {1: (CONFIRMATORY, "stage1_tasks.json"), 2: (STAGE2, "stage2_tasks.json"),
                       3: (list(L.STAGE3_PHASES), "stage3_tasks.json"),
                       4: (list(L.STAGE4_PHASES), "stage4_tasks.json")}[a.stage]
    real = L.STAGE1_FILE, L.STAGE2_FILE, L.STAGE3_FILE, L.STAGE4_FILE
    with tempfile.TemporaryDirectory() as d:
        L.STAGE1_FILE = L.STAGE2_FILE = L.STAGE3_FILE = L.STAGE4_FILE = Path(d) / "dry_build"
        L.STAGE1_FILE.write_text("dry build of the task lists only\n")
        out, sets = {}, set()
        try:
            for name in names:
                tasks = registration_build(name)()
                if not tasks:
                    raise SystemExit(f"{name}: no tasks (targets, levels or pool missing?)")
                out[name] = dict(n_tasks=len(tasks), sha256=L.task_list_hash(tasks))
                sets.update(t["meta"]["set"] for t in tasks if "set" in t.get("meta", {}))
                if a.stage >= 2:                  # Stages 2 and 3 also pin every task's settings (row 135; review m4)
                    out[name]["sha256_full"] = L.task_list_full_hash(tasks)
        finally:
            L.STAGE1_FILE, L.STAGE2_FILE, L.STAGE3_FILE, L.STAGE4_FILE = real
    if a.stage == 3:                              # each set's lookup, as the re-runs will record it (review m4)
        from reach import grammar as G
        from reach.revisit import Lookup, enum_root
        lk = {g: Lookup(G.get(g), enum_root(g)).sha256 for g in sorted(sets)}
        (ROOT / "data/registration/stage3_lookups.json").write_text(json.dumps(lk, indent=1) + "\n")
        print(f"lookups of {len(lk)} sets -> data/registration/stage3_lookups.json")
    p = ROOT / "data/registration" / out_name
    p.write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out.items():
        print(f"{k:15s} {v['n_tasks']:6d}  {v['sha256'][:16]}")
    print(f"total {sum(v['n_tasks'] for v in out.values()):,} tasks -> {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

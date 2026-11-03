#!/bin/bash
# K50-versus-budget sweep (Definition 2 is budget-relative; K50 was measured at one budget).
# Runs sequentially, resumable, ordered so the informative low-K levels complete first.
cd "$(dirname "$0")"
LOG() { echo "[$(date +%H:%M)] $*"; }

# 1. wait for the in-flight gen-800 main-band run
while pgrep -f "run_synthetic.*generations 800.*budget_pilot_g800" >/dev/null; do sleep 60; done
LOG "gen-800 main band complete"

# 2. gen-800, K=3 and K=5 (anchors the upper asymptote of the logistic fit)
LOG "starting gen-800 low-K"
python3 run_synthetic.py --targets targets_lowk.json --out g800_lowk.csv \
        --generations 800 --n_seeds 5 --workers 9
LOG "gen-800 low-K complete"

# 3. gen-3200, transition band first (K<=13): the levels that determine K50
LOG "starting gen-3200 low-K"
python3 run_synthetic.py --targets targets_lowk.json --out g3200_lowk.csv \
        --generations 3200 --n_seeds 5 --workers 9
LOG "starting gen-3200 K<=13"
python3 run_synthetic.py --k_max 13 --out g3200_main.csv \
        --generations 3200 --n_seeds 5 --workers 9
LOG "gen-3200 transition band complete"

# 4. extend gen-3200 to K=15,17 only if time allows; same output file, resumable
LOG "starting gen-3200 K<=17 extension"
python3 run_synthetic.py --k_max 17 --out g3200_main.csv \
        --generations 3200 --n_seeds 5 --workers 9
LOG "SWEEP COMPLETE"

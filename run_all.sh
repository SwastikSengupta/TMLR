#!/usr/bin/env bash
# Reproduce every number in the paper. ~35 min single core.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:$PWD/scripts:${PYTHONPATH:-}"
mkdir -p results figures
[ -d data/d52 ] || bash scripts/get_data.sh
echo "[1/5] Class I: estimation leakage, 70 conditions"
python3 scripts/class1_estimation.py
echo "[2/5] Class III: per-dataset effects, 12 learners"
python3 scripts/class3_final.py
python3 scripts/c3_rest.py
echo "[3/5] measured capacity via random-label fit"
python3 scripts/capacity_measure.py
echo "[4/5] statistical analysis"
python3 scripts/statistics.py
echo "[5/5] figures"
python3 scripts/figures.py
echo; echo "Done. Compare results/*.json to the tables in docs/paper.pdf"

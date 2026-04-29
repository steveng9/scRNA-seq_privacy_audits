#!/usr/bin/env bash
# sweep.sh — end-to-end v2 spot check.
#
# Trains v2 copulas (uncentered second moment), generates v2 synthetic data
# at no-DP and at the configured epsilons, runs quality+MIA evaluations, and
# emits a v1-vs-v2 comparison table.
#
# Designed to be safe to run alongside experiments/sdg_comparison/run_baselines_sweep.py:
#   - --max-workers 2 for R training (vs 4 in the v1 pipeline)
#   - generation is sequential per cell type
#   - MIA workers = 2
#
# Edit DATASET / ND / TRIALS / EPSILONS at the top to change the spot.

set -euo pipefail

DATASET="${DATASET:-ok}"
ND="${ND:-20}"
TRIALS="${TRIALS:-1 2}"
EPSILONS="${EPSILONS:-1 100 10000 1000000 100000000}"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
V2_DIR="$REPO_ROOT/experiments/dp/v2"
LOG_DIR="$V2_DIR/_logs"
mkdir -p "$LOG_DIR"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
banner() { echo; echo "============================================================"; echo "  $* — $(ts)"; echo "============================================================"; }

banner "v2 spot check: dataset=$DATASET nd=$ND trials=[$TRIALS] eps=[$EPSILONS]"

cd "$REPO_ROOT"

# ---- 1. Train v2 copulas ----
banner "1/5  Train v2 copulas (uncentered second moment)"
python "$V2_DIR/train.py" \
    --dataset "$DATASET" --nd "$ND" --trial $TRIALS --max-workers 2 \
    2>&1 | tee "$LOG_DIR/train_${DATASET}_${ND}d.log"

# ---- 2. Generate v2 no-DP synthetic ----
banner "2/5  Generate v2 no-DP synthetic"
python "$V2_DIR/generate.py" \
    --dataset "$DATASET" --nd "$ND" --trial $TRIALS --no-dp \
    2>&1 | tee "$LOG_DIR/gen_nodp_${DATASET}_${ND}d.log"

# ---- 3. Generate v2 + DP synthetic at each epsilon ----
banner "3/5  Generate v2+DP synthetic for each ε"
python "$V2_DIR/generate.py" \
    --dataset "$DATASET" --nd "$ND" --trial $TRIALS --epsilon $EPSILONS \
    2>&1 | tee "$LOG_DIR/gen_dp_${DATASET}_${ND}d.log"

# ---- 4. Quality + MIA evals ----
banner "4/5  Quality + MIA evaluations"
python "$V2_DIR/run_evals.py" \
    --dataset "$DATASET" --nd "$ND" --trial $TRIALS \
    2>&1 | tee "$LOG_DIR/evals_${DATASET}_${ND}d.log"

# ---- 5. Side-by-side comparison ----
banner "5/5  v1 vs v2 comparison table"
python "$V2_DIR/compare.py" --dataset "$DATASET" --nd "$ND" \
    2>&1 | tee "$LOG_DIR/compare_${DATASET}_${ND}d.log"

banner "v2 spot check done"

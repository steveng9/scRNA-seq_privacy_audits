#!/usr/bin/env bash
# sweep.sh — end-to-end lowrank spot check.
#
# Trains lowrank copulas (same fit as v2, plus saved per-cell quantile_normal),
# generates lowrank synthetic data at each (rank, {no-DP, epsilon}) combo,
# runs quality+MIA evaluations, and emits a lowrank-vs-v2 comparison table.
#
# Also regenerates the v2 (full-rank) baseline at ok/20d/trials 1-2 with the
# corrected TRUE_CLIP_VALUE (see notes/DP_clip_value_bug.txt) so the
# comparison is apples-to-apples -- the old v2_eps_* results for this
# scope were moved to
#   ~/data/scMAMAMIA/ok/scdesign2/_stale_clip3_backup/
# rather than deleted.
#
# Designed to be safe to run alongside other sweeps (max-workers 2, same
# convention as experiments/dp/v2/sweep.sh).
#
# Edit DATASET / ND / TRIALS / RANKS / EPSILONS at the top to change the spot.

set -euo pipefail

DATASET="${DATASET:-ok}"
ND="${ND:-20}"
TRIALS="${TRIALS:-1 2}"
RANKS="${RANKS:-5 10 20 50}"
EPSILONS="${EPSILONS:-1 100 10000 1000000 100000000}"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LR_DIR="$REPO_ROOT/experiments/dp/lowrank"
V2_DIR="$REPO_ROOT/experiments/dp/v2"
LOG_DIR="$LR_DIR/_logs"
mkdir -p "$LOG_DIR"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
banner() { echo; echo "============================================================"; echo "  $* — $(ts)"; echo "============================================================"; }

banner "lowrank spot check: dataset=$DATASET nd=$ND trials=[$TRIALS] ranks=[$RANKS] eps=[$EPSILONS]"

cd "$REPO_ROOT"

# ---- 0. Regenerate v2 (full-rank) baseline with corrected clip_value ----
banner "0/6  Regenerate v2 full-rank baseline (corrected TRUE_CLIP_VALUE)"
python "$V2_DIR/generate.py" \
    --dataset "$DATASET" --nd "$ND" --trial $TRIALS --epsilon $EPSILONS \
    2>&1 | tee "$LOG_DIR/gen_v2_baseline_${DATASET}_${ND}d.log"

# ---- 1. Train lowrank copulas (adds saved quantile_normal) ----
banner "1/6  Train lowrank copulas"
python "$LR_DIR/train.py" \
    --dataset "$DATASET" --nd "$ND" --trial $TRIALS --max-workers 2 \
    2>&1 | tee "$LOG_DIR/train_${DATASET}_${ND}d.log"

# ---- 2. Generate rank-only (no DP) ablation, one rank at a time ----
banner "2/6  Generate rank-only (no DP) synthetic, for each rank"
for R in $RANKS; do
    python "$LR_DIR/generate.py" \
        --dataset "$DATASET" --nd "$ND" --trial $TRIALS --rank "$R" --no-dp \
        2>&1 | tee -a "$LOG_DIR/gen_nodp_${DATASET}_${ND}d.log"
done

# ---- 3. Generate lowrank+DP synthetic, for each (rank, epsilon) ----
banner "3/6  Generate lowrank+DP synthetic, for each rank"
for R in $RANKS; do
    python "$LR_DIR/generate.py" \
        --dataset "$DATASET" --nd "$ND" --trial $TRIALS --rank "$R" --epsilon $EPSILONS \
        2>&1 | tee -a "$LOG_DIR/gen_dp_${DATASET}_${ND}d.log"
done

# ---- 4. Quality + MIA evals for v2 baseline ----
banner "4/6  Quality + MIA evaluations (v2 baseline)"
python "$V2_DIR/run_evals.py" \
    --dataset "$DATASET" --nd "$ND" --trial $TRIALS \
    2>&1 | tee "$LOG_DIR/evals_v2_${DATASET}_${ND}d.log"

# ---- 5. Quality + MIA evals for lowrank ----
banner "5/6  Quality + MIA evaluations (lowrank)"
python "$LR_DIR/run_evals.py" \
    --dataset "$DATASET" --nd "$ND" --trial $TRIALS \
    2>&1 | tee "$LOG_DIR/evals_lowrank_${DATASET}_${ND}d.log"

# ---- 6. Side-by-side comparison ----
banner "6/6  lowrank vs v2 comparison table"
python "$LR_DIR/compare.py" --dataset "$DATASET" --nd "$ND" \
    2>&1 | tee "$LOG_DIR/compare_${DATASET}_${ND}d.log"

banner "lowrank spot check done"

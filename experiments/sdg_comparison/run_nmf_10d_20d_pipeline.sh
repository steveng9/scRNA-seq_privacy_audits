#!/bin/bash
# run_nmf_10d_20d_pipeline.sh
# Full pipeline for NMF 10d/20d: generate → quad MIA attacks → quality evals.
# Runs entirely in the background; safe to log out.
#
# Launch with:
#   nohup bash experiments/sdg_comparison/run_nmf_10d_20d_pipeline.sh \
#       > /tmp/nmf_10d_20d_pipeline.log 2>&1 &
#   echo "PID: $!"
#
# Monitor with:
#   tail -f /tmp/nmf_10d_20d_pipeline.log
#   grep -E "^(===|---|\[ERROR\]|Progress)" /tmp/nmf_10d_20d_pipeline.log

set -e
REPO=/home/golobs/scRNA-seq_privacy_audits

echo "================================================================"
echo "NMF 10d/20d pipeline started"
date
echo "================================================================"

# ---------------------------------------------------------------------------
# Step 1: Generate NMF synthetic data (serial)
# ---------------------------------------------------------------------------
echo ""
echo "=== STEP 1: Generate NMF data (10d/20d, no_dp + eps 1..1e5) ==="
date
conda run --no-capture-output -n tabddpm_ \
    python $REPO/experiments/sdg_comparison/gen_nmf_10d_20d_sweep.py
echo "--- Generation complete ---"
date

# ---------------------------------------------------------------------------
# Step 2: Quad MIA attacks (BB +/-aux, with/without Class B)
# ---------------------------------------------------------------------------
echo ""
echo "=== STEP 2: Quad MIA attacks — NMF no_dp 10d ==="
date
python $REPO/experiments/sdg_comparison/run_mia_sweep.py --sdg nmf --nd 10

echo ""
echo "=== STEP 2: Quad MIA attacks — NMF no_dp 20d ==="
date
python $REPO/experiments/sdg_comparison/run_mia_sweep.py --sdg nmf --nd 20

echo ""
echo "=== STEP 2: Quad MIA attacks — NMF DP 10d ==="
date
python $REPO/experiments/sdg_comparison/run_mia_sweep.py --sdg nmf_dp --nd 10

echo ""
echo "=== STEP 2: Quad MIA attacks — NMF DP 20d ==="
date
python $REPO/experiments/sdg_comparison/run_mia_sweep.py --sdg nmf_dp --nd 20

echo "--- MIA attacks complete ---"
date

# ---------------------------------------------------------------------------
# Step 3: Quality evals (auto-discovers all NMF variants/nd/trial)
# ---------------------------------------------------------------------------
echo ""
echo "=== STEP 3: Quality evals — NMF (all variants) ==="
date
python $REPO/experiments/sdg_comparison/run_quality_evals.py \
    --dataset-filter nmf \
    --workers 4 \
    --max-donors 100

echo ""
echo "================================================================"
echo "NMF 10d/20d pipeline COMPLETE"
date
echo "================================================================"

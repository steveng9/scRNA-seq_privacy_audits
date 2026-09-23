#!/usr/bin/env bash
# Sequential launcher for scDiffusion-v3 UMAPs.
# Runs 2-panel comparison first, then the 5×4 grid.
# Intended to be run under nohup.
set -euo pipefail

REPO=/home/golobs/scRNA-seq_privacy_audits
PYTHON=/home/golobs/miniconda3/envs/recon_/bin/python
LOG=/tmp/scdiff_v3_umaps.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Starting scDiffusion-v3 UMAP pipeline ==="

log "--- [1/2] 2-panel comparison (10d+50d, no subsampling) ---"
$PYTHON "$REPO/experiments/sdg_comparison/make_scdiffusion_v3_umaps.py" \
    2>&1 | tee -a "$LOG"
log "--- [1/2] done ---"

log "--- [2/2] 5x4 grid UMAP (50d, 5 trials, no subsampling) ---"
$PYTHON "$REPO/experiments/sdg_comparison/make_scdiffusion_v3_grid_umap.py" \
    2>&1 | tee -a "$LOG"
log "--- [2/2] done ---"

log "=== All UMAPs complete ==="

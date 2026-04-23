#!/usr/bin/env bash
# run_zinbwave_generation.sh
#
# Generate OneK1K ZINBWave synthetic datasets for 10d, 20d, and 50d,
# 5 trials each.  Runs in the background so you can log out safely.
#
# Launch with:
#   nohup bash experiments/sdg_comparison/run_zinbwave_generation.sh \
#       > /tmp/zinbwave_generation.log 2>&1 &
#   echo "PID: $!"
#
# Check progress:
#   tail -f /tmp/zinbwave_generation.log
#   python experiments/sdg_comparison/check_generated_data.py

set -euo pipefail

REPO="/home/golobs/scRNA-seq_privacy_audits"
DATASET="/home/golobs/data/scMAMAMIA/ok/full_dataset_cleaned.h5ad"
HVG_PATH="/home/golobs/data/scMAMAMIA/ok/hvg_full.csv"
OUT_BASE="/home/golobs/data/scMAMAMIA/ok/zinbwave/no_dp"
SPLITS_BASE="/home/golobs/data/scMAMAMIA/ok/splits"
GENERATE_SCRIPT="${REPO}/experiments/sdg_comparison/generate_trial.py"

# Latent factors and cell-type fitting cap
N_LATENT=10
MAX_CELLS=3000
N_WORKERS=4   # parallel R processes per trial

echo "============================================================"
echo "  ZINBWave generation — OneK1K (ok), 5 trials each"
echo "  Donor counts: 10, 20, 50"
echo "  K=${N_LATENT}  max_cells/type=${MAX_CELLS}  workers=${N_WORKERS}"
echo "  Start: $(date)"
echo "============================================================"

for ND in 10 20 50; do
    for TRIAL in 1 2 3 4 5; do
        SPLITS_DIR="${SPLITS_BASE}/${ND}d/${TRIAL}"
        OUT_DIR="${OUT_BASE}/${ND}d/${TRIAL}"
        SYNTH_OUT="${OUT_DIR}/datasets/synthetic.h5ad"

        if [ -f "${SYNTH_OUT}" ]; then
            echo "[SKIP] ${ND}d trial ${TRIAL} already done: ${SYNTH_OUT}"
            continue
        fi

        echo ""
        echo "------------------------------------------------------------"
        echo "  ${ND}d trial ${TRIAL}  →  ${OUT_DIR}"
        echo "  Start: $(date)"
        echo "------------------------------------------------------------"

        python "${GENERATE_SCRIPT}" \
            --generator         zinbwave \
            --dataset           "${DATASET}" \
            --splits-dir        "${SPLITS_DIR}" \
            --out-dir           "${OUT_DIR}" \
            --hvg-path          "${HVG_PATH}" \
            --n-latent          "${N_LATENT}" \
            --max-cells-per-type "${MAX_CELLS}" \
            --zinbwave-workers  "${N_WORKERS}" \
            --zinbwave-seed     "${TRIAL}"

        echo "  Done: $(date)"
    done
done

echo ""
echo "============================================================"
echo "  ZINBWave generation complete: $(date)"
echo "============================================================"

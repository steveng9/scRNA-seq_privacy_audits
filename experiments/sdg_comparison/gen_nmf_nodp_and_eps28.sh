#!/bin/bash
# gen_nmf_nodp_and_eps28.sh
# Regenerate ok/nmf/no_dp/50d and ok/nmf/eps_2.8/50d serially.
# Launched after deleting uncertain-provenance no_dp data.
# Run with: nohup bash experiments/sdg_comparison/gen_nmf_nodp_and_eps28.sh > /tmp/nmf_nodp_eps28.log 2>&1 &

set -e
REPO=/home/golobs/scRNA-seq_privacy_audits
DATA=/home/golobs/data/scMAMAMIA
SPLITS_BASE=$DATA/ok/scdesign2/no_dp

echo "=== NMF no_dp + eps_2.8 serial generation (ok, 50d, trials 1-5) ==="
date

# ---- no_dp ----
for trial in 1 2 3 4 5; do
    echo ""
    echo "--- no_dp  trial $trial ---"
    date
    conda run --no-capture-output -n tabddpm_ \
        python $REPO/experiments/sdg_comparison/generate_trial.py \
            --generator nmf \
            --dataset $DATA/ok/full_dataset_cleaned.h5ad \
            --splits-dir $SPLITS_BASE/50d/$trial/datasets \
            --out-dir $DATA/ok/nmf/no_dp/50d/$trial \
            --hvg-path $DATA/ok/hvg_full.csv \
            --conda-env nmf_ \
            --dp-mode none
done

# ---- eps_2.8 (CAMDA defaults: eps_nmf=0.5, eps_kmeans=2.1, eps_summaries=0.2) ----
for trial in 1 2 3 4 5; do
    echo ""
    echo "--- eps_2.8  trial $trial ---"
    date
    conda run --no-capture-output -n tabddpm_ \
        python $REPO/experiments/sdg_comparison/generate_trial.py \
            --generator nmf \
            --dataset $DATA/ok/full_dataset_cleaned.h5ad \
            --splits-dir $SPLITS_BASE/50d/$trial/datasets \
            --out-dir $DATA/ok/nmf/eps_2.8/50d/$trial \
            --hvg-path $DATA/ok/hvg_full.csv \
            --conda-env nmf_ \
            --dp-mode all \
            --dp-eps-nmf 0.5 \
            --dp-eps-kmeans 2.1 \
            --dp-eps-summaries 0.2
done

echo ""
echo "=== All done ==="
date

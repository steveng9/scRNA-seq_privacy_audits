"""
scVI Membership Inference Attack — experiment runner.

Runs N trials of the ELBO-based MIA against an scVI model trained on
scRNA-seq data.  For each trial:

  1. Sample balanced train / holdout / auxiliary donor splits
  2. Train scVI on D_train
  3. Score all target cells (D_train + D_holdout) with the target model
  4. Train an auxiliary scVI on D_aux; score target cells (for +aux variant)
  5. Aggregate cell ELBO scores to donor level; compute AUC
  6. Save per-trial and summary results to <out_dir>/scvi_mia_results.csv

Usage
-----
  python experiments/scvi/run_scvi_mia.py \\
      --dataset  /path/to/full_dataset_cleaned.h5ad \\
      --out-dir  /path/to/results/scvi/10d \\
      --n-donors 10 \\
      --n-trials 5 \\
      --individual-col individual \\
      --cell-type-col  cell_type

Requirements
------------
  - conda env "scvi_" with scvi-tools >= 1.1 (see experiments/scvi/setup_scvi_env.sh)
  - The caller env needs: anndata, scanpy, numpy, pandas, scikit-learn
    (i.e., tabddpm_ or camda_ both work)
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_DIR   = os.path.join(_REPO_ROOT, "src")
for _d in [_REPO_ROOT, _SRC_DIR]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from attacks.scvi_mia.attack import (
    attack_scvi_elbo,
    attack_scvi_elbo_no_aux,
    aggregate_to_donors,
)
from sdg.scvi.model import ScVI


# ---------------------------------------------------------------------------
# Donor sampling (balanced MIA setup)
# ---------------------------------------------------------------------------

def sample_donors(all_donors, n_donors, rng, min_aux=20):
    """
    Sample equal-sized train and holdout pools (each n_donors donors).
    Auxiliary pool: non-target donors first, falling back to targets if needed.
    """
    n_total = len(all_donors)
    n_used  = min(n_donors, n_total // 2)

    target_idx   = rng.choice(n_total, size=n_used * 2, replace=False)
    target       = all_donors[target_idx]
    train_donors = target[:n_used]
    hold_donors  = target[n_used:]

    non_target   = np.setdiff1d(all_donors, target)
    n_aux        = max(min_aux, n_used)
    aux_pool     = np.concatenate([
        rng.permutation(non_target),
        rng.permutation(target),
    ])[:n_aux]

    return train_donors, hold_donors, aux_pool


# ---------------------------------------------------------------------------
# Data slicing helpers
# ---------------------------------------------------------------------------

def slice_donors(adata, donors, individual_col):
    mask = adata.obs[individual_col].isin(donors)
    subset = adata[mask].copy()
    return subset


def add_member_column(adata, train_donors, individual_col):
    """Add 'member' column: 1 for train donors, 0 for holdout."""
    adata.obs["member"] = adata.obs[individual_col].isin(train_donors).astype(int)
    return adata


# ---------------------------------------------------------------------------
# One MIA trial
# ---------------------------------------------------------------------------

def run_trial(
    adata_full,
    trial_num,
    n_donors,
    out_dir,
    individual_col,
    cell_type_col,
    hvg_path,
    conda_env,
    scvi_kwargs,
    seed,
    run_aux,
):
    rng = np.random.default_rng(seed)
    all_donors = adata_full.obs[individual_col].unique()

    train_donors, hold_donors, aux_donors = sample_donors(
        all_donors, n_donors, rng
    )

    print(f"\n[Trial {trial_num}]  train={len(train_donors)}  "
          f"hold={len(hold_donors)}  aux={len(aux_donors)}", flush=True)

    trial_dir = os.path.join(out_dir, str(trial_num))
    os.makedirs(trial_dir, exist_ok=True)

    # Save donor splits
    np.save(os.path.join(trial_dir, "train_donors.npy"),   train_donors)
    np.save(os.path.join(trial_dir, "holdout_donors.npy"), hold_donors)
    np.save(os.path.join(trial_dir, "aux_donors.npy"),     aux_donors)

    # Build H5ADs
    ds_dir = os.path.join(trial_dir, "datasets")
    os.makedirs(ds_dir, exist_ok=True)

    train_h5ad  = os.path.join(ds_dir, "train.h5ad")
    target_h5ad = os.path.join(ds_dir, "target.h5ad")
    aux_h5ad    = os.path.join(ds_dir, "aux.h5ad")

    # Training data (D_train only — fed to scVI)
    adata_train = slice_donors(adata_full, train_donors, individual_col)
    adata_train.write_h5ad(train_h5ad, compression="gzip")
    print(f"  D_train: {adata_train.n_obs} cells", flush=True)

    # Target data (D_train + D_hold, labelled) — scored for MIA
    adata_target = slice_donors(
        adata_full,
        np.concatenate([train_donors, hold_donors]),
        individual_col,
    )
    adata_target = add_member_column(adata_target, train_donors, individual_col)
    adata_target.write_h5ad(target_h5ad, compression="gzip")

    # Auxiliary data (D_aux)
    adata_aux = slice_donors(adata_full, aux_donors, individual_col)
    adata_aux.write_h5ad(aux_h5ad, compression="gzip")
    print(f"  D_aux: {adata_aux.n_obs} cells", flush=True)

    # ----- Build a minimal config dict for ScVI wrapper -----
    model_dir_target = os.path.join(trial_dir, "models", "target")
    model_dir_aux    = os.path.join(trial_dir, "models", "aux")

    def _make_scvi_cfg(model_dir, train_file, test_file):
        return {
            "generator_name": "scvi",
            "dir_list": {"home": trial_dir},
            "dataset_config": {
                "name":                "dataset",
                "train_count_file":    train_file,
                "test_count_file":     test_file,
                "cell_type_col_name":  cell_type_col,
                "cell_label_col_name": cell_type_col,
            },
            "scvi_config": {
                "model_dir":   os.path.relpath(model_dir, trial_dir),
                "hvg_path":    hvg_path,
                "conda_env":   conda_env,
                **scvi_kwargs,
            },
        }

    # ----- Train target model -----
    t0 = time.time()
    target_cfg = _make_scvi_cfg(
        model_dir_target,
        train_file=os.path.relpath(train_h5ad, ds_dir),
        test_file=os.path.relpath(train_h5ad, ds_dir),
    )
    # Override data_dir to ds_dir so filenames resolve
    target_cfg["dir_list"]["data"] = ds_dir
    target_cfg["dataset_config"]["train_count_file"] = "train.h5ad"
    target_cfg["dataset_config"]["test_count_file"]  = "train.h5ad"

    scvi_target = ScVI(target_cfg)
    print("  Training target scVI ...", flush=True)
    scvi_target.train()
    train_time = time.time() - t0
    print(f"  Target model trained in {train_time:.0f}s", flush=True)

    # ----- Score target cells with target model -----
    scores_target_path = os.path.join(trial_dir, "scores_target.npy")
    print("  Scoring target cells ...", flush=True)
    elbo_target = scvi_target.score_cells(target_h5ad, scores_target_path)

    # ----- No-aux attack -----
    obs = adata_target.obs.copy()
    obs.index = adata_target.obs_names
    obs["individual"] = adata_target.obs[individual_col]

    _, y_true_noaux, y_pred_noaux = attack_scvi_elbo_no_aux(
        elbo_target, obs, tm_code="110"
    )
    auc_noaux = roc_auc_score(y_true_noaux, y_pred_noaux)
    print(f"  AUC (no-aux): {auc_noaux:.4f}", flush=True)

    auc_aux = None
    if run_aux:
        # ----- Train auxiliary model -----
        aux_cfg = _make_scvi_cfg(
            model_dir_aux,
            train_file="aux.h5ad",
            test_file="aux.h5ad",
        )
        aux_cfg["dir_list"]["data"] = ds_dir
        scvi_aux = ScVI(aux_cfg)
        print("  Training auxiliary scVI ...", flush=True)
        scvi_aux.train()

        # ----- Score target cells with aux model -----
        scores_aux_path = os.path.join(trial_dir, "scores_aux.npy")
        elbo_aux = scvi_aux.score_cells(target_h5ad, scores_aux_path)

        # ----- Aux-calibrated attack -----
        _, y_true_aux, y_pred_aux = attack_scvi_elbo(
            elbo_target, elbo_aux, obs, tm_code="100"
        )
        auc_aux = roc_auc_score(y_true_aux, y_pred_aux)
        print(f"  AUC (+aux):   {auc_aux:.4f}", flush=True)

    return dict(
        trial=trial_num,
        n_donors=n_donors,
        seed=seed,
        n_train_cells=adata_train.n_obs,
        auc_noaux=auc_noaux,
        auc_aux=auc_aux,
        train_time_s=train_time,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="scVI MIA experiment runner")
    parser.add_argument("--dataset",        required=True,
                        help="Full dataset .h5ad path")
    parser.add_argument("--out-dir",        required=True,
                        help="Output directory for results")
    parser.add_argument("--n-donors",       type=int, default=10,
                        help="Number of train (= holdout) donors per trial")
    parser.add_argument("--n-trials",       type=int, default=5,
                        help="Number of independent MIA trials")
    parser.add_argument("--base-seed",      type=int, default=0,
                        help="Base random seed; trial k uses seed base+k")
    parser.add_argument("--individual-col", default="individual",
                        help="obs column name for donor ID")
    parser.add_argument("--cell-type-col",  default="cell_type",
                        help="obs column name for cell type")
    parser.add_argument("--hvg-path",       default=None,
                        help="Path to HVG CSV (dataset-level, shared across trials)")
    parser.add_argument("--conda-env",      default="scvi_",
                        help="Conda environment that has scvi-tools installed")
    parser.add_argument("--no-aux",         action="store_true",
                        help="Skip the auxiliary-model (+aux) attack variant")
    # scVI hyperparameters
    parser.add_argument("--n-latent",   type=int, default=30)
    parser.add_argument("--n-layers",   type=int, default=2)
    parser.add_argument("--n-hidden",   type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=512)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset}", flush=True)
    adata_full = sc.read_h5ad(args.dataset)
    print(f"  {adata_full.n_obs} cells × {adata_full.n_vars} genes  "
          f"| donors: {adata_full.obs[args.individual_col].nunique()}", flush=True)

    scvi_kwargs = dict(
        n_latent=args.n_latent,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
    )

    all_results = []
    for trial in range(1, args.n_trials + 1):
        seed = args.base_seed + trial
        try:
            result = run_trial(
                adata_full     = adata_full,
                trial_num      = trial,
                n_donors       = args.n_donors,
                out_dir        = args.out_dir,
                individual_col = args.individual_col,
                cell_type_col  = args.cell_type_col,
                hvg_path       = args.hvg_path,
                conda_env      = args.conda_env,
                scvi_kwargs    = scvi_kwargs,
                seed           = seed,
                run_aux        = not args.no_aux,
            )
            all_results.append(result)
        except Exception as e:
            print(f"[Trial {trial}] FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()

    if not all_results:
        print("No trials completed.", flush=True)
        return

    df = pd.DataFrame(all_results)
    results_path = os.path.join(args.out_dir, "scvi_mia_results.csv")
    df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    print("\n=== Summary ===")
    print(f"n_donors = {args.n_donors}   n_trials = {len(df)}")
    print(f"AUC (no-aux)  mean={df['auc_noaux'].mean():.4f}  "
          f"std={df['auc_noaux'].std():.4f}")
    if df["auc_aux"].notna().any():
        print(f"AUC (+aux)    mean={df['auc_aux'].mean():.4f}  "
              f"std={df['auc_aux'].std():.4f}")
    print(df[["trial", "n_donors", "auc_noaux", "auc_aux"]].to_string(index=False))


if __name__ == "__main__":
    main()

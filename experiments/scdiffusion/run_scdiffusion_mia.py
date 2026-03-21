"""
scDiffusion MIA experiment runner.

Usage
-----
  conda run -n tabddpm_ python experiments/scdiffusion/run_scdiffusion_mia.py \\
      --dataset  /path/to/full_dataset_cleaned.h5ad \\
      --out-dir  /path/to/results/scdiffusion/10d \\
      --n-donors 10  --n-trials 5 \\
      --hvg-path /path/to/hvg.csv

Key options
-----------
  --dataset           full h5ad (all donors)
  --out-dir           results root; trials land in <out-dir>/<trial>/
  --n-donors          train (= holdout) donors per trial
  --n-trials          number of independent MIA trials
  --individual-col    obs column for donor ID        (default: individual)
  --cell-type-col     obs column for cell type       (default: cell_type)
  --hvg-path          shared HVG CSV; if set, used for all trials
  --conda-env         conda env with scDiffusion     (default: scdiff_)
  --vae-steps         VAE training iterations        (default: 150000)
  --diff-steps        diffusion training iterations  (default: 300000)
  --n-score-times     noise levels for MIA score     (default: 50)
  --base-seed         base random seed               (default: 0)

Output
------
  <out-dir>/scdiffusion_mia_results.csv   — per-trial AUC + timing
  <out-dir>/<trial>/
    datasets/  train.h5ad  target.h5ad
    models/vae/   model_seed=0_step=*.pt
    models/diff/  model*.pt
    scores.npy    per-cell denoising-loss MIA scores
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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_DIR   = os.path.join(_REPO_ROOT, "src")
for d in [_REPO_ROOT, _SRC_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

from sdg.scdiffusion.model import ScDiffusion
from attacks.scdiffusion_mia.attack import attack_scdiffusion


# ---------------------------------------------------------------------------
# Donor split helpers (identical to scVI runner)
# ---------------------------------------------------------------------------

def sample_donors(all_donors, n_donors, rng, min_aux=20):
    n = min(n_donors, len(all_donors) // 2)
    idx = rng.choice(len(all_donors), size=n*2, replace=False)
    target      = all_donors[idx]
    train, hold = target[:n], target[n:]
    non_target  = np.setdiff1d(all_donors, target)
    n_aux = max(min_aux, n)
    aux = np.concatenate([rng.permutation(non_target), rng.permutation(target)])[:n_aux]
    return train, hold, aux


def slice_donors(adata, donors, col):
    return adata[adata.obs[col].isin(donors)].copy()


# ---------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------

def run_trial(adata_full, trial_num, n_donors, out_dir, individual_col,
              cell_type_col, hvg_path, conda_env, model_kwargs, seed):
    rng = np.random.default_rng(seed)
    all_donors = adata_full.obs[individual_col].unique()
    train_donors, hold_donors, _ = sample_donors(all_donors, n_donors, rng)

    print(f"\n[Trial {trial_num}]  train={len(train_donors)}  hold={len(hold_donors)}", flush=True)

    trial_dir = os.path.join(out_dir, str(trial_num))
    ds_dir    = os.path.join(trial_dir, "datasets")
    os.makedirs(ds_dir, exist_ok=True)

    train_h5ad  = os.path.join(ds_dir, "train.h5ad")
    target_h5ad = os.path.join(ds_dir, "target.h5ad")

    adata_train = slice_donors(adata_full, train_donors, individual_col)
    adata_train.write_h5ad(train_h5ad, compression="gzip")

    adata_target = slice_donors(
        adata_full, np.concatenate([train_donors, hold_donors]), individual_col
    )
    adata_target.obs["member"] = adata_target.obs[individual_col].isin(train_donors).astype(int)
    adata_target.write_h5ad(target_h5ad, compression="gzip")

    # Build ScDiffusion config
    cfg = {
        "generator_name": "scdiffusion",
        "dir_list":       {"home": trial_dir},
        "dataset_config": {
            "name":                "dataset",
            "train_count_file":    "train.h5ad",
            "test_count_file":     "train.h5ad",
            "cell_type_col_name":  cell_type_col,
            "cell_label_col_name": cell_type_col,
        },
        "scdiffusion_config": {
            "vae_dir":    "models/vae",
            "diff_dir":   "models/diff",
            "hvg_path":   hvg_path,
            "conda_env":  conda_env,
            **model_kwargs,
        },
    }
    cfg["dataset_config"]["name"] = "dataset"
    # point data_dir to ds_dir so filenames resolve
    cfg["dir_list"]["data"] = ds_dir

    gen = ScDiffusion(cfg)

    t0 = time.time()
    print("  Training scDiffusion (VAE + diffusion backbone) ...", flush=True)
    gen.train()
    train_time = time.time() - t0
    print(f"  Training done in {train_time/60:.1f} min", flush=True)

    scores_npy = os.path.join(trial_dir, "scores.npy")
    scores = gen.score_cells(target_h5ad, scores_npy)

    obs = adata_target.obs.copy()
    obs.index = adata_target.obs_names
    obs["individual"] = adata_target.obs[individual_col]

    _, y_true, y_pred = attack_scdiffusion(scores, obs, tm_code="110")
    auc = roc_auc_score(y_true, y_pred)
    print(f"  AUC = {auc:.4f}", flush=True)

    return dict(trial=trial_num, n_donors=n_donors, seed=seed,
                n_train_cells=adata_train.n_obs, auc=auc, train_time_s=train_time)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",        required=True)
    ap.add_argument("--out-dir",        required=True)
    ap.add_argument("--n-donors",       type=int, default=10)
    ap.add_argument("--n-trials",       type=int, default=5)
    ap.add_argument("--base-seed",      type=int, default=0)
    ap.add_argument("--individual-col", default="individual")
    ap.add_argument("--cell-type-col",  default="cell_type")
    ap.add_argument("--hvg-path",       default=None)
    ap.add_argument("--conda-env",      default="scdiff_")
    ap.add_argument("--latent-dim",     type=int, default=128)
    ap.add_argument("--vae-steps",      type=int, default=150000)
    ap.add_argument("--diff-steps",     type=int, default=300000)
    ap.add_argument("--batch-size",     type=int, default=512)
    ap.add_argument("--save-interval",  type=int, default=50000)
    ap.add_argument("--n-score-times",  type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    adata_full = sc.read_h5ad(args.dataset)
    print(f"Dataset: {adata_full.n_obs} cells  "
          f"{adata_full.obs[args.individual_col].nunique()} donors", flush=True)

    model_kwargs = dict(
        latent_dim=args.latent_dim,
        vae_steps=args.vae_steps,
        diff_steps=args.diff_steps,
        batch_size=args.batch_size,
        save_interval=args.save_interval,
    )

    results = []
    for trial in range(1, args.n_trials + 1):
        try:
            r = run_trial(
                adata_full, trial, args.n_donors, args.out_dir,
                args.individual_col, args.cell_type_col,
                args.hvg_path, args.conda_env, model_kwargs,
                seed=args.base_seed + trial,
            )
            results.append(r)
        except Exception as e:
            print(f"[Trial {trial}] FAILED: {e}")
            import traceback; traceback.print_exc()

    if not results:
        print("No trials completed.")
        return

    df = pd.DataFrame(results)
    out_csv = os.path.join(args.out_dir, "scdiffusion_mia_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nResults: {out_csv}")
    print(f"AUC  mean={df['auc'].mean():.4f}  std={df['auc'].std():.4f}")
    print(df[["trial", "n_donors", "auc", "train_time_s"]].to_string(index=False))


if __name__ == "__main__":
    main()

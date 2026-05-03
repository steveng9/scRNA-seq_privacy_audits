"""
Extract per-cell-type (n_cells, k_max, n_genes) for every (dataset, n_donors, trial)
and emit two CSVs:

  - notes/dp_norms/copula_dims.csv      — one row per (ds, nd, trial, cell_type)
  - notes/dp_norms/sigma_table.csv      — one row per (ds, nd, trial, ct, dp_variant, epsilon)

Constants pulled from src/sdg/dp/sensitivity.py and src/generators/gen_dp_quality_data.py:
    c       = 3.0   (entry-wise clip in v1/v2 proof)
    delta   = 1e-5
    v1 prefactor on per-entry sensitivity = 4   (centered covariance)
    v2 prefactor on per-entry sensitivity = 1   (uncentered second moment)
    Per-cell-type Frobenius sensitivity:
        Δ_F = prefactor * c² * G * k_max / (n_cells - k_max)
    Classical Gaussian noise scale:
        σ = Δ_F * sqrt(2 ln(1.25/δ)) / ε

The sigma is **per cell type** and **per the v1 proof's interpretation** that ε
is the per-cell-type budget passed to gaussian_noise_scale at runtime.
"""
import os, sys, json, math, glob
import numpy as np
import pandas as pd
import anndata as ad

ROOT = "/home/golobs/data/scMAMAMIA"
OUT_DIR = "/home/golobs/scRNA-seq_privacy_audits/notes/dp_norms"
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = ["ok", "aida"]
N_DONORS_LIST = ["5d", "10d", "20d", "50d", "100d", "200d", "490d"]
TRIALS = ["1", "2", "3", "4", "5", "6"]

CLIP_VALUE = 3.0
DELTA = 1e-5
EPS_GRID = [1, 10, 100, 1000, 10_000, 100_000, 1_000_000,
            10_000_000, 100_000_000, 1_000_000_000]


def load_full_obs(dataset):
    h5 = os.path.join(ROOT, dataset, "full_dataset_cleaned.h5ad")
    a = ad.read_h5ad(h5, backed="r")
    obs = a.obs[["individual", "cell_type"]].copy()
    a.file.close()
    obs["individual"] = obs["individual"].astype(str)
    obs["cell_type"] = obs["cell_type"].astype(str)
    return obs


def k_max_for_donors(full_obs, train_donors, ct):
    sub = full_obs[(full_obs["individual"].isin(train_donors)) &
                   (full_obs["cell_type"] == ct)]
    if len(sub) == 0:
        return 0, 0
    counts = sub.groupby("individual").size()
    return int(counts.max()), int(len(sub))


def extract_copula_metadata_via_R(rds_path):
    """Returns (n_cell, n_primary_genes, has_cov_mat) for one cell type."""
    from rpy2.robjects import r as R
    obj = R["readRDS"](rds_path)
    # The .rds in models/{ct}.rds is a list keyed by ct; pick first key
    ct_key = list(obj.names)[0]
    ct_obj = obj.rx2(ct_key)
    try:
        n_cell = int(ct_obj.rx2("n_cell")[0])
    except Exception:
        n_cell = -1
    try:
        gene_sel1 = ct_obj.rx2("gene_sel1")
        n_primary = len(list(gene_sel1.names))
    except Exception:
        n_primary = -1
    cov_mat = ct_obj.rx2("cov_mat")
    from rpy2.rinterface_lib.sexp import NULLType as _NULL
    has_cov_mat = not isinstance(cov_mat, _NULL)
    return n_cell, n_primary, has_cov_mat, ct_key


def main():
    rows = []
    for ds in DATASETS:
        print(f"=== {ds} ===", flush=True)
        full_obs = load_full_obs(ds)
        nodp_root = os.path.join(ROOT, ds, "scdesign2", "no_dp")
        if not os.path.isdir(nodp_root):
            continue

        for nd in N_DONORS_LIST:
            nd_dir = os.path.join(nodp_root, nd)
            if not os.path.isdir(nd_dir):
                continue
            for tr in TRIALS:
                tr_dir = os.path.join(nd_dir, tr)
                models_dir = os.path.join(tr_dir, "models")
                splits_dir = os.path.join(ROOT, ds, "splits", nd, tr)
                train_npy = os.path.join(splits_dir, "train.npy")
                if not os.path.isdir(models_dir) or not os.path.exists(train_npy):
                    continue
                train_donors = set(np.load(train_npy, allow_pickle=True).astype(str).tolist())

                rds_files = sorted(glob.glob(os.path.join(models_dir, "*.rds")))
                print(f"  [{ds} {nd} t{tr}] {len(rds_files)} cell types", flush=True)

                for rds in rds_files:
                    ct = os.path.splitext(os.path.basename(rds))[0]
                    try:
                        n_cell, n_primary, has_cov, ct_key = extract_copula_metadata_via_R(rds)
                    except Exception as e:
                        print(f"    [WARN] {rds}: {e}")
                        continue
                    k_max, n_train_cells_obs = k_max_for_donors(full_obs, train_donors, ct)
                    rows.append({
                        "dataset": ds,
                        "n_donors": nd,
                        "trial": int(tr),
                        "cell_type": ct,
                        "n_cells_in_copula": n_cell,
                        "n_train_cells_in_obs": n_train_cells_obs,
                        "n_genes_primary": n_primary,
                        "k_max": k_max,
                        "has_cov_mat": has_cov,
                    })

                # Periodic save
                pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "copula_dims.csv"), index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "copula_dims.csv"), index=False)
    print(f"\nSaved {len(df)} rows to copula_dims.csv", flush=True)

    # Build sigma_table.csv
    out_rows = []
    sqrt_term = math.sqrt(2.0 * math.log(1.25 / DELTA))
    for _, r in df.iterrows():
        n, k, G = r["n_cells_in_copula"], r["k_max"], r["n_genes_primary"]
        if n <= 0 or k <= 0 or G <= 0 or k >= n:
            continue
        for variant, prefactor in [("v1", 4.0), ("v2", 1.0)]:
            delta_F = prefactor * (CLIP_VALUE ** 2) * G * k / (n - k)
            for eps in EPS_GRID:
                sigma = delta_F * sqrt_term / eps
                out_rows.append({
                    "dataset": r["dataset"],
                    "n_donors": r["n_donors"],
                    "trial": r["trial"],
                    "cell_type": r["cell_type"],
                    "n_cells": n,
                    "k_max": k,
                    "n_genes": G,
                    "dp_variant": variant,
                    "prefactor": prefactor,
                    "clip_value": CLIP_VALUE,
                    "delta": DELTA,
                    "epsilon": eps,
                    "delta_F": delta_F,
                    "sigma": sigma,
                })
    sigma_df = pd.DataFrame(out_rows)
    sigma_df.to_csv(os.path.join(OUT_DIR, "sigma_table.csv"), index=False)
    print(f"Saved {len(sigma_df)} rows to sigma_table.csv", flush=True)


if __name__ == "__main__":
    main()

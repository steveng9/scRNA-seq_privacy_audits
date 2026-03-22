"""
Recompute HVGs from the full dataset (all donors) for OneK1K and AIDA.

Saves to <data_dir>/<dataset>/hvg_full.csv — a separate file from the
existing hvg.csv (computed from per-trial train subsets) so that existing
scDesign2 experiments are not disturbed.

Also reports overlap statistics vs the existing hvg.csv.

Usage
-----
    conda run -n tabddpm_ python experiments/sdg_comparison/compute_hvgs.py

Output files
------------
    /home/golobs/data/ok/hvg_full.csv
    /home/golobs/data/aida/hvg_full.csv
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc

DATA_DIR = "/home/golobs/data"

DATASETS = {
    "ok":   os.path.join(DATA_DIR, "ok",   "full_dataset_cleaned.h5ad"),
    "aida": os.path.join(DATA_DIR, "aida", "full_dataset_cleaned.h5ad"),
}

HVG_PARAMS = dict(min_mean=0.0125, max_mean=3, min_disp=0.5)


def compute_hvgs_for_dataset(name: str, h5ad_path: str) -> pd.DataFrame:
    existing_path = os.path.join(DATA_DIR, name, "hvg.csv")
    new_path      = os.path.join(DATA_DIR, name, "hvg_full.csv")

    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"  Loading {h5ad_path} ...", flush=True)
    adata = sc.read_h5ad(h5ad_path)
    print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes | "
          f"{adata.obs['individual'].nunique()} donors", flush=True)

    # Normalise + log1p in a scratch layer so raw counts are unchanged.
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, layer="counts", target_sum=1e4)
    sc.pp.log1p(adata, layer="counts")

    sc.pp.highly_variable_genes(adata, layer="counts", **HVG_PARAMS)
    new_hvg = adata.var[["highly_variable"]].copy()
    new_hvg.to_csv(new_path)

    n_new = int(new_hvg["highly_variable"].sum())
    print(f"  New HVGs (full dataset):   {n_new}")

    # -------------------------------------------------------------------
    # Overlap comparison
    # -------------------------------------------------------------------
    if os.path.exists(existing_path):
        old_hvg = pd.read_csv(existing_path, index_col=0)
        old_set = set(old_hvg.index[old_hvg["highly_variable"]])
        new_set = set(new_hvg.index[new_hvg["highly_variable"]])

        n_old       = len(old_set)
        n_overlap   = len(old_set & new_set)
        n_only_old  = len(old_set - new_set)
        n_only_new  = len(new_set - old_set)
        jaccard     = n_overlap / len(old_set | new_set) if (old_set | new_set) else 0.0

        print(f"  Old HVGs (train subset):   {n_old}")
        print(f"  Overlap:                   {n_overlap} "
              f"({100*n_overlap/max(n_old,1):.1f}% of old, "
              f"{100*n_overlap/max(n_new,1):.1f}% of new)")
        print(f"  Only in old (dropped):     {n_only_old}")
        print(f"  Only in new (added):       {n_only_new}")
        print(f"  Jaccard similarity:        {jaccard:.4f}")
    else:
        print(f"  (no existing hvg.csv found at {existing_path})")

    print(f"  Saved → {new_path}")
    return new_hvg


def main():
    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping {name}.")
            continue
        compute_hvgs_for_dataset(name, path)

    print("\nDone.")


if __name__ == "__main__":
    main()

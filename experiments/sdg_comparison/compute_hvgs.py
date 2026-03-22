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

# Maximum cells to load into memory for HVG computation.  For large datasets
# (e.g. AIDA 57 GB / 1M cells) loading everything OOMs the machine.
# 200k cells from all donors is more than sufficient for stable HVG selection.
MAX_CELLS = 200_000


def compute_hvgs_for_dataset(name: str, h5ad_path: str) -> pd.DataFrame:
    existing_path = os.path.join(DATA_DIR, name, "hvg.csv")
    new_path      = os.path.join(DATA_DIR, name, "hvg_full.csv")

    print(f"\n{'='*60}")
    print(f"Dataset: {name}")

    if os.path.exists(new_path):
        print(f"  [SKIP] {new_path} already exists.")
        return pd.read_csv(new_path, index_col=0)

    # Read obs metadata only (backed='r') to decide whether to subsample.
    print(f"  Reading obs metadata from {h5ad_path} ...", flush=True)
    adata_backed = sc.read_h5ad(h5ad_path, backed="r")
    n_total = adata_backed.n_obs
    n_donors = adata_backed.obs["individual"].nunique()
    print(f"  {n_total:,} cells × {adata_backed.n_vars:,} genes | "
          f"{n_donors} donors", flush=True)

    if n_total > MAX_CELLS:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_total, size=MAX_CELLS, replace=False)
        idx.sort()   # backed h5ad requires sorted indices
        print(f"  Subsampling to {MAX_CELLS:,} cells for HVG computation ...", flush=True)
        adata = adata_backed[idx].to_memory()
    else:
        adata = adata_backed.to_memory()
    adata_backed.file.close()

    print(f"  Loaded {adata.n_obs:,} cells into memory.", flush=True)

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

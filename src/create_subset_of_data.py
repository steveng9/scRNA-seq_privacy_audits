data_dir = "/Users/stevengolob/Documents/school/PhD/Ghent_project_mia_scRNAseq/data/"
train = "onek1k/onek1k_annotated_train_release.h5ad"
test = "onek1k/onek1k_annotated_test_release.h5ad"

import sys
import anndata as ad
import numpy as np
from pathlib import Path


def main():

    for dataset in [train, test]:
        input_path = Path(data_dir + dataset)
        n_cells = 100_000

        # Read the full AnnData object
        print(f"Reading {input_path} ...")
        adata = ad.read_h5ad(input_path)

        # Randomly sample cells (rows = observations)
        n_cells = min(n_cells, adata.n_obs)
        subset_idx = np.random.choice(adata.n_obs, size=n_cells, replace=False)
        adata_subset = adata[subset_idx, :].copy()

        # Optionally: also downsample genes (uncomment if needed)
        # n_genes = 1000
        # subset_genes = np.random.choice(adata.n_vars, size=n_genes, replace=False)
        # adata_subset = adata_subset[:, subset_genes].copy()

        # Save the smaller file
        output_path = input_path.with_name(input_path.stem + "_subset_100k.h5ad")
        print(f"Saving subset ({adata_subset.n_obs} cells, {adata_subset.n_vars} genes) → {output_path}")
        adata_subset.write_h5ad(output_path)
        print("Done.")

if __name__ == "__main__":
    main()

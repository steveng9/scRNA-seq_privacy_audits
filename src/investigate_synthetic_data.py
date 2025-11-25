import anndata as ad
import pandas as pd
import numpy as np


synthetic_path = "/Users/stevengolob/Documents/school/PhD/Ghent_project_mia_scRNAseq/data/onek1k/synthetic_data/scdesign2/onek1k_annotated_synthetic_all.h5ad"
hvg_path = "/models/ok/hvg.csv"

adata = ad.read_h5ad(synthetic_path)

hvg_mask_raw = pd.read_csv(hvg_path, index_col=0)
if 'highly_variable' in hvg_mask_raw.columns:
    hvg_mask = hvg_mask_raw['highly_variable'].values
else:
    # In case the CSV is just a single column of booleans
    hvg_mask = hvg_mask_raw.values.flatten()

# Get dense data if sparse
X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

# Sanity check
print(f"Data shape: {X.shape}")
print(f"Num HVGs: {hvg_mask.sum()}")

# Compute whether any non-HVG gene has nonzero expression
non_hvg_zeros = np.all(X[:, ~hvg_mask] == 0)

if non_hvg_zeros:
    print("✅ All non-HVG genes are zero.")
else:
    print("collecting non zeroes found")
    num_nonzero_in_nonhvg = np.count_nonzero(X[:, ~hvg_mask])
    print(f"❌ Found {num_nonzero_in_nonhvg} nonzero values in non-HVG genes!")

import scanpy as sc
import numpy as np

def sparse_matrix_variance(sparse_matrix):
    sparse_matrix_csc = sparse_matrix.tocsc()
    means = sparse_matrix_csc.mean(axis=0).A1
    
    n_rows = sparse_matrix.shape[0]
    squared = sparse_matrix_csc.multiply(sparse_matrix_csc)
    mean_of_squares = squared.mean(axis=0).A1
    variances = mean_of_squares - means**2
    return variances

adata = sc.read_h5ad("data/processed/onek1k/onek1k_annotated_train.h5ad")
genes = adata.var.gene_ids
cell_ids = adata.obs.cell_label
cell_type_vector = adata.obs.cell_type
counts = adata.X
assert counts.shape[0] == len(cell_ids)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
print(adata.var[['highly_variable', 'means', 'dispersions']])
print(len(adata.var[adata.var['highly_variable']]))
hvg_mask = adata.var['highly_variable']
adata_hvg = adata[:, hvg_mask].copy()
adata_hvg.write("subset.h5ad")
exit(0)

expr_var = sparse_matrix_variance(counts)
print(f"MAX: {np.max(expr_var)}")
print(f"AVG: {np.mean(expr_var)}")

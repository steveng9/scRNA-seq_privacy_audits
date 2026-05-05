"""
Gaussian copula parsing and shared-gene algebra for scMAMA-MIA.

parse_copula()               — rpy2 R object → Python dict
load_copula()                — read .rds file and parse a single cell type
build_shared_covariance_matrix() — extract shared-gene sub-cov + marginals
get_shared_genes()           — intersection of primary gene sets
"""

import numpy as np
from rpy2.robjects import r as R
from rpy2.rinterface_lib.sexp import NULLType as _NullType


def parse_copula(r_copula_obj):
    """
    Parse a scDesign2 per-cell-type copula R object into plain Python.

    Parameters
    ----------
    r_copula_obj : rpy2 ListVector — one cell type's copula from readRDS(...)[[ct]]

    Returns
    -------
    dict with keys:
        primary_genes, secondary_genes, cov_matrix (None for vine copulas),
        primary_marginals, secondary_marginals,
        get_correlation(g1, g2) → float,
        get_gene_params(gene)   → (pi, theta, mu)
    """
    primary_genes   = list(r_copula_obj.rx2("gene_sel1").names)
    secondary_genes = list(r_copula_obj.rx2("gene_sel2").names)
    n_primary       = len(primary_genes)
    n_secondary     = len(secondary_genes)

    cov_mat_r = r_copula_obj.rx2("cov_mat")
    cov_matrix = None if isinstance(cov_mat_r, _NullType) else cov_mat_r

    primary_marginals   = r_copula_obj.rx2("marginal_param1")
    secondary_marginals = r_copula_obj.rx2("marginal_param2")

    gene_to_idx = {g: i for i, g in enumerate(primary_genes)}
    gene_loc    = {g: (0, i) for i, g in enumerate(primary_genes)}
    gene_loc.update({g: (1, i) for i, g in enumerate(secondary_genes)})

    def get_correlation(gene1, gene2):
        i1 = gene_to_idx[gene1]
        i2 = gene_to_idx[gene2]
        # R matrices are stored column-major; flat index = row + col * nrow
        return float(cov_matrix[i1 + i2 * n_primary])

    def get_gene_params(gene):
        cls, idx = gene_loc[gene]
        n   = n_primary if cls == 0 else n_secondary
        marg = primary_marginals if cls == 0 else secondary_marginals
        pi    = float(marg[idx])
        theta = float(marg[idx + n])
        mu    = float(marg[idx + 2 * n])
        return pi, theta, mu

    return dict(
        primary_genes       = primary_genes,
        secondary_genes     = secondary_genes,
        cov_matrix          = cov_matrix,
        primary_marginals   = primary_marginals,
        secondary_marginals = secondary_marginals,
        get_correlation     = get_correlation,
        get_gene_params     = get_gene_params,
    )


def load_copula(rds_path, cell_type):
    """
    Read a scDesign2 .rds file and return the parsed copula dict for one cell type.
    """
    r_obj = R["readRDS"](rds_path).rx2(str(cell_type))
    return parse_copula(r_obj)


def build_shared_covariance_matrix(shared_genes, all_primary_genes,
                                    cov_matrix_r, marginal_params_r):
    """
    Extract the sub-covariance matrix and marginals for the shared gene subset.

    Parameters
    ----------
    shared_genes       : list[str] — genes present in both copulas' primary set
    all_primary_genes  : list[str] — full primary gene list for this copula
    cov_matrix_r       : rpy2 matrix or numpy array (n_primary × n_primary)
    marginal_params_r  : rpy2 matrix or numpy array (n_primary × 3: pi, theta, mu)

    Returns
    -------
    shared_cov       : np.ndarray  (n_shared × n_shared)
    shared_marginals : np.ndarray  (n_shared × 3)
    """
    n_shared  = len(shared_genes)
    cov_np    = np.array(cov_matrix_r,    dtype=np.float64)
    marg_np   = np.array(marginal_params_r, dtype=np.float64)
    gene_idx  = np.array([all_primary_genes.index(g) for g in shared_genes])

    shared_cov = np.zeros((n_shared, n_shared))
    for i, gi in enumerate(gene_idx):
        shared_cov[i] = cov_np[gi][gene_idx]
    shared_marginals = marg_np[gene_idx]   # (n_shared, 3)

    return shared_cov, shared_marginals


def get_shared_genes(primary_s, secondary_s, primary_a, secondary_a):
    """
    Return (covariate_genes, all_genes):
      covariate_genes — primary genes in BOTH copulas (used for Mahalanobis)
      all_genes       — all genes (primary + secondary) in both copulas
    """
    covariate_genes = list(set(primary_s) & set(primary_a))
    all_genes = list(
        (set(primary_s) | set(secondary_s)) & (set(primary_a) | set(secondary_a))
    )
    return covariate_genes, all_genes

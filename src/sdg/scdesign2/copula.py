"""
Helpers for working with scDesign2's Gaussian copula.

Covers three responsibilities:
  1. Parsing: converting rpy2 R objects into numpy structures (parse_copula)
  2. Algebra:  building a shared-gene sub-covariance matrix (build_shared_covariance_matrix)
  3. Config:   creating the YAML config block that drives a scDesign2 training run
               (make_sdg_config)
"""

import os
import numpy as np
from box import Box
from rpy2.rinterface_lib.sexp import NULLType as _R_NULLType


# ---------------------------------------------------------------------------
# 1. Copula parsing
# ---------------------------------------------------------------------------

def parse_copula(r_copula_obj):
    """
    Parse a scDesign2 copula R object (loaded via rpy2) into plain Python
    structures that the attack algorithms can consume.

    Parameters
    ----------
    r_copula_obj : rpy2 ListVector
        The per-cell-type copula object as returned by scDesign2's
        fit_Gaussian_copula(), e.g. after
            r["readRDS"](path).rx2(cell_type)

    Returns
    -------
    dict with keys:
        primary_genes       : list[str]  — "group 1" genes (in the copula)
        secondary_genes     : list[str]  — "group 2" genes (marginal only)
        cov_matrix          : rpy2 vector (raw; indexed as flat array)
        primary_marginals   : rpy2 matrix  — (pi, theta, mu) for primary genes
        secondary_marginals : rpy2 matrix  — (pi, theta, mu) for secondary genes
        get_correlation     : Callable[[str, str], float]
        get_gene_params     : Callable[[str], tuple[float, float, float]]
    """
    primary_genes = list(r_copula_obj.rx2("gene_sel1").names)
    secondary_genes = list(r_copula_obj.rx2("gene_sel2").names)
    len_primary = len(primary_genes)
    len_secondary = len(secondary_genes)
    cov_matrix_r = r_copula_obj.rx2("cov_mat")
    cov_matrix = None if isinstance(cov_matrix_r, _R_NULLType) else cov_matrix_r
    primary_marginals = r_copula_obj.rx2("marginal_param1")
    secondary_marginals = r_copula_obj.rx2("marginal_param2")

    gene_to_idx = {gene: i for i, gene in enumerate(primary_genes)}
    all_gene_location = {}
    for i, gene in enumerate(primary_genes):
        all_gene_location[gene] = (0, i)
    for i, gene in enumerate(secondary_genes):
        all_gene_location[gene] = (1, i)

    def get_correlation(gene1, gene2):
        idx1 = gene_to_idx[gene1]
        idx2 = gene_to_idx[gene2]
        return cov_matrix[idx1 * len_primary + idx2]

    def get_gene_params(gene_name):
        gene_class, idx = all_gene_location[gene_name]
        l = len_primary if gene_class == 0 else len_secondary
        marginals = primary_marginals if gene_class == 0 else secondary_marginals
        dist_pi = marginals[idx]
        dist_theta = marginals[idx + l]
        dist_mu = marginals[idx + 2 * l]
        return dist_pi, dist_theta, dist_mu

    return dict(
        primary_genes=primary_genes,
        secondary_genes=secondary_genes,
        len_primary=len_primary,
        len_secondary=len_secondary,
        cov_matrix=cov_matrix,
        primary_marginals=primary_marginals,
        secondary_marginals=secondary_marginals,
        get_correlation=get_correlation,
        get_gene_params=get_gene_params,
    )


def load_copula(path: str, cell_type: str):
    """
    Read a scDesign2 .rds file and return the parsed copula dict for one
    cell type.  Wraps rpy2 so callers don't need to import it directly.
    """
    from rpy2.robjects import r
    r_obj = r["readRDS"](path).rx2(str(cell_type))
    return parse_copula(r_obj)


# ---------------------------------------------------------------------------
# 2. Shared-gene covariance sub-matrix
# ---------------------------------------------------------------------------

def build_shared_covariance_matrix(shared_genes, all_primary_genes, cov_matrix_r, marginal_params_r):
    """
    Extract the sub-covariance matrix and marginal parameters for the subset
    of genes that appear in *both* the synthetic and auxiliary copulas.

    Parameters
    ----------
    shared_genes        : list[str]  — genes present in both copulas' primary set
    all_primary_genes   : list[str]  — full primary gene list for this copula
    cov_matrix_r        : rpy2 vector  — flat covariance matrix (n_primary^2)
    marginal_params_r   : rpy2 matrix  — marginal params (n_primary x 3)

    Returns
    -------
    shared_cov : np.ndarray  shape (len(shared_genes), len(shared_genes))
    shared_marginals : np.ndarray  shape (len(shared_genes), 3)  columns: (pi, theta, mu)
    """
    n_shared = len(shared_genes)
    cov_np = np.array(cov_matrix_r)
    marginals_np = np.array(marginal_params_r)
    gene_indices = np.array([all_primary_genes.index(g) for g in shared_genes])

    shared_cov = np.zeros((n_shared, n_shared))
    for i, gene_idx in enumerate(gene_indices):
        shared_cov[i] = cov_np[gene_idx][gene_indices]
    shared_marginals = marginals_np[gene_indices]

    return shared_cov, shared_marginals


def get_shared_genes(primary_s, secondary_s, primary_a, secondary_a):
    """
    Return (covariate_genes, all_genes) where:
      - covariate_genes: primary genes present in BOTH copulas (used for Mahalanobis)
      - all_genes:       all genes (primary + secondary) present in both copulas
    """
    covariate_genes = list(set(primary_s).intersection(set(primary_a)))
    all_genes = list(
        (set(primary_s) | set(secondary_s)) & (set(primary_a) | set(secondary_a))
    )
    return covariate_genes, all_genes


# ---------------------------------------------------------------------------
# 3. SDG config builder
# ---------------------------------------------------------------------------

def make_sdg_config(cfg, generate: bool, model_path: str,
                    full_hvg_path: str, train_file_name: str) -> Box:
    """
    Build the scDesign2 YAML config dict for one phase of the pipeline
    (target training, synthetic shadow model, or auxiliary shadow model).

    Parameters
    ----------
    cfg            : Box  — top-level experiment config
    generate       : bool — whether this run should also generate synthetic data
    model_path     : str  — relative path (from trial_dir) where .rds copulas are saved
    full_hvg_path  : str  — absolute path to the HVG mask CSV
    train_file_name: str  — filename of the .h5ad input inside datasets_path
    """
    s2_cfg = Box()

    s2_cfg.dir_list = Box()
    s2_cfg.dir_list.home = cfg.trial_dir
    s2_cfg.dir_list.data = cfg.datasets_path

    s2_cfg.generator_name = "scdesign2"
    s2_cfg.train = True
    s2_cfg.generate = generate
    s2_cfg.load_from_checkpoint = False

    s2_cfg.scdesign2_config = Box()
    s2_cfg.scdesign2_config.out_model_path = model_path
    s2_cfg.scdesign2_config.hvg_path = full_hvg_path

    s2_cfg.dataset_config = Box()
    s2_cfg.dataset_config.name = cfg.dataset_name
    s2_cfg.dataset_config.train_count_file = train_file_name
    s2_cfg.dataset_config.test_count_file = train_file_name
    s2_cfg.dataset_config.cell_type_col_name = "cell_type"
    s2_cfg.dataset_config.cell_label_col_name = "cell_label"
    s2_cfg.dataset_config.random_seed = 42

    return s2_cfg

"""
Helpers for working with scDesign3's Gaussian (and vine) copula.

Provides the same interface as sdg.scdesign2.copula so that existing attack
algorithms can consume scDesign3 Gaussian-copula models without modification.

Key differences from scDesign2
------------------------------
- scDesign3's Gaussian copula is a plain correlation matrix (genes × genes)
  with gene names as row/col names.  There is no "group 1 / group 2" split;
  all genes in the matrix are treated as primary genes.
- Marginal parameters come from gamlss NBI/ZINBI models (mu, sigma, zero_prob)
  and are mapped to scDesign2's (pi, theta, mu) convention:
      pi    = zero_prob
      theta = 1 / sigma      (NBI sigma is reciprocal of NB size)
      mu    = mu
  so the same zinb_cdf() in data.cdf_utils works without changes.

Vine copula support
-------------------
For vine copulas, `load_copula_sd3` returns the same marginal-parameter keys
PLUS a `vine_copula` entry containing the raw rpy2 vinecop object.  The
Mahalanobis attack (cov_matrix) is not applicable to vines; those keys are
None.  A future vine-specific attack will call rvinecopulib::dvinecop() on
each cell using the stored `vine_copula` rpy2 object.

API
---
Both copula types return a dict with:
    copula_type         : str           "gaussian" or "vine"
    primary_genes       : list[str]     genes in the copula
    secondary_genes     : list[str]     always []
    len_primary         : int
    len_secondary       : int           always 0
    primary_marginals   : np.ndarray (n, 3)  columns: (pi, theta, mu)
    secondary_marginals : None
    get_gene_params     : Callable[[str], tuple[float,float,float]]

Gaussian-only keys (None for vine):
    cov_matrix          : np.ndarray (n, n)
    get_correlation     : Callable[[str,str], float]

Vine-only keys (None for gaussian):
    vine_copula         : rpy2 ListVector  (the rvinecopulib vinecop object)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_copula_sd3(path: str, cell_type: str) -> dict:
    """
    Load a scDesign3 per-cell-type model .rds and return a parsed copula dict.

    Dispatches automatically to the Gaussian or vine path based on the
    copula_type field stored in the model.

    Parameters
    ----------
    path      : str  — path to the .rds saved by scdesign3.r train
    cell_type : str  — cell type label used at training time

    Returns
    -------
    dict — see module docstring for full key listing
    """
    from rpy2.robjects import r
    model_r = r["readRDS"](path)
    return _parse_model_r(model_r, cell_type)


def parse_copula_sd3(model_r, cell_type: str) -> dict:
    """
    Parse an already-loaded rpy2 scDesign3 model object.

    Parameters
    ----------
    model_r   : rpy2 ListVector — the object returned by readRDS(path)
    cell_type : str
    """
    return _parse_model_r(model_r, cell_type)


# ---------------------------------------------------------------------------
# Internal dispatch
# ---------------------------------------------------------------------------

def _parse_model_r(model_r, cell_type: str) -> dict:
    copula_type_r = model_r.rx2("copula_type")
    copula_type   = str(copula_type_r[0])

    copula_list_r = model_r.rx2("copula_list")
    copula_obj_r  = copula_list_r.rx2(cell_type)

    marginals = _parse_marginals(model_r, cell_type, copula_obj_r, copula_type)

    if copula_type == "gaussian":
        return _parse_gaussian(copula_obj_r, marginals)
    elif copula_type == "vine":
        return _parse_vine(copula_obj_r, marginals)
    else:
        raise ValueError(f"Unknown scDesign3 copula_type: '{copula_type}'")


# ---------------------------------------------------------------------------
# Marginal parameter extraction (shared by Gaussian and vine paths)
# ---------------------------------------------------------------------------

def _parse_marginals(model_r, cell_type: str, copula_obj_r, copula_type: str):
    """
    Extract per-gene marginal parameters from the saved model and subset
    to only the genes present in the copula object.

    Returns
    -------
    dict with:
        gene_names        : list[str]
        mu_arr            : np.ndarray (n_copula_genes,)
        sigma_arr         : np.ndarray
        zero_arr          : np.ndarray
        theta_arr         : np.ndarray  (= 1 / sigma_arr)
        primary_marginals : np.ndarray (n_copula_genes, 3) — (pi, theta, mu)
        gene_to_idx       : dict[str, int]
    """
    from rpy2.robjects import r as R

    # Gene names in the copula object
    if copula_type == "gaussian":
        # Gaussian: plain R matrix — gene names are colnames
        gene_names = list(R["colnames"](copula_obj_r))
    else:
        # Vine: rvinecopulib vinecop — gene names are in $names
        gene_names = list(copula_obj_r.rx2("names"))

    # Full set of gene names in the marginal parameter matrices
    mean_rep_r  = model_r.rx2("mean_mat_rep")
    sigma_rep_r = model_r.rx2("sigma_mat_rep")
    zero_rep_r  = model_r.rx2("zero_mat_rep")

    all_gene_names       = list(R["colnames"](mean_rep_r))
    all_gene_idx         = {g: i for i, g in enumerate(all_gene_names)}
    corr_gene_positions  = [all_gene_idx[g] for g in gene_names]

    mu_all    = np.array(mean_rep_r).reshape(-1)
    sigma_all = np.array(sigma_rep_r).reshape(-1)
    zero_all  = np.array(zero_rep_r).reshape(-1)

    mu_arr    = mu_all[corr_gene_positions]
    sigma_arr = sigma_all[corr_gene_positions]
    zero_arr  = zero_all[corr_gene_positions]
    theta_arr = np.where(sigma_arr > 0, 1.0 / sigma_arr, np.inf)

    primary_marginals = np.column_stack([zero_arr, theta_arr, mu_arr])
    gene_to_idx       = {g: i for i, g in enumerate(gene_names)}

    return dict(
        gene_names=gene_names,
        mu_arr=mu_arr,
        sigma_arr=sigma_arr,
        zero_arr=zero_arr,
        theta_arr=theta_arr,
        primary_marginals=primary_marginals,
        gene_to_idx=gene_to_idx,
    )


# ---------------------------------------------------------------------------
# Gaussian copula path
# ---------------------------------------------------------------------------

def _parse_gaussian(corr_mat_r, marginals: dict) -> dict:
    gene_names        = marginals["gene_names"]
    gene_to_idx       = marginals["gene_to_idx"]
    zero_arr          = marginals["zero_arr"]
    theta_arr         = marginals["theta_arr"]
    n_genes           = len(gene_names)

    # Correlation matrix → 2-D numpy (n_genes × n_genes)
    corr_np = np.array(corr_mat_r).reshape(n_genes, n_genes)

    def get_correlation(gene1: str, gene2: str) -> float:
        return corr_np[gene_to_idx[gene1], gene_to_idx[gene2]]

    def get_gene_params(gene: str):
        i = gene_to_idx[gene]
        return float(zero_arr[i]), float(theta_arr[i]), float(marginals["mu_arr"][i])

    return dict(
        copula_type="gaussian",
        primary_genes=gene_names,
        secondary_genes=[],
        len_primary=n_genes,
        len_secondary=0,
        cov_matrix=corr_np,
        primary_marginals=marginals["primary_marginals"],
        secondary_marginals=None,
        get_correlation=get_correlation,
        get_gene_params=get_gene_params,
        # vine-specific key absent for Gaussian
        vine_copula=None,
    )


# ---------------------------------------------------------------------------
# Vine copula path
# ---------------------------------------------------------------------------

def _parse_vine(vine_r, marginals: dict) -> dict:
    """
    Parse a vine copula model.

    The raw rvinecopulib vinecop object is preserved under `vine_copula` for
    use in future vine-specific attack algorithms (e.g. per-cell log-likelihood
    via rvinecopulib::dvinecop()).  Summary statistics are also extracted into
    plain Python structures for inspection and logging.

    Vine structure summary (for reference / future attack use)
    ----------------------------------------------------------
    vine_structure : dict with
        order       : list[int]   — variable ordering in the R-vine
        d           : int         — number of variables (= len(primary_genes))
        trunc_lvl   : int         — truncation level of the vine
    pair_copulas_summary : list of lists (one per tree) of dicts, each with
        family      : str         — copula family (e.g. "gaussian", "indep", "t")
        parameters  : list[float] — fitted parameters (e.g. correlation)
    """
    gene_names  = marginals["gene_names"]
    gene_to_idx = marginals["gene_to_idx"]
    zero_arr    = marginals["zero_arr"]
    theta_arr   = marginals["theta_arr"]
    n_genes     = len(gene_names)

    # Extract vine structure summary into plain Python
    structure_r  = vine_r.rx2("structure")
    order        = list(np.array(structure_r.rx2("order")).astype(int))
    trunc_lvl    = int(np.array(structure_r.rx2("trunc_lvl"))[0])

    # Pair copula summaries (family + parameters per edge)
    pair_copulas_r = vine_r.rx2("pair_copulas")
    n_trees = len(pair_copulas_r)
    pair_copulas_summary = []
    for t in range(n_trees):
        tree_r = pair_copulas_r[t]
        tree_summary = []
        for e in range(len(tree_r)):
            pc = tree_r[e]
            family = str(pc.rx2("family")[0])
            params = list(np.array(pc.rx2("parameters")).reshape(-1))
            tree_summary.append({"family": family, "parameters": params})
        pair_copulas_summary.append(tree_summary)

    def get_gene_params(gene: str):
        i = gene_to_idx[gene]
        return float(zero_arr[i]), float(theta_arr[i]), float(marginals["mu_arr"][i])

    return dict(
        copula_type="vine",
        primary_genes=gene_names,
        secondary_genes=[],
        len_primary=n_genes,
        len_secondary=0,
        # Gaussian-specific keys are None for vine
        cov_matrix=None,
        get_correlation=None,
        primary_marginals=marginals["primary_marginals"],
        secondary_marginals=None,
        get_gene_params=get_gene_params,
        # Vine-specific: raw rpy2 object + Python summary
        vine_copula=vine_r,
        vine_structure=dict(order=order, d=n_genes, trunc_lvl=trunc_lvl),
        pair_copulas_summary=pair_copulas_summary,
    )


# ---------------------------------------------------------------------------
# Config builder (mirrors make_sdg_config in scdesign2/copula.py)
# ---------------------------------------------------------------------------

def make_sdg_config_sd3(cfg, generate: bool, model_path: str,
                         full_hvg_path: str, train_file_name: str,
                         copula_type: str = "gaussian",
                         family_use: str = "nb"):
    """
    Build a scDesign3 config dict for one pipeline phase.
    Parameters mirror make_sdg_config() in scdesign2/copula.py.
    """
    from box import Box
    s3_cfg = Box()

    s3_cfg.dir_list = Box()
    s3_cfg.dir_list.home = cfg.trial_dir
    s3_cfg.dir_list.data = cfg.datasets_path

    s3_cfg.generator_name = "scdesign3"
    s3_cfg.train    = True
    s3_cfg.generate = generate
    s3_cfg.load_from_checkpoint = False

    s3_cfg.scdesign3_config = Box()
    s3_cfg.scdesign3_config.out_model_path = model_path
    s3_cfg.scdesign3_config.hvg_path       = full_hvg_path
    s3_cfg.scdesign3_config.copula_type    = copula_type
    s3_cfg.scdesign3_config.family_use     = family_use

    s3_cfg.dataset_config = Box()
    s3_cfg.dataset_config.name                = cfg.dataset_name
    s3_cfg.dataset_config.train_count_file    = train_file_name
    s3_cfg.dataset_config.test_count_file     = train_file_name
    s3_cfg.dataset_config.cell_type_col_name  = "cell_type"
    s3_cfg.dataset_config.cell_label_col_name = "cell_label"
    s3_cfg.dataset_config.random_seed         = 42

    return s3_cfg

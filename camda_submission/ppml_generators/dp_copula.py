"""
Gaussian-mechanism DP noise injection for the scDesign2 Gaussian copula.

Main entry point: apply_gaussian_dp(copula_dict, epsilon, delta, n_cells, k_max, ...)
Returns a new copula dict with a DP-noised covariance matrix.

Post-processing steps (PSD projection + re-normalization to correlation matrix)
do not affect the DP guarantee (post-processing theorem, Dwork & Roth 2014, Prop 2.1).
"""

import os
import sys
import numpy as np

# Ensure this directory is on the path so sensitivity.py is importable
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from sensitivity import gaussian_noise_scale


def apply_gaussian_dp(copula_dict, epsilon, delta, n_cells, k_max,
                      clip_value=3.0, rng=None, dp_variant="v2"):
    """
    Inject (ε,δ)-DP Gaussian noise into a parsed copula dict.

    Parameters
    ----------
    copula_dict : dict from parse_copula() — must have cov_matrix, primary_genes,
                  primary_marginals keys.
    epsilon     : DP privacy budget ε
    delta       : DP failure probability δ
    n_cells     : total cells of this cell type in training set
    k_max       : max cells any one donor contributes to this cell type
    clip_value  : quantile-normal clipping bound c (default 3.0)
    rng         : numpy Generator; created fresh if None
    dp_variant  : "v1" (centered cov) or "v2" (uncentered 2nd moment, 4× less noise)

    Returns
    -------
    New copula dict with noised cov_matrix (numpy array) and updated callables.
    """
    if copula_dict.get("cov_matrix") is None:
        raise ValueError("apply_gaussian_dp: cov_matrix is None (vine copula not supported)")

    if rng is None:
        rng = np.random.default_rng()

    cov_np = _to_numpy(copula_dict["cov_matrix"])
    n_genes = cov_np.shape[0]

    sigma = gaussian_noise_scale(epsilon, delta, n_cells, k_max,
                                 n_genes, clip_value, dp_variant)

    noised = _add_symmetric_gaussian_noise(cov_np, sigma, rng)
    psd    = _project_to_psd(noised)
    corr   = _normalise_to_correlation(psd)

    return _rebuild_copula_dict(copula_dict, corr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(cov_matrix):
    arr = np.array(cov_matrix, dtype=np.float64)
    if arr.ndim == 1:
        side = int(round(arr.size ** 0.5))
        if side * side != arr.size:
            raise ValueError(f"cov_matrix has {arr.size} elements — not a perfect square")
        arr = arr.reshape(side, side)
    return arr


def _add_symmetric_gaussian_noise(cov, sigma, rng):
    G = cov.shape[0]
    E = rng.normal(loc=0.0, scale=sigma, size=(G, G))
    return cov + (E + E.T) / 2.0


def _project_to_psd(cov, floor=1e-8):
    cov_sym = (cov + cov.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(cov_sym)
    return eigvecs @ np.diag(np.maximum(eigvals, floor)) @ eigvecs.T


def _normalise_to_correlation(cov):
    std = np.sqrt(np.diag(cov))
    std = np.where(std > 0, std, 1.0)
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def _rebuild_copula_dict(original, noised_corr):
    out = dict(original)
    out["cov_matrix"] = noised_corr

    gene_names  = original["primary_genes"]
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    def get_correlation(gene1, gene2):
        return float(noised_corr[gene_to_idx[gene1], gene_to_idx[gene2]])

    out["get_correlation"] = get_correlation

    marginals_np = np.array(original["primary_marginals"], dtype=np.float64)
    out["primary_marginals"] = marginals_np

    def get_gene_params(gene):
        i = gene_to_idx[gene]
        row = marginals_np[i]
        return float(row[0]), float(row[1]), float(row[2])

    out["get_gene_params"] = get_gene_params
    return out

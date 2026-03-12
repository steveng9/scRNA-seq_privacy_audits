"""
Gaussian-mechanism DP noise injection for the Gaussian copula.

Works with both scDesign2 and scDesign3 copula dicts (they share the same
schema — see sdg/scdesign2/copula.py and sdg/scdesign3/copula.py).

Core function
-------------
apply_gaussian_dp(copula_dict, epsilon, delta, n_cells, k_max, ...)
    → new copula dict with a DP-noised cov_matrix (numpy array) and
      refreshed get_correlation / get_gene_params callables.

Design notes
------------
DP is applied to the RAW covariance, not the correlation matrix:
  1. Convert the existing cov_matrix to a numpy array if needed.
  2. Add symmetric Gaussian noise N(0, σ²) calibrated to the Frobenius
     sensitivity (see sensitivity.py).
  3. Project to the nearest positive-semidefinite (PSD) matrix by clipping
     negative eigenvalues to a small positive floor.
  4. Re-normalise to a correlation matrix (diagonal → 1).

Steps 3 and 4 are deterministic post-processing of the noised output and
therefore do not affect the DP guarantee (post-processing theorem, Dwork &
Roth 2014, Proposition 2.1).

The returned dict is a plain Python dict with numpy arrays, compatible with
both scDesign2 and scDesign3 attack paths.  The rpy2-backed fields
(get_correlation, get_gene_params) are replaced by pure-Python equivalents
operating on the noised arrays.
"""

import numpy as np

from sdg.dp.sensitivity import gaussian_noise_scale


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_gaussian_dp(
    copula_dict: dict,
    epsilon: float,
    delta: float,
    n_cells: int,
    k_max: int,
    clip_value: float = 3.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Inject (ε, δ)-DP Gaussian noise into a parsed copula dict.

    Parameters
    ----------
    copula_dict : dict
        Output of parse_copula() (scDesign2) or load_copula_sd3() (scDesign3).
        Must have: cov_matrix, primary_genes, primary_marginals.
    epsilon     : float — privacy budget ε > 0
    delta       : float — privacy failure probability δ ∈ (0, 1)
    n_cells     : int   — total cells of this cell type in D_train
    k_max       : int   — max cells any one donor contributes to this cell type
    clip_value  : float — quantile-normal clipping bound c (default 3.0)
    rng         : numpy Generator for reproducibility; created if None

    Returns
    -------
    dict — same schema as copula_dict but with a DP-noised numpy cov_matrix
           and updated get_correlation / get_gene_params callables.

    Raises
    ------
    ValueError if cov_matrix is None (vine copula — DP not supported here).
    """
    if copula_dict.get("cov_matrix") is None:
        raise ValueError(
            "apply_gaussian_dp: cov_matrix is None.  "
            "DP injection is only defined for Gaussian copulas."
        )

    if rng is None:
        rng = np.random.default_rng()

    cov_np = _to_numpy(copula_dict["cov_matrix"])
    n_genes = cov_np.shape[0]

    sigma = gaussian_noise_scale(
        epsilon=epsilon,
        delta=delta,
        n_cells=n_cells,
        k_max=k_max,
        n_genes=n_genes,
        clip_value=clip_value,
    )

    noised_cov = _add_symmetric_gaussian_noise(cov_np, sigma, rng)
    psd_cov    = _project_to_psd(noised_cov)
    corr       = _normalise_to_correlation(psd_cov)

    return _rebuild_copula_dict(copula_dict, corr)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy(cov_matrix) -> np.ndarray:
    """Accept either a numpy array or an rpy2 matrix and return a 2-D float64 array."""
    arr = np.array(cov_matrix, dtype=np.float64)
    if arr.ndim == 1:
        # rpy2 sometimes flattens R matrices; infer square shape
        side = int(round(arr.size ** 0.5))
        if side * side != arr.size:
            raise ValueError(
                f"cov_matrix has {arr.size} elements — expected a perfect square."
            )
        arr = arr.reshape(side, side)
    return arr


def _add_symmetric_gaussian_noise(
    cov: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add symmetric Gaussian noise to a covariance matrix.

    We draw a full G×G noise matrix E ~ N(0, σ²) and symmetrise:
        Ẽ = (E + Eᵀ) / 2
    Symmetrisation is valid post-processing; it halves the effective
    per-entry variance (σ²/2), which is conservative (safe side).
    """
    G = cov.shape[0]
    E = rng.normal(loc=0.0, scale=sigma, size=(G, G))
    E_sym = (E + E.T) / 2.0
    return cov + E_sym


def _project_to_psd(cov: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """
    Project a symmetric matrix to the cone of positive-semidefinite matrices
    by clipping negative eigenvalues to `floor`.

    This is a valid post-processing step (Proposition 2.1, Dwork & Roth 2014)
    and does not affect the DP guarantee of the noised covariance.

    Parameters
    ----------
    cov   : (G, G) symmetric numpy array (may not be PSD after noise addition)
    floor : minimum eigenvalue after projection (small positive to avoid
            singularity in the subsequent Mahalanobis inversion)

    Returns
    -------
    (G, G) PSD numpy array
    """
    # Force exact symmetry first (avoid numerical drift)
    cov_sym = (cov + cov.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(cov_sym)
    eigvals_clipped = np.maximum(eigvals, floor)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T


def _normalise_to_correlation(cov: np.ndarray) -> np.ndarray:
    """
    Normalise a covariance matrix to a correlation matrix (diagonal → 1).

    This is post-processing and does not affect the DP guarantee.
    Off-diagonal entries are clipped to [-1, 1] to avoid floating-point
    artefacts after the PSD projection step.
    """
    std = np.sqrt(np.diag(cov))
    std = np.where(std > 0, std, 1.0)   # guard against zero-variance genes
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)
    return corr


def _rebuild_copula_dict(original: dict, noised_corr: np.ndarray) -> dict:
    """
    Return a new copula dict that replaces cov_matrix with noised_corr and
    rebuilds the get_correlation / get_gene_params callables over plain numpy.

    All other keys are copied from the original dict unchanged.
    """
    out = dict(original)  # shallow copy
    out["cov_matrix"] = noised_corr

    gene_names  = original["primary_genes"]
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Re-build pure-Python get_correlation using the noised numpy array
    def get_correlation(gene1: str, gene2: str) -> float:
        return float(noised_corr[gene_to_idx[gene1], gene_to_idx[gene2]])

    out["get_correlation"] = get_correlation

    # get_gene_params reads only the marginals, which are unchanged by DP.
    # Re-wrap it over numpy so it no longer holds any rpy2 references.
    marginals_np = np.array(original["primary_marginals"], dtype=np.float64)
    out["primary_marginals"] = marginals_np

    def get_gene_params(gene: str):
        i   = gene_to_idx[gene]
        row = marginals_np[i]
        # columns: (pi, theta, mu) for scDesign2 & scDesign3
        return float(row[0]), float(row[1]), float(row[2])

    out["get_gene_params"] = get_gene_params

    return out

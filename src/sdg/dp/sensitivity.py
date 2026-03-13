"""
Frobenius-norm sensitivity of the Gaussian copula covariance matrix
under donor-level differential privacy.

This module is intentionally kept free of any project-specific dependencies
so that the calculation can be read directly alongside the formal proof.

Background
----------
scDesign2 and scDesign3 (Gaussian mode) estimate a per-cell-type correlation
matrix R by:

  1. Transforming each cell's expression x_i ∈ ℝ^G to a normal-quantile vector
         z_i = Φ⁻¹(F̂_j(x_ij))   for each gene j,
     where F̂_j is the fitted ZINB marginal CDF.

  2. Clipping z_i entry-wise to [-c, c]  (clip_value = c).
     Without clipping the sensitivity is unbounded.

  3. Computing the sample (centered) covariance
         Ĉ = (1/(N-1)) · ZᵀZ    (after column-centering Z)
     and normalising to a correlation matrix R = D^{-½} Ĉ D^{-½},
     where D = diag(Ĉ).

Privacy unit: one donor, who may contribute up to k_max cells to a single
cell type.  Removing that donor changes Ĉ by at most Δ_F in Frobenius norm.
Adding Gaussian noise with scale σ ≥ Δ_F · √(2 ln(1.25/δ)) / ε to each
(symmetrised) entry of Ĉ yields (ε, δ)-DP (Gaussian mechanism, Dwork & Roth
2014, Theorem A.1).

Theorem (informal)
------------------
For clip_value c, at most k_max cells per donor, N total cells in the cell-type
training slice, and G genes:

    Δ_F ≤ (4 · k_max · c²· G) / N

The factor 4 accounts for centering: after column-centering, each entry of Z
lies in [-2c, 2c], so the per-entry sensitivity of Ĉ is 4c²k_max / N, and the
Frobenius norm sums G² such entries, giving the bound via the inequality
√(G² · (4c²k_max/N)²) = G · 4c²k_max / N.

Notes
-----
- This bound is conservative.  The true sensitivity for a specific dataset may
  be lower, but the conservative bound is safe and simpler to prove.
- The bound is linear in G.  For typical scRNA-seq HVG sets (G ≈ 300–600) and
  k_max / N ≈ 0.001–0.01, Δ_F is on the order of 10–100, which is large
  relative to the covariance values in [−c², c²].  Consequently even ε = 10
  requires σ ≈ Δ_F · 1.5, which often dwarfs the signal.  This is a feature,
  not a bug: it quantifies how much DP costs for this class of generators.
- For the (ε, δ=0) pure-DP Laplace mechanism the sensitivity in L1 norm is
  needed instead.  See `laplace_noise_scale` below for that variant.
"""

import math


# ---------------------------------------------------------------------------
# Primary bound: Frobenius sensitivity for the Gaussian mechanism
# ---------------------------------------------------------------------------

def frobenius_sensitivity(
    n_cells: int,
    k_max: int,
    n_genes: int,
    clip_value: float,
) -> float:
    """
    Conservative upper bound on the Frobenius-norm sensitivity of the
    sample covariance matrix to the removal of one donor.

    Parameters
    ----------
    n_cells    : int   — total cells of this cell type in the training set (N)
    k_max      : int   — maximum cells any single donor contributes (k_max)
    n_genes    : int   — number of copula genes (G)
    clip_value : float — quantile-normal clipping bound (c); entries ∈ [-c, c]

    Returns
    -------
    float — Δ_F, a conservative upper bound on the Frobenius sensitivity
    """
    if n_cells <= k_max:
        raise ValueError(
            f"n_cells ({n_cells}) must exceed k_max ({k_max}); "
            "cannot remove a donor with more cells than the whole training set."
        )
    # After column-centering each entry of z_i is in [-2c, 2c], so each
    # cross-product z_ij * z_ik ∈ [-4c², 4c²].
    # Removing k_max rows shifts each of the G² covariance entries by at most
    #   4c² * k_max / (N - k_max)   ≲   4c² * k_max / N
    # (the denominator decreases when we remove rows, making this conservative).
    # Frobenius norm: sqrt over G² entries, each bounded by the same amount.
    per_entry = 4.0 * (clip_value ** 2) * k_max / (n_cells - k_max)
    delta_f = per_entry * n_genes  # sqrt(G² · per_entry²) = G · per_entry
    return delta_f


# ---------------------------------------------------------------------------
# Gaussian mechanism: noise scale for (ε, δ)-DP
# ---------------------------------------------------------------------------

def gaussian_noise_scale(
    epsilon: float,
    delta: float,
    n_cells: int,
    k_max: int,
    n_genes: int,
    clip_value: float,
) -> float:
    """
    Standard deviation σ of the Gaussian noise to add to each covariance
    entry so that the overall mechanism is (ε, δ)-DP.

    Uses the classical Gaussian mechanism bound:
        σ ≥ Δ_F · √(2 ln(1.25/δ)) / ε

    (Dwork & Roth 2014, Theorem A.1; also Near & Abuah, Ch. 3)

    Parameters
    ----------
    epsilon, delta : DP parameters (ε > 0, 0 < δ < 1)
    n_cells, k_max, n_genes, clip_value : passed to frobenius_sensitivity

    Returns
    -------
    float — σ (noise standard deviation per covariance entry)
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")

    delta_f = frobenius_sensitivity(n_cells, k_max, n_genes, clip_value)
    sigma = delta_f * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
    return sigma


# ---------------------------------------------------------------------------
# Laplace mechanism: noise scale for ε-DP (pure DP)
# ---------------------------------------------------------------------------

def l1_sensitivity(
    n_cells: int,
    k_max: int,
    n_genes: int,
    clip_value: float,
) -> float:
    """
    Conservative upper bound on the L1 (entry-sum) sensitivity of the
    sample covariance matrix.  Used for the Laplace mechanism (pure DP).

    Each of the G² entries changes by at most 4c²k_max / (N - k_max).
    L1 sensitivity = G² · per_entry.

    Note: G² entries is a loose bound (the matrix is symmetric, so only
    G(G+1)/2 are independent), but the bound is still valid.
    """
    per_entry = 4.0 * (clip_value ** 2) * k_max / (n_cells - k_max)
    return (n_genes ** 2) * per_entry


def laplace_noise_scale(
    epsilon: float,
    n_cells: int,
    k_max: int,
    n_genes: int,
    clip_value: float,
) -> float:
    """
    Scale b of the Laplace noise (each entry drawn iid from Lap(0, b)) so
    that the mechanism is ε-DP (pure differential privacy).

    b = Δ_1 / ε   (Dwork & Roth 2014, Theorem 3.6)
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    delta_1 = l1_sensitivity(n_cells, k_max, n_genes, clip_value)
    return delta_1 / epsilon


# ---------------------------------------------------------------------------
# Convenience: summarise the sensitivity budget for a given configuration
# ---------------------------------------------------------------------------

def sensitivity_report(
    n_cells: int,
    k_max: int,
    n_genes: int,
    clip_value: float,
    epsilons=None,
    delta: float = 1e-5,
) -> dict:
    """
    Return a dict summarising Δ_F, Δ_1, and σ / b for a range of ε values.
    Useful for sanity-checking before a sweep.
    """
    if epsilons is None:
        epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    delta_f = frobenius_sensitivity(n_cells, k_max, n_genes, clip_value)
    delta_1 = l1_sensitivity(n_cells, k_max, n_genes, clip_value)

    report = {
        "n_cells":    n_cells,
        "k_max":      k_max,
        "n_genes":    n_genes,
        "clip_value": clip_value,
        "delta_F":    delta_f,
        "delta_1":    delta_1,
        "gaussian": {
            eps: gaussian_noise_scale(eps, delta, n_cells, k_max, n_genes, clip_value)
            for eps in epsilons
        },
        "laplace": {
            eps: laplace_noise_scale(eps, n_cells, k_max, n_genes, clip_value)
            for eps in epsilons
        },
    }
    return report

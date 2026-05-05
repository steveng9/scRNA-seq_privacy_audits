"""
Frobenius-norm sensitivity of the Gaussian copula covariance matrix
under donor-level differential privacy.

See the full derivation in the paper (Golob et al., RECOMB-Privacy 2026).
Short summary:

scDesign2 estimates a per-cell-type correlation matrix R by transforming
each cell's expression to normal-quantile space (clipped to [-c, c]) and
computing a sample covariance.  Removing one donor (up to k_max cells) shifts
each covariance entry by at most:

    v1 (centered cov):      4 · c² · k_max / (N − k_max)   per entry
    v2 (uncentered 2nd mom): c² · k_max / (N − k_max)       per entry

Frobenius sensitivity = G · (per-entry bound).

Adding Gaussian noise N(0, σ²) per entry with
    σ = Δ_F · √(2 ln(1.25/δ)) / ε
gives (ε, δ)-DP at the donor level (Gaussian mechanism, Dwork & Roth 2014).
"""

import math

_VARIANT_PREFACTOR = {"v1": 4.0, "v2": 1.0}


def frobenius_sensitivity(n_cells, k_max, n_genes, clip_value, dp_variant="v1"):
    if n_cells <= k_max:
        raise ValueError(f"n_cells ({n_cells}) must exceed k_max ({k_max})")
    prefactor = _VARIANT_PREFACTOR[dp_variant]
    per_entry = prefactor * (clip_value ** 2) * k_max / (n_cells - k_max)
    return per_entry * n_genes


def gaussian_noise_scale(epsilon, delta, n_cells, k_max, n_genes, clip_value, dp_variant="v1"):
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")
    delta_f = frobenius_sensitivity(n_cells, k_max, n_genes, clip_value, dp_variant)
    return delta_f * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon

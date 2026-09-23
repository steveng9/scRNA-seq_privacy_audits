"""
Low-rank donor-level DP for the Gaussian copula covariance matrix.

Reduces dimensionality BEFORE noising -- a genuine sensitivity reduction --
rather than noising the full G x G matrix (sdg.dp.dp_copula.apply_gaussian_dp)
and truncating to rank r afterward, which only denoises (Analyze Gauss-style)
and does not lower the noise scale needed for a given epsilon.

Mechanism
---------
1. Fix P in R^{G x r}: a semi-orthogonal (orthonormal columns) matrix, seeded
   PUBLICLY -- deterministically from dataset/cell-type/trial/rank identifiers,
   NEVER from the private expression data.  See make_projection().
2. Project each cell's quantile-normal vector: y_i = P^T z_i  in R^r.
3. Re-clip y_i entrywise to [-clip_value, clip_value]^r.  This step is not
   optional -- see "Why the naive shortcut fails" below.
4. Raw second moment M_r = Y^T Y / (N-1)  (same uncentered-moment convention
   as dp_variant="v2" in sensitivity.py / dp_copula.py).
5. Noise M_r using the EXISTING, already-proven Gaussian mechanism
   (sensitivity.gaussian_noise_scale), called with n_genes=r instead of G.
   This is the actual sensitivity win: Delta_F scales with r, not G.
6. PSD-project the noised r x r matrix (post-processing, free).
7. Reconstruct a full G x G object: Sigma_full = P @ Sigma_r_psd @ P^T +
   RESIDUAL_FLOOR * I_G (post-processing, free -- required because the
   Mahalanobis attack, src/attacks/scmamamia/attack.py, needs a dense
   invertible G x G matrix, but P @ Sigma_r @ P^T alone is exactly rank r).
8. Normalize to a correlation matrix (post-processing, free), matching
   dp_copula._normalise_to_correlation.

Steps 1-5 are the only steps that touch the privacy budget; 6-8 are
post-processing and free by the post-processing theorem (Dwork & Roth,
Prop. 2.1) -- exactly as PSD-projection and correlation-normalization already
are in dp_copula.apply_gaussian_dp.

Why not just noise the true top-r eigenvectors and re-orthonormalize?
-----------------------------------------------------------------------
Rejected. Re-orthonormalizing a noised matrix (via QR or polar decomposition)
IS valid, free post-processing -- exactly like the PSD-projection step above.
The problem is one step earlier: there is no dataset-independent sensitivity
bound to calibrate the PRE-orthonormalization noise to. Eigenvector
perturbation is governed by the Davis-Kahan sin-Theta theorem, whose bound is
||perturbation|| / eigengap, and the eigengap (lambda_r - lambda_{r+1}) can be
arbitrarily small for an otherwise unremarkable covariance matrix -- DP needs
a bound that holds for every neighboring dataset, and no such bound exists
here without an eigengap assumption, which is not something a DP proof is
allowed to assume. Rigorous alternatives that release genuine data-dependent
eigenvectors exist (exponential-mechanism / matrix-Bingham sampling per
Chaudhuri, Sarwate & Sinha 2012; noisy power iteration per Hardt & Roth 2014)
but calibrate noise to the COVARIANCE MATRIX's own sensitivity at each step,
never to the eigenvectors directly -- and are substantially more
implementation risk than the construction here.

Why not just project the ALREADY-FITTED G x G matrix (no per-cell access)?
------------------------------------------------------------------------------
Also rejected -- worth spelling out because it looks like it should work and
would avoid needing per-cell data (i.e. would let this run on ordinary v1/v2
.rds files with no R changes). Let M_hat be the already-fitted G x G raw
second moment and M_r = P^T M_hat P (P orthonormal, r columns). Bounding the
sensitivity of M_r this way requires bounding entries of Delta_r = P^T
Delta_hat P, where Delta_hat is the (entrywise-bounded, per the existing
proof in sensitivity.py) perturbation to M_hat from removing k_max cells.
Two ways to try to bound this fail:
  (a) Operator/Frobenius-norm composition (||P^T Delta_hat P||_2 <=
      ||Delta_hat||_2, via Cauchy interlacing on principal submatrices) is
      exact but ||Delta_hat||_2 itself still scales with G (a single removed
      cell's rank-1 contribution z z^T has ||z||_2^2 <= G*c^2 in the worst
      case), so no reduction is achieved.
  (b) Entrywise composition (|Delta_r[i,j]| <= per_entry * ||P[:,i]||_1 *
      ||P[:,j]||_1) DOES give the desired r-dependent bound, but only if P's
      columns are L1-normalized (<=1). For a DENSE orthonormal P (the kind
      needed to preserve real correlation signal across genes), L2-unit
      columns imply L1 norm ~ sqrt(G) (not <=1) by Cauchy-Schwarz -- so this
      route needs shrinking, signal-destroying column weights (or a sparse
      hashing/bucketing P, which reintroduces the same G-dependence at the
      bucket-size scale once weights are kept large enough to preserve
      signal -- worked through by hand, does not help).
The only construction that is both correct (r-dependent bound) and useful
(preserves real gene-gene structure) is: orthonormal P + explicit per-cell
re-clip after projection, which requires the per-cell quantile-normal matrix
Z, not just the fitted aggregate. This is why this module requires a
copula_dict produced by scdesign2_lowrank.r (which saves Z), not an ordinary
v1/v2 .rds (which discards it after computing cov_mat).

TRUE_CLIP_VALUE
----------------
Neither scdesign2.r nor scdesign2_v2.r/scdesign2_lowrank.r clips z = qnorm(u)
directly to a configurable clip_value. What they actually enforce (DT branch
of fit_marginals) is clamping the quantile u to [epsilon, 1-epsilon] with
epsilon=1e-5, which implicitly bounds |z| <= qnorm(1 - 1e-5) =
4.264890793923841 -- NOT 3.0, which the existing v1/v2 Python pipeline has
been passing as clip_value (see notes/DP_clip_value_bug.txt: this has been
under-calibrating v1/v2's sigma by roughly (4.265/3.0)^2 ~= 2x). This module
defaults clip_value to the true, code-consistent bound.
"""

import numpy as np

from sdg.dp.sensitivity import gaussian_noise_scale, TRUE_CLIP_VALUE
from sdg.dp.dp_copula import (
    _to_numpy,
    _add_symmetric_gaussian_noise,
    _project_to_psd,
    _normalise_to_correlation,
    _rebuild_copula_dict,
)

# Floor added to the reconstructed G x G matrix's diagonal (post-processing,
# free -- P and psd_r are already fully determined by the DP-released output,
# so any fixed, data-independent constant added here costs nothing further).
# P @ Sigma_r @ P^T alone is EXACTLY rank r (not just numerically singular),
# so a tiny numerical-stability epsilon (e.g. dp_copula._project_to_psd's
# 1e-8) is NOT enough here -- confirmed empirically: it leaves the
# reconstructed matrix's smallest eigenvalue at ~1e-8, i.e. barely
# invertible, which destabilizes the attack's matrix inversion.
#
# RESIDUAL_FLOOR = 1.0 instead gives the reconstruction a defensible
# statistical reading: Sigma_full = P @ Sigma_r @ P^T + 1.0 * I_G is exactly
# a factor-analysis model (Sigma = Lambda Lambda^T + Psi, homoscedastic
# Psi = I) -- genes/directions outside the modeled rank-r subspace are
# treated as independent, unit-variance z's, which is consistent with the
# z ~= N(0, 1) marginal assumption this whole v2/lowrank line of proofs
# already relies on (see sensitivity.py's "mu ~= 0" justification). This is
# a modeling choice, not something derived from the DP proof itself -- worth
# revisiting against the spot-check quality numbers.
RESIDUAL_FLOOR = 1.0


def make_projection(n_genes: int, rank: int, seed: int) -> np.ndarray:
    """
    Deterministic, PUBLIC (data-independent) semi-orthogonal projection
    P in R^{n_genes x rank}: QR-decompose a seeded iid-Gaussian matrix and
    keep Q (orthonormal columns, P^T P = I_rank).

    `seed` MUST be derived only from PUBLIC identifiers (dataset name, cell
    type, trial, rank) -- never from the private expression data. If P
    depended on the data, using it would itself cost privacy budget, and this
    whole construction collapses back into the eigenvector-sensitivity
    problem this module exists to avoid (see module docstring).
    """
    if rank > n_genes:
        raise ValueError(f"rank ({rank}) cannot exceed n_genes ({n_genes})")
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n_genes, rank))
    Q, _ = np.linalg.qr(A)
    return Q  # (n_genes, rank), orthonormal columns


def _extract_z(quantile_normal, n_genes: int) -> np.ndarray:
    """
    Normalize the raw 'quantile_normal' object (rpy2 R matrix or numpy array,
    saved by scdesign2_lowrank.r as genes x cells) to a (n_cells, n_genes)
    numpy array, inferring orientation from n_genes (the known cov_matrix
    side length) rather than assuming a fixed axis order.
    """
    Z = np.array(quantile_normal, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError(f"quantile_normal must be 2-D, got shape {Z.shape}")
    if Z.shape[0] == n_genes:
        return Z.T  # (genes, cells) -> (cells, genes)
    if Z.shape[1] == n_genes:
        return Z
    raise ValueError(
        f"quantile_normal shape {Z.shape} incompatible with n_genes={n_genes}"
    )


def apply_gaussian_dp_lowrank(
    copula_dict: dict,
    epsilon: float,
    delta: float,
    n_cells: int,
    k_max: int,
    rank: int,
    clip_value: float = TRUE_CLIP_VALUE,
    rng=None,
    clip: bool = True,
    projection_seed: int = 0,
    residual_floor: float = RESIDUAL_FLOOR,
) -> dict:
    """
    Inject (epsilon, delta)-DP Gaussian noise into a rank-`rank` reduction of
    a parsed copula dict's covariance matrix. See module docstring for the
    full mechanism and why this specific construction (orthonormal public
    projection + per-cell re-clip) is the one that is both correct and
    useful.

    Parameters
    ----------
    copula_dict : dict
        Output of parse_copula() (scdesign2.copula) PLUS a 'quantile_normal'
        key holding the raw per-cell normal-quantile matrix -- only produced
        by scdesign2_lowrank.r's fit_Gaussian_copula, NOT present in ordinary
        v1/v2 .rds files (they discard it after computing cov_mat). Raises
        if missing.
    epsilon, delta : DP parameters (epsilon > 0, 0 < delta < 1)
    n_cells        : total cells of this cell type in D_train (N)
    k_max          : max cells any single donor contributes to this cell type
    rank           : target reduced dimension r (1 <= r <= n_genes)
    clip_value     : quantile-normal clipping bound c applied AFTER
                      projection. Defaults to TRUE_CLIP_VALUE, the bound
                      actually enforced by the R fitting code (see module
                      docstring) -- not 3.0, which v1/v2 have been (wrongly)
                      using.
    rng            : numpy Generator for reproducibility; created if None
    clip           : bool -- whether to PSD-project + clip correlation
                      entries to [-1, 1] (default True); mirrors
                      dp_copula.apply_gaussian_dp's `clip` param, same
                      epsilon-to-infinity sanity-check use.
    projection_seed : seed for make_projection() -- caller must derive this
                      from PUBLIC identifiers only (see make_projection).
    residual_floor : diagonal floor added when reconstructing the full G x G
                      matrix (see RESIDUAL_FLOOR).

    Returns
    -------
    dict -- same schema as dp_copula.apply_gaussian_dp's return value
            (cov_matrix, get_correlation, get_gene_params, unchanged
            primary_marginals) -- drop-in compatible with the existing
            Mahalanobis attack and generation pipeline.
    """
    if copula_dict.get("cov_matrix") is None:
        raise ValueError(
            "apply_gaussian_dp_lowrank: cov_matrix is None. "
            "DP injection is only defined for Gaussian copulas."
        )
    if copula_dict.get("quantile_normal") is None:
        raise ValueError(
            "apply_gaussian_dp_lowrank requires 'quantile_normal' (the raw "
            "per-cell normal-quantile matrix) in copula_dict. Ordinary v1/v2 "
            ".rds files do not have this -- re-fit with scdesign2_lowrank.r "
            "(see src/sdg/scdesign2/scdesign2_lowrank.r)."
        )

    if rng is None:
        rng = np.random.default_rng()

    cov_np = _to_numpy(copula_dict["cov_matrix"])
    n_genes = cov_np.shape[0]

    Z = _extract_z(copula_dict["quantile_normal"], n_genes=n_genes)  # (N, G)

    P = make_projection(n_genes, rank, seed=projection_seed)  # (G, r), public
    Y = Z @ P  # (N, r)
    Y = np.clip(Y, -clip_value, clip_value)  # licenses reusing the existing proof at n_genes=r
    n_for_norm = max(Y.shape[0] - 1, 1)
    M_r = (Y.T @ Y) / n_for_norm  # (r, r), same uncentered-moment convention as v2

    sigma = gaussian_noise_scale(
        epsilon=epsilon,
        delta=delta,
        n_cells=n_cells,
        k_max=k_max,
        n_genes=rank,  # <- the actual sensitivity win vs. dp_copula.apply_gaussian_dp
        clip_value=clip_value,
        dp_variant="v2",
    )

    noised_r = _add_symmetric_gaussian_noise(M_r, sigma, rng)

    if clip:
        psd_r = _project_to_psd(noised_r)
    else:
        psd_r = noised_r

    sigma_full = P @ psd_r @ P.T + residual_floor * np.eye(n_genes)

    corr = _normalise_to_correlation(sigma_full, clip=clip)

    return _rebuild_copula_dict(copula_dict, corr)


def project_lowrank_no_dp(
    copula_dict: dict,
    rank: int,
    clip_value: float = TRUE_CLIP_VALUE,
    projection_seed: int = 0,
    residual_floor: float = RESIDUAL_FLOOR,
) -> dict:
    """
    Rank-r reduction with NO privacy noise added -- isolates the utility cost
    of the projection step alone (steps 1-4, 6-8 of the module docstring's
    mechanism, skipping step 5's noise injection). This is the ablation the
    low-rank idea was originally proposed to answer: how much does rank
    truncation by itself cost, before any DP is layered on top?

    Same interface/output schema as apply_gaussian_dp_lowrank, minus the
    epsilon/delta/n_cells/k_max/rng DP-specific arguments.
    """
    if copula_dict.get("cov_matrix") is None:
        raise ValueError(
            "project_lowrank_no_dp: cov_matrix is None. "
            "Only defined for Gaussian copulas."
        )
    if copula_dict.get("quantile_normal") is None:
        raise ValueError(
            "project_lowrank_no_dp requires 'quantile_normal' in copula_dict "
            "-- re-fit with scdesign2_lowrank.r."
        )

    cov_np = _to_numpy(copula_dict["cov_matrix"])
    n_genes = cov_np.shape[0]

    Z = _extract_z(copula_dict["quantile_normal"], n_genes=n_genes)
    P = make_projection(n_genes, rank, seed=projection_seed)
    Y = Z @ P
    Y = np.clip(Y, -clip_value, clip_value)
    n_for_norm = max(Y.shape[0] - 1, 1)
    M_r = (Y.T @ Y) / n_for_norm

    psd_r = _project_to_psd(M_r)
    sigma_full = P @ psd_r @ P.T + residual_floor * np.eye(n_genes)
    corr = _normalise_to_correlation(sigma_full, clip=True)

    return _rebuild_copula_dict(copula_dict, corr)

"""
scMAMA-MIA attack algorithms.

Two core strategies are implemented:

  attack_pairwise_correlation  — original MAMA-MIA focal-point approach:
      iterates over all gene pairs in the copula and accumulates closeness-to-
      correlation scores as focal points.

  attack_mahalanobis           — improved approach (primary):
      maps each cell to the uniform space via the marginal CDFs, then computes
      the Mahalanobis distance to the copula mean; combines synth and aux
      distances as λ = d_aux / (d_synth + d_aux).

  attack_mahalanobis_no_aux    — same but without auxiliary data;
      returns 1 / (d_synth + ε) as the membership score.
"""

import sys
import os
import numpy as np

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from sdg.scdesign2.copula import parse_copula, build_shared_covariance_matrix, get_shared_genes
from attacks.scmamamia.scoring import compute_cell_scores


# ---------------------------------------------------------------------------
# Helper: map one gene's expression values to Uniform(0,1)
# ---------------------------------------------------------------------------

def _map_gene_to_uniform(cfg, gene, get_gene_params_fn, targets):
    dist_pi, dist_theta, dist_mu = get_gene_params_fn(gene)
    return cfg.uniform_remapping_fn(targets[gene].values, dist_pi, dist_theta, dist_mu)


# ---------------------------------------------------------------------------
# Attack 1: pairwise correlation focal points
# ---------------------------------------------------------------------------

def attack_pairwise_correlation(cfg, targets, cell_type, copula_synth_r, copula_aux_r=None):
    """
    MAMA-MIA original: for each pair of copula genes, compare the observed
    gene-gene correlation in the target cell to the synth and aux copulas.
    Accumulates focal-point sums; higher = more likely member.
    """
    cs = parse_copula(copula_synth_r)
    ca = parse_copula(copula_aux_r)
    covariate_genes, all_genes = get_shared_genes(
        cs["primary_genes"], cs["secondary_genes"],
        ca["primary_genes"], ca["secondary_genes"],
    )

    FP_sums = np.zeros(len(targets))

    for i in range(len(covariate_genes) - 1):
        gene1 = covariate_genes[i]
        vals_s1 = _map_gene_to_uniform(cfg, gene1, cs["get_gene_params"], targets)
        vals_a1 = _map_gene_to_uniform(cfg, gene1, ca["get_gene_params"], targets)

        for j in range(i + 1, len(covariate_genes)):
            gene2 = covariate_genes[j]
            vals_s2 = _map_gene_to_uniform(cfg, gene2, cs["get_gene_params"], targets)
            vals_a2 = _map_gene_to_uniform(cfg, gene2, ca["get_gene_params"], targets)

            corr_s = cs["get_correlation"](gene1, gene2)
            corr_a = ca["get_correlation"](gene1, gene2)
            close_s = cfg.closeness_to_correlation_fn(vals_s1, vals_s2, corr_s)
            close_a = cfg.closeness_to_correlation_fn(vals_a1, vals_a2, corr_a)

            avg_strength = (abs(corr_s) + abs(corr_a)) / 2
            lambda_ = avg_strength * close_s / (close_a + cfg.mamamia_params.epsilon)
            FP_sums += lambda_

    for gene in all_genes:
        vals_s = _map_gene_to_uniform(cfg, gene, cs["get_gene_params"], targets)
        vals_a = _map_gene_to_uniform(cfg, gene, cs["get_gene_params"], targets)
        centrality_s = 1 - 2 * np.abs(vals_s - 0.5)
        centrality_a = 1 - 2 * np.abs(vals_a - 0.5)
        FP_sums += (
            centrality_s / (centrality_a + cfg.mamamia_params.epsilon)
        ) * cfg.mamamia_params.IMPORTANCE_OF_CLASS_B_FPs

    return compute_cell_scores(cfg, cell_type, FP_sums, targets)


# ---------------------------------------------------------------------------
# Attack 2: Mahalanobis distance (primary attack)
# ---------------------------------------------------------------------------

def attack_mahalanobis(cfg, targets, cell_type, copula_synth_r, copula_aux_r=None):
    """
    Maps each cell's expression to Uniform(0,1) via the marginal CDFs, then
    computes the Mahalanobis distance to the copula mean under both the
    synthetic and auxiliary copulas.

    Membership score:  λ = d_aux / (d_synth + d_aux)
    Higher λ → cell is closer to the synthetic copula → more likely a member.

    Vectorized: iterates over G genes (not N cells), calling the CDF once per gene
    with all N cell values — O(G) Python calls instead of O(N).  The quadratic form
    δᵀΣ⁻¹δ is then computed for all cells in one einsum.
    """
    cs = copula_synth_r if isinstance(copula_synth_r, dict) else parse_copula(copula_synth_r)
    ca = copula_aux_r   if isinstance(copula_aux_r,   dict) else parse_copula(copula_aux_r)

    # Vine copulas don't have a covariance matrix — Mahalanobis not applicable.
    if cs.get("copula_type") == "vine" or cs.get("cov_matrix") is None:
        scores = np.full(len(targets), 0.5)
        return compute_cell_scores(cfg, cell_type, scores, targets, None, None)

    covariate_genes, _ = get_shared_genes(
        cs["primary_genes"], cs["secondary_genes"],
        ca["primary_genes"], ca["secondary_genes"],
    )

    shared_cov_s, shared_marginals_s = build_shared_covariance_matrix(
        covariate_genes, cs["primary_genes"], cs["cov_matrix"], cs["primary_marginals"]
    )
    shared_cov_a, shared_marginals_a = build_shared_covariance_matrix(
        covariate_genes, ca["primary_genes"], ca["cov_matrix"], ca["primary_marginals"]
    )

    inv_s = cfg.lin_alg_inverse_fn(shared_cov_s)
    inv_a = cfg.lin_alg_inverse_fn(shared_cov_a)

    X = targets[covariate_genes].values  # (N, G)
    G = len(covariate_genes)
    pi_s, theta_s, mu_s = shared_marginals_s[:, 0], shared_marginals_s[:, 1], shared_marginals_s[:, 2]
    pi_a, theta_a, mu_a = shared_marginals_a[:, 0], shared_marginals_a[:, 1], shared_marginals_a[:, 2]

    # Map all cells to uniform space gene-by-gene; each call is vectorized over N cells.
    mapped_s = np.empty_like(X, dtype=float)
    mapped_a = np.empty_like(X, dtype=float)
    mean_s   = np.empty(G, dtype=float)
    mean_a   = np.empty(G, dtype=float)
    remap = cfg.uniform_remapping_fn
    for j in range(G):
        mapped_s[:, j] = remap(X[:, j], pi_s[j], theta_s[j], mu_s[j])
        mapped_a[:, j] = remap(X[:, j], pi_a[j], theta_a[j], mu_a[j])
        mean_s[j]      = remap(mu_s[j],  pi_s[j], theta_s[j], mu_s[j])
        mean_a[j]      = remap(mu_a[j],  pi_a[j], theta_a[j], mu_a[j])

    delta_s = mapped_s - mean_s   # (N, G)
    delta_a = mapped_a - mean_a

    # Mahalanobis for all N cells in one batched quadratic form
    d_s = np.sqrt(np.einsum("ij,jk,ik->i", delta_s, inv_s, delta_s))  # (N,)
    d_a = np.sqrt(np.einsum("ij,jk,ik->i", delta_a, inv_a, delta_a))

    scores = d_a / (d_s + d_a)
    nan_mask = np.isnan(scores)
    error_count = int(nan_mask.sum())
    if error_count > 0:
        scores = scores.copy()
        scores[nan_mask] = 0.5
        print(f"  [WARN] {error_count} NaN scores for cell type {cell_type} — set to 0.5")

    return compute_cell_scores(cfg, cell_type, scores, targets,
                               d_s.tolist(), d_a.tolist())


# ---------------------------------------------------------------------------
# Attack 3: Mahalanobis without auxiliary data
# ---------------------------------------------------------------------------

def attack_mahalanobis_no_aux(cfg, targets, cell_type, copula_synth_r, copula_aux_r=None):
    """
    Mahalanobis-based attack when no auxiliary data is available.
    Membership score: 1 / (d_synth + ε)   (inverse distance to synth copula)

    Vectorized the same way as attack_mahalanobis.
    """
    cs = copula_synth_r if isinstance(copula_synth_r, dict) else parse_copula(copula_synth_r)

    if cs.get("copula_type") == "vine" or cs.get("cov_matrix") is None:
        scores = np.full(len(targets), 0.5)
        return compute_cell_scores(cfg, cell_type, scores, targets, None, None)

    shared_cov_s, shared_marginals_s = build_shared_covariance_matrix(
        cs["primary_genes"], cs["primary_genes"], cs["cov_matrix"], cs["primary_marginals"]
    )

    inv_s = cfg.lin_alg_inverse_fn(shared_cov_s)

    X = targets[cs["primary_genes"]].values  # (N, G)
    G = len(cs["primary_genes"])
    pi_s, theta_s, mu_s = shared_marginals_s[:, 0], shared_marginals_s[:, 1], shared_marginals_s[:, 2]

    mapped_s = np.empty_like(X, dtype=float)
    mean_s   = np.empty(G, dtype=float)
    remap = cfg.uniform_remapping_fn
    for j in range(G):
        mapped_s[:, j] = remap(X[:, j], pi_s[j], theta_s[j], mu_s[j])
        mean_s[j]      = remap(mu_s[j],  pi_s[j], theta_s[j], mu_s[j])

    delta_s = mapped_s - mean_s  # (N, G)
    d_s = np.sqrt(np.einsum("ij,jk,ik->i", delta_s, inv_s, delta_s))  # (N,)

    scores = 1.0 / (d_s + cfg.mamamia_params.epsilon)
    nan_mask = np.isnan(scores)
    error_count = int(nan_mask.sum())
    if error_count > 0:
        scores = scores.copy()
        scores[nan_mask] = 0.5
        print(f"  [WARN] {error_count} NaN scores for cell type {cell_type} — set to 0.5")

    return compute_cell_scores(cfg, cell_type, scores, targets, d_s.tolist(), None)

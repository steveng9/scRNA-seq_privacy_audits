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
    """
    cs = parse_copula(copula_synth_r)
    ca = parse_copula(copula_aux_r)
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

    remap = np.vectorize(cfg.uniform_remapping_fn)
    inv_s = cfg.lin_alg_inverse_fn(shared_cov_s)
    inv_a = cfg.lin_alg_inverse_fn(shared_cov_a)

    distances_s = []
    distances_a = []
    error_count = 0

    def _score_one_cell(gene_expr):
        nonlocal error_count
        mapped_s = remap(gene_expr, *np.moveaxis(shared_marginals_s, 1, 0))
        mapped_a = remap(gene_expr, *np.moveaxis(shared_marginals_a, 1, 0))
        mean_s = remap(shared_marginals_s[:, 2], *np.moveaxis(shared_marginals_s, 1, 0))
        mean_a = remap(shared_marginals_a[:, 2], *np.moveaxis(shared_marginals_a, 1, 0))
        delta_s = mapped_s - mean_s
        delta_a = mapped_a - mean_a
        d_s = float(np.sqrt(delta_s.T @ inv_s @ delta_s))
        d_a = float(np.sqrt(delta_a.T @ inv_a @ delta_a))
        distances_s.append(d_s)
        distances_a.append(d_a)
        result = d_a / (d_s + d_a)
        if np.isnan(result):
            error_count += 1
            return 0.5
        return result

    scores = targets[covariate_genes].apply(_score_one_cell, axis=1)
    if error_count > 0:
        print(f"  [WARN] {error_count} NaN scores for cell type {cell_type} — set to 0.5")

    return compute_cell_scores(cfg, cell_type, scores, targets, distances_s, distances_a)


# ---------------------------------------------------------------------------
# Attack 3: Mahalanobis without auxiliary data
# ---------------------------------------------------------------------------

def attack_mahalanobis_no_aux(cfg, targets, cell_type, copula_synth_r, copula_aux_r=None):
    """
    Mahalanobis-based attack when no auxiliary data is available.
    Membership score: 1 / (d_synth + ε)   (inverse distance to synth copula)
    """
    cs = parse_copula(copula_synth_r)
    shared_cov_s, shared_marginals_s = build_shared_covariance_matrix(
        cs["primary_genes"], cs["primary_genes"], cs["cov_matrix"], cs["primary_marginals"]
    )

    remap = np.vectorize(cfg.uniform_remapping_fn)
    inv_s = cfg.lin_alg_inverse_fn(shared_cov_s)

    distances_s = []
    error_count = 0

    def _score_one_cell(gene_expr):
        nonlocal error_count
        mapped_s = remap(gene_expr, *np.moveaxis(shared_marginals_s, 1, 0))
        mean_s = remap(shared_marginals_s[:, 2], *np.moveaxis(shared_marginals_s, 1, 0))
        delta_s = mapped_s - mean_s
        d_s = float(np.sqrt(delta_s.T @ inv_s @ delta_s))
        distances_s.append(d_s)
        result = 1.0 / (d_s + cfg.mamamia_params.epsilon)
        if np.isnan(result):
            error_count += 1
            return 0.5
        return result

    scores = targets[cs["primary_genes"]].apply(_score_one_cell, axis=1)
    if error_count > 0:
        print(f"  [WARN] {error_count} NaN scores for cell type {cell_type} — set to 0.5")

    return compute_cell_scores(cfg, cell_type, scores, targets, distances_s, None)

"""
scMAMA-MIA: Black-box Mahalanobis attack with Class B focal points.

Primary function: attack_mahalanobis_classb_bb()

Algorithm (BB+aux):
  1. Map each cell's gene expression to Uniform(0,1) via marginal ZINB CDFs.
  2. Compute Mahalanobis distance to the copula mean under synth (d_s) and aux (d_a).
  3. Primary log-odds: log(d_a) - log(d_s)   [higher = closer to synth = member]
  4. Class B (secondary genes): Σ_g [ log p_synth(x_g) - log p_aux(x_g) ]
     — Neyman-Pearson optimal statistic under gene independence.
  5. Combine: combined_logit = log(d_a/d_s) + γ_eff · LLR_B
     where γ_eff = 1/√(n_secondary_genes)  (auto-normalised).
  6. Score activation: z-score + sigmoid.

For BB-aux (no auxiliary data), falls back to: combined_logit = -log(d_s).
Class B is disabled for BB-aux by default (LLR without a reference adds noise).

Public API
----------
attack_mahalanobis_classb_bb(cfg, targets, cell_type, copula_synth, copula_aux)
    → pd.DataFrame of per-cell membership scores

attack_mahalanobis_classb_bb_noaux(cfg, targets, cell_type, copula_synth)
    → pd.DataFrame of per-cell membership scores (no auxiliary copula)
"""

import numpy as np

from copula import build_shared_covariance_matrix, get_shared_genes
from cdf_utils import zinb_log_pmf, activate_from_logits
from scoring import compute_cell_scores_classb


# ---------------------------------------------------------------------------
# Core Mahalanobis helper
# ---------------------------------------------------------------------------

def _mahalanobis_distances(cfg, targets, cs, ca=None):
    """
    Compute per-cell Mahalanobis distances under the synth copula (and
    optionally the aux copula).

    Returns (covariate_genes, d_s, d_a).
    d_a is None when ca is None.
    Returns (None, None, None) for vine/no-covariance copulas.
    """
    if cs.get("cov_matrix") is None:
        return None, None, None

    if ca is not None:
        covariate_genes, _ = get_shared_genes(
            cs["primary_genes"], cs["secondary_genes"],
            ca["primary_genes"], ca["secondary_genes"],
        )
        shared_cov_a, shared_marg_a = build_shared_covariance_matrix(
            covariate_genes, ca["primary_genes"], ca["cov_matrix"], ca["primary_marginals"]
        )
        inv_a  = cfg["lin_alg_inverse_fn"](shared_cov_a)
        pi_a   = shared_marg_a[:, 0]
        theta_a = shared_marg_a[:, 1]
        mu_a   = shared_marg_a[:, 2]
    else:
        covariate_genes = cs["primary_genes"]

    shared_cov_s, shared_marg_s = build_shared_covariance_matrix(
        covariate_genes, cs["primary_genes"], cs["cov_matrix"], cs["primary_marginals"]
    )
    inv_s  = cfg["lin_alg_inverse_fn"](shared_cov_s)
    pi_s   = shared_marg_s[:, 0]
    theta_s = shared_marg_s[:, 1]
    mu_s   = shared_marg_s[:, 2]

    X = targets[covariate_genes].values   # (N, G)
    G = len(covariate_genes)
    remap = cfg["uniform_remapping_fn"]

    mapped_s = np.empty_like(X, dtype=float)
    mean_s   = np.empty(G, dtype=float)
    for j in range(G):
        mapped_s[:, j] = remap(X[:, j], pi_s[j], theta_s[j], mu_s[j])
        mean_s[j]      = remap(mu_s[j],  pi_s[j], theta_s[j], mu_s[j])

    delta_s = mapped_s - mean_s
    d_s = np.sqrt(np.einsum("ij,jk,ik->i", delta_s, inv_s, delta_s))

    if ca is None:
        return covariate_genes, d_s, None

    mapped_a = np.empty_like(X, dtype=float)
    mean_a   = np.empty(G, dtype=float)
    for j in range(G):
        mapped_a[:, j] = remap(X[:, j], pi_a[j], theta_a[j], mu_a[j])
        mean_a[j]      = remap(mu_a[j],  pi_a[j], theta_a[j], mu_a[j])

    delta_a = mapped_a - mean_a
    d_a = np.sqrt(np.einsum("ij,jk,ik->i", delta_a, inv_a, delta_a))
    return covariate_genes, d_s, d_a


# ---------------------------------------------------------------------------
# Class B LLR helper
# ---------------------------------------------------------------------------

def _class_b_llr(cfg, targets, cs, ca, gene_set="secondary", gamma="auto"):
    """
    Per-cell log-likelihood ratio over secondary genes (Class B evidence).

    Returns (log_evidence, gamma_eff).
    log_evidence[i] > 0 → cell i fits synth better → member evidence.
    """
    covariate_genes, all_genes = get_shared_genes(
        cs["primary_genes"], cs["secondary_genes"],
        ca["primary_genes"], ca["secondary_genes"],
    )
    covariate_set = set(covariate_genes)
    b_genes = [g for g in all_genes if g not in covariate_set] if gene_set == "secondary" \
              else all_genes

    n_b = len(b_genes)
    if n_b == 0:
        return np.zeros(len(targets)), 0.0

    X_b    = targets[b_genes].values
    N      = len(targets)
    log_ps = np.zeros(N)
    log_pa = np.zeros(N)
    for j, gene in enumerate(b_genes):
        pi_s, th_s, mu_s = cs["get_gene_params"](gene)
        pi_a, th_a, mu_a = ca["get_gene_params"](gene)
        x_j = X_b[:, j].astype(int)
        log_ps += zinb_log_pmf(x_j, pi_s, th_s, mu_s)
        log_pa += zinb_log_pmf(x_j, pi_a, th_a, mu_a)

    gamma_eff = 1.0 / np.sqrt(n_b) if gamma == "auto" else float(gamma)
    return log_ps - log_pa, gamma_eff


# ---------------------------------------------------------------------------
# Public attack functions
# ---------------------------------------------------------------------------

def attack_mahalanobis_classb_bb(cfg, targets, cell_type, copula_synth, copula_aux):
    """
    BB+aux Mahalanobis attack with Class B secondary-gene LLR.

    Parameters
    ----------
    cfg          : dict with keys:
                     "uniform_remapping_fn" : callable(x, pi, theta, mu) → uniform values
                     "lin_alg_inverse_fn"   : callable(matrix) → inverse matrix
                     "epsilon"              : small float for numerical stability (e.g. 1e-10)
                     "class_b_gamma"        : "auto" | float | 0 (0 = disable Class B)
                     "class_b_gene_set"     : "secondary" | "all"
    targets      : pd.DataFrame — cells × genes, with 'individual' and 'member' columns
    cell_type    : str
    copula_synth : parsed copula dict (from copula.parse_copula)
    copula_aux   : parsed copula dict
    """
    eps     = cfg.get("epsilon",         1e-10)
    gamma   = cfg.get("class_b_gamma",   "auto")
    gene_set = cfg.get("class_b_gene_set", "secondary")

    cs, ca = copula_synth, copula_aux
    covariate_genes, d_s, d_a = _mahalanobis_distances(cfg, targets, cs, ca)

    if covariate_genes is None:
        # Vine/no-cov copula — Class B only
        if gamma != 0 and gamma != 0.0:
            log_b, g_eff = _class_b_llr(cfg, targets, cs, ca, gene_set, gamma)
            raw = activate_from_logits(np.clip(g_eff * log_b, -500, 500))
        else:
            raw = np.full(len(targets), 0.5)
        return compute_cell_scores_classb(targets, cell_type, raw,
                                           None, None, use_aux=True)

    log_primary = np.log(d_a + eps) - np.log(d_s + eps)

    if gamma == 0 or gamma == 0.0:
        combined = log_primary
    else:
        log_b, g_eff = _class_b_llr(cfg, targets, cs, ca, gene_set, gamma)
        combined = log_primary + g_eff * log_b

    _fix_nonfinite(combined, cell_type, "BB+aux")
    raw = activate_from_logits(combined)
    return compute_cell_scores_classb(targets, cell_type, raw,
                                       d_s.tolist(), d_a.tolist(), use_aux=True)


def attack_mahalanobis_classb_bb_noaux(cfg, targets, cell_type, copula_synth):
    """
    BB-aux Mahalanobis attack (no auxiliary data).

    Class B is disabled by default (cfg["class_b_gamma_noaux"] = 0).
    """
    eps          = cfg.get("epsilon",              1e-10)
    gamma_noaux  = cfg.get("class_b_gamma_noaux",  0.0)
    gene_set     = cfg.get("class_b_gene_set",     "secondary")

    cs = copula_synth
    covariate_genes, d_s, _ = _mahalanobis_distances(cfg, targets, cs, ca=None)

    if covariate_genes is None:
        raw = np.full(len(targets), 0.5)
        return compute_cell_scores_classb(targets, cell_type, raw,
                                           None, None, use_aux=False)

    log_primary = -np.log(d_s + eps)

    if gamma_noaux == 0 or gamma_noaux == 0.0:
        combined = log_primary
    else:
        # No aux: sum log p_synth over secondary genes
        b_genes_all  = cs["primary_genes"] + cs["secondary_genes"]
        covariate_set = set(cs["primary_genes"])
        b_genes = [g for g in b_genes_all if g not in covariate_set] \
                  if gene_set == "secondary" else b_genes_all
        X_b = targets[b_genes].values
        log_ps = np.zeros(len(targets))
        for j, gene in enumerate(b_genes):
            pi, th, mu = cs["get_gene_params"](gene)
            log_ps += zinb_log_pmf(X_b[:, j].astype(int), pi, th, mu)
        g_eff   = 1.0 / np.sqrt(len(b_genes)) if gamma_noaux == "auto" else float(gamma_noaux)
        combined = log_primary + g_eff * log_ps

    _fix_nonfinite(combined, cell_type, "BB-aux")
    raw = activate_from_logits(combined)
    return compute_cell_scores_classb(targets, cell_type, raw,
                                       d_s.tolist(), None, use_aux=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fix_nonfinite(arr, cell_type, label, fill=0.0):
    n = int((~np.isfinite(arr)).sum())
    if n > 0:
        arr[~np.isfinite(arr)] = fill
        print(f"  [WARN] {n} non-finite logits ({label}) for {cell_type} — set to {fill}")

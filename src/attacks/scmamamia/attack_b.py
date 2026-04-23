"""
Mahalanobis attack augmented with Class B focal points.

Class B = per-gene log-likelihood ratio (LLR) over secondary genes — those in the
scDesign2 copula but NOT included in the covariance matrix (group 1 / "secondary"
genes).  These are omitted from the Mahalanobis computation because their
inter-gene correlations are weak, which also makes the LLR assumption of
approximate independence accurate.

Core idea
---------
The existing Mahalanobis attack extracts a primary log-odds score:
    log_A = log(d_aux / d_synth)   — from the Gaussian copula covariance

Class B adds:
    LLR_B = Σ_g [ log p_synth(x_g) - log p_aux(x_g) ]   over secondary genes

Combination (additive in log-odds space, principled under the independence assumption):
    combined_logit = log_A + γ · LLR_B

This is identical in spirit to adding two independent Neyman-Pearson test statistics
— the optimal approach when the gene contributions are uncorrelated.

New public API
--------------
  attack_mahalanobis_b           — BB+aux + Class B
  attack_mahalanobis_b_no_aux    — BB-aux + Class B
  attack_mahalanobis_b_both      — BB+aux AND BB-aux in a single pass

Configuration keys (in mamamia_params):
  class_b_gene_set  : "secondary" (default) | "all"
  class_b_scoring   : "llr" (default) | "diag_mahal"
  class_b_gamma     : float | "auto" (default "auto")

gamma="auto" normalises by 1/sqrt(n_b_genes) so the secondary contribution has
unit scale regardless of how many secondary genes are shared between copulas.
"""

import sys
import os
import numpy as np

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from sdg.scdesign2.copula import parse_copula, build_shared_covariance_matrix, get_shared_genes
from attacks.scmamamia.scoring import compute_cell_scores
from data.cdf_utils import zinb_log_pmf, activate_from_logits


# ---------------------------------------------------------------------------
# Core helper: Class B log-membership evidence per cell
# ---------------------------------------------------------------------------

def _class_b_log_evidence(
    cfg,
    targets,
    cs: dict,
    ca: dict,
    gene_set: str = "secondary",
    scoring: str = "llr",
    gamma = "auto",
    eps: float = 1e-300,
) -> tuple:
    """
    Compute per-cell Class B log-membership evidence.

    Parameters
    ----------
    cfg        : experiment config (provides uniform_remapping_fn, epsilon)
    targets    : pd.DataFrame of cells (rows), genes (columns)
    cs, ca     : parsed copula dicts for synth and aux
    gene_set   : "secondary" — only non-copula genes (recommended for SD2)
                 "all"       — all genes in both copulas (primary + secondary)
    scoring    : "llr"        — sum of log(p_synth/p_aux) per gene (optimal statistic)
                 "diag_mahal" — diagonal Mahalanobis in uniform space, converted to log-ratio
    gamma      : float — weight; "auto" normalises by 1/sqrt(n_b_genes)

    Returns
    -------
    (log_evidence, gamma_eff)
    log_evidence[i] > 0  →  cell i fits synth distribution better  →  member evidence
    """
    covariate_genes, all_genes = get_shared_genes(
        cs["primary_genes"], cs["secondary_genes"],
        ca["primary_genes"], ca["secondary_genes"],
    )

    if gene_set == "secondary":
        covariate_set = set(covariate_genes)
        b_genes = [g for g in all_genes if g not in covariate_set]
    else:  # "all"
        b_genes = all_genes

    n_b = len(b_genes)
    if n_b == 0:
        return np.zeros(len(targets)), 0.0

    N = len(targets)
    X_b = targets[b_genes].values  # (N, n_b)

    if scoring == "llr":
        log_p_s = np.zeros(N)
        log_p_a = np.zeros(N)
        for j, gene in enumerate(b_genes):
            pi_s, theta_s, mu_s = cs["get_gene_params"](gene)
            pi_a, theta_a, mu_a = ca["get_gene_params"](gene)
            x_j = X_b[:, j].astype(int)
            log_p_s += zinb_log_pmf(x_j, pi_s, theta_s, mu_s)
            log_p_a += zinb_log_pmf(x_j, pi_a, theta_a, mu_a)
        log_evidence = log_p_s - log_p_a   # LLR: > 0 → member evidence

    elif scoring == "diag_mahal":
        remap = cfg.uniform_remapping_fn
        d_sq_s = np.zeros(N)
        d_sq_a = np.zeros(N)
        for j, gene in enumerate(b_genes):
            pi_s, theta_s, mu_s = cs["get_gene_params"](gene)
            pi_a, theta_a, mu_a = ca["get_gene_params"](gene)
            x_j = X_b[:, j]
            u_s = remap(x_j, pi_s, theta_s, mu_s) - 0.5
            u_a = remap(x_j, pi_a, theta_a, mu_a) - 0.5
            d_sq_s += u_s ** 2
            d_sq_a += u_a ** 2
        # Normalise: Var(Uniform[0,1]) = 1/12  →  scale to proper distances
        d_b_s = np.sqrt(d_sq_s * 12.0)
        d_b_a = np.sqrt(d_sq_a * 12.0)
        # Higher d_a, lower d_s → closer to synth → member
        log_evidence = np.log(d_b_a + eps) - np.log(d_b_s + eps)

    else:
        raise ValueError(f"Unknown Class B scoring method: {scoring!r}")

    gamma_eff = 1.0 / np.sqrt(n_b) if gamma == "auto" else float(gamma)
    return log_evidence, gamma_eff


def _class_b_log_evidence_noaux(
    targets,
    cs: dict,
    gene_set: str = "secondary",
    gamma = "auto",
    eps: float = 1e-300,
) -> tuple:
    """
    Class B evidence without auxiliary data: sum of log p_synth per secondary gene.

    For no-aux, we cannot form a ratio; instead we compute the log-likelihood
    under the synth marginals alone.  activate_from_logits z-scores across cells,
    so the absolute offset cancels — only the relative fit between cells matters.
    Members systematically score higher (their expression was drawn from p_synth).
    """
    covariate_set = set(cs["primary_genes"])  # no aux → use all synth primary genes
    all_synth_genes = cs["primary_genes"] + cs["secondary_genes"]

    if gene_set == "secondary":
        b_genes = [g for g in all_synth_genes if g not in covariate_set]
    else:
        b_genes = all_synth_genes

    n_b = len(b_genes)
    if n_b == 0:
        return np.zeros(len(targets)), 0.0

    N = len(targets)
    X_b = targets[b_genes].values

    log_p_s = np.zeros(N)
    for j, gene in enumerate(b_genes):
        pi_s, theta_s, mu_s = cs["get_gene_params"](gene)
        x_j = X_b[:, j].astype(int)
        log_p_s += zinb_log_pmf(x_j, pi_s, theta_s, mu_s)

    gamma_eff = 1.0 / np.sqrt(n_b) if gamma == "auto" else float(gamma)
    return log_p_s, gamma_eff


# ---------------------------------------------------------------------------
# Primary Mahalanobis helper (shared between aux and no-aux variants)
# ---------------------------------------------------------------------------

def _mahalanobis_distances(cfg, targets, cs, ca=None):
    """
    Compute per-cell Mahalanobis distances under synth (and optionally aux) copulas.

    Returns
    -------
    (covariate_genes, d_s, d_a)
    d_a is None when ca is None (no-aux mode).
    Returns (None, None, None) if the copula has no covariance matrix.
    """
    if cs.get("copula_type") == "vine" or cs.get("cov_matrix") is None:
        return None, None, None

    if ca is not None:
        covariate_genes, _ = get_shared_genes(
            cs["primary_genes"], cs["secondary_genes"],
            ca["primary_genes"], ca["secondary_genes"],
        )
        shared_cov_a, shared_marginals_a = build_shared_covariance_matrix(
            covariate_genes, ca["primary_genes"], ca["cov_matrix"], ca["primary_marginals"]
        )
        inv_a = cfg.lin_alg_inverse_fn(shared_cov_a)
        pi_a = shared_marginals_a[:, 0]
        theta_a = shared_marginals_a[:, 1]
        mu_a = shared_marginals_a[:, 2]
    else:
        covariate_genes = cs["primary_genes"]

    shared_cov_s, shared_marginals_s = build_shared_covariance_matrix(
        covariate_genes, cs["primary_genes"], cs["cov_matrix"], cs["primary_marginals"]
    )
    inv_s = cfg.lin_alg_inverse_fn(shared_cov_s)

    X = targets[covariate_genes].values   # (N, G)
    G = len(covariate_genes)
    pi_s   = shared_marginals_s[:, 0]
    theta_s = shared_marginals_s[:, 1]
    mu_s   = shared_marginals_s[:, 2]

    remap = cfg.uniform_remapping_fn
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
# Convenience: read Class B params from cfg (with defaults)
# ---------------------------------------------------------------------------

def _b_params(cfg):
    p = cfg.mamamia_params
    gene_set = p.get("class_b_gene_set",       "secondary")
    scoring  = p.get("class_b_scoring",        "llr")
    gamma    = p.get("class_b_gamma",          "auto")
    # Separate no-aux gamma: defaults to 0 (disabled) because log p_s alone without
    # an aux reference adds noise rather than discriminative signal for BB-aux.
    # Set class_b_gamma_noaux explicitly in config to enable Class B for BB-aux.
    gamma_noaux = p.get("class_b_gamma_noaux", 0.0)
    if gamma == 0 or gamma == 0.0:
        gamma = 0.0
    return gene_set, scoring, gamma, gamma_noaux


# ---------------------------------------------------------------------------
# Public attack functions
# ---------------------------------------------------------------------------

def attack_mahalanobis_b(cfg, targets, cell_type, copula_synth_r, copula_aux_r=None):
    """
    BB+aux Mahalanobis attack augmented with Class B secondary gene LLR.

    Reads class_b_gene_set, class_b_scoring, class_b_gamma from cfg.mamamia_params.
    Falls back to pure Class B evidence for vine/no-covariance copulas.
    """
    gene_set, scoring, gamma, _ = _b_params(cfg)
    eps = cfg.mamamia_params.epsilon

    cs = copula_synth_r if isinstance(copula_synth_r, dict) else parse_copula(copula_synth_r)
    ca = copula_aux_r   if isinstance(copula_aux_r,   dict) else parse_copula(copula_aux_r)

    covariate_genes, d_s, d_a = _mahalanobis_distances(cfg, targets, cs, ca)

    if covariate_genes is None:
        # Vine copula: primary Mahalanobis unavailable — use Class B alone
        log_b, gamma_eff = _class_b_log_evidence(cfg, targets, cs, ca, gene_set, scoring, gamma)
        if gamma_eff == 0.0:
            raw = np.full(len(targets), 0.5)
        else:
            raw = np.exp(np.clip(gamma_eff * log_b, -500, 500))
        return compute_cell_scores(cfg, cell_type, raw, targets, None, None)

    log_primary = np.log(d_a + eps) - np.log(d_s + eps)

    if gamma == 0.0:
        combined_logit = log_primary
    else:
        log_b, gamma_eff = _class_b_log_evidence(
            cfg, targets, cs, ca, gene_set, scoring, gamma
        )
        combined_logit = log_primary + gamma_eff * log_b

    _fix_nonfinite(combined_logit, cell_type, "BB+aux")
    raw = activate_from_logits(combined_logit)
    return compute_cell_scores(cfg, cell_type, raw, targets, d_s.tolist(), d_a.tolist())


def attack_mahalanobis_b_no_aux(cfg, targets, cell_type, copula_synth_r, copula_aux_r=None):
    """
    BB-aux Mahalanobis attack augmented with Class B secondary gene log-probabilities.

    No auxiliary copula.  Primary: -log(d_synth).  Secondary: Σ log p_synth(x_g).
    Controlled by class_b_gamma_noaux (default 0 = disabled), separate from the
    aux variant, since log p_s alone (without an aux ratio) is a weaker signal.
    """
    gene_set, scoring, _, gamma = _b_params(cfg)
    eps = cfg.mamamia_params.epsilon

    cs = copula_synth_r if isinstance(copula_synth_r, dict) else parse_copula(copula_synth_r)

    covariate_genes, d_s, _ = _mahalanobis_distances(cfg, targets, cs, ca=None)

    if covariate_genes is None:
        log_b, gamma_eff = _class_b_log_evidence_noaux(targets, cs, gene_set, gamma)
        if gamma_eff == 0.0:
            raw = np.full(len(targets), 0.5)
        else:
            raw = np.exp(np.clip(gamma_eff * log_b, -500, 500))
        return compute_cell_scores(cfg, cell_type, raw, targets, None, None)

    log_primary = -np.log(d_s + eps)   # higher → closer to synth → member

    if gamma == 0.0:
        combined_logit = log_primary
    else:
        log_b, gamma_eff = _class_b_log_evidence_noaux(targets, cs, gene_set, gamma)
        combined_logit = log_primary + gamma_eff * log_b

    _fix_nonfinite(combined_logit, cell_type, "BB-aux")
    raw = activate_from_logits(combined_logit)
    return compute_cell_scores(cfg, cell_type, raw, targets, d_s.tolist(), None)


def attack_mahalanobis_quad(cfg, targets, cell_type, copula_synth_r, copula_aux_r,
                             tm_aux="100", tm_noaux="101"):
    """
    Compute all 4 variants in one pass:
      r_std_aux,   r_std_noaux — standard Mahalanobis +aux / -aux (no Class B)
      r_b_aux,     r_b_noaux   — Class B augmented +aux / -aux

    tm_aux / tm_noaux control the TM code used in result column names.
    Defaults ("100"/"101") are BB; pass ("000"/"001") for white-box runs.

    Mahalanobis distances and Class B LLR vectors are each computed once and
    reused across all four outputs.  Returns (r_std_aux, r_std_noaux, r_b_aux, r_b_noaux).
    """
    gene_set, scoring, gamma, gamma_noaux = _b_params(cfg)
    eps = cfg.mamamia_params.epsilon

    cs = copula_synth_r if isinstance(copula_synth_r, dict) else parse_copula(copula_synth_r)
    ca = copula_aux_r   if isinstance(copula_aux_r,   dict) else parse_copula(copula_aux_r)

    half = np.full(len(targets), 0.5)

    if cs.get("copula_type") == "vine" or cs.get("cov_matrix") is None:
        r_std_aux   = compute_cell_scores(cfg, cell_type, half.copy(), targets, None, None, tm=tm_aux)
        r_std_noaux = compute_cell_scores(cfg, cell_type, half.copy(), targets, None, None, tm=tm_noaux)

        if gamma != 0.0:
            log_b, gamma_eff = _class_b_log_evidence(cfg, targets, cs, ca, gene_set, scoring, gamma)
            r_b_aux = compute_cell_scores(cfg, cell_type,
                                          activate_from_logits(np.clip(gamma_eff * log_b, -500, 500)),
                                          targets, None, None, tm=tm_aux)
        else:
            r_b_aux = compute_cell_scores(cfg, cell_type, half.copy(), targets, None, None, tm=tm_aux)

        if gamma_noaux != 0.0:
            log_b_na, g_na = _class_b_log_evidence_noaux(targets, cs, gene_set, gamma_noaux)
            r_b_noaux = compute_cell_scores(cfg, cell_type,
                                            activate_from_logits(np.clip(g_na * log_b_na, -500, 500)),
                                            targets, None, None, tm=tm_noaux)
        else:
            r_b_noaux = compute_cell_scores(cfg, cell_type, half.copy(), targets, None, None, tm=tm_noaux)

        return r_std_aux, r_std_noaux, r_b_aux, r_b_noaux

    covariate_genes, d_s, d_a = _mahalanobis_distances(cfg, targets, cs, ca)

    log_primary_aux   = np.log(d_a + eps) - np.log(d_s + eps)
    log_primary_noaux = -np.log(d_s + eps)

    # Standard variants: pure Mahalanobis, no Class B
    logit_std_aux   = log_primary_aux.copy()
    logit_std_noaux = log_primary_noaux.copy()
    _fix_nonfinite(logit_std_aux,   cell_type, f"{tm_aux} std")
    _fix_nonfinite(logit_std_noaux, cell_type, f"{tm_noaux} std")
    r_std_aux   = compute_cell_scores(cfg, cell_type, activate_from_logits(logit_std_aux),
                                      targets, d_s.tolist(), d_a.tolist(), tm=tm_aux)
    r_std_noaux = compute_cell_scores(cfg, cell_type, activate_from_logits(logit_std_noaux),
                                      targets, d_s.tolist(), None, tm=tm_noaux)

    # Class B variants: Mahalanobis + LLR secondary genes
    logit_b_aux   = log_primary_aux.copy()
    logit_b_noaux = log_primary_noaux.copy()

    if gamma != 0.0:
        log_b_aux, gamma_eff = _class_b_log_evidence(
            cfg, targets, cs, ca, gene_set, scoring, gamma
        )
        logit_b_aux = log_primary_aux + gamma_eff * log_b_aux

    if gamma_noaux != 0.0:
        log_b_noaux_vals, gamma_noaux_eff = _class_b_log_evidence_noaux(
            targets, cs, gene_set, gamma_noaux
        )
        logit_b_noaux = log_primary_noaux + gamma_noaux_eff * log_b_noaux_vals

    _fix_nonfinite(logit_b_aux,   cell_type, f"{tm_aux} ClassB")
    _fix_nonfinite(logit_b_noaux, cell_type, f"{tm_noaux} ClassB")
    r_b_aux   = compute_cell_scores(cfg, cell_type, activate_from_logits(logit_b_aux),
                                    targets, d_s.tolist(), d_a.tolist(), tm=tm_aux)
    r_b_noaux = compute_cell_scores(cfg, cell_type, activate_from_logits(logit_b_noaux),
                                    targets, d_s.tolist(), None, tm=tm_noaux)

    return r_std_aux, r_std_noaux, r_b_aux, r_b_noaux


def attack_mahalanobis_b_both(cfg, targets, cell_type, copula_synth_r, copula_aux_r):
    """
    Compute BB+aux (tm:100) and BB-aux (tm:101) Class B attacks in a single pass,
    sharing the synth-side Mahalanobis computation and the Class B LLR vectors.

    Uses class_b_gamma for BB+aux and class_b_gamma_noaux for BB-aux (default 0).
    Returns (result_df_100, result_df_101).
    """
    gene_set, scoring, gamma, gamma_noaux = _b_params(cfg)
    eps = cfg.mamamia_params.epsilon

    cs = copula_synth_r if isinstance(copula_synth_r, dict) else parse_copula(copula_synth_r)
    ca = copula_aux_r   if isinstance(copula_aux_r,   dict) else parse_copula(copula_aux_r)

    half = np.full(len(targets), 0.5)

    if cs.get("copula_type") == "vine" or cs.get("cov_matrix") is None:
        # Vine fallback: Class B only (no covariance matrix available)
        if gamma != 0.0:
            log_b, gamma_eff = _class_b_log_evidence(cfg, targets, cs, ca, gene_set, scoring, gamma)
            r100 = compute_cell_scores(cfg, cell_type,
                                       activate_from_logits(np.clip(gamma_eff * log_b, -500, 500)),
                                       targets, None, None, tm="100")
        else:
            r100 = compute_cell_scores(cfg, cell_type, half.copy(), targets, None, None, tm="100")

        if gamma_noaux != 0.0:
            log_b_noaux, g_noaux = _class_b_log_evidence_noaux(targets, cs, gene_set, gamma_noaux)
            r101 = compute_cell_scores(cfg, cell_type,
                                       activate_from_logits(np.clip(g_noaux * log_b_noaux, -500, 500)),
                                       targets, None, None, tm="101")
        else:
            r101 = compute_cell_scores(cfg, cell_type, half.copy(), targets, None, None, tm="101")
        return r100, r101

    covariate_genes, d_s, d_a = _mahalanobis_distances(cfg, targets, cs, ca)

    log_primary_aux   = np.log(d_a + eps) - np.log(d_s + eps)
    log_primary_noaux = -np.log(d_s + eps)

    logit_aux   = log_primary_aux
    logit_noaux = log_primary_noaux

    if gamma != 0.0:
        log_b_aux, gamma_eff = _class_b_log_evidence(
            cfg, targets, cs, ca, gene_set, scoring, gamma
        )
        logit_aux = log_primary_aux + gamma_eff * log_b_aux

    if gamma_noaux != 0.0:
        log_b_noaux_vals, gamma_noaux_eff = _class_b_log_evidence_noaux(
            targets, cs, gene_set, gamma_noaux
        )
        logit_noaux = log_primary_noaux + gamma_noaux_eff * log_b_noaux_vals

    _fix_nonfinite(logit_aux,   cell_type, "BB+aux")
    _fix_nonfinite(logit_noaux, cell_type, "BB-aux")

    scores_aux   = activate_from_logits(logit_aux)
    scores_noaux = activate_from_logits(logit_noaux)

    r100 = compute_cell_scores(cfg, cell_type, scores_aux,   targets, d_s.tolist(), d_a.tolist(), tm="100")
    r101 = compute_cell_scores(cfg, cell_type, scores_noaux, targets, d_s.tolist(), None,          tm="101")
    return r100, r101


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fix_nonfinite(arr, cell_type, label, fill=0.0):
    mask = ~np.isfinite(arr)
    n = int(mask.sum())
    if n > 0:
        arr[mask] = fill
        print(f"  [WARN] {n} non-finite logits ({label}) for {cell_type} — set to {fill}")

"""
Utility functions for mapping scRNA-seq count data into the [0, 1] uniform space
required by the Gaussian copula, and for computing donor membership scores.
"""

import numpy as np
from scipy.stats import nbinom, poisson
from scipy import stats


# ---------------------------------------------------------------------------
# Uniform remapping: ZINB / Poisson CDF transforms
# ---------------------------------------------------------------------------

def zinb_cdf(x, pi, theta, mu):
    """CDF of Zero-Inflated Negative Binomial at value x.  Returns P(X <= x)."""
    x = np.asarray(x)
    if np.isinf(theta):  # Poisson case
        F = poisson.cdf(x, mu)
    else:
        n = theta
        p = theta / (theta + mu)
        F = nbinom.cdf(x, n, p)
    return pi + (1 - pi) * F


def zinb_cdf_DT(x, pi, theta, mu, jitter=True):
    """Distributional-transform (DT) variant of the ZINB CDF."""
    x = np.asarray(x)
    if np.isinf(theta):
        F_x = poisson.cdf(x, mu)
        F_xm1 = poisson.cdf(x - 1, mu)
    else:
        n = theta
        p = theta / (theta + mu)
        F_x = nbinom.cdf(x, n, p)
        F_xm1 = nbinom.cdf(x - 1, n, p)
    F_x = pi + (1 - pi) * F_x
    F_xm1 = pi + (1 - pi) * F_xm1 * (x > 0)

    v = np.random.rand(*x.shape) if jitter else 0.5
    return F_xm1 + v * (F_x - F_xm1)


def zinb_uniform_transform(x, pi, theta, mu, jitter=True):
    """Properly maps ZINB-distributed counts to Uniform(0,1) via the
    distributional transform, removing zero-inflation bias."""
    x = np.asarray(x, dtype=int)

    if np.isinf(theta):
        F_nb = poisson.cdf(x, mu)
        F_nb_prev = poisson.cdf(x - 1, mu)
        p0_nb = poisson.pmf(0, mu)
    else:
        n = theta
        p = theta / (theta + mu)
        F_nb = nbinom.cdf(x, n, p)
        F_nb_prev = nbinom.cdf(x - 1, n, p)
        p0_nb = nbinom.pmf(0, n, p)

    p_zero_total = pi + (1 - pi) * p0_nb
    v = np.random.rand(*x.shape) if jitter else 0.5
    u = np.empty_like(x, dtype=float)

    mask_nonzero = (x != 0)
    if np.any(mask_nonzero):
        F_x = F_nb[mask_nonzero]
        F_xm1 = F_nb_prev[mask_nonzero]
        F_cond_x = (F_x - p0_nb) / (1 - p0_nb)
        F_cond_xm1 = (F_xm1 - p0_nb) / (1 - p0_nb)
        u[mask_nonzero] = p_zero_total + (1 - p_zero_total) * (
            F_cond_xm1 + v[mask_nonzero] * (F_cond_x - F_cond_xm1)
        )

    return u


# ---------------------------------------------------------------------------
# Correlation closeness metrics (for pairwise-correlation attack variant)
# ---------------------------------------------------------------------------

def closeness_to_correlation_1(vals1, vals2, correlation):
    """Simple absolute difference, scaled by distance to origin."""
    return 1 - (np.abs(vals1 - vals2) / np.maximum(vals1, vals2))


def closeness_to_correlation_2(vals1, vals2, correlation, epsilon=1e-9):
    ratios = vals2 / (vals1 + epsilon)
    ratios = np.where(ratios > 1, (1 / ratios), ratios)
    return ratios


def closeness_to_correlation_3(vals1, vals2, correlation):
    """Like #1, but flips to y = -x + 1 line when correlation is negative."""
    if correlation >= 0:
        return 1 - (np.abs(vals1 - vals2) / np.maximum(vals1, vals2))
    else:
        return 1 - (np.abs(vals1 - (1 - vals2)) / np.maximum(vals1, (1 - vals2)))


def closeness_to_correlation_4(vals1, vals2, correlation):
    """Evaluates the point against line y = c*x + 0.5 - c/2, which passes
    through (0.5, 0.5) and represents the expected correlation."""
    expected_vals2 = correlation * vals1 + 0.5 - correlation / 2
    return 1 - np.abs(expected_vals2 - vals2)


# ---------------------------------------------------------------------------
# PMF: log-probability of ZINB counts (for LLR-based Class B attacks)
# ---------------------------------------------------------------------------

def zinb_log_pmf(x, pi, theta, mu):
    """Log-PMF of Zero-Inflated Negative Binomial at integer counts x (vectorized).

    ZINB(pi, theta, mu):
        P(X=0) = pi + (1-pi) * NB(0)
        P(X=k) = (1-pi) * NB(k)   for k > 0

    Handles the Poisson limit when theta == inf.
    """
    x = np.asarray(x, dtype=int)
    if np.isinf(theta):
        log_nb = poisson.logpmf(x, mu)
    else:
        p_nb = theta / (theta + mu)
        log_nb = nbinom.logpmf(x, theta, p_nb)

    # For zero counts: log( pi + (1-pi)*exp(log_nb_pmf(0)) )
    # For nonzero:     log(1-pi) + log_nb_pmf(k)
    log_pmf = np.where(
        x == 0,
        np.log(np.clip(pi + (1.0 - pi) * np.exp(log_nb), 1e-300, None)),
        np.log(np.clip(1.0 - pi, 1e-300, None)) + log_nb,
    )
    return log_pmf


def activate_from_logits(logits, confidence=1, center=True):
    """Sigmoid activation for pre-computed log-odds scores (skips the log step in activate()).

    Equivalent to activate(np.exp(logits)) but avoids the overflow-prone exp/log roundtrip.
    """
    logits = np.asarray(logits, dtype=float)
    zscores = stats.zscore(logits)
    median = np.median(zscores) if center else 0.0
    return 1.0 / (1.0 + np.exp(-confidence * (zscores - median)))


# ---------------------------------------------------------------------------
# Score activation: maps raw focal-point sums to [0, 1] membership scores
# ---------------------------------------------------------------------------

def activate(p_rel, confidence=1, center=True) -> np.ndarray:
    """Convert raw log-ratio scores to sigmoid-activated membership probabilities."""
    logs = np.log(p_rel)
    zscores = stats.zscore(logs)
    median = np.median(zscores) if center else 0
    probabilities = 1 / (1 + np.exp(-1 * confidence * (zscores - median)))
    return probabilities

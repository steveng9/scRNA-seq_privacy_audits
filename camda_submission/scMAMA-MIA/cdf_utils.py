"""
CDF and scoring utilities for the scMAMA-MIA attack.

Covers:
  - ZINB distributional-transform CDF (maps count data → Uniform(0,1))
  - ZINB log-PMF (for Class B LLR statistic)
  - Score activation (sigmoid over z-scored log-odds)
"""

import numpy as np
from scipy.stats import nbinom, poisson
from scipy import stats


def zinb_cdf(x, pi, theta, mu):
    """CDF of Zero-Inflated Negative Binomial: P(X ≤ x)."""
    x = np.asarray(x)
    if np.isinf(theta):
        F = poisson.cdf(x, mu)
    else:
        F = nbinom.cdf(x, theta, theta / (theta + mu))
    return pi + (1 - pi) * F


def zinb_uniform_transform(x, pi, theta, mu, jitter=True):
    """Map ZINB-distributed counts to Uniform(0,1) via the distributional transform."""
    x = np.asarray(x, dtype=int)

    if np.isinf(theta):
        F_nb      = poisson.cdf(x,     mu)
        F_nb_prev = poisson.cdf(x - 1, mu)
        p0_nb     = poisson.pmf(0,     mu)
    else:
        n = theta; p = theta / (theta + mu)
        F_nb      = nbinom.cdf(x,     n, p)
        F_nb_prev = nbinom.cdf(x - 1, n, p)
        p0_nb     = nbinom.pmf(0,     n, p)

    p_zero_total = pi + (1 - pi) * p0_nb
    v = np.random.rand(*x.shape) if jitter else 0.5
    u = np.empty_like(x, dtype=float)

    mask_nz = (x != 0)
    if np.any(mask_nz):
        F_x   = F_nb[mask_nz]
        F_xm1 = F_nb_prev[mask_nz]
        F_c_x   = (F_x   - p0_nb) / (1 - p0_nb)
        F_c_xm1 = (F_xm1 - p0_nb) / (1 - p0_nb)
        u[mask_nz] = p_zero_total + (1 - p_zero_total) * (
            F_c_xm1 + v[mask_nz] * (F_c_x - F_c_xm1)
        )

    return u


def zinb_log_pmf(x, pi, theta, mu):
    """Log-PMF of Zero-Inflated Negative Binomial at integer counts x (vectorized)."""
    x = np.asarray(x, dtype=int)
    if np.isinf(theta):
        log_nb = poisson.logpmf(x, mu)
    else:
        log_nb = nbinom.logpmf(x, theta, theta / (theta + mu))

    return np.where(
        x == 0,
        np.log(np.clip(pi + (1.0 - pi) * np.exp(log_nb), 1e-300, None)),
        np.log(np.clip(1.0 - pi, 1e-300, None)) + log_nb,
    )


def activate_from_logits(logits, confidence=1, center=True):
    """Sigmoid activation for pre-computed log-odds scores."""
    logits  = np.asarray(logits, dtype=float)
    zscores = stats.zscore(logits)
    median  = np.median(zscores) if center else 0.0
    return 1.0 / (1.0 + np.exp(-confidence * (zscores - median)))


def activate(p_rel, confidence=1, center=True):
    """Convert raw log-ratio scores to sigmoid-activated membership probabilities."""
    logs    = np.log(p_rel)
    zscores = stats.zscore(logs)
    median  = np.median(zscores) if center else 0
    return 1.0 / (1.0 + np.exp(-confidence * (zscores - median)))

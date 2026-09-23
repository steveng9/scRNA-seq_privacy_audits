#!/usr/bin/env python3
"""
B2 (marginal-noise defense) — noise injection into a trained scDesign2 copula.

Motivation
----------
The existing DP path (sdg/dp/dp_copula.py) noises ONLY the Gaussian-copula
covariance matrix (cov_mat) and leaves the per-gene marginals untouched
(dp_copula.py line 227).  A6 showed that covariance noise alone does NOT defeat
the *enhanced* (Class B) attack: enhanced BB+aux stays ~0.72 even at the paper's
η=1e-1 "crossover" level.  That is because Class B evidence comes from the
per-gene log-likelihood-ratio over the **secondary** genes (gene_sel2), whose
marginals covariance noise never touches.

B2 tests the complementary defense: perturb the fitted **marginal parameters**
before generation, then measure how the standard vs. enhanced attack degrade.
Hypothesis: marginal noise (especially on the secondary genes) is what actually
kills the Class B enhancement.

Noise model (unified perturbation level `s`, dimensionless)
-----------------------------------------------------------
scDesign2 marginals are stored as a (G, 3) matrix with columns (pi, theta, mu):
    pi     zero-inflation probability   in [0, 1]
    theta  NB dispersion / size         > 0
    mu     mean expression              > 0
We perturb each on its natural scale so the family is preserved:
    mu'    = mu    * exp(N(0, s^2))                 (multiplicative, log scale)
    theta' = theta * exp(N(0, s^2))                 (multiplicative, log scale)
    pi'    = sigmoid(logit(pi) + N(0, s^2))         (additive on logit scale)
Boundary pi (exactly 0 = pure NB, or 1) is left unchanged so we never flip a
gene's distribution family.

Covariance noise (mode 'cov'/'both') uses the same level `s` as a direct
per-entry Gaussian σ on the correlation matrix, followed by PSD projection and
re-normalisation (reusing the validated helpers in sdg/dp/dp_copula.py).  This
keeps every mode on one comparable knob rather than mixing an ε- and an
η-parametrisation.

This is an empirical perturbation study (like the paper's covariance-η sweep),
NOT a formal (ε,δ)-DP mechanism — consistent with the AC's guidance that a
rigorous DP mechanism is out of scope ("another paper").

Modes
-----
    none  : no noise (baseline; `s` ignored)
    cov   : noise cov_mat only            (reproduces A6 under the B2 pipeline)
    marg1 : noise primary marginals only  (gene_sel1 — affects standard PIT)
    marg2 : noise secondary marginals only(gene_sel2 — the Class B LLR genes)
    marg  : noise all marginals (marg1 + marg2)
    both  : noise cov_mat + all marginals
"""
import numpy as np

# Reuse the validated covariance post-processing (PSD projection + renormalise).
import os, sys
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from sdg.dp.dp_copula import (           # noqa: E402
    _add_symmetric_gaussian_noise, _project_to_psd, _normalise_to_correlation,
)

MODES = ["none", "cov", "marg1", "marg2", "marg", "both"]

_EPS = 1e-6


def _noise_marginals(marg_np: np.ndarray, s: float, rng) -> np.ndarray:
    """Perturb a (G, 3) marginal matrix (pi, theta, mu) at level s. Returns a copy."""
    out = np.array(marg_np, dtype=np.float64).copy()
    if out.ndim != 2 or out.shape[1] != 3 or s <= 0:
        return out
    G = out.shape[0]
    pi, theta, mu = out[:, 0], out[:, 1], out[:, 2]

    # mu, theta: multiplicative log-normal
    theta_new = theta * np.exp(rng.normal(0.0, s, size=G))
    mu_new = mu * np.exp(rng.normal(0.0, s, size=G))
    out[:, 1] = np.maximum(theta_new, _EPS)
    out[:, 2] = np.maximum(mu_new, _EPS)

    # pi: additive on logit scale, interior only (preserve NB/ZINB family at boundaries)
    interior = (pi > _EPS) & (pi < 1.0 - _EPS)
    if interior.any():
        logit = np.log(pi[interior] / (1.0 - pi[interior]))
        logit = logit + rng.normal(0.0, s, size=interior.sum())
        pi_new = 1.0 / (1.0 + np.exp(-logit))
        out[interior, 0] = np.clip(pi_new, _EPS, 1.0 - _EPS)
    return out


def _noise_cov(cov_np: np.ndarray, s: float, rng) -> np.ndarray:
    """Symmetric Gaussian noise σ=s on the correlation matrix, PSD-projected+renormalised."""
    if cov_np is None or s <= 0:
        return cov_np
    noised = _add_symmetric_gaussian_noise(np.asarray(cov_np, dtype=np.float64), s, rng)
    psd = _project_to_psd(noised)
    return _normalise_to_correlation(psd, clip=True)


def noise_and_save_rds(copula_path: str, cell_type: str, s: float, mode: str,
                       out_path: str, rng) -> dict:
    """
    Load one cell type's copula from `copula_path`, apply noise per `mode`/`s`,
    patch the R object in place (cov_mat, marginal_param1, marginal_param2 as
    dictated by mode) preserving dimnames, and saveRDS to `out_path`.

    Returns a small dict of what was noised (for logging).
    """
    from rpy2.robjects import r as R
    from rpy2.robjects.vectors import FloatVector

    copula_rds = R["readRDS"](copula_path)
    ct = copula_rds.rx2(str(cell_type))

    info = {"mode": mode, "s": s}
    # Assign once up front so saveRDS always has a valid object, even if every
    # component is skipped (mode 'none', NULL cov_mat, empty marginals, s<=0).
    R.assign("b2_obj", copula_rds)

    def _patch_matrix(field, new_np):
        """Rebuild ct[[field]] as a matrix with new values, preserving dimnames."""
        nr, ncol = new_np.shape
        flat = new_np.flatten(order="F").tolist()   # column-major for R matrix()
        R.assign("b2_vals", FloatVector(flat))
        R(f'b2_old <- b2_obj[["{cell_type}"]][["{field}"]]')
        R(f'b2_new <- matrix(b2_vals, nrow={nr}, ncol={ncol})')
        R('if (!is.null(dimnames(b2_old))) dimnames(b2_new) <- dimnames(b2_old)')
        R(f'b2_obj[["{cell_type}"]][["{field}"]] <- b2_new')

    do_cov = mode in ("cov", "both")
    do_m1 = mode in ("marg1", "marg", "both")
    do_m2 = mode in ("marg2", "marg", "both")

    if do_cov:
        cov_r = ct.rx2("cov_mat")
        from rpy2.rinterface_lib.sexp import NULLType
        if not isinstance(cov_r, NULLType):
            cov_np = np.array(cov_r, dtype=np.float64)
            noised_cov = _noise_cov(cov_np, s, rng)
            _patch_matrix("cov_mat", noised_cov)
            info["cov_G"] = int(noised_cov.shape[0])

    if do_m1:
        m1 = np.array(ct.rx2("marginal_param1"), dtype=np.float64)
        if m1.ndim == 2 and m1.shape[0] > 0:
            _patch_matrix("marginal_param1", _noise_marginals(m1, s, rng))
            info["m1_G"] = int(m1.shape[0])

    if do_m2:
        m2 = np.array(ct.rx2("marginal_param2"), dtype=np.float64)
        if m2.ndim == 2 and m2.shape[0] > 0:
            _patch_matrix("marginal_param2", _noise_marginals(m2, s, rng))
            info["m2_G"] = int(m2.shape[0])

    R(f'saveRDS(b2_obj, file="{out_path}")')
    return info

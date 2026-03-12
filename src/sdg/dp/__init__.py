"""
Differential privacy mechanisms for synthetic single-cell RNA-seq data generators.

The primary target is the Gaussian copula used by scDesign2 and scDesign3.
Both generators store the copula as a correlation matrix R ∈ ℝ^{G×G} (diagonal = 1)
plus per-gene marginal parameters.  DP noise is injected into R via the Gaussian
mechanism (approximate DP) calibrated to donor-level sensitivity.

Modules
-------
sensitivity  — pure-math sensitivity bounds (no project dependencies)
dp_copula    — Gaussian mechanism applied to a parsed copula dict
"""

"""
scDesign3 SDG — Gaussian and vine copula single-cell data generator.

Uses the official scDesign3 R package (Bioconductor) via subprocess,
mirroring the scDesign2 integration pattern.

Copula types supported:
  - "gaussian"  (default) — Gaussian copula; scMAMA-MIA attack applies directly
  - "vine"                — vine copula (rvinecopulib); attack uses same interface

Reference: sun2023scdesign3
"""

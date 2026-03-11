"""
scVAE SDG — NOT YET IMPLEMENTED.

scVAE is a VAE-based generator; it does not expose a Gaussian copula, so a
different MIA strategy is required (likelihood-based or shadow model approach).

To implement:
  1. Add model.py that wraps or re-implements the scVAE architecture.
  2. Add a corresponding attack module under src/attacks/ (not scmamamia/).
  3. Register "scvae" in src/sdg/run.py under generator_classes.
  4. Add experiment configs under experiments/scvae/.

Reference: gronbech2020scvae
"""

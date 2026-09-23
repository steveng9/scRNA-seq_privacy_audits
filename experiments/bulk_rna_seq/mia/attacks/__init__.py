"""Membership inference attacks.

This is a PRUNED copy (see ../../README.md for provenance): only `mahalamia`
is included. The source repo (MIA_on_bulkRNAseq_CAMDA2026) also has
`mamamia` (targets DP-PGM) and the two MeLoMIA variants (loss-trajectory
attacks against CVAE/ND), deliberately left out here -- out of scope for the
MVN-focused DP work this copy exists for.
"""

from .base import REGISTRY, Attack, build, register  # noqa: F401
from .mahalamia import MahalaMIA  # noqa: F401

__all__ = ["Attack", "build", "register", "REGISTRY", "MahalaMIA"]

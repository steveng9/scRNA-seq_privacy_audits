"""Target generators.

This is a PRUNED copy (see ../../README.md for provenance): only `mvn` is
included. The source repo (MIA_on_bulkRNAseq_CAMDA2026) also has `cvae`,
`nd`, and `pgm` generators, deliberately left out here -- they pull in
heavier deps (torch, an external private-PGM repo) not needed for the
DP-MVN work this copy exists for.
"""

from .base import REGISTRY, Generator, build, register  # noqa: F401
from .mvn import MVNGenerator  # noqa: F401

__all__ = ["Generator", "build", "register", "REGISTRY"]

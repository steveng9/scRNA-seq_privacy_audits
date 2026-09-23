"""Common interface for the four target generators.

A generator is anything that can be fitted to a labelled real cohort and then
sampled from.  It always speaks *raw* gene-expression space: `fit` receives the
untransformed VST values and `sample` returns them, so callers never have to
know whether a particular generator normalises internally.

The same classes serve two roles:
  * target   -- trained on a split's member half to produce the synthetic
                dataset an attack is evaluated against;
  * shadow   -- trained by MeLoMIA on data it controls, to build the
                membership-signal extractor.
Keeping one implementation for both is deliberate: a shadow that differs from
the target is a silent source of attack degradation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Generator(ABC):
    """Base class.  Subclasses declare their hyperparameters as dataclass fields."""

    seed: int = 42
    device: str = "cuda"
    verbose: bool = True

    #: registry key, e.g. "mvn"
    name: str = field(init=False, default="base")

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> "Generator":
        """Train on raw expression `X` (n, n_genes) with integer labels `y`."""

    @abstractmethod
    def sample(self, n: int) -> tuple:
        """Return (X_syn raw float32 (n, n_genes), y_syn int64 (n,))."""

    def save(self, path: Path) -> None:  # pragma: no cover - optional
        raise NotImplementedError(f"{type(self).__name__} does not support save()")

    def load(self, path: Path) -> "Generator":  # pragma: no cover - optional
        raise NotImplementedError(f"{type(self).__name__} does not support load()")

    def params(self) -> dict:
        """Hyperparameters as a plain dict, for the run record."""
        from dataclasses import asdict
        d = asdict(self)
        d.pop("verbose", None)
        d["generator"] = self.name
        return d


REGISTRY: dict[str, Any] = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def build(name: str, **params) -> Generator:
    if name not in REGISTRY:
        raise KeyError(f"Unknown generator {name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)

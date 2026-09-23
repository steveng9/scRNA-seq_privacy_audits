"""Common interface for the membership inference attacks.

An attack has two phases, split so the expensive one happens once:

    prepare(dataset)                 build whatever reusable machinery the
                                     attack needs -- for MeLoMIA, the whole
                                     shadow / synth-shadow / meta-classifier
                                     stack.  Cached on disk, keyed by the
                                     attack's own hyperparameters.

    score(dataset, generator, split) score every real sample against one
                                     released synthetic dataset.  Returns an
                                     array aligned to
                                     `datasets.load_expression(dataset).index`,
                                     higher = more likely a training member.

`prepare` deliberately does not see the target.  The adversary may reuse one
shadow stack against any number of released datasets, and keeping the split
explicit stops target information from leaking into the meta-classifier.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .. import datasets as D
from .. import metrics as M
from .. import runs as R


@dataclass
class Attack(ABC):
    seed: int = 42
    device: str = "cuda"
    verbose: bool = True

    name: str = field(init=False, default="base")

    # ── Interface ───────────────────────────────────────────────────────────

    def prepare(self, dataset: str) -> None:
        """Build reusable machinery.  Default: nothing to build."""
        return None

    @abstractmethod
    def score(self, dataset: str, generator: str, split: int) -> np.ndarray:
        """Membership scores for all real samples, in expression-matrix order."""

    # ── Bookkeeping ─────────────────────────────────────────────────────────

    def params(self) -> dict:
        d = asdict(self)
        d["attack"] = self.name
        d.pop("verbose", None)
        return d

    def tag(self) -> str:
        """Short label distinguishing hyperparameter variants of this attack."""
        return "default"

    def evaluate(self, dataset: str, generator: str, split: int,
                 save: bool = True, experiment: str = "", variant: str = "",
                 notes: str = "") -> dict:
        """Score one target, compute metrics, and (by default) record the run."""
        scores = self.score(dataset, generator, split)
        y = D.membership_labels(dataset, split)
        ids = list(D.load_expression(dataset).index)
        metrics = M.evaluate(y, scores)

        if save:
            # Record what was attacked, not only its name: a target rebuilt in
            # place keeps its name, and only the fingerprint tells them apart.
            from .. import targets as T
            target = T.target_record(dataset, generator, split)
            R.save_run(
                dataset=dataset, attack=self.params().get("attack", self.name),
                generator=generator, split=split,
                params=self.params(), sample_ids=ids, scores=scores, y_member=y,
                metrics=metrics, tag=self.tag(), experiment=experiment,
                variant=variant, notes=notes, target=target,
            )
        return metrics


REGISTRY: dict[str, Any] = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def build(name: str, **params) -> Attack:
    if name not in REGISTRY:
        raise KeyError(f"Unknown attack {name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)

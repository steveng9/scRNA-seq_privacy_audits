"""Filesystem locations.

Everything the project reads or writes resolves through this module, so a new
machine only needs the four environment variables below (or edits here).
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_path(var: str, default: str) -> Path:
    return Path(os.environ.get(var, default)).expanduser().resolve()


# ── External inputs ──────────────────────────────────────────────────────────
# Challenge data as distributed by the ELSA benchmark platform.
CHALLENGE_DATA = _env_path("CAMDA_DATA", "~/data/CAMDA26")

# Blue-team NoisyDiffusion repo: supplies the canonical 5 train/test splits and
# the ND synthetic data generated from them.
ND_REPO = _env_path("CAMDA_ND_REPO", "~/CAMDA25_NoisyDiffusion")

# Private-PGM generator repo (DP-PGM target generator).
PGM_REPO = _env_path("CAMDA_PGM_REPO", "~/private-pgm-rnaseq-camda2026")

# ── Project outputs ──────────────────────────────────────────────────────────
# Large, regenerable intermediates: target synthetic datasets, shadow models,
# extracted features, meta-classifiers.  Git-ignored.
ARTIFACTS = _env_path("CAMDA_ARTIFACTS", str(PROJECT_ROOT / "artifacts"))

# Small, permanent experiment records: per-run config, row-level scores, metrics.
# Tracked in git so results are reviewable and diffable.
RESULTS = _env_path("CAMDA_RESULTS", str(PROJECT_ROOT / "results"))

CONFIGS = PROJECT_ROOT / "configs"

# ── Derived artifact subtrees ────────────────────────────────────────────────
CACHE_DIR = ARTIFACTS / "cache"           # decompressed label files etc.
SPLITS_DIR = ARTIFACTS / "splits"         # canonical target splits per dataset
TARGETS_DIR = ARTIFACTS / "targets"       # target synthetic datasets
ATTACK_DIR = ARTIFACTS / "attacks"        # shadow models / features / classifiers

RUNS_DIR = RESULTS / "runs"
INDEX_CSV = RESULTS / "index.csv"


def ensure(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def target_dir(dataset: str, generator: str, split: int) -> Path:
    """Where one target synthetic dataset lives.

    `generator` may carry a variant, `pgm@epsilon=0.1`, naming the parameters
    that differ from the defaults; each variant gets its own directory so a
    sweep never overwrites the canonical target.  See `targets.variant_name`.
    """
    base, _, variant = generator.partition("@")
    root = TARGETS_DIR / dataset / base
    return (root / variant if variant else root) / f"split_{split}"


def attack_cache(attack: str, dataset: str, tag: str = "default") -> Path:
    """Where an attack keeps its reusable shadow-model machinery."""
    return ATTACK_DIR / attack / dataset / tag

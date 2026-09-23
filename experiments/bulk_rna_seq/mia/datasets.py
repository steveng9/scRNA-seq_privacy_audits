"""Dataset registry and loaders for the two TCGA bulk RNA-seq cohorts.

File formats, as distributed by the challenge:

  expression TSV : rows = 978 landmark genes, columns = sample IDs.  Transposed
                   on load so every downstream array is (n_samples, n_genes).
  subtype CSV    : inside the BLUE_*.zip, indexed by sample ID.
  splits YAML    : from the NoisyDiffusion blue-team repo; lists the *test*
                   (non-member) sample IDs for each of the 5 canonical splits.

The splits YAML is the anchor of the whole evaluation: all four target
generators are trained on the same five 80/20 partitions, so a sample's
membership label is identical across every cell of the attack x generator grid.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from . import paths


@dataclass(frozen=True)
class DatasetSpec:
    name: str                       # short handle used everywhere ("BRCA")
    full_name: str                  # challenge name ("TCGA-BRCA")
    subtype_col: str                # column of the subtype CSV holding the class
    n_genes: int = 978
    has_reference: bool = False     # auxiliary non-member set (COMBINED only)

    # ── Challenge inputs ────────────────────────────────────────────────────
    @property
    def red_dir(self) -> Path:
        return paths.CHALLENGE_DATA / f"RED_{self.full_name}"

    @property
    def expr_tsv(self) -> Path:
        return self.red_dir / f"{self.full_name}_primary_tumor_star_deseq_VST_lmgenes.tsv"

    @property
    def reference_tsv(self) -> Optional[Path]:
        if not self.has_reference:
            return None
        return self.red_dir / f"{self.full_name}_primary_tumor_star_deseq_VST_lmgenes_reference.tsv"

    @property
    def blue_zip(self) -> Path:
        return paths.CHALLENGE_DATA / f"BLUE_{self.full_name}.zip"

    @property
    def subtype_member(self) -> str:
        """Path of the subtype CSV *inside* the blue zip."""
        return f"BLUE_{self.full_name}/{self.full_name}_primary_tumor_subtypes.csv"

    # ── NoisyDiffusion blue-team repo ───────────────────────────────────────
    @property
    def nd_dir(self) -> Path:
        return paths.ND_REPO / self.full_name

    @property
    def splits_yaml(self) -> Path:
        return self.nd_dir / f"{self.full_name}_splits.yaml"

    @property
    def nd_synthetic_dir(self) -> Path:
        return self.nd_dir / "synthetic_data"


DATASETS = {
    "BRCA": DatasetSpec(
        name="BRCA",
        full_name="TCGA-BRCA",
        subtype_col="Subtype",
        has_reference=False,
    ),
    "COMBINED": DatasetSpec(
        name="COMBINED",
        full_name="TCGA-COMBINED",
        subtype_col="project",
        has_reference=True,
    ),
}


def spec(dataset: str) -> DatasetSpec:
    if dataset not in DATASETS:
        raise KeyError(f"Unknown dataset {dataset!r}. Known: {sorted(DATASETS)}")
    return DATASETS[dataset]


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=4)
def load_expression(dataset: str) -> pd.DataFrame:
    """Full real cohort as (n_samples, n_genes), index = sample IDs."""
    ds = spec(dataset)
    df = pd.read_csv(ds.expr_tsv, sep="\t", index_col=0)
    return df.T


@lru_cache(maxsize=4)
def load_reference(dataset: str) -> Optional[pd.DataFrame]:
    """Auxiliary set of known non-members, or None when the cohort has none."""
    ds = spec(dataset)
    if ds.reference_tsv is None or not ds.reference_tsv.exists():
        return None
    return pd.read_csv(ds.reference_tsv, sep="\t", index_col=0).T


@lru_cache(maxsize=4)
def load_subtypes(dataset: str) -> pd.Series:
    """Ground-truth class label per sample, aligned to `load_expression` order.

    The CSV ships inside the blue-team zip; it is extracted once into the
    artifact cache so repeated runs do not re-open the archive.
    """
    ds = spec(dataset)
    cached = paths.CACHE_DIR / f"{ds.full_name}_subtypes.csv"
    if not cached.exists():
        paths.ensure(paths.CACHE_DIR)
        with zipfile.ZipFile(ds.blue_zip) as zf, open(cached, "wb") as out:
            out.write(zf.read(ds.subtype_member))

    df = pd.read_csv(cached, index_col=0)
    if ds.subtype_col not in df.columns:
        raise KeyError(
            f"{cached} has no column {ds.subtype_col!r}; found {list(df.columns)}"
        )
    # The COMBINED CSV repeats the sample ID in a 'samplesID' column and uses it
    # as the index too; BRCA indexes by sample ID directly.  Either way the
    # index is the join key.
    labels = df[ds.subtype_col]
    expr_index = load_expression(dataset).index
    missing = expr_index.difference(labels.index)
    if len(missing):
        raise ValueError(f"{len(missing)} samples have no subtype label, e.g. {list(missing[:3])}")
    return labels.reindex(expr_index)


@lru_cache(maxsize=4)
def class_names(dataset: str) -> tuple:
    """Sorted class vocabulary — the canonical int encoding for conditioning."""
    return tuple(sorted(load_subtypes(dataset).unique()))


def encode_subtypes(dataset: str, labels: pd.Series | np.ndarray) -> np.ndarray:
    vocab = {c: i for i, c in enumerate(class_names(dataset))}
    return np.array([vocab[l] for l in np.asarray(labels)], dtype=np.int64)


def n_classes(dataset: str) -> int:
    return len(class_names(dataset))


# ─────────────────────────────────────────────────────────────────────────────
# Canonical target splits
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=4)
def load_target_splits(dataset: str) -> dict:
    """The five canonical 80/20 partitions shared by all target generators.

    Returns {split_no: {"member_ids": [...], "nonmember_ids": [...]}}.

    Sourced from the NoisyDiffusion repo's splits YAML, which lists the held-out
    (non-member) IDs per split.  Using these rather than freshly drawn splits is
    what lets the ND column of the grid reuse the blue team's already-generated
    synthetic data instead of retraining it.
    """
    ds = spec(dataset)
    with open(ds.splits_yaml) as f:
        raw = yaml.safe_load(f)["splits"]

    all_ids = list(load_expression(dataset).index)
    out = {}
    for key, entry in raw.items():
        split_no = int(key.split("_")[-1])
        nonmember = set(entry["test_index"])
        out[split_no] = {
            "member_ids": [s for s in all_ids if s not in nonmember],
            "nonmember_ids": [s for s in all_ids if s in nonmember],
        }
    return dict(sorted(out.items()))


def membership_labels(dataset: str, split: int) -> np.ndarray:
    """0/1 membership vector over all real samples, in `load_expression` order."""
    nonmember = set(load_target_splits(dataset)[split]["nonmember_ids"])
    return np.array(
        [0 if sid in nonmember else 1 for sid in load_expression(dataset).index],
        dtype=np.int64,
    )


def training_subset(dataset: str, split: int) -> tuple:
    """(X, y_int, sample_ids) for the member half of one split — what a target
    generator is trained on."""
    expr = load_expression(dataset)
    member = set(load_target_splits(dataset)[split]["member_ids"])
    mask = np.array([sid in member for sid in expr.index])
    X = expr.values[mask].astype(np.float32)
    y = encode_subtypes(dataset, load_subtypes(dataset).values[mask])
    ids = list(expr.index[mask])
    return X, y, ids


def gene_names(dataset: str) -> list:
    return list(load_expression(dataset).columns)

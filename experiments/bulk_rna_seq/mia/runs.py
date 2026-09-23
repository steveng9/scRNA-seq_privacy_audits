"""Experiment record keeping.

Every attack evaluation writes exactly one run directory:

    results/runs/<run_id>/
        config.json    fully resolved parameters (what was run)
        scores.csv     sample_id, score, y_member  -- row level, never overwritten
        metrics.json   auc, aupr, tpr@fpr, ...

plus one line appended to results/index.csv, which is the flat table to load
with pandas when building figures or paper tables.

`run_id` is deterministic: the same configuration always maps to the same
directory, so re-running an experiment updates it in place instead of
accumulating near-duplicates.  The trailing hash covers every parameter that is
not already in the readable prefix, so two runs that differ only in, say, shadow
count get different directories.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import csvlock, paths

INDEX_COLUMNS = [
    "run_id", "timestamp", "dataset", "experiment", "attack", "variant",
    "generator", "split",
    "auc", "aupr", "tpr_at_fpr_0.01", "tpr_at_fpr_0.1",
    "precision_at_5pct", "n", "n_members", "tag", "notes",
]

#: Fields copied verbatim from a run's config.json into the index.
_META_FIELDS = ("run_id", "timestamp", "dataset", "experiment", "attack",
                "variant", "generator", "split", "tag", "notes")

#: Metric fields promoted into the index; the rest stay in metrics.json.
_METRIC_FIELDS = ("auc", "aupr", "tpr_at_fpr_0.01", "tpr_at_fpr_0.1",
                  "precision_at_5pct", "n", "n_members")


def _hash(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def make_run_id(dataset: str, attack: str, generator: str, split, params: dict) -> str:
    split_part = "all" if split is None else f"s{split}"
    return f"{dataset}__{attack}__{generator}__{split_part}__{_hash(params)}"


def save_run(
    *,
    dataset: str,
    attack: str,
    generator: str,
    split,
    params: dict,
    sample_ids,
    scores,
    y_member,
    metrics: dict,
    tag: str = "",
    experiment: str = "",
    variant: str = "",
    notes: str = "",
    target: dict | None = None,
) -> Path:
    """Persist one attack evaluation and index it.  Returns the run directory."""
    run_id = make_run_id(dataset, attack, generator, split, params)
    out = paths.RUNS_DIR / run_id
    out.mkdir(parents=True, exist_ok=True)

    record = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset,
        "attack": attack,
        "generator": generator,
        "split": split,
        "tag": tag,
        "experiment": experiment,
        "variant": variant or attack,
        "notes": notes,
        "params": params,
        "target": target,
    }
    (out / "config.json").write_text(json.dumps(record, indent=2, default=str))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    pd.DataFrame(
        {
            "sample_id": list(sample_ids),
            "score": np.asarray(scores, dtype=float),
            "y_member": np.asarray(y_member, dtype=int),
        }
    ).to_csv(out / "scores.csv", index=False)

    _append_index(record, metrics)
    return out


def _append_index(record: dict, metrics: dict) -> None:
    """Replace this run's row in the index, atomically and under a lock.

    De-duplicating on run_id means the whole file is rewritten every time, which
    is a read-modify-write and therefore a race.  Two experiment processes
    sharing one index -- one per cohort, say -- will interleave and leave NUL
    bytes where one truncated the file while the other was reading it, and the
    next reader dies with `_csv.Error: line contains NUL`.

    `csvlock.atomic_update` holds an exclusive flock across the whole
    read-modify-write and installs the result with `os.replace`.  A reader that
    takes no lock still never sees a torn file, only the old one or the new one.
    The run directory is written before this is called and is the source of
    truth -- `scripts/reindex.py` rebuilds the index from it.
    """
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    row = {c: "" for c in INDEX_COLUMNS}
    row.update({k: record.get(k, "") for k in _META_FIELDS})
    for k in _METRIC_FIELDS:
        if k in metrics:
            row[k] = metrics[k]

    with csvlock.atomic_update(paths.INDEX_CSV) as tmp:
        existing = []
        if paths.INDEX_CSV.exists():
            with open(paths.INDEX_CSV, newline="") as f:
                existing = [r for r in csv.DictReader(f)
                            if r.get("run_id") != row["run_id"]]

        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
            w.writeheader()
            for r in existing:
                w.writerow({c: r.get(c, "") for c in INDEX_COLUMNS})
            w.writerow(row)


def load_index() -> pd.DataFrame:
    if not paths.INDEX_CSV.exists():
        return pd.DataFrame(columns=INDEX_COLUMNS)
    return pd.read_csv(paths.INDEX_CSV)


def load_run(run_id: str) -> dict:
    out = paths.RUNS_DIR / run_id
    return {
        "config": json.loads((out / "config.json").read_text()),
        "metrics": json.loads((out / "metrics.json").read_text()),
        "scores": pd.read_csv(out / "scores.csv"),
    }


def rebuild_index() -> pd.DataFrame:
    """Regenerate results/index.csv from the run directories on disk.

    The index is a convenience view, not the source of truth.  Deleting runs by
    hand, or changing an attack's parameters so old runs get different ids,
    leaves rows in it that point nowhere -- this drops them.
    """
    rows = []
    for d in sorted(paths.RUNS_DIR.glob("*")):
        cfg_path, met_path = d / "config.json", d / "metrics.json"
        if not (cfg_path.exists() and met_path.exists()):
            continue
        record = json.loads(cfg_path.read_text())
        metrics = json.loads(met_path.read_text())
        row = {c: "" for c in INDEX_COLUMNS}
        row.update({k: record.get(k, "") for k in _META_FIELDS})
        row.setdefault("variant", record.get("attack", ""))
        for k in _METRIC_FIELDS:
            if k in metrics:
                row[k] = metrics[k]
        rows.append(row)

    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    with open(paths.INDEX_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return pd.DataFrame(rows)


def prune(dataset=None, attack=None, dry_run: bool = True) -> list:
    """Delete run directories matching a filter, then reindex."""
    import shutil
    removed = []
    for d in sorted(paths.RUNS_DIR.glob("*")):
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            continue
        rec = json.loads(cfg_path.read_text())
        if dataset and rec.get("dataset") != dataset:
            continue
        if attack and rec.get("attack") != attack:
            continue
        removed.append(d.name)
        if not dry_run:
            shutil.rmtree(d)
    if not dry_run:
        rebuild_index()
    return removed

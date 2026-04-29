"""
train.py — train v2 scDesign2 copulas.

The v2 R script (src/sdg/scdesign2/scdesign2_v2.r) replaces cor(t(z))
with the uncentered second moment ZZᵀ/(N−1).  This module orchestrates
training one (dataset, n_donors, trial) combination by:

  1. Loading the full anndata file (backed='r' to avoid OOM).
  2. Filtering to the donors listed in
       ~/data/scMAMAMIA/{dataset}/splits/{nd}d/{trial}/train.npy
  3. HVG-subsetting using the dataset's hvg.csv.
  4. Writing a temp h5ad and forking parallel Rscript subprocesses
     (one per cell type) calling scdesign2_v2.r train.
  5. Saving the .rds copulas to
       ~/data/scMAMAMIA/{dataset}/scdesign2/v2_no_dp/{nd}d/{trial}/models/
     and a provenance.json next to it.

Defaults to MAX_WORKERS=2 to leave room for the running baselines sweep.
"""

import argparse
import datetime
import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scanpy as sc

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_ROOT = "/home/golobs/data/scMAMAMIA"
R_SCRIPT  = os.path.join(REPO_ROOT, "src", "sdg", "scdesign2", "scdesign2_v2.r")

DEFAULT_MAX_WORKERS = 2
DELTA = 1e-5
DONOR_COL     = "individual"
CELL_TYPE_COL = "cell_type"


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_r_train(hvg_subset_path, cell_type, out_rds_path):
    cmd = ["Rscript", R_SCRIPT, "train", hvg_subset_path, str(cell_type), out_rds_path]
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return (cell_type, "ok", None)
    except subprocess.CalledProcessError as e:
        msg = e.output.decode("utf-8", errors="replace")
        return (cell_type, "fail", msg[-500:])


def train_one(dataset, nd, trial, max_workers=DEFAULT_MAX_WORKERS, force=False):
    base = os.path.join(DATA_ROOT, dataset)
    full_h5ad   = os.path.join(base, "full_dataset_cleaned.h5ad")
    hvg_path    = os.path.join(base, "hvg.csv")
    splits_dir  = os.path.join(base, "splits", f"{nd}d", str(trial))
    train_npy   = os.path.join(splits_dir, "train.npy")

    out_root  = os.path.join(base, "scdesign2", "v2_no_dp", f"{nd}d", str(trial))
    models_dir = os.path.join(out_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(full_h5ad):
        raise FileNotFoundError(full_h5ad)
    if not os.path.exists(train_npy):
        raise FileNotFoundError(train_npy)
    if not os.path.exists(hvg_path):
        raise FileNotFoundError(hvg_path)

    print(f"[v2/train] {dataset} {nd}d t{trial} — loading full anndata (backed)…", flush=True)
    full = sc.read_h5ad(full_h5ad, backed="r")
    train_donors = np.load(train_npy, allow_pickle=True)
    train_mask = full.obs[DONOR_COL].isin(train_donors).values
    print(f"[v2/train]   train donors={len(train_donors)}, train cells={train_mask.sum()}", flush=True)

    # Materialize the train slice into memory (subset only — fits comfortably for nd=20)
    X_train = full[train_mask, :].to_memory()
    full.file.close()

    # HVG mask
    hvg_df = pd.read_csv(hvg_path, index_col=0)
    if len(hvg_df) != len(X_train.var_names):
        hvg_df = hvg_df.reindex(X_train.var_names).fillna(False)
    hvg_mask = hvg_df["highly_variable"].values.astype(bool)
    print(f"[v2/train]   {hvg_mask.sum()} HVGs", flush=True)
    X_hvg = X_train[:, hvg_mask].copy()

    # Mean expression per cell type (matches v1 layout — saved to models/mean_expr.csv)
    import scipy.sparse as sp
    X_sp = X_train.X if sp.issparse(X_train.X) else sp.csr_matrix(X_train.X)
    cts  = X_train.obs[CELL_TYPE_COL].values
    unique_cts = sorted(Counter(cts).keys(), key=lambda c: str(c))
    means = {ct: np.asarray(X_sp[cts == ct].mean(axis=0)).flatten() for ct in unique_cts}
    means_df = pd.DataFrame(means, index=X_train.var_names).T
    means_df.to_csv(os.path.join(models_dir, "mean_expr.csv"), index=True)

    # Skip cell types that already have a .rds (idempotent unless --force)
    cell_types_to_train = []
    for ct in unique_cts:
        out_rds = os.path.join(models_dir, f"{ct}.rds")
        if os.path.exists(out_rds) and not force:
            print(f"[v2/train]   skip {ct} — {out_rds} exists", flush=True)
            continue
        cell_types_to_train.append(ct)

    # Free RAM before fork
    del X_train

    if not cell_types_to_train:
        print(f"[v2/train]   all cell types already trained.", flush=True)
        _write_provenance(out_root, dataset, nd, trial, full_h5ad, hvg_path, splits_dir,
                          n_cell_types=len(unique_cts))
        return out_root

    # Write the temp HVG-subset h5ad
    with tempfile.TemporaryDirectory(prefix="v2_train_") as tmp_dir:
        hvg_h5ad = os.path.join(tmp_dir, "hvg_train.h5ad")
        X_hvg.write(hvg_h5ad)
        del X_hvg

        ctx = multiprocessing.get_context("spawn")
        print(f"[v2/train]   launching parallel R training "
              f"({len(cell_types_to_train)} cell types, max_workers={max_workers})…",
              flush=True)
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as exe:
            futures = {
                exe.submit(_run_r_train, hvg_h5ad, ct,
                           os.path.join(models_dir, f"{ct}.rds")): ct
                for ct in cell_types_to_train
            }
            for fut in as_completed(futures):
                ct, status, err = fut.result()
                if status == "ok":
                    print(f"[v2/train]   ✓ {ct}", flush=True)
                else:
                    print(f"[v2/train]   ✗ {ct} — {err[:200] if err else ''}", flush=True)

    _write_provenance(out_root, dataset, nd, trial, full_h5ad, hvg_path, splits_dir,
                      n_cell_types=len(unique_cts))
    print(f"[v2/train] done: {out_root}", flush=True)
    return out_root


def _write_provenance(out_root, dataset, nd, trial,
                      full_h5ad, hvg_path, splits_dir, n_cell_types):
    prov = {
        "stage":         "train",
        "dp_variant":    "v2",
        "dataset":       dataset,
        "n_donors":      nd,
        "trial":         trial,
        "n_cell_types_attempted": n_cell_types,
        "r_script_path": R_SCRIPT,
        "r_script_sha256": _file_sha256(R_SCRIPT),
        "inputs": {
            "full_h5ad":   full_h5ad,
            "hvg_path":    hvg_path,
            "splits_dir":  splits_dir,
        },
        "trained_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(out_root, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ok")
    ap.add_argument("--nd",      type=int, required=True)
    ap.add_argument("--trial",   type=int, nargs="+", required=True,
                    help="One or more trial numbers (e.g. --trial 1 2)")
    ap.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                    help=f"Parallel R workers (default: {DEFAULT_MAX_WORKERS})")
    ap.add_argument("--force",   action="store_true",
                    help="Re-train cell types whose .rds already exists")
    args = ap.parse_args()

    for t in args.trial:
        train_one(args.dataset, args.nd, t,
                  max_workers=args.max_workers, force=args.force)


if __name__ == "__main__":
    main()

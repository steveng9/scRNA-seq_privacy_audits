"""
Targeted re-generation script for sd3v (scDesign3 vine copula) trials.

All existing sd3v model .rds files were trained correctly on 1141 HVGs.
Only the Python assembly step was broken (column-index bug, now fixed in
src/sdg/scdesign3/model.py).  This script therefore:

  - Skips model training for any trial that already has model .rds files
  - Re-runs R generation (simu_new, fast) + fixed Python assembly
  - Falls back to full train+generate for trials with no model files (50d/t1)

Runs one trial at a time (N_WORKERS=1) to avoid OOM: each trial spawns up to
15 R subprocesses for parallel cell-type generation.

Usage
-----
    conda run --no-capture-output -n tabddpm_ \\
        python experiments/sdg_comparison/regenerate_sd3v.py \\
        > /tmp/regenerate_sd3v.log 2>&1 &
    echo $! > /tmp/regenerate_sd3v.pid
    tail -f /tmp/regenerate_sd3v.log
"""

import os
import sys
import glob
import shutil

import numpy as np
import anndata as ad
import scanpy as sc

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC  = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

DATA = "/home/golobs/data"

# Trials to regenerate (donor_count, trial_index)
OK_SD3V = [10, 20, 50]
N_TRIALS = 5

INDIVIDUAL_COL = "individual"
CELL_TYPE_COL  = "cell_type"

OK_DATASET    = os.path.join(DATA, "ok", "full_dataset_cleaned.h5ad")
OK_HVG_PATH   = os.path.join(DATA, "ok", "hvg_full.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_train_h5ad(dataset_path, train_donors, out_path, hvg_path):
    """Write HVG-filtered train subset to disk (if not already present)."""
    if os.path.exists(out_path):
        return ad.read_h5ad(out_path, backed="r").n_obs
    import pandas as pd
    adata = sc.read_h5ad(dataset_path, backed="r")
    mask  = adata.obs[INDIVIDUAL_COL].isin(set(train_donors))
    sub   = adata[mask].to_memory()
    adata.file.close()
    hvg_df = pd.read_csv(hvg_path, index_col=0)
    hvgs   = set(hvg_df[hvg_df["highly_variable"]].index)
    keep   = [g for g in sub.var_names if g in hvgs]
    sub    = sub[:, keep].copy()
    sub.write_h5ad(out_path)
    n = sub.n_obs
    del sub
    return n


def _models_complete(models_dir, expected_n=14):
    """Return True if at least expected_n cell-type .rds files exist."""
    rds = [f for f in glob.glob(os.path.join(models_dir, "*.rds"))
           if "mean_expr" not in os.path.basename(f)]
    return len(rds) >= expected_n


# ---------------------------------------------------------------------------
# Per-trial generation (skip training when models exist)
# ---------------------------------------------------------------------------

def regenerate_trial(nd, trial):
    from sdg.scdesign3.model import ScDesign3

    out_dir   = os.path.join(DATA, "ok_sd3v", f"{nd}d", str(trial))
    ds_dir    = os.path.join(out_dir, "datasets")
    synth_out = os.path.join(ds_dir, "synthetic.h5ad")
    models_dir = os.path.join(out_dir, "models")

    if os.path.exists(synth_out):
        print(f"  [SKIP] {nd}d/t{trial}: synthetic.h5ad already exists.")
        return True

    splits_dir = os.path.join(DATA, "ok", f"{nd}d", str(trial), "datasets")
    train_donors = np.load(os.path.join(splits_dir, "train.npy"), allow_pickle=True)

    os.makedirs(ds_dir, exist_ok=True)

    # Copy split files if missing
    for fname in ["train.npy", "holdout.npy", "auxiliary.npy"]:
        src = os.path.join(splits_dir, fname)
        dst = os.path.join(ds_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Write train.h5ad (needed by model.generate() → load_test_anndata)
    train_h5ad = os.path.join(ds_dir, "train.h5ad")
    n_cells = _write_train_h5ad(OK_DATASET, train_donors, train_h5ad, OK_HVG_PATH)
    print(f"  {nd}d/t{trial}: {n_cells:,} cells", flush=True)

    config = {
        "generator_name": "scdesign3",
        "dir_list": {
            "home": out_dir,
            "data": ds_dir,
        },
        "scdesign3_config": {
            "out_model_path": "models",
            "hvg_path":       OK_HVG_PATH,
            "copula_type":    "vine",
            "family_use":     "nb",
            "trunc_lvl":      1,
        },
        "dataset_config": {
            "name":                "data",
            "train_count_file":    "train.h5ad",
            "test_count_file":     "train.h5ad",
            "cell_type_col_name":  CELL_TYPE_COL,
            "cell_label_col_name": "cell_label",
            "random_seed":         42,
        },
    }

    model = ScDesign3(config)

    if _models_complete(models_dir):
        print(f"  {nd}d/t{trial}: models exist — skipping training, running generation only.", flush=True)
    else:
        print(f"  {nd}d/t{trial}: no models found — running full train+generate.", flush=True)
        model.train()

    print(f"  {nd}d/t{trial}: generating ...", flush=True)
    synth = model.generate()
    synth.write_h5ad(synth_out)
    print(f"  {nd}d/t{trial}: saved → {synth_out}", flush=True)

    if os.path.exists(train_h5ad):
        os.remove(train_h5ad)

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("sd3v re-generation (skipping training where models exist)")
    print("=" * 60, flush=True)

    failed = []
    for nd in OK_SD3V:
        for trial in range(1, N_TRIALS + 1):
            out_dir = os.path.join(DATA, "ok_sd3v", f"{nd}d", str(trial))
            if not os.path.isdir(out_dir):
                print(f"  [SKIP] {nd}d/t{trial}: directory missing ({out_dir})")
                continue
            synth_out = os.path.join(out_dir, "datasets", "synthetic.h5ad")
            if os.path.exists(synth_out):
                print(f"  [SKIP] {nd}d/t{trial}: already done.")
                continue
            print(f"\n--- {nd}d/t{trial} ---", flush=True)
            try:
                regenerate_trial(nd, trial)
            except Exception as e:
                print(f"  [ERROR] {nd}d/t{trial}: {e}")
                failed.append(f"{nd}d/t{trial}")

    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED ({len(failed)}): {failed}")
    else:
        print("All sd3v trials regenerated successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()

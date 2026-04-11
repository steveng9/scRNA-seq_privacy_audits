"""
Full retrain of all ok_sd3v trials using the standard 1118-HVG set (ok/hvg.csv).

Previous runs used ok/hvg_full.csv (1141 HVGs), a slightly different gene set than
all other SDG methods.  This script:

  1. Clears existing synthetic.h5ad, models/*.rds, and tmp_sd3/ for each trial
  2. Rewrites the HVG-filtered train.h5ad using the 1118-HVG set
  3. Trains the vine copula from scratch with trunc_lvl=1
  4. Generates synthetic cells and saves synthetic.h5ad

After this script completes, ok_sd3v trials will use the same 1118-HVG gene set
as ok_sd3g, ok_scvi, ok_scdiff, and ok_dp.

Usage
-----
    nohup conda run --no-capture-output -n tabddpm_ \\
        python experiments/sdg_comparison/retrain_sd3v_1118hvg.py \\
        > /tmp/retrain_sd3v_1118.log 2>&1 &
    echo $! > /tmp/retrain_sd3v_1118.pid
    tail -f /tmp/retrain_sd3v_1118.log
"""

import os
import sys
import glob
import shutil

import numpy as np
import anndata as ad
import scanpy as sc
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC  = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

DATA = "/home/golobs/data"

OK_SD3V_SIZES = [10, 20, 50]
N_TRIALS      = 5

INDIVIDUAL_COL = "individual"
CELL_TYPE_COL  = "cell_type"

OK_DATASET = os.path.join(DATA, "ok", "full_dataset_cleaned.h5ad")
# Standard 1118-HVG set used by all other SDG methods
OK_HVG_PATH = os.path.join(DATA, "ok", "hvg.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_trial(out_dir):
    """Delete models, tmp_sd3, and synthetic.h5ad so everything is regenerated fresh."""
    synth = os.path.join(out_dir, "datasets", "synthetic.h5ad")
    if os.path.exists(synth):
        os.remove(synth)
        print(f"    Removed {synth}")

    models_dir = os.path.join(out_dir, "models")
    if os.path.isdir(models_dir):
        rds_files = glob.glob(os.path.join(models_dir, "*.rds"))
        mean_csv  = os.path.join(models_dir, "mean_expr.csv")
        for f in rds_files + ([mean_csv] if os.path.exists(mean_csv) else []):
            os.remove(f)
        print(f"    Removed {len(rds_files)} model .rds files from {models_dir}")

    tmp_dir = os.path.join(out_dir, "tmp_sd3")
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
        print(f"    Removed tmp_sd3/")


def _write_train_h5ad(dataset_path, train_donors, out_path, hvg_path):
    """Write HVG-filtered (1118-gene) train subset to disk."""
    adata  = sc.read_h5ad(dataset_path, backed="r")
    mask   = adata.obs[INDIVIDUAL_COL].isin(set(train_donors))
    sub    = adata[mask].to_memory()
    adata.file.close()

    hvg_df = pd.read_csv(hvg_path, index_col=0)
    hvgs   = set(hvg_df[hvg_df["highly_variable"]].index)
    keep   = [g for g in sub.var_names if g in hvgs]
    sub    = sub[:, keep].copy()
    sub.write_h5ad(out_path)
    n = sub.n_obs
    del sub
    return n


# ---------------------------------------------------------------------------
# Per-trial full retrain + generate
# ---------------------------------------------------------------------------

def retrain_trial(nd, trial):
    from sdg.scdesign3.model import ScDesign3

    out_dir    = os.path.join(DATA, "ok_sd3v", f"{nd}d", str(trial))
    ds_dir     = os.path.join(out_dir, "datasets")
    synth_out  = os.path.join(ds_dir, "synthetic.h5ad")
    models_dir = os.path.join(out_dir, "models")

    print(f"\n--- {nd}d/t{trial} ---", flush=True)

    _clear_trial(out_dir)

    # Copy split files from the base ok directory if missing
    splits_dir = os.path.join(DATA, "ok", f"{nd}d", str(trial), "datasets")
    os.makedirs(ds_dir, exist_ok=True)
    for fname in ["train.npy", "holdout.npy", "auxiliary.npy"]:
        src = os.path.join(splits_dir, fname)
        dst = os.path.join(ds_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    train_donors = np.load(os.path.join(ds_dir, "train.npy"), allow_pickle=True)

    # Write 1118-HVG train.h5ad
    train_h5ad = os.path.join(ds_dir, "train.h5ad")
    n_cells = _write_train_h5ad(OK_DATASET, train_donors, train_h5ad, OK_HVG_PATH)
    print(f"  {nd}d/t{trial}: {n_cells:,} cells, {OK_HVG_PATH}", flush=True)

    os.makedirs(models_dir, exist_ok=True)

    config = {
        "generator_name": "scdesign3",
        "dir_list": {
            "home": out_dir,
            "data": ds_dir,
        },
        "scdesign3_config": {
            "out_model_path":   "models",
            "hvg_path":         OK_HVG_PATH,
            "copula_type":      "vine",
            "family_use":       "nb",
            "trunc_lvl":        1,
            # Vine copulas are memory-intensive: 15 parallel R processes OOMs
            # for 50d (~64k cells).  4 workers keeps peak RSS well under 125GB.
            "parallel_workers": 4,
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

    print(f"  {nd}d/t{trial}: training vine copula (trunc_lvl=1, 1118 HVGs) ...", flush=True)
    model.train()

    print(f"  {nd}d/t{trial}: generating ...", flush=True)
    synth = model.generate()
    synth.write_h5ad(synth_out)
    print(f"  {nd}d/t{trial}: saved → {synth_out} ({synth.n_obs:,} cells, {synth.n_vars} genes)", flush=True)

    # Clean up tmp train.h5ad to save space
    if os.path.exists(train_h5ad):
        os.remove(train_h5ad)

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("sd3v full retrain with standard 1118-HVG set (ok/hvg.csv)")
    print(f"HVG file: {OK_HVG_PATH}")
    hvg_df = pd.read_csv(OK_HVG_PATH, index_col=0)
    n_hvgs = hvg_df["highly_variable"].sum()
    print(f"Number of HVGs: {n_hvgs}")
    print(f"Trials: {OK_SD3V_SIZES} × {N_TRIALS}")
    print("=" * 60, flush=True)

    failed = []
    for nd in OK_SD3V_SIZES:
        for trial in range(1, N_TRIALS + 1):
            out_dir = os.path.join(DATA, "ok_sd3v", f"{nd}d", str(trial))
            if not os.path.isdir(out_dir):
                print(f"  [SKIP] {nd}d/t{trial}: directory missing ({out_dir})")
                continue
            try:
                retrain_trial(nd, trial)
            except Exception as e:
                import traceback
                print(f"  [ERROR] {nd}d/t{trial}: {e}")
                traceback.print_exc()
                failed.append(f"{nd}d/t{trial}")

    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED ({len(failed)}): {failed}")
    else:
        print("All sd3v trials retrained with 1118-HVG set successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()

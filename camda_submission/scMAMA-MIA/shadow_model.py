"""
Shadow model: run scDesign2 on a given h5ad to extract a BB shadow copula.

Used by run_attack.py for the black-box (BB) attack variant: we treat the
synthetic data (or auxiliary data) as if it were training data and fit
scDesign2 on it to obtain shadow copulas.

Main function
-------------
fit_shadow_copulas(h5ad_path, output_dir, hvg_csv, cell_type_col, n_workers)
    → {cell_type: path_to_rds}
"""

import os
import subprocess
import tempfile
import shutil

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

from concurrent.futures import ProcessPoolExecutor, as_completed

_DIR       = os.path.dirname(os.path.abspath(__file__))
_R_SCRIPT  = os.path.join(_DIR, "scdesign2.r")


# ---------------------------------------------------------------------------
# Module-level R subprocess helper (picklable for multiprocessing)
# ---------------------------------------------------------------------------

def _run_train_ct(hvg_h5ad, cell_type, out_rds):
    os.makedirs(os.path.dirname(out_rds), exist_ok=True)
    cmd = f"Rscript {_R_SCRIPT} train {hvg_h5ad} {cell_type!r} {out_rds}"
    try:
        subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        return cell_type, True
    except subprocess.CalledProcessError as e:
        return cell_type, False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_shadow_copulas(h5ad_path, output_dir, hvg_csv=None,
                        cell_type_col="cell_type", n_workers=4):
    """
    Fit one scDesign2 copula per cell type in h5ad_path.

    Parameters
    ----------
    h5ad_path     : str  — path to the input AnnData (.h5ad)
    output_dir    : str  — directory where {cell_type}.rds files are saved
    hvg_csv       : str or None — CSV with a 'highly_variable' column; if None,
                    HVGs are computed from h5ad_path using standard parameters
    cell_type_col : str  — obs column with cell-type labels (default: "cell_type")
    n_workers     : int  — parallel R processes (default: 4)

    Returns
    -------
    dict mapping cell_type → path_to_rds (only for successfully trained cell types)
    """
    import multiprocessing

    os.makedirs(output_dir, exist_ok=True)

    adata = ad.read_h5ad(h5ad_path)
    hvg_mask = _get_hvg_mask(adata, hvg_csv)
    cell_types = list(adata.obs[cell_type_col].unique())

    tmp_dir  = tempfile.mkdtemp(dir=output_dir, prefix="shadow_tmp_")
    hvg_h5ad = os.path.join(tmp_dir, "hvg_subset.h5ad")
    try:
        adata[:, hvg_mask].copy().write_h5ad(hvg_h5ad)
        del adata

        copula_paths = {}
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as exe:
            futures = {}
            for ct in cell_types:
                out_rds = os.path.join(output_dir, f"{ct}.rds")
                if os.path.exists(out_rds):
                    copula_paths[ct] = out_rds
                    continue
                futures[exe.submit(_run_train_ct, hvg_h5ad, ct, out_rds)] = ct
            for fut in as_completed(futures):
                ct, ok = fut.result()
                if ok:
                    copula_paths[ct] = os.path.join(output_dir, f"{ct}.rds")
                    print(f"  [shadow] Fitted copula: {ct}", flush=True)
                else:
                    print(f"  [shadow] FAILED: {ct}", flush=True)

        return copula_paths

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# HVG helper
# ---------------------------------------------------------------------------

def _get_hvg_mask(adata, hvg_csv):
    """Return a boolean array of length n_vars indicating HVG membership."""
    if hvg_csv and os.path.exists(hvg_csv):
        hvg_df = pd.read_csv(hvg_csv, index_col=0)
        if len(hvg_df) != len(adata.var_names):
            hvg_df = hvg_df.reindex(adata.var_names).fillna(False)
        return hvg_df["highly_variable"].values.astype(bool)

    # Compute from data using standard scMAMA-MIA parameters
    tmp = adata.copy()
    if "counts" not in tmp.layers:
        tmp.layers["counts"] = tmp.X.copy()
    sc.pp.normalize_total(tmp, layer="counts", target_sum=1e4)
    sc.pp.log1p(tmp, layer="counts")
    sc.pp.highly_variable_genes(tmp, layer="counts",
                                 min_mean=0.0125, max_mean=3, min_disp=0.5)
    mask = tmp.var["highly_variable"].values.astype(bool)
    print(f"  [shadow] Computed {mask.sum()} HVGs from {adata.n_vars} genes", flush=True)
    return mask

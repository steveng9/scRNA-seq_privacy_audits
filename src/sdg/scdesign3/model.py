"""
scDesign3 synthetic data generator (Python wrapper around the R implementation).

Trains a per-cell-type Gaussian (or vine) copula via parallel R subprocesses,
then generates synthetic counts by sampling from those copulas.

The R driver is src/sdg/scdesign3/scdesign3.r.
Copula parsing for the attack is in src/sdg/scdesign3/copula.py.
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import anndata as ad
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any

# Absolute path to the R script — CWD-independent
_R_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scdesign3.r")

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from sdg.base import BaseSingleCellDataGenerator


# ---------------------------------------------------------------------------
# Per-process R subprocess helpers (module-level so they are picklable)
# ---------------------------------------------------------------------------

def _run_train(home_dir, hvg_subset_path, cell_type, out_model_path,
               family_use, copula_type, trunc_lvl):
    """Fit a copula for one cell type via Rscript."""
    model_path = os.path.join(home_dir, out_model_path, f"{cell_type}.rds")
    # Cell type names may contain spaces — quote the argument
    cmd = (
        f'Rscript {_R_SCRIPT} train '
        f'{hvg_subset_path} "{cell_type}" {model_path} '
        f'{family_use} {copula_type} {trunc_lvl}'
    )
    try:
        subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        return cell_type
    except subprocess.CalledProcessError as e:
        output = e.output.decode("utf-8", errors="replace")
        print(f"[SKIP] scDesign3 training failed for '{cell_type}':\n{output[-800:]}")
        return None


def _run_generate(home_dir, tmp_dir, out_model_path, cell_type, num_to_gen):
    """Sample synthetic counts for one cell type via Rscript."""
    model_path = os.path.join(home_dir, out_model_path, f"{cell_type}.rds")
    out_path   = os.path.join(tmp_dir, f"out_{cell_type}.rds")
    cmd = f'Rscript {_R_SCRIPT} gen {num_to_gen} {model_path} {out_path}'
    try:
        subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        output = e.output.decode("utf-8", errors="replace")
        print(f"[WARN] scDesign3 generation failed for '{cell_type}': {output[:300]}")
    return cell_type


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class ScDesign3(BaseSingleCellDataGenerator):
    """
    Fits a per-cell-type Gaussian (default) or vine copula with NB/ZINB marginals
    (via R's scDesign3 package) and samples synthetic scRNA-seq counts.

    Training and generation are parallelised across cell types.

    Config keys (under scdesign3_config):
        out_model_path  : relative path (from home_dir) for .rds model files
        hvg_path        : path to/from which the HVG mask CSV is read/written
        copula_type     : "gaussian" (default) or "vine"
        family_use      : "nb" (default), "zinb", "poisson", or "gaussian"
        trunc_lvl       : vine truncation level — integer or "Inf" (default "Inf")
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cell_type_col = self.dataset_config["cell_type_col_name"]
        self.tmp_dir    = os.path.join(self.home_dir, "tmp_sd3")
        self.means_path = os.path.join(
            self.home_dir, self.generator_config["out_model_path"], "mean_expr.csv"
        )
        self.hvg_path     = self.generator_config["hvg_path"]
        self.copula_type  = self.generator_config.get("copula_type", "gaussian")
        self.family_use   = self.generator_config.get("family_use", "nb")
        self.trunc_lvl    = self.generator_config.get("trunc_lvl", "Inf")
        self.mean_expression = None
        self.hvg_mask        = None
        os.makedirs(self.tmp_dir, exist_ok=True)

    # ------------------------------------------------------------------

    def train(self):
        X_train = self.load_train_anndata()
        print("Train dims:", X_train.X.shape)

        cell_types = X_train.obs[self.cell_type_col].values
        cell_type_counts = Counter(cell_types)

        # Compute mean expression per cell type
        X_dense = X_train.X.toarray() if not isinstance(X_train.X, np.ndarray) else X_train.X
        self.mean_expression = (
            pd.DataFrame(X_dense, columns=X_train.var_names, index=X_train.obs_names)
            .groupby(X_train.obs[self.cell_type_col], observed=False)
            .mean()
            .T
        )
        os.makedirs(os.path.dirname(self.means_path), exist_ok=True)
        self.mean_expression.to_csv(self.means_path, index=True)

        # HVG selection — scDesign3 uses the same HVG subset as scDesign2
        X_train.layers["counts"] = X_train.X.copy()
        sc.pp.normalize_total(X_train, layer="counts", target_sum=1e4)
        sc.pp.log1p(X_train, layer="counts")

        if not os.path.exists(self.hvg_path):
            sc.pp.highly_variable_genes(
                X_train, layer="counts",
                min_mean=0.0125, max_mean=3, min_disp=0.5
            )
            self.hvg_mask = X_train.var["highly_variable"]
            self.hvg_mask.to_csv(self.hvg_path)
            print(f"{self.hvg_mask.sum()} HVGs determined")
        else:
            self.hvg_mask = np.array(pd.read_csv(self.hvg_path)["highly_variable"])
            print(f"{self.hvg_mask.sum()} HVGs found")

        hvg_subset_path = os.path.join(self.tmp_dir, "hvg_train.h5ad")
        n_hvg = int(self.hvg_mask.sum()) if hasattr(self.hvg_mask, "sum") else sum(self.hvg_mask)
        if X_train.n_vars == n_hvg:
            # train.h5ad was already pre-filtered to HVGs — skip boolean indexing
            print("  train.h5ad already HVG-filtered; skipping subsetting step.", flush=True)
            X_train.copy().write(hvg_subset_path)
        else:
            X_train[:, self.hvg_mask].copy().write(hvg_subset_path)

        out_model_path = self.generator_config["out_model_path"]
        os.makedirs(os.path.join(self.home_dir, out_model_path), exist_ok=True)

        print(f"Launching parallel scDesign3 training ({self.copula_type} copula) ...")
        with ProcessPoolExecutor(max_workers=15) as executor:
            futures = [
                executor.submit(
                    _run_train, self.home_dir, hvg_subset_path, ct, out_model_path,
                    self.family_use, self.copula_type, self.trunc_lvl
                )
                for ct in cell_type_counts.keys()
            ]
            for f in as_completed(futures):
                result = f.result()
                if result is not None:
                    print(f"  Finished training cell type: {result}")
                else:
                    print(f"  A cell type failed — skipped.")
        print("Training complete.")

    # ------------------------------------------------------------------

    def generate(self) -> ad.AnnData:
        from rpy2.robjects import r as R

        if self.mean_expression is None:
            self.mean_expression = pd.read_csv(self.means_path, index_col=0)
        if self.hvg_mask is None:
            self.hvg_mask = np.array(pd.read_csv(self.hvg_path)["highly_variable"])

        X_train = self.load_train_anndata()
        cell_types = X_train.obs[self.cell_type_col].values
        cell_type_counts = Counter(cell_types)

        out_model_path = self.generator_config["out_model_path"]

        X_test = self.load_test_anndata()
        test_cell_types = X_test.obs[self.cell_type_col].values

        print("Launching parallel scDesign3 generation ...")
        with ProcessPoolExecutor(max_workers=15) as executor:
            futures = []
            for ct in cell_type_counts.keys():
                n = int((test_cell_types == ct).sum())
                futures.append(
                    executor.submit(
                        _run_generate, self.home_dir, self.tmp_dir, out_model_path, ct, n
                    )
                )
            for f in as_completed(futures):
                print(f"  Generated cell type: {f.result()}")

        print("Assembling synthetic AnnData ...")
        total_counts = X_test.X.toarray() if not isinstance(X_test.X, np.ndarray) else X_test.X
        synthetic_counts = sp.lil_matrix(total_counts.shape, dtype=np.float64)

        hvg_indices = np.where(self.hvg_mask)[0]

        for ct in cell_type_counts.keys():
            out_path = os.path.join(self.tmp_dir, f"out_{ct}.rds")
            try:
                # Use rpy2 to read the matrix (genes × cells) saved by simu_new
                counts_r = R["readRDS"](out_path)
                counts_np = np.array(counts_r)   # shape: (n_genes, n_cells)
                cell_indices = np.where(test_cell_types == ct)[0]
                n_cells = min(len(cell_indices), counts_np.shape[1])
                for i in range(n_cells):
                    row_idx = cell_indices[i]
                    gene_vals = counts_np[:, i]   # genes for cell i
                    n_assign = min(len(hvg_indices), len(gene_vals))
                    synthetic_counts[row_idx, hvg_indices[:n_assign]] = gene_vals[:n_assign]
            except Exception as e:
                print(f"[SKIP] Could not read {out_path}: {e}")
                continue

        synthetic_adata = ad.AnnData(X=synthetic_counts.tocsr())
        synthetic_adata.obs[self.cell_type_col] = test_cell_types
        synthetic_adata.var_names = X_test.var_names
        print("Synthetic AnnData ready.")
        return synthetic_adata

    # ------------------------------------------------------------------

    def load_from_checkpoint(self):
        pass

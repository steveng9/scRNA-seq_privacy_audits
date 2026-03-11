"""
scDesign2 synthetic data generator (Python wrapper around the R implementation).

Trains a per-cell-type Gaussian copula via parallel R subprocesses, then
generates synthetic counts by sampling from those copulas.
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
_R_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scdesign2.r")

# Ensure src/ is on the path so sibling packages resolve correctly
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from sdg.base import BaseSingleCellDataGenerator


# ---------------------------------------------------------------------------
# Per-process R subprocess helpers (module-level so they are picklable)
# ---------------------------------------------------------------------------

def _run_train(home_dir, hvg_subset_path, cell_type, out_model_path):
    """Fit a Gaussian copula for one cell type via Rscript."""
    copula_path = os.path.join(home_dir, out_model_path, f"{cell_type}.rds")
    cmd = f"Rscript {_R_SCRIPT} train {hvg_subset_path} {cell_type} {copula_path}"
    try:
        subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        return cell_type
    except subprocess.CalledProcessError as e:
        print(f"[SKIP] Training failed for '{cell_type}': {e.output[:200]}")
        return None


def _run_generate(home_dir, tmp_dir, out_model_path, cell_type, num_to_gen):
    """Sample synthetic counts for one cell type via Rscript."""
    copula_path = os.path.join(home_dir, out_model_path, f"{cell_type}.rds")
    out_path = os.path.join(tmp_dir, f"out{cell_type}.rds")
    cmd = f"Rscript {_R_SCRIPT} gen {num_to_gen} {copula_path} {out_path}"
    try:
        subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Generation failed for '{cell_type}': {e.output[:200]}")
    return cell_type


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class ScDesign2(BaseSingleCellDataGenerator):
    """
    Fits a per-cell-type Gaussian copula with ZINB/NB/Poisson marginals
    (via R's scDesign2 package) and samples synthetic scRNA-seq counts.

    Training and generation are parallelised across cell types.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cell_type_col = self.dataset_config["cell_type_col_name"]
        self.tmp_dir = os.path.join(self.home_dir, "tmp")
        self.means_path = os.path.join(
            self.home_dir, self.generator_config["out_model_path"], "mean_expr.csv"
        )
        self.hvg_path = self.generator_config["hvg_path"]
        self.mean_expression = None
        self.hvg_mask = None
        os.makedirs(self.tmp_dir, exist_ok=True)

    # ------------------------------------------------------------------

    def train(self):
        import pyreadr  # noqa: F401 — imported here to keep optional

        X_train = self.load_train_anndata()
        print("Train dims:", X_train.X.shape)

        cell_types = X_train.obs[self.cell_type_col].values
        cell_type_counts = Counter(cell_types)

        # Compute mean expression per cell type without densifying the full matrix.
        # Calling .toarray() on large datasets (e.g. 200K cells × 35K genes) would
        # create a ~30+ GB dense array and OOM-kill the process.
        import scipy.sparse as sp
        X_sp = X_train.X if sp.issparse(X_train.X) else sp.csr_matrix(X_train.X)
        unique_cts = list(cell_type_counts.keys())
        ct_labels = X_train.obs[self.cell_type_col].values
        means_dict = {
            ct: np.asarray(X_sp[ct_labels == ct].mean(axis=0)).flatten()
            for ct in unique_cts
        }
        self.mean_expression = pd.DataFrame(means_dict, index=X_train.var_names).T
        self.mean_expression.to_csv(self.means_path, index=True)

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
        X_train[:, self.hvg_mask].copy().write(hvg_subset_path)

        # Release X_train before forking worker processes to avoid CoW overhead
        # multiplying the ~11 GB in-memory AnnData across all workers.
        del X_train
        import gc; gc.collect()

        out_model_path = self.generator_config["out_model_path"]
        import multiprocessing
        print("Launching parallel training processes...")
        ctx = multiprocessing.get_context("spawn")  # avoids inheriting parent's memory
        with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as executor:
            futures = [
                executor.submit(_run_train, self.home_dir, hvg_subset_path, ct, out_model_path)
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
        import pyreadr

        if self.mean_expression is None:
            self.mean_expression = pd.read_csv(self.means_path, index_col=0)
        if self.hvg_mask is None:
            self.hvg_mask = np.array(pd.read_csv(self.hvg_path)["highly_variable"])

        X_train = self.load_train_anndata()
        cell_types = X_train.obs[self.cell_type_col].values
        cell_type_counts = Counter(cell_types)

        out_model_path = self.generator_config["out_model_path"]

        print("Launching parallel generation processes...")
        X_test = self.load_test_anndata()
        test_cell_types = X_test.obs[self.cell_type_col].values
        with ProcessPoolExecutor(max_workers=15) as executor:
            futures = []
            for ct in cell_type_counts.keys():
                n = int((test_cell_types == ct).sum())
                futures.append(
                    executor.submit(_run_generate, self.home_dir, self.tmp_dir, out_model_path, ct, n)
                )
            for f in as_completed(futures):
                print(f"  Generated cell type: {f.result()}")

        print("Assembling synthetic AnnData...")
        total_counts = X_test.X.toarray()
        synthetic_counts = sp.lil_matrix(total_counts.shape, dtype=np.int64)

        for ct in cell_type_counts.keys():
            out_path = os.path.join(self.tmp_dir, f"out{ct}.rds")
            try:
                r_matrix = list(pyreadr.read_r(out_path).values())[0]
                counts_np = r_matrix.to_numpy() if hasattr(r_matrix, "to_numpy") else np.array(r_matrix)
                cell_indices = np.where(test_cell_types == ct)[0]
                for i, row_idx in enumerate(cell_indices):
                    try:
                        synthetic_counts[row_idx, np.asarray(self.hvg_mask)] = counts_np[:, i]
                    except Exception as e:
                        print(f"[WARN] row assignment error for {ct}: {e}")
            except Exception as e:
                print(f"[SKIP] Could not read {out_path}: {e}")
                continue

        synthetic_adata = ad.AnnData(X=synthetic_counts.tocsr().astype(np.float64))
        synthetic_adata.obs[self.cell_type_col] = test_cell_types
        synthetic_adata.var_names = X_test.var_names
        print("Synthetic AnnData ready.")
        return synthetic_adata

    # ------------------------------------------------------------------

    def load_from_checkpoint(self):
        pass

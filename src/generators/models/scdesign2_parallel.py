import os
import sys
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import anndata as ad
from collections import Counter
from typing import Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyreadr
import rpy2.robjects as robjects

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)

from src.generators.models.sc_base import BaseSingleCellDataGenerator


def run_rscript_train(home_dir, hvg_subset_path, cell_type, out_model_path):
    """Run one training subprocess for a given cell type."""
    copula_path = os.path.join(home_dir, out_model_path, f"{cell_type}.rds")
    cmd = f"Rscript models/scdesign2.r train {hvg_subset_path} {cell_type} {copula_path}"
    os.system(cmd)
    return cell_type


def run_rscript_generate(home_dir, tmp_dir, out_model_path, cell_type, num_to_gen):
    """Run one generation subprocess for a given cell type."""
    copula_path = os.path.join(home_dir, out_model_path, f"{cell_type}.rds")
    out_path = os.path.join(tmp_dir, f"out{cell_type}.rds")
    cmd = f"Rscript models/scdesign2.r gen {num_to_gen} {copula_path} {out_path}"
    os.system(cmd)
    return cell_type


class ScDesign2GeneratorParallel(BaseSingleCellDataGenerator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cell_type_col_name = self.dataset_config["cell_type_col_name"]
        self.tmp_dir = os.path.join(self.home_dir, "tmp")

        self.means_path = os.path.join(self.home_dir, self.generator_config["out_model_path"], "mean_expr.csv")
        self.mean_expression = None

        self.hvg_mask = None
        self.hvg_path = os.path.join(self.home_dir, self.generator_config["out_model_path"], "hvg.csv")

        os.makedirs(self.tmp_dir, exist_ok=True)

    def initialize_random_seeds(self):
        np.random.seed(self.random_seed)

    def train(self):
        X_train_adata = self.load_train_anndata()
        print("Train dims:", X_train_adata.X.shape)

        cell_types = X_train_adata.obs[self.cell_type_col_name].values
        cell_types_dist = Counter(cell_types)
        print("Num cell types:", len(cell_types_dist))

        # compute mean expression
        print("Calculating means")
        self.mean_expression = pd.DataFrame(
            X_train_adata.X.toarray() if not isinstance(X_train_adata.X, np.ndarray) else X_train_adata.X,
            columns=X_train_adata.var_names,
            index=X_train_adata.obs_names
        ).groupby(X_train_adata.obs[self.cell_type_col_name]).mean().T
        self.mean_expression.to_csv(self.means_path, index=True)

        print("Copying counts matrix")
        X_train_adata.layers["counts"] = X_train_adata.X.copy()

        # Determine HVGs
        print("Determining HVGs")
        sc.pp.normalize_total(X_train_adata, layer="counts", target_sum=1e4)
        sc.pp.log1p(X_train_adata, layer="counts")
        sc.pp.highly_variable_genes(X_train_adata, layer="counts", min_mean=0.0125, max_mean=3, min_disp=0.5)
        self.hvg_mask = X_train_adata.var['highly_variable']
        self.hvg_mask.to_csv(self.hvg_path)

        print(f"Detected {self.hvg_mask.sum()} HVGs")
        hvg_subset_path = os.path.join(self.tmp_dir, "hvg_train.h5ad")
        X_train_adata_hvg = X_train_adata[:, self.hvg_mask].copy()
        X_train_adata_hvg.write(hvg_subset_path)

        # Parallelize per-cell-type training
        out_model_path = self.generator_config["out_model_path"]
        futures = []
        print("Launching parallel training processes...")
        with ProcessPoolExecutor(max_workers=15) as executor:
            for cell_type in cell_types_dist.keys():
                futures.append(executor.submit(run_rscript_train, self.home_dir, hvg_subset_path, cell_type, out_model_path))

            for f in as_completed(futures):
                print(f"Finished training cell type {f.result()}")

        print("Training completed successfully!")

    def generate(self):
        print("Generating synthetic data...")
        if self.mean_expression is None:
            self.mean_expression = pd.read_csv(self.means_path, index_col=0)
        if self.hvg_mask is None:
            hvg_mask_raw = pd.read_csv(self.hvg_path, index_col=0)
            self.hvg_mask = hvg_mask_raw['highly_variable']

        X_train_adata = self.load_train_anndata()
        cell_types = X_train_adata.obs[self.cell_type_col_name].values
        cell_types_dist = Counter(cell_types)

        avail_models = []
        total_cells = 0
        for cell_type, num in cell_types_dist.items():
            copula_path = os.path.join(self.home_dir, self.generator_config["out_model_path"], f"{cell_type}.rds")
            if os.path.exists(copula_path):
                avail_models.append(cell_type)
                total_cells += num
            else:
                print(f"{copula_path} is missing")
                exit()

        X_test_adata = self.load_test_anndata()
        total_counts = X_test_adata.X.toarray()
        counts = total_counts

        print("Launching parallel generation processes...")
        out_model_path = self.generator_config["out_model_path"]
        futures = []
        cell_types = X_test_adata.obs[self.cell_type_col_name].values
        with ProcessPoolExecutor(max_workers=15) as executor:
            for cell_type in cell_types_dist.keys():
                cell_type_mask = cell_types == cell_type
                cell_indices = np.where(cell_type_mask)[0]
                num_to_gen = len(cell_indices)
                futures.append(executor.submit(run_rscript_generate, self.home_dir, self.tmp_dir, out_model_path, cell_type, num_to_gen))

            for f in as_completed(futures):
                print(f"Finished generating {f.result()}")

        print("Reading generated RDS files...")
        synthetic_counts = sp.lil_matrix(counts.shape, dtype=np.int64)

        for cell_type in cell_types_dist.keys():
            out_path = os.path.join(self.tmp_dir, f"out{cell_type}.rds")
            counts_res = pyreadr.read_r(out_path)
            r_matrix = list(counts_res.values())[0]
            counts_np_array = r_matrix.to_numpy() if hasattr(r_matrix, "to_numpy") else np.array(r_matrix)

            cell_type_mask = cell_types == cell_type
            cell_indices = np.where(cell_type_mask)[0]
            for i, row_idx in enumerate(cell_indices):
                synthetic_counts[row_idx, self.hvg_mask.values] = counts_np_array[:, i]

        synthetic_counts_csr = synthetic_counts.tocsr().astype(np.float64)
        synthetic_adata = ad.AnnData(X=synthetic_counts_csr)
        synthetic_adata.obs[self.cell_type_col_name] = cell_types
        synthetic_adata.var_names = X_test_adata.var_names

        print("Returning synthetic AnnData object")
        return synthetic_adata

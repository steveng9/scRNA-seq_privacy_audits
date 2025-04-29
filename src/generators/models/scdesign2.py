import os
import sys
import pandas as pd
import numpy as np
import scipy.stats as st
import scipy.sparse as sp
import scanpy as sc
import anndata as ad
import math
import pyreadr
from collections import Counter
from typing import Dict, Any

import rpy2.robjects as robjects

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)

from generators.models.sc_base import BaseSingleCellDataGenerator

class ScDesign2Generator(BaseSingleCellDataGenerator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cell_type_col_name = self.dataset_config["cell_type_col_name"]
        self.cell_label_col_name = self.dataset_config["cell_label_col_name"]
        self.tmp_dir = os.path.join(self.home_dir, "tmp")

        self.means_path = os.path.join(self.home_dir, self.generator_config["out_model_path"], "mean_expr.csv")
        self.mean_expression = None

        self.hvg_mask = None
        self.hvg_path = os.path.join(self.home_dir, self.generator_config["out_model_path"], "hvg.csv")

        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir, exist_ok=True)

        #print("Importing scdesign2 R code")
        #robjects.r['source']('src/generators/models/scdesign2.r')
        #print("Import succeeded")

    def initialize_random_seeds(self):
        np.random.seed(self.random_seed)

    def train(self):
        """Compute gene expression parameters for each cell type from training data."""
        X_train_adata = self.load_train_anndata()
        cell_types = X_train_adata.obs[self.cell_type_col_name].values
        cell_types_dist = Counter(cell_types)

        genes = X_train_adata.var.gene_ids
        cell_ids = X_train_adata.obs.cell_label
        cell_type_vector = X_train_adata.obs.cell_type
        counts = X_train_adata.X
        assert counts.shape[0] == len(cell_ids)

        print("Calculating means")
        self.mean_expression = pd.DataFrame(
            X_train_adata.X.toarray() if not isinstance(X_train_adata.X, np.ndarray) else X_train_adata.X,
            columns=X_train_adata.var_names,
            index=X_train_adata.obs_names
        ).groupby(X_train_adata.obs['cell_type']).mean().T
        self.mean_expression.to_csv(self.means_path, index=True)
        
        sc.pp.normalize_total(X_train_adata, target_sum=1e4)
        sc.pp.log1p(X_train_adata)
        sc.pp.highly_variable_genes(X_train_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

        hvg_df = X_train_adata.var[X_train_adata.var['highly_variable']]
        hvg_df = hvg_df.copy()
        hvg_df['gene'] = hvg_df.index
        print(f"Detected {len(X_train_adata.var[X_train_adata.var['highly_variable']])} HVGs")

        self.hvg_mask = X_train_adata.var['highly_variable']
        self.hvg_mask.to_csv(self.hvg_path)

        hvg_subset_path = os.path.join(self.tmp_dir, "hvg_train.h5ad")
        X_train_adata_hvg = X_train_adata[:, self.hvg_mask].copy()
        X_train_adata_hvg.write(hvg_subset_path)

        avail_models = []
        total_cells = 0
        for cell_type, num in cell_types_dist.items():
            copula_path = os.path.join(self.home_dir, self.generator_config["out_model_path"], f"{cell_type}.rds")
            print(copula_path)
            if not os.path.exists(copula_path):
                print(f"Training cell type {cell_type}")
                os.system(f"Rscript src/generators/models/scdesign2.r train {hvg_subset_path} {cell_type} {copula_path} > /dev/null 2>&1")

        print("Training completed successfully!")

    def generate(self):
        if self.mean_expression is None:
            self.mean_expression = pd.read_csv(self.means_path, index_col=0)
            self.mean_expression.columns = [int(col) if col.isdigit() else col for col in self.mean_expression.columns]
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
        total_counts = X_test_adata.X.toarray() if isinstance(X_test_adata.X, np.ndarray) else X_test_adata.X.A
        counts = total_counts
        num_test_cells = counts.shape[0]

        synthetic_cell_types = []
        synthetic_counts = sp.lil_matrix(counts.shape, dtype=np.int64)
        print(f"COUNTS MAT SHAPE: {synthetic_counts.shape}")
        cell_types = X_test_adata.obs[self.cell_type_col_name].values
        for cell_type in cell_types_dist.keys():
            print(f"Generating for {cell_type}")
            copula_path = os.path.join(self.home_dir, self.generator_config["out_model_path"], f"{cell_type}.rds")
            cell_type_mask = cell_types == cell_type
            cell_indices = np.where(cell_type_mask)[0]
            num_to_gen = len(cell_indices)

            out_path = f"out{cell_type}.rds"
            print(f"Rscript src/generators/models/scdesign2.r gen {num_to_gen} {copula_path} {out_path}")
            os.system(f"Rscript src/generators/models/scdesign2.r gen {num_to_gen} {copula_path} {out_path} > /dev/null 2>&1")

            counts_res = pyreadr.read_r(out_path)
            r_matrix = list(counts_res.values())[0]
            counts_np_array = r_matrix.to_numpy() if hasattr(r_matrix, "to_numpy") else np.array(r_matrix)
            print(counts_np_array.shape)
            #synthetic_counts[cell_indices, :] = self.mean_expression[cell_type]

            #print("HVG MASK")
            #print(self.hvg_mask)
            #print(self.hvg_mask.shape)

            #print(synthetic_counts[cell_indices][:, self.hvg_mask].shape)
            #for i, row_idx in enumerate(cell_indices):
            #    a = synthetic_counts[row_idx, self.hvg_mask]
            #    #print(i, row_idx)
            #    synthetic_counts[row_idx, self.hvg_mask] = cell_type_synth[i].X.toarray().astype(float).flatten()
            for i, row_idx in enumerate(cell_indices):
                synthetic_counts[row_idx, self.hvg_mask] = counts_np_array[:, i]
            #synthetic_counts[cell_indices][:, self.hvg_mask] = np.transpose(counts_np_array)
            #synthetic_cell_types.extend([cell_type] * num_to_gen)

        synthetic_counts_csr = synthetic_counts.tocsr().astype(np.float64)
        synthetic_adata = ad.AnnData(X=synthetic_counts_csr)
        #synthetic_adata.obs[self.cell_type_col_name] = synthetic_cell_types
        synthetic_adata.obs[self.cell_type_col_name] = cell_types
        synthetic_adata.var_names = X_test_adata.var_names

        print("RETURNING")
        return synthetic_adata

        exit(1)

        counts = X_test_adata.X.toarray() if isinstance(X_test_adata.X, np.ndarray) else X_test_adata.X.A
        print("Original counts shape:", counts.shape)
        copula_path = os.path.join(self.home_dir, self.generator_config["out_model_path"], "copula.rds")
        if not os.path.exists(copula_path):
            raise ValueError("Training has not yet completed")

        test_data_path = os.path.join(self.home_dir, self.dataset_config["test_count_file"])
        out_path = "out.h5ad"
        os.system(f"Rscript src/generators/models/scdesign2.r gen {test_data_path} {copula_path} {out_path}")
        exit(1)
        #gen_synth_data(test_data_path, copula_path, out_path)

        if self.max_real_value is None:
            raise ValueError("Training must be completed before generating data!")

        X_test_adata = self.load_test_anndata()
        counts = X_test_adata.X.toarray() if isinstance(X_test_adata.X, np.ndarray) else X_test_adata.X.A
        print("Original counts shape:", counts.shape)

        cell_types = X_test_adata.obs[self.cell_type_col_name].values
        synthetic_counts = sp.lil_matrix(counts.shape, dtype=np.int64)
        synthetic_cell_types = []

        for cell_type in np.unique(cell_types):
            print(f"Generating for Cell Type: {cell_type}")

            if str(cell_type) not in self.cell_type_params:
                print(f"Cell type {cell_type} not found in training data! Skipping...")
                continue

            cell_type_mask = cell_types == cell_type
            cell_indices = np.where(cell_type_mask)[0]
            num_cells = len(cell_indices)

            means = self.cell_type_params[str(cell_type)]['means'].astype(np.float64)
            means = np.clip(means, 1e-6, None)  # Avoid zeros

            if self.distribution == 'NB':
                dispersions = self.cell_type_params[str(cell_type)]['dispersions'].astype(np.float64)
                dispersions = np.clip(dispersions, 1e-3, 10)  # Prevent extreme values

                # Compute Negative Binomial parameters
                n_param = np.clip(1 / (dispersions + 1e-6), 1e-2, 10) 
                p_param = np.clip(means / (means + n_param), 0.01, 0.99)  

                # Debugging prints
                print(f"n_param range for {cell_type}: min={n_param.min()}, max={n_param.max()}")
                print(f"p_param range for {cell_type}: min={p_param.min()}, max={p_param.max()}")

                expected_variance = means + (means ** 2) / n_param
                print(f"Expected variance for {cell_type}: min={expected_variance.min()}, max={expected_variance.max()}")

                # Generate Negative Binomial samples
                generated_data = st.nbinom.rvs(n=n_param, p=p_param, size=(num_cells, means.shape[0])).astype(np.int64)

            elif self.distribution == 'Poisson':
                generated_data = st.poisson.rvs(means, size=(num_cells, means.shape[0])).astype(np.int64)

            # Limit extreme values to prevent memory explosion
            upper_clip = np.percentile(generated_data, 99.5)
            generated_data = np.clip(generated_data, 0, min(upper_clip, self.max_real_value * 2))

            # Store generated data
            synthetic_counts[cell_indices, :] = generated_data
            synthetic_cell_types.extend([cell_type] * num_cells)

        synthetic_counts_csr = synthetic_counts.tocsr().astype(np.int64)
        synthetic_adata = ad.AnnData(X=synthetic_counts_csr)
        synthetic_adata.obs[self.cell_type_col_name] = synthetic_cell_types
        synthetic_adata.var_names = X_test_adata.var_names

        return synthetic_adata


    def load_from_checkpoint(self):
        pass


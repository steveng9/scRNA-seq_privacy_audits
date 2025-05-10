import os
import pickle
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

def merge_sparse_matrices(mat1, mat2, selector):
    # There are smarter ways to do this, but this is simple.
    mat1 = mat1.tocsc()
    mat2 = mat2.tocsc()

    assert mat1.shape[0] == mat2.shape[0], "Row mismatch"
    cols = []
    ptr1, ptr2 = 0, 0

    for i, sel in enumerate(selector):
        if sel:
            cols.append(mat1[:, i])
        else:
            cols.append(mat2[:, i])

    merged = sp.hstack(cols)
    return merged.tolil()


class ScDesign2EnsembleGenerator(BaseSingleCellDataGenerator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cell_type_col_name = self.dataset_config["cell_type_col_name"]
        self.cell_label_col_name = self.dataset_config["cell_label_col_name"]
        self.tmp_dir = os.path.join(self.home_dir, "tmp")

        self.means_path = os.path.join(self.home_dir, self.generator_config["out_model_path"], "mean_expr.csv")
        self.mean_expression = None

        self.hvg_mask = None
        self.hvg_path = os.path.join(self.home_dir, self.generator_config["out_model_path"], "hvg.csv")

        self.dist_path = os.path.join(self.home_dir, self.generator_config["out_model_path"], "distribution.pkl")
        self.max_real_val_path = os.path.join(self.home_dir, self.generator_config["out_model_path"], "max_real_value.npy")

        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir, exist_ok=True)

        self.noise_level = self.generator_config["noise_level"]
        self.random_seed = self.generator_config["random_seed"]
        self.distribution = self.generator_config["distribution"]
        self.max_real_value = None

        self.gene_means = None
        self.num_samples = None
        self.X_train_features = None
        self.cell_type_params = {}

        self.initialize_random_seeds()


    def initialize_random_seeds(self):
        np.random.seed(self.random_seed)

    def train(self):
        self.train_distribution()
        self.train_scdesign2()

    def train_distribution(self):
        """Compute gene expression parameters for each cell type from training data."""
        if os.path.exists(self.dist_path) and os.path.exists(self.max_real_val_path):
            print("Distributions already trained, skipping training")
            return

        print("INSIDE TRAIN DISTRIBUTION")
        X_train_adata = self.load_train_anndata()

        counts = X_train_adata.X.toarray() if isinstance(X_train_adata.X, np.ndarray) else X_train_adata.X.A
        cell_types = X_train_adata.obs[self.cell_type_col_name].values
        cell_labels = X_train_adata.obs[self.cell_label_col_name].values

        self.cell_type_to_label = dict(set(zip(cell_types, cell_labels)))

        print("Cell Type to Label Mapping:", self.cell_type_to_label)

        # Store the max gene expression value from training
        self.max_real_value = counts.max()
        print(f"Max real expression value from training: {self.max_real_value}")

        for cell_type in np.unique(cell_types):
            print(f"Training on Cell Type: {cell_type}")

            cell_type_mask = cell_types == cell_type
            cell_type_counts = counts[cell_type_mask, :]

            means = cell_type_counts.mean(axis=0)
            means = np.clip(means, 1e-6, None)  # Avoid zero means

            if self.distribution == 'NB':
                variances = cell_type_counts.var(axis=0)

                # variance >= mean to prevent negative dispersions
                variances = np.maximum(variances, means)

                dispersions = (variances - means) / (means ** 2)
                dispersions = np.clip(dispersions, 1e-3, 10)  # Avoid extreme values

                # debugging
                print(f"Dispersion values for {cell_type}: min={dispersions.min()}, max={dispersions.max()}")

                if np.any(np.isnan(dispersions)):
                    raise ValueError(f"NaN detected in dispersions for {cell_type}!")

                self.cell_type_params[str(cell_type)] = {
                    'means': means.astype(np.float32),
                    'dispersions': dispersions.astype(np.float32)
                }

            elif self.distribution == 'Poisson':
                self.cell_type_params[str(cell_type)] = {
                    'means': means.astype(np.float32)
                }

        print(f"Completed training {self.distribution}")
        with open(self.dist_path, "wb+") as f:
            pickle.dump(self.cell_type_params, f)
        np.save(self.max_real_val_path, self.max_real_value)


    def train_scdesign2(self):
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
        if not os.path.exists(self.hvg_path):
            sc.pp.highly_variable_genes(X_train_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
            self.hvg_mask = X_train_adata.var['highly_variable']
            self.hvg_mask.to_csv(self.hvg_path)
        else:
            self.hvg_mask = pd.read_csv(self.hvg_path)

        hvg_df = X_train_adata.var[self.hvg_mask]
        hvg_df = hvg_df.copy()
        hvg_df['gene'] = hvg_df.index
        print(f"Detected {len(X_train_adata.var[X_train_adata.var['highly_variable']])} HVGs")

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
                self.cmd_no_output(f"Rscript src/generators/models/scdesign2.r train {hvg_subset_path} {cell_type} {copula_path}")
            else:
                print(f"Model exists for {cell_type}, skipping")

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
                exit(1)

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

            out_path = os.path.join(self.tmp_dir, f"out{cell_type}.rds")
            print(self.tmp_dir)
            print(out_path)
            print(f"Rscript src/generators/models/scdesign2.r gen {num_to_gen} {copula_path} {out_path}")
            self.cmd_no_output(f"Rscript src/generators/models/scdesign2.r gen {num_to_gen} {copula_path} {out_path}")

            counts_res = pyreadr.read_r(out_path)
            r_matrix = list(counts_res.values())[0]
            counts_np_array = r_matrix.to_numpy() if hasattr(r_matrix, "to_numpy") else np.array(r_matrix)
            print(counts_np_array.shape)

            for i, row_idx in enumerate(cell_indices):
                synthetic_counts[row_idx, self.hvg_mask] = counts_np_array[:, i]

        final_counts = synthetic_counts
        dist_counts = self.gen_from_dist()

        #synthetic_counts_csr = final_counts.tocsr().astype(np.float64)
        #synthetic_adata = ad.AnnData(X=synthetic_counts_csr)
        #synthetic_adata.obs[self.cell_type_col_name] = cell_types
        #synthetic_adata.var_names = X_test_adata.var_names
        #synthetic_adata.write_h5ad("scdesign2.h5ad")

        #final_counts = dist_counts
        #synthetic_counts_csr = final_counts.tocsr().astype(np.float64)
        #synthetic_adata = ad.AnnData(X=synthetic_counts_csr)
        #synthetic_adata.obs[self.cell_type_col_name] = cell_types
        #synthetic_adata.var_names = X_test_adata.var_names
        #synthetic_adata.write_h5ad("poisson.h5ad")

        final_counts = merge_sparse_matrices(synthetic_counts, dist_counts, self.hvg_mask)
        synthetic_counts_csr = final_counts.tocsr().astype(np.float64)
        synthetic_adata = ad.AnnData(X=synthetic_counts_csr)
        synthetic_adata.obs[self.cell_type_col_name] = cell_types
        synthetic_adata.var_names = X_test_adata.var_names
        synthetic_adata.write_h5ad("together.h5ad")

        return synthetic_adata

    def gen_from_dist(self):
        if self.max_real_value is None:
            if not os.path.exists(self.dist_path) or not os.path.exists(self.max_real_val_path):
                print(f"Path {self.dist_path} or {self.max_real_val_path} does not exist")
                raise ValueError("Training must be completed before generating data!")

            with open(self.dist_path, "rb") as f:
                self.cell_type_params = pickle.load(f)

            self.max_real_value = np.load(self.max_real_val_path)

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
            #synthetic_cell_types.extend([cell_type] * num_cells)

        return synthetic_counts


    def load_from_checkpoint(self):
        pass


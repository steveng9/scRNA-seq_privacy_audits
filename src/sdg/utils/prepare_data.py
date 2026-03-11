"""
Bulk RNA-seq data loader.  Used by Track I (bulk RNA-seq) experiments.
Preserved from the CAMDA 2025 starter package.
"""

import os
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


class RealDataLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.home_dir = self.config["dir_list"]["home"]
        self.dataset_config = config["dataset_config"]

        self.dataset_name = self.dataset_config["name"]
        self.original_data_path = os.path.join(
            self.home_dir, self.dataset_config["count_file"]
        )
        self.original_lbl_path = os.path.join(
            self.home_dir, self.dataset_config["annot_file"]
        )
        self.num_splits = self.dataset_config["num_splits"]
        self.random_seed = self.dataset_config["random_seed"]
        print(f"Random seed set to {self.random_seed}.")

        self.data_save_dir = os.path.join(
            self.home_dir, self.config["dir_list"]["data_save_dir"]
        )
        self.split_indices_dir = os.path.join(
            self.home_dir, self.config["dir_list"]["split_save_dir"]
        )
        self.sample_col_name = self.dataset_config["sample_col_name"]
        self.subtype_col_name = self.dataset_config["subtype_col_name"]

        self.read_original_data()
        self.read_subtype_labels()

    def read_original_data(self):
        if not os.path.exists(self.original_data_path):
            raise FileNotFoundError("Original data is missing.")
        self.original_data = pd.read_csv(
            self.original_data_path, sep="\t", index_col=0
        ).T

    def read_subtype_labels(self):
        if not os.path.exists(self.original_lbl_path):
            raise FileNotFoundError("Labels file is missing.")
        self.subtype_labels = pd.read_csv(self.original_lbl_path)
        ordered = self.subtype_labels.set_index(self.sample_col_name).loc[
            self.original_data.index
        ]
        self.subtype_labels = ordered.reset_index()
        self.subtype_labels = self.subtype_labels.rename(
            columns={"index": self.sample_col_name}
        )
        self.subtype_col_ix = self.subtype_labels.columns.get_loc(self.subtype_col_name)

    def generate_stratified_splits(self):
        skf = StratifiedKFold(
            n_splits=self.num_splits, shuffle=True, random_state=self.random_seed
        )
        self.split_indices = list(
            skf.split(
                self.original_data,
                self.subtype_labels.iloc[:, self.subtype_col_ix],
            )
        )
        self.split_indices_named = [
            (
                self.original_data.index[train_idx],
                self.original_data.index[test_idx],
            )
            for train_idx, test_idx in self.split_indices
        ]
        print("Stratified splits generated.")

    def standardize_split_data(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        scaling_params = pd.DataFrame(
            {"mean": scaler.mean_, "std": scaler.scale_}, index=X_train.columns
        )
        return X_train_scaled, X_test_scaled, scaling_params

    def save_split_indices(self):
        os.makedirs(self.split_indices_dir, exist_ok=True)
        splits_file = os.path.join(
            self.split_indices_dir, f"{self.dataset_name}_splits.yaml"
        )
        self.generate_stratified_splits()
        splits = {}
        for i, (train_idx, test_idx) in enumerate(self.split_indices_named):
            splits[f"split_{i+1}"] = {
                "train_index": train_idx.tolist(),
                "test_index": test_idx.tolist(),
            }
        with open(splits_file, "w") as f:
            yaml.dump({"splits": splits}, f)
        print(f"Splits file saved to {splits_file}.")

    def save_split_data(self):
        splits_file = os.path.join(
            self.split_indices_dir, f"{self.dataset_name}_splits.yaml"
        )
        try:
            with open(splits_file, "r") as f:
                loaded = yaml.load(f, Loader=yaml.FullLoader)
                splits = loaded.get("splits", {})
            split_indices = [
                (np.array(s["train_index"]), np.array(s["test_index"]))
                for s in splits.values()
            ]
        except Exception:
            raise FileNotFoundError("Generate split indices first.")

        split_save_dir = os.path.join(self.data_save_dir, self.dataset_name, "real")
        os.makedirs(split_save_dir, exist_ok=True)

        for i, (train_idx, test_idx) in enumerate(split_indices):
            X_train = self.original_data.loc[train_idx]
            y_train = self.subtype_labels.set_index(self.sample_col_name).loc[train_idx]
            X_test = self.original_data.loc[test_idx]
            y_test = self.subtype_labels.set_index(self.sample_col_name).loc[test_idx]

            pd.DataFrame(X_train, columns=self.original_data.columns).to_csv(
                os.path.join(split_save_dir, f"X_train_real_split_{i+1}.csv"), index=False
            )
            pd.DataFrame(
                y_train.iloc[:, 1].values, columns=[self.subtype_col_name]
            ).to_csv(
                os.path.join(split_save_dir, f"y_train_real_split_{i+1}.csv"), index=False
            )
            pd.DataFrame(X_test, columns=self.original_data.columns).to_csv(
                os.path.join(split_save_dir, f"X_test_real_split_{i+1}.csv"), index=False
            )
            pd.DataFrame(
                y_test.iloc[:, 1].values, columns=[self.subtype_col_name]
            ).to_csv(
                os.path.join(split_save_dir, f"y_test_real_split_{i+1}.csv"), index=False
            )

        pd.DataFrame(
            self.original_data.columns.values, columns=["column_names"]
        ).to_csv(os.path.join(split_save_dir, "column_names.csv"), index=False)

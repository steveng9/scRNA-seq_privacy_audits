import os
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

### conditioning on the subtypes 
class RealDataLoader:
    def __init__(self, config: Dict[str, Any]):
        ## reading config
        self.config = config
        self.home_dir = self.config["dir_list"]["home"]
        self.dataset_config = config["dataset_config"]
        
        self.dataset_name = self.dataset_config["name"]
        self.original_data_path = os.path.join(self.home_dir, 
                                               self.dataset_config["count_file"])
        self.original_lbl_path = os.path.join(self.home_dir, 
                                              self.dataset_config["annot_file"])
        self.num_splits = self.dataset_config["num_splits"]
        self.random_seed = self.dataset_config["random_seed"]
        print(f"Random seed set to {self.random_seed}.")

        self.data_save_dir = os.path.join(self.home_dir, 
                                          self.config["dir_list"]["data_save_dir"])
        self.split_indices_dir = os.path.join(self.home_dir,
                                              self.config["dir_list"]["split_save_dir"])

        self.sample_col_name = self.dataset_config["sample_col_name"]
        self.subtype_col_name = self.dataset_config["subtype_col_name"]

        #self.read_original_data_and_remove_valid()

        self.read_original_data()
        self.read_subtype_labels()

    def read_original_data(self):
        if not os.path.exists(self.original_data_path):
            raise FileNotFoundError("Original data is missing.")
        self.original_data = pd.read_csv(self.original_data_path, 
                                         sep="\t", 
                                         index_col=0).T
        

    def read_subtype_labels(self):
        if not os.path.exists(self.original_lbl_path):
            raise FileNotFoundError("Labels file is missing.")
        
        self.subtype_labels = pd.read_csv(self.original_lbl_path) #????
        ordered_subtype_labels = self.subtype_labels.set_index(
                                        self.sample_col_name).loc[self.original_data.index]
        self.subtype_labels = ordered_subtype_labels.reset_index()
        self.subtype_labels = self.subtype_labels.rename(columns={"index":self.sample_col_name})
        ## order labels by index 
        self.subtype_col_ix = self.subtype_labels.columns.get_loc(self.subtype_col_name)
        



    def generate_stratified_splits(self):
        skf = StratifiedKFold(n_splits=self.num_splits, 
                              shuffle=True, 
                              random_state=self.random_seed)
        self.split_indices = list(skf.split(self.original_data, 
                                            self.subtype_labels.iloc[:, self.subtype_col_ix]))

        self.split_indices_named = [
            (self.original_data.index[train_index], self.original_data.index[test_index])
            for train_index, test_index in self.split_indices
        ]
        print("Stratified splits have been generated.")


    def standardize_split_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        means = scaler.mean_
        stds = scaler.scale_
        scaling_params_df = pd.DataFrame({'mean': means, 'std': stds},
                                          index=X_train.columns)
        
        return X_train_scaled, X_test_scaled, scaling_params_df


    def save_split_indices(self):
        if not os.path.exists(self.split_indices_dir):
            os.makedirs(self.split_indices_dir)

        splits_file = os.path.join(self.split_indices_dir, f"{self.dataset_name}_splits.yaml")
        self.generate_stratified_splits()
        splits = {}
        for i, (train_index, test_index) in enumerate(self.split_indices_named):
                splits[f"split_{i+1}"] = {
                    "train_index": train_index.tolist(),
                    "test_index": test_index.tolist()
                }
        try:
            with open(splits_file, 'w') as file:
                yaml.dump({"splits": splits}, file)
            print(f"Splits file saved successfully at {splits_file}.")
        except Exception as e:
            print(f"Failed to save splits file: {e}")

    def generate_nested_stratified_splits(self):
        outer_skf = StratifiedKFold(n_splits=self.num_splits, shuffle=True, random_state=self.random_seed)
        self.split_indices_named = []

        for train_val_index, test_index in outer_skf.split(self.original_data, self.subtype_labels.iloc[:, self.subtype_col_ix]):
            train_val_data = self.original_data.iloc[train_val_index]
            train_val_labels = self.subtype_labels.iloc[train_val_index, self.subtype_col_ix]

            inner_skf = StratifiedKFold(n_splits=self.num_splits, shuffle=True, random_state=self.random_seed)
            # Here we take only the first fold for the inner split to get a single validation set
            inner_train_index, val_index = next(inner_skf.split(train_val_data, train_val_labels))
            
            train_index = train_val_data.index[inner_train_index]
            val_index = train_val_data.index[val_index]
            test_index = self.original_data.index[test_index]

            self.split_indices_named.append((train_index, val_index, test_index))
        
        print("Stratified splits have been generated.")

    def save_nested_split_indices(self):
        if not os.path.exists(self.split_indices_dir):
            os.makedirs(self.split_indices_dir)

        splits_file = os.path.join(self.split_indices_dir, f"{self.dataset_name}_splits.yaml")
        self.generate_nested_stratified_splits()
        splits = {}
        for i, (train_index, val_index, test_index) in enumerate(self.split_indices_named):
            splits[f"split_{i+1}"] = {
                "train_index": train_index.tolist(),
                "val_index": val_index.tolist(),
                "test_index": test_index.tolist()
            }
        try:
            with open(splits_file, 'w') as file:
                yaml.dump({"splits": splits}, file)
            print(f"Splits file saved successfully at {splits_file}.")
        except Exception as e:
            print(f"Failed to save splits file: {e}")

    def save_validation_dataset(self, i: int, val_index: List[str], split_save_dir: str):
        val_data_df = self.original_data.loc[val_index].copy()
        val_data_df.to_csv(os.path.join(split_save_dir, f"reference_split_{i+1}.csv"), index=True)


    def save_split_data(self):
        try:
            splits_file = os.path.join(self.split_indices_dir, f"{self.dataset_name}_splits.yaml")
            with open(splits_file, 'r') as file:
                loaded_data = yaml.load(file, Loader=yaml.FullLoader)
                splits = loaded_data.get("splits", {})

            split_indices_named = [
                (np.array(split["train_index"]), np.array(split["test_index"]))  
                 #np.array(split["val_index"]))
                for split in splits.values()
            ]
        except:
            raise FileNotFoundError("Generate split indices")

        # Save the actual data splits
        split_save_dir = os.path.join(self.data_save_dir, self.dataset_name, "real")
        if not os.path.exists(split_save_dir):
            os.makedirs(split_save_dir)

        for i, (train_index, test_index) in enumerate(split_indices_named):
            X_train = self.original_data.loc[train_index]
            y_train = self.subtype_labels.set_index(self.sample_col_name).loc[train_index]
            X_test = self.original_data.loc[test_index]
            y_test = self.subtype_labels.set_index(self.sample_col_name).loc[test_index]

            # Standardize the data (fit on train, transform on both train and test)
            #X_train_scaled, X_test_scaled, scaling_params_df = self.standardize_split_data(X_train, X_test)
            #scaling_params_df.to_csv(os.path.join(split_save_dir, f"X_train_scale_params_split_{i+1}.csv"))
            
            X_train_df = pd.DataFrame(X_train, columns=self.original_data.columns)
            X_test_df = pd.DataFrame(X_test, columns=self.original_data.columns)

            X_train_df.to_csv(os.path.join(split_save_dir, f"X_train_real_split_{i+1}.csv"), index=False)
            pd.DataFrame(y_train.iloc[:, 1].values, columns=[self.subtype_col_name]).to_csv(
                os.path.join(split_save_dir, f"y_train_real_split_{i+1}.csv"), index=False)
            X_test_df.to_csv(os.path.join(split_save_dir, f"X_test_real_split_{i+1}.csv"), index=False)
            pd.DataFrame(y_test.iloc[:, 1].values, columns=[self.subtype_col_name]).to_csv(
                os.path.join(split_save_dir, f"y_test_real_split_{i+1}.csv"), index=False)
            
            
        # Save column names
        pd.DataFrame(self.original_data.columns.values, columns=['column_names']).to_csv(
             os.path.join(split_save_dir, "column_names.csv"), index=False)




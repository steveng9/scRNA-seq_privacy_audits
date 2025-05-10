import os
import subprocess
import scanpy as sc
import anndata as ad
from typing import Dict, Any
from abc import ABC, abstractmethod


def check_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


class AbstractSingleCellDataGenerator(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def save_synthetic_anndata(self, 
                                synthetic_features: ad.AnnData, 
                                experiment_name: str):
        pass

    @abstractmethod
    def train(self):
        pass
        
    @abstractmethod
    def load_from_checkpoint(self):
        pass



class BaseSingleCellDataGenerator(AbstractSingleCellDataGenerator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dataset_config = self.config["dataset_config"]

        # Common configuration loading logic
        self.generator_name = self.config.get('generator_name')
        config_key = f"{self.generator_name}_config"

        if config_key in config:
            self.generator_config = config[config_key]
        else:
            raise ValueError(f"No config found for generator: {self.generator_name}")

        self.dataset_name = self.dataset_config["name"]
        self.home_dir = self.config["dir_list"]["home"]

    def cmd_no_output(self, cmd):
        try:
            output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode('utf-8')
            return output
        except subprocess.CalledProcessError as e:
            print(f"Command '{cmd}' failed with return code {e.returncode} "\
                    f"and error {e.output}")
            exit(1)

    def load_train_anndata(self):
        try:
            train_data_pth = os.path.join(self.home_dir, self.dataset_config["train_count_file"])
            train_data = sc.read_h5ad(train_data_pth)

            return train_data
        except:
            raise Exception(f"Failed to load train anndata.")
        

    def load_test_anndata(self):
        try: 
            test_data_pth = os.path.join(self.home_dir, self.dataset_config["test_count_file"])
            test_data = sc.read_h5ad(test_data_pth)

            return test_data
        except:
            raise Exception(f"Failed to load test anndata.")
        
    
    def load_external_anndata(self):
        if self.dataset_config["external_count_file"]: 
            external_data_pth = os.path.join(self.home_dir, self.dataset_config["external_count_file"])
            external_data = sc.read_h5ad(external_data_pth)

            return external_data
        else:
            raise Exception(f"Failed to load test anndata.")
         

    def save_synthetic_anndata(self, 
                            synthetic_features: ad.AnnData, 
                            experiment_name: str = ""):
        data_save_dir = os.path.join(self.config["dir_list"]["home"],
                                     self.config["dir_list"]["data_splits"])
        
        syn_save_dir = os.path.join(
                                    data_save_dir, 
                                    self.dataset_name, 
                                    "synthetic", 
                                    self.generator_name,
                                    experiment_name
                                    )
        check_dirs(syn_save_dir)
                
        # save synthetic features and labels
        synthetic_features.write(os.path.join(syn_save_dir, "onek1k_annotated_synthetic.h5ad" ),
                                 compression="gzip")
        print(f"Synthetic data saved in {syn_save_dir}.")




    @abstractmethod
    def generate(self):
        """Generate synthetic data based on your configuration."""
        pass

    @abstractmethod
    def train(self):
        """Train your generator model."""
        pass

    @abstractmethod
    def load_from_checkpoint(self):
        """Load your model from a checkpoint."""
        pass

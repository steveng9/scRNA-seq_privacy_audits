"""
Abstract base class for Synthetic Data Generators (SDGs).

All new generators (scDesign3, scVAE, etc.) should subclass
BaseSingleCellDataGenerator and implement the three abstract methods.
"""

import os
import subprocess
import anndata as ad
from abc import ABC, abstractmethod
from typing import Dict, Any


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class AbstractSingleCellDataGenerator(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def train(self):
        """Fit the generative model on training data."""
        pass

    @abstractmethod
    def generate(self) -> ad.AnnData:
        """Sample synthetic cells and return an AnnData object."""
        pass

    @abstractmethod
    def load_from_checkpoint(self):
        """Restore a previously trained model from disk."""
        pass

    @abstractmethod
    def save_synthetic_anndata(self,
                               synthetic_data: ad.AnnData,
                               experiment_name: str,
                               synthetic_path: str = None):
        """Persist generated AnnData to disk."""
        pass


class BaseSingleCellDataGenerator(AbstractSingleCellDataGenerator):
    """
    Concrete base that handles config parsing and common I/O.
    Subclasses implement train() and generate().
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dataset_config = self.config["dataset_config"]

        self.generator_name = self.config.get("generator_name")
        config_key = f"{self.generator_name}_config"
        if config_key not in config:
            raise ValueError(f"No config block found for generator: {self.generator_name}")
        self.generator_config = config[config_key]

        self.dataset_name = self.dataset_config["name"]
        self.home_dir = self.config["dir_list"]["home"]
        self.data_dir = self.config["dir_list"]["data"]
        self.random_seed = self.dataset_config.get("random_seed", 42)

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def load_train_anndata(self) -> ad.AnnData:
        import scanpy as sc
        path = os.path.join(self.data_dir, self.dataset_config["train_count_file"])
        try:
            return sc.read_h5ad(path)
        except Exception:
            raise IOError(f"Failed to load training data from {path}")

    def load_test_anndata(self) -> ad.AnnData:
        import scanpy as sc
        path = os.path.join(self.data_dir, self.dataset_config["test_count_file"])
        try:
            return sc.read_h5ad(path)
        except Exception:
            raise IOError(f"Failed to load test data from {path}")

    # ------------------------------------------------------------------
    # Saving helpers
    # ------------------------------------------------------------------

    def save_synthetic_anndata(self,
                               synthetic_data: ad.AnnData,
                               experiment_name: str = "",
                               synthetic_path: str = None):
        if synthetic_path is None:
            save_dir = os.path.join(
                self.config["dir_list"]["data"],
                self.dataset_name,
                "synthetic_data",
                self.generator_name,
            )
            ensure_dir(save_dir)
            synthetic_path = os.path.join(
                save_dir, self.config["dataset_config"]["synthetic_data_name"]
            )
        synthetic_data.write(synthetic_path, compression="gzip")
        print(f"Synthetic data saved to {synthetic_path}.")

    # ------------------------------------------------------------------
    # Subprocess helper
    # ------------------------------------------------------------------

    @staticmethod
    def run_command(cmd: str) -> str:
        try:
            return subprocess.check_output(
                cmd, shell=True, stderr=subprocess.STDOUT
            ).decode("utf-8")
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Command failed (code {e.returncode}): {cmd}")
            print(f"[WARN] Error (first 300 chars): {e.output[:300]}")
            return None

"""
scVI synthetic data generator (Python wrapper around run_scvi_standalone.py).

Trains scvi.model.SCVI in a dedicated conda environment ("scvi_") via
subprocess, exactly as scDesign3 calls Rscript.  The three modes of the
standalone script (train / generate / score) map to the three methods here.

Config keys (under scvi_config):
    model_dir    : relative path (from home_dir) where model checkpoints land
    hvg_path     : shared HVG CSV path (gene × highly_variable bool)
    conda_env    : conda env name (default "scvi_")
    n_latent     : latent dimension (default 30)
    n_layers     : number of encoder/decoder layers (default 2)
    n_hidden     : hidden layer width (default 128)
    max_epochs   : training epochs cap (default 400)
    batch_size   : mini-batch size (default 512)
"""

import os
import sys
import subprocess
import numpy as np
import anndata as ad
import scanpy as sc

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from sdg.base import BaseSingleCellDataGenerator

# Absolute path to the standalone script — CWD-independent
_STANDALONE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "run_scvi_standalone.py")


def _run(cmd: list[str], label: str) -> None:
    """Run a command, streaming stdout/stderr; raise on non-zero exit."""
    full_cmd = " ".join(cmd)
    print(f"[scVI:{label}] {full_cmd}", flush=True)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"[scVI:{label}] subprocess exited with code {result.returncode}"
        )


class ScVI(BaseSingleCellDataGenerator):
    """
    Wraps scvi.model.SCVI for training and synthetic data generation.

    All heavy computation (PyTorch training, ELBO scoring) runs inside the
    ``scvi_`` conda environment via subprocess so that the rest of the
    pipeline (tabddpm_ / camda_) is not affected by scvi-tools dependencies.
    """

    def __init__(self, config):
        super().__init__(config)
        gcfg = self.generator_config

        self.model_dir    = os.path.join(self.home_dir, gcfg["model_dir"])
        self.hvg_path     = gcfg.get("hvg_path", None)
        self.conda_env    = gcfg.get("conda_env", "scvi_")
        self.n_latent     = gcfg.get("n_latent",   30)
        self.n_layers     = gcfg.get("n_layers",    2)
        self.n_hidden     = gcfg.get("n_hidden",   128)
        self.max_epochs   = gcfg.get("max_epochs", 400)
        self.batch_size   = gcfg.get("batch_size", 512)

        # Resolved paths for data I/O
        cell_type_col = self.dataset_config["cell_type_col_name"]
        self.cell_type_col = cell_type_col

    # ------------------------------------------------------------------
    # Internal helper: build the "conda run" prefix
    # ------------------------------------------------------------------

    def _conda_run(self, *args) -> list[str]:
        return ["conda", "run", "--no-capture-output", "-n", self.conda_env,
                "python", _STANDALONE] + list(args)

    # ------------------------------------------------------------------

    def train(self):
        train_path = os.path.join(
            self.data_dir, self.dataset_config["train_count_file"]
        )
        os.makedirs(self.model_dir, exist_ok=True)

        cmd = self._conda_run(
            "train", train_path, self.model_dir,
            "--n-latent",   str(self.n_latent),
            "--n-layers",   str(self.n_layers),
            "--n-hidden",   str(self.n_hidden),
            "--max-epochs", str(self.max_epochs),
            "--batch-size", str(self.batch_size),
        )
        if self.hvg_path:
            cmd += ["--hvg-path", self.hvg_path]

        _run(cmd, "train")

    # ------------------------------------------------------------------

    def generate(self) -> ad.AnnData:
        train_path = os.path.join(
            self.data_dir, self.dataset_config["train_count_file"]
        )
        test_path  = os.path.join(
            self.data_dir, self.dataset_config["test_count_file"]
        )
        out_path   = os.path.join(self.home_dir, "tmp_scvi_synth.h5ad")

        # Use the same number of cells as the test set
        test_adata = sc.read_h5ad(test_path)
        n_cells    = test_adata.n_obs

        cmd = self._conda_run(
            "generate", train_path, self.model_dir, out_path, str(n_cells)
        )
        _run(cmd, "generate")

        synth = sc.read_h5ad(out_path)
        os.remove(out_path)
        return synth

    # ------------------------------------------------------------------

    def score_cells(self, target_h5ad: str, scores_out: str) -> np.ndarray:
        """
        Compute per-cell ELBO for every cell in target_h5ad.
        Saves scores to scores_out (.npy) and returns the array.
        """
        cmd = self._conda_run("score", target_h5ad, self.model_dir, scores_out)
        _run(cmd, "score")
        return np.load(scores_out)

    # ------------------------------------------------------------------

    def load_from_checkpoint(self):
        """No-op: model is loaded inside the subprocess on demand."""
        pass

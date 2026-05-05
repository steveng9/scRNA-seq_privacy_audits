"""
ZINBWave synthetic data generator wrapper (CAMDA Track II submission).

Fits a per-cell-type ZINB latent-factor model (ZINBWave, Risso et al. 2018)
via R subprocess and generates synthetic counts by bootstrapping from fitted
ZINB parameters.

The train() method runs the full ZINBWave pipeline (train R models + generate
synthetic counts) and caches the result.  generate() returns the cached data.

Config keys (under zinbwave_config):
  out_model_path     : directory for per-cell-type .rds model files  (required)
  hvg_path           : path to HVG mask CSV; computed if absent
  n_latent           : ZINBWave latent factors K  (default: 10)
  max_cells_per_type : subsample cap per cell type  (default: 3000)
  cell_type_col      : obs column with cell-type labels  (default: from dataset_config)
  n_workers          : parallel R processes  (default: 4)
  seed               : random seed  (default: 42)
"""

import os
import sys
import subprocess
import shutil
import tempfile

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from concurrent.futures import ProcessPoolExecutor

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from models.sc_base import BaseSingleCellDataGenerator

_ZINBWAVE_STANDALONE = os.path.join(_PKG_DIR, "run_zinbwave_standalone.py")


class ZINBWaveGenerator(BaseSingleCellDataGenerator):
    """
    ZINBWave-based single-cell data generator.

    Set generator_name: zinbwave in config.yaml.
    """

    def __init__(self, config):
        super().__init__(config)
        gcfg = self.generator_config

        self.cell_type_col    = self.dataset_config.get("cell_type_col_name", "cell_type")
        self.hvg_path         = gcfg.get("hvg_path", None)
        self.n_latent         = int(gcfg.get("n_latent",           10))
        self.max_cells_per_type = int(gcfg.get("max_cells_per_type", 3000))
        self.n_workers        = int(gcfg.get("n_workers",           4))
        self.seed             = int(gcfg.get("seed",                42))

        self.model_dir   = os.path.join(self.home_dir, gcfg["out_model_path"])
        self._synth_path = os.path.join(self.home_dir, "_zinbwave_synth_cache.h5ad")

    # ------------------------------------------------------------------

    def train(self):
        """Run the full ZINBWave pipeline (train + generate) and cache the output."""
        os.makedirs(self.model_dir, exist_ok=True)
        train_adata = self.load_train_anndata()
        hvg_mask    = self._get_or_compute_hvg_mask(train_adata)

        tmp_dir    = tempfile.mkdtemp(dir=self.home_dir, prefix="zinbw_train_")
        train_hvg  = os.path.join(tmp_dir, "train_hvg.h5ad")
        try:
            train_adata[:, hvg_mask].copy().write_h5ad(train_hvg)

            cmd = (
                f"{sys.executable} {_ZINBWAVE_STANDALONE} "
                f"--train-h5ad {train_hvg} "
                f"--output-h5ad {self._synth_path} "
                f"--model-dir {self.model_dir} "
                f"--n-latent {self.n_latent} "
                f"--max-cells-per-type {self.max_cells_per_type} "
                f"--cell-type-col {self.cell_type_col} "
                f"--n-workers {self.n_workers} "
                f"--seed {self.seed}"
            )
            print(f"Running ZINBWave pipeline...", flush=True)
            result = subprocess.run(cmd, shell=True, capture_output=False)
            if result.returncode != 0:
                raise RuntimeError("ZINBWave standalone script failed")
            print(f"ZINBWave output cached → {self._synth_path}", flush=True)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------

    def generate(self):
        """Return synthetic data generated during train()."""
        if not os.path.exists(self._synth_path):
            raise RuntimeError(
                f"ZINBWave synthetic cache not found at {self._synth_path}. "
                "Run train() first."
            )
        synth = ad.read_h5ad(self._synth_path)
        print(f"Loaded ZINBWave synthetic data: {synth.n_obs:,} cells × {synth.n_vars} genes",
              flush=True)
        return synth

    # ------------------------------------------------------------------

    def load_from_checkpoint(self):
        pass

    # ------------------------------------------------------------------

    def _get_or_compute_hvg_mask(self, adata):
        if self.hvg_path and os.path.exists(self.hvg_path):
            hvg_df = pd.read_csv(self.hvg_path, index_col=0)
            if len(hvg_df) != len(adata.var_names):
                hvg_df = hvg_df.reindex(adata.var_names).fillna(False)
            return hvg_df["highly_variable"].values.astype(bool)

        tmp = adata.copy()
        tmp.layers["counts"] = tmp.X.copy()
        sc.pp.normalize_total(tmp, layer="counts", target_sum=1e4)
        sc.pp.log1p(tmp, layer="counts")
        sc.pp.highly_variable_genes(tmp, layer="counts",
                                     min_mean=0.0125, max_mean=3, min_disp=0.5)
        mask = tmp.var["highly_variable"].values.astype(bool)
        print(f"  Computed {mask.sum()} HVGs from training data", flush=True)
        if self.hvg_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.hvg_path)), exist_ok=True)
            pd.Series(mask, index=adata.var_names,
                      name="highly_variable").to_csv(self.hvg_path)
        return mask

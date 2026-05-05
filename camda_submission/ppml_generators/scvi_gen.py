"""
scVI synthetic data generator wrapper (CAMDA Track II submission).

Wraps scvi.model.SCVI via a subprocess into the separate "scvi_" conda
environment (see environment_scvi.yaml).

Config keys (under scvi_config):
  model_dir    : where to save model checkpoints  (required)
  hvg_path     : path to HVG mask CSV; computed if absent
  conda_env    : conda env name for scvi-tools  (default: "scvi_")
  n_latent     : latent dimension  (default: 30)
  n_layers     : encoder/decoder depth  (default: 2)
  n_hidden     : hidden layer width  (default: 128)
  max_epochs   : training epoch cap  (default: 400)
  batch_size   : mini-batch size  (default: 512)
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

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from models.sc_base import BaseSingleCellDataGenerator

_SCVI_STANDALONE = os.path.join(_PKG_DIR, "run_scvi_standalone.py")


class ScVIGenerator(BaseSingleCellDataGenerator):
    """
    scVI-based single-cell data generator.

    Requires a separate conda environment with scvi-tools installed.
    Set generator_name: scvi in config.yaml and specify conda_env.
    """

    def __init__(self, config):
        super().__init__(config)
        gcfg = self.generator_config

        self.cell_type_col = self.dataset_config.get("cell_type_col_name", "cell_type")
        self.hvg_path    = gcfg.get("hvg_path", None)
        self.conda_env   = gcfg.get("conda_env",   "scvi_")
        self.n_latent    = int(gcfg.get("n_latent",    30))
        self.n_layers    = int(gcfg.get("n_layers",     2))
        self.n_hidden    = int(gcfg.get("n_hidden",   128))
        self.max_epochs  = int(gcfg.get("max_epochs", 400))
        self.batch_size  = int(gcfg.get("batch_size", 512))

        self.model_dir   = os.path.join(self.home_dir, gcfg["model_dir"])
        self._train_hvg  = None   # set during train(), used by generate()

    # ------------------------------------------------------------------

    def _conda_run(self, *args):
        return (f"conda run --no-capture-output -n {self.conda_env} "
                f"python {_SCVI_STANDALONE} " + " ".join(str(a) for a in args))

    def _run(self, cmd):
        print(f"  $ {cmd}", flush=True)
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            raise RuntimeError(f"scVI subprocess exited {result.returncode}")

    # ------------------------------------------------------------------

    def train(self):
        """Train scVI model (runs inside the scvi_ conda env via subprocess)."""
        os.makedirs(self.model_dir, exist_ok=True)
        train_adata = self.load_train_anndata()
        hvg_mask    = self._get_or_compute_hvg_mask(train_adata)

        tmp_dir   = tempfile.mkdtemp(dir=self.home_dir, prefix="scvi_train_")
        train_hvg = os.path.join(tmp_dir, "train_hvg.h5ad")
        self._train_hvg = train_hvg   # keep for generate()

        try:
            train_adata[:, hvg_mask].copy().write_h5ad(train_hvg)

            hvg_flag = f"--hvg-path {self.hvg_path}" if self.hvg_path else ""
            cmd = self._conda_run(
                "train", train_hvg, self.model_dir,
                f"--n-latent {self.n_latent}",
                f"--n-layers {self.n_layers}",
                f"--n-hidden {self.n_hidden}",
                f"--max-epochs {self.max_epochs}",
                f"--batch-size {self.batch_size}",
                hvg_flag,
            )
            self._run(cmd)
            print(f"scVI model saved → {self.model_dir}", flush=True)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            self._train_hvg = None
            raise

    # ------------------------------------------------------------------

    def generate(self):
        """Generate synthetic cells (runs inside the scvi_ conda env)."""
        train_adata = self.load_train_anndata()
        n_cells     = train_adata.n_obs

        hvg_mask    = self._get_or_compute_hvg_mask(train_adata)
        tmp_dir     = tempfile.mkdtemp(dir=self.home_dir, prefix="scvi_gen_")
        train_hvg   = self._train_hvg or os.path.join(tmp_dir, "train_hvg.h5ad")
        out_h5ad    = os.path.join(tmp_dir, "synth_scvi.h5ad")

        try:
            if not os.path.exists(train_hvg):
                train_adata[:, hvg_mask].copy().write_h5ad(train_hvg)

            cmd = self._conda_run(
                "generate", train_hvg, self.model_dir, out_h5ad, str(n_cells)
            )
            self._run(cmd)

            synth = ad.read_h5ad(out_h5ad)
            print(f"scVI synthetic: {synth.n_obs:,} cells × {synth.n_vars} genes",
                  flush=True)
            return synth
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

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
        if self.hvg_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.hvg_path)), exist_ok=True)
            pd.Series(mask, index=adata.var_names,
                      name="highly_variable").to_csv(self.hvg_path)
        return mask

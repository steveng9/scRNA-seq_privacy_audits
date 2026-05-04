"""
scDiffusion generator wrapper (Luo et al. 2024, Bioinformatics).

Dispatches all heavy work to run_scdiffusion_standalone.py via
  conda run -n scdiff_ python run_scdiffusion_standalone.py <mode> ...

IMPLEMENTATION VERSION HISTORY
-------------------------------
v1 (prior to 2026-05-04):
    Two-stage pipeline: train_vae + train (diffusion backbone).
    Cell types assigned post-hoc via 1-NN. Wrong hyperparameters.
    Data stored in ~/data/scMAMAMIA/{dataset}/scdiffusion/.

v2 (2026-05-04):
    Three-stage pipeline matching Luo et al. 2024:
      1. train_vae        -- VAE autoencoder
      2. train            -- DDPM backbone in latent space
      3. train_classifier -- Cell_classifier for guided sampling
    Hyperparameters now match paper defaults (see defaults below).
    Cell types generated via classifier guidance, not post-hoc 1-NN.
    Data stored in ~/data/scMAMAMIA/{dataset}/scdiffusion_v2/.

Config keys (under scdiffusion_config):
  vae_dir            relative path for VAE checkpoint directory
  diff_dir           relative path for diffusion checkpoint directory
  classifier_dir     relative path for classifier checkpoint directory
  hvg_path           shared HVG CSV path
  conda_env          conda env name           (default: "scdiff_")
  latent_dim         VAE latent dim           (default: 128)
  vae_steps          VAE training iterations  (default: 200000)
  diff_steps         diffusion iterations     (default: 800000)
  batch_size         mini-batch size          (default: 128)
  save_interval      checkpoint save freq     (default: 100000)
  classifier_steps   classifier iterations    (default: 200000)
  classifier_scale   guidance scale           (default: 2.0)
  start_guide_steps  guidance threshold       (default: 500)
  generation_batch_size  sampling batch size  (default: 3000)
"""

import os
import sys
import subprocess
import glob
import numpy as np
import anndata as ad

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from sdg.base import BaseSingleCellDataGenerator

_STANDALONE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "run_scdiffusion_standalone.py"
)


def _run(cmd, label):
    print(f"[scDiffusion:{label}]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"[scDiffusion:{label}] exited {r.returncode}")


class ScDiffusion(BaseSingleCellDataGenerator):

    def __init__(self, config):
        super().__init__(config)
        gc = self.generator_config

        self.vae_dir        = os.path.join(self.home_dir, gc["vae_dir"])
        self.diff_dir       = os.path.join(self.home_dir, gc["diff_dir"])
        self.classifier_dir = os.path.join(self.home_dir,
                                           gc.get("classifier_dir", "models/classifier"))
        self.hvg_path       = gc.get("hvg_path", None)
        self.conda_env      = gc.get("conda_env", "scdiff_")
        self.latent_dim     = gc.get("latent_dim",    128)
        self.vae_steps      = gc.get("vae_steps",   200000)
        self.diff_steps     = gc.get("diff_steps",  800000)
        self.batch_size     = gc.get("batch_size",     128)
        self.save_interval  = gc.get("save_interval", 100000)
        self.classifier_steps       = gc.get("classifier_steps",       200000)
        self.classifier_scale       = gc.get("classifier_scale",          2.0)
        self.start_guide_steps      = gc.get("start_guide_steps",         500)
        self.generation_batch_size  = gc.get("generation_batch_size",    3000)
        self.cell_type_col          = self.dataset_config["cell_type_col_name"]

    def _base_cmd(self, *args):
        return [
            "conda", "run", "--no-capture-output", "-n", self.conda_env,
            "python", _STANDALONE,
        ] + list(args) + self._shared_flags()

    def _shared_flags(self):
        flags = ["--cell-type-col", self.cell_type_col,
                 "--latent-dim",    str(self.latent_dim)]
        if self.hvg_path:
            flags += ["--hvg-path", self.hvg_path]
        return flags

    def _vae_ckpt(self):
        pts = sorted(glob.glob(os.path.join(self.vae_dir, "model_seed=0_step=*.pt")))
        if not pts:
            raise FileNotFoundError(f"No VAE checkpoint in {self.vae_dir}")
        return pts[-1]

    def _diff_ckpt(self):
        pts = sorted(glob.glob(os.path.join(self.diff_dir, "diffusion", "model*.pt")))
        if not pts:
            pts = sorted(glob.glob(os.path.join(self.diff_dir, "model*.pt")))
        if not pts:
            raise FileNotFoundError(f"No diffusion checkpoint in {self.diff_dir}")
        return pts[-1]

    def _classifier_ckpt(self):
        pts = sorted(glob.glob(os.path.join(self.classifier_dir, "model*.pt")))
        if not pts:
            raise FileNotFoundError(f"No classifier checkpoint in {self.classifier_dir}")
        return pts[-1]

    def _train_h5ad(self):
        return os.path.join(self.data_dir,
                            self.dataset_config["train_count_file"])

    # ------------------------------------------------------------------

    def train(self):
        train_h5ad = self._train_h5ad()
        os.makedirs(self.vae_dir,        exist_ok=True)
        os.makedirs(self.diff_dir,       exist_ok=True)
        os.makedirs(self.classifier_dir, exist_ok=True)

        # Stage 1: VAE
        print("==> Stage 1: Training VAE ...", flush=True)
        _run(self._base_cmd("train_vae", train_h5ad, self.vae_dir,
                             "--vae-steps",     str(self.vae_steps),
                             "--batch-size",    str(self.batch_size),
                             "--save-interval", str(self.save_interval)), "train_vae")

        # Stage 2: Diffusion backbone
        print("==> Stage 2: Training diffusion backbone ...", flush=True)
        _run(self._base_cmd("train", train_h5ad, self._vae_ckpt(), self.diff_dir,
                             "--diff-steps",    str(self.diff_steps),
                             "--batch-size",    str(self.batch_size),
                             "--save-interval", str(self.save_interval)), "train")

        # Stage 3: Classifier
        print("==> Stage 3: Training classifier ...", flush=True)
        _run(self._base_cmd("train_classifier", train_h5ad, self._vae_ckpt(),
                             self.classifier_dir,
                             "--classifier-steps",  str(self.classifier_steps),
                             "--batch-size",         str(self.batch_size),
                             "--start-guide-steps",  str(self.start_guide_steps)), "train_classifier")

    def generate(self) -> ad.AnnData:
        import scanpy as sc
        test_path = os.path.join(self.data_dir,
                                  self.dataset_config["test_count_file"])
        n_cells   = sc.read_h5ad(test_path).n_obs
        out_path  = os.path.join(self.home_dir, "tmp_scdiff_synth.h5ad")

        _run(self._base_cmd("generate",
                             self._train_h5ad(),
                             self._vae_ckpt(),
                             self._diff_ckpt(),
                             self._classifier_ckpt(),
                             out_path, str(n_cells),
                             "--classifier-scale",      str(self.classifier_scale),
                             "--start-guide-steps",     str(self.start_guide_steps),
                             "--generation-batch-size", str(self.generation_batch_size)),
             "generate")

        synth = sc.read_h5ad(out_path)
        os.remove(out_path)
        return synth

    def score_cells(self, target_h5ad: str, scores_out: str) -> np.ndarray:
        """Compute per-cell MIA scores. Returns array of shape (n_cells,)."""
        _run(self._base_cmd("score",
                             target_h5ad, self._vae_ckpt(), self._diff_ckpt(),
                             scores_out), "score")
        return np.load(scores_out)

    def load_from_checkpoint(self):
        pass

"""
scDiffusion generator wrapper (Luo et al. 2024, Bioinformatics).

Dispatches all heavy work to run_scdiffusion_standalone.py via
  conda run -n scdiff_ python run_scdiffusion_standalone.py <mode> ...

Training is two-stage:
  1. train_vae  — VAE autoencoder (gene space → 128-dim latent)
  2. train      — DDPM backbone in latent space

Config keys (under scdiffusion_config):
  vae_dir        relative path for VAE checkpoint directory
  diff_dir       relative path for diffusion checkpoint directory
  hvg_path       shared HVG CSV path
  conda_env      conda env name  (default: "scdiff_")
  latent_dim     VAE latent dim  (default: 128)
  vae_steps      VAE training iterations   (default: 150000)
  diff_steps     diffusion iterations      (default: 300000)
  batch_size     mini-batch size           (default: 512)
  save_interval  checkpoint save frequency (default: 50000)
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

        self.vae_dir       = os.path.join(self.home_dir, gc["vae_dir"])
        self.diff_dir      = os.path.join(self.home_dir, gc["diff_dir"])
        self.hvg_path      = gc.get("hvg_path", None)
        self.conda_env     = gc.get("conda_env", "scdiff_")
        self.latent_dim    = gc.get("latent_dim",    128)
        self.vae_steps     = gc.get("vae_steps",   150000)
        self.diff_steps    = gc.get("diff_steps",  300000)
        self.batch_size    = gc.get("batch_size",     512)
        self.save_interval = gc.get("save_interval", 50000)
        self.cell_type_col = self.dataset_config["cell_type_col_name"]

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
        """Return path to the latest VAE checkpoint in vae_dir."""
        pts = sorted(glob.glob(os.path.join(self.vae_dir, "model_seed=0_step=*.pt")))
        if not pts:
            raise FileNotFoundError(f"No VAE checkpoint in {self.vae_dir}")
        return pts[-1]

    def _diff_ckpt(self):
        """Return path to the latest diffusion checkpoint.
        TrainLoop saves to <diff_dir>/diffusion/model*.pt."""
        # Try both direct and the model_name subdirectory
        pts = sorted(glob.glob(os.path.join(self.diff_dir, "diffusion", "model*.pt")))
        if not pts:
            pts = sorted(glob.glob(os.path.join(self.diff_dir, "model*.pt")))
        if not pts:
            raise FileNotFoundError(f"No diffusion checkpoint in {self.diff_dir}")
        return pts[-1]

    def _train_h5ad(self):
        return os.path.join(self.data_dir,
                            self.dataset_config["train_count_file"])

    # ------------------------------------------------------------------

    def train(self):
        train_h5ad = self._train_h5ad()
        os.makedirs(self.vae_dir,  exist_ok=True)
        os.makedirs(self.diff_dir, exist_ok=True)

        # Step 1: VAE
        print("==> Training VAE ...", flush=True)
        _run(self._base_cmd("train_vae", train_h5ad, self.vae_dir,
                             "--vae-steps",  str(self.vae_steps),
                             "--batch-size", str(self.batch_size)), "train_vae")

        # Step 2: Diffusion backbone
        print("==> Training diffusion backbone ...", flush=True)
        _run(self._base_cmd("train", train_h5ad, self._vae_ckpt(), self.diff_dir,
                             "--diff-steps",    str(self.diff_steps),
                             "--batch-size",    str(self.batch_size),
                             "--save-interval", str(self.save_interval)), "train")

    def generate(self) -> ad.AnnData:
        import scanpy as sc
        test_path = os.path.join(self.data_dir,
                                  self.dataset_config["test_count_file"])
        n_cells   = sc.read_h5ad(test_path).n_obs
        out_path  = os.path.join(self.home_dir, "tmp_scdiff_synth.h5ad")

        _run(self._base_cmd("generate",
                             self._train_h5ad(), self._vae_ckpt(), self._diff_ckpt(),
                             out_path, str(n_cells)), "generate")

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

"""
Generate synthetic scRNA-seq data for one trial of one SDG method.

Loads pre-existing donor splits from the shared splits/ directory under the
dataset root, trains the target generator on D_train, and saves the synthetic
output.  No MIA attack is run — this is generation only.

Supported generators
--------------------
  sd3_gaussian   scDesign3 with Gaussian copula
  sd3_vine       scDesign3 with vine copula
  scvi           scVI VAE
  scdiffusion    scDiffusion v1 LEGACY — do NOT use for new experiments.
                 Two-stage pipeline (VAE + diffusion only), post-hoc 1-NN
                 cell-type assignment, wrong hyperparameters. Data stored
                 under scdiffusion/. Kept only for reproducibility.
  scdiffusion_v2 scDiffusion v2 (2026-05-04, LEGACY) — classifier-guided
                 sampling. Correct hyperparameters but wrong generation mode.
                 Data stored under scdiffusion_v2/. Do not use for new runs.
  scdiffusion_v3 scDiffusion v3 (2026-05-04+) — PAPER-FAITHFUL. Unconditional
                 DDPM generation + CellTypist post-hoc annotation. Matches the
                 main evaluation pipeline in Luo et al. 2024. Correct
                 hyperparameters (vae=200k, diff=800k, batch=128). Data stored
                 under scdiffusion_v3/. Use this for all new experiments.
  nmf            SingleCellNMFGenerator (NMF + KMeans + ZINB sampling)
  zinbwave       ZINBWave (ZINB latent-factor model, Risso et al. 2018)

Output layout
-------------
  <out-dir>/
    datasets/
      synthetic.h5ad      final output
      train.h5ad          temporary training subset (deleted after generation)
    models/
      <cell_type>.rds     (scDesign3 only)
      model/              (scVI only)
      vae/  diff/  classifier/  (scDiffusion v2 only)

  Donor splits (train.npy, holdout.npy, auxiliary.npy) live in the shared
  splits/ directory at the dataset root, not in each SDG trial dir.

Usage
-----
  # scDesign3 Gaussian, OneK1K, 10 donors, trial 3
  python experiments/sdg_comparison/generate_trial.py \\
      --generator  sd3_gaussian \\
      --dataset    /home/golobs/data/scMAMAMIA/ok/full_dataset_cleaned.h5ad \\
      --splits-dir /home/golobs/data/scMAMAMIA/ok/splits/10d/3 \\
      --out-dir    /home/golobs/data/scMAMAMIA/ok/scdesign3/gaussian/10d/3 \\
      --hvg-path   /home/golobs/data/scMAMAMIA/ok/hvg_full.csv

  # scVI
  python experiments/sdg_comparison/generate_trial.py \\
      --generator scvi \\
      --dataset   /home/golobs/data/scMAMAMIA/ok/full_dataset_cleaned.h5ad \\
      --splits-dir /home/golobs/data/scMAMAMIA/ok/splits/10d/3 \\
      --out-dir    /home/golobs/data/scMAMAMIA/ok/scvi/no_dp/10d/3 \\
      --hvg-path   /home/golobs/data/scMAMAMIA/ok/hvg_full.csv \\
      --conda-env  scvi_

  # ZINBWave, 10 donors, trial 3:
  python experiments/sdg_comparison/generate_trial.py \\
      --generator zinbwave \\
      --dataset   /home/golobs/data/scMAMAMIA/ok/full_dataset_cleaned.h5ad \\
      --splits-dir /home/golobs/data/scMAMAMIA/ok/splits/10d/3 \\
      --out-dir    /home/golobs/data/scMAMAMIA/ok/zinbwave/no_dp/10d/3 \\
      --hvg-path   /home/golobs/data/scMAMAMIA/ok/hvg_full.csv
"""

import argparse
import glob
import os
import re
import shutil
import sys
import subprocess

import numpy as np
import anndata as ad
import scanpy as sc

# ---------------------------------------------------------------------------
# Resolve src/ directory so we can import project modules
# ---------------------------------------------------------------------------
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC  = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_splits(splits_dir):
    """Return (train_donors, holdout_donors) from a datasets/ directory."""
    train   = np.load(os.path.join(splits_dir, "train.npy"),   allow_pickle=True)
    holdout = np.load(os.path.join(splits_dir, "holdout.npy"), allow_pickle=True)
    return train, holdout


def _write_train_h5ad(dataset_path, train_donors, individual_col, out_path, hvg_path=None):
    """Subset to train donors (using backed='r') and write to disk.

    If hvg_path is given, further filters to HVGs before writing — use this for
    scVI/scDiffusion to avoid loading the full 35k-gene matrix into RAM.

    If the file already exists, checks whether it has the expected gene count.
    A stale full-gene file (from a crashed run before the HVG fix) is overwritten.
    """
    if os.path.exists(out_path):
        existing = ad.read_h5ad(out_path, backed="r")
        n_obs = existing.n_obs
        n_vars_existing = existing.n_vars
        existing.file.close()
        if hvg_path is not None:
            import pandas as pd
            hvg_df = pd.read_csv(hvg_path, index_col=0)
            n_hvg = hvg_df["highly_variable"].sum()
            if n_vars_existing == n_hvg:
                return n_obs  # already filtered correctly
            print(f"  [WARN] {os.path.basename(out_path)} has {n_vars_existing} genes "
                  f"(expected {n_hvg} HVGs) — rewriting with HVG filter.", flush=True)
        else:
            return n_obs
    adata_backed = sc.read_h5ad(dataset_path, backed="r")
    mask = adata_backed.obs[individual_col].isin(set(train_donors))
    subset = adata_backed[mask].to_memory()
    adata_backed.file.close()
    if hvg_path is not None:
        import pandas as pd
        hvg_df = pd.read_csv(hvg_path, index_col=0)
        hvgs = set(hvg_df[hvg_df["highly_variable"]].index)
        keep = [g for g in subset.var_names if g in hvgs]
        subset = subset[:, keep].copy()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    subset.write_h5ad(out_path)
    n, n_vars = subset.n_obs, subset.n_vars
    del subset
    print(f"  Wrote {os.path.basename(out_path)}: {n:,} cells x {n_vars} genes", flush=True)
    return n


def _copy_splits(splits_dir, ds_dir):
    """Copy .npy split files into the output datasets/ directory."""
    os.makedirs(ds_dir, exist_ok=True)
    for fname in ["train.npy", "holdout.npy", "auxiliary.npy"]:
        src = os.path.join(splits_dir, fname)
        dst = os.path.join(ds_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)


def _run(cmd, check=True, **kwargs):
    print(f"  $ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check, **kwargs)


def _latest_checkpoint(directory, pattern="*.pt"):
    """Return the path to the checkpoint with the largest step number."""
    ckpts = glob.glob(os.path.join(directory, pattern))
    if not ckpts:
        raise FileNotFoundError(f"No {pattern} checkpoints found in {directory}")
    # File names include the step: model_seed=0_step=150000.pt or model300000.pt
    def _step(p):
        nums = re.findall(r"\d+", os.path.basename(p))
        return int(nums[-1]) if nums else 0
    return max(ckpts, key=_step)


# ---------------------------------------------------------------------------
# Per-generator generation functions
# ---------------------------------------------------------------------------

def generate_sd3(out_dir, dataset_path, splits_dir, hvg_path,
                 individual_col, cell_type_col, copula_type):
    """Train scDesign3 and generate synthetic data."""
    from sdg.scdesign3.model import ScDesign3

    ds_dir    = os.path.join(out_dir, "datasets")
    synth_out = os.path.join(ds_dir, "synthetic.h5ad")
    if os.path.exists(synth_out):
        print(f"  [SKIP] synthetic.h5ad already exists: {synth_out}")
        return

    train_donors, _ = _load_splits(splits_dir)

    # Pre-filter to HVGs to avoid loading full 35k-gene matrix into R
    train_h5ad = os.path.join(ds_dir, "train.h5ad")
    n_cells = _write_train_h5ad(dataset_path, train_donors, individual_col, train_h5ad,
                                hvg_path=hvg_path)

    trunc_lvl = 1 if copula_type == "vine" else "Inf"

    config = {
        "generator_name": "scdesign3",
        "dir_list": {
            "home": out_dir,
            "data": ds_dir,
        },
        "scdesign3_config": {
            "out_model_path": "models",
            "hvg_path":       hvg_path,
            "copula_type":    copula_type,
            "family_use":     "nb",
            "trunc_lvl":      trunc_lvl,
        },
        "dataset_config": {
            "name":                 "data",
            "train_count_file":     "train.h5ad",
            "test_count_file":      "train.h5ad",
            "cell_type_col_name":   cell_type_col,
            "cell_label_col_name":  "cell_label",
            "random_seed":          42,
        },
    }

    try:
        model = ScDesign3(config)
        print(f"  Training scDesign3 ({copula_type}) ...", flush=True)
        model.train()
        print(f"  Generating {n_cells:,} synthetic cells ...", flush=True)
        synth = model.generate()
        synth.write_h5ad(synth_out)
        print(f"  Saved → {synth_out}")
    finally:
        if os.path.exists(train_h5ad):
            os.remove(train_h5ad)


def generate_scvi(out_dir, dataset_path, splits_dir, hvg_path,
                  individual_col, cell_type_col, conda_env,
                  max_epochs=400, batch_size=512):
    """Train scVI and generate synthetic data."""
    ds_dir      = os.path.join(out_dir, "datasets")
    model_dir   = os.path.join(out_dir, "models", "model")
    synth_out   = os.path.join(ds_dir, "synthetic.h5ad")

    if os.path.exists(synth_out):
        print(f"  [SKIP] synthetic.h5ad already exists: {synth_out}")
        return

    train_donors, _ = _load_splits(splits_dir)

    # Write HVG-filtered train set to avoid loading 35k genes into RAM (OOM on large donors)
    train_h5ad = os.path.join(ds_dir, "train_hvg.h5ad")
    n_cells = _write_train_h5ad(dataset_path, train_donors, individual_col, train_h5ad,
                                hvg_path=hvg_path)

    os.makedirs(model_dir, exist_ok=True)
    scvi_script = os.path.join(_SRC, "sdg", "scvi", "run_scvi_standalone.py")
    conda_prefix = f"conda run --no-capture-output -n {conda_env}"

    try:
        # Train
        _run(f"{conda_prefix} python {scvi_script} train "
             f"{train_h5ad} {model_dir} "
             f"--hvg-path {hvg_path} "
             f"--max-epochs {max_epochs} --batch-size {batch_size}")

        # Generate
        _run(f"{conda_prefix} python {scvi_script} generate "
             f"{train_h5ad} {model_dir} {synth_out} {n_cells}")

        print(f"  Saved → {synth_out}")
    finally:
        if os.path.exists(train_h5ad):
            os.remove(train_h5ad)


def generate_nmf(out_dir, dataset_path, splits_dir, hvg_path,
                 individual_col, cell_type_col, conda_env,
                 n_components=20, dp_mode="none", seed=42, batch_size=1000,
                 dp_eps_nmf=0.5, dp_eps_kmeans=2.1, dp_eps_summaries=0.2):
    """Train NMF-based generator and produce synthetic data."""
    ds_dir    = os.path.join(out_dir, "datasets")
    synth_out = os.path.join(ds_dir, "synthetic.h5ad")

    if os.path.exists(synth_out):
        print(f"  [SKIP] synthetic.h5ad already exists: {synth_out}")
        return

    train_donors, _ = _load_splits(splits_dir)

    os.makedirs(ds_dir, exist_ok=True)
    train_h5ad = os.path.join(ds_dir, "train_hvg.h5ad")
    n_cells = _write_train_h5ad(dataset_path, train_donors, individual_col, train_h5ad,
                                hvg_path=hvg_path)

    nmf_script   = os.path.join(_SRC, "sdg", "nmf", "run_nmf_standalone.py")
    conda_prefix = f"conda run --no-capture-output -n {conda_env}"

    try:
        _run(
            f"{conda_prefix} python {nmf_script} "
            f"--train-h5ad {train_h5ad} "
            f"--output-h5ad {synth_out} "
            f"--n-components {n_components} "
            f"--dp-mode {dp_mode} "
            f"--dp-eps-nmf {dp_eps_nmf} "
            f"--dp-eps-kmeans {dp_eps_kmeans} "
            f"--dp-eps-summaries {dp_eps_summaries} "
            f"--cell-type-col {cell_type_col} "
            f"--seed {seed} "
            f"--batch-size {batch_size}"
        )
        print(f"  Saved → {synth_out}")
    finally:
        if os.path.exists(train_h5ad):
            os.remove(train_h5ad)


def generate_zinbwave(out_dir, dataset_path, splits_dir, hvg_path,
                      individual_col, cell_type_col,
                      n_latent=10, max_cells_per_type=3000, n_workers=4, seed=42):
    """Train ZINBWave and generate synthetic data."""
    ds_dir      = os.path.join(out_dir, "datasets")
    model_dir   = os.path.join(out_dir, "models")
    synth_out   = os.path.join(ds_dir, "synthetic.h5ad")

    if os.path.exists(synth_out):
        print(f"  [SKIP] synthetic.h5ad already exists: {synth_out}")
        return

    train_donors, _ = _load_splits(splits_dir)

    os.makedirs(ds_dir,    exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    train_h5ad = os.path.join(ds_dir, "train_hvg.h5ad")
    _write_train_h5ad(dataset_path, train_donors, individual_col, train_h5ad,
                      hvg_path=hvg_path)

    zinbwave_script = os.path.join(_SRC, "sdg", "zinbwave", "run_zinbwave_standalone.py")

    try:
        _run(
            f"python {zinbwave_script} "
            f"--train-h5ad {train_h5ad} "
            f"--output-h5ad {synth_out} "
            f"--model-dir {model_dir} "
            f"--n-latent {n_latent} "
            f"--max-cells-per-type {max_cells_per_type} "
            f"--cell-type-col {cell_type_col} "
            f"--n-workers {n_workers} "
            f"--seed {seed}"
        )
        print(f"  Saved → {synth_out}")
    finally:
        if os.path.exists(train_h5ad):
            os.remove(train_h5ad)


def generate_scdiffusion(out_dir, dataset_path, splits_dir, hvg_path,
                         individual_col, cell_type_col, conda_env,
                         vae_steps=150000, diff_steps=300000, batch_size=512):
    """
    LEGACY v1 — do NOT call for new experiments.

    Two-stage scDiffusion with post-hoc 1-NN cell-type assignment and
    wrong hyperparameters (vae=150k, diff=300k, bs=512). Kept for
    reproducibility of data in scdiffusion/. Use generate_scdiffusion_v2
    for all new runs.
    """
    ds_dir     = os.path.join(out_dir, "datasets")
    vae_dir    = os.path.join(out_dir, "models", "vae")
    diff_dir   = os.path.join(out_dir, "models", "diff")
    synth_out  = os.path.join(ds_dir, "synthetic.h5ad")

    if os.path.exists(synth_out):
        print(f"  [SKIP] synthetic.h5ad already exists: {synth_out}")
        return

    train_donors, _ = _load_splits(splits_dir)

    train_h5ad = os.path.join(ds_dir, "train_hvg.h5ad")
    n_cells = _write_train_h5ad(dataset_path, train_donors, individual_col, train_h5ad,
                                hvg_path=hvg_path)

    os.makedirs(vae_dir,  exist_ok=True)
    os.makedirs(diff_dir, exist_ok=True)
    scd_script   = os.path.join(_SRC, "sdg", "scdiffusion", "run_scdiffusion_standalone.py")
    conda_prefix = f"conda run --no-capture-output -n {conda_env}"

    try:
        try:
            vae_ckpt = _latest_checkpoint(vae_dir)
            print(f"  [SKIP] VAE checkpoint already exists: {vae_ckpt}", flush=True)
        except FileNotFoundError:
            _run(f"{conda_prefix} python {scd_script} train_vae "
                 f"{train_h5ad} {vae_dir} "
                 f"--hvg-path {hvg_path} "
                 f"--vae-steps {vae_steps} --batch-size {batch_size}")
            vae_ckpt = _latest_checkpoint(vae_dir)
        print(f"  VAE checkpoint: {vae_ckpt}", flush=True)

        diff_subdir = os.path.join(diff_dir, "diffusion")
        _diff_ckpt_exists = False
        try:
            candidate = _latest_checkpoint(diff_subdir, pattern="model*.pt")
            step = int(re.findall(r"\d+", os.path.basename(candidate))[-1])
            if step >= diff_steps:
                _diff_ckpt_exists = True
                diff_ckpt = candidate
                print(f"  [SKIP] Diffusion at step {step}: {diff_ckpt}", flush=True)
        except (FileNotFoundError, IndexError):
            pass
        if not _diff_ckpt_exists:
            _run(f"{conda_prefix} python {scd_script} train "
                 f"{train_h5ad} {vae_ckpt} {diff_dir} "
                 f"--hvg-path {hvg_path} "
                 f"--diff-steps {diff_steps} --batch-size {batch_size}")
            diff_ckpt = _latest_checkpoint(diff_subdir, pattern="model*.pt")
        print(f"  Diff checkpoint: {diff_ckpt}", flush=True)

        _run(f"{conda_prefix} python {scd_script} generate "
             f"{train_h5ad} {vae_ckpt} {diff_ckpt} {synth_out} {n_cells} "
             f"--hvg-path {hvg_path}")

        print(f"  Saved -> {synth_out}", flush=True)
    finally:
        if os.path.exists(train_h5ad):
            os.remove(train_h5ad)


def generate_scdiffusion_v2(out_dir, dataset_path, splits_dir, hvg_path,
                             individual_col, cell_type_col, conda_env,
                             vae_steps=200000, diff_steps=800000, batch_size=128,
                             classifier_steps=200000, classifier_scale=2.0,
                             start_guide_steps=500, generation_batch_size=3000,
                             save_interval=100000):
    """
    Three-stage scDiffusion matching Luo et al. 2024 (v2, 2026-05-04+).

    Stages:
      1. VAE autoencoder       (vae_steps=200k, batch=128)
      2. DDPM backbone         (diff_steps=800k, batch=128)
      3. Cell_classifier       (classifier_steps=200k, batch=128)
    Generation: classifier-guided sampling (classifier_scale=2, start_guide_steps=500).

    Output in <out_dir>/datasets/synthetic.h5ad.
    """
    ds_dir          = os.path.join(out_dir, "datasets")
    vae_dir         = os.path.join(out_dir, "models", "vae")
    diff_dir        = os.path.join(out_dir, "models", "diff")
    classifier_dir  = os.path.join(out_dir, "models", "classifier")
    synth_out       = os.path.join(ds_dir, "synthetic.h5ad")

    if os.path.exists(synth_out):
        print(f"  [SKIP] synthetic.h5ad already exists: {synth_out}")
        return

    train_donors, _ = _load_splits(splits_dir)

    train_h5ad = os.path.join(ds_dir, "train_hvg.h5ad")
    n_cells = _write_train_h5ad(dataset_path, train_donors, individual_col, train_h5ad,
                                hvg_path=hvg_path)

    for d in (vae_dir, diff_dir, classifier_dir):
        os.makedirs(d, exist_ok=True)

    scd_script   = os.path.join(_SRC, "sdg", "scdiffusion", "run_scdiffusion_standalone.py")
    conda_prefix = f"conda run --no-capture-output -n {conda_env}"

    try:
        # ------------------------------------------------------------------
        # Stage 1: VAE
        # ------------------------------------------------------------------
        try:
            vae_ckpt = _latest_checkpoint(vae_dir)
            print(f"  [SKIP] VAE checkpoint: {vae_ckpt}", flush=True)
        except FileNotFoundError:
            _run(f"{conda_prefix} python {scd_script} train_vae "
                 f"{train_h5ad} {vae_dir} "
                 f"--hvg-path {hvg_path} "
                 f"--vae-steps {vae_steps} --batch-size {batch_size} "
                 f"--save-interval {save_interval}")
            vae_ckpt = _latest_checkpoint(vae_dir)
        print(f"  VAE checkpoint: {vae_ckpt}", flush=True)

        # ------------------------------------------------------------------
        # Stage 2: Diffusion backbone
        # ------------------------------------------------------------------
        diff_subdir = os.path.join(diff_dir, "diffusion")
        diff_ckpt   = None
        try:
            candidate = _latest_checkpoint(diff_subdir, pattern="model*.pt")
            step = int(re.findall(r"\d+", os.path.basename(candidate))[-1])
            if step >= diff_steps:
                diff_ckpt = candidate
                print(f"  [SKIP] Diffusion checkpoint at step {step}: {diff_ckpt}", flush=True)
        except (FileNotFoundError, IndexError):
            pass
        if diff_ckpt is None:
            _run(f"{conda_prefix} python {scd_script} train "
                 f"{train_h5ad} {vae_ckpt} {diff_dir} "
                 f"--hvg-path {hvg_path} "
                 f"--diff-steps {diff_steps} --batch-size {batch_size} "
                 f"--save-interval {save_interval}")
            diff_ckpt = _latest_checkpoint(diff_subdir, pattern="model*.pt")
        print(f"  Diff checkpoint: {diff_ckpt}", flush=True)

        # ------------------------------------------------------------------
        # Stage 3: Classifier
        # ------------------------------------------------------------------
        clf_ckpt = None
        try:
            candidate = _latest_checkpoint(classifier_dir, pattern="model*.pt")
            step = int(re.findall(r"\d+", os.path.basename(candidate))[-1])
            if step >= classifier_steps:
                clf_ckpt = candidate
                print(f"  [SKIP] Classifier checkpoint at step {step}: {clf_ckpt}", flush=True)
        except (FileNotFoundError, IndexError):
            pass
        if clf_ckpt is None:
            _run(f"{conda_prefix} python {scd_script} train_classifier "
                 f"{train_h5ad} {vae_ckpt} {classifier_dir} "
                 f"--hvg-path {hvg_path} "
                 f"--classifier-steps {classifier_steps} --batch-size {batch_size} "
                 f"--start-guide-steps {start_guide_steps}")
            clf_ckpt = _latest_checkpoint(classifier_dir, pattern="model*.pt")
        print(f"  Classifier checkpoint: {clf_ckpt}", flush=True)

        # ------------------------------------------------------------------
        # Generate
        # ------------------------------------------------------------------
        _run(f"{conda_prefix} python {scd_script} generate "
             f"{train_h5ad} {vae_ckpt} {diff_ckpt} {clf_ckpt} {synth_out} {n_cells} "
             f"--hvg-path {hvg_path} "
             f"--classifier-scale {classifier_scale} "
             f"--start-guide-steps {start_guide_steps} "
             f"--generation-batch-size {generation_batch_size}")

        print(f"  Saved -> {synth_out}", flush=True)
    finally:
        if os.path.exists(train_h5ad):
            os.remove(train_h5ad)


def generate_scdiffusion_v3(out_dir, dataset_path, splits_dir, hvg_path,
                             individual_col, cell_type_col, conda_env,
                             vae_steps=200000, diff_steps=800000, batch_size=128,
                             generation_batch_size=3000, save_interval=100000,
                             model_source_dir=None):
    """
    Paper-faithful scDiffusion pipeline (v3, 2026-05-04+).

    Two-stage training:
      1. VAE autoencoder       (vae_steps=200k, batch=128)
      2. DDPM backbone         (diff_steps=800k, batch=128)
    Generation: unconditional p_sample_loop (no classifier, no cond_fn,
                clip_denoised=False), matching cell_sample.py from Luo et al.
    Cell type labels: CellTypist trained on D_train, applied to decoded output.

    If model_source_dir is given, VAE and diffusion checkpoints are looked up
    there instead of out_dir (allows reusing completed v2 model weights).
    Output always goes to out_dir/datasets/synthetic.h5ad.
    """
    ds_dir       = os.path.join(out_dir, "datasets")
    synth_out    = os.path.join(ds_dir, "synthetic.h5ad")

    if os.path.exists(synth_out):
        print(f"  [SKIP] synthetic.h5ad already exists: {synth_out}")
        return

    # Model dirs: search model_source_dir first, fall back to out_dir
    model_root      = model_source_dir if model_source_dir else out_dir
    vae_dir         = os.path.join(model_root, "models", "vae")
    diff_dir        = os.path.join(model_root, "models", "diff")
    celltypist_dir  = os.path.join(out_dir,    "models", "celltypist")

    train_donors, _ = _load_splits(splits_dir)
    train_h5ad      = os.path.join(ds_dir, "train_hvg.h5ad")
    n_cells         = _write_train_h5ad(dataset_path, train_donors, individual_col,
                                        train_h5ad, hvg_path=hvg_path)

    os.makedirs(vae_dir,        exist_ok=True)
    os.makedirs(diff_dir,       exist_ok=True)
    os.makedirs(celltypist_dir, exist_ok=True)

    scd_script   = os.path.join(_SRC, "sdg", "scdiffusion", "run_scdiffusion_standalone.py")
    conda_prefix = f"conda run --no-capture-output -n {conda_env}"

    try:
        # ------------------------------------------------------------------
        # Stage 1: VAE
        # ------------------------------------------------------------------
        try:
            vae_ckpt = _latest_checkpoint(vae_dir)
            print(f"  [SKIP] VAE checkpoint: {vae_ckpt}", flush=True)
        except FileNotFoundError:
            _run(f"{conda_prefix} python {scd_script} train_vae "
                 f"{train_h5ad} {vae_dir} "
                 f"--hvg-path {hvg_path} "
                 f"--vae-steps {vae_steps} --batch-size {batch_size} "
                 f"--save-interval {save_interval}")
            vae_ckpt = _latest_checkpoint(vae_dir)
        print(f"  VAE checkpoint: {vae_ckpt}", flush=True)

        # ------------------------------------------------------------------
        # Stage 2: Diffusion backbone (resumes automatically from latest ckpt)
        # ------------------------------------------------------------------
        diff_subdir = os.path.join(diff_dir, "diffusion")
        diff_ckpt   = None
        ema_ckpt    = None
        try:
            candidate = _latest_checkpoint(diff_subdir, pattern="model*.pt")
            step = int(re.findall(r"\d+", os.path.basename(candidate))[-1])
            if step >= diff_steps:
                diff_ckpt = candidate
                print(f"  [SKIP] Diffusion checkpoint at step {step}: {diff_ckpt}",
                      flush=True)
        except (FileNotFoundError, IndexError):
            pass
        if diff_ckpt is None:
            _run(f"{conda_prefix} python {scd_script} train "
                 f"{train_h5ad} {vae_ckpt} {diff_dir} "
                 f"--hvg-path {hvg_path} "
                 f"--diff-steps {diff_steps} --batch-size {batch_size} "
                 f"--save-interval {save_interval}")
            diff_ckpt = _latest_checkpoint(diff_subdir, pattern="model*.pt")

        # Prefer EMA weights for generation (better quality)
        step     = int(re.findall(r"\d+", os.path.basename(diff_ckpt))[-1])
        ema_path = os.path.join(diff_subdir, f"ema_0.9999_{step:06d}.pt")
        ema_ckpt = ema_path if os.path.exists(ema_path) else diff_ckpt
        print(f"  Diff checkpoint for generation: {ema_ckpt}", flush=True)

        # ------------------------------------------------------------------
        # Stage 3: CellTypist (fast CPU logistic regression)
        # ------------------------------------------------------------------
        celltypist_ckpt = os.path.join(celltypist_dir, "model.pkl")
        if os.path.exists(celltypist_ckpt):
            print(f"  [SKIP] CellTypist model: {celltypist_ckpt}", flush=True)
        else:
            _run(f"{conda_prefix} python {scd_script} train_celltypist "
                 f"{train_h5ad} {celltypist_ckpt} "
                 f"--hvg-path {hvg_path} "
                 f"--cell-type-col {cell_type_col}")
        print(f"  CellTypist model: {celltypist_ckpt}", flush=True)

        # ------------------------------------------------------------------
        # Generate: unconditional + CellTypist annotation
        # ------------------------------------------------------------------
        _run(f"{conda_prefix} python {scd_script} generate_unconditional "
             f"{vae_ckpt} {ema_ckpt} {celltypist_ckpt} {synth_out} {n_cells} "
             f"--hvg-path {hvg_path} "
             f"--cell-type-col {cell_type_col} "
             f"--generation-batch-size {generation_batch_size}")

        print(f"  Saved -> {synth_out}", flush=True)
    finally:
        if os.path.exists(train_h5ad):
            os.remove(train_h5ad)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate synthetic data for one SDG trial")
    ap.add_argument("--generator",      required=True,
                    choices=["sd3_gaussian", "sd3_vine", "scvi",
                             "scdiffusion",     # LEGACY v1 — do not use for new data
                             "scdiffusion_v2",  # LEGACY v2 — classifier-guided (wrong mode)
                             "scdiffusion_v3",  # paper-faithful: unconditional + CellTypist
                             "nmf", "zinbwave"])
    ap.add_argument("--dataset",        required=True,
                    help="Path to full_dataset_cleaned.h5ad")
    ap.add_argument("--splits-dir",     required=True,
                    help="Path to a datasets/ folder with train.npy, holdout.npy")
    ap.add_argument("--out-dir",        required=True,
                    help="Trial output directory")
    ap.add_argument("--hvg-path",       required=True,
                    help="Path to hvg_full.csv")
    ap.add_argument("--individual-col", default="individual")
    ap.add_argument("--cell-type-col",  default="cell_type")
    ap.add_argument("--conda-env",      default=None,
                    help="Conda env for scVI or scDiffusion")
    # scVI options
    ap.add_argument("--max-epochs",  type=int, default=400)
    ap.add_argument("--batch-size",  type=int, default=512,
                    help="scVI batch size (default: 512)")
    # scDiffusion v2 options (paper-matching defaults)
    ap.add_argument("--vae-steps",              type=int,   default=200000,
                    help="VAE training steps (paper: 200000)")
    ap.add_argument("--diff-steps",             type=int,   default=800000,
                    help="Diffusion backbone steps (paper: 800000)")
    ap.add_argument("--classifier-steps",       type=int,   default=200000,
                    help="Classifier training steps (paper: 400k-500k)")
    ap.add_argument("--classifier-scale",       type=float, default=2.0,
                    help="Guidance scale (paper: 2.0)")
    ap.add_argument("--start-guide-steps",      type=int,   default=500,
                    help="Guidance active for t < this (paper: 500)")
    ap.add_argument("--generation-batch-size",  type=int,   default=3000,
                    help="Sampling batch size (paper: 3000)")
    ap.add_argument("--scd-batch-size",          type=int,   default=128,
                    help="scDiffusion v2 batch size (paper: 128; default: 128)")
    ap.add_argument("--scd-save-interval",      type=int,   default=100000,
                    help="Checkpoint save frequency for scDiffusion")
    ap.add_argument("--model-source-dir",       default=None,
                    help="scdiffusion_v3 only: reuse VAE+diff checkpoints from this "
                         "directory (e.g. a completed scdiffusion_v2 trial dir) "
                         "instead of training from scratch")
    # NMF options
    ap.add_argument("--n-components", type=int, default=20,
                    help="NMF latent components (default: 20)")
    ap.add_argument("--dp-mode",
                    choices=["all", "nmf", "kmeans", "sampling", "none"],
                    default="none",
                    help="NMF DP noise stages (default: none)")
    ap.add_argument("--nmf-seed",    type=int, default=42)
    ap.add_argument("--dp-eps-nmf",       type=float, default=0.5,
                    help="NMF basis DP epsilon (used only when --dp-mode includes nmf)")
    ap.add_argument("--dp-eps-kmeans",    type=float, default=2.1,
                    help="KMeans centroid DP epsilon (used only when --dp-mode includes kmeans)")
    ap.add_argument("--dp-eps-summaries", type=float, default=0.2,
                    help="Cluster summary DP epsilon (used only when --dp-mode includes sampling)")
    # ZINBWave options
    ap.add_argument("--n-latent",         type=int, default=10,
                    help="ZINBWave latent factors K (default: 10)")
    ap.add_argument("--max-cells-per-type", type=int, default=3000,
                    help="Max cells per cell type for zinbwave fitting (default: 3000)")
    ap.add_argument("--zinbwave-workers", type=int, default=4,
                    help="Parallel R workers for ZINBWave (default: 4)")
    ap.add_argument("--zinbwave-seed",    type=int, default=42,
                    help="Random seed for ZINBWave generation (default: 42)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n{'='*60}", flush=True)
    print(f"Generator: {args.generator}  |  out: {args.out_dir}", flush=True)

    if args.generator in ("sd3_gaussian", "sd3_vine"):
        copula_type = args.generator.split("_")[1]
        generate_sd3(
            out_dir=args.out_dir,
            dataset_path=args.dataset,
            splits_dir=args.splits_dir,
            hvg_path=args.hvg_path,
            individual_col=args.individual_col,
            cell_type_col=args.cell_type_col,
            copula_type=copula_type,
        )

    elif args.generator == "scvi":
        if not args.conda_env:
            ap.error("--conda-env is required for scvi")
        generate_scvi(
            out_dir=args.out_dir,
            dataset_path=args.dataset,
            splits_dir=args.splits_dir,
            hvg_path=args.hvg_path,
            individual_col=args.individual_col,
            cell_type_col=args.cell_type_col,
            conda_env=args.conda_env,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
        )

    elif args.generator == "scdiffusion":
        # LEGACY v1 — produces data in scdiffusion/ dirs
        if not args.conda_env:
            ap.error("--conda-env is required for scdiffusion")
        generate_scdiffusion(
            out_dir=args.out_dir,
            dataset_path=args.dataset,
            splits_dir=args.splits_dir,
            hvg_path=args.hvg_path,
            individual_col=args.individual_col,
            cell_type_col=args.cell_type_col,
            conda_env=args.conda_env,
            vae_steps=args.vae_steps,
            diff_steps=args.diff_steps,
            batch_size=args.batch_size,
        )

    elif args.generator == "scdiffusion_v2":
        # v2 LEGACY: classifier-guided generation (kept for reference)
        if not args.conda_env:
            ap.error("--conda-env is required for scdiffusion_v2")
        generate_scdiffusion_v2(
            out_dir=args.out_dir,
            dataset_path=args.dataset,
            splits_dir=args.splits_dir,
            hvg_path=args.hvg_path,
            individual_col=args.individual_col,
            cell_type_col=args.cell_type_col,
            conda_env=args.conda_env,
            vae_steps=args.vae_steps,
            diff_steps=args.diff_steps,
            batch_size=args.scd_batch_size,
            classifier_steps=args.classifier_steps,
            classifier_scale=args.classifier_scale,
            start_guide_steps=args.start_guide_steps,
            generation_batch_size=args.generation_batch_size,
            save_interval=args.scd_save_interval,
        )

    elif args.generator == "scdiffusion_v3":
        # v3: paper-faithful unconditional generation + CellTypist annotation
        if not args.conda_env:
            ap.error("--conda-env is required for scdiffusion_v3")
        generate_scdiffusion_v3(
            out_dir=args.out_dir,
            dataset_path=args.dataset,
            splits_dir=args.splits_dir,
            hvg_path=args.hvg_path,
            individual_col=args.individual_col,
            cell_type_col=args.cell_type_col,
            conda_env=args.conda_env,
            vae_steps=args.vae_steps,
            diff_steps=args.diff_steps,
            batch_size=args.scd_batch_size,
            generation_batch_size=args.generation_batch_size,
            save_interval=args.scd_save_interval,
            model_source_dir=args.model_source_dir,
        )

    elif args.generator == "nmf":
        if not args.conda_env:
            ap.error("--conda-env is required for nmf")
        generate_nmf(
            out_dir=args.out_dir,
            dataset_path=args.dataset,
            splits_dir=args.splits_dir,
            hvg_path=args.hvg_path,
            individual_col=args.individual_col,
            cell_type_col=args.cell_type_col,
            conda_env=args.conda_env,
            n_components=args.n_components,
            dp_mode=args.dp_mode,
            seed=args.nmf_seed,
            batch_size=args.batch_size,
            dp_eps_nmf=args.dp_eps_nmf,
            dp_eps_kmeans=args.dp_eps_kmeans,
            dp_eps_summaries=args.dp_eps_summaries,
        )

    elif args.generator == "zinbwave":
        generate_zinbwave(
            out_dir=args.out_dir,
            dataset_path=args.dataset,
            splits_dir=args.splits_dir,
            hvg_path=args.hvg_path,
            individual_col=args.individual_col,
            cell_type_col=args.cell_type_col,
            n_latent=args.n_latent,
            max_cells_per_type=args.max_cells_per_type,
            n_workers=args.zinbwave_workers,
            seed=args.zinbwave_seed,
        )

    print(f"Done: {args.generator}  {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

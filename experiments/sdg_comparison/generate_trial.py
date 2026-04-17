"""
Generate synthetic scRNA-seq data for one trial of one SDG method.

Loads pre-existing donor splits from a scDesign2 trial directory, trains the
target generator on D_train, and saves the synthetic output.  No MIA attack
is run — this is generation only.

Supported generators
--------------------
  sd3_gaussian   scDesign3 with Gaussian copula
  sd3_vine       scDesign3 with vine copula
  scvi           scVI VAE
  scdiffusion    scDiffusion (VAE + diffusion backbone)
  nmf            SingleCellNMFGenerator (NMF + KMeans + ZINB sampling)

Output layout
-------------
  <out-dir>/
    datasets/
      train.npy           copied donor ID array (reference for later attacks)
      holdout.npy         copied donor ID array
      train.h5ad          temporary training subset (deleted after generation)
      synthetic.h5ad      final output
    models/
      <cell_type>.rds     (scDesign3 only)
      model/              (scVI only)
      vae/  diff/         (scDiffusion only)

Usage
-----
  # scDesign3 Gaussian, OneK1K, 10 donors, trial 3
  python experiments/sdg_comparison/generate_trial.py \\
      --generator  sd3_gaussian \\
      --dataset    /home/golobs/data/ok/full_dataset_cleaned.h5ad \\
      --splits-dir /home/golobs/data/ok/10d/3/datasets \\
      --out-dir    /home/golobs/data/ok_sd3g/10d/3 \\
      --hvg-path   /home/golobs/data/ok/hvg_full.csv

  # scVI
  python experiments/sdg_comparison/generate_trial.py \\
      --generator scvi \\
      --dataset   /home/golobs/data/ok/full_dataset_cleaned.h5ad \\
      --splits-dir /home/golobs/data/ok/10d/3/datasets \\
      --out-dir    /home/golobs/data/ok_scvi/10d/3 \\
      --hvg-path   /home/golobs/data/ok/hvg_full.csv \\
      --conda-env  scvi_
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

    _copy_splits(splits_dir, ds_dir)
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

    model = ScDesign3(config)
    print(f"  Training scDesign3 ({copula_type}) ...", flush=True)
    model.train()
    print(f"  Generating {n_cells:,} synthetic cells ...", flush=True)
    synth = model.generate()
    synth.write_h5ad(synth_out)
    print(f"  Saved → {synth_out}")

    # Clean up temporary train.h5ad (it's large; splits are already saved)
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

    _copy_splits(splits_dir, ds_dir)
    train_donors, _ = _load_splits(splits_dir)

    # Write HVG-filtered train set to avoid loading 35k genes into RAM (OOM on large donors)
    train_h5ad = os.path.join(ds_dir, "train_hvg.h5ad")
    n_cells = _write_train_h5ad(dataset_path, train_donors, individual_col, train_h5ad,
                                hvg_path=hvg_path)

    os.makedirs(model_dir, exist_ok=True)
    scvi_script = os.path.join(_SRC, "sdg", "scvi", "run_scvi_standalone.py")
    conda_prefix = f"conda run --no-capture-output -n {conda_env}"

    # Train
    _run(f"{conda_prefix} python {scvi_script} train "
         f"{train_h5ad} {model_dir} "
         f"--hvg-path {hvg_path} "
         f"--max-epochs {max_epochs} --batch-size {batch_size}")

    # Generate
    _run(f"{conda_prefix} python {scvi_script} generate "
         f"{train_h5ad} {model_dir} {synth_out} {n_cells}")

    print(f"  Saved → {synth_out}")
    if os.path.exists(train_h5ad):
        os.remove(train_h5ad)


def generate_nmf(out_dir, dataset_path, splits_dir, hvg_path,
                 individual_col, cell_type_col, conda_env,
                 n_components=20, dp_mode="none", seed=42, batch_size=1000):
    """Train NMF-based generator and produce synthetic data."""
    ds_dir    = os.path.join(out_dir, "datasets")
    synth_out = os.path.join(ds_dir, "synthetic.h5ad")

    if os.path.exists(synth_out):
        print(f"  [SKIP] synthetic.h5ad already exists: {synth_out}")
        return

    _copy_splits(splits_dir, ds_dir)
    train_donors, _ = _load_splits(splits_dir)

    train_h5ad = os.path.join(ds_dir, "train_hvg.h5ad")
    n_cells = _write_train_h5ad(dataset_path, train_donors, individual_col, train_h5ad,
                                hvg_path=hvg_path)

    nmf_script   = os.path.join(_SRC, "sdg", "nmf", "run_nmf_standalone.py")
    conda_prefix = f"conda run --no-capture-output -n {conda_env}"

    _run(
        f"{conda_prefix} python {nmf_script} "
        f"--train-h5ad {train_h5ad} "
        f"--output-h5ad {synth_out} "
        f"--n-components {n_components} "
        f"--dp-mode {dp_mode} "
        f"--cell-type-col {cell_type_col} "
        f"--seed {seed} "
        f"--batch-size {batch_size}"
    )

    print(f"  Saved → {synth_out}")
    if os.path.exists(train_h5ad):
        os.remove(train_h5ad)


def generate_scdiffusion(out_dir, dataset_path, splits_dir, hvg_path,
                         individual_col, cell_type_col, conda_env,
                         vae_steps=150000, diff_steps=300000, batch_size=512):
    """Train scDiffusion (VAE + diffusion) and generate synthetic data."""
    ds_dir     = os.path.join(out_dir, "datasets")
    vae_dir    = os.path.join(out_dir, "models", "vae")
    diff_dir   = os.path.join(out_dir, "models", "diff")
    synth_out  = os.path.join(ds_dir, "synthetic.h5ad")

    if os.path.exists(synth_out):
        print(f"  [SKIP] synthetic.h5ad already exists: {synth_out}")
        return

    _copy_splits(splits_dir, ds_dir)
    train_donors, _ = _load_splits(splits_dir)

    # Write HVG-filtered train set to avoid loading 35k genes into RAM (OOM on large donors)
    train_h5ad = os.path.join(ds_dir, "train_hvg.h5ad")
    n_cells = _write_train_h5ad(dataset_path, train_donors, individual_col, train_h5ad,
                                hvg_path=hvg_path)

    os.makedirs(vae_dir,  exist_ok=True)
    os.makedirs(diff_dir, exist_ok=True)
    scd_script  = os.path.join(_SRC, "sdg", "scdiffusion", "run_scdiffusion_standalone.py")
    conda_prefix = f"conda run --no-capture-output -n {conda_env}"

    # Train VAE (skip if checkpoint already exists)
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

    # Train diffusion (skip only if final checkpoint at diff_steps already exists)
    diff_subdir = os.path.join(diff_dir, "diffusion")
    _diff_ckpt_exists = False
    try:
        candidate = _latest_checkpoint(diff_subdir, pattern="model*.pt")
        step = int(re.findall(r"\d+", os.path.basename(candidate))[-1])
        if step >= diff_steps:
            _diff_ckpt_exists = True
            diff_ckpt = candidate
            print(f"  [SKIP] Diffusion checkpoint already at step {step}: {diff_ckpt}", flush=True)
    except (FileNotFoundError, IndexError):
        pass
    if not _diff_ckpt_exists:
        _run(f"{conda_prefix} python {scd_script} train "
             f"{train_h5ad} {vae_ckpt} {diff_dir} "
             f"--hvg-path {hvg_path} "
             f"--diff-steps {diff_steps} --batch-size {batch_size}")
        # Use model*.pt (not ema_*.pt or opt*.pt) — the model weights, not EMA/optimizer state
        diff_ckpt = _latest_checkpoint(diff_subdir, pattern="model*.pt")
    print(f"  Diff checkpoint: {diff_ckpt}", flush=True)

    # Generate
    _run(f"{conda_prefix} python {scd_script} generate "
         f"{train_h5ad} {vae_ckpt} {diff_ckpt} {synth_out} {n_cells} "
         f"--hvg-path {hvg_path}")

    print(f"  Saved → {synth_out}", flush=True)
    if os.path.exists(train_h5ad):
        os.remove(train_h5ad)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate synthetic data for one SDG trial")
    ap.add_argument("--generator",      required=True,
                    choices=["sd3_gaussian", "sd3_vine", "scvi", "scdiffusion", "nmf"])
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
    ap.add_argument("--batch-size",  type=int, default=512)
    # scDiffusion options
    ap.add_argument("--vae-steps",   type=int, default=150000)
    ap.add_argument("--diff-steps",  type=int, default=300000)
    # NMF options
    ap.add_argument("--n-components", type=int, default=20,
                    help="NMF latent components (default: 20)")
    ap.add_argument("--dp-mode",
                    choices=["all", "nmf", "kmeans", "sampling", "none"],
                    default="none",
                    help="NMF DP noise stages (default: none)")
    ap.add_argument("--nmf-seed",    type=int, default=42)
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
        )

    print(f"Done: {args.generator}  {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

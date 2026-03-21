"""
Standalone scVI runner — executed inside the scvi_ conda environment.

This script is called via subprocess from src/sdg/scvi/model.py.
It handles three modes:

  train   <train_h5ad> <model_dir> [options]
      Fit an SCVI model on the training data and save it to model_dir/.

  generate <train_h5ad> <model_dir> <out_h5ad> <n_cells>
      Load a saved model and sample n_cells synthetic cells, writing
      an h5ad with the same gene vocabulary as the training data.

  score    <target_h5ad> <model_dir> <scores_npy>
      Load a saved model and compute per-cell ELBO for every cell in
      target_h5ad.  Higher ELBO = better fit = more likely a member.
      Saves a float32 numpy array of shape (n_cells,) to scores_npy.

Usage examples
--------------
  python run_scvi_standalone.py train  train.h5ad  model_dir/
  python run_scvi_standalone.py generate  train.h5ad  model_dir/  synth.h5ad  5000
  python run_scvi_standalone.py score   target.h5ad  model_dir/  scores.npy
"""

import argparse
import sys
import os

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch

# ---------------------------------------------------------------------------
# PyTorch 2.6+ changed the default of weights_only from False to True, which
# breaks scvi-tools model loading (saved checkpoints include numpy metadata).
# We patch torch.load to restore the old default for our trusted internal
# models (we only load checkpoints that we trained ourselves in this script).
# ---------------------------------------------------------------------------
_orig_torch_load = torch.load

def _load_compat(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(f, *args, **kwargs)

torch.load = _load_compat


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _prepare_adata(adata: ad.AnnData) -> ad.AnnData:
    """
    Ensure raw integer counts are in adata.layers["counts"], which is
    what scvi.model.SCVI.setup_anndata expects.
    """
    import scipy.sparse as sp

    adata = adata.copy()
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    adata.layers["counts"] = X.astype(np.float32)
    return adata


def _load_and_prepare(h5ad_path: str) -> ad.AnnData:
    adata = sc.read_h5ad(h5ad_path)
    return _prepare_adata(adata)


def _select_hvgs(adata: ad.AnnData, hvg_path: str = None,
                 min_mean: float = 0.0125, max_mean: float = 3.0,
                 min_disp: float = 0.5) -> ad.AnnData:
    """
    Subset to HVGs.  If hvg_path is given and exists, load the gene list
    from there.  Otherwise select HVGs from the current data and optionally
    save the result.

    Falls back to all genes if fewer than 10 HVGs are found (e.g., test data).
    """
    if hvg_path and os.path.exists(hvg_path):
        hvg_df = pd.read_csv(hvg_path, index_col=0)
        hvg_genes = hvg_df.index[hvg_df["highly_variable"]].tolist()
        hvg_genes = [g for g in hvg_genes if g in adata.var_names]
        print(f"Loaded {len(hvg_genes)} HVGs from {hvg_path}")
        return adata[:, hvg_genes].copy()

    # Compute HVGs from scratch on a normalised/log-transformed copy
    tmp = adata.copy()
    sc.pp.normalize_total(tmp, layer="counts", target_sum=1e4)
    sc.pp.log1p(tmp, layer="counts")
    sc.pp.highly_variable_genes(
        tmp, layer="counts",
        min_mean=min_mean, max_mean=max_mean, min_disp=min_disp
    )
    hvg_mask = tmp.var["highly_variable"]
    n_hvg = int(hvg_mask.sum())
    print(f"Selected {n_hvg} HVGs from data")

    if n_hvg < 10:
        print(f"  [WARN] Fewer than 10 HVGs found ({n_hvg}); using all {adata.n_vars} genes.")
        hvg_mask[:] = True

    if hvg_path:
        os.makedirs(os.path.dirname(os.path.abspath(hvg_path)), exist_ok=True)
        hvg_mask.to_csv(hvg_path)
        print(f"Saved HVG mask to {hvg_path}")

    return adata[:, hvg_mask].copy()


# ---------------------------------------------------------------------------
# Per-cell ELBO computation
# ---------------------------------------------------------------------------

def _get_per_cell_elbo(model, adata: ad.AnnData, batch_size: int = 256) -> np.ndarray:
    """
    Compute per-cell ELBO using the trained SCVI model's internal
    data loader and module forward pass.

    In scvi-tools 1.2.x, LossOutput fields are dicts:
      - reconstruction_loss['reconstruction_loss'] : (batch,) Tensor
      - kl_local['kl_divergence_z'] + kl_local['kl_divergence_l'] : (batch,) each

    Returns an array of shape (n_cells,) where higher values indicate
    better fit to the model (i.e., more likely to be a training member).
    """
    device = next(model.module.parameters()).device
    model.module.eval()

    scdl = model._make_data_loader(adata=adata, batch_size=batch_size)

    all_elbo = []

    with torch.no_grad():
        for tensors in scdl:
            tensors = {k: v.to(device) for k, v in tensors.items()}

            # Forward pass
            inf_inputs = model.module._get_inference_input(tensors)
            inf_out    = model.module.inference(**inf_inputs)
            gen_inputs = model.module._get_generative_input(tensors, inf_out)
            gen_out    = model.module.generative(**gen_inputs)
            loss_out   = model.module.loss(
                tensors, inf_out, gen_out, kl_weight=1.0
            )

            # reconstruction_loss is a dict in 1.2.x, a Tensor in older versions
            recon_raw = loss_out.reconstruction_loss
            if isinstance(recon_raw, dict):
                # Primary key is 'reconstruction_loss'
                recon = recon_raw.get(
                    "reconstruction_loss",
                    next(iter(recon_raw.values()))
                )
            else:
                recon = recon_raw  # older API: already a Tensor

            # kl_local is also a dict (kl_divergence_z, kl_divergence_l)
            kl_raw = loss_out.kl_local
            if isinstance(kl_raw, dict):
                kl = torch.stack(list(kl_raw.values()), dim=-1).sum(-1)
            elif isinstance(kl_raw, torch.Tensor):
                kl = kl_raw
            else:
                kl = torch.zeros_like(recon)

            # ELBO = -(reconstruction_loss + kl_local); higher → better fit → member
            elbo = -(recon + kl).cpu().float().numpy()  # (batch,)
            all_elbo.append(elbo)

    return np.concatenate(all_elbo, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Mode: train
# ---------------------------------------------------------------------------

def cmd_train(args):
    import scvi

    adata = _load_and_prepare(args.train_h5ad)
    adata = _select_hvgs(adata, hvg_path=args.hvg_path)
    print(f"Training on {adata.n_obs} cells × {adata.n_vars} HVGs")

    scvi.model.SCVI.setup_anndata(adata, layer="counts")
    model = scvi.model.SCVI(
        adata,
        n_latent=args.n_latent,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        dropout_rate=0.1,
        gene_likelihood="nb",
    )

    use_gpu = torch.cuda.is_available()
    model.train(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        early_stopping=True,
        early_stopping_patience=20,
        plan_kwargs={"lr": 1e-3},
        accelerator="gpu" if use_gpu else "cpu",
    )

    os.makedirs(args.model_dir, exist_ok=True)
    model.save(args.model_dir, overwrite=True)
    print(f"Model saved to {args.model_dir}")

    # Save gene list for generate / score step
    gene_list_path = os.path.join(args.model_dir, "gene_names.txt")
    with open(gene_list_path, "w") as fh:
        fh.write("\n".join(adata.var_names.tolist()))
    print(f"Gene list saved to {gene_list_path}")


# ---------------------------------------------------------------------------
# Mode: generate
# ---------------------------------------------------------------------------

def cmd_generate(args):
    import scvi

    adata_train = _load_and_prepare(args.train_h5ad)

    # Subset to the same genes used at training time
    gene_list_path = os.path.join(args.model_dir, "gene_names.txt")
    with open(gene_list_path) as fh:
        gene_names = [line.strip() for line in fh if line.strip()]
    gene_names = [g for g in gene_names if g in adata_train.var_names]
    adata_train = adata_train[:, gene_names].copy()

    # Load model
    use_gpu = torch.cuda.is_available()
    model = scvi.model.SCVI.load(
        args.model_dir,
        adata=adata_train,
        accelerator="gpu" if use_gpu else "cpu",
    )
    model.module.eval()

    n_cells = int(args.n_cells)
    print(f"Sampling {n_cells} synthetic cells ...")

    # scvi-tools sample_from_posterior gives latent → generative → counts
    synth = model.posterior_predictive_sample(
        adata_train,
        n_samples=1,
        gene_list=gene_names,
    )  # shape: (n_training_cells, n_genes) — we subsample to n_cells

    # synth is a numpy array; wrap in AnnData
    indices = np.random.choice(synth.shape[0], size=min(n_cells, synth.shape[0]),
                               replace=n_cells > synth.shape[0])
    synth_counts = synth[indices, :]

    synth_adata = ad.AnnData(
        X=synth_counts,
        var=pd.DataFrame(index=gene_names),
    )
    if "cell_type" in adata_train.obs.columns:
        synth_adata.obs["cell_type"] = adata_train.obs["cell_type"].values[indices]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_h5ad)), exist_ok=True)
    synth_adata.write_h5ad(args.out_h5ad, compression="gzip")
    print(f"Synthetic data ({synth_adata.n_obs} cells) saved to {args.out_h5ad}")


# ---------------------------------------------------------------------------
# Mode: score
# ---------------------------------------------------------------------------

def cmd_score(args):
    import scvi

    adata_target = _load_and_prepare(args.target_h5ad)

    # Subset to the same genes used at training time
    gene_list_path = os.path.join(args.model_dir, "gene_names.txt")
    with open(gene_list_path) as fh:
        gene_names = [line.strip() for line in fh if line.strip()]
    gene_names = [g for g in gene_names if g in adata_target.var_names]
    adata_target = adata_target[:, gene_names].copy()

    print(f"Scoring {adata_target.n_obs} cells × {adata_target.n_vars} genes")

    use_gpu = torch.cuda.is_available()
    model = scvi.model.SCVI.load(
        args.model_dir,
        adata=adata_target,
        accelerator="gpu" if use_gpu else "cpu",
    )

    per_cell_elbo = _get_per_cell_elbo(model, adata_target)
    print(f"  ELBO  mean={per_cell_elbo.mean():.2f}  "
          f"std={per_cell_elbo.std():.2f}  "
          f"range=[{per_cell_elbo.min():.1f}, {per_cell_elbo.max():.1f}]")

    os.makedirs(os.path.dirname(os.path.abspath(args.scores_npy)), exist_ok=True)
    np.save(args.scores_npy, per_cell_elbo)
    print(f"Per-cell ELBO scores saved to {args.scores_npy}  shape={per_cell_elbo.shape}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Standalone scVI runner (train / generate / score)"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # --- train ---
    p_train = sub.add_parser("train")
    p_train.add_argument("train_h5ad",  help="Path to training AnnData (.h5ad)")
    p_train.add_argument("model_dir",   help="Directory where the model will be saved")
    p_train.add_argument("--hvg-path",  default=None,
                         help="Path to HVG CSV (gene × highly_variable bool). "
                              "If missing, HVGs are computed and saved here.")
    p_train.add_argument("--n-latent",  type=int, default=30)
    p_train.add_argument("--n-layers",  type=int, default=2)
    p_train.add_argument("--n-hidden",  type=int, default=128)
    p_train.add_argument("--max-epochs", type=int, default=400)
    p_train.add_argument("--batch-size", type=int, default=512)

    # --- generate ---
    p_gen = sub.add_parser("generate")
    p_gen.add_argument("train_h5ad", help="Training AnnData (for gene vocabulary)")
    p_gen.add_argument("model_dir",  help="Saved model directory")
    p_gen.add_argument("out_h5ad",   help="Output synthetic AnnData path")
    p_gen.add_argument("n_cells",    help="Number of synthetic cells to generate")

    # --- score ---
    p_score = sub.add_parser("score")
    p_score.add_argument("target_h5ad", help="Target AnnData to score")
    p_score.add_argument("model_dir",   help="Saved model directory")
    p_score.add_argument("scores_npy",  help="Output path for per-cell ELBO scores (.npy)")

    args = parser.parse_args()

    if args.mode == "train":
        cmd_train(args)
    elif args.mode == "generate":
        cmd_generate(args)
    elif args.mode == "score":
        cmd_score(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

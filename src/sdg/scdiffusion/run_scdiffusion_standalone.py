"""
Standalone scDiffusion runner — executed inside the scdiff_ conda environment.

Called via subprocess from src/sdg/scdiffusion/model.py.

Repo root: /home/golobs/scDiffusion  (Luo et al. 2024, Bioinformatics)

Pipeline
--------
  train_vae   <train_h5ad> <vae_out> [options]
      Train the VAE autoencoder from scratch.
      Output: <vae_out>/model_seed=0_step=<N>.pt

  train       <train_h5ad> <vae_ckpt> <diff_out> [options]
      Train the diffusion backbone in VAE latent space.
      Output: <diff_out>/model<N>.pt

  generate    <train_h5ad> <vae_ckpt> <diff_ckpt> <out_h5ad> <n_cells>
      Sample latent codes from the diffusion model, decode through the VAE,
      and write an h5ad of synthetic gene expression.

  score       <target_h5ad> <vae_ckpt> <diff_ckpt> <scores_npy>
      Compute per-cell diffusion denoising loss as the MIA membership score.
      Higher score (lower loss) = more likely a training member.
      Output: float32 numpy array of shape (n_cells,) saved to scores_npy.

Key options (see --help for each subcommand)
--------------------------------------------
  --cell-type-col   obs column for cell type labels  (default: cell_type)
  --hvg-path        pre-computed HVG CSV (highly_variable bool); if given,
                    uses our standard HVGs instead of scDiffusion's own filter
  --latent-dim      VAE latent dimension            (default: 128)
  --vae-steps       VAE training iterations         (default: 150000)
  --diff-steps      diffusion training iterations   (default: 300000)
  --n-score-times   noise levels sampled for MIA    (default: 50)
"""

import argparse
import os
import sys

import numpy as np
import torch
import scanpy as sc
import anndata as ad
import pandas as pd

# ---------------------------------------------------------------------------
# scDiffusion repo on sys.path
# ---------------------------------------------------------------------------
SCDIFF_ROOT = "/home/golobs/scDiffusion"
if SCDIFF_ROOT not in sys.path:
    sys.path.insert(0, SCDIFF_ROOT)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _preprocess(adata, cell_type_col, hvg_path=None):
    """
    Normalise + log1p (scDiffusion standard), optionally subset to HVGs.
    Returns (cell_data np.ndarray float32, cell_types np.ndarray int).
    """
    import scipy.sparse as sp
    from sklearn.preprocessing import LabelEncoder

    adata = adata.copy()
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=10)
    adata.var_names_make_unique()

    if hvg_path and os.path.exists(hvg_path):
        hvg_df = pd.read_csv(hvg_path, index_col=0)
        hvg_genes = [g for g in hvg_df.index[hvg_df["highly_variable"]]
                     if g in adata.var_names]
        adata = adata[:, hvg_genes].copy()
        print(f"Using {len(hvg_genes)} pre-computed HVGs")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    cell_data = X.astype(np.float32)

    le = LabelEncoder()
    cell_types = le.fit_transform(adata.obs[cell_type_col].values).astype(np.int64)

    return cell_data, cell_types, adata.var_names.tolist()


def _load_vae(vae_ckpt, n_genes, latent_dim, device):
    from VAE.VAE_model import VAE
    autoencoder = VAE(
        num_genes=n_genes,
        device=str(device),
        seed=0,
        loss_ae="mse",
        hidden_dim=latent_dim,
        decoder_activation="ReLU",
    )
    ckpt = torch.load(vae_ckpt, map_location=device)
    autoencoder.load_state_dict(ckpt)
    autoencoder.to(device)
    autoencoder.eval()
    return autoencoder


def _load_diffusion_model(diff_ckpt, latent_dim, device):
    from guided_diffusion.script_util import (
        model_and_diffusion_defaults,
        create_model_and_diffusion,
    )
    defaults = model_and_diffusion_defaults()
    defaults["input_dim"] = latent_dim
    model, diffusion = create_model_and_diffusion(**defaults)
    ckpt = torch.load(diff_ckpt, map_location=device)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model, diffusion


# ---------------------------------------------------------------------------
# Mode: train_vae
# ---------------------------------------------------------------------------

def cmd_train_vae(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adata = sc.read_h5ad(args.train_h5ad)
    cell_data, _, gene_names = _preprocess(
        adata, args.cell_type_col, args.hvg_path
    )
    n_genes = cell_data.shape[1]
    print(f"Training VAE: {len(cell_data)} cells × {n_genes} genes  latent_dim={args.latent_dim}")

    from VAE.VAE_model import VAE
    from torch.utils.data import TensorDataset, DataLoader

    os.makedirs(args.vae_out, exist_ok=True)

    # Save gene list for downstream steps
    with open(os.path.join(args.vae_out, "gene_names.txt"), "w") as f:
        f.write("\n".join(gene_names))

    vae = VAE(
        num_genes=n_genes, device=str(device), seed=0,
        loss_ae="mse", hidden_dim=args.latent_dim, decoder_activation="ReLU",
    )
    vae.to(device)

    X_tensor = torch.tensor(cell_data)
    ds = TensorDataset(X_tensor)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    step = 0
    while step < args.vae_steps:
        for (batch,) in loader:
            if step >= args.vae_steps:
                break
            loss_dict = vae.train_step(batch.to(device))
            if step % 10000 == 0:
                print(f"  VAE step {step}/{args.vae_steps}  "
                      f"loss={loss_dict['loss_reconstruction']:.4f}", flush=True)
            step += 1

    ckpt_path = os.path.join(args.vae_out, f"model_seed=0_step={step}.pt")
    torch.save(vae.state_dict(), ckpt_path)
    print(f"VAE saved: {ckpt_path}")


# ---------------------------------------------------------------------------
# Mode: train (diffusion backbone)
# ---------------------------------------------------------------------------

def cmd_train(args):
    """Train the diffusion backbone using dist_util (requires mpi4py)."""
    from guided_diffusion import dist_util, logger
    from guided_diffusion.cell_datasets_loader import load_data, load_VAE
    from guided_diffusion.resample import create_named_schedule_sampler
    from guided_diffusion.script_util import (
        model_and_diffusion_defaults,
        create_model_and_diffusion,
        args_to_dict,
    )
    from guided_diffusion.train_util import TrainLoop

    dist_util.setup_dist()
    os.makedirs(args.diff_out, exist_ok=True)
    logger.configure(dir=os.path.join(args.diff_out, "logs"))

    defaults = model_and_diffusion_defaults()
    defaults["input_dim"] = args.latent_dim

    model, diffusion = create_model_and_diffusion(**defaults)
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    # Preprocess data to h5ad and use scDiffusion's data loader
    # We write a tmp h5ad with 'celltype' column as required by cell_datasets_loader
    adata = sc.read_h5ad(args.train_h5ad)
    cell_data, cell_types, gene_names = _preprocess(
        adata, args.cell_type_col, args.hvg_path
    )
    n_genes = cell_data.shape[1]

    # scDiffusion's load_data expects obs['celltype'] and raw h5ad — we feed latents directly
    device = dist_util.dev()
    vae = _load_vae(args.vae_ckpt, n_genes, args.latent_dim, device)

    with torch.no_grad():
        latents = []
        bs = 512
        for i in range(0, len(cell_data), bs):
            batch = torch.tensor(cell_data[i:i+bs]).to(device)
            latents.append(vae(batch, return_latent=True).cpu().numpy())
    latents = np.concatenate(latents, axis=0)

    from guided_diffusion.cell_datasets_loader import CellDataset
    from torch.utils.data import DataLoader

    def _data_gen():
        ds = CellDataset(latents, cell_types)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, drop_last=True)
        while True:
            yield from loader

    data = _data_gen()

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=-1,
        lr=1e-4,
        ema_rate="0.9999",
        log_interval=500,
        save_interval=args.save_interval,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=schedule_sampler,
        weight_decay=1e-4,
        lr_anneal_steps=args.diff_steps,
        model_name="diffusion",
        save_dir=args.diff_out,
    ).run_loop()


# ---------------------------------------------------------------------------
# Mode: generate
# ---------------------------------------------------------------------------

def cmd_generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adata = sc.read_h5ad(args.train_h5ad)
    cell_data, _, gene_names = _preprocess(
        adata, args.cell_type_col, args.hvg_path
    )
    n_genes = cell_data.shape[1]

    vae   = _load_vae(args.vae_ckpt, n_genes, args.latent_dim, device)
    model, diffusion = _load_diffusion_model(args.diff_ckpt, args.latent_dim, device)

    n_cells    = int(args.n_cells)
    batch_size = min(n_cells, 1000)
    all_latents = []

    while sum(x.shape[0] for x in all_latents) < n_cells:
        with torch.no_grad():
            sample, _ = diffusion.p_sample_loop(
                model,
                (batch_size, args.latent_dim),
                clip_denoised=False,
                model_kwargs={},
                start_time=diffusion.betas.shape[0],
            )
        all_latents.append(sample.cpu())

    latents = torch.cat(all_latents, dim=0)[:n_cells]

    print(f"Decoding {latents.shape[0]} latent codes → gene expression ...")
    with torch.no_grad():
        gene_expr = vae(latents.to(device), return_decoded=True).cpu().numpy()

    # The VAE was trained on normalize_total+log1p data, so decoder output is in
    # log-normalized space.  Reverse the log1p so the stored values are in
    # normalize_total space (~sum 10 000 per cell).  This makes the output
    # compatible with the quality evaluator, which applies normalize_total+log1p
    # itself and assumes raw/pseudo-count input.
    gene_expr = np.expm1(gene_expr)

    synth = ad.AnnData(
        X=gene_expr,
        var=pd.DataFrame(index=gene_names),
    )

    # Assign cell types via 1-NN in latent space against training cells.
    # The diffusion backbone is unconditional, so we label each generated cell
    # with the cell type of its nearest training cell (cosine distance on the
    # L2-normalised encoder output).
    if args.cell_type_col in adata.obs.columns:
        print("Assigning cell types via 1-NN in latent space ...")
        from sklearn.neighbors import NearestNeighbors

        with torch.no_grad():
            train_latents = []
            for i in range(0, len(cell_data), 512):
                batch = torch.tensor(cell_data[i : i + 512]).to(device)
                train_latents.append(vae(batch, return_latent=True).cpu().numpy())
        train_latents_np = np.concatenate(train_latents, axis=0)

        nn = NearestNeighbors(n_neighbors=1, metric="cosine", n_jobs=-1)
        nn.fit(train_latents_np)
        _, idxs = nn.kneighbors(latents.cpu().numpy()[:n_cells])

        train_cell_types = adata.obs[args.cell_type_col].values
        synth.obs[args.cell_type_col] = train_cell_types[idxs.flatten()]
        print(f"  Cell type distribution: {synth.obs[args.cell_type_col].value_counts().to_dict()}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_h5ad)), exist_ok=True)
    synth.write_h5ad(args.out_h5ad, compression="gzip")
    print(f"Synthetic data saved: {args.out_h5ad}  ({synth.n_obs} cells × {synth.n_vars} genes)")


# ---------------------------------------------------------------------------
# Mode: score  (MIA)
# ---------------------------------------------------------------------------

def cmd_score(args):
    """
    Diffusion denoising loss MIA (Carlini et al. 2022).

    For each target cell:
      1. Encode through VAE → latent z
      2. At T random noise timesteps t:
           z_t = sqrt(alphabar_t)*z + sqrt(1-alphabar_t)*eps
           loss_t = ||eps - model(z_t, t)||^2
      3. Membership score = -mean(loss_t)   [higher = more likely member]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adata = sc.read_h5ad(args.target_h5ad)
    cell_data, _, _ = _preprocess(adata, args.cell_type_col, args.hvg_path)
    n_genes = cell_data.shape[1]
    n_cells = cell_data.shape[0]

    vae   = _load_vae(args.vae_ckpt, n_genes, args.latent_dim, device)
    model, diffusion = _load_diffusion_model(args.diff_ckpt, args.latent_dim, device)

    # Noise schedule alphas_cumprod
    alphas_bar = torch.tensor(diffusion.alphas_cumprod, dtype=torch.float32, device=device)
    T = len(alphas_bar)
    n_t = args.n_score_times

    # Encode all cells to latent space
    print(f"Encoding {n_cells} cells to latent space ...", flush=True)
    all_latents = []
    bs = 512
    with torch.no_grad():
        for i in range(0, n_cells, bs):
            batch = torch.tensor(cell_data[i:i+bs]).to(device)
            z = vae(batch, return_latent=True)
            all_latents.append(z)
    latents = torch.cat(all_latents, dim=0)   # (n_cells, latent_dim)

    # Compute denoising loss at random noise levels
    print(f"Computing denoising loss at {n_t} noise levels ...", flush=True)
    rng = np.random.default_rng(0)
    t_indices = rng.choice(T, size=n_t, replace=False)
    t_indices = np.sort(t_indices)

    all_losses = torch.zeros(n_cells, device=device)

    with torch.no_grad():
        for t_idx in t_indices:
            t_tensor = torch.full((n_cells,), t_idx, dtype=torch.long, device=device)
            noise = torch.randn_like(latents)

            ab  = alphas_bar[t_idx]
            z_t = torch.sqrt(ab) * latents + torch.sqrt(1 - ab) * noise

            pred_noise = model(z_t, t_tensor)
            loss_t = ((noise - pred_noise) ** 2).mean(dim=1)   # (n_cells,)
            all_losses += loss_t

    avg_loss = (all_losses / n_t).cpu().numpy()
    scores   = -avg_loss.astype(np.float32)   # negate: lower loss = higher membership score

    print(f"  scores mean={scores.mean():.3f}  std={scores.std():.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.scores_npy)), exist_ok=True)
    np.save(args.scores_npy, scores)
    print(f"Per-cell scores saved: {args.scores_npy}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Standalone scDiffusion runner (train_vae/train/generate/score)"
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # shared args helper
    def _shared(sp):
        sp.add_argument("--cell-type-col", default="cell_type")
        sp.add_argument("--hvg-path",      default=None)
        sp.add_argument("--latent-dim",    type=int, default=128)

    # train_vae
    sv = sub.add_parser("train_vae")
    sv.add_argument("train_h5ad")
    sv.add_argument("vae_out",    help="Directory to save VAE checkpoint")
    _shared(sv)
    sv.add_argument("--vae-steps",  type=int, default=150000)
    sv.add_argument("--batch-size", type=int, default=512)

    # train
    st = sub.add_parser("train")
    st.add_argument("train_h5ad")
    st.add_argument("vae_ckpt",  help="Path to trained VAE .pt file")
    st.add_argument("diff_out",  help="Directory to save diffusion checkpoints")
    _shared(st)
    st.add_argument("--diff-steps",    type=int, default=300000)
    st.add_argument("--batch-size",    type=int, default=512)
    st.add_argument("--save-interval", type=int, default=50000)

    # generate
    sg = sub.add_parser("generate")
    sg.add_argument("train_h5ad")
    sg.add_argument("vae_ckpt")
    sg.add_argument("diff_ckpt",  help="Path to trained diffusion .pt file")
    sg.add_argument("out_h5ad")
    sg.add_argument("n_cells",    type=int)
    _shared(sg)

    # score
    ss = sub.add_parser("score")
    ss.add_argument("target_h5ad")
    ss.add_argument("vae_ckpt")
    ss.add_argument("diff_ckpt")
    ss.add_argument("scores_npy")
    _shared(ss)
    ss.add_argument("--n-score-times", type=int, default=50,
                    help="Number of noise levels to average for MIA score")

    args = p.parse_args()

    dispatch = {
        "train_vae": cmd_train_vae,
        "train":     cmd_train,
        "generate":  cmd_generate,
        "score":     cmd_score,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()

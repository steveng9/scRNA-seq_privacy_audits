"""
Standalone scDiffusion runner — executed inside the scdiff_ conda environment.

Called via subprocess from src/sdg/scdiffusion/model.py.

Repo root: /home/golobs/scDiffusion  (Luo et al. 2024, Bioinformatics)

IMPLEMENTATION VERSION HISTORY
-------------------------------
v1 (prior to 2026-05-04):
    Two-stage pipeline (VAE + diffusion backbone only). Cell types assigned
    post-hoc via 1-NN in VAE latent space. Wrong hyperparameters. INVALID.

v2 (2026-05-04, LEGACY):
    Three-stage pipeline. Classifier-guided sampling for generation.
    Correct hyperparameters (vae=200k, diff=800k, batch=128).
    Wrong generation mode: paper's main results use unconditional, not guided.

v3 (2026-05-04+, PAPER-FAITHFUL — use this):
    Two-stage training (VAE + DDPM backbone, same hyperparams as v2).
    Generation: unconditional p_sample_loop (no classifier, no cond_fn,
    clip_denoised=False), matching cell_sample.py from Luo et al.
    Cell type labels assigned post-hoc via CellTypist trained on D_train,
    matching the paper's main evaluation pipeline exactly.
    Data stored under ~/data/scMAMAMIA/{dataset}/scdiffusion_v3/.

Pipeline
--------
  train_vae   <train_h5ad> <vae_out> [options]
      Train the VAE autoencoder from scratch. Supports resume.
      Output: <vae_out>/model_seed=0_step=<N>.pt

  train       <train_h5ad> <vae_ckpt> <diff_out> [options]
      Train the diffusion backbone in VAE latent space. Supports resume
      from the latest checkpoint found in <diff_out>/diffusion/.
      Output: <diff_out>/diffusion/model<N>.pt + ema_0.9999_<N>.pt

  train_celltypist  <train_h5ad> <celltypist_out> [options]  [v3]
      Train a CellTypist logistic-regression model on D_train for post-hoc
      annotation of unconditionally generated cells. Fast (CPU, minutes).
      Mirrors celltypist_train.py from Luo et al.
      Output: <celltypist_out>  (a .pkl file)

  generate_unconditional  <vae_ckpt> <diff_ckpt> <celltypist_ckpt>
                          <out_h5ad> <n_cells> [options]  [v3]
      Unconditional DDPM sampling + CellTypist annotation. Paper-faithful.
      clip_denoised=False (matches cell_sample.py).
      Uses EMA model weights for best generation quality.

  generate    <train_h5ad> <vae_ckpt> <diff_ckpt> <classifier_ckpt>
              <out_h5ad> <n_cells> [options]  [v2 LEGACY]
      Classifier-guided sampling. Kept for reference; not used for v3.

  score       <target_h5ad> <vae_ckpt> <diff_ckpt> <scores_npy>
      Compute per-cell diffusion denoising loss as the MIA membership score.

Key options (see --help for each subcommand)
--------------------------------------------
  --cell-type-col        obs column for cell type labels  (default: cell_type)
  --hvg-path             pre-computed HVG CSV (highly_variable bool)
  --latent-dim           VAE latent dimension             (default: 128)
  --vae-steps            VAE training iterations          (default: 200000)
  --diff-steps           diffusion training iterations    (default: 800000)
  --batch-size           mini-batch size for all training (default: 128)
  --generation-batch-size  sampling batch size            (default: 3000)
  --n-score-times        noise levels sampled for MIA     (default: 50)
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
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
    Normalize + log1p (scDiffusion standard), optionally subset to HVGs.
    Returns (cell_data float32, cell_types int64, gene_names list, LabelEncoder).
    """
    import scipy.sparse as sp
    from sklearn.preprocessing import LabelEncoder

    adata = adata.copy()
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=10)
    adata.var_names_make_unique()

    if hvg_path and os.path.exists(hvg_path):
        hvg_df    = pd.read_csv(hvg_path, index_col=0)
        hvg_genes = [g for g in hvg_df.index[hvg_df["highly_variable"]]
                     if g in adata.var_names]
        adata = adata[:, hvg_genes].copy()
        print(f"Using {len(hvg_genes)} pre-computed HVGs")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    cell_data  = X.astype(np.float32)

    le         = LabelEncoder()
    cell_types = le.fit_transform(adata.obs[cell_type_col].values).astype(np.int64)

    return cell_data, cell_types, adata.var_names.tolist(), le


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


def _load_classifier(classifier_ckpt, latent_dim, num_class, device):
    from guided_diffusion.script_util import (
        classifier_and_diffusion_defaults,
        create_classifier_and_diffusion,
    )
    defaults = classifier_and_diffusion_defaults()
    defaults["input_dim"] = latent_dim
    defaults["num_class"]  = num_class
    classifier, _ = create_classifier_and_diffusion(**defaults)
    ckpt = torch.load(classifier_ckpt, map_location=device)
    classifier.load_state_dict(ckpt)
    classifier.to(device)
    classifier.eval()
    return classifier


def _encode_latents(cell_data, vae, device, batch_size=512):
    """Encode all cells to VAE latent space. Returns (N, latent_dim) np.ndarray."""
    latents = []
    with torch.no_grad():
        for i in range(0, len(cell_data), batch_size):
            batch = torch.tensor(cell_data[i:i + batch_size]).to(device)
            latents.append(vae(batch, return_latent=True).cpu().numpy())
    return np.concatenate(latents, axis=0)


# ---------------------------------------------------------------------------
# Mode: train_vae
# ---------------------------------------------------------------------------

def cmd_train_vae(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adata = sc.read_h5ad(args.train_h5ad)
    cell_data, _, gene_names, _ = _preprocess(
        adata, args.cell_type_col, args.hvg_path
    )
    n_genes = cell_data.shape[1]
    print(f"Training VAE: {len(cell_data)} cells x {n_genes} genes  "
          f"latent_dim={args.latent_dim}  steps={args.vae_steps}  bs={args.batch_size}")

    from VAE.VAE_model import VAE
    from torch.utils.data import TensorDataset, DataLoader

    os.makedirs(args.vae_out, exist_ok=True)

    with open(os.path.join(args.vae_out, "gene_names.txt"), "w") as f:
        f.write("\n".join(gene_names))

    vae = VAE(
        num_genes=n_genes, device=str(device), seed=0,
        loss_ae="mse", hidden_dim=args.latent_dim, decoder_activation="ReLU",
    )
    vae.to(device)

    X_tensor = torch.tensor(cell_data)
    ds     = TensorDataset(X_tensor)
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
    """Train the diffusion backbone using dist_util (single- or multi-GPU)."""
    from guided_diffusion import dist_util, logger
    from guided_diffusion.resample import create_named_schedule_sampler
    from guided_diffusion.script_util import (
        model_and_diffusion_defaults,
        create_model_and_diffusion,
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

    adata = sc.read_h5ad(args.train_h5ad)
    cell_data, cell_types, _, _ = _preprocess(
        adata, args.cell_type_col, args.hvg_path
    )
    n_genes = cell_data.shape[1]
    print(f"Training diffusion backbone: {len(cell_data)} cells x {n_genes} genes  "
          f"latent_dim={args.latent_dim}  steps={args.diff_steps}  bs={args.batch_size}")

    device  = dist_util.dev()
    vae     = _load_vae(args.vae_ckpt, n_genes, args.latent_dim, device)
    latents = _encode_latents(cell_data, vae, device)
    print(f"  Encoded {len(latents)} cells to latent space")

    # Resume from latest checkpoint in save_dir if one exists
    diff_subdir = os.path.join(args.diff_out, "diffusion")
    resume_ckpt = ""
    existing = sorted(glob.glob(os.path.join(diff_subdir, "model*.pt")))
    if existing:
        resume_ckpt = existing[-1]
        print(f"  Resuming from checkpoint: {resume_ckpt}", flush=True)

    from guided_diffusion.cell_datasets_loader import CellDataset
    from torch.utils.data import DataLoader

    def _data_gen():
        ds     = CellDataset(latents, cell_types)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, drop_last=True)
        while True:
            yield from loader

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=_data_gen(),
        batch_size=args.batch_size,
        microbatch=-1,
        lr=1e-4,
        ema_rate="0.9999",
        log_interval=500,
        save_interval=args.save_interval,
        resume_checkpoint=resume_ckpt,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=schedule_sampler,
        weight_decay=1e-4,
        lr_anneal_steps=args.diff_steps,
        model_name="diffusion",
        save_dir=args.diff_out,
    ).run_loop()


# ---------------------------------------------------------------------------
# Mode: train_classifier
# ---------------------------------------------------------------------------

def cmd_train_classifier(args):
    """
    Train a Cell_classifier on noisy VAE latents for classifier-guided sampling.

    Matches classifier_train.py from Luo et al. 2024:
      - Architecture: Cell_classifier(input=128, hidden=[512,512,256,128], dropout=0.1)
      - Trains on q_sample(latent, t) for t ~ Uniform[0, start_guide_steps]
      - Loss: cross-entropy over cell-type labels
      - Optimizer: AdamW(lr=3e-4, weight_decay=0.0)
    """
    from guided_diffusion.script_util import (
        classifier_and_diffusion_defaults,
        create_classifier_and_diffusion,
    )
    from guided_diffusion.resample import create_named_schedule_sampler
    from torch.optim import AdamW
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adata = sc.read_h5ad(args.train_h5ad)
    cell_data, cell_types, _, le = _preprocess(
        adata, args.cell_type_col, args.hvg_path
    )
    n_genes   = cell_data.shape[1]
    num_class = int(cell_types.max()) + 1
    print(f"Training classifier: {len(cell_data)} cells  "
          f"{num_class} cell types  steps={args.classifier_steps}  bs={args.batch_size}")
    print(f"  Cell types: {le.classes_.tolist()}")

    vae     = _load_vae(args.vae_ckpt, n_genes, args.latent_dim, device)
    latents = _encode_latents(cell_data, vae, device)
    print(f"  Encoded {len(latents)} latents  shape={latents.shape}")

    defaults = classifier_and_diffusion_defaults()
    defaults["input_dim"] = args.latent_dim
    defaults["num_class"]  = num_class
    classifier, diffusion = create_classifier_and_diffusion(**defaults)
    classifier.to(device)

    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    latents_t = torch.tensor(latents, dtype=torch.float32)
    types_t   = torch.tensor(cell_types, dtype=torch.long)
    loader    = DataLoader(TensorDataset(latents_t, types_t),
                           batch_size=args.batch_size, shuffle=True,
                           num_workers=0, drop_last=True)

    opt = AdamW(classifier.parameters(), lr=3e-4, weight_decay=0.0)

    os.makedirs(args.classifier_out, exist_ok=True)
    # Persist class-name mapping so generate can label cells correctly
    np.save(os.path.join(args.classifier_out, "class_names.npy"),
            np.array(le.classes_))

    step = 0
    while step < args.classifier_steps:
        for (lat_batch, type_batch) in loader:
            if step >= args.classifier_steps:
                break

            lat_batch  = lat_batch.to(device)
            type_batch = type_batch.to(device)

            # t ~ Uniform[0, start_guide_steps]  (matches classifier_train.py)
            t, _ = schedule_sampler.sample(lat_batch.shape[0], device,
                                           start_guide_time=args.start_guide_steps)

            # Noisy latent: x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*eps
            with torch.no_grad():
                noisy_lat = diffusion.q_sample(lat_batch, t)

            classifier.train()
            logits = classifier(noisy_lat, t)
            loss   = F.cross_entropy(logits, type_batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 5000 == 0:
                acc = (logits.detach().argmax(dim=1) == type_batch).float().mean().item()
                print(f"  Classifier step {step}/{args.classifier_steps}  "
                      f"loss={loss.item():.4f}  acc@1={acc:.4f}", flush=True)

            if step > 0 and step % args.save_interval == 0:
                ckpt = os.path.join(args.classifier_out, f"model{step:06d}.pt")
                torch.save(classifier.state_dict(), ckpt)
                print(f"  Saved: {ckpt}")

            step += 1

    ckpt = os.path.join(args.classifier_out, f"model{step:06d}.pt")
    torch.save(classifier.state_dict(), ckpt)
    print(f"Classifier saved: {ckpt}")


# ---------------------------------------------------------------------------
# Mode: train_celltypist  [v3 — paper-faithful]
# ---------------------------------------------------------------------------

def cmd_train_celltypist(args):
    """
    Train a CellTypist logistic-regression model on D_train for post-hoc
    annotation of unconditionally generated cells.

    Mirrors celltypist_train.py from Luo et al. 2024 (fast, CPU-only).
    Input must be log1p-normalized (CellTypist requirement).
    """
    import celltypist

    adata = sc.read_h5ad(args.train_h5ad)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=10)
    adata.var_names_make_unique()

    if args.hvg_path and os.path.exists(args.hvg_path):
        hvg_df    = pd.read_csv(args.hvg_path, index_col=0)
        hvg_genes = [g for g in hvg_df.index[hvg_df["highly_variable"]]
                     if g in adata.var_names]
        adata = adata[:, hvg_genes].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    labels = sorted(adata.obs[args.cell_type_col].unique().tolist())
    print(f"Training CellTypist: {adata.n_obs} cells x {adata.n_vars} genes  "
          f"{len(labels)} cell types")
    print(f"  Labels: {labels}")

    new_model = celltypist.train(
        adata, labels=args.cell_type_col, n_jobs=8, feature_selection=False,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.celltypist_out)), exist_ok=True)
    new_model.write(args.celltypist_out)
    print(f"CellTypist model saved: {args.celltypist_out}")


# ---------------------------------------------------------------------------
# Mode: generate_unconditional  [v3 — paper-faithful]
# ---------------------------------------------------------------------------

def cmd_generate_unconditional(args):
    """
    Generate cells unconditionally (paper-faithful, matching cell_sample.py),
    then assign cell types via a pre-trained CellTypist model.

    Matches the paper's main evaluation pipeline:
      - Unconditional DDPM reverse process (no classifier, no cond_fn)
      - clip_denoised=False  (cell_sample.py default)
      - CellTypist for post-hoc cell type assignment (applied before expm1)
    Uses EMA model weights (ema_0.9999_*.pt) if diff_ckpt is an EMA file,
    otherwise uses the raw model checkpoint as given.
    """
    import celltypist

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae_dir         = os.path.dirname(args.vae_ckpt)
    gene_names_path = os.path.join(vae_dir, "gene_names.txt")
    with open(gene_names_path) as f:
        gene_names = [ln.strip() for ln in f if ln.strip()]
    n_genes = len(gene_names)
    n_total = int(args.n_cells)

    print(f"Generating {n_total} cells unconditionally  ({n_genes} genes)")

    vae              = _load_vae(args.vae_ckpt, n_genes, args.latent_dim, device)
    model, diffusion = _load_diffusion_model(args.diff_ckpt, args.latent_dim, device)

    gen_bs      = args.generation_batch_size
    all_latents = []
    generated   = 0

    while generated < n_total:
        this_bs = min(gen_bs, n_total - generated)
        with torch.no_grad():
            sample, _ = diffusion.p_sample_loop(
                model,
                (this_bs, args.latent_dim),
                clip_denoised=False,   # matches cell_sample.py
                model_kwargs={},
                cond_fn=None,
                device=device,
            )
        all_latents.append(sample.cpu())
        generated += this_bs
        print(f"  Generated {generated}/{n_total}", flush=True)

    latents = torch.cat(all_latents, dim=0)
    print(f"Decoding {latents.shape[0]} latents -> gene expression ...")

    with torch.no_grad():
        gene_expr_log1p = vae(latents.to(device), return_decoded=True).cpu().numpy()

    # CellTypist requires log1p-normalized to exactly 10000 counts/cell.
    # Re-normalize from VAE output to satisfy its format check.
    print("Annotating cell types with CellTypist ...")
    ct_model     = celltypist.models.Model.load(args.celltypist_ckpt)
    synth_for_ct = ad.AnnData(
        X=np.expm1(gene_expr_log1p).clip(0).astype(np.float32),
        var=pd.DataFrame(index=gene_names),
    )
    sc.pp.normalize_total(synth_for_ct, target_sum=1e4)
    sc.pp.log1p(synth_for_ct)
    predictions      = celltypist.annotate(synth_for_ct, model=ct_model,
                                           majority_voting=False)
    cell_type_labels = predictions.predicted_labels["predicted_labels"].values

    print(f"CellTypist distribution: "
          f"{pd.Series(cell_type_labels).value_counts().to_dict()}")

    # Reverse log1p: output in normalize_total space (downstream evals re-apply log1p)
    gene_expr = np.expm1(gene_expr_log1p)

    synth = ad.AnnData(
        X=gene_expr.astype(np.float32),
        var=pd.DataFrame(index=gene_names),
    )
    synth.obs[args.cell_type_col] = cell_type_labels

    os.makedirs(os.path.dirname(os.path.abspath(args.out_h5ad)), exist_ok=True)
    synth.write_h5ad(args.out_h5ad, compression="gzip")
    print(f"Synthetic data saved: {args.out_h5ad}  "
          f"({synth.n_obs} cells x {synth.n_vars} genes)")


# ---------------------------------------------------------------------------
# Mode: generate  (classifier-guided)  [v2 LEGACY]
# ---------------------------------------------------------------------------

def cmd_generate(args):
    """
    Generate synthetic cells using classifier-guided DDPM sampling.

    For each cell type present in the training data, generates cells in
    proportion to their training-set frequency using the classifier gradient
    as a cond_fn. Cell type labels are assigned from the generation target,
    not post-hoc 1-NN.

    Matches classifier_sample.py from Luo et al. 2024:
      - classifier_scale = 2.0  (gradient scale)
      - start_guide_steps = 500  (guidance active for t < 500)
      - clip_denoised = True
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adata = sc.read_h5ad(args.train_h5ad)
    cell_data, cell_types, gene_names, le = _preprocess(
        adata, args.cell_type_col, args.hvg_path
    )
    n_genes   = cell_data.shape[1]
    num_class = int(cell_types.max()) + 1
    n_total   = int(args.n_cells)

    class_counts = np.bincount(cell_types, minlength=num_class)
    class_props  = class_counts / class_counts.sum()
    class_names  = le.classes_.tolist()

    print(f"Generating {n_total} cells  ({num_class} cell types)")
    for i, (name, cnt, prop) in enumerate(zip(class_names, class_counts, class_props)):
        print(f"  [{i}] {name}: {cnt} train cells ({prop:.3f}) "
              f"-> {round(n_total * prop)} synthetic")

    vae              = _load_vae(args.vae_ckpt, n_genes, args.latent_dim, device)
    model, diffusion = _load_diffusion_model(args.diff_ckpt, args.latent_dim, device)
    classifier       = _load_classifier(args.classifier_ckpt, args.latent_dim,
                                        num_class, device)

    gen_bs = args.generation_batch_size

    # Cell_Unet backbone is unconditional — y kwarg is accepted but ignored.
    # Conditioning is entirely through cond_fn (classifier gradient guidance).
    def model_fn(x, t, y=None):
        return model(x, t)

    # cond_fn: grad(log p(y | x_t)) * classifier_scale
    # Matches cond_fn_ori in classifier_sample.py (Luo et al. 2024)
    def make_cond_fn(target_class_idx):
        def cond_fn(x, t, y=None):
            with torch.enable_grad():
                x_in     = x.detach().requires_grad_(True)
                logits   = classifier(x_in, t)
                log_prob = F.log_softmax(logits, dim=-1)
                selected = log_prob[:, target_class_idx]
                grad     = torch.autograd.grad(selected.sum(), x_in)[0]
                return grad * args.classifier_scale
        return cond_fn

    all_latents   = []
    all_celltypes = []

    for ct_idx in range(num_class):
        n_ct = round(n_total * class_props[ct_idx])
        if n_ct == 0:
            continue

        ct_name  = class_names[ct_idx]
        cond_fn  = make_cond_fn(ct_idx)
        ct_done  = 0

        while ct_done < n_ct:
            this_bs      = min(gen_bs, n_ct - ct_done)
            model_kwargs = {"y": torch.full((this_bs,), ct_idx,
                                            dtype=torch.long, device=device)}
            with torch.no_grad():
                sample, _ = diffusion.p_sample_loop(
                    model_fn,
                    (this_bs, args.latent_dim),
                    clip_denoised=True,           # matches classifier_sample.py default
                    model_kwargs=model_kwargs,
                    cond_fn=cond_fn,
                    device=device,
                    start_guide_steps=args.start_guide_steps,
                    start_time=diffusion.betas.shape[0],
                )
            all_latents.append(sample.cpu())
            all_celltypes.extend([ct_name] * this_bs)
            ct_done += this_bs

        print(f"  Generated {n_ct} cells for '{ct_name}'", flush=True)

    latents = torch.cat(all_latents, dim=0)
    print(f"Decoding {latents.shape[0]} latents -> gene expression ...")

    with torch.no_grad():
        gene_expr = vae(latents.to(device), return_decoded=True).cpu().numpy()

    # Reverse log1p so output is in normalize_total space (~sum 10,000/cell).
    # Quality evaluator re-applies normalize_total + log1p on its own.
    gene_expr = np.expm1(gene_expr)

    synth = ad.AnnData(
        X=gene_expr,
        var=pd.DataFrame(index=gene_names),
    )
    synth.obs[args.cell_type_col] = all_celltypes

    print(f"Cell type distribution: "
          f"{synth.obs[args.cell_type_col].value_counts().to_dict()}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_h5ad)), exist_ok=True)
    synth.write_h5ad(args.out_h5ad, compression="gzip")
    print(f"Synthetic data saved: {args.out_h5ad}  "
          f"({synth.n_obs} cells x {synth.n_vars} genes)")


# ---------------------------------------------------------------------------
# Mode: score  (MIA)
# ---------------------------------------------------------------------------

def cmd_score(args):
    """
    Diffusion denoising loss MIA (Carlini et al. 2022).

    For each target cell:
      1. Encode through VAE -> latent z
      2. At n_score_times random noise timesteps t:
           z_t = sqrt(alpha_bar_t)*z + sqrt(1-alpha_bar_t)*eps
           loss_t = ||eps - model(z_t, t)||^2
      3. Membership score = -mean(loss_t)   [higher = more likely member]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adata = sc.read_h5ad(args.target_h5ad)
    cell_data, _, _, _ = _preprocess(adata, args.cell_type_col, args.hvg_path)
    n_genes = cell_data.shape[1]
    n_cells = cell_data.shape[0]

    vae              = _load_vae(args.vae_ckpt, n_genes, args.latent_dim, device)
    model, diffusion = _load_diffusion_model(args.diff_ckpt, args.latent_dim, device)

    alphas_bar = torch.tensor(diffusion.alphas_cumprod, dtype=torch.float32, device=device)
    T   = len(alphas_bar)
    n_t = args.n_score_times

    print(f"Encoding {n_cells} cells to latent space ...", flush=True)
    with torch.no_grad():
        parts = []
        for i in range(0, n_cells, 512):
            batch = torch.tensor(cell_data[i:i + 512]).to(device)
            parts.append(vae(batch, return_latent=True))
    latents = torch.cat(parts, dim=0)

    print(f"Computing denoising loss at {n_t} noise levels ...", flush=True)
    rng        = np.random.default_rng(0)
    t_indices  = np.sort(rng.choice(T, size=n_t, replace=False))
    all_losses = torch.zeros(n_cells, device=device)

    with torch.no_grad():
        for t_idx in t_indices:
            t_tensor = torch.full((n_cells,), t_idx, dtype=torch.long, device=device)
            noise    = torch.randn_like(latents)
            ab       = alphas_bar[t_idx]
            z_t      = torch.sqrt(ab) * latents + torch.sqrt(1 - ab) * noise
            pred     = model(z_t, t_tensor)
            all_losses += ((noise - pred) ** 2).mean(dim=1)

    avg_loss = (all_losses / n_t).cpu().numpy()
    scores   = -avg_loss.astype(np.float32)

    print(f"  scores mean={scores.mean():.3f}  std={scores.std():.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.scores_npy)), exist_ok=True)
    np.save(args.scores_npy, scores)
    print(f"Per-cell scores saved: {args.scores_npy}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Standalone scDiffusion runner v2 -- "
                    "train_vae / train / train_classifier / generate / score"
    )
    sub = p.add_subparsers(dest="mode", required=True)

    def _shared(sp):
        sp.add_argument("--cell-type-col", default="cell_type")
        sp.add_argument("--hvg-path",      default=None)
        sp.add_argument("--latent-dim",    type=int, default=128)

    # ------------------------------------------------------------------
    # train_vae
    # ------------------------------------------------------------------
    sv = sub.add_parser("train_vae", help="Stage 1: train VAE autoencoder")
    sv.add_argument("train_h5ad")
    sv.add_argument("vae_out", help="Directory to save VAE checkpoint")
    _shared(sv)
    sv.add_argument("--vae-steps",     type=int, default=200000,
                    help="Training iterations (paper: 200000)")
    sv.add_argument("--batch-size",    type=int, default=128,
                    help="Mini-batch size (paper: 128)")
    sv.add_argument("--save-interval", type=int, default=50000)

    # ------------------------------------------------------------------
    # train  (diffusion backbone)
    # ------------------------------------------------------------------
    st = sub.add_parser("train", help="Stage 2: train DDPM backbone in latent space")
    st.add_argument("train_h5ad")
    st.add_argument("vae_ckpt",  help="Path to trained VAE .pt file")
    st.add_argument("diff_out",  help="Directory to save diffusion checkpoints")
    _shared(st)
    st.add_argument("--diff-steps",    type=int, default=800000,
                    help="Training iterations (paper: 800000)")
    st.add_argument("--batch-size",    type=int, default=128,
                    help="Mini-batch size (paper: 128)")
    st.add_argument("--save-interval", type=int, default=100000,
                    help="Checkpoint save frequency")

    # ------------------------------------------------------------------
    # train_classifier  [v2 LEGACY]
    # ------------------------------------------------------------------
    sc_p = sub.add_parser("train_classifier",
                           help="[v2 LEGACY] Stage 3: train Cell_classifier on noisy latents")
    sc_p.add_argument("train_h5ad")
    sc_p.add_argument("vae_ckpt",       help="Path to trained VAE .pt file")
    sc_p.add_argument("classifier_out", help="Directory to save classifier checkpoints")
    _shared(sc_p)
    sc_p.add_argument("--classifier-steps",  type=int, default=200000)
    sc_p.add_argument("--batch-size",        type=int, default=128)
    sc_p.add_argument("--start-guide-steps", type=int, default=500)
    sc_p.add_argument("--save-interval",     type=int, default=50000)

    # ------------------------------------------------------------------
    # train_celltypist  [v3 — paper-faithful]
    # ------------------------------------------------------------------
    sct = sub.add_parser("train_celltypist",
                          help="[v3] Train CellTypist on D_train for post-hoc annotation")
    sct.add_argument("train_h5ad")
    sct.add_argument("celltypist_out", help="Output path for CellTypist model (.pkl)")
    _shared(sct)

    # ------------------------------------------------------------------
    # generate_unconditional  [v3 — paper-faithful]
    # ------------------------------------------------------------------
    sgu = sub.add_parser("generate_unconditional",
                          help="[v3] Unconditional DDPM sampling + CellTypist annotation")
    sgu.add_argument("vae_ckpt")
    sgu.add_argument("diff_ckpt",        help="Path to trained diffusion .pt file (raw or EMA)")
    sgu.add_argument("celltypist_ckpt",  help="Path to trained CellTypist model (.pkl)")
    sgu.add_argument("out_h5ad")
    sgu.add_argument("n_cells", type=int)
    _shared(sgu)
    sgu.add_argument("--generation-batch-size", type=int, default=3000,
                     help="Sampling batch size (default: 3000)")

    # ------------------------------------------------------------------
    # generate  (classifier-guided)  [v2 LEGACY]
    # ------------------------------------------------------------------
    sg = sub.add_parser("generate",
                         help="[v2 LEGACY] Classifier-guided generation")
    sg.add_argument("train_h5ad")
    sg.add_argument("vae_ckpt")
    sg.add_argument("diff_ckpt",       help="Path to trained diffusion .pt file")
    sg.add_argument("classifier_ckpt", help="Path to trained classifier .pt file")
    sg.add_argument("out_h5ad")
    sg.add_argument("n_cells", type=int)
    _shared(sg)
    sg.add_argument("--classifier-scale",      type=float, default=2.0)
    sg.add_argument("--start-guide-steps",     type=int,   default=500)
    sg.add_argument("--generation-batch-size", type=int,   default=3000)

    # ------------------------------------------------------------------
    # score  (MIA)
    # ------------------------------------------------------------------
    ss = sub.add_parser("score", help="Compute per-cell MIA membership scores")
    ss.add_argument("target_h5ad")
    ss.add_argument("vae_ckpt")
    ss.add_argument("diff_ckpt")
    ss.add_argument("scores_npy")
    _shared(ss)
    ss.add_argument("--n-score-times", type=int, default=50,
                    help="Number of noise levels to average for MIA score")

    args = p.parse_args()
    dispatch = {
        "train_vae":              cmd_train_vae,
        "train":                  cmd_train,
        "train_celltypist":       cmd_train_celltypist,
        "generate_unconditional": cmd_generate_unconditional,
        "train_classifier":       cmd_train_classifier,
        "generate":               cmd_generate,
        "score":                  cmd_score,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()

"""
run_nmf_standalone.py — Standalone NMF-based synthetic single-cell data generator.

Adapted from SingleCellNMFGenerator (github.com/AndrewJWicks/SingleCellNMFGenerator).

Algorithm:
  1. MiniBatchNMF factorization of the count matrix → latent basis H, coefficients W
  2. KMeans clustering of W to discover cell-type clusters
  3. Per-cluster summary statistics (mean, variance, zero-inflation rate)
  4. ZINB (or Poisson) sampling of synthetic cells
  5. RandomForest classifier on (W, true_cell_type) → assign cell types to synthetic cells

Differential privacy (optional):
  --dp-mode all    applies Gaussian noise to NMF basis + Laplace noise to KMeans
                   centroids and cluster summaries
  --dp-mode none   no noise (default)
  Individual stages can be enabled: nmf, kmeans, sampling

Usage:
    python run_nmf_standalone.py \\
        --train-h5ad  /path/to/train.h5ad \\
        --output-h5ad /path/to/synthetic.h5ad \\
        [--n-cells N] [--n-components 20] [--batch-size 1000] \\
        [--dp-mode {all,nmf,kmeans,sampling,none}] \\
        [--dp-eps-nmf 1.0] [--dp-nmf-noise-scale 1.0] \\
        [--dp-eps-kmeans 1.0] [--dp-eps-summaries 1.0] \\
        [--cell-type-col cell_type] [--sampling-method {zinb,poisson}] \\
        [--seed 42]
"""

import argparse
import os
import sys

import numpy as np
import anndata as ad
import scanpy as sc
from sklearn.decomposition import MiniBatchNMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


def main():
    ap = argparse.ArgumentParser(description="NMF-based synthetic scRNA-seq generator")
    ap.add_argument("--train-h5ad",    required=True,  help="Path to training AnnData (.h5ad)")
    ap.add_argument("--output-h5ad",   required=True,  help="Output path for synthetic AnnData")
    ap.add_argument("--n-cells",       type=int, default=-1,
                    help="Number of synthetic cells to generate (-1 = match training count)")
    ap.add_argument("--n-components",  type=int, default=20,  help="NMF components")
    ap.add_argument("--batch-size",    type=int, default=1000, help="MiniBatchNMF batch size")
    ap.add_argument("--dp-mode",
                    choices=["all", "nmf", "kmeans", "sampling", "none"],
                    default="none",
                    help="Which DP stages to apply (default: none)")
    # DP epsilon defaults match the published CAMDA 2024 submission config.yaml:
    #   eps_nmf=0.5, eps_kmeans=2.1, eps_summaries=0.2
    ap.add_argument("--dp-eps-nmf",         type=float, default=0.5)
    ap.add_argument("--dp-nmf-noise-scale",  type=float, default=1.0,
                    help="Scale factor for Gaussian noise on NMF basis")
    ap.add_argument("--dp-eps-kmeans",       type=float, default=2.1)
    ap.add_argument("--dp-eps-summaries",    type=float, default=0.2)
    ap.add_argument("--cell-type-col",       default="cell_type")
    # CAMDA submission used Poisson; ZINB is also supported
    ap.add_argument("--sampling-method",     choices=["zinb", "poisson"], default="poisson")
    # sample_fraction: fraction of training cells used for NMF/KMeans fitting.
    # CAMDA submission used 0.1 (10%). We default to 1.0 for best quality.
    ap.add_argument("--sample-fraction",     type=float, default=1.0,
                    help="Fraction of training cells used for NMF fitting (default: 1.0)")
    ap.add_argument("--seed",                type=int, default=42)
    args = ap.parse_args()

    do_nmf       = args.dp_mode in ("all", "nmf")
    do_kmeans    = args.dp_mode in ("all", "kmeans")
    do_summaries = args.dp_mode in ("all", "sampling")

    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Load training data
    # ------------------------------------------------------------------
    print(f"[NMF] Loading training data: {args.train_h5ad}", flush=True)
    adata = sc.read_h5ad(args.train_h5ad)
    n_train = adata.n_obs
    n_synth = args.n_cells if args.n_cells > 0 else n_train
    print(f"[NMF] Training cells: {n_train:,}  →  Generating: {n_synth:,}", flush=True)

    labels_all = adata.obs[args.cell_type_col].astype(str).values
    unique_types = np.unique(labels_all)
    n_clusters = len(unique_types)
    print(f"[NMF] Cell types (n_clusters={n_clusters}): {unique_types}", flush=True)

    counts = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
    n_genes = counts.shape[1]
    var_df  = adata.var.copy()

    # Subsample for NMF/KMeans fitting (CAMDA used 10%; we default to 100%)
    n_fit = max(1, int(n_train * args.sample_fraction))
    if n_fit < n_train:
        fit_idx = np.random.choice(n_train, n_fit, replace=False)
        counts_fit   = counts[fit_idx]
        labels_fit   = labels_all[fit_idx]
    else:
        counts_fit   = counts
        labels_fit   = labels_all
    print(f"[NMF] Using {n_fit:,}/{n_train:,} cells for NMF/KMeans fitting "
          f"(sample_fraction={args.sample_fraction})", flush=True)

    # ------------------------------------------------------------------
    # 2. MiniBatchNMF factorization
    # ------------------------------------------------------------------
    print(f"[NMF] Fitting MiniBatchNMF (n_components={args.n_components}) ...", flush=True)
    nmf = MiniBatchNMF(n_components=args.n_components, random_state=args.seed)
    bs = args.batch_size
    for i in range(0, n_fit, bs):
        nmf.partial_fit(counts_fit[i : i + bs])
    H_clean = nmf.components_.copy()

    # Optional DP noise on NMF basis
    if do_nmf:
        print(f"[NMF] Applying Gaussian DP noise to NMF basis (scale={args.dp_nmf_noise_scale})",
              flush=True)
        H_noised = H_clean + np.random.normal(0, args.dp_nmf_noise_scale, H_clean.shape)
        nmf.components_ = H_noised
    else:
        H_noised = H_clean

    # ------------------------------------------------------------------
    # 3. Transform fitting cells to latent space (W = latent coefficients)
    # ------------------------------------------------------------------
    W_parts, lbl_parts = [], []
    for i in range(0, n_fit, bs):
        batch = counts_fit[i : i + bs]
        W_parts.append(nmf.transform(batch))
        lbl_parts.append(labels_fit[i : i + bs])
    W = np.vstack(W_parts)
    y = np.concatenate(lbl_parts)

    # ------------------------------------------------------------------
    # 4. Train RandomForest classifier
    # ------------------------------------------------------------------
    print("[NMF] Training RandomForest classifier ...", flush=True)
    rf = RandomForestClassifier(n_estimators=100, random_state=args.seed, n_jobs=-1)
    rf.fit(W, y)

    # ------------------------------------------------------------------
    # 5. KMeans clustering (one cluster per cell type)
    # ------------------------------------------------------------------
    print(f"[NMF] KMeans clustering (n_clusters={n_clusters}) ...", flush=True)
    if do_kmeans:
        baseline_km = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init="auto")
        baseline_km.fit(W)
        centroids = baseline_km.cluster_centers_

        W_min, W_max = W.min(axis=0), W.max(axis=0)
        sensitivity  = (W_max - W_min) / n_clusters
        noise_scale  = sensitivity / args.dp_eps_kmeans
        noisy_centroids = centroids + np.random.laplace(0, noise_scale, centroids.shape)

        km = KMeans(n_clusters=n_clusters, init=noisy_centroids.astype(np.float32),
                    n_init=1, random_state=args.seed)
        labels = km.fit_predict(W.astype(np.float32))
        print(f"[NMF] Applied Laplace DP noise to KMeans centroids "
              f"(eps_kmeans={args.dp_eps_kmeans})", flush=True)
    else:
        km = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init="auto")
        labels = km.fit_predict(W)

    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    print(f"[NMF] Cluster sizes: min={cluster_sizes.min()}  max={cluster_sizes.max()}", flush=True)

    # ------------------------------------------------------------------
    # 6. Per-cluster summary statistics (with optional DP noise)
    # ------------------------------------------------------------------
    eps_small = 1e-6
    gene_means = np.zeros((n_clusters, n_genes))
    gene_vars  = np.zeros((n_clusters, n_genes))
    gene_zprob = np.zeros((n_clusters, n_genes))

    for cid in range(n_clusters):
        mask = labels == cid
        if mask.sum() > 0:
            sub = counts_fit[mask]
            m = sub.mean(axis=0)
            v = sub.var(axis=0)
            z = (sub == 0).mean(axis=0)
            if do_summaries:
                sens  = 1.0 / mask.sum()
                scale = sens / args.dp_eps_summaries
                m += np.random.laplace(0, scale, m.shape)
                v += np.random.laplace(0, scale, v.shape)
                z = np.clip(z + np.random.laplace(0, scale, z.shape), 0, 1)
            gene_means[cid] = np.clip(m, 0, None)
            gene_vars[cid]  = np.clip(v, eps_small, None)
            gene_zprob[cid] = z
        else:
            gene_means[cid] = 0
            gene_vars[cid]  = eps_small
            gene_zprob[cid] = 1.0

    probs = cluster_sizes / cluster_sizes.sum()

    # ------------------------------------------------------------------
    # 7. Sample synthetic cells
    # ------------------------------------------------------------------
    print(f"[NMF] Sampling {n_synth:,} synthetic cells ({args.sampling_method}) ...",
          flush=True)
    synth_list     = []
    synth_clusters = []

    for _ in range(n_synth):
        cid = np.random.choice(n_clusters, p=probs)
        mu  = gene_means[cid]

        if args.sampling_method == "poisson":
            counts_s = np.random.poisson(lam=np.maximum(mu, 0))
        else:  # zinb
            var   = gene_vars[cid]
            denom = np.maximum(var - mu, eps_small)
            theta = np.minimum(mu ** 2 / denom, 1e6)
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = np.divide(mu, theta, out=np.zeros_like(mu), where=theta > 0)
            lam = np.random.gamma(shape=np.maximum(theta, eps_small), scale=scale)
            lam = np.nan_to_num(np.minimum(lam, 1e4), nan=0.0, posinf=1e4)
            counts_s = np.random.poisson(lam)
            zi_mask = np.random.rand(n_genes) < gene_zprob[cid]
            counts_s[zi_mask] = 0

        synth_list.append(counts_s)
        synth_clusters.append(cid)

    synth_array = np.vstack(synth_list)

    # ------------------------------------------------------------------
    # 8. Assign cell types via RandomForest
    # ------------------------------------------------------------------
    print("[NMF] Assigning cell types via RandomForest ...", flush=True)
    synth_W = nmf.transform(synth_array)
    prob_matrix = rf.predict_proba(synth_W)
    classes = rf.classes_
    synth_labels = np.array([
        classes[np.random.choice(len(classes), p=prob_matrix[i])]
        for i in range(len(synth_array))
    ])

    # ------------------------------------------------------------------
    # 9. Build and save AnnData
    # ------------------------------------------------------------------
    synth_adata = ad.AnnData(X=synth_array.astype(np.float32))
    synth_adata.var = var_df
    synth_adata.obs[args.cell_type_col] = synth_labels
    synth_adata.obs["cluster"] = np.array(synth_clusters).astype(str)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_h5ad)), exist_ok=True)
    synth_adata.write_h5ad(args.output_h5ad)
    print(f"[NMF] Saved synthetic data → {args.output_h5ad}  "
          f"({n_synth:,} cells × {n_genes:,} genes)", flush=True)

    # Reconstruction error (diagnostic)
    frob = np.linalg.norm(counts[:n_synth] - synth_array, ord="fro")
    print(f"[NMF] Frobenius norm (real vs synth): {frob:.2f}", flush=True)


if __name__ == "__main__":
    main()

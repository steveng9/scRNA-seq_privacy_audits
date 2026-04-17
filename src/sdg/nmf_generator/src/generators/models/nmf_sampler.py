#!/usr/bin/env python3

import os
import yaml
import numpy as np
import scanpy as sc
import click
from sklearn.decomposition import MiniBatchNMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import anndata as ad
import random


@click.command()
@click.option(
    "--dp",
    type=click.Choice(["all", "nmf", "kmeans", "sampling", "none"]),
    default="all",
    help="Which DP steps to apply…"
)
@click.option(
    "--experiment_name",
    default="",
    help="Subfolder name under synthetic/nmf_sampler/"
)
def main(dp, experiment_name):
    # 1) Load configuration
    cfg           = yaml.safe_load(open('config.yaml'))
    nmf_cfg       = cfg['nmf_sampler_config']
    sampling_method = nmf_cfg.get("sampling_method", "zinb").lower()
    home          = cfg['dir_list']['home']
    dp_cfg        = cfg.get('dp_config', {})

    # Decide which DP steps to run
    do_nmf      = dp in ("all", "nmf")
    do_kmeans   = dp in ("all", "kmeans")
    do_summaries= dp in ("all", "sampling")

    # label column name
    label_col = cfg['dataset_config']['cell_label_col_name']

    # 2) Hyperparameters
    sample_frac = float(nmf_cfg.get('sample_fraction', 1.0))
    bs          = int(nmf_cfg['nmf_batch_size'])
    n_comp      = int(nmf_cfg['n_components'])
    n_clusters  = int(nmf_cfg['n_clusters'])
    seed        = int(nmf_cfg.get('seed', 42))
    requested   = int(nmf_cfg.get('n_synth_samples', -1))

    # Load full annotated training data early for sizing
    full_adata = sc.read_h5ad(os.path.join(home, cfg['dataset_config']['train_count_file']))
    full_n     = full_adata.n_obs

    # Determine real vs synthetic cell counts
    n_cells = max(1, int(full_n * sample_frac))
    if requested > 0:
        n_synth, override = requested, True
    else:
        n_synth, override = n_cells, False
    print(f"[INFO] Generating {n_synth} synthetic cells" + (" (override)" if override else " to match real count"))

    # Seeds
    random.seed(seed)
    np.random.seed(seed)

    # Paths
    # build dynamic output path
    ds = cfg["dataset_config"]["name"]
    outd = os.path.join(
        home,
        cfg["dir_list"]["data_splits"],
        ds,
        "synthetic",
        "nmf_sampler",
        experiment_name or nmf_cfg.get("experiment_name", "")
    )
    os.makedirs(outd, exist_ok=True)

    # include dp in the filename
    filename = f"{ds}_{dp}_synthetic.h5ad"
    output_h5 = os.path.join(outd, filename)

    # 3) Load counts & labels
    labels_all = full_adata.obs[label_col].astype(str).values
    unique_types  = np.unique(labels_all)
    n_clusters    = len(unique_types)
    print(f"[INFO] Auto-setting n_clusters = {n_clusters} (one per real cell type)")
    counts_all = full_adata.X.toarray() if hasattr(full_adata.X, 'toarray') else full_adata.X
    n_genes    = counts_all.shape[1]

    # 4) Batch‐wise NMF fitting
    nmf = MiniBatchNMF(n_components=n_comp, random_state=seed)
    for i in range(0, n_cells, bs):
        nmf.partial_fit(counts_all[i : min(i+bs, n_cells), :])
    H = nmf.components_

    # Keep an un‐noised copy for debugging
    H_clean = H.copy()

    # DP output perturbation of H
    eps_nmf         = dp_cfg.get("eps_nmf")
    nmf_noise_scale = dp_cfg.get("nmf_noise_scale", 1.0)
    if do_nmf and eps_nmf is not None:
        print("[INFO] Applying DP noise to NMF basis H.")
        H_noised = H_clean + np.random.normal(loc=0, scale=nmf_noise_scale, size=H_clean.shape)
        nmf.components_ = H_noised
    else:
        H_noised = H_clean  # downstream code uses this

    # 5) Transform to get W, V, and labels
    W_parts, V_parts, lbl_parts = [], [], []
    for i in range(0, n_cells, bs):
        batch = counts_all[i : min(i+bs, n_cells), :]
        Wb    = nmf.transform(batch)
        W_parts.append(Wb)
        V_parts.append(batch)
        lbl_parts.append(labels_all[i : min(i+bs, n_cells)])
    W = np.vstack(W_parts)
    V = np.vstack(V_parts)
    y = np.concatenate(lbl_parts)

    # ==== DEBUG: Check cluster flips from NMF noise ====
    if do_nmf and eps_nmf is not None:
        # clean labels on the un‐noised embedding
        nmf.components_ = H_clean
        clean_labels = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(W)
        # noised labels on the noised embedding
        nmf.components_ = H_noised
        noised_labels = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(W)
        print(">>> NMF-only cluster flips:", np.sum(clean_labels != noised_labels))
        # restore the noised H for the rest of the pipeline
        nmf.components_ = H_noised
    # ================================================


    # 6) Train Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    rf.fit(W, y)
    print(f"[INFO] Trained RF on {len(y)} real embeddings.")

    # 7) KMeans with centroid perturbation
    eps_kmeans = dp_cfg.get('eps_kmeans')
    if do_kmeans and eps_kmeans is not None:
        print(f"[INFO] Applying Laplace perturbation to KMeans centroids with ε={eps_kmeans}")
        # 7a) Fit a clean “baseline” KMeans
        baseline_km = KMeans(n_clusters=n_clusters, random_state=seed)
        baseline_km.fit(W)
        clean_labels = baseline_km.labels_
        centroids   = baseline_km.cluster_centers_

        # 7b) Compute sensitivity per latent dimension
        W_min, W_max = W.min(axis=0), W.max(axis=0)
        sensitivity  = (W_max - W_min) / n_clusters

        # 7c) Draw Laplace noise and perturb centroids
        noise_scale     = sensitivity / eps_kmeans
        noise           = np.random.laplace(loc=0.0, scale=noise_scale, size=centroids.shape)
        noisy_centroids = centroids + noise

        # ==== DEBUG: how many labels flip? ====
        noised_km = KMeans(
            n_clusters=n_clusters,
            init=noisy_centroids.astype(np.float32),
            n_init=1,
            random_state=seed
        )
        flip_labels = noised_km.fit_predict(W.astype(np.float32))
        print(">>> KMeans-only cluster flips:", np.sum(clean_labels != flip_labels))
        # ======================================

        # 7d) Use the perturbed centroids for your downstream pipeline
        baseline_km.cluster_centers_ = noisy_centroids.astype(np.float32)
        labels = baseline_km.predict(W.astype(np.float32))

    else:
        # “none” or other flags: just do a standard KMeans
        clean_km = KMeans(n_clusters=n_clusters, random_state=seed)
        labels   = clean_km.fit_predict(W)



    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    print(f"[INFO] KMeans cluster sizes: {cluster_sizes}")

    # 8) Cluster summaries with optional Laplace noise
    gene_means = np.zeros((n_clusters, n_genes))
    gene_vars  = np.zeros((n_clusters, n_genes))
    gene_zprob = np.zeros((n_clusters, n_genes))
    eps_small  = 1e-6
    eps_sum    = dp_cfg.get('eps_summaries')
    lap_scale  = dp_cfg.get('laplace_scale', 1.0)
    for cid in range(n_clusters):
        mask = (labels == cid)
        if mask.sum() > 0:
            subV = V[mask]
            m = subV.mean(axis=0)
            v = subV.var(axis=0)
            z = (subV == 0).mean(axis=0)
            if do_summaries and eps_sum is not None:
                # sensitivity = 1/|Ic|
                sens = 1.0 / mask.sum()
                scale = sens / eps_sum * lap_scale
                m += np.random.laplace(0, scale, size=m.shape)
                v += np.random.laplace(0, scale, size=v.shape)
                z = np.clip(z + np.random.laplace(0, scale, size=z.shape), 0, 1)
            gene_means[cid] = np.clip(m, 0, None)
            gene_vars[cid]  = np.clip(v, eps_small, None)
            gene_zprob[cid] = z
        else:
            gene_means[cid] = 0
            gene_vars[cid]  = eps_small
            gene_zprob[cid] = 1.0

    # sampling proportions
    prop_aware = bool(nmf_cfg.get('proportion_aware', True))
    if prop_aware:
        probs = cluster_sizes / cluster_sizes.sum()
    else:
        probs = np.ones(n_clusters) / n_clusters
    print("[INFO] Sampling clusters %s." % ("proportionally" if prop_aware else "uniformly"))

    # 9) Synthetic sampling
    synth_list, synth_clusters = [], []
    for _ in range(n_synth):
        cid = np.random.choice(n_clusters, p=probs)
        mu  = gene_means[cid]

        if sampling_method == "poisson":
            # simple Poisson sampler (uses the noised gene_means if do_summaries=True)
            counts = np.random.poisson(lam=mu)

        elif sampling_method == "zinb":
            # original ZINB sampler
            var   = gene_vars[cid]
            denom = np.maximum(var - mu, eps_small)
            theta = np.minimum(mu**2 / denom, 1e6)
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = np.divide(mu, theta, out=np.zeros_like(mu), where=theta>0)
            lam    = np.random.gamma(shape=theta, scale=scale)
            lam    = np.nan_to_num(lam, nan=0.0, posinf=1e4)
            lam    = np.minimum(lam, 1e4)
            counts = np.random.poisson(lam)
            # zero-inflate
            zi_mask = np.random.rand(n_genes) < gene_zprob[cid]
            counts[zi_mask] = 0

        else:
            raise ValueError(f"Unknown sampling_method: {sampling_method!r}. "
                             f"Expected 'poisson' or 'zinb'.")

        synth_list.append(counts)
        synth_clusters.append(cid)


    # 10) Batched embedding transform & label sampling
    synth_array = np.vstack(synth_list)
    synth_W     = nmf.transform(synth_array)
    prob_matrix = rf.predict_proba(synth_W)
    classes     = rf.classes_
    synth_labels_str = [
        classes[np.random.choice(len(classes), p=prob_matrix[i])]
        for i in range(len(synth_array))
    ]

    # 11) Build and save synthetic AnnData with integer labels
    synth_adata = ad.AnnData(X=synth_array)
    synth_adata.obs['cluster'] = synth_clusters
    # map string labels to ints for both fields
    class_to_int = {cls: i for i, cls in enumerate(classes)}
    synth_labels_int = [class_to_int[l] for l in synth_labels_str]
    synth_adata.obs[label_col]       = synth_labels_int
    cell_type_col = cfg['dataset_config']['cell_type_col_name']
    synth_adata.obs[cell_type_col]   = synth_adata.obs[label_col]
    synth_adata.var                  = full_adata.var.copy()
    synth_adata.write_h5ad(output_h5)
    print(f"[INFO] Synthetic data with DP='{dp}' saved to {output_h5}")

    # 12) Report Frobenius error
    frob = np.linalg.norm(V - synth_array, ord='fro')
    print(f"[INFO] Frobenius norm: {frob:.4f}")

if __name__ == '__main__':
    main()

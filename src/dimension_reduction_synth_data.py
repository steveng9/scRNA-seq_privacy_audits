
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import numpy as np
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap.umap_ as umap


data_dir = "/Users/stevengolob/Documents/school/PhD/Ghent_project_mia_scRNAseq/data/onek1k/"
model_dir = "/Users/stevengolob/PycharmProjects/camda_hpc/models/"


# synth_file = data_dir + "synthetic_data/scdesign2/onek1k_annotated_synthetic_1k.h5ad"
# train_file = data_dir + "onek1k_annotated_train_subset.h5ad"
# test_file = data_dir + "onek1k_annotated_test_subset.h5ad"

synth_file = data_dir + "synthetic_data/scdesign2/onek1k_annotated_synthetic_all.h5ad"
train_file = data_dir + "onek1k_annotated_train_release.h5ad"
test_file = data_dir + "onek1k_annotated_test_release.h5ad"

hvg_file = model_dir + "hvg.csv"

synth_ = ad.read_h5ad(synth_file)
train_ = ad.read_h5ad(train_file)
test_ = ad.read_h5ad(test_file)
hvg_mask = pd.read_csv(hvg_file)


def main(reduction_method, sample=None):
    hvgs = hvg_mask[hvg_mask["highly_variable"]].values[:, 0]
    cell_types = train_.obs["cell_type"].unique()

    for cell_type in cell_types:
        print(f"Processing Cell Type: {cell_type}")
        train_df = train_[train_.obs["cell_type"] == cell_type].to_df()
        synth_df = synth_[synth_.obs["cell_type"] == cell_type].to_df()
        # frac = 1 if sample is None else max(sample / len(train_df), .2)
        frac = 1 if sample is None else (sample / len(train_df))
        print(f"Number of cells: {len(train_df)}, {len(synth_df)}")

        reduction_method(train_df.sample(frac=frac), synth_df.sample(frac=frac), hvgs=hvgs, cell_type=cell_type)

    train_df = train_.to_df()
    # test_df = test_.to_df()
    synth_df = synth_.to_df()

    if sample is not None:
        train_df = train_df.sample(n=sample)
        # test_df = test_df.sample(n=sample)
        synth_df = synth_df.sample(n=sample)

    # reduction_method(train_df, test_df, hvgs=hvgs)
    reduction_method(train_df, synth_df, hvgs=hvgs)


def run_pca_plot(train: pd.DataFrame, synth: pd.DataFrame, hvgs, cell_type: str = None, normalize=True, n_components=2):
    cols = hvgs

    tr = train.loc[:, cols].copy()
    sy = synth.loc[:, cols].copy()

    if normalize:
        scaler = StandardScaler()
        scaler.fit(tr.values)  # fit on real data only
        tr_vals = scaler.transform(tr.values)
        sy_vals = scaler.transform(sy.values)
    else:
        tr_vals, sy_vals = tr.values, sy.values

    pca = PCA(n_components=n_components, random_state=0)
    combined_vals = np.vstack([tr_vals, sy_vals])
    X_pca = pca.fit_transform(combined_vals)
    n_train = tr_vals.shape[0]

    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:n_train, 0], X_pca[:n_train, 1], s=12, alpha=0.6, label='Train', color='blue')
    plt.scatter(X_pca[n_train:, 0], X_pca[n_train:, 1], s=12, alpha=0.6, label='Synthetic', color='orange')
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    title = f"PCA: Train vs Synth" + (f" ({cell_type})" if cell_type else "")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return pca, X_pca



def run_umap_plot(train: pd.DataFrame, synth: pd.DataFrame, hvgs, cell_type: str = None, normalize=True, n_neighbors=15, min_dist=0.3):
    """
    UMAP visualization comparing train vs synthetic data for one cell type.
    """
    cols = hvgs

    tr = train.loc[:, cols].copy()
    sy = synth.loc[:, cols].copy()

    # --- normalize ---
    if normalize:
        scaler = StandardScaler()
        scaler.fit(tr.values)
        tr_vals = scaler.transform(tr.values)
        sy_vals = scaler.transform(sy.values)
    else:
        tr_vals, sy_vals = tr.values, sy.values

    # --- combine and run UMAP ---
    combined_vals = np.vstack([tr_vals, sy_vals])
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=0)
    X_umap = reducer.fit_transform(combined_vals)
    n_train = tr_vals.shape[0]

    # --- plot ---
    plt.figure(figsize=(6, 5))
    plt.scatter(X_umap[:n_train, 0], X_umap[:n_train, 1], s=12, alpha=0.6, label='Train', color='blue')
    plt.scatter(X_umap[n_train:, 0], X_umap[n_train:, 1], s=12, alpha=0.6, label='Synthetic', color='orange')
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    title = f"UMAP: Train vs Synth" + (f" ({cell_type})" if cell_type else "")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return reducer, X_umap


def run_tsne_plot(train: pd.DataFrame, synth: pd.DataFrame, hvgs, cell_type: str = None, normalize=True, perplexity=30):
    """
    t-SNE visualization comparing train vs synthetic data for one cell type.
    """
    perplexity = min(perplexity, min(len(train), len(synth)))
    cols = hvgs

    tr = train.loc[:, cols].copy()
    sy = synth.loc[:, cols].copy()

    # --- normalize ---
    if normalize:
        scaler = StandardScaler()
        scaler.fit(tr.values)
        tr_vals = scaler.transform(tr.values)
        sy_vals = scaler.transform(sy.values)
    else:
        tr_vals, sy_vals = tr.values, sy.values

    # --- combine and run t-SNE ---
    combined_vals = np.vstack([tr_vals, sy_vals])
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(combined_vals)
    n_train = tr_vals.shape[0]

    # --- plot ---
    plt.figure(figsize=(6, 5))
    plt.scatter(X_tsne[:n_train, 0], X_tsne[:n_train, 1], s=12, alpha=0.6, label='Train', color='blue')
    plt.scatter(X_tsne[n_train:, 0], X_tsne[n_train:, 1], s=12, alpha=0.6, label='Synthetic', color='orange')
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    title = f"t-SNE: Train vs Synth" + (f" ({cell_type})" if cell_type else "")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return tsne, X_tsne



if __name__ == "__main__":
    # main(run_pca_plot)
    main(run_umap_plot, sample=10_000)
    # main(run_tsne_plot, sample=10_000)

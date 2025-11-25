from typing import Tuple, Dict
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



data_dir = "/Users/stevengolob/Documents/school/PhD/Ghent_project_mia_scRNAseq/data/onek1k/"
model_dir = "/Users/stevengolob/PycharmProjects/camda_hpc/models/"

n_donors = 500
single_MI = False


# synth_file = data_dir + "synthetic_data/scdesign2/onek1k_annotated_synthetic_1k.h5ad"
# train_file = data_dir + "onek1k_annotated_train_subset.h5ad"
# test_file = data_dir + "onek1k_annotated_test_subset.h5ad"
# sample_frac = 1

synth_file = data_dir + "synthetic_data/scdesign2/onek1k_annotated_synthetic_all.h5ad"
train_file = data_dir + "onek1k_annotated_train_release.h5ad"
test_file = data_dir + "onek1k_annotated_test_release.h5ad"
sample_frac = .05

hvg_file = model_dir + "hvg.csv"

synth_ = ad.read_h5ad(synth_file)
train_ = ad.read_h5ad(train_file)
test_ = ad.read_h5ad(test_file)
hvg_mask = pd.read_csv(hvg_file)




def main():
    demo_results, demo_agg = demo()
    # demo_agg.to_csv('mia_aggregated_demo.csv', index=False)


def run_naive_mia_per_celltype(train: pd.DataFrame,
                                test: pd.DataFrame,
                                synth: pd.DataFrame,
                                hvg_mask: pd.Series or list or np.ndarray,
                                k: int = 5,
                                normalize: bool = True) -> pd.DataFrame:
    if not (list(train.columns) == list(test.columns) == list(synth.columns)):
        raise ValueError("train/test/synth must have identical columns (gene names) in the same order.")

    # Resolve HVG columns
    cols = hvg_mask

    if len(cols) == 0:
        raise ValueError("No HVG columns selected. Check HVG mask or names.")

    tr = train.loc[:, cols].copy()
    te = test.loc[:, cols].copy()
    sy = synth.loc[:, cols].copy()

    if normalize:
        comb = pd.concat([tr, te], axis=0)
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(comb.values)
        tr_vals = scaler.transform(tr.values)
        te_vals = scaler.transform(te.values)
        sy_vals = scaler.transform(sy.values)
    else:
        tr_vals = tr.values.astype(float)
        te_vals = te.values.astype(float)
        sy_vals = sy.values.astype(float)

    combined_vals = np.vstack([tr_vals, te_vals])
    k_used = min(k, sy_vals.shape[0])
    sims = cosine_similarity(combined_vals, sy_vals)
    print(sims.shape)
    topk_mean = np.mean(np.sort(sims, axis=1)[:, -k_used:], axis=1)
    scores = (topk_mean + 1.0) / 2.0
    combined_df = pd.concat([train.copy(), test.copy()], axis=0).reset_index(drop=True)
    combined_df['member'] = [True]*tr.shape[0] + [False]*te.shape[0]
    combined_df['mia_score'] = scores
    return combined_df[['donor', 'member', 'mia_score']]

def aggregate_across_celltypes(results_by_celltype: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ct, df in results_by_celltype.items():
        for i, row in df.reset_index().iterrows():
            rows.append({'cell_type': ct, 'cell_index_within_type': row['index'],
                         'member': row['member'], 'mia_score': row['mia_score']})
    out = pd.DataFrame(rows)
    out['mia_score_type_norm'] = out.groupby('cell_type')['mia_score'].transform(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.0)
    out['mia_score_aggregated'] = 0.5 * out['mia_score'] + 0.5 * out['mia_score_type_norm']
    return out

def demo():
    rng = np.random.RandomState(0)
    n_genes = 200
    genes = train_.to_df().columns
    hvg = hvg_mask[hvg_mask["highly_variable"]].values[:, 0]
    results = {}

    cell_types = train_.obs["cell_type"].unique()

    for cell_type in cell_types:
        print(f"Processing Cell Type: {cell_type}")
        synth_df = synth_[synth_.obs["cell_type"] == cell_type].to_df().sample(frac=sample_frac)

        if single_MI: # by cell
            train_df = train_[train_.obs["cell_type"] == cell_type].to_df().sample(frac=sample_frac)
            test_df = test_[test_.obs["cell_type"] == cell_type].to_df().sample(frac=sample_frac)
        else: # by donor
            train_df = sample_cells_by_donor(train_, cell_type, n_donors=n_donors)
            test_df = sample_cells_by_donor(test_, cell_type, n_donors=n_donors)
            synth_df["donor"] = None

        df = run_naive_mia_per_celltype(train_df, test_df, synth_df, hvg_mask=hvg, k=5)

        if single_MI:
            results[cell_type] = df
        else:
            grouped_results = df.groupby("donor")
            results[cell_type] = grouped_results.mean()

        try:
            auc_val = roc_auc_score(df['member'].astype(int), df['mia_score'].values)
        except Exception:
            auc_val = np.nan
        print(f"Cell type {cell_type}: n_train={train_df.shape[0]}, n_test={test_df.shape[0]}, n_synth={synth_df.shape[0]}, AUC={auc_val:.3f}")
        # break
    # agg = aggregate_across_celltypes(results)

    # print("\nAggregated results (first 10 rows):")
    # print(agg.head(10), agg.shape)
    plt.figure(figsize=(6,4))
    for cell_type, df in results.items():
        try:
            fpr, tpr, _ = roc_curve(df['member'].astype(int), df['mia_score'].values)
            plt.plot(fpr, tpr, label=f"{cell_type} (AUC={roc_auc_score(df['member'].astype(int), df['mia_score'].values):.3f})")
        except Exception:
            continue
    plt.plot([0,1],[0,1],'k--', linewidth=0.6)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC per cell type (demo)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return results, agg

def sample_cells_by_donor(adata, cell_type, n_donors, cells_per_donor=10, random_state=42):
    np.random.seed(random_state)

    unique_donors = adata.obs["individual"].unique()
    chosen_donors = np.random.choice(unique_donors, size=min(n_donors, len(unique_donors)), replace=False)
    print(f"Selected {len(chosen_donors)} donors: {chosen_donors}")

    sampled_indices = []
    for donor in chosen_donors:
        # donor_mask = adata.obs["individual"] == donor
        donor_mask = (adata.obs["individual"] == donor) & (adata.obs["cell_type"] == cell_type)
        donor_indices = np.where(donor_mask)[0]
        # if fewer than requested cells exist, sample all
        n_cells = min(cells_per_donor, len(donor_indices))
        sampled_indices.extend(np.random.choice(donor_indices, size=n_cells, replace=False))

    # Create new AnnData subset
    adata_sampled = adata[sampled_indices].copy()
    print(f"Sampled {adata_sampled.n_obs} cells from {len(chosen_donors)} donors.")
    adata_sampled_df = adata_sampled.to_df()
    adata_sampled_df["donor"] = adata_sampled.obs["individual"]

    return adata_sampled_df


if __name__ == "__main__":
    main()
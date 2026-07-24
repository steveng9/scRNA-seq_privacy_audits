import os
import pandas as pd
import numpy as np

# ── Hardcoded settings ──────────────────────────────────────────────────────
HOME_DIR       = "/Users/charlene/Documents/00_TCSS499-Research/HPC_starter_repo"
DATASET_NAME   = "TCGA-BRCA"
SUBTYPE_COL    = "Subtype"          # "cancer_type" for TCGA-COMBINED
NOISE_LEVEL    = 0.5
RANDOM_SEED    = 42
SPLIT_NO       = 1
EXPERIMENT     = "noise_0.5"
# ────────────────────────────────────────────────────────────────────────────

np.random.seed(RANDOM_SEED)

# Paths
data_save_dir = os.path.join(HOME_DIR, "data_splits") #data_splits
real_save_dir = os.path.join(data_save_dir, DATASET_NAME, "real")
syn_save_dir  = os.path.join(data_save_dir, DATASET_NAME, "test_synthetic", "multivariate", EXPERIMENT) #synthetic
os.makedirs(syn_save_dir, exist_ok=True)

# ── Load real training data ──────────────────────────────────────────────────
X_train = pd.read_csv(os.path.join(real_save_dir, f"X_train_real_split_{SPLIT_NO}.csv"))
y_train = pd.read_csv(os.path.join(real_save_dir, f"y_train_real_split_{SPLIT_NO}.csv"))
X_train_features = pd.read_csv(os.path.join(real_save_dir, "column_names.csv")).values.flatten()

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# ── Compute per-subtype gene statistics ──────────────────────────────────────
original_data_df = X_train.copy()
labels_df = y_train.copy()

merged_data = original_data_df.join(labels_df)
subtype_column = merged_data[SUBTYPE_COL]
numeric_data = merged_data.select_dtypes(include=[np.number])

gene_means = numeric_data.groupby(subtype_column).mean()
gene_covs  = numeric_data.groupby(subtype_column).apply(lambda g: g.cov())

# ── Generate synthetic data ───────────────────────────────────────────────────
synthetic_data_list = []

for subtype in gene_means.index:
    n_samples   = sum(labels_df.iloc[:, 0] == subtype)
    cov_matrix  = gene_covs.loc[subtype].values
    mean_vector = gene_means.loc[subtype]

    # Ensure symmetry and positive semi-definite
    cov_matrix = (cov_matrix + cov_matrix.T) / 2
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    eigvals = np.maximum(eigvals, 0)
    cov_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T

    min_eig = np.min(np.real(np.linalg.eigvals(cov_matrix)))
    if min_eig < 0:
        cov_matrix -= 10 * min_eig * np.eye(*cov_matrix.shape)

    # Sample from multivariate normal
    synthetic_subtype = np.random.multivariate_normal(
        mean=mean_vector,
        cov=cov_matrix,
        size=n_samples
    )

    # Add multivariate Gaussian noise
    noise_cov = cov_matrix * NOISE_LEVEL
    noise = np.random.multivariate_normal(
        mean=np.zeros(len(mean_vector)),
        cov=noise_cov,
        size=n_samples
    )

    noisy_subtype = synthetic_subtype + noise
    df_subtype = pd.DataFrame(noisy_subtype, columns=X_train_features)
    df_subtype['Subtype'] = subtype
    synthetic_data_list.append(df_subtype)

# Combine and shuffle
synthetic_data = pd.concat(synthetic_data_list, ignore_index=True)
synthetic_data = synthetic_data.sample(frac=1).reset_index(drop=True)

# Separate features and labels
synthetic_features = synthetic_data[synthetic_data.columns.difference(['Subtype'])]
synthetic_labels   = synthetic_data['Subtype']

# ── Save ──────────────────────────────────────────────────────────────────────
synthetic_features.to_csv(os.path.join(syn_save_dir, f"synthetic_data_split_{SPLIT_NO}.csv"), index=False)
synthetic_labels.to_csv(os.path.join(syn_save_dir,   f"synthetic_labels_split_{SPLIT_NO}.csv"), index=False)

print(f"Synthetic data saved to {syn_save_dir}")
print(f"Synthetic features shape: {synthetic_features.shape}")
print(f"Synthetic labels shape:   {synthetic_labels.shape}")




# ── run attack ──────────────────────────────────────────────────────────────────────

from np.linalg import pinv
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Uniform remapping: ZINB / Poisson CDF transforms
# ---------------------------------------------------------------------------

def remapping_fn(x, pi, theta, mu):
    """CDF of Zero-Inflated Negative Binomial at value x.  Returns P(X <= x)."""
    x = np.asarray(x)
    if np.isinf(theta):  # Poisson case
        F = poisson.cdf(x, mu)
    else:
        n = theta
        p = theta / (theta + mu)
        F = nbinom.cdf(x, n, p)
    return pi + (1 - pi) * F


def activate(p_rel, confidence=1, center=True) -> np.ndarray:
    """Convert raw log-ratio scores to sigmoid-activated membership probabilities."""
    logs = np.log(p_rel)
    zscores = stats.zscore(logs)
    median = np.median(zscores) if center else 0
    probabilities = 1 / (1 + np.exp(-1 * confidence * (zscores - median)))
    return probabilities

def compute_sample_scores(raw_scores, target_labels):
    activated_scores = activate(np.array(raw_scores))
    return activated_scores

attack_config = {
    "lin_alg_inverse_fn": pinv,
    "eps": .0001
}

def attack_mahalanobis(targets, synth, genes, mean_vector, covariance_matrix):

    inv_cov_matrix = attack_config.lin_alg_inverse_fn(covariance_matrix)

    X = targets[genes].values  # (N, G)
    G = len(genes)

    # Map all cells to uniform space gene-by-gene; each call is vectorized over N cells.
    mapped_targets = np.empty_like(X, dtype=float)
    gene_means = np.empty(G, dtype=float)
    for j in range(G):
        mapped_targets[:, j] = remapping_fn(X[j], mean_vector[j])
        gene_means[j] = remapping_fn(mean_vector[j], mean_vector[j])

    delta = mapped_targets - gene_means   # (N, G)

    # Mahalanobis for all N cells in one batched quadratic form
    distance_synth = np.sqrt(np.einsum("ij,jk,ik->i", delta, inv_cov_matrix, delta))  # (N,)

    scores = 1.0 / (distance_synth + attack_config.eps)
    nan_mask = np.isnan(scores)
    error_count = int(nan_mask.sum())
    if error_count > 0:
        scores = scores.copy()
        scores[nan_mask] = 0.5
        print(f"  [WARN] {error_count} NaN scores -- setting to 0.5")

    return compute_sample_scores(scores, target_labels)


predictions = attack_mahalanobis(targets, synthetic_data, genes, synth_mean_vector, synth_covariance_matrix)

auc = roc_auc_score(target_labels, predictions)
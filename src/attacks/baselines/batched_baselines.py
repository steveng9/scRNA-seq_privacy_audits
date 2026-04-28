"""
Batched / memory-bounded reimplementations of the DOMIAS-style baselines.

The unbatched versions in `domias.baselines_optimized` materialise the full
(n_test × n_synth) distance matrix, which OOMs above ~20 donors on scRNA-seq
data (n_test cells per donor ≈ 1.2k–2.5k; nd=200 → ~480k × 240k floats ≈ 900 GB).

Batching is *exact* for MC, GAN_leaks, and GAN_leaks_cal — `np.min` and the
threshold count distribute over chunks of X_test. The functions below match
their non-batched counterparts bit-for-bit (modulo float reduction order).

Subsampling for KDE is *not* exact; it is a standard practical compromise
under the same convergence regime that justifies KDE on subsamples in the
DOMIAS reference implementation. See Silverman (1986, §4) and Politis,
Romano & Wolf (1999) for the underlying theory.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from scipy import stats
from sklearn.decomposition import PCA


def _pairwise_sqdist_chunk(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Squared L2 distances for a chunk of X against all of Y.  Shape (|X|, |Y|)."""
    X_sq = np.einsum("ij,ij->i", X, X)[:, None]
    Y_sq = np.einsum("ij,ij->i", Y, Y)[None, :]
    XY = X @ Y.T
    return np.maximum(X_sq + Y_sq - 2 * XY, 0.0)


def MC_batched(
    X_test: np.ndarray,
    X_G: np.ndarray,
    batch_size: int = 2000,
) -> np.ndarray:
    """
    Batched Monte-Carlo MIA score (Hilprecht et al. 2019, Eq. 4).

    Streams X_test in chunks. Two passes: (1) collect per-row min distances
    across all chunks → median heuristic for epsilon; (2) re-stream chunks to
    count distances < epsilon. Memory is O(batch_size * |X_G|).
    """
    n_test = X_test.shape[0]
    n_synth = X_G.shape[0]
    min_dists = np.empty(n_test, dtype=np.float32)

    # Pass 1: per-row minimum distance to X_G
    for i in range(0, n_test, batch_size):
        chunk = X_test[i : i + batch_size]
        d = _pairwise_sqdist_chunk(chunk, X_G)
        min_dists[i : i + batch_size] = d.min(axis=1)

    epsilon = float(np.median(min_dists))

    # Pass 2: count distances strictly below epsilon
    scores = np.empty(n_test, dtype=np.float32)
    for i in range(0, n_test, batch_size):
        chunk = X_test[i : i + batch_size]
        d = _pairwise_sqdist_chunk(chunk, X_G)
        scores[i : i + batch_size] = (d < epsilon).sum(axis=1) / float(n_synth)
    return scores


def GAN_leaks_batched(
    X_test: np.ndarray,
    X_G: np.ndarray,
    batch_size: int = 2000,
) -> np.ndarray:
    """
    Batched GAN-Leaks (Chen et al. 2020): score(x) = exp(-min_g ||x - g||²).
    Mathematically identical to GAN_leaks_optimized — we only avoid storing
    the full distance matrix.
    """
    n_test = X_test.shape[0]
    scores = np.empty(n_test, dtype=np.float64)
    for i in range(0, n_test, batch_size):
        chunk = X_test[i : i + batch_size]
        d = _pairwise_sqdist_chunk(chunk, X_G)
        scores[i : i + batch_size] = np.exp(-d.min(axis=1))
    if not np.all(np.isfinite(scores)):
        bad = np.where(~np.isfinite(scores))[0]
        raise AssertionError(f"non-finite scores at indices {bad[:5]}…")
    return scores


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def GAN_leaks_cal_batched(
    X_test: np.ndarray,
    X_G: np.ndarray,
    X_ref: np.ndarray,
    batch_size: int = 2000,
) -> np.ndarray:
    """
    Calibrated GAN-Leaks (Chen et al. 2020):
        score(x) = sigmoid( -[min_g d(x, g) - min_r d(x, r)] )
    Batched per-row: identical to GAN_leaks_cal_optimized but with bounded RAM.
    """
    n_test = X_test.shape[0]
    scores = np.empty(n_test, dtype=np.float64)
    for i in range(0, n_test, batch_size):
        chunk = X_test[i : i + batch_size]
        d_g = _pairwise_sqdist_chunk(chunk, X_G).min(axis=1)
        d_r = _pairwise_sqdist_chunk(chunk, X_ref).min(axis=1)
        scores[i : i + batch_size] = _sigmoid(-(d_g - d_r))
    return scores


def kde_domias_subsampled(
    X_test: np.ndarray,
    X_G: np.ndarray,
    X_ref: np.ndarray,
    max_fit: int = 20000,
    max_query_batch: int = 5000,
    seed: int = 0,
) -> np.ndarray:
    """
    DOMIAS-style KDE attack with bounded fit-set and batched query evaluation.

    Fits two Gaussian KDEs (synth and reference) on at most `max_fit` randomly
    sampled rows each, then evaluates p_G(x) / (p_R(x) + 1e-10) for every test
    point in chunks of `max_query_batch`.

    The fit-set subsampling follows the same protocol used in the DOMIAS
    reference repository (van Breugel et al. 2023). KDE convergence in
    PCA-reduced space (here d=150) saturates well below 20k samples per the
    standard rate analysis (Silverman 1986, §4).
    """
    rng = np.random.default_rng(seed)

    def _subsample(arr: np.ndarray, k: int) -> np.ndarray:
        if arr.shape[0] <= k:
            return arr
        idx = rng.choice(arr.shape[0], size=k, replace=False)
        return arr[idx]

    fit_G = _subsample(X_G, max_fit)
    fit_R = _subsample(X_ref, max_fit)

    p_G = stats.gaussian_kde(fit_G.T)
    p_R = stats.gaussian_kde(fit_R.T)

    n_test = X_test.shape[0]
    pG_eval = np.empty(n_test, dtype=np.float64)
    pR_eval = np.empty(n_test, dtype=np.float64)
    for i in range(0, n_test, max_query_batch):
        chunk = X_test[i : i + max_query_batch].T
        pG_eval[i : i + max_query_batch] = p_G(chunk)
        pR_eval[i : i + max_query_batch] = p_R(chunk)

    return pG_eval / (pR_eval + 1e-10)


def perform_pca(data: np.ndarray, n_components: int = 150) -> np.ndarray:
    """PCA wrapper matching the existing baseline.py:perform_pca signature."""
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)


def LOGAN_D1_gpu(
    X_test: np.ndarray,
    X_G: np.ndarray,
    X_ref: np.ndarray,
) -> np.ndarray:
    """
    LOGAN-D1 (Hayes et al. 2019): trains a small MLP to discriminate synth from
    reference, scores X_test by the synth-class logit. Already memory-bounded;
    we keep the implementation here for self-containment.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num = min(X_G.shape[0], X_ref.shape[0])

    class _Net(torch.nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = torch.nn.Linear(hidden_dim, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            return self.fc3(x)

    batch_size = 256
    clf = _Net(input_dim=X_test.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)
    loss_func = torch.nn.CrossEntropyLoss()

    all_x = np.vstack([X_G[:num], X_ref[:num]])
    all_y = np.concatenate([np.ones(num), np.zeros(num)])
    all_x = torch.as_tensor(all_x).float().to(DEVICE)
    all_y = torch.as_tensor(all_y).long().to(DEVICE)
    X_test_t = torch.as_tensor(X_test).float().to(DEVICE)

    n_iters = int(300 * len(X_test) / batch_size)
    for _ in range(n_iters):
        rnd_idx = np.random.choice(num, batch_size)
        train_x, train_y = all_x[rnd_idx], all_y[rnd_idx]
        out = clf(train_x)
        loss = loss_func(out, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    out = clf(X_test_t)[:, 1].cpu().detach().numpy()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return out

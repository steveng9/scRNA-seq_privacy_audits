import sys
import numpy as np
from scipy.stats import nbinom, poisson
from scipy import stats
import anndata as ad




def sample_donors_strategy_1(cfg, all_data, cell_types):
    all_train, all_holdout, all_aux = sample_cells_from_sampled_donors(cfg, all_data, cell_types[0])
    for cell_type in cell_types[1:]:
        train_data, holdout_data, aux_data = sample_cells_from_sampled_donors(cfg, all_data, cell_type)
        all_train = ad.concat([all_train, train_data])
        all_holdout = ad.concat([all_holdout, holdout_data])
        all_aux = ad.concat([all_aux, aux_data])

    return all_train, all_holdout, all_aux


def sample_cells_from_sampled_donors(cfg, all_data, cell_type):
    all_meta = all_data.obs

    # step 1: sample donors
    all_only_cell_meta = all_meta[all_meta["cell_type"] == cell_type]
    unique_donors = all_meta["individual"].unique()
    cell_counts = all_only_cell_meta["individual"].value_counts()
    unique_donors_w_enough_of_celltype = list(cell_counts[cell_counts >= cfg.mia_setting.cells_per_donor_min].index)
    n_donors_used = min(cfg.mia_setting.num_donors, len(unique_donors_w_enough_of_celltype) // 2)
    train_donors = np.random.choice(unique_donors_w_enough_of_celltype, size=n_donors_used, replace=False)
    all_holdout_donors = list(set(unique_donors_w_enough_of_celltype).difference(set(train_donors)))
    used_holdout_donors = np.random.choice(all_holdout_donors, size=n_donors_used, replace=False)
    aux_donors = np.random.choice(unique_donors_w_enough_of_celltype, size=n_donors_used, replace=False)
    print(f"Selected {n_donors_used} donors of {len(unique_donors_w_enough_of_celltype)} w/ cell type {cell_type} ({len(used_holdout_donors)} holdout, {len(unique_donors)} total)")

    # step 2: then sample cells from donors
    def get_cell_sample_from_donor(donors):
        sampled_indices = []
        for donor in donors:
            donor_mask = (all_meta["individual"] == donor) & (all_meta["cell_type"] == cell_type)
            donor_indices = np.where(donor_mask)[0]
            # if fewer than requested cells exist, sample all
            n_cells_from_donor = min(cfg.mia_setting.cells_per_donor_max, len(donor_indices))
            sampled_indices.extend(np.random.choice(donor_indices, size=n_cells_from_donor, replace=False))
        return all_data[sampled_indices].copy()

    train_data = get_cell_sample_from_donor(train_donors)
    holdout_data = get_cell_sample_from_donor(used_holdout_donors)
    aux_data = get_cell_sample_from_donor(aux_donors)

    return train_data, holdout_data, aux_data



def sample_donors_strategy_2(cfg, all_data, cell_types):
    MIN_AUX_DONORS = 10
    all_meta = all_data.obs

    # step 1: sample donors
    all_donors = all_meta["individual"].unique()
    n_donors_used = min(cfg.mia_setting.num_donors, len(all_donors) // 2)
    target_donors = np.random.choice(all_donors, size=n_donors_used*2, replace=False)
    train_donors = target_donors[:n_donors_used]
    holdout_donors = target_donors[n_donors_used:]

    # create aux dataset
    non_target_donors = list(set(all_donors).difference(set(target_donors)))
    num_aux_donors = max(MIN_AUX_DONORS, n_donors_used)
    # The following approach samples from the non_target_donors first, until it needs more donors to
    # meet the minimum number of donors, at which point it will sample from the train + holdout donors
    non_target_donors_shuffled = np.random.permutation(non_target_donors)
    target_donors_shuffled = np.random.permutation(target_donors)
    aux_donors = np.concatenate((non_target_donors_shuffled, target_donors_shuffled))[:num_aux_donors]

    all_train = all_data[all_data.obs["individual"].isin(train_donors)]
    all_holdout = all_data[all_data.obs["individual"].isin(holdout_donors)]
    all_aux = all_data[all_data.obs["individual"].isin(aux_donors)]
    print(f"Num train donors: {len(train_donors)}, Holdout: {len(holdout_donors)}, Auxiliary: {len(aux_donors)}")
    print(f"Num train cells: {len(all_train)}, Holdout: {len(all_holdout)}, Auxiliary: {len(all_aux)}")

    if cfg.mia_setting.num_donors > 1.95*len(train_donors):
        print("Too few donors for MIA", flush=True)
        print(f"Requested: {cfg.mia_setting.num_donors}, found: {len(train_donors)}.", flush=True)
        print("exiting.", flush=True)
        sys.exit(0)

    return all_train, all_holdout, all_aux





# TODO: define a better metric here, where higher score means closer to y=x correlation

# simple absolute different, scaled by distance to origin
def closeness_to_correlation_1(vals1, vals2, correlation):
    similarity = 1 - (np.abs(vals1 - vals2) / np.maximum(vals1, vals2))
    return similarity

def closeness_to_correlation_2(vals1, vals2, correlation):
    ratios = vals2 / (vals1 + epsilon)
    ratios = np.where(ratios > 1, (1/ratios), ratios)
    return ratios

# same as previous, but switch to y=-x+1 line if correlation is negative
def closeness_to_correlation_3(vals1, vals2, correlation):
    if correlation >= 0:
        similarity = 1 - (np.abs(vals1 - vals2) / np.maximum(vals1, vals2))
    else:
        similarity = 1 - (np.abs(vals1 - (1-vals2)) / np.maximum(vals1, (1-vals2)))
    return similarity

# more sophisticated, evaluates point against the line y = cx + .5 - c/2
# which is just a line that passes through (.5, .5) and represents the correlation
def closeness_to_correlation_4(vals1, vals2, correlation):
    expected_vals2 = correlation * vals1 + .5 - correlation / 2
    similarity = 1-np.abs(expected_vals2 - vals2)
    return similarity




def activate(p_rel, confidence=1, center=True) -> np.ndarray:
    logs = np.log(p_rel)
    zscores = stats.zscore(logs)
    median = np.median(zscores) if center else 0
    probabilities = 1 / (1 + np.exp(-1 * confidence * (zscores - median)))
    return probabilities

def zinb_cdf(x, pi, theta, mu):
    """
    CDF of Zero-Inflated Negative Binomial at value x.
    Returns the probability P(X <= x).
    """
    x = np.asarray(x)
    if np.isinf(theta):  # Poisson case
        F = poisson.cdf(x, mu)
    # elif mu == 0 or mu == np.inf:
    #     F = .5
    else:
        n = theta
        p = theta / (theta + mu)
        F = nbinom.cdf(x, n, p)
    return pi + (1 - pi) * F

def zinb_cdf_DT(x, pi, theta, mu, jitter=True):
    x = np.asarray(x)
    if np.isinf(theta):
        F_x = poisson.cdf(x, mu)
        F_xm1 = poisson.cdf(x - 1, mu)
    else:
        n = theta
        p = theta / (theta + mu)
        F_x = nbinom.cdf(x, n, p)
        F_xm1 = nbinom.cdf(x - 1, n, p)
    F_x = pi + (1 - pi) * F_x
    F_xm1 = pi + (1 - pi) * F_xm1 * (x > 0)

    if jitter:
        v = np.random.rand(*x.shape)
    else:
        v = 0.5  # midpoint
    return F_xm1 + v * (F_x - F_xm1)

def zinb_uniform_transform(x, pi, theta, mu, jitter=True):
    """
    Properly maps ZINB-distributed counts to uniform(0,1) via
    the distributional transform, removing zero inflation bias.
    """
    x = np.asarray(x, dtype=int)

    if np.isinf(theta):
        F_nb = poisson.cdf(x, mu)
        F_nb_prev = poisson.cdf(x - 1, mu)
        p0_nb = poisson.pmf(0, mu)
    else:
        n = theta
        p = theta / (theta + mu)
        F_nb = nbinom.cdf(x, n, p)
        F_nb_prev = nbinom.cdf(x - 1, n, p)
        p0_nb = nbinom.pmf(0, n, p)

    # Overall probability of X=0
    p_zero_total = pi + (1 - pi) * p0_nb

    # Uniform jitter
    v = np.random.rand(*x.shape) if jitter else 0.5

    u = np.empty_like(x, dtype=float)

    # Case 1: zero observations
    mask_zero = (x == 0)
    # if np.any(mask_zero):
    #     # Spread zeros uniformly within [0, p_zero_total)
    #     u[mask_zero] = v[mask_zero] * p_zero_total

    # Case 2: nonzero observations
    mask_nonzero = ~mask_zero
    if np.any(mask_nonzero):
        F_x = F_nb[mask_nonzero]
        F_xm1 = F_nb_prev[mask_nonzero]
        # conditional CDF given X>0
        F_cond_x = (F_x - p0_nb) / (1 - p0_nb)
        F_cond_xm1 = (F_xm1 - p0_nb) / (1 - p0_nb)
        # mix into upper (1 - p_zero_total) portion
        u[mask_nonzero] = p_zero_total + (1 - p_zero_total) * (
            F_cond_xm1 + v[mask_nonzero] * (F_cond_x - F_cond_xm1)
        )

    return u



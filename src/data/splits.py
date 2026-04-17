"""
Donor sampling strategies for constructing train / holdout / auxiliary splits
for MIA experiments.

Each strategy takes (cfg, all_data, cell_types) and returns three numpy arrays:
    (train_donors, holdout_donors, aux_donors)
"""

import sys
import numpy as np


def sample_donors_strategy_2(cfg, all_data, _cell_types):
    """
    Sample equal-sized train and holdout sets from all donors, then build an
    auxiliary set that prioritises non-target donors but falls back to target
    donors when needed to reach `min_aux_donors`.
    """
    all_meta = all_data.obs

    all_donors = all_meta["individual"].unique()
    n_donors_used = min(cfg.mia_setting.num_donors, len(all_donors) // 2)

    target_donors = np.random.choice(all_donors, size=n_donors_used * 2, replace=False)
    train_donors = target_donors[:n_donors_used]
    holdout_donors = target_donors[n_donors_used:]

    non_target_donors = list(set(all_donors).difference(set(target_donors)))
    num_aux_donors = max(cfg.min_aux_donors, n_donors_used)
    non_target_shuffled = np.random.permutation(non_target_donors)
    target_shuffled = np.random.permutation(target_donors)
    aux_donors = np.concatenate((non_target_shuffled, target_shuffled))[:num_aux_donors]

    print(f"Num train donors: {len(train_donors)}, "
          f"Holdout: {len(holdout_donors)}, "
          f"Auxiliary: {len(aux_donors)}")

    if cfg.mia_setting.num_donors > 1.95 * len(train_donors):
        print("Too few donors for MIA", flush=True)
        print(f"Requested: {cfg.mia_setting.num_donors}, found: {len(train_donors)}.", flush=True)
        print("exiting.", flush=True)
        sys.exit(0)

    return train_donors, holdout_donors, aux_donors


def sample_donors_strategy_490(cfg, all_data, _cell_types):
    """
    Large-scale strategy for datasets that support ~490 train donors (e.g. OneK1K).

    Train / holdout are split equally (up to half the donor pool).
    Auxiliary is a random subsample of 200 HOLDOUT donors — guaranteeing that aux
    donors are disjoint from the training set, which keeps BB+aux semantically clean.
    Holdout donors are known non-members, so training a shadow copula on them is valid.

    When fewer than 400 total donors exist, falls back to strategy 2 behaviour.
    """
    all_meta = all_data.obs
    all_donors = all_meta["individual"].unique()
    n_donors_used = min(cfg.mia_setting.num_donors, len(all_donors) // 2)

    target_donors = np.random.choice(all_donors, size=n_donors_used * 2, replace=False)
    train_donors   = target_donors[:n_donors_used]
    holdout_donors = target_donors[n_donors_used:]

    # Aux = up to 200 holdout donors (disjoint from train by construction)
    n_aux = min(200, len(holdout_donors))
    aux_donors = np.random.choice(holdout_donors, size=n_aux, replace=False)

    print(f"Num train donors: {len(train_donors)}, "
          f"Holdout: {len(holdout_donors)}, "
          f"Auxiliary: {len(aux_donors)} (subsample of holdout)")

    if cfg.mia_setting.num_donors > 1.95 * len(train_donors):
        print("Too few donors for MIA", flush=True)
        print(f"Requested: {cfg.mia_setting.num_donors}, found: {len(train_donors)}.", flush=True)
        print("exiting.", flush=True)
        import sys
        sys.exit(0)

    return train_donors, holdout_donors, aux_donors


def sample_donors_strategy_3(cfg, all_data, _cell_types):
    """
    Identical to strategy 2.  Preserved as a separate entry for config-file
    compatibility and future divergence.
    """
    all_meta = all_data.obs

    all_donors = all_meta["individual"].unique()
    n_donors_used = min(cfg.mia_setting.num_donors, len(all_donors) // 2)

    target_donors = np.random.choice(all_donors, size=n_donors_used * 2, replace=False)
    train_donors = target_donors[:n_donors_used]
    holdout_donors = target_donors[n_donors_used:]

    non_target_donors = list(set(all_donors).difference(set(target_donors)))
    num_aux_donors = max(cfg.min_aux_donors, n_donors_used)
    non_target_shuffled = np.random.permutation(non_target_donors)
    target_shuffled = np.random.permutation(target_donors)
    aux_donors = np.concatenate((non_target_shuffled, target_shuffled))[:num_aux_donors]

    print(f"Num train donors: {len(train_donors)}, "
          f"Holdout: {len(holdout_donors)}, "
          f"Auxiliary: {len(aux_donors)}")

    if cfg.mia_setting.num_donors > 1.95 * len(train_donors):
        print("Too few donors for MIA", flush=True)
        print(f"Requested: {cfg.mia_setting.num_donors}, found: {len(train_donors)}.", flush=True)
        print("exiting.", flush=True)
        sys.exit(0)

    return train_donors, holdout_donors, aux_donors

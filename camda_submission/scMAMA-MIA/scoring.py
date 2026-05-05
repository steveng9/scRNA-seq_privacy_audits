"""
Cell-level and donor-level score aggregation for scMAMA-MIA.
"""

import numpy as np
import pandas as pd

from cdf_utils import activate


def compute_cell_scores_classb(targets, cell_type, raw_scores,
                                distances_synth=None, distances_aux=None,
                                use_aux=True):
    """
    Package per-cell membership scores into a DataFrame.

    Parameters
    ----------
    targets          : pd.DataFrame with 'individual', 'member', and gene columns
    cell_type        : str
    raw_scores       : (N,) array — already activated [0,1] membership scores
    distances_synth  : optional list of per-cell Mahalanobis distances to synth copula
    distances_aux    : optional list of per-cell Mahalanobis distances to aux copula
    use_aux          : bool — True for BB+aux (tm 100), False for BB-aux (tm 101)

    Returns
    -------
    pd.DataFrame with columns: cell id, donor id, cell type, membership, score
    """
    tm = "BB+aux" if use_aux else "BB-aux"
    return pd.DataFrame({
        "cell id":            targets.index,
        "donor id":           targets["individual"].values,
        "cell type":          cell_type,
        "membership":         targets["member"].values.astype(int),
        f"dist_synth:{tm}":   distances_synth,
        f"dist_aux:{tm}":     distances_aux,
        f"score:{tm}":        raw_scores,
    })


def merge_cell_type_results(per_cell_type_results, use_aux=True):
    """
    Concatenate per-cell-type result DataFrames, skipping any failures.

    Returns
    -------
    merged : pd.DataFrame
    total_runtime : float (seconds)
    """
    tm  = "BB+aux" if use_aux else "BB-aux"
    col = f"score:{tm}"
    merged  = pd.DataFrame()
    runtime = 0.0
    for cell_type, result_df, rt in per_cell_type_results:
        if result_df is not None and col in result_df.columns:
            if not result_df[col].isna().any():
                merged  = pd.concat([merged, result_df], ignore_index=True)
                runtime += rt
            else:
                print(f"  [SKIP] {cell_type}: NaN scores")
        else:
            print(f"  [SKIP] {cell_type}: no results")
    return merged, runtime


def aggregate_to_donor(merged_df, use_aux=True):
    """
    Average cell-level membership scores within each donor.

    Returns
    -------
    true_labels  : list[int] — ground-truth membership per donor
    predictions  : np.ndarray — predicted membership probability per donor
    donor_ids    : list — donor IDs in matching order
    """
    tm  = "BB+aux" if use_aux else "BB-aux"
    col = f"score:{tm}"

    donors  = merged_df["donor id"].unique().tolist()
    grouped = merged_df.groupby("donor id", observed=True)

    true_labels  = []
    raw_preds    = []

    for donor in donors:
        grp = grouped.get_group(donor)
        mem = grp["membership"].mean()
        assert mem in (0.0, 1.0), f"Inconsistent membership for donor {donor}: {mem}"
        true_labels.append(int(mem))
        raw_preds.append(grp[col].astype(float).mean())

    predictions = activate(np.array(raw_preds))
    return true_labels, predictions, donors

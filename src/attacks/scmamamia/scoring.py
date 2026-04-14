"""
Cell-level and donor-level score aggregation for scMAMA-MIA.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from data.cdf_utils import activate


def compute_cell_scores(cfg, cell_type, raw_scores, targets,
                        distances_synth=None, distances_aux=None, tm=None):
    """
    Activate raw focal-point sums / Mahalanobis ratios into [0, 1] membership
    scores and package them into a per-cell DataFrame.

    Parameters
    ----------
    cfg          : Box  — experiment config (provides threat model code)
    cell_type    : str
    raw_scores   : array-like  — one score per target cell
    targets      : pd.DataFrame — must contain 'individual' and 'member' columns
    distances_synth, distances_aux : optional lists of per-cell distances
    tm           : str or None — explicit threat model code override (e.g. "100");
                   if None, derived from cfg via _threat_model_code

    Returns
    -------
    pd.DataFrame with columns: cell id, donor id, cell type, membership,
        distance_to_synth:<tm>, distance_to_aux:<tm>, score:<tm>
    """
    if tm is None:
        tm = _threat_model_code(cfg)
    return pd.DataFrame({
        "cell id":               targets.index,
        "donor id":              targets["individual"].values,
        "cell type":             cell_type,
        "membership":            targets["member"].values,
        "distance_to_synth:" + tm: distances_synth,
        "distance_to_aux:" + tm:   distances_aux,
        "score:" + tm:           activate(np.array(raw_scores)),
    })


def merge_cell_type_results(cfg, per_cell_type_results):
    """
    Concatenate per-cell-type result DataFrames, skip any that are empty or
    contain NaN scores, and return the merged DataFrame plus total runtime.
    """
    tm = _threat_model_code(cfg)
    merged = pd.DataFrame(columns=[
        "cell id", "donor id", "cell type",
        "distance_to_synth:" + tm, "distance_to_aux:" + tm,
        "membership", "score:" + tm,
    ])
    total_runtime = 0.0
    for cell_type, result_df, runtime in per_cell_type_results:
        if result_df is not None and not result_df["score:" + tm].isna().any():
            merged = pd.concat([merged, result_df])
            total_runtime += runtime
        else:
            print(f"  Skipping cell type: {cell_type}")
    return merged, total_runtime


def aggregate_scores_by_donor(cfg, full_result_df):
    """
    Average cell-level membership scores within each donor, then activate.

    Returns
    -------
    membership_true : list[int]   — ground-truth 0/1 per donor
    predictions     : np.ndarray  — predicted membership probability per donor
    """
    tm = _threat_model_code(cfg)
    donors = full_result_df["donor id"].unique().tolist()
    grouped = full_result_df.groupby("donor id", observed=True)

    membership_true = []
    raw_predictions = []

    for donor in donors:
        group = grouped.get_group(donor)
        donor_membership = group["membership"].mean()
        assert donor_membership in (0.0, 1.0), (
            f"Inconsistent membership labels for donor {donor}: {donor_membership}"
        )
        membership_true.append(int(donor_membership))
        raw_predictions.append(group["score:" + tm].astype(float).mean())

    predictions = activate(np.array(raw_predictions))
    return membership_true, predictions


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _threat_model_code(cfg) -> str:
    """Return the 3-bit threat model string, e.g. '110' for BB+aux."""
    code = ""
    code += "0" if cfg.mia_setting.white_box  else "1"
    code += "0" if cfg.mia_setting.use_wb_hvgs else "1"
    code += "0" if cfg.mia_setting.use_aux      else "1"
    return code

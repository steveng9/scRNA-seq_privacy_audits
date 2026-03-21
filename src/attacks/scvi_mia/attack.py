"""
ELBO-based membership inference attack against scVI.

Strategy
--------
scVI is a VAE: it learns p(x|z) (decoder) and q(z|x) (encoder).
The ELBO for a cell x is:

    ELBO(x) = E_{q(z|x)}[log p(x|z)] - KL(q(z|x) || p(z))

Members (training donors) tend to have **higher** ELBO than non-members
because the model has overfit to their gene expression patterns.
We use ELBO as the cell-level membership score and aggregate to the
donor level via sigmoid + averaging — the same pipeline as scMAMA-MIA.

Two variants
------------
  attack_scvi_elbo          — with auxiliary data calibration (BB+aux analogue)
  attack_scvi_elbo_no_aux   — without auxiliary data (BB-aux analogue)

For the aux-calibrated variant we score target cells against BOTH the
target model AND an auxiliary model trained on the auxiliary set, then
compute  λ = elbo_synth / (elbo_synth + |elbo_aux|)  which maps to [0, 1].
(Analogous to the Mahalanobis ratio  d_aux / (d_synth + d_aux)  in
scMAMA-MIA, but inverted because higher ELBO means better fit.)

Donor-level aggregation
-----------------------
  raw cell score → sigmoid → mean per donor → sigmoid
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Cell score → DataFrame packaging
# ---------------------------------------------------------------------------

def _package_cell_scores(raw_scores: np.ndarray, obs: pd.DataFrame,
                          tm_code: str) -> pd.DataFrame:
    """
    Apply sigmoid to raw_scores and package into a per-cell DataFrame
    matching the scMAMA-MIA format expected by aggregate_scores_by_donor().
    """
    scores = _sigmoid(raw_scores)
    return pd.DataFrame({
        "cell id":           obs.index,
        "donor id":          obs["individual"].values,
        "cell type":         obs.get("cell_type", pd.Series(["unknown"] * len(obs))).values,
        "membership":        obs["member"].values,
        f"score:{tm_code}":  scores,
    })


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Donor-level aggregation
# ---------------------------------------------------------------------------

def aggregate_to_donors(cell_df: pd.DataFrame, tm_code: str):
    """
    Average cell-level scores within each donor, then sigmoid again.

    Returns
    -------
    y_true : list[int]        — ground-truth membership (0/1) per donor
    y_pred : np.ndarray       — predicted membership probability per donor
    """
    donors = cell_df["donor id"].unique()
    grouped = cell_df.groupby("donor id", observed=True)

    y_true, raw_preds = [], []
    for donor in donors:
        grp = grouped.get_group(donor)
        membership = grp["membership"].mean()
        assert membership in (0.0, 1.0), (
            f"Mixed membership labels for donor {donor}: {membership}"
        )
        y_true.append(int(membership))
        raw_preds.append(float(grp[f"score:{tm_code}"].mean()))

    y_pred = _sigmoid(np.array(raw_preds))
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Attack without auxiliary data
# ---------------------------------------------------------------------------

def attack_scvi_elbo_no_aux(
    elbo_scores: np.ndarray,
    obs: pd.DataFrame,
    tm_code: str = "110",
) -> tuple[pd.DataFrame, list, np.ndarray]:
    """
    Membership score = sigmoid(elbo_synth / scale)

    Parameters
    ----------
    elbo_scores : (n_cells,) — per-cell ELBO from the target scVI model
    obs         : pd.DataFrame with columns 'individual', 'member', optionally 'cell_type'
    tm_code     : 3-char threat-model code (e.g. "110" = BB-aux)

    Returns
    -------
    cell_df, y_true, y_pred
    """
    # Normalise ELBOs to a reasonable range so sigmoid isn't saturated
    scale = np.std(elbo_scores) + 1e-8
    normed = elbo_scores / scale

    cell_df = _package_cell_scores(normed, obs, tm_code)
    y_true, y_pred = aggregate_to_donors(cell_df, tm_code)
    return cell_df, y_true, y_pred


# ---------------------------------------------------------------------------
# Attack with auxiliary data calibration
# ---------------------------------------------------------------------------

def attack_scvi_elbo(
    elbo_synth: np.ndarray,
    elbo_aux:   np.ndarray,
    obs:        pd.DataFrame,
    tm_code:    str = "100",
) -> tuple[pd.DataFrame, list, np.ndarray]:
    """
    Calibrated attack using an auxiliary scVI model.

    Membership score:
        λ = elbo_synth / (elbo_synth + |elbo_aux| + ε)

    Intuition: if the target cell fits the *synthetic* model much better than
    the *auxiliary* model, it is likely a training member.

    Parameters
    ----------
    elbo_synth : (n_cells,) — ELBO from the model trained on D_train
    elbo_aux   : (n_cells,) — ELBO from the model trained on D_aux
    obs        : pd.DataFrame with 'individual', 'member', optionally 'cell_type'
    tm_code    : threat-model code

    Returns
    -------
    cell_df, y_true, y_pred
    """
    eps = 1e-8
    # Shift both to positive range before ratio
    min_val = min(elbo_synth.min(), elbo_aux.min())
    s = elbo_synth - min_val + eps   # positive
    a = elbo_aux   - min_val + eps   # positive

    lambda_ = s / (s + a)           # in (0, 1); higher = better fit to synth = member

    # Logit-transform to get unbounded score for sigmoid
    raw = np.log(lambda_ + eps) - np.log(1 - lambda_ + eps)

    cell_df = _package_cell_scores(raw, obs, tm_code)
    y_true, y_pred = aggregate_to_donors(cell_df, tm_code)
    return cell_df, y_true, y_pred

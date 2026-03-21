"""
Diffusion denoising loss MIA against scDiffusion (Carlini et al. 2022 strategy).

Membership score per cell = -mean_t[ ||eps - eps_theta(z_t, t)||^2 ]
  (computed in run_scdiffusion_standalone.py score mode)

Lower denoising loss = model knows this cell well = more likely a member.
Negated so higher score = member, matching the convention used by scMAMA-MIA.

Aggregation: same sigmoid+mean pipeline as scVI MIA and scMAMA-MIA.
"""

import numpy as np
import pandas as pd


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _package(raw_scores, obs, tm_code):
    return pd.DataFrame({
        "cell id":          obs.index,
        "donor id":         obs["individual"].values,
        "cell type":        obs.get("cell_type", pd.Series(["unknown"]*len(obs))).values,
        "membership":       obs["member"].values,
        f"score:{tm_code}": _sigmoid(raw_scores),
    })


def _aggregate(cell_df, tm_code):
    donors = cell_df["donor id"].unique()
    y_true, raw = [], []
    for d in donors:
        g = cell_df[cell_df["donor id"] == d]
        y_true.append(int(g["membership"].mean()))
        raw.append(float(g[f"score:{tm_code}"].mean()))
    return y_true, _sigmoid(np.array(raw))


def attack_scdiffusion(scores, obs, tm_code="110"):
    """
    Parameters
    ----------
    scores   : (n_cells,) float — per-cell membership scores from score mode
    obs      : pd.DataFrame with columns 'individual', 'member', optionally 'cell_type'
    tm_code  : threat-model code string

    Returns
    -------
    cell_df, y_true, y_pred
    """
    scale  = np.std(scores) + 1e-8
    normed = scores / scale
    cell_df = _package(normed, obs, tm_code)
    y_true, y_pred = _aggregate(cell_df, tm_code)
    return cell_df, y_true, y_pred

"""Attack evaluation metrics.

One function, `evaluate`, used by every attack so numbers are comparable across
the grid.  The headline four match the CAMDA abstract's table: AUC-ROC, AUPR,
and TPR at 1% and 10% FPR.

TPR@FPR is read off the ROC curve as the largest TPR whose FPR does not exceed
the target.  That is the conservative reading: it never credits an attack with
operating at a lower false-positive rate than it actually achieves.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

FPR_TARGETS = (0.001, 0.01, 0.05, 0.10, 0.20)


def tpr_at_fpr(y_true, y_score, fpr_target: float = 0.10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = fpr <= fpr_target
    return float(tpr[ok][-1]) if ok.any() else 0.0


def precision_at_top_k(y_true, y_score, top_percent: float = 5.0) -> float:
    """Fraction of true members among the highest-scoring `top_percent`."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    k = max(1, int(np.ceil(len(y_score) * top_percent / 100)))
    top = np.argsort(y_score)[-k:]
    return float(y_true[top].sum() / k)


def evaluate(y_true, y_score) -> dict:
    """Full metric dictionary for one (labels, scores) pair."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if np.isnan(y_score).any():
        y_score = np.nan_to_num(y_score, nan=float(np.nanmedian(y_score)))

    out = {
        "n": int(len(y_true)),
        "n_members": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
    }
    if len(np.unique(y_true)) < 2:
        return {**out, "auc": float("nan"), "aupr": float("nan")}

    out["auc"] = float(roc_auc_score(y_true, y_score))
    out["aupr"] = float(average_precision_score(y_true, y_score))
    out["accuracy_at_median"] = float(
        accuracy_score(y_true, y_score > np.median(y_score))
    )
    for t in FPR_TARGETS:
        out[f"tpr_at_fpr_{t:g}"] = tpr_at_fpr(y_true, y_score, t)
    out["precision_at_5pct"] = precision_at_top_k(y_true, y_score, 5.0)
    return out


def aggregate(per_split: list) -> dict:
    """Mean and standard deviation of each metric across splits."""
    if not per_split:
        return {}
    keys = [k for k in per_split[0] if isinstance(per_split[0][k], (int, float))]
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in per_split if k in m], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    out["n_splits"] = len(per_split)
    return out

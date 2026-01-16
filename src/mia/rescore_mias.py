import yaml
import os
import sys
import importlib
import anndata as ad
import numpy as np
import pandas as pd
from box import Box
import warnings

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from sklearn.metrics import (accuracy_score, roc_curve, f1_score,
                             roc_auc_score, average_precision_score,
                             precision_recall_curve, auc)



from src.mia.baseline_mias import create_config_for_baselines_code, create_config
from src.mia.models.base import BaseMIAModel

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)



env = "server" if sys.argv[1] == "T" else "local"
config_path = sys.argv[2]


src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)



def main():
    cfg = create_config()

    results_list = []
    def save_result(mia_name, predictions, runtime_):
        grp_predictions, grp_labels = perform_donor_level_avg(donors, predictions, labels)
        (acc_median, acc_best, fpr, tpr, threshold, auc_, ap, pr_auc, f1_median, f1_best,
            tpr_at_fpr_001, tpr_at_fpr_002, tpr_at_fpr_005,
            tpr_at_fpr_01, tpr_at_fpr_02, tpr_at_fpr_05) = compute_metrics(grp_predictions, grp_labels)
        results_list.append({
            "mia": mia_name,
            "accuracy_median": acc_median,
            "accuracy_best": acc_best,
            "roc_auc": auc_,
            "average_precision": ap,
            "pr_auc": pr_auc,
            "f1_median": f1_median,
            "f1_best": f1_best,
            "tpr_at_fpr_001": tpr_at_fpr_001,
            "tpr_at_fpr_002": tpr_at_fpr_002,
            "tpr_at_fpr_005": tpr_at_fpr_005,
            "tpr_at_fpr_01": tpr_at_fpr_01,
            "tpr_at_fpr_02": tpr_at_fpr_02,
            "tpr_at_fpr_05": tpr_at_fpr_05,
            "runtime": runtime_
        })


    # MAMA-MIA
    mamamia_file = os.path.join(cfg.results_base_path, "mamamia_all_scores.csv")
    if not os.path.exists(mamamia_file):
        sys.exit("mamamia results file does not exist")

    mamamia_results = pd.read_csv(mamamia_file)
    mamamia_runtimes = pd.read_csv(os.path.join(cfg.results_base_path, "mamamia_results.csv"))
    labels = mamamia_results["membership"]
    donors = mamamia_results["donor id"]

    for mm_tm in ["000", "100", "001", "101"]:
        col_name = f"tm:{mm_tm}"
        if col_name in mamamia_runtimes:
            mm_predictions = mamamia_results[f"score:{mm_tm}"].values
            runtime = mamamia_runtimes[mamamia_runtimes["metric"] == "runtime"][col_name].values[0]
            save_result(f"mamamia:{mm_tm}", mm_predictions, runtime)

    # BASELINE MIAs
    baselines_file = os.path.join(cfg.results_path, "baselines_evaluation_results.csv")
    if os.path.exists(baselines_file):
        baseline_predictions = pd.read_csv(baselines_file)

        for bl in baseline_predictions["method"].values:
            # this file will exist if "baselines_evaluation_results.csv" does
            bl_predictions = pd.read_csv(os.path.join(cfg.results_path, f"{bl}_predictions.csv")).values
            runtime = baseline_predictions[baseline_predictions["method"] == bl]["runtime"].values[0]
            save_result(bl, bl_predictions, runtime)


    results_df = pd.DataFrame(results_list)
    results_df.to_csv(os.path.join(cfg.results_base_path, "all_mias_evaluation_results.csv"), index=False)

    print("Done combining results")


def perform_donor_level_avg(donors, predictions, labels):

    unique_donors = list(np.unique(donors))
    scores_df = pd.DataFrame({
        'donor': donors,
        'score': predictions,
        'y_test': labels,
    })
    grouped = scores_df.groupby('donor')
    grp_predictions = np.array([grouped.get_group(donor)['score'].mean() for donor in unique_donors])
    grp_labels = np.array([grouped.get_group(donor)['y_test'].mean() for donor in unique_donors])

    return grp_predictions, grp_labels

def compute_metrics(
        y_scores: np.ndarray,
        y_true: np.ndarray,
        sample_weight: Optional[np.ndarray] = None):
    y_pred_median = y_scores > np.median(y_scores)
    np.save("yscores.npy", y_scores)
    np.save("ytrue.npy", y_true)

    # compute F1 for multiple thresholds
    thresholds = np.sort(np.unique(y_scores))
    # if len(thresholds) < 2:
    #        raise ValueError("Not enough unique prediction scores..")

    f1_scores = [f1_score(y_true, y_scores > t, sample_weight=sample_weight)
                 for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    y_pred_best = y_scores > best_threshold

    ## compare the accuracy and f1 computed with best_threshold vs median
    acc_median = accuracy_score(y_true, y_pred_median, sample_weight=sample_weight)
    acc_best = accuracy_score(y_true, y_pred_best, sample_weight=sample_weight)

    auc_sc = roc_auc_score(y_true, y_scores, sample_weight=sample_weight)
    ap = average_precision_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    # compute PR-AUC
    pr_auc = auc(recall, precision)

    fpr, tpr, threshold = roc_curve(y_true, y_scores, pos_label=1)

    # Get TPR at specific FPR thresholds
    tpr_at_fpr_001 = tpr[(fpr >= 0.01).argmax()]
    tpr_at_fpr_002 = tpr[(fpr >= 0.02).argmax()]
    tpr_at_fpr_005 = tpr[(fpr >= 0.05).argmax()]
    tpr_at_fpr_01 = tpr[(fpr >= 0.1).argmax()]
    tpr_at_fpr_02 = tpr[(fpr >= 0.2).argmax()]
    tpr_at_fpr_05 = tpr[(fpr >= 0.5).argmax()]

    f1_median = f1_score(y_true, y_pred_median, sample_weight=sample_weight)
    f1_best = f1_score(y_true, y_pred_best, sample_weight=sample_weight)

    # return acc, fpr, tpr, threshold, auc, ap
    return (acc_median, acc_best, fpr, tpr, threshold,
            auc_sc, ap, pr_auc, f1_median, f1_best,
            tpr_at_fpr_001, tpr_at_fpr_002, tpr_at_fpr_005,
            tpr_at_fpr_01, tpr_at_fpr_02, tpr_at_fpr_05)


if __name__ == '__main__':
    main()



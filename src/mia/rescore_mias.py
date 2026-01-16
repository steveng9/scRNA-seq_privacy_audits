import yaml
import os
import sys
import importlib
import anndata as ad
import numpy as np
import pandas as pd
from box import Box
import warnings
from itertools import zip_longest

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


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)



env = "server" if sys.argv[1] == "T" else "local"
config_path = sys.argv[2]
trial_number = int(sys.argv[3])


src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)



def main():
    cfg = create_config()


    # MAMA-MIA
    mamamia_file = os.path.join(cfg.results_base_path, "mamamia_all_scores.csv")
    if not os.path.exists(mamamia_file):
        sys.exit("mamamia results file does not exist")

    mamamia_results = pd.read_csv(mamamia_file)
    mamamia_runtimes = pd.read_csv(os.path.join(cfg.results_base_path, "mamamia_results.csv"))
    labels = mamamia_results["membership"]
    donors = mamamia_results["donor id"]


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
    padded_arrays = [
        list(zip_longest(donors.values, list(predictions), labels.values, fillvalue=0))
    ]
    difference_in_len = len(donors) - len(predictions)
    if difference_in_len > 0:
        print("Discrepency of {difference_in_len} missing predictions.")
    scores_df = pd.DataFrame(padded_arrays[0], columns=['donor', 'score', 'y_test'])
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

def create_config():
    with open(config_path) as f:
        cfg = Box(yaml.load(f, Loader=yaml.FullLoader))
        cfg.split_name = f"{cfg.mia_setting.num_donors}d"
        cfg.experiment_setting_path = os.path.join(cfg.dir_list[env].data, cfg.dataset_name, cfg.split_name)
        cfg.experiment_tracking_file = os.path.join(cfg.experiment_setting_path, "tracking.csv")
        cfg.trial_num = trial_number

        cfg.experiment_data_path = os.path.join(cfg.experiment_setting_path, str(cfg.trial_num), "datasets")
        cfg.results_base_path = os.path.join(cfg.experiment_setting_path, str(cfg.trial_num), "results")
        cfg.results_path = os.path.join(cfg.experiment_setting_path, str(cfg.trial_num), "results", "baseline_mias")
        os.makedirs(cfg.results_path, exist_ok=True)
        cfg.train_donor_path = os.path.join(cfg.experiment_data_path, "train.npy")
        cfg.holdout_donor_path = os.path.join(cfg.experiment_data_path, "holdout.npy")
        cfg.aux_donor_path = os.path.join(cfg.experiment_data_path, "auxiliary.npy")

        cfg.train_path = os.path.join(cfg.experiment_data_path, "train.h5ad")
        cfg.holdout_path = os.path.join(cfg.experiment_data_path, "holdout.h5ad")
        cfg.aux_path = os.path.join(cfg.experiment_data_path, "auxiliary.h5ad")
        cfg.synth_path = os.path.join(cfg.experiment_data_path, "synthetic.h5ad")
        cfg.targets_path = os.path.join(cfg.experiment_data_path, "targets.h5ad")
        cfg.labels_path = os.path.join(cfg.experiment_data_path, "labels.csv")

    print("Experiment Configuration:")
    print("dataset: ", cfg.dataset_name)
    print("num_donors: ", cfg.mia_setting.num_donors)
    print(f"TRIAL number: ", cfg.trial_num)
    return cfg



if __name__ == '__main__':
    main()



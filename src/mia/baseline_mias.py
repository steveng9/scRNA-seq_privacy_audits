import yaml
import os
import sys
import importlib
import anndata as ad
import numpy as np
import pandas as pd
from box import Box
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)



env = "server" if sys.argv[1] == "T" else "local"
config_path = sys.argv[2]

if env == "server":
    from mia.models.sc_baseline import DOMIASSingleCellBaselineModels
else:
    from src.mia.models.sc_baseline import DOMIASSingleCellBaselineModels


src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)



def main():
    cfg = create_config(config_path)
    create_datasets_for_baseline_experiment(cfg)
    baselines_cfg = create_config_for_baselines_code(cfg)

    mia_model = DOMIASSingleCellBaselineModels(baselines_cfg, cfg.synth_path, cfg.targets_path, cfg.labels_path, "", cfg.aux_path)

    predictions, y_test, runtimes = mia_model.run_attack()
    mia_model.results_save_dir = cfg.results_path
    mia_model.save_predictions(predictions)

    if y_test is not None:
        grp_preds, grp_y = mia_model.perform_donor_level_avg(predictions, y_test)
        mia_model.evaluate_attack(grp_preds, runtimes, grp_y, "baselines_evaluation_results.csv")

    register_baseline_experiment(cfg)
    delete_h5ad_datasets_for_baselines(cfg)



def get_next_baseline_trial_num(cfg):
    if os.path.exists(cfg.experiment_tracking_file):
        experiment_tracking_df = pd.read_csv(cfg.experiment_tracking_file, header=0)
        if experiment_tracking_df["baselines"].all():
            print("Synthetic data not yet available for additional experiments in this setting.")
            sys.exit(0)
        else:
            incomplete_experiments = experiment_tracking_df[experiment_tracking_df["baselines"] == 0]
            trial_num = int(incomplete_experiments['trial'].iloc[0])
            return trial_num
    else:
        print("Experiment tracking file not yet created for this setting")
        sys.exit(0)


def create_config(config_path):
    with open(config_path) as f:
        cfg = Box(yaml.load(f, Loader=yaml.FullLoader))
        cfg.split_name = f"{cfg.mia_setting.num_donors}d"
        cfg.experiment_setting_path = os.path.join(cfg.dir_list[env].data, cfg.dataset_name, cfg.split_name)
        cfg.experiment_tracking_file = os.path.join(cfg.experiment_setting_path, "tracking.csv")
        cfg.trial_num = get_next_baseline_trial_num(cfg)

        cfg.experiment_data_path = os.path.join(cfg.experiment_setting_path, str(cfg.trial_num), "datasets")
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



def create_datasets_for_baseline_experiment(cfg):
    all_data = ad.read_h5ad(os.path.join(cfg.dir_list[env].data, cfg.dataset_name, "full_dataset_cleaned.h5ad"))
    train_donors = np.load(cfg.train_donor_path, allow_pickle=True)
    holdout_donors = np.load(cfg.holdout_donor_path, allow_pickle=True)
    aux_donors = np.load(cfg.aux_donor_path, allow_pickle=True)

    all_train = all_data[all_data.obs["individual"].isin(train_donors)]
    all_holdout = all_data[all_data.obs["individual"].isin(holdout_donors)]
    all_aux = all_data[all_data.obs["individual"].isin(aux_donors)]
    targets = ad.concat([all_train, all_holdout])
    targets.obs["barcode_col"] = targets.to_df().index


    labels = pd.DataFrame()
    labels["membership"] = np.concatenate((np.ones(all_train.shape[0], dtype=int), np.zeros(all_holdout.shape[0], dtype=int)))
    labels["barcode_col"] = targets.to_df().index
    labels["individual"] = targets.obs["individual"].values

    # save so that these are available during this experiment (will be removed after)
    all_train.write_h5ad(cfg.train_path)
    all_holdout.write_h5ad(cfg.holdout_path)
    all_aux.write_h5ad(cfg.aux_path)
    targets.write_h5ad(cfg.targets_path)
    labels.to_csv(cfg.labels_path)



def create_config_for_baselines_code(cfg):
    baselines_cfg = dict()
    baselines_cfg["dir_list"] = dict()
    baselines_cfg["dir_list"]["home"] = cfg.experiment_data_path
    baselines_cfg["dir_list"]["mia_files"] = cfg.experiment_data_path
    baselines_cfg["generator_config"] = dict()
    baselines_cfg["generator_config"]["model_name"] = ""
    baselines_cfg["generator_config"]["experiment_name"] = ""
    baselines_cfg["attack_model"] = ""
    baselines_cfg["dataset_config"] = dict()
    baselines_cfg["dataset_config"]["name"] = ""
    baselines_cfg["dataset_config"]["membership_label_col"] = "membership"
    return baselines_cfg


def delete_h5ad_datasets_for_baselines(cfg):
    os.remove(cfg.train_path)
    os.remove(cfg.holdout_path)
    os.remove(cfg.aux_path)
    os.remove(cfg.targets_path)
    os.remove(cfg.labels_path)


def register_baseline_experiment(cfg):
    experiment_tracking_df = pd.read_csv(cfg.experiment_tracking_file, header=0)
    experiment_tracking_df.loc[experiment_tracking_df['trial'] == cfg.trial_num, "baselines"] = 1
    experiment_tracking_df.to_csv(cfg.experiment_tracking_file, index=False)




if __name__ == '__main__':
    main()



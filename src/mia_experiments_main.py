import datetime
import sys
import time
import warnings

import numpy as np

env = "server" if sys.argv[1] == "T" else "local"
config_path = sys.argv[2]
print_out = (len(sys.argv) > 3 and sys.argv[3] == "P")

from box import Box
from rpy2.robjects import r
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
import os
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil

from  numpy.linalg import inv as inv
from  numpy.linalg import solve as solve
from  numpy.linalg import pinv as pinv

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.simplefilter(action='ignore', category=FutureWarning)

from discretionary_functions import *

if env == "local":
    from src.generators.blue_team import run_singlecell_generator, format_ct_name
else:
    from generators.blue_team import run_singlecell_generator, format_ct_name




def main():

    # 1. create experiment configuration
    set_up_cuda()
    cfg = create_config(config_path)

    # 2. create data splits and necessary file structure
    make_dir_structure(cfg)
    make_experiment_config_files(cfg)
    resampled, cell_types = create_datasets_from_splits(cfg)

    # 3. generate target synthetic data
    regenerated = generate_target_synthetic_data(cfg, cell_types, force=resampled)

    # 4. generate focal points
    if not cfg.mia_setting.white_box:
        run_scDesign2(cfg, cfg.synth_model_config_path, cell_types, "SYNTHETIC_DATA_SHADOW_MODEL", force=regenerated)
    else: print("\n\n(simulating scDesign2 on synthetic data not necessary in white_box setting... skipping)", flush=True)
    run_scDesign2(cfg, cfg.aux_model_config_path, cell_types, "AUXILIARY_DATA_SHADOW_MODEL", force=resampled)

    # 5. set up wandb (experiment tracking)

    # 6. attack and save results
    results = mamamia_on_scdesign2(cfg)
    save_results(cfg, results)
    register_experiment(cfg)

    # 7. analyze results
    # analyze_final_results(cfg)
    delete_h5ad_datasets(cfg) # not necessary to store datasets, since list of donors are saved





def register_experiment(cfg):
    experiment_tracking_df, _trial_num = get_experiment_tracking(cfg)
    experiment_column_name = "tm:" + get_threat_model_code(cfg)
    experiment_tracking_df.loc[experiment_tracking_df['trial'] == cfg.trial_num, experiment_column_name] = 1
    experiment_tracking_df.to_csv(cfg.experiment_tracking_file, index=False)


def pinv_gpu(a):

    if DEVICE.type == "cpu":
        return pinv(a)
    else:
        a = a.astype(np.float32)
        """GPU-accelerated pseudoinverse using PyTorch"""
        a_torch = torch.from_numpy(a).cuda()
        a_inv_torch = torch.linalg.pinv(a_torch)
        return a_inv_torch.cpu().numpy()


def set_up_cuda():
    print(f"Using device: {DEVICE}")



def create_config(config_path):
    with open(config_path) as f:
        cfg = Box(yaml.load(f, Loader=yaml.FullLoader))
        cfg.split_name = f"{cfg.mia_setting.num_donors}d"
        cfg.top_data_dir = os.path.join(cfg.dir_list[env].data, cfg.dataset_name)
        cfg.cfg_dir = os.path.join(cfg.top_data_dir, cfg.split_name)
        cfg.experiment_tracking_file = os.path.join(cfg.cfg_dir, "tracking.csv")

        experiment_tracking_df, trial_num = get_experiment_tracking(cfg)
        cfg.trial_num = trial_num
        cfg.trial_dir = os.path.join(cfg.cfg_dir, str(trial_num))

        cfg.datasets_path = os.path.join(cfg.trial_dir, "datasets")
        cfg.results_path = os.path.join(cfg.trial_dir, "results")
        cfg.figures_path = os.path.join(cfg.results_path, "figures")
        cfg.models_path = os.path.join(cfg.trial_dir, "models")
        cfg.artifacts_path = os.path.join(cfg.trial_dir, "artifacts")
        cfg.synth_artifacts_path = os.path.join(cfg.artifacts_path, "synth")
        cfg.aux_artifacts_path = os.path.join(cfg.artifacts_path, "aux")
        cfg.train_donors_path = os.path.join(cfg.datasets_path, "train.npy")
        cfg.train_path = os.path.join(cfg.datasets_path, "train.h5ad")
        cfg.target_synthetic_data_path = os.path.join(cfg.datasets_path, "synthetic.h5ad")
        cfg.holdout_donors_path = os.path.join(cfg.datasets_path, "holdout.npy")
        cfg.holdout_path = os.path.join(cfg.datasets_path, "holdout.h5ad")
        cfg.aux_donors_path = os.path.join(cfg.datasets_path, "auxiliary.npy")
        cfg.aux_path = os.path.join(cfg.datasets_path, "auxiliary.h5ad")
        cfg.all_scores_file = os.path.join(cfg.results_path, "mamamia_all_scores.csv")
        cfg.results_file = os.path.join(cfg.results_path, "mamamia_results.csv")
        cfg.runtime_file = os.path.join(cfg.results_path, "mamamia_runtimes.csv")
        cfg.target_model_config_path = os.path.join(cfg.models_path, "config.yaml")
        cfg.synth_model_config_path = os.path.join(cfg.synth_artifacts_path, f"config.yaml")
        cfg.aux_model_config_path = os.path.join(cfg.aux_artifacts_path, "config.yaml")
        cfg.permanent_hvg_mask_path = os.path.join(cfg.top_data_dir, "hvg.csv")
        cfg.shadow_modelling_hvg_path = cfg.permanent_hvg_mask_path if cfg.mia_setting.use_wb_hvgs else cfg.artifacts_path

        cfg.parallelize = cfg.parallelize and not cfg.mamamia_params.mahalanobis #mahalanobis is too cpu-intensive to parallelize
        # cfg.parallelize = False
        cfg.lin_alg_inverse_fn = globals()[cfg.mamamia_params.lin_alg_inverse_fn]
        cfg.uniform_remapping_fn = globals()[cfg.mamamia_params.uniform_remapping_fn]
        cfg.closeness_to_correlation_fn = globals()[cfg.mamamia_params.closeness_to_correlation_fn]
        cfg.sample_donors_strategy_fn = globals()[cfg.mia_setting.sample_donors_strategy_fn]

    if print_out:
        print("Experiment Configuration:")
        for k, v in cfg.mia_setting.to_dict().items():
            print(f"\t{k}: {v}")
        print("mamamia parameters:")
        for k, v in cfg.mamamia_params.to_dict().items():
            print(f"\t{k}: {v}")
    return cfg


def new_experiment_row(trial_num=[1]):
    return dict({
        'trial': trial_num,
        # 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # 'synth_generated': False,
        # 'synth_copula_trained': False,
        # 'aux_copula_trained': False,
        'tm:000': 0,
        'tm:001': 0,
        'tm:010': 0,
        'tm:011': 0,
        'tm:100': 0,
        'tm:101': 0,
        'tm:110': 0,
        'tm:111': 0,
        'baselines': 0,
        'quality': 0
    })


def get_experiment_tracking(cfg):
    if os.path.exists(cfg.experiment_tracking_file):
        experiment_tracking_df = pd.read_csv(cfg.experiment_tracking_file, header=0)
    else:
        experiment_tracking_df = pd.DataFrame(new_experiment_row())
        os.makedirs(cfg.cfg_dir, exist_ok=True)
        experiment_tracking_df.to_csv(cfg.experiment_tracking_file, index=False)
    trial_num = get_next_trial_num(cfg, experiment_tracking_df)

    return experiment_tracking_df, trial_num


def get_next_trial_num(cfg, experiment_tracking_df):
    threat_model = "tm:" + get_threat_model_code(cfg)
    if experiment_tracking_df[threat_model].all():
        trial_num = experiment_tracking_df["trial"].max() + 1
        experiment_tracking_df.loc[trial_num] = new_experiment_row(trial_num)
        experiment_tracking_df.to_csv(cfg.experiment_tracking_file, index=False)
    else:
        incomplete_experiments = experiment_tracking_df[experiment_tracking_df[threat_model] == 0]
        trial_num = int(incomplete_experiments['trial'].iloc[0])

    return trial_num


def get_threat_model_code(cfg):
    threat_model = ""
    if cfg.mia_setting.white_box: threat_model += "0"
    else: threat_model += "1"
    if cfg.mia_setting.use_wb_hvgs: threat_model += "0"
    else: threat_model += "1"
    if cfg.mia_setting.use_aux: threat_model += "0"
    else: threat_model += "1"
    return threat_model


def make_dir_structure(cfg):
    os.makedirs(cfg.datasets_path, exist_ok=True)
    os.makedirs(cfg.figures_path, exist_ok=True)
    os.makedirs(cfg.models_path, exist_ok=True)
    os.makedirs(cfg.synth_artifacts_path, exist_ok=True)
    os.makedirs(cfg.aux_artifacts_path, exist_ok=True)


def make_experiment_config_files(cfg):

    # target model
    if not os.path.exists(cfg.target_model_config_path):
        target_model_config = make_scdesign2_config(cfg, True, "models", cfg.permanent_hvg_mask_path, "train.h5ad")
        with open(cfg.target_model_config_path, "w") as f:
            yaml.safe_dump(target_model_config.to_dict(), f, sort_keys=False)

    # synthetic data shadow model
    if not os.path.exists(cfg.synth_model_config_path):
        synth_model_config = make_scdesign2_config(cfg, False, "artifacts/synth", cfg.shadow_modelling_hvg_path, "synthetic.h5ad")
        with open(cfg.synth_model_config_path, "w") as f:
            yaml.safe_dump(synth_model_config.to_dict(), f, sort_keys=False)

    # aux data shadow model
    if not os.path.exists(cfg.aux_model_config_path):
        aux_model_config = make_scdesign2_config(cfg, False, "artifacts/aux", cfg.shadow_modelling_hvg_path, "auxiliary.h5ad")
        with open(cfg.aux_model_config_path, "w") as f:
            yaml.safe_dump(aux_model_config.to_dict(), f, sort_keys=False)


def make_scdesign2_config(cfg, generate, model_path, full_hvg_path, train_file_name):
    s2_cfg = Box()

    s2_cfg.dir_list = Box()
    s2_cfg.dir_list.home = cfg.trial_dir
    s2_cfg.dir_list.data = cfg.datasets_path

    s2_cfg.generator_name = "scdesign2"
    s2_cfg.train = True
    s2_cfg.generate = generate
    s2_cfg.load_from_checkpoint = False

    s2_cfg.scdesign2_config = Box()
    s2_cfg.scdesign2_config.out_model_path = model_path
    # s2_cfg.scdesign2_config.hvg_path = os.path.basename(os.path.normpath(hvg_path))
    s2_cfg.scdesign2_config.hvg_path = full_hvg_path

    s2_cfg.dataset_config = Box()
    s2_cfg.dataset_config.name = cfg.dataset_name
    s2_cfg.dataset_config.train_count_file = train_file_name
    s2_cfg.dataset_config.test_count_file = train_file_name
    s2_cfg.dataset_config.cell_type_col_name = "cell_type"
    s2_cfg.dataset_config.cell_label_col_name = "cell_label"
    s2_cfg.dataset_config.random_seed = 42

    return s2_cfg


def create_datasets_from_splits(cfg):
    all_data = ad.read_h5ad(os.path.join(cfg.top_data_dir, "full_dataset_cleaned.h5ad"))
    cell_types = all_data.obs["cell_type"].unique()

    if os.path.exists(cfg.train_donors_path) and os.path.exists(cfg.holdout_donors_path) and os.path.exists(cfg.aux_donors_path):
        resampled = False
        print("(skipping sampling)", flush=True)
    else:
        resampled = True
        create_data_splits_donor_MI(cfg, all_data)

    train_donors = np.load(cfg.train_donors_path, allow_pickle=True)
    holdout_donors = np.load(cfg.holdout_donors_path, allow_pickle=True)
    aux_donors = np.load(cfg.aux_donors_path, allow_pickle=True)

    all_train = all_data[all_data.obs["individual"].isin(train_donors)]
    all_holdout = all_data[all_data.obs["individual"].isin(holdout_donors)]
    all_aux = all_data[all_data.obs["individual"].isin(aux_donors)]
    print(f"Num Train cells: {len(all_train)}, Holdout: {len(all_holdout)}, Auxiliary: {len(all_aux)}")

    # save so that these are available during this experiment (will be removed after)
    all_train.write_h5ad(cfg.train_path)
    all_holdout.write_h5ad(cfg.holdout_path)
    all_aux.write_h5ad(cfg.aux_path)
    targets = ad.concat([all_train, all_holdout])

    # make full_results_file for target cells
    if not os.path.exists(cfg.all_scores_file):
        cols = targets.obs.columns.tolist()
        all_scores_df = pd.DataFrame({
            'cell id': targets.to_df().index,
            'donor id': targets.obs['individual'].values,
            'cell type': targets.obs['cell_type'].values,
            'sex': targets.obs['sex'].values if 'sex' in cols else None,
            'ethnicity': targets.obs['ethnicity'].values if 'ethnicity' in cols else None,
            'age': targets.obs['age'].values if 'age' in cols else None,
            'membership': np.concatenate([np.ones(len(all_train)), np.zeros(len(all_holdout))]),
        })
        all_scores_df.to_csv(cfg.all_scores_file, index=False)

    if not os.path.exists(cfg.results_file):
        donor_scores_df = pd.DataFrame({"metric": ["runtime", "auc"]})
        donor_scores_df.to_csv(cfg.results_file, index=False)

    return resampled, cell_types


def delete_h5ad_datasets(cfg):
    os.remove(cfg.train_path)
    os.remove(cfg.holdout_path)
    os.remove(cfg.aux_path)


def create_data_splits_donor_MI(cfg, all_data):
    cell_types = list(all_data.obs["cell_type"].unique())
    sample_strategy = cfg.sample_donors_strategy_fn

    all_train, all_holdout, all_aux = sample_strategy(cfg, all_data, cell_types)

    # all_train.write_h5ad(cfg.train_path)
    # all_holdout.write_h5ad(cfg.holdout_path)
    # all_aux.write_h5ad(cfg.aux_path)
    np.save(cfg.train_donors_path, all_train, allow_pickle=True)
    np.save(cfg.holdout_donors_path, all_holdout, allow_pickle=True)
    np.save(cfg.aux_donors_path, all_aux, allow_pickle=True)

    return cell_types


def generate_target_synthetic_data(cfg, cell_types, force=False):
    if not os.path.exists(cfg.target_synthetic_data_path) or force:
        run_scDesign2(cfg, cfg.target_model_config_path, cell_types, "TARGET_SYNTHETIC_DATA", force=True)
        return True
    else:
        print(f"\n\n(previously generated target data... skipping)", flush=True)
        return False


def run_scDesign2(cfg, scdesign2_cfg_path, cell_types, name, force=False):
    if not force:
        # check if copulas already generated from prior run
        with open(scdesign2_cfg_path, "r") as f:
            scdesign2_cfg = Box(yaml.load(f, Loader=yaml.FullLoader))
            copulas_path = scdesign2_cfg.scdesign2_config.out_model_path
            already_generated = True
            for cell_type in cell_types:
                if not os.path.exists(os.path.join(cfg.trial_dir, copulas_path, f"{cell_type}.rds")):
                    already_generated = False
                    break
            if already_generated:
                print(f"\n\n(previously simulated {name}... skipping)", flush=True)
                return

    print(f"\n\n\nGENERATING {name}\n_______________________", flush=True)
    start_process_time = time.process_time()
    start_wall_time = time.time()
    start_perf_time = time.perf_counter()

    run_singlecell_generator.callback(cfg_file=scdesign2_cfg_path)

    elapsed_process_time = time.process_time() - start_process_time
    elapsed_wall_time = time.time() - start_wall_time
    elapsed_perf_time = time.perf_counter() - start_perf_time

    pd.DataFrame({
        "elapsed_process_time": [elapsed_process_time],
        "elapsed_wall_time": [elapsed_wall_time],
        "elapsed_perf_time": [elapsed_perf_time],
    }).to_csv(os.path.join(cfg.results_path, f"{name}_runtime.csv"), index=False)


def mamamia_on_scdesign2(cfg):
    print("\n\n\nRUNNING MAMA-MIA\n_______________________", flush=True)
    train = ad.read_h5ad(cfg.train_path)
    holdout = ad.read_h5ad(cfg.holdout_path)
    cell_types = list(train.obs["cell_type"].unique())

    results = []
    if cfg.parallelize:
        print(f"\tparallelizing...", flush=True)
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_cell_type, cfg, ct, train, holdout): ct for ct in cell_types}
            for fut in as_completed(futures):
                fut_result = fut.result()
                print(f"\tfinished attacking cell {fut_result[0]}", flush=True)
                results.append(fut_result)
        results.sort(key=lambda x: x[0])
    else:
        print(f"\trunning sequentially...", flush=True)
        for ct in cell_types:
            result = process_cell_type(cfg, ct, train, holdout)
            results.append(result)
            print(f"\tfinished attacking cell {ct}", flush=True)

    return results


def get_params_based_on_threat_model(cfg):
    # Threat model parameter #1 white-box vs. black-box
    # If white-box, look at target model's copula, else generate one based on synthetic data
    copula_synth_path_ = cfg.models_path if cfg.mia_setting.white_box else cfg.synth_artifacts_path

    # Threat model parameter #2 knowledge of auxiliary data
    # If no knowledge of aux data, then use special function that omits it
    if cfg.mia_setting.use_aux:
        if cfg.mamamia_params.mahalanobis: attack_fn = attack_w_mahalanobis_algorithm
        else: attack_fn = attack_algorithm
    else:
        if cfg.mamamia_params.mahalanobis: attack_fn = attack_w_mahalanobis_algorithm_no_aux
        else:
            print("error!!! non-aux, non-mahalanobis attack not implemented!", flush=True)
            sys.exit(1)

    # Threat model parameter #3 knowledge of hvgs used by target model
    hvg_mask = pd.read_csv(cfg.shadow_modelling_hvg_path)
    hvgs = hvg_mask[hvg_mask["highly_variable"]].values[:, 0]

    return copula_synth_path_, attack_fn, hvgs



def process_cell_type(cfg, cell_type_, train, holdout):
    copula_synth_path_, attack_fn, hvgs = get_params_based_on_threat_model(cfg)

    copula_synth_path = os.path.join(copula_synth_path_, f"{cell_type_}.rds")
    copula_aux_path = os.path.join(cfg.aux_artifacts_path, f"{cell_type_}.rds")
    if not (os.path.exists(copula_aux_path) and os.path.exists(copula_synth_path)):
        return (cell_type_, None, None)  # skip this one

    # scDesign2's weird, particular way of storing the copula in an R object
    copula_aux = r["readRDS"](copula_aux_path).rx2(str(cell_type_))
    copula_synth = r["readRDS"](copula_synth_path).rx2(str(cell_type_))

    targets = create_target_dataset(cell_type_, hvgs, train, holdout)

    start = time.process_time()
    result_df = attack_fn(cfg, targets, cell_type_, copula_synth, copula_aux=copula_aux)
    runtime = time.process_time() - start

    return (cell_type_, result_df, runtime)


def attack_algorithm(cfg, targets, cell_type, copula_synth, copula_aux=None):
    primary_genes_s, secondary_genes_s, len_primary_s, len_secondary_s, cov_s, primary_marginal_params_s, secondary_marginal_params_s, get_correlation_fn_s, get_gene_params_fn_s = extract_copula_information(copula_synth)
    primary_genes_a, secondary_genes_a, len_primary_a, len_secondary_a, cov_a, primary_marginal_params_a, secondary_marginal_params_a, get_correlation_fn_a, get_gene_params_fn_a = extract_copula_information(copula_aux)
    covariate_genes_in_both_copulas, all_genes_in_both_copulas = get_shared_genes(primary_genes_s, secondary_genes_s, primary_genes_a, secondary_genes_a)

    FP_sums = np.zeros(len(targets))
    neg_synth_corrs = []
    neg_aux_corrs = []
    total_corrs = 0
    for i in range(len(covariate_genes_in_both_copulas)-1):
        gene1 = covariate_genes_in_both_copulas[i]
        targetcell_vals_s_1 = uniform_remapping(cfg, gene1, get_gene_params_fn_s, targets)
        targetcell_vals_a_1 = uniform_remapping(cfg, gene1, get_gene_params_fn_a, targets)

        for j in range(i+1, len(covariate_genes_in_both_copulas)):
            gene2 = covariate_genes_in_both_copulas[j]
            targetcell_vals_s_2 = uniform_remapping(cfg, gene2, get_gene_params_fn_s, targets)
            targetcell_vals_a_2 = uniform_remapping(cfg, gene2, get_gene_params_fn_a, targets)

            corr_s = get_correlation_fn_s(gene1, gene2)
            corr_a = get_correlation_fn_a(gene1, gene2)
            if corr_s < 0: neg_synth_corrs.append(corr_s)
            if corr_a < 0: neg_aux_corrs.append(corr_a)
            total_corrs += 1

            targets_closeness_to_correlation_s = cfg.closeness_to_correlation_fn(targetcell_vals_s_1, targetcell_vals_s_2, corr_s)
            targets_closeness_to_correlation_a = cfg.closeness_to_correlation_fn(targetcell_vals_a_1, targetcell_vals_a_2, corr_a)

            ave_corr_strength = (abs(corr_s) + abs(corr_a)) / 2
            lambda_ = ave_corr_strength * targets_closeness_to_correlation_s / (targets_closeness_to_correlation_a + cfg.mamamia_params.epsilon)
            # lambda_ = targets_closeness_to_correlation_s / (targets_closeness_to_correlation_a + cfg.mamamia_params.epsilon)
            FP_sums += lambda_


    # 1 - 2 * | CDF(x) - 0.5 |
    for gene in all_genes_in_both_copulas:
        targetcell_vals_s = uniform_remapping(cfg, gene, get_gene_params_fn_s, targets)
        targetcell_vals_a = uniform_remapping(cfg, gene, get_gene_params_fn_s, targets)
        targetcell_vals_s_ = 1 - 2 * np.abs(targetcell_vals_s - .5)
        targetcell_vals_a_ = 1 - 2 * np.abs(targetcell_vals_a - .5)
        FP_sums += (targetcell_vals_s_ / (targetcell_vals_a_ + cfg.mamamia_params.epsilon)) * cfg.mamamia_params.IMPORTANCE_OF_CLASS_B_FPs

    result_df = score_aggregations(cfg, cell_type, FP_sums, targets)
    return result_df


def attack_w_mahalanobis_algorithm(cfg, targets, cell_type, copula_synth, copula_aux=None):
    primary_genes_s, secondary_genes_s, len_primary_s, len_secondary_s, cov_s, primary_marginal_params_s, secondary_marginal_params_s, get_correlation_fn_s, get_gene_params_fn_s = extract_copula_information(copula_synth)
    primary_genes_a, secondary_genes_a, len_primary_a, len_secondary_a, cov_a, primary_marginal_params_a, secondary_marginal_params_a, get_correlation_fn_a, get_gene_params_fn_a = extract_copula_information(copula_aux)
    covariate_genes_in_both_copulas, all_genes_in_both_copulas = get_shared_genes(primary_genes_s, secondary_genes_s, primary_genes_a, secondary_genes_a)
    shared_cov_s, shared_genes_primary_marginals_s = create_shared_gene_corr_matrix(covariate_genes_in_both_copulas, primary_genes_s, cov_s, primary_marginal_params_s)
    shared_cov_a, shared_genes_primary_marginals_a = create_shared_gene_corr_matrix(covariate_genes_in_both_copulas, primary_genes_a, cov_a, primary_marginal_params_a)
    remapping_fn_vec = np.vectorize(cfg.uniform_remapping_fn)
    error_count = 0
    m_distances_s = []
    m_distances_a = []

    def mahalanobis_as_FPs(target_gene_expr):
        nonlocal error_count
        target_gene_expr_mapped_s = remapping_fn_vec(target_gene_expr, *np.moveaxis(shared_genes_primary_marginals_s, 1, 0))
        target_gene_expr_mapped_a = remapping_fn_vec(target_gene_expr, *np.moveaxis(shared_genes_primary_marginals_a, 1, 0))
        mean_s = remapping_fn_vec(shared_genes_primary_marginals_s[:,2], *np.moveaxis(shared_genes_primary_marginals_s, 1, 0))
        mean_a = remapping_fn_vec(shared_genes_primary_marginals_a[:,2], *np.moveaxis(shared_genes_primary_marginals_a, 1, 0))
        delta_s = target_gene_expr_mapped_s - mean_s
        delta_a = target_gene_expr_mapped_a - mean_a
        m_dist_synth = np.sqrt(delta_s.T @ cfg.lin_alg_inverse_fn(shared_cov_s) @ delta_s)
        m_distances_s.append(m_dist_synth)
        m_dist_aux = np.sqrt(delta_a.T @ cfg.lin_alg_inverse_fn(shared_cov_a) @ delta_a)
        m_distances_a.append(m_dist_aux)
        result = m_dist_aux / (m_dist_synth + m_dist_aux)
        if np.isnan(result):
            error_count += 1
            result = .5
        return result

    if error_count > 0: print(f"encountered {error_count} nans for cell type: {cell_type}")
    FP_sums = targets[covariate_genes_in_both_copulas].apply(mahalanobis_as_FPs, axis=1)

    result_df = score_aggregations(cfg, cell_type, FP_sums, targets, m_distances_s, m_distances_a)
    return result_df


def attack_w_mahalanobis_algorithm_no_aux(cfg, copula_synth, targets, cell_type):
    primary_genes_s, secondary_genes_s, len_primary_s, len_secondary_s, cov_s, primary_marginal_params_s, secondary_marginal_params_s, get_correlation_fn_s, get_gene_params_fn_s = extract_copula_information(copula_synth)
    shared_cov_s, shared_genes_primary_marginals_s = create_shared_gene_corr_matrix(primary_genes_s, primary_genes_s, cov_s, primary_marginal_params_s)
    remapping_fn_vec = np.vectorize(cfg.uniform_remapping_fn)
    error_count = 0
    m_distances_s = []

    def mahalanobis_as_FPs_no_aux(target_gene_expr):
        nonlocal error_count
        target_gene_expr_mapped_s = remapping_fn_vec(target_gene_expr, *np.moveaxis(shared_genes_primary_marginals_s, 1, 0))
        mean_s = remapping_fn_vec(shared_genes_primary_marginals_s[:,2], *np.moveaxis(shared_genes_primary_marginals_s, 1, 0))
        delta_s = target_gene_expr_mapped_s - mean_s
        m_dist_synth = np.sqrt(delta_s.T @ cfg.lin_alg_inverse_fn(shared_cov_s) @ delta_s)
        m_distances_s.append(m_dist_synth)
        result = 1 / (m_dist_synth + cfg.mamamia_params.epsilon)
        if np.isnan(result):
            error_count += 1
            result = .5
        return result

    if error_count > 0: print(f"encountered {error_count} nans for cell type: {cell_type}")
    FP_sums = targets[primary_genes_s].apply(mahalanobis_as_FPs_no_aux, axis=1)

    result_df = score_aggregations(cfg, cell_type, FP_sums, targets, m_distances_s, None)
    return result_df


def extract_copula_information(copula):
    primary_genes = copula.rx2("gene_sel1").names
    secondary_genes = copula.rx2("gene_sel2").names
    len_primary = len(primary_genes)
    len_secondary = len(secondary_genes)
    cov = copula.rx2("cov_mat")
    primary_marginal_params = copula.rx2("marginal_param1")
    secondary_marginal_params = copula.rx2("marginal_param2")

    genes_indexes = {gene: i for i, gene in enumerate(primary_genes)}

    all_genes_indexes = dict()
    for i, gene in enumerate(primary_genes):
        all_genes_indexes[gene] = (0, i)
    for i, gene in enumerate(secondary_genes):
        all_genes_indexes[gene] = (1, i)

    def get_correlation_fn(gene1, gene2):
        idx1 = genes_indexes[gene1]
        idx2 = genes_indexes[gene2]
        return cov[idx1 * len_primary + idx2]

    def get_gene_params_fn(gene_name):
        gene_class, idx = all_genes_indexes[gene_name]
        l = len_primary if gene_class == 0 else len_secondary
        marginal_params = (primary_marginal_params, secondary_marginal_params)[gene_class]
        dist_pi = marginal_params[idx]
        dist_theta = marginal_params[idx + l]
        dist_mu = marginal_params[idx + 2 * l]
        return dist_pi, dist_theta, dist_mu

    return primary_genes, secondary_genes, len_primary, len_secondary, cov, primary_marginal_params, secondary_marginal_params, get_correlation_fn, get_gene_params_fn


def create_target_dataset(cell_type, hvgs, train_h5, holdout_h5):
    train = train_h5[train_h5.obs["cell_type"] == cell_type].to_df().loc[:, hvgs]
    holdout = holdout_h5[holdout_h5.obs["cell_type"] == cell_type].to_df().loc[:, hvgs]
    train["member"] = True
    holdout["member"] = False

    train["individual"] = train_h5.obs["individual"]
    holdout["individual"] = holdout_h5.obs["individual"]

    for additional_feature in ["ethnicity", "age", "sex"]:
        train[additional_feature] = train_h5.obs[additional_feature] if additional_feature in train_h5.obs.columns else None
        holdout[additional_feature] = holdout_h5.obs[additional_feature] if additional_feature in holdout_h5.obs.columns else None
    return pd.concat([train, holdout])


def get_shared_genes(genes1_s, genes2_s, genes1_a, genes2_a):
    covariate_genes_in_both_copulas = list(set(genes1_s).intersection(set(genes1_a)))
    all_genes_in_both_copulas = list(
        ( set(genes1_s).union(set(genes2_s)) )
        .intersection(
        ( set(genes1_a).union(set(genes2_a)) )
    ))
    return covariate_genes_in_both_copulas, all_genes_in_both_copulas


def uniform_remapping(cfg, gene, get_gene_params_fn, targets):
    dist_pi, dist_theta, dist_mu = get_gene_params_fn(gene)
    # TODO: try adjusting these to actually be uniform: that is, spread out (uniformly or
    # TODO: proportionally, based on this array or all four arrays) to span the entire [0,1] range
    targetcell_vals = cfg.uniform_remapping_fn(targets[gene].values, dist_pi, dist_theta, dist_mu)
    return targetcell_vals


def create_shared_gene_corr_matrix(covariate_genes_in_both_copulas, genes, cov_matrix, marginal_params):
    l_both = len(covariate_genes_in_both_copulas)
    cov_matrix = np.array(cov_matrix)
    marginal_params = np.array(marginal_params)
    gene_indexes_ = np.array([genes.index(g) for g in covariate_genes_in_both_copulas])
    shared_genes_cov_matrix = np.zeros((l_both, l_both))
    for i, gene_idx in enumerate(gene_indexes_):
        shared_genes_cov_matrix[i] = cov_matrix[gene_idx][gene_indexes_]
    shared_genes_marginal_params = marginal_params[gene_indexes_]

    return shared_genes_cov_matrix, shared_genes_marginal_params


def score_aggregations(cfg, cell_type, FP_sums, targets, distances_s=None, distances_a=None):
    tm = get_threat_model_code(cfg)
    result_df = pd.DataFrame({
        'cell id': targets.index,
        'donor id': targets['individual'].values,
        'cell type': cell_type,
        'membership': targets['member'].values,
        # 'sex': targets['sex'].values,
        # 'ethnicity': targets['ethnicity'].values,
        # 'age': targets['age'].values,
        'distance_to_synth:' + tm: distances_s,
        'distance_to_aux:' + tm: distances_a,
        'score:' + tm: activate(np.array(FP_sums)),
    })
    return result_df


def save_results(cfg, results):
    tm = get_threat_model_code(cfg)
    new_full_result_df, full_runtime = concat_scores_for_all_celltypes(cfg, results)
    prior_full_results_df = pd.read_csv(cfg.all_scores_file, dtype=str)
    merged = pd.merge(prior_full_results_df, new_full_result_df, on=['cell id', 'donor id', 'cell type', 'membership'], suffixes=['', '_'+tm])
    merged.to_csv(cfg.all_scores_file, index=False)

    true_, predictions_ = aggregate_scores_by_donor(cfg, new_full_result_df)
    overall_auc = roc_auc_score(true_, predictions_)

    experiment_column_name = "tm:" + get_threat_model_code(cfg)
    prior_results_df = pd.read_csv(cfg.results_file)
    prior_results_df[experiment_column_name] = [full_runtime, overall_auc]
    prior_results_df.to_csv(cfg.results_file, index=False)

    # with open(os.path.join(cfg.results_path, cfg.results_file), "w") as results_file:
    #     results_file.write("membership, prediction\n")
    #     for membership, prediction in zip(true_, predictions_):
    #         message = f"{membership}, {prediction:.3f}"
    #         results_file.write(message + "\n")
    #     results_file.write(f"num cells, {new_full_result_df.shape[0]}" + "\n")
    #     results_file.write(f"aggregation runtime, {full_runtime}" + "\n")
    #     results_file.write(f"OVERALL, {overall_auc}" + "\n")

    print(f"num cells: {new_full_result_df.shape[0]}")
    print(f"aggregation runtime: {full_runtime}")
    print(f"OVERALL: {overall_auc}")


def aggregate_scores_by_donor(cfg, full_result_df):
    tm = get_threat_model_code(cfg)
    membership_true = []
    predictions = []
    grouped_predictions = full_result_df.groupby('donor id', observed=True)
    donors = full_result_df["donor id"].unique().tolist()
    for donor in donors:
        donor_membership = grouped_predictions.get_group(donor).membership.mean()
        assert donor_membership == 1 or donor_membership == 0, f"group membership ground truth is not consistent: {donor_membership}"
        membership_true.append(int(donor_membership))
        predictions.append(grouped_predictions.get_group(donor)["score:"+tm].mean())
    predictions = activate(np.array(predictions))

    return membership_true, predictions


# def analyze_final_results(cfg):
#     results = pd.read_csv(os.path.join(cfg.results_path, cfg.results_file))
#     print("FINAL RESULTS:")
#     print(results.iloc[-3:].to_numpy())


def concat_scores_for_all_celltypes(cfg, results):
    tm = get_threat_model_code(cfg)
    full_result_df = pd.DataFrame(columns=['cell id', 'donor id', 'cell type', 'distance_to_synth:'+tm, 'distance_to_aux:'+tm, 'membership', 'score:'+tm])

    full_runtime = 0
    for _cell_type, result_df, runtime in results:
        if (result_df is not None) and (not result_df["score:"+tm].isna().any()):
            full_result_df = pd.concat([full_result_df, result_df])
            full_runtime += runtime
        else:
            print("Skipping cell type", _cell_type)
            continue
    return full_result_df, full_runtime

#
# def plot_fn(cfg, cell_type, result_df):
#     auc_ = roc_auc_score(result_df["membership"].values, result_df["score"].values)
#     fpr, tpr, _ = roc_curve(result_df["membership"].values, result_df["score"].values)
#     plt.plot(fpr, tpr, label=f"{cell_type} (AUC={auc_:.2f})")
#     plt.plot([0,1],[0,1],'k--', linewidth=0.6)
#     plt.xlabel("FPR")
#     plt.ylabel("TPR")
#     plt.title(f"ROC for cell type {cell_type}")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(cfg.figures_path, f"{cfg.threat_model}_{cell_type}.png"))
#     if cfg.plot_results:
#         plt.show()

#
# def plot_fn(cfg, cell_type, result_df):
#     # --- ROC curve ---
#     auc_ = roc_auc_score(result_df["membership"].values, result_df["score"].values)
#     fpr, tpr, _ = roc_curve(result_df["membership"].values, result_df["score"].values)
#
#     plt.figure()
#     plt.plot(fpr, tpr, label=f"{cell_type} (AUC={auc_:.2f})")
#     plt.plot([0, 1], [0, 1], 'k--', linewidth=0.6)
#     plt.xlabel("FPR")
#     plt.ylabel("TPR")
#     plt.title(f"ROC for cell type {cell_type}")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(cfg.figures_path, f"{cfg.threat_model}_{cell_type}_roc.png"))
#
#     if cfg.plot_results:
#         plt.show()
#     plt.close()
#
#     # --- Boxplot: distance_to_synth by membership ---
#     plt.figure()
#
#     true_mask = result_df["membership"] == True
#     false_mask = result_df["membership"] == False
#
#     true_vals = result_df.loc[true_mask, "distance_to_synth"].values
#     false_vals = result_df.loc[false_mask, "distance_to_synth"].values
#
#     n_true = true_mask.sum()
#     n_false = false_mask.sum()
#
#     plt.boxplot(
#         [false_vals, true_vals],
#         labels=[f"Non-member (n={n_false})", f"Member (n={n_true})"],
#         showfliers=True
#     )
#
#     plt.ylabel("Distance to Synthetic")
#     plt.title(f"Distance to Synthetic for cell type: {cell_type}")
#     plt.tight_layout()
#
#     plt.savefig(
#         os.path.join(cfg.figures_path, f"{cfg.threat_model}_{cell_type}_distance_boxplot.png")
#     )
#
#     if cfg.plot_results:
#         plt.show()
#     plt.close()






if __name__ == "__main__":
    main()







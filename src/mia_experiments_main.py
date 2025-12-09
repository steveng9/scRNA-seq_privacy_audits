import sys

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

from discretionary_functions import *

if env == "local":
    from src.generators.blue_team import run_singlecell_generator, format_ct_name
else:
    from generators.blue_team import run_singlecell_generator, format_ct_name



def main():
    cfg = create_config(config_path)

    # 2. create data splits and necessary file structure
    make_dir_structure(cfg)
    make_experiment_config_files(cfg)
    resampled, cell_types = create_data_splits(cfg)
    determine_hvgs(cfg)

    # 3. generate target synthetic data
    regenerated = generate_target_synthetic_data(cfg, cell_types, force=resampled)

    # 4. generate focal points
    if not cfg.mia_setting.white_box:
        run_scDesign2(cfg, cfg.synth_model_config_path, cell_types, "SYNTHETIC DATA SHADOW MODEL", force=regenerated)
    else: print("\n\n(simulating scDesign2 on synthetic data not necessary in white_box setting... skipping)", flush=True)
    run_scDesign2(cfg, cfg.aux_model_config_path, cell_types, "AUXILIARY DATA SHADOW MODEL", force=resampled)

    # 5. set up wandb (experiment tracking)

    # 6. attack
    mamamia_on_scdesign2(cfg)


def create_config(config_path):
    with open(config_path) as f:
        cfg = Box(yaml.load(f, Loader=yaml.FullLoader))
        if cfg.mia_setting.donor_level:
            cfg.split_name = f"{cfg.mia_setting.num_donors}d"
        else:
            cfg.split_name = f"{cfg.mia_setting.dataset_size_k}k"
        cfg.top_data_dir = os.path.join(cfg.dir_list[env].data, cfg.dataset_name)
        cfg.cfg_dir = os.path.join(cfg.top_data_dir, cfg.split_name)
        cfg.data_path = os.path.join(cfg.cfg_dir, "datasets")
        cfg.results_path = os.path.join(cfg.cfg_dir, "results")
        cfg.figures_path = os.path.join(cfg.results_path, "figures")
        cfg.models_path = os.path.join(cfg.cfg_dir, "models")
        cfg.artifacts_path = os.path.join(cfg.cfg_dir, "artifacts")
        cfg.synth_artifacts_path = os.path.join(cfg.artifacts_path, "synth")
        cfg.aux_artifacts_path = os.path.join(cfg.artifacts_path, "aux")
        cfg.train_path = os.path.join(cfg.data_path, "train.h5ad")
        cfg.target_synthetic_data_path = os.path.join(cfg.data_path, "synthetic.h5ad")
        cfg.holdout_path = os.path.join(cfg.data_path, "holdout.h5ad")
        cfg.aux_path = os.path.join(cfg.data_path, "auxiliary.h5ad")
        cfg.threat_model = 'wb' if cfg.mia_setting.white_box else 'bb'
        cfg.results_filename = f"results_{cfg.threat_model}.csv"

        cfg.target_model_config_path = os.path.join(cfg.models_path, "config.yaml")
        cfg.synth_model_config_path = os.path.join(cfg.synth_artifacts_path, f"config_{cfg.threat_model}.yaml")
        cfg.aux_model_config_path = os.path.join(cfg.aux_artifacts_path, "config.yaml")

        cfg.parallelize = cfg.parallelize and not cfg.mamamia_params.mahalanobis

        cfg.hvg_mask_path = os.path.join(cfg.models_path, "hvg.csv")
        if cfg.mia_setting.use_wb_hvgs:
            cfg.shadow_modelling_hvg_path = cfg.models_path
        else:
            cfg.shadow_modelling_hvg_path = cfg.artifacts_path

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


def make_dir_structure(cfg):
    os.makedirs(cfg.data_path, exist_ok=True)
    os.makedirs(cfg.figures_path, exist_ok=True)
    os.makedirs(cfg.models_path, exist_ok=True)
    os.makedirs(cfg.synth_artifacts_path, exist_ok=True)
    os.makedirs(cfg.aux_artifacts_path, exist_ok=True)



def make_experiment_config_files(cfg):

    # target model
    if not os.path.exists(cfg.target_model_config_path):
        target_model_config = make_scdesign2_config(cfg, True, "models", cfg.models_path, "train.h5ad")
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


def make_scdesign2_config(cfg, generate, model_path, hvg_path, train_file_name):
    s2_cfg = Box()

    s2_cfg.dir_list = Box()
    s2_cfg.dir_list.home = cfg.cfg_dir
    s2_cfg.dir_list.data = cfg.data_path

    s2_cfg.generator_name = "scdesign2"
    s2_cfg.train = True
    s2_cfg.generate = generate
    s2_cfg.load_from_checkpoint = False

    s2_cfg.scdesign2_config = Box()
    s2_cfg.scdesign2_config.out_model_path = model_path
    s2_cfg.scdesign2_config.hvg_path = os.path.basename(os.path.normpath(hvg_path))

    s2_cfg.dataset_config = Box()
    s2_cfg.dataset_config.name = cfg.dataset_name
    s2_cfg.dataset_config.train_count_file = train_file_name
    s2_cfg.dataset_config.test_count_file = train_file_name
    s2_cfg.dataset_config.cell_type_col_name = "cell_type"
    s2_cfg.dataset_config.cell_label_col_name = "cell_label"
    s2_cfg.dataset_config.random_seed = 42

    return s2_cfg


def create_data_splits(cfg):
    if os.path.exists(cfg.train_path) and os.path.exists(cfg.holdout_path) and os.path.exists(cfg.aux_path):
        print("(skipping sampling)", flush=True)
        return False, get_cell_types_only(cfg)

    if cfg.mia_setting.donor_level:
        return True, create_data_splits_donor_MI(cfg)
    else:
        return True, create_data_splits_cell_MI(cfg)


def get_cell_types_only(cfg):
    all_data = ad.read_h5ad(cfg.train_path)
    all_data.obs["cell_type"] = all_data.obs["cell_type"].apply(format_ct_name)
    cell_types = list(all_data.obs["cell_type"].unique())
    return cell_types


def create_data_splits_donor_MI(cfg):
    all_data = ad.read_h5ad(os.path.join(cfg.top_data_dir, "full_dataset.h5ad"))
    all_data.obs["cell_type"] = all_data.obs["cell_type"].apply(format_ct_name)
    cell_types = list(all_data.obs["cell_type"].unique())
    sample_strategy = cfg.sample_donors_strategy_fn

    all_train, all_holdout, all_aux, num_donors_used = sample_strategy(cfg, all_data, cell_types)

    if cfg.mia_setting.num_donors > 1.95*num_donors_used:
        print("Too few donors for MIA", flush=True)
        print(f"Requested: {cfg.mia_setting.num_donors}, found: {num_donors_used}.", flush=True)
        print("exiting.", flush=True)
        sys.exit(0)

    all_train.write_h5ad(cfg.train_path)
    all_holdout.write_h5ad(cfg.holdout_path)
    all_aux.write_h5ad(cfg.aux_path)

    return cell_types


def create_data_splits_cell_MI(cfg):
    all_data = ad.read_h5ad(os.path.join(cfg.top_data_dir, "full_dataset.h5ad"))
    all_data.obs["cell_type"] = all_data.obs["cell_type"].apply(format_ct_name)
    cell_types = list(all_data.obs["cell_type"].unique())
    n_cells = min(cfg.mia_setting.dataset_size_k * 1000, all_data.n_obs)
    subset_idx = np.random.choice(all_data.n_obs, size=n_cells*3, replace=False)

    train_data = all_data[subset_idx[:n_cells], :].copy()
    train_data.write_h5ad(cfg.train_path)

    holdout_data = all_data[subset_idx[n_cells: n_cells*2], :].copy()
    holdout_data.write_h5ad(cfg.holdout_path)

    aux_data = all_data[subset_idx[n_cells*2: ], :].copy()
    aux_data.write_h5ad(cfg.aux_path)
    return cell_types


def determine_hvgs(cfg):
    if cfg.determine_new_hvgs:
        print("Determining new HVGs")
        raise NotImplementedError
    else:
        if not os.path.exists(cfg.hvg_mask_path): # then file should exist in root dir of dataset
            permenant_hvg_mask_path = os.path.join(cfg.top_data_dir, "hvg.csv")
            try:
                shutil.copy(permenant_hvg_mask_path, cfg.hvg_mask_path)
            except FileNotFoundError:
                print("HVG mask not found, creating new HVGs")


def generate_target_synthetic_data(cfg, cell_types, force=False):
    if not os.path.exists(cfg.target_synthetic_data_path) or force:
        run_scDesign2(cfg, cfg.target_model_config_path, cell_types, "TARGET DATA", force=True)
        return True
    else:
        print(f"\n\n(previously generated target data... skipping)", flush=True)
        return False


def run_scDesign2(cfg, scdesign2_cfg_path, cell_types, name, force=False):
    if not force:
        # check of copulas already generated from prior run
        with open(scdesign2_cfg_path, "r") as f:
            scdesign2_cfg = Box(yaml.load(f, Loader=yaml.FullLoader))
            copulas_path = scdesign2_cfg.scdesign2_config.out_model_path
            already_generated = True
            for cell_type in cell_types:
                if not os.path.exists(os.path.join(cfg.cfg_dir, copulas_path, f"{cell_type}.rds")):
                    already_generated = False
                    break
            if already_generated:
                print(f"\n\n(previously simulated {name}... skipping)", flush=True)
                return
    print(f"\n\n\nGENERATING {name}\n_______________________", flush=True)
    run_singlecell_generator.callback(cfg_file=scdesign2_cfg_path)


def mamamia_on_scdesign2(cfg):
    print("\n\n\nRUNNING MAMA-MIA\n_______________________", flush=True)
    train = ad.read_h5ad(cfg.train_path)
    holdout = ad.read_h5ad(cfg.holdout_path)
    cell_types = list(train.obs["cell_type"].unique())
    hvg_mask = pd.read_csv(os.path.join(cfg.shadow_modelling_hvg_path, "hvg.csv"))
    hvgs = hvg_mask[hvg_mask["highly_variable"]].values[:, 0]

    results = []
    if cfg.parallelize:
        print(f"\tparallelizing...", flush=True)
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_cell_type, cfg, ct, train, holdout, hvgs): ct for ct in cell_types}
            for fut in as_completed(futures):
                fut_result = fut.result()
                print(f"\tfinished attacking cell {fut_result[0]}", flush=True)
                results.append(fut_result)
        results.sort(key=lambda x: x[0])
    else:
        print(f"\trunning sequentially...", flush=True)
        for ct in cell_types:
            result = process_cell_type(cfg, ct, train, holdout, hvgs)
            results.append(result)
            print(f"\tfinished attacking cell {ct}", flush=True)

    print_final_results(cfg, results)


def process_cell_type(cfg, cell_type_, train, holdout, hvgs):
    targets = create_target_dataset(cell_type_, hvgs, train, holdout)
    copula_synth_path_ = cfg.models_path if cfg.mia_setting.white_box else cfg.synth_artifacts_path
    copula_synth_path = os.path.join(copula_synth_path_, f"{cell_type_}.rds")
    copula_aux_path = os.path.join(cfg.aux_artifacts_path, f"{cell_type_}.rds")
    if not (os.path.exists(copula_aux_path) and os.path.exists(copula_synth_path)):
        return (cell_type_, None)  # skip this one

    # scDesign2's weird, particular way of storing the copula
    copula_aux = r["readRDS"](copula_aux_path).rx2(str(cell_type_))
    copula_synth = r["readRDS"](copula_synth_path).rx2(str(cell_type_))

    if cfg.mamamia_params.mahalanobis:
        result_df = attack_w_mahalanobis_algorithm(cfg, copula_synth, copula_aux, targets, cell_type_)
    else:
        result_df = attack_algorithm(cfg, copula_synth, copula_aux, targets, cell_type_)

    return (cell_type_, result_df)


def attack_algorithm(cfg, copula_synth, copula_aux, targets, cell_type):
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
    try:
        plot_fn(cfg, cell_type, result_df)
    except ValueError:
        pass

    return result_df


def attack_w_mahalanobis_algorithm(cfg, copula_synth, copula_aux, targets, cell_type):
    primary_genes_s, secondary_genes_s, len_primary_s, len_secondary_s, cov_s, primary_marginal_params_s, secondary_marginal_params_s, get_correlation_fn_s, get_gene_params_fn_s = extract_copula_information(copula_synth)
    primary_genes_a, secondary_genes_a, len_primary_a, len_secondary_a, cov_a, primary_marginal_params_a, secondary_marginal_params_a, get_correlation_fn_a, get_gene_params_fn_a = extract_copula_information(copula_aux)
    covariate_genes_in_both_copulas, all_genes_in_both_copulas = get_shared_genes(primary_genes_s, secondary_genes_s, primary_genes_a, secondary_genes_a)
    shared_cov_s, shared_genes_primary_marginals_s = create_shared_gene_corr_matrix(covariate_genes_in_both_copulas, primary_genes_s, cov_s, primary_marginal_params_s)
    shared_cov_a, shared_genes_primary_marginals_a = create_shared_gene_corr_matrix(covariate_genes_in_both_copulas, primary_genes_a, cov_a, primary_marginal_params_a)
    remapping_fn_vec = np.vectorize(cfg.uniform_remapping_fn)
    error_count = 0

    def mahalanobis_as_FPs(target_gene_expr):
        nonlocal error_count
        target_gene_expr_mapped_s = remapping_fn_vec(target_gene_expr, *np.moveaxis(shared_genes_primary_marginals_s, 1, 0))
        target_gene_expr_mapped_a = remapping_fn_vec(target_gene_expr, *np.moveaxis(shared_genes_primary_marginals_a, 1, 0))
        mean_s = remapping_fn_vec(shared_genes_primary_marginals_s[:,2], *np.moveaxis(shared_genes_primary_marginals_s, 1, 0))
        mean_a = remapping_fn_vec(shared_genes_primary_marginals_a[:,2], *np.moveaxis(shared_genes_primary_marginals_a, 1, 0))
        delta_s = target_gene_expr_mapped_s - mean_s
        delta_a = target_gene_expr_mapped_a - mean_a
        m_dist_synth = np.sqrt(delta_s.T @ cfg.lin_alg_inverse_fn(shared_cov_s) @ delta_s)
        m_dist_aux = np.sqrt(delta_a.T @ cfg.lin_alg_inverse_fn(shared_cov_a) @ delta_a)
        result = m_dist_aux / (m_dist_synth + m_dist_aux)
        if np.isnan(result):
            error_count += 1
            result = .5
        return result

    if error_count > 0: print(f"encountered {error_count} nans for cell type: {cell_type}")
    FP_sums = targets[covariate_genes_in_both_copulas].apply(mahalanobis_as_FPs, axis=1)

    result_df = score_aggregations(cfg, cell_type, FP_sums, targets)
    try:
        plot_fn(cfg, cell_type, result_df)
    except ValueError:
        pass

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


def score_aggregations(cfg, cell_type, FP_sums, targets):
    result_df = pd.DataFrame({
        'cell id': targets.index,
        'donor id': targets['individual'].values,
        'cell type': cell_type,
        'membership': targets['member'].values,
        'score': activate(np.array(FP_sums))
    })
    return result_df


def print_final_results(cfg, results):
    true_, predictions_, num_cells = concat_scores_for_all_celltypes(cfg, results)
    overall_auc = roc_auc_score(true_, predictions_)

    with open(os.path.join(cfg.results_path, cfg.results_filename), "w") as results_file:
        results_file.write("membership, prediction\n")
        for membership, prediction in zip(true_, predictions_):
            message = f"{membership}, {prediction:.3f}"
            results_file.write(message + "\n")
        print(f"Average auc ({num_cells} cells): {overall_auc}")
        results_file.write(f"num cells, {num_cells}" + "\n")
        results_file.write(f"OVERALL, {overall_auc}" + "\n")


def concat_scores_for_all_celltypes(cfg, results):
    full_result_df = pd.DataFrame(columns=['cell id', 'donor id', 'cell type', 'membership', 'score'])

    for _cell_type, result_df in results:
        if (result_df is not None) and (not result_df.isna().any().any()):
            full_result_df = pd.concat([full_result_df, result_df])

    num_cells = full_result_df.shape[0]
    membership_true = []
    predictions = []
    if cfg.mia_setting.donor_level:
        grouped_predictions = full_result_df.groupby('donor id', observed=True)
        donors = full_result_df["donor id"].unique().tolist()
        for donor in donors:
            donor_membership = grouped_predictions.get_group(donor).membership.mean()
            assert donor_membership == 1 or donor_membership == 0, f"group membership ground truth is not consistent: {donor_membership}"
            membership_true.append(int(donor_membership))
            predictions.append(grouped_predictions.get_group(donor).score.mean())
        predictions = activate(np.array(predictions))
    else:
        membership_true = full_result_df["membership"].values
        predictions = full_result_df["score"].values

    return membership_true, predictions, num_cells

def plot_fn(cfg, cell_type, result_df):
    auc_ = roc_auc_score(result_df["membership"].values, result_df["score"].values)
    fpr, tpr, _ = roc_curve(result_df["membership"].values, result_df["score"].values)
    plt.plot(fpr, tpr, label=f"{cell_type} (AUC={auc_:.2f})")
    plt.plot([0,1],[0,1],'k--', linewidth=0.6)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC for cell type {cell_type}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.figures_path, f"{cfg.threat_model}_{cell_type}.png"))
    if cfg.plot_results:
        plt.show()




if __name__ == "__main__":
    main()







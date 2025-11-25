import sys
on_server = sys.argv[1] == "T"

from rpy2.robjects import r
import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import nbinom, poisson
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
import os
import itertools
import math
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed



dataset_name = "cg"
dataset_size_k = 1
mahalanobis = False
white_box = True
IMPORTANCE_OF_CLASS_B_FPs = .17
epsilon = .0001
plot_results = False
parallelize = not mahalanobis
# parallelize = False
donor_level = False
n_donors_per_cell_type = 50


if on_server:
    from generators.blue_team import run_singlecell_generator, format_ct_name
    home_dir = "/home/golobs/scRNA-seq_privacy_audits/"
    data_dir = "/home/golobs/data/sc/"
else:
    from src.generators.blue_team import run_singlecell_generator, format_ct_name
    home_dir = "/Users/stevengolob/PycharmProjects/camda_hpc/"
    data_dir = "/Users/stevengolob/Documents/school/PhD/Ghent_project_mia_scRNAseq/data/"

mia_artifacts_dir = home_dir + f"src/mia/{dataset_name}_artifacts/"



def main():

    print(f"\n\n\n\nNum Donors {n_donors_per_cell_type}" if donor_level else f"\n\n\n\nDataset size {dataset_size_k}K")
    update_config_files()

    # STEP 0: generate target synthetic data
    # run_singlecell_generator.callback(cfg_file=home_dir + f"{dataset_name}_config.yaml")

    # STEP 1: sample aux data
    # create_sample_of_aux_data()

    # STEP 2: generate focal points
    print(f"\tgenerating synth copulas")
    run_singlecell_generator.callback(cfg_file=mia_artifacts_dir + "run_on_synth.yaml")
    print(f"\tgenerating aux copulas")
    run_singlecell_generator.callback(cfg_file=mia_artifacts_dir + "run_on_aux.yaml")

    # STEP 3: make sense of fps in attack
    print(f"\tattacking...")
    dataset_name_train = f"train_subset_d{n_donors_per_cell_type}" if donor_level else f"train_subset_{dataset_size_k}k"
    dataset_name_test = f"test_subset_d{n_donors_per_cell_type}" if donor_level else f"test_subset_{dataset_size_k}k"
    train = ad.read_h5ad(data_dir + f"{dataset_name}/{dataset_name_train}.h5ad")
    test = ad.read_h5ad(data_dir + f"{dataset_name}/{dataset_name_test}.h5ad")
    train.obs["cell_type"] = train.obs["cell_type"].apply(format_ct_name)
    test.obs["cell_type"] = test.obs["cell_type"].apply(format_ct_name)
    hvg_mask = pd.read_csv(home_dir + f"models/{dataset_name}/hvg.csv")
    hvgs = hvg_mask[hvg_mask["highly_variable"]].values[:, 0]
    cell_types = [format_ct_name(cell_type) for cell_type in train.obs["cell_type"].unique()]


    aucs = []
    total_cells = 0
    results = []
    if parallelize:
        print(f"\tparallelizing...")
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_cell_type, ct, train, test, hvgs): ct for ct in cell_types}
            for fut in as_completed(futures):
                fut_result = fut.result()
                print(f"\tfinished attacking cell {fut_result[0]}, {fut_result[1]}")
                results.append(fut_result)
        # Sort results to preserve the order of cell_type
        results.sort(key=lambda x: x[0])
    else:
        print(f"\trunning sequentially...")
        for ct in cell_types:
            result = process_cell_type(ct, train, test, hvgs)
            results.append(result)
            print(f"\tfinished attacking cell {ct}, {result[1]}")

    for cell_type, auc_, num_scores in results:
        if auc_ is None:
            continue
        print(f"\t\tATTACK on {num_scores} {'Donors' if donor_level else 'Cells'} [{cell_type}]: {auc_:.3f}")
        aucs.append(auc_ * num_scores)
        total_cells += num_scores

    print(f"Average auc: {round(np.sum(aucs) / total_cells, 3)}")



def process_cell_type(cell_type_, train, test, hvgs):
    copula_path_part = f"d{n_donors_per_cell_type}/{cell_type_}.rds" if donor_level else f"{dataset_size_k}k/{cell_type_}.rds"

    copula_aux_path = mia_artifacts_dir + f"models_aux/{copula_path_part}"
    copula_synth_path = home_dir + f"models/{dataset_name}/{copula_path_part}" if white_box else mia_artifacts_dir + f"models_synth/{copula_path_part}"

    if not (os.path.exists(copula_aux_path) and os.path.exists(copula_synth_path)):
        return (cell_type_, None, None)  # skip this one

    copula_aux = r["readRDS"](copula_aux_path).rx2(str(cell_type_))
    copula_synth = r["readRDS"](copula_synth_path).rx2(str(cell_type_))
    if mahalanobis:
        _, _, score, num_targets = attack_w_mahalanobis(hvgs, copula_synth, copula_aux, train, test, cell_type_)
    else:
        _, _, score, num_targets = first_attack(hvgs, copula_synth, copula_aux, train, test, cell_type_)

    return (cell_type_, score, num_targets)



def update_config_files():
    config_filename = home_dir + f"{dataset_name}_config.yaml"

    with open(config_filename, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["dataset_config"]["train_count_file"] = f"{dataset_name}/train_subset_d{n_donors_per_cell_type}.h5ad" if donor_level else f"{dataset_name}/train_subset_{dataset_size_k}k.h5ad"
    cfg["dataset_config"]["test_count_file"] = f"{dataset_name}/test_subset_d{n_donors_per_cell_type}.h5ad" if donor_level else f"{dataset_name}/test_subset_{dataset_size_k}k.h5ad"
    cfg["dataset_config"]["synthetic_data_name"] = f"synthetic_d{n_donors_per_cell_type}.h5ad" if donor_level else f"synthetic_{dataset_size_k}k.h5ad"
    cfg["scdesign2_config"]["out_model_path"] = f"models/{dataset_name}/d{n_donors_per_cell_type}" if donor_level else f"models/{dataset_name}/{dataset_size_k}k"
    os.makedirs(home_dir + cfg["scdesign2_config"]["out_model_path"], exist_ok=True)
    with open(config_filename, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    config_filename = mia_artifacts_dir + "run_on_synth.yaml"
    with open(config_filename, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["dataset_config"]["train_count_file"] = f"{dataset_name}/synthetic_data/scdesign2/synthetic_d{n_donors_per_cell_type}.h5ad" if donor_level else f"{dataset_name}/synthetic_data/scdesign2/synthetic_{dataset_size_k}k.h5ad"
    cfg["scdesign2_config"]["out_model_path"] = f"src/mia/{dataset_name}_artifacts/models_synth/d{n_donors_per_cell_type}" if donor_level else f"src/mia/{dataset_name}_artifacts/models_synth/{dataset_size_k}k"
    os.makedirs(home_dir + cfg["scdesign2_config"]["out_model_path"], exist_ok=True)
    with open(config_filename, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    config_filename = mia_artifacts_dir + "run_on_aux.yaml"
    with open(config_filename, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["dataset_config"]["train_count_file"] = f"{dataset_name}/aux_subset_d{n_donors_per_cell_type}.h5ad" if donor_level else f"{dataset_name}/aux_subset_{dataset_size_k}k.h5ad"
    cfg["scdesign2_config"]["out_model_path"] = f"src/mia/{dataset_name}_artifacts/models_aux/d{n_donors_per_cell_type}" if donor_level else f"src/mia/{dataset_name}_artifacts/models_aux/{dataset_size_k}k"
    os.makedirs(home_dir + cfg["scdesign2_config"]["out_model_path"], exist_ok=True)
    with open(config_filename, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def plot_fn(cell_type, membership_true, membership_scores):
    auc_ = roc_auc_score(membership_true, membership_scores)
    if plot_results:
        fpr, tpr, _ = roc_curve(membership_true, membership_scores)
        plt.plot(fpr, tpr, label=f"{cell_type} (AUC={auc_:.2f})")
        plt.plot([0,1],[0,1],'k--', linewidth=0.6)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC for cell type {cell_type}")
        plt.legend()
        plt.tight_layout()
        plt.show()
    return auc_


def first_attack(hvgs, copula_synth, copula_aux, train_h5, test_h5, cell_type):
    train = train_h5[train_h5.obs["cell_type"] == cell_type].to_df().loc[:, hvgs]
    test = test_h5[test_h5.obs["cell_type"] == cell_type].to_df().loc[:, hvgs]
    train["member"] = True
    test["member"] = False
    train["individual"] = train_h5.obs["individual"]
    test["individual"] = test_h5.obs["individual"]
    targets = pd.concat([train, test])

    synth_genes1 = copula_synth.rx2("gene_sel1").names
    synth_genes2 = copula_synth.rx2("gene_sel2").names
    l1_synth = len(synth_genes1)
    l2_synth = len(synth_genes2)
    cov_synth = copula_synth.rx2("cov_mat")
    marginal_params1_synth = copula_synth.rx2("marginal_param1")
    marginal_params2_synth = copula_synth.rx2("marginal_param2")
    aux_genes1 = copula_aux.rx2("gene_sel1").names
    aux_genes2 = copula_aux.rx2("gene_sel2").names
    l1_aux = len(aux_genes1)
    l2_aux = len(aux_genes2)
    cov_aux = copula_aux.rx2("cov_mat")
    marginal_params1_aux = copula_aux.rx2("marginal_param1")
    marginal_params2_aux = copula_aux.rx2("marginal_param2")

    covariate_genes_in_both_copulas = list(set(synth_genes1).intersection(set(aux_genes1)))
    all_genes_in_both_copulas = list( (set(synth_genes1).union(set(synth_genes2))) \
                                      .intersection(
                                      (set(aux_genes1).union(set(aux_genes2))) ) )
    # print(f"Cell type {cell_type} has {len(covariate_genes_in_both_copulas)} shared covariate genes in synth and aux")
    # print(f"synth: {len(synth_genes1)}, aux: {len(aux_genes1)}")


    genes_indexes_aux, genes_indexes_synth = dict(), dict()
    for gene in covariate_genes_in_both_copulas:
        genes_indexes_synth[gene] = synth_genes1.index(gene)
        genes_indexes_aux[gene] = aux_genes1.index(gene)

    def get_gene_params_from_copula(gene_name):
        i_synth = genes_indexes_synth[gene_name]
        synth_dist_pi = marginal_params1_synth[i_synth]
        synth_dist_theta = marginal_params1_synth[i_synth + l1_synth]
        synth_dist_mu = marginal_params1_synth[i_synth + 2 * l1_synth]
        i_aux = genes_indexes_aux[gene_name]
        aux_dist_pi = marginal_params1_aux[i_aux]
        aux_dist_theta = marginal_params1_aux[i_aux + l1_aux]
        aux_dist_mu = marginal_params1_aux[i_aux + 2 * l1_aux]
        return i_synth, i_aux, synth_dist_pi, synth_dist_theta, synth_dist_mu, aux_dist_pi, aux_dist_theta, aux_dist_mu

    all_genes_indexes_aux, all_genes_indexes_synth = dict(), dict()
    for gene in synth_genes1:
        all_genes_indexes_synth[gene] = (0, synth_genes1.index(gene))
    for gene in synth_genes2:
        all_genes_indexes_synth[gene] = (1, synth_genes2.index(gene))
    for gene in aux_genes1:
        all_genes_indexes_aux[gene] = (0, aux_genes1.index(gene))
    for gene in aux_genes2:
        all_genes_indexes_aux[gene] = (1, aux_genes2.index(gene))

    def get_arbitrary_gene_params_from_copula(gene_name):
        gene_class_synth, i_synth = all_genes_indexes_synth[gene_name]
        l = l1_synth if gene_class_synth == 0 else l2_synth
        synth_params = (marginal_params1_synth, marginal_params2_synth)[gene_class_synth]
        synth_dist_pi = synth_params[i_synth]
        synth_dist_theta = synth_params[i_synth + l]
        synth_dist_mu = synth_params[i_synth + 2 * l]

        gene_class_aux, i_aux = all_genes_indexes_aux[gene_name]
        l = l1_aux if gene_class_aux == 0 else l2_aux
        aux_params = (marginal_params1_aux, marginal_params2_aux)[gene_class_aux]
        aux_dist_pi = aux_params[i_aux]
        aux_dist_theta = aux_params[i_aux + l]
        aux_dist_mu = aux_params[i_aux + 2 * l]
        return synth_dist_pi, synth_dist_theta, synth_dist_mu, aux_dist_pi, aux_dist_theta, aux_dist_mu


    FP_sums = np.zeros(len(targets))
    # uniform_remapping_fn = zinb_uniform_transform
    uniform_remapping_fn = zinb_cdf
    # uniform_remapping_fn = zinb_cdf_DT
    neg_synth_corrs = []
    neg_aux_corrs = []
    total_corrs = 0
    for i in range(len(covariate_genes_in_both_copulas)-1):
        gene1 = covariate_genes_in_both_copulas[i]
        i_synth_1, i_aux_1, synth_dist_pi_1, synth_dist_theta_1, synth_dist_mu_1, aux_dist_pi_1, aux_dist_theta_1, aux_dist_mu_1 = get_gene_params_from_copula(gene1)

        # TODO: try adjusting these to actually be uniform: that is, spread out (uniformly or
        # TODO: proportionally, based on this array or all four arrays) to span the entire [0,1] range
        synth_targetcell_vals_1 = uniform_remapping_fn(targets[gene1].values, synth_dist_pi_1, synth_dist_theta_1, synth_dist_mu_1)
        aux_targetcell_vals_1 = uniform_remapping_fn(targets[gene1].values, aux_dist_pi_1, aux_dist_theta_1, aux_dist_mu_1)

        for j in range(i+1, len(covariate_genes_in_both_copulas)):
            gene2 = covariate_genes_in_both_copulas[j]
            i_synth_2, i_aux_2, synth_dist_pi_2, synth_dist_theta_2, synth_dist_mu_2, aux_dist_pi_2, aux_dist_theta_2, aux_dist_mu_2 = get_gene_params_from_copula(gene2)
            synth_targetcell_vals_2 = uniform_remapping_fn(targets[gene2].values, synth_dist_pi_2, synth_dist_theta_2, synth_dist_mu_2)
            aux_targetcell_vals_2 = uniform_remapping_fn(targets[gene2].values, aux_dist_pi_2, aux_dist_theta_2, aux_dist_mu_2)

            corr_synth = cov_synth[i_synth_1 * l1_synth + i_synth_2]
            corr_aux = cov_aux[i_aux_1 * l1_aux + i_aux_2]
            if corr_synth < 0: neg_synth_corrs.append(corr_synth)
            if corr_aux < 0: neg_aux_corrs.append(corr_aux)
            total_corrs += 1

            # TODO: define a better metric here, where higher score means closer to y=x correlation
            # def corrs_closeness_to_one(vals1, vals2):
            #     ratios = vals2 / (vals1 + epsilon)
            #     ratios = np.where(ratios > 1, (1/ratios), ratios)
            #     return ratios

            # simple absolute different, scaled by distance to origin
            def closeness_to_correlation(vals1, vals2, _correlation):
                similarity = 1 - (np.abs(vals1 - vals2) / np.maximum(vals1, vals2))
                return similarity

            # same as previous, but switch to y=-x+1 line if correlation is negative
            # def closeness_to_correlation(vals1, vals2, correlation):
            #     if correlation >= 0:
            #         similarity = 1 - (np.abs(vals1 - vals2) / np.maximum(vals1, vals2))
            #     else:
            #         similarity = 1 - (np.abs(vals1 - (1-vals2)) / np.maximum(vals1, (1-vals2)))
            #     return similarity

            # more sophisticated, evaluates point against the line y = cx + .5 - c/2
            # which is just a line that passes through (.5, .5) and represents the correlation
            # def closeness_to_correlation(vals1, vals2, correlation):
            #     expected_vals2 = correlation * vals1 + .5 - correlation / 2
            #     similarity = 1-np.abs(expected_vals2 - vals2)
            #     return similarity


            targets_closeness_to_correlation_synth = closeness_to_correlation(synth_targetcell_vals_1, synth_targetcell_vals_2, corr_synth)
            targets_closeness_to_correlation_aux = closeness_to_correlation(aux_targetcell_vals_1, aux_targetcell_vals_2, corr_aux)

            ave_corr_strength = (abs(corr_synth) + abs(corr_aux)) / 2
            lambda_ = ave_corr_strength * targets_closeness_to_correlation_synth / (targets_closeness_to_correlation_aux + epsilon)
            # lambda_ = targets_closeness_to_correlation_synth / (targets_closeness_to_correlation_aux + epsilon)
            FP_sums += lambda_


    # 1 - 2 * | CDF(x) - 0.5 |
    for i in range(len(all_genes_in_both_copulas)):
        gene = all_genes_in_both_copulas[i]
        synth_dist_pi_, synth_dist_theta_, synth_dist_mu_, aux_dist_pi_, aux_dist_theta_, aux_dist_mu_ = get_arbitrary_gene_params_from_copula(gene)
        synth_targetcell_vals_ = uniform_remapping_fn(targets[gene].values, synth_dist_pi_, synth_dist_theta_, synth_dist_mu_)
        aux_targetcell_vals_ = uniform_remapping_fn(targets[gene].values, aux_dist_pi_, aux_dist_theta_, aux_dist_mu_)
        synth_targetcell_vals = 1 - 2 * np.abs(synth_targetcell_vals_ - .5)
        aux_targetcell_vals = 1 - 2 * np.abs(aux_targetcell_vals_ - .5)
        FP_sums += (synth_targetcell_vals / (aux_targetcell_vals + epsilon)) * IMPORTANCE_OF_CLASS_B_FPs


    membership_true, membership_scores = score_aggregations(cell_type, FP_sums, targets, train_h5, test_h5)
    auc_ = plot_fn(cell_type, membership_true, membership_scores)

    return membership_true, membership_scores, auc_, len(membership_true)


def create_shared_gene_corr_matrix(covariate_genes_in_both_copulas, genes, cov_matrix, marginal_params):
    l_both = len(covariate_genes_in_both_copulas)
    genes_indexes =  dict()
    for gene in covariate_genes_in_both_copulas:
        genes_indexes[gene] = genes.index(gene)

    gene_indexes_ = np.array([genes_indexes[s] for s in covariate_genes_in_both_copulas])
    shared_genes_cov_matrix = np.zeros((l_both, l_both))
    for i, gene_idx in enumerate(gene_indexes_):
        shared_genes_cov_matrix[i] = cov_matrix[gene_idx][gene_indexes_]
    shared_genes_marginal_params = marginal_params[gene_indexes_]

    return genes_indexes, shared_genes_cov_matrix, shared_genes_marginal_params


def score_aggregations(cell_type, FP_sums, targets, train_h5, test_h5):

    if donor_level:
        predictions = pd.DataFrame({
            'id': targets['individual'].values,
            'A': pd.Series(FP_sums),
        })
        scores = []
        membership_true = []
        grouped_predictions = predictions.groupby('id')
        individuals = targets["individual"].unique().tolist()
        individuals_train = train_h5[train_h5.obs["cell_type"] == cell_type].obs["individual"].unique().tolist()
        individuals_holdout = test_h5[test_h5.obs["cell_type"] == cell_type].obs["individual"].unique().tolist()
        for id in individuals:
            scores.append(grouped_predictions.get_group(id).A.mean())
            if id in individuals_train:
                membership_true.append(1)
            if id in individuals_holdout:
                membership_true.append(0)
        membership_scores = activate_3(np.array(scores))
    else:
        membership_true = targets["member"].astype(int).values
        membership_scores = activate_3(np.array(FP_sums))
    return membership_true, membership_scores


def attack_w_mahalanobis(hvgs, copula_synth, copula_aux, train_h5, test_h5, cell_type):
    train = train_h5[train_h5.obs["cell_type"] == cell_type].to_df().loc[:, hvgs]
    test = test_h5[test_h5.obs["cell_type"] == cell_type].to_df().loc[:, hvgs]
    train["member"] = True
    test["member"] = False
    train["individual"] = train_h5.obs["individual"]
    test["individual"] = test_h5.obs["individual"]
    targets = pd.concat([train, test])

    synth_genes1 = copula_synth.rx2("gene_sel1").names
    synth_genes2 = copula_synth.rx2("gene_sel2").names
    l1_synth = len(synth_genes1)
    l2_synth = len(synth_genes2)
    cov_synth = np.array(copula_synth.rx2("cov_mat"))
    marginal_params1_synth = np.array(copula_synth.rx2("marginal_param1"))
    marginal_params2_synth = np.array(copula_synth.rx2("marginal_param2"))
    aux_genes1 = copula_aux.rx2("gene_sel1").names
    aux_genes2 = copula_aux.rx2("gene_sel2").names
    l1_aux = len(aux_genes1)
    l2_aux = len(aux_genes2)
    cov_aux = np.array(copula_aux.rx2("cov_mat"))
    marginal_params1_aux = np.array(copula_aux.rx2("marginal_param1"))
    marginal_params2_aux = np.array(copula_aux.rx2("marginal_param2"))
    covariate_genes_in_both_copulas = list(set(synth_genes1).intersection(set(aux_genes1)))
    all_genes_in_both_copulas = list( (set(synth_genes1).union(set(synth_genes2))) \
                                      .intersection(
                                      (set(aux_genes1).union(set(aux_genes2))) ) )
    gene_indexes_synth, shared_cov_synth, shared_genes_marginals1_synth = create_shared_gene_corr_matrix(covariate_genes_in_both_copulas, synth_genes1, cov_synth, marginal_params1_synth)
    gene_indexes_aux, shared_cov_aux, shared_genes_marginals1_aux = create_shared_gene_corr_matrix(covariate_genes_in_both_copulas, aux_genes1, cov_aux, marginal_params1_aux)


    uniform_remapping_fn = zinb_cdf
    remapping_fn_vec = np.vectorize(zinb_cdf)


    def mahalanobis_as_FPs(target_gene_expr):
        target_gene_expr_mapped_synth = remapping_fn_vec(target_gene_expr, *np.moveaxis(shared_genes_marginals1_synth, 1, 0))
        target_gene_expr_mapped_aux = remapping_fn_vec(target_gene_expr, *np.moveaxis(shared_genes_marginals1_aux, 1, 0))
        # synth_mean_mapped = remapping_fn_vec(mean_expr_synth.loc[covariate_genes_in_both_copulas], *np.moveaxis(shared_genes_marginals1_synth, 1, 0))
        # aux_mean_mapped = remapping_fn_vec(mean_expr_aux.loc[covariate_genes_in_both_copulas], *np.moveaxis(shared_genes_marginals1_aux, 1, 0))
        synth_mean_mapped = remapping_fn_vec(shared_genes_marginals1_synth[:,2], *np.moveaxis(shared_genes_marginals1_synth, 1, 0))
        aux_mean_mapped = remapping_fn_vec(shared_genes_marginals1_aux[:,2], *np.moveaxis(shared_genes_marginals1_aux, 1, 0))
        delta_synth = target_gene_expr_mapped_synth - synth_mean_mapped
        delta_aux = target_gene_expr_mapped_aux - aux_mean_mapped

        # m_dist_synth = np.sqrt(delta_synth.T @ np.linalg.inv(shared_cov_synth) @ delta_synth)
        # m_dist_aux = np.sqrt(delta_aux.T @ np.linalg.inv(shared_cov_aux) @ delta_aux)

        # m_dist_synth = np.sqrt(delta_synth.T @ np.linalg.solve(shared_cov_synth, delta_synth))
        # m_dist_aux = np.sqrt(delta_aux.T @ np.linalg.solve(shared_cov_aux, delta_aux))

        m_dist_synth = np.sqrt(delta_synth.T @ np.linalg.pinv(shared_cov_synth) @ delta_synth)
        m_dist_aux = np.sqrt(delta_aux.T @ np.linalg.pinv(shared_cov_aux) @ delta_aux)

        result = m_dist_aux / (m_dist_synth + m_dist_aux)
        if np.isnan(result):
            print("hi")
            result = .5
        return result

    FP_sums = targets[covariate_genes_in_both_copulas].apply(mahalanobis_as_FPs, axis=1)

    membership_true, membership_scores = score_aggregations(cell_type, FP_sums, targets, train_h5, test_h5)
    auc_ = plot_fn(cell_type, membership_true, membership_scores)

    return membership_true, membership_scores, auc_, len(membership_true)




def activate_3(p_rel, confidence=1, center=True) -> np.ndarray:
    logs = np.log(p_rel)
    zscores = stats.zscore(logs)
    median = np.median(zscores) if center else 0
    probabilities = 1 / (1 + np.exp(-1 * confidence * (zscores - median)))
    return probabilities

def zinb_cdf(x, pi, theta, mu):
    """
    CDF of Zero-Inflated Negative Binomial at value x.
    Returns the probability P(X <= x).
    """
    x = np.asarray(x)
    if np.isinf(theta):  # Poisson case
        F = poisson.cdf(x, mu)
    # elif mu == 0 or mu == np.inf:
    #     F = .5
    else:
        n = theta
        p = theta / (theta + mu)
        F = nbinom.cdf(x, n, p)
    return pi + (1 - pi) * F

def zinb_cdf_DT(x, pi, theta, mu, jitter=True):
    x = np.asarray(x)
    if np.isinf(theta):
        F_x = poisson.cdf(x, mu)
        F_xm1 = poisson.cdf(x - 1, mu)
    else:
        n = theta
        p = theta / (theta + mu)
        F_x = nbinom.cdf(x, n, p)
        F_xm1 = nbinom.cdf(x - 1, n, p)
    F_x = pi + (1 - pi) * F_x
    F_xm1 = pi + (1 - pi) * F_xm1 * (x > 0)

    if jitter:
        v = np.random.rand(*x.shape)
    else:
        v = 0.5  # midpoint
    return F_xm1 + v * (F_x - F_xm1)

def zinb_uniform_transform(x, pi, theta, mu, jitter=True):
    """
    Properly maps ZINB-distributed counts to uniform(0,1) via
    the distributional transform, removing zero inflation bias.
    """
    x = np.asarray(x, dtype=int)

    if np.isinf(theta):
        F_nb = poisson.cdf(x, mu)
        F_nb_prev = poisson.cdf(x - 1, mu)
        p0_nb = poisson.pmf(0, mu)
    else:
        n = theta
        p = theta / (theta + mu)
        F_nb = nbinom.cdf(x, n, p)
        F_nb_prev = nbinom.cdf(x - 1, n, p)
        p0_nb = nbinom.pmf(0, n, p)

    # Overall probability of X=0
    p_zero_total = pi + (1 - pi) * p0_nb

    # Uniform jitter
    v = np.random.rand(*x.shape) if jitter else 0.5

    u = np.empty_like(x, dtype=float)

    # Case 1: zero observations
    mask_zero = (x == 0)
    # if np.any(mask_zero):
    #     # Spread zeros uniformly within [0, p_zero_total)
    #     u[mask_zero] = v[mask_zero] * p_zero_total

    # Case 2: nonzero observations
    mask_nonzero = ~mask_zero
    if np.any(mask_nonzero):
        F_x = F_nb[mask_nonzero]
        F_xm1 = F_nb_prev[mask_nonzero]
        # conditional CDF given X>0
        F_cond_x = (F_x - p0_nb) / (1 - p0_nb)
        F_cond_xm1 = (F_xm1 - p0_nb) / (1 - p0_nb)
        # mix into upper (1 - p_zero_total) portion
        u[mask_nonzero] = p_zero_total + (1 - p_zero_total) * (
            F_cond_xm1 + v[mask_nonzero] * (F_cond_x - F_cond_xm1)
        )

    return u


def create_sample_of_aux_data():
    input_path = Path(data_dir + f"{dataset_name}/train_full.h5ad")
    adata = ad.read_h5ad(input_path)
    n_cells = min(dataset_size_k*1000, adata.n_obs)
    subset_idx = np.random.choice(adata.n_obs, size=n_cells, replace=False)
    adata_subset = adata[subset_idx, :].copy()
    adata_subset.write_h5ad(data_dir + f"{dataset_name}/aux_subset_{n_cells//1_000}k.h5ad")



if __name__ == "__main__":
    main()







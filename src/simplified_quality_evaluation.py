import sys
env = "server" if sys.argv[1] == "T" else "local"
config_path = sys.argv[2]



import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from box import Box
import yaml
import anndata as ad
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


if env == "server":
    from evaluation.sc_evaluate import SingleCellEvaluator
else:
    from src.evaluation.sc_evaluate import SingleCellEvaluator

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=ImportWarning)

# src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(src_dir)



def main():
    cfg = create_config()

    train_donors = np.load(cfg.train_donor_path, allow_pickle=True)
    all_data = ad.read_h5ad(os.path.join(cfg.dir_list[env].data, cfg.dataset_name, "full_dataset_cleaned.h5ad"))
    train = all_data[all_data.obs["individual"].isin(train_donors)]
    synth = ad.read_h5ad(cfg.synth_path)

    quality_eval_cfg = make_quality_eval_cfg(cfg)

    results = evaluate(quality_eval_cfg, train, synth)

    # pd.DataFrame([results]).to_csv(cfg.results_file, index=False)
    register_quality_check(cfg)
    print("\nDONE! Saved results to:", cfg.results_file)


def create_config():
    with open(config_path) as f:
        cfg = Box(yaml.load(f, Loader=yaml.FullLoader))
        cfg.split_name = f"{cfg.mia_setting.num_donors}d"
        cfg.top_data_dir = os.path.join(cfg.dir_list[env].data, cfg.dataset_name)
        cfg.cfg_dir = os.path.join(cfg.top_data_dir, cfg.split_name)
        cfg.experiment_tracking_file = os.path.join(cfg.cfg_dir, "tracking.csv")
        cfg.trial_num = get_next_quality_trial_num(cfg)
        cfg.trial_dir = os.path.join(cfg.cfg_dir, str(cfg.trial_num))

        cfg.experiment_data_path = os.path.join(cfg.trial_dir, "datasets")
        cfg.results_path = os.path.join(cfg.trial_dir, "results")
        cfg.results_file = os.path.join(cfg.results_path, "quality_results.csv")
        os.makedirs(cfg.results_path, exist_ok=True)
        cfg.train_donor_path = os.path.join(cfg.experiment_data_path, "train.npy")
        cfg.train_path = os.path.join(cfg.experiment_data_path, "train.h5ad")
        cfg.synth_path = os.path.join(cfg.experiment_data_path, "synthetic.h5ad")

    print("Experiment Configuration:")
    print("dataset: ", cfg.dataset_name)
    print("num_donors: ", cfg.mia_setting.num_donors)
    print(f"TRIAL number: ", cfg.trial_num)
    return cfg

def make_quality_eval_cfg(cfg):
    quality_eval_cfg = dict()

    quality_eval_cfg["dir_list"] = dict()
    quality_eval_cfg["dir_list"]["home"] = os.path.join(cfg.results_path, "quality_eval_results")
    quality_eval_cfg["dir_list"]["figures"] = "figures"
    quality_eval_cfg["dir_list"]["res_files"] = "results"
    quality_eval_cfg["full_data_path"] = os.path.join(cfg.top_data_dir, "full_dataset_cleaned.h5ad")
    quality_eval_cfg["synthetic_file"] = cfg.synth_path


    quality_eval_cfg["dataset_config"] = dict()
    quality_eval_cfg["dataset_config"]["name"] = cfg.dataset_name
    quality_eval_cfg["dataset_config"]["test_count_file"] = cfg.train_donor_path
    quality_eval_cfg["dataset_config"]["synthetic_file"] = cfg.synth_path
    quality_eval_cfg["dataset_config"]["cell_type_col_name"] = "cell_type"
    quality_eval_cfg["dataset_config"]["cell_label_col_name"] = "cell_label"
    quality_eval_cfg["dataset_config"]["celltypist_model"] = os.path.join(cfg.top_data_dir, "Immune_All_High.pkl")

    quality_eval_cfg["evaluator_config"] = dict()
    quality_eval_cfg["evaluator_config"]["random_seed"] = 1
    quality_eval_cfg["n_hvgs"] = 1000

    # quality_eval_cfg["generator_config"] = dict()
    # quality_eval_cfg["generator_config"]["experiment_name"] = "quality_eval"
    # quality_eval_cfg["generator_config"]["name"] =

    return quality_eval_cfg



def get_next_quality_trial_num(cfg):
    if os.path.exists(cfg.experiment_tracking_file):
        experiment_tracking_df = pd.read_csv(cfg.experiment_tracking_file, header=0)
        if experiment_tracking_df["quality"].all():
            print("Synthetic data not yet available for additional experiments in this setting.")
            sys.exit(0)
        else:
            incomplete_experiments = experiment_tracking_df[experiment_tracking_df["quality"] == 0]
            trial_num = int(incomplete_experiments['trial'].iloc[0])
            return trial_num
    else:
        print("Experiment tracking file not yet created for this setting")
        sys.exit(0)


def register_quality_check(cfg):
    experiment_tracking_df = pd.read_csv(cfg.experiment_tracking_file, header=0)
    experiment_tracking_df.loc[experiment_tracking_df['trial'] == cfg.trial_num, "quality"] = 1
    experiment_tracking_df.to_csv(cfg.experiment_tracking_file, index=False)




# ====== Utility Metrics Reimplemented from Evaluation Code ======

# def wasserstein_distance(X_real, X_syn):
#     def wasserstein_distance(cfg, d1, d2, columns):
#         d1 = pd.DataFrame(binarize_discrete_features_evenly(cfg, d1, columns)[0])
#         d2 = pd.DataFrame(binarize_discrete_features_evenly(cfg, d2, columns)[0])
#         wd = 0
#         ratio = d1.shape[0] / d2.shape[0]
#         for col in d1.columns:
#             wd += abs(d1[col].sum() - d2[col].sum() * ratio) / d1.shape[0]
#         return wd
#
#
#
# # fit encoders to convert discrete data into one hot encoded form
# def fit_discrete_features_evenly(name, aux_data, meta, columns):
#     meta = pd.DataFrame(meta)
#     columns_encodings = {}
#     columns_domain = {}
#     for col in columns:
#         col_data = aux_data[col]
#         if is_numeric_dtype(col_data) and C.rap_bucket_numeric:
#
#             # if n_bins doesn't divide the values nicely, then this logic more evenly
#             # distributes the data to bins than the above logic
#             splits = np.array_split(sorted(col_data.values), C.n_bins)
#             basket_edges = [0]
#             for i in range(1, C.n_bins):
#                 # don't duplicate basket edges when basket is overfull
#                 basket_edges.append(splits[i][0] if splits[i][0] > basket_edges[i-1] else basket_edges[i-1]+1)
#             columns_encodings[col] = basket_edges
#             columns_domain[col] = C.n_bins
#         else:
#             categories = meta[meta["name"] == col].representation.values[0]
#             ohe = OneHotEncoder(categories=[categories]).fit(np.reshape(col_data.to_numpy(), (-1, 1)))
#             columns_encodings[col] = ohe
#             columns_domain[col] = len(categories)
#
#     dump_artifact(columns_encodings, f"{name}_thresholds_for_discrete_features_{C.n_bins}bins")
#     dump_artifact(columns_domain, f"{name}_ohe_domain_{C.n_bins}bins")
#
#
# # convert discrete data into one hot encoded form
# def binarize_discrete_features_evenly(cfg, data, columns):
#     columns_encodings = load_artifact(f"{cfg.data_name}_thresholds_for_discrete_features_{C.n_bins}bins")
#     columns_domain = load_artifact(f"{cfg.data_name}_ohe_domain_{C.n_bins}bins")
#
#     ohe_data = []
#     for col in columns:
#         col_data = data[col]
#         col_encoding = columns_encodings[col]
#         if is_numeric_dtype(col_data) and C.rap_bucket_numeric:
#             bins = np.digitize(col_data, col_encoding)
#             ohe_data.append(np.eye(C.n_bins)[bins - 1])
#         else:
#             ohe_data.append(col_encoding.transform(np.reshape(col_data.to_numpy(), (-1, 1))).toarray())
#
#     return np.hstack(ohe_data), columns_domain

#
# def mmd_rbf(X, Y, gamma=1.0):
#     """Simple RBF MMD for dense matrices."""
#     from sklearn.metrics.pairwise import rbf_kernel
#     XX = rbf_kernel(X, X, gamma=gamma)
#     YY = rbf_kernel(Y, Y, gamma=gamma)
#     XY = rbf_kernel(X, Y, gamma=gamma)
#     return XX.mean() + YY.mean() - 2 * XY.mean()
#
# def distance_to_closest_neighbor(X_real, X_syn):
#     """Average Euclidean distance from each synthetic cell to nearest real cell."""
#     from sklearn.metrics import pairwise_distances
#     D = pairwise_distances(X_syn, X_real)
#     return np.min(D, axis=1).mean()
#
# def discriminative_score(X_real, X_syn, seed=0):
#     """Real vs Synthetic classifier F1."""
#     X = np.vstack([X_real, X_syn])
#     y = np.array([1]*len(X_real) + [0]*len(X_syn))
#     X, y = shuffle(X, y, random_state=seed)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
#
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#
#     clf = LogisticRegression(max_iter=1000)
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     return f1_score(y_test, y_pred)
#
# def feature_overlap(feats_a, feats_b):
#     """Count overlapping top features (fake reimplementation)."""
#     a = set(feats_a)
#     b = set(feats_b)
#     overlap = len(a & b)
#     prop = overlap / max(len(a), 1)
#     return overlap, prop
#
# def top_features(model, top_n=10):
#     """Extract top features from a One-vs-Rest logistic model."""
#     feats = []
#     for est in model.estimators_:
#         coef = est.coef_[0]
#         idx = coef.argsort()[-top_n:]
#         feats.extend(idx.tolist())
#     return feats

# ========= SIMPLE EVALUATION PIPELINE =============

def evaluate(evaluator_cfg, real, syn):

    # Ensure dense arrays
    X_real = real.X.toarray() if hasattr(real.X, "toarray") else real.X
    X_syn = syn.X.toarray() if hasattr(syn.X, "toarray") else syn.X

    # ---------- Basic shape alignment ----------
    common_genes = real.var_names.intersection(syn.var_names)
    real = real[:, common_genes]
    syn = syn[:, common_genes]
    X_real = real.X.toarray()
    X_syn = syn.X.toarray()



    # ---------- UMAP eval ----------

    evaluator = SingleCellEvaluator(config=evaluator_cfg)
    evaluator.get_umap_evals(evaluator_cfg["n_hvgs"])


    # ---------- Train classifier on synthetic → test on real ----------

    evaluator = SingleCellEvaluator(config=evaluator_cfg)
    results = evaluator.get_classification_evals()

    output_file = os.path.join(evaluator.res_files_dir, f"classification_evals.csv")
    ##
    evaluator.save_results_to_csv(results, output_file)

    # ---------- Statistical metrics ----------

    evaluator = SingleCellEvaluator(config=evaluator_cfg)
    results = evaluator.get_statistical_evals()

    output_file = os.path.join(evaluator.res_files_dir, f"statistics_evals.csv")
    ##
    evaluator.save_results_to_csv(results, output_file)



    return None

    #
    # # ---------- Train classifier on synthetic → test on real ----------
    # print("Running classification metrics...")
    # y_real = real.obs.iloc[:, 0].astype(str).values
    # y_syn = syn.obs.iloc[:, 0].astype(str).values
    #
    # encoder = LabelEncoder()
    # encoder.fit(np.concatenate([y_real, y_syn]))
    #
    # y_real_enc = encoder.transform(y_real)
    # y_syn_enc = encoder.transform(y_syn)
    #
    # model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    # model.fit(X_syn, y_syn_enc)
    #
    # y_pred = model.predict(X_real)
    # y_proba = model.predict_proba(X_real)
    #
    # acc_syn = accuracy_score(y_real_enc, y_pred)
    # print("\n\ny_real_acc: ", y_real_enc)
    # print("\n\ny_proba: ", y_proba)
    # avgpr_syn = average_precision_score(y_real_enc, y_proba, average='macro')
    #
    # # ---------- Train on real → test on real (baseline) ----------
    # print("Running baseline metrics...")
    # model2 = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    # model2.fit(X_real, y_real_enc)
    # y_pred2 = model2.predict(X_real)
    # y_proba2 = model2.predict_proba(X_real)
    # acc_real = accuracy_score(y_real_enc, y_pred2)
    # avgpr_real = average_precision_score(y_real_enc, y_proba2, average='macro')
    #
    # feats_syn = top_features(model)
    # feats_real = top_features(model2)
    # overlap_cnt, overlap_prop = feature_overlap(feats_syn, feats_real)
    #
    # # ---------- Statistical metrics ----------
    # print("Running statistical metrics...")
    # mmd_score = mmd_rbf(X_real, X_syn)
    # dist_syn = distance_to_closest_neighbor(X_real, X_syn)
    # dist_base = distance_to_closest_neighbor(X_real, X_real)
    #
    # disc = discriminative_score(X_real, X_syn)
    #
    # print("Running wasserstein distance metric ...")
    # print("Not Yet Implemented.")
    #
    #
    # return {
    #     "accuracy_synthetic": acc_syn,
    #     "avgpr_synthetic": avgpr_syn,
    #     "accuracy_real": acc_real,
    #     "avgpr_real": avgpr_real,
    #     "feature_overlap_count": overlap_cnt,
    #     "feature_overlap_prop": overlap_prop,
    #     "mmd": mmd_score,
    #     "distance_to_closest": dist_syn,
    #     "distance_to_closest_base": dist_base,
    #     "discriminative_score": disc,
    # }


# ============================================================
# =============== COMMAND-LINE INTERFACE =====================
# ============================================================


if __name__ == "__main__":
    main()


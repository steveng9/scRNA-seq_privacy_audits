import time
from typing import Optional, Tuple, Dict, Any
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.decomposition import PCA

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)


if sys.argv[1] == 'T':
    from mia.utils.prepare_data import MIADataLoader
    from mia.models.base import BaseMIAModel
else:
    from src.mia.utils.prepare_data import MIADataLoader
    from src.mia.models.base import BaseMIAModel
from domias.bnaf.density_estimation import compute_log_p_x, density_estimator_trainer
from domias.baselines import (MC, LOGAN_D1,
                              GAN_leaks, GAN_leaks_cal, MC_optimized, GAN_leaks_optimized, GAN_leaks_cal_optimized)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

#Adapted from https://github.com/holarissun/DOMIAS/blob/main/src/domias/baselines.py


class DOMIASBaselineModels(BaseMIAModel):
    def __init__(self, 
                 config: Dict[str, Any], 
                 synthetic_file: str,
                 membership_test_file: str,
                 membership_lbl_file: str,
                 mia_experiment_name:str,
                 reference_file:str = None):
        super().__init__(config, 
                         synthetic_file, 
                         membership_test_file, 
                         membership_lbl_file,
                         mia_experiment_name,
                         reference_file)

    def run_attack(self):
        data_loader = MIADataLoader(
            synthetic_file=self.synthetic_file,
            membership_test_file=self.membership_test_file,
            membership_lbl_file = self.membership_lbl_file,
            membership_label_col=self.membership_label_col,
            generator_model=self.generator_model,
            reference_file=self.reference_file
        )
        synthetic_data = data_loader.load_synthetic_data().to_numpy()
        X_test = data_loader.load_membership_dataset().to_numpy()
        y_test = data_loader.load_membership_labels()
        if y_test is not None:
            assert len(X_test) == len(y_test), "mismatch in test data and label lengths."
        
        reference = data_loader.load_reference_data().to_numpy()

        scores = run_baselines(X_test, synthetic_data, reference, reference, None)

        return scores, y_test



def run_baselines(
    X_test: np.ndarray,
    #Y_test: np.ndarray,
    X_G: np.ndarray,
    X_ref: np.ndarray,
    X_ref_GLC: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[dict, dict]:
    score = {}
    runtimes = {}

    print("\n\nrunning MC baseline")
    start = time.process_time()
    score["MC"] = MC_optimized(X_test, X_G)
    runtime = time.process_time() - start
    print("took %.1f seconds" % runtime)
    runtimes["MC"] = runtime

    print("\n\nrunning gan_leaks baseline")
    start = time.process_time()
    score["gan_leaks"] = GAN_leaks_optimized(X_test, X_G)
    runtime = time.process_time() - start
    print("took %.1f seconds" % runtime)
    runtimes["gan_leaks"] = runtime

    if X_ref is not None:

        print("\n\nrunning logan baseline")
        start = time.process_time()
        score["LOGAN_D1"] = LOGAN_D1(X_test, X_G, X_ref)
        runtime = time.process_time() - start
        print("took %.1f seconds" % runtime)
        runtimes["LOGAN_D1"] = runtime

        print("\n\nrunning gan_leaks_cal baseline")
        start = time.process_time()
        score["gan_leaks_cal"] = GAN_leaks_cal_optimized(X_test, X_G, X_ref_GLC)
        runtime = time.process_time() - start
        print("took %.1f seconds" % runtime)
        runtimes["gan_leaks_cal"] = runtime

        ### apply PCA

        print("\n\nrunning pca")
        start = time.process_time()
        pca_ref = perform_pca(X_ref, n_components=150)
        pca_test = perform_pca(X_test, n_components=150)
        pca_synth = perform_pca(X_G, n_components=150)
        pca_runtime = time.process_time() - start
        print("took %.1f seconds" % pca_runtime)

        print("\n\nrunning domias baseline")
        start = time.process_time()
        score["domias_kde"] = kde_domias(pca_test, pca_synth, pca_ref)
        runtime = time.process_time() - start
        print("took %.1f seconds" % runtime)
        runtimes["domias_kde"] = runtime + pca_runtime

        #score["domias_bnaf"] = kde_domias(pca_test, pca_synth, pca_ref, "bnaf")

    return score, runtimes


def perform_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    #data = StandardScaler().fit_transform(data)
    principal_components = pca.fit_transform(data)
    # print("3:::", np.sum(pca.explained_variance_ratio_))

    return principal_components

def kde_domias(
            X_test: np.ndarray,
            synth_set: np.ndarray,
            reference_set: np.ndarray,
            density_estimator:str = "kde"):
  
    # BNAF was memory intensive and couldnt be tested 
    # BNAF for pG
    if density_estimator == "bnaf":
        _, p_G_model = density_estimator_trainer(
                synth_set, #values
                None,
                None,
            )
        _, p_R_model = density_estimator_trainer(reference_set)
        p_G_evaluated = np.exp(
                compute_log_p_x(p_G_model, torch.as_tensor(X_test).float().to(DEVICE))
                .cpu()
                .detach()
                .numpy()
            )
        p_R_evaluated = np.exp(
                compute_log_p_x(p_R_model, torch.as_tensor(X_test).float().to(DEVICE))
                .cpu()
                .detach()
                .numpy()
            )
            # KDE for pG
    elif density_estimator == "kde":
            density_gen = stats.gaussian_kde(synth_set.transpose(1, 0)) #values
            density_data = stats.gaussian_kde(reference_set.transpose(1, 0))
            p_G_evaluated = density_gen(X_test.transpose(1, 0))
            p_R_evaluated = density_data(X_test.transpose(1, 0))

    p_rel = p_G_evaluated / (p_R_evaluated + 1e-10)

    return p_rel



"""
scMAMA-MIA experiment orchestrator.

Usage:
    python src/run_experiment.py <config_path>            # normal
    python src/run_experiment.py <config_path> --print    # verbose
    python src/run_experiment.py T <config_path> P        # legacy server form

Pipeline:
    1. Load config, resolve paths, pick next trial number
    2. Sample train / holdout / aux donor sets
    3. Train target SDG on train set; generate synthetic data
    4. Fit shadow copulas on synthetic data (BB) and on aux data
    5. Run scMAMA-MIA attack for each cell type
    6. Aggregate to donor-level scores; compute and save AUC
    7. Register completed trial in tracking.csv
"""

import sys
import os
import time
import warnings

import numpy as np
import pandas as pd
import anndata as ad
import yaml
from box import Box
from sklearn.metrics import roc_auc_score
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch

# ---------------------------------------------------------------------------
# Path setup: make src/ importable regardless of CWD
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ---------------------------------------------------------------------------
# CLI arg parsing (backward-compatible with legacy "T/F <config> P" form)
# ---------------------------------------------------------------------------
_args = sys.argv[1:]
# Legacy "T/F <config> P" form — T = server, F = local
_env_flag = _args[0] if _args and _args[0] in ("T", "F") else None
if _env_flag is not None:
    _args = _args[1:]
_ENV = "server" if _env_flag == "T" else "local"
config_path = _args[0]
print_out = "--print" in sys.argv or (len(sys.argv) > 3 and sys.argv[3] == "P")

warnings.simplefilter(action="ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Local imports (after path setup)
# ---------------------------------------------------------------------------
from sdg.run import run_singlecell_generator          # noqa: E402
from sdg.scdesign2.copula import make_sdg_config       # noqa: E402
from attacks.scmamamia.attack import (                 # noqa: E402
    attack_pairwise_correlation,
    attack_mahalanobis,
    attack_mahalanobis_no_aux,
)
from attacks.scmamamia.scoring import (                # noqa: E402
    merge_cell_type_results,
    aggregate_scores_by_donor,
    _threat_model_code,
)
from data.splits import sample_donors_strategy_2, sample_donors_strategy_3   # noqa: E402
from data.cdf_utils import (                           # noqa: E402
    zinb_cdf, zinb_cdf_DT, zinb_uniform_transform,
    closeness_to_correlation_1, closeness_to_correlation_2,
    closeness_to_correlation_3, closeness_to_correlation_4,
    activate,
)

from numpy.linalg import inv, solve
from numpy.linalg import pinv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pinv_gpu(a):
    if DEVICE.type == "cpu":
        return pinv(a)
    a = a.astype(np.float32)
    t = torch.from_numpy(a).cuda()
    return torch.linalg.pinv(t).cpu().numpy()


# Registry of functions that can be referenced by name in YAML configs.
# Using an explicit registry instead of globals() keeps this clean.
FUNCTION_REGISTRY = {
    "zinb_cdf":                    zinb_cdf,
    "zinb_cdf_DT":                 zinb_cdf_DT,
    "zinb_uniform_transform":      zinb_uniform_transform,
    "pinv_gpu":                    pinv_gpu,
    "pinv":                        pinv,
    "inv":                         inv,
    "solve":                       solve,
    "closeness_to_correlation_1":  closeness_to_correlation_1,
    "closeness_to_correlation_2":  closeness_to_correlation_2,
    "closeness_to_correlation_3":  closeness_to_correlation_3,
    "closeness_to_correlation_4":  closeness_to_correlation_4,
    "sample_donors_strategy_2":    sample_donors_strategy_2,
    "sample_donors_strategy_3":    sample_donors_strategy_3,
}


# ===========================================================================
# Main pipeline
# ===========================================================================

def main():
    print(f"Using device: {DEVICE}")
    cfg = create_config(config_path)

    make_dir_structure(cfg)
    write_sdg_config_files(cfg)
    resampled, cell_types = create_data_splits(cfg)

    regenerated = generate_target_synthetic_data(cfg, cell_types, force=resampled)

    if not cfg.mia_setting.white_box:
        run_sdg(cfg, cfg.synth_model_config_path, cell_types,
                "SYNTHETIC_DATA_SHADOW_MODEL", force=regenerated)
    else:
        print("\n(white-box: skipping synthetic shadow model)", flush=True)

    run_sdg(cfg, cfg.aux_model_config_path, cell_types,
            "AUXILIARY_DATA_SHADOW_MODEL", force=resampled)

    results = run_mamamia_attack(cfg)
    save_results(cfg, results)
    register_trial(cfg)

    delete_interim_h5ad(cfg)


# ===========================================================================
# Configuration
# ===========================================================================

def create_config(path):
    with open(path) as f:
        cfg = Box(yaml.load(f, Loader=yaml.FullLoader))

    cfg.split_name = f"{cfg.mia_setting.num_donors}d"
    cfg.top_data_dir = os.path.join(cfg.dir_list[_ENV].data, cfg.dataset_name)
    cfg.cfg_dir = os.path.join(cfg.top_data_dir, cfg.split_name)
    cfg.experiment_tracking_file = os.path.join(cfg.cfg_dir, "tracking.csv")

    tracking_df, trial_num = _get_tracking(cfg)
    cfg.trial_num = trial_num
    cfg.trial_dir = os.path.join(cfg.cfg_dir, str(trial_num))

    cfg.datasets_path     = os.path.join(cfg.trial_dir, "datasets")
    cfg.results_path      = os.path.join(cfg.trial_dir, "results")
    cfg.figures_path      = os.path.join(cfg.results_path, "figures")
    cfg.models_path       = os.path.join(cfg.trial_dir, "models")
    cfg.artifacts_path    = os.path.join(cfg.trial_dir, "artifacts")
    cfg.synth_artifacts_path = os.path.join(cfg.artifacts_path, "synth")
    cfg.aux_artifacts_path   = os.path.join(cfg.artifacts_path, "aux")

    cfg.train_donors_path   = os.path.join(cfg.datasets_path, "train.npy")
    cfg.holdout_donors_path = os.path.join(cfg.datasets_path, "holdout.npy")
    cfg.aux_donors_path     = os.path.join(cfg.datasets_path, "auxiliary.npy")
    cfg.train_path          = os.path.join(cfg.datasets_path, "train.h5ad")
    cfg.holdout_path        = os.path.join(cfg.datasets_path, "holdout.h5ad")
    cfg.aux_path            = os.path.join(cfg.datasets_path, "auxiliary.h5ad")

    cfg.target_synthetic_data_path = os.path.join(cfg.datasets_path, "synthetic.h5ad")
    cfg.all_scores_file   = os.path.join(cfg.results_path, "mamamia_all_scores.csv")
    cfg.results_file      = os.path.join(cfg.results_path, "mamamia_results.csv")
    cfg.target_model_config_path = os.path.join(cfg.models_path, "config.yaml")
    cfg.synth_model_config_path  = os.path.join(cfg.synth_artifacts_path, "config.yaml")
    cfg.aux_model_config_path    = os.path.join(cfg.aux_artifacts_path, "config.yaml")
    cfg.permanent_hvg_mask_path  = os.path.join(cfg.top_data_dir, "hvg.csv")
    cfg.shadow_modelling_hvg_path = (
        cfg.permanent_hvg_mask_path if cfg.mia_setting.use_wb_hvgs
        else cfg.artifacts_path
    )

    # Mahalanobis is too CPU-intensive to parallelize well
    cfg.parallelize = cfg.parallelize and not cfg.mamamia_params.mahalanobis

    cfg.lin_alg_inverse_fn        = FUNCTION_REGISTRY[cfg.mamamia_params.lin_alg_inverse_fn]
    cfg.uniform_remapping_fn      = FUNCTION_REGISTRY[cfg.mamamia_params.uniform_remapping_fn]
    cfg.closeness_to_correlation_fn = FUNCTION_REGISTRY[cfg.mamamia_params.closeness_to_correlation_fn]
    cfg.sample_donors_strategy_fn = FUNCTION_REGISTRY[cfg.mia_setting.sample_donors_strategy_fn]

    if print_out:
        print("Experiment configuration:")
        for k, v in cfg.mia_setting.to_dict().items():
            print(f"  {k}: {v}")
        print("scMAMA-MIA parameters:")
        for k, v in cfg.mamamia_params.to_dict().items():
            print(f"  {k}: {v}")
        print(f"Trial: {trial_num}")

    return cfg


# ===========================================================================
# Trial tracking
# ===========================================================================

def _new_tracking_row(trial_num=1):
    return {
        "trial": trial_num,
        "tm:000": 0, "tm:001": 0, "tm:010": 0, "tm:011": 0,
        "tm:100": 0, "tm:101": 0, "tm:110": 0, "tm:111": 0,
        "baselines": 0, "quality": 0,
    }


def _get_tracking(cfg):
    if os.path.exists(cfg.experiment_tracking_file):
        df = pd.read_csv(cfg.experiment_tracking_file, header=0)
    else:
        df = pd.DataFrame(_new_tracking_row())
        os.makedirs(cfg.cfg_dir, exist_ok=True)
        df.to_csv(cfg.experiment_tracking_file, index=False)
    trial_num = _next_trial_num(cfg, df)
    return df, trial_num


def _next_trial_num(cfg, df):
    col = "tm:" + _threat_model_code(cfg)
    if df[col].all():
        trial_num = int(df["trial"].max()) + 1
        df.loc[trial_num] = _new_tracking_row(trial_num)
        df.to_csv(cfg.experiment_tracking_file, index=False)
    else:
        trial_num = int(df[df[col] == 0]["trial"].iloc[0])
    return trial_num


def register_trial(cfg):
    df, _ = _get_tracking(cfg)
    col = "tm:" + _threat_model_code(cfg)
    df.loc[df["trial"] == cfg.trial_num, col] = 1
    df.to_csv(cfg.experiment_tracking_file, index=False)


# ===========================================================================
# Directory and config-file setup
# ===========================================================================

def make_dir_structure(cfg):
    for path in [
        cfg.datasets_path, cfg.figures_path, cfg.models_path,
        cfg.synth_artifacts_path, cfg.aux_artifacts_path,
    ]:
        os.makedirs(path, exist_ok=True)


def write_sdg_config_files(cfg):
    """Write per-phase scDesign2 config YAMLs if they don't already exist."""
    if not os.path.exists(cfg.target_model_config_path):
        c = make_sdg_config(cfg, True, "models",
                            cfg.permanent_hvg_mask_path, "train.h5ad")
        with open(cfg.target_model_config_path, "w") as f:
            yaml.safe_dump(c.to_dict(), f, sort_keys=False)

    if not os.path.exists(cfg.synth_model_config_path):
        c = make_sdg_config(cfg, False, "artifacts/synth",
                            cfg.shadow_modelling_hvg_path, "synthetic.h5ad")
        with open(cfg.synth_model_config_path, "w") as f:
            yaml.safe_dump(c.to_dict(), f, sort_keys=False)

    if not os.path.exists(cfg.aux_model_config_path):
        c = make_sdg_config(cfg, False, "artifacts/aux",
                            cfg.shadow_modelling_hvg_path, "auxiliary.h5ad")
        with open(cfg.aux_model_config_path, "w") as f:
            yaml.safe_dump(c.to_dict(), f, sort_keys=False)


# ===========================================================================
# Data splitting
# ===========================================================================

def create_data_splits(cfg):
    all_data = ad.read_h5ad(os.path.join(cfg.top_data_dir, "full_dataset_cleaned.h5ad"))
    cell_types = all_data.obs["cell_type"].unique()

    if (os.path.exists(cfg.train_donors_path)
            and os.path.exists(cfg.holdout_donors_path)
            and os.path.exists(cfg.aux_donors_path)):
        resampled = False
        print("(donor split already exists — skipping)", flush=True)
    else:
        resampled = True
        _sample_and_save_donors(cfg, all_data)

    train_donors   = np.load(cfg.train_donors_path,   allow_pickle=True)
    holdout_donors = np.load(cfg.holdout_donors_path, allow_pickle=True)
    aux_donors     = np.load(cfg.aux_donors_path,     allow_pickle=True)

    all_train   = all_data[all_data.obs["individual"].isin(train_donors)]
    all_holdout = all_data[all_data.obs["individual"].isin(holdout_donors)]
    all_aux     = all_data[all_data.obs["individual"].isin(aux_donors)]
    print(f"Cells — train: {len(all_train)}, holdout: {len(all_holdout)}, aux: {len(all_aux)}")

    all_train.write_h5ad(cfg.train_path)
    all_holdout.write_h5ad(cfg.holdout_path)
    all_aux.write_h5ad(cfg.aux_path)

    targets = ad.concat([all_train, all_holdout])
    _initialise_results_files(cfg, targets)

    return resampled, cell_types


def _sample_and_save_donors(cfg, all_data):
    train_donors, holdout_donors, aux_donors = cfg.sample_donors_strategy_fn(
        cfg, all_data, list(all_data.obs["cell_type"].unique())
    )
    np.save(cfg.train_donors_path,   train_donors,   allow_pickle=True)
    np.save(cfg.holdout_donors_path, holdout_donors, allow_pickle=True)
    np.save(cfg.aux_donors_path,     aux_donors,     allow_pickle=True)


def _initialise_results_files(cfg, targets):
    if not os.path.exists(cfg.all_scores_file):
        cols = targets.obs.columns.tolist()
        train_donors   = np.load(cfg.train_donors_path,   allow_pickle=True)
        holdout_donors = np.load(cfg.holdout_donors_path, allow_pickle=True)
        membership = targets.obs["individual"].isin(train_donors).astype(int).values
        pd.DataFrame({
            "cell id":    targets.to_df().index,
            "donor id":   targets.obs["individual"].values,
            "cell type":  targets.obs["cell_type"].values,
            "sex":        targets.obs["sex"].values       if "sex"       in cols else None,
            "ethnicity":  targets.obs["ethnicity"].values if "ethnicity" in cols else None,
            "age":        targets.obs["age"].values       if "age"       in cols else None,
            "membership": membership,
        }).to_csv(cfg.all_scores_file, index=False)

    if not os.path.exists(cfg.results_file):
        pd.DataFrame({"metric": ["runtime", "auc"]}).to_csv(cfg.results_file, index=False)


def delete_interim_h5ad(cfg):
    for path in [cfg.train_path, cfg.holdout_path, cfg.aux_path]:
        if os.path.exists(path):
            os.remove(path)


# ===========================================================================
# SDG runner (generic wrapper)
# ===========================================================================

def generate_target_synthetic_data(cfg, cell_types, force=False):
    if not os.path.exists(cfg.target_synthetic_data_path) or force:
        run_sdg(cfg, cfg.target_model_config_path, cell_types,
                "TARGET_SYNTHETIC_DATA", force=True)
        return True
    print("(target synthetic data already exists — skipping)", flush=True)
    return False


def run_sdg(cfg, sdg_cfg_path, cell_types, label, force=False):
    """Run one phase of the SDG (train+generate or train-only) via run_singlecell_generator."""
    if not force:
        with open(sdg_cfg_path) as f:
            sdg_cfg = Box(yaml.load(f, Loader=yaml.FullLoader))
        copulas_dir = sdg_cfg.scdesign2_config.out_model_path
        if all(
            os.path.exists(os.path.join(cfg.trial_dir, copulas_dir, f"{ct}.rds"))
            for ct in cell_types
        ):
            print(f"(copulas for {label} already exist — skipping)", flush=True)
            return

    print(f"\n\nGENERATING {label}\n{'_'*40}", flush=True)
    t0_wall = time.time()
    t0_proc = time.process_time()

    run_singlecell_generator.callback(cfg_file=sdg_cfg_path)

    pd.DataFrame({
        "elapsed_wall_time":    [time.time() - t0_wall],
        "elapsed_process_time": [time.process_time() - t0_proc],
    }).to_csv(os.path.join(cfg.results_path, f"{label}_runtime.csv"), index=False)


# ===========================================================================
# Attack runner
# ===========================================================================

def run_mamamia_attack(cfg):
    """Run scMAMA-MIA over all cell types (parallel or sequential per config)."""
    print("\n\nRUNNING scMAMA-MIA\n" + "_" * 40, flush=True)
    train   = ad.read_h5ad(cfg.train_path)
    holdout = ad.read_h5ad(cfg.holdout_path)
    cell_types = list(train.obs["cell_type"].unique())

    results = []
    if cfg.parallelize:
        print("  parallelising across cell types...", flush=True)
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(_attack_cell_type, cfg, ct, train, holdout): ct
                for ct in cell_types
            }
            for fut in as_completed(futures):
                result = fut.result()
                print(f"  finished: {result[0]}", flush=True)
                results.append(result)
        results.sort(key=lambda x: x[0])
    else:
        for ct in cell_types:
            result = _attack_cell_type(cfg, ct, train, holdout)
            results.append(result)
            print(f"  finished: {ct}", flush=True)

    return results


def _resolve_attack_fn(cfg):
    """Return the appropriate attack function and paths based on the threat model."""
    copula_synth_dir = cfg.models_path if cfg.mia_setting.white_box else cfg.synth_artifacts_path

    if cfg.mia_setting.use_aux:
        attack_fn = (attack_mahalanobis if cfg.mamamia_params.mahalanobis
                     else attack_pairwise_correlation)
    else:
        if cfg.mamamia_params.mahalanobis:
            attack_fn = attack_mahalanobis_no_aux
        else:
            print("ERROR: non-aux, non-Mahalanobis attack not implemented.", flush=True)
            sys.exit(1)

    hvg_mask = pd.read_csv(cfg.shadow_modelling_hvg_path)
    hvgs = hvg_mask[hvg_mask["highly_variable"]].values[:, 0]

    return copula_synth_dir, attack_fn, hvgs


def _attack_cell_type(cfg, cell_type, train, holdout):
    from rpy2.robjects import r

    copula_synth_dir, attack_fn, hvgs = _resolve_attack_fn(cfg)
    synth_rds = os.path.join(copula_synth_dir, f"{cell_type}.rds")
    aux_rds   = os.path.join(cfg.aux_artifacts_path, f"{cell_type}.rds")

    if not (os.path.exists(synth_rds) and os.path.exists(aux_rds)):
        return (cell_type, None, None)

    copula_synth_r = r["readRDS"](synth_rds).rx2(str(cell_type))
    copula_aux_r   = r["readRDS"](aux_rds).rx2(str(cell_type))

    targets = _build_target_dataset(cell_type, hvgs, train, holdout)

    t0 = time.process_time()
    result_df = attack_fn(cfg, targets, cell_type,
                          copula_synth_r, copula_aux_r=copula_aux_r)
    runtime = time.process_time() - t0

    return (cell_type, result_df, runtime)


def _build_target_dataset(cell_type, hvgs, train_h5, holdout_h5):
    """Subset to one cell type, restrict to HVGs, attach membership labels."""
    train   = train_h5[train_h5.obs["cell_type"] == cell_type].to_df().loc[:, hvgs]
    holdout = holdout_h5[holdout_h5.obs["cell_type"] == cell_type].to_df().loc[:, hvgs]
    train["member"]   = True
    holdout["member"] = False
    train["individual"]   = train_h5.obs["individual"]
    holdout["individual"] = holdout_h5.obs["individual"]
    for col in ["ethnicity", "age", "sex"]:
        train[col]   = train_h5.obs[col]   if col in train_h5.obs.columns   else None
        holdout[col] = holdout_h5.obs[col] if col in holdout_h5.obs.columns else None
    return pd.concat([train, holdout])


# ===========================================================================
# Results saving
# ===========================================================================

def save_results(cfg, results):
    tm = _threat_model_code(cfg)
    new_df, total_runtime = merge_cell_type_results(cfg, results)

    prior_df = pd.read_csv(cfg.all_scores_file, dtype=str)
    prior_df["membership"] = (
        pd.to_numeric(prior_df["membership"], errors="coerce").fillna(0).astype(int)
    )
    new_df["membership"] = (
        pd.to_numeric(new_df["membership"], errors="coerce").fillna(0).astype(int)
    )
    merged = pd.merge(
        prior_df, new_df,
        on=["cell id", "donor id", "cell type", "membership"],
        suffixes=["", "_" + tm],
    )
    merged.to_csv(cfg.all_scores_file, index=False)

    true_labels, predictions = aggregate_scores_by_donor(cfg, new_df)
    overall_auc = roc_auc_score(true_labels, predictions)

    results_df = pd.read_csv(cfg.results_file)
    results_df["tm:" + tm] = [total_runtime, overall_auc]
    results_df.to_csv(cfg.results_file, index=False)

    print(f"Cells scored: {len(new_df)}")
    print(f"Runtime:      {total_runtime:.1f}s")
    print(f"AUC (tm:{tm}): {overall_auc:.4f}")


# ===========================================================================

if __name__ == "__main__":
    main()

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
from concurrent.futures import ProcessPoolExecutor, as_completed, BrokenExecutor
import multiprocessing
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
    attack_mahalanobis_both,
)
from attacks.scmamamia.attack_b import (               # noqa: E402
    attack_mahalanobis_b,
    attack_mahalanobis_b_no_aux,
    attack_mahalanobis_b_both,
    attack_mahalanobis_quad,
)
from attacks.scmamamia.scoring import (                # noqa: E402
    merge_cell_type_results,
    aggregate_scores_by_donor,
    _threat_model_code,
)
from data.splits import (                                                      # noqa: E402
    sample_donors_strategy_2,
    sample_donors_strategy_3,
    sample_donors_strategy_490,
)
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
    "sample_donors_strategy_490":  sample_donors_strategy_490,
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

    _run_aux_shadow_model_shared(cfg, cell_types, force=resampled)

    if cfg.mia_setting.get("run_quad_bb", False):
        results = run_mamamia_attack_quad(cfg)
        save_results_quad(cfg, results)
        register_trial_quad(cfg)
    elif cfg.mia_setting.get("run_both_bb", False):
        results = run_mamamia_attack_both(cfg)
        save_results_both(cfg, results)
        register_trial_both(cfg)
    else:
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
    # base_data_dir: the dataset root (e.g. scMAMAMIA/ok/), independent of SDG variant.
    # dataset_name may be 'ok' (old-style) or 'ok/scdesign2/no_dp' (new-style);
    # the first path component is always the dataset identifier.
    base_dataset = cfg.dataset_name.split('/')[0]
    cfg.base_data_dir = os.path.join(cfg.dir_list[_ENV].data, base_dataset)
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
    # Shared aux copulas across all SDG variants for the same (base_dataset, nd, trial).
    # The first SDG attack on a given (nd, trial) trains scDesign2 on the aux data and
    # saves here; subsequent SDG attacks reuse these without retraining.
    cfg.shared_aux_artifacts_path = os.path.join(
        cfg.base_data_dir, "aux_artifacts", cfg.split_name, str(trial_num)
    )

    # Donor splits live in a shared directory under the dataset root, not in each
    # SDG trial dir. This avoids duplicating the same tiny .npy files across every
    # SDG variant while keeping a single authoritative source of truth.
    cfg.splits_path       = os.path.join(cfg.base_data_dir, "splits",
                                          cfg.split_name, str(trial_num))
    cfg.train_donors_path   = os.path.join(cfg.splits_path, "train.npy")
    cfg.holdout_donors_path = os.path.join(cfg.splits_path, "holdout.npy")
    cfg.aux_donors_path     = os.path.join(cfg.splits_path, "auxiliary.npy")
    cfg.train_path          = os.path.join(cfg.datasets_path, "train.h5ad")
    cfg.holdout_path        = os.path.join(cfg.datasets_path, "holdout.h5ad")
    cfg.aux_path            = os.path.join(cfg.datasets_path, "auxiliary.h5ad")

    cfg.target_synthetic_data_path = os.path.join(cfg.datasets_path, "synthetic.h5ad")
    cfg.all_scores_file         = os.path.join(cfg.results_path, "mamamia_all_scores.csv")
    cfg.results_file            = os.path.join(cfg.results_path, "mamamia_results.csv")
    cfg.all_scores_file_classb  = os.path.join(cfg.results_path, "mamamia_all_scores_classb.csv")
    cfg.results_file_classb     = os.path.join(cfg.results_path, "mamamia_results_classb.csv")
    cfg.target_model_config_path = os.path.join(cfg.models_path, "config.yaml")
    cfg.synth_model_config_path  = os.path.join(cfg.synth_artifacts_path, "config.yaml")
    cfg.aux_model_config_path    = os.path.join(cfg.aux_artifacts_path, "config.yaml")
    cfg.permanent_hvg_mask_path  = (cfg.get("hvg_path")
                                     or os.path.join(cfg.base_data_dir, "hvg.csv"))
    cfg.shadow_modelling_hvg_path = (
        cfg.permanent_hvg_mask_path if cfg.mia_setting.use_wb_hvgs
        else cfg.artifacts_path
    )

    cfg.parallel_workers = cfg.get("parallel_workers", 4)

    # Generator selection (default scdesign2 for backward compat)
    cfg.generator_name = cfg.get("generator_name", "scdesign2")
    if cfg.generator_name == "scdesign3":
        sd3p = cfg.get("scdesign3_params", {})
        cfg.sd3_copula_type = sd3p.get("copula_type", "gaussian")
        cfg.sd3_family_use  = sd3p.get("family_use",  "nb")
        cfg.sd3_trunc_lvl   = str(sd3p.get("trunc_lvl", "Inf"))

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
        "classb:100": 0, "classb:101": 0,
        "classb:000": 0, "classb:001": 0,
        "baselines": 0, "quality": 0,
    }


def _get_tracking(cfg):
    if os.path.exists(cfg.experiment_tracking_file):
        df = pd.read_csv(cfg.experiment_tracking_file, header=0)
        # Migrate old tracking files that lack tm:* columns (e.g. quality-only runs)
        template = _new_tracking_row()
        changed = False
        for col, default in template.items():
            if col not in df.columns:
                df[col] = default
                changed = True
        if changed:
            df.to_csv(cfg.experiment_tracking_file, index=False)
    else:
        df = pd.DataFrame(_new_tracking_row(), index=[0])
        os.makedirs(cfg.cfg_dir, exist_ok=True)
        df.to_csv(cfg.experiment_tracking_file, index=False)
    trial_num = _next_trial_num(cfg, df)
    return df, trial_num


def _next_trial_num(cfg, df):
    # WB quad tracks via classb:000 so it doesn't collide with existing tm:000 entries
    # from earlier (non-Class-B) white-box runs.
    is_wb_quad = (cfg.mia_setting.get("run_quad_bb", False) and cfg.mia_setting.white_box)
    col = "classb:000" if is_wb_quad else ("tm:" + _threat_model_code(cfg))
    if col not in df.columns:
        df[col] = 0
        df.to_csv(cfg.experiment_tracking_file, index=False)
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
        cfg.splits_path, cfg.datasets_path, cfg.figures_path,
        cfg.models_path, cfg.synth_artifacts_path, cfg.aux_artifacts_path,
        cfg.shared_aux_artifacts_path,
    ]:
        os.makedirs(path, exist_ok=True)


def write_sdg_config_files(cfg):
    """Write per-phase SDG config YAMLs if they don't already exist."""
    if cfg.generator_name == "scdesign3":
        _write_sdg_config_files_sd3(cfg)
    else:
        _write_sdg_config_files_sd2(cfg)


def _write_sdg_config_files_sd2(cfg):
    """Write scDesign2 per-phase config YAMLs."""
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


def _write_sdg_config_files_sd3(cfg):
    """Write scDesign3 per-phase config YAMLs."""
    from sdg.scdesign3.copula import make_sdg_config_sd3
    if not os.path.exists(cfg.target_model_config_path):
        c = make_sdg_config_sd3(cfg, True, "models",
                                cfg.permanent_hvg_mask_path, "train.h5ad")
        with open(cfg.target_model_config_path, "w") as f:
            yaml.safe_dump(c.to_dict(), f, sort_keys=False)

    if not os.path.exists(cfg.synth_model_config_path):
        c = make_sdg_config_sd3(cfg, False, "artifacts/synth",
                                cfg.shadow_modelling_hvg_path, "synthetic.h5ad")
        with open(cfg.synth_model_config_path, "w") as f:
            yaml.safe_dump(c.to_dict(), f, sort_keys=False)

    if not os.path.exists(cfg.aux_model_config_path):
        c = make_sdg_config_sd3(cfg, False, "artifacts/aux",
                                cfg.shadow_modelling_hvg_path, "auxiliary.h5ad")
        with open(cfg.aux_model_config_path, "w") as f:
            yaml.safe_dump(c.to_dict(), f, sort_keys=False)


# ===========================================================================
# Data splitting
# ===========================================================================

def create_data_splits(cfg):
    # Use backed='r' so only obs metadata is loaded initially; avoids loading multi-GB
    # full datasets (e.g. AIDA 57 GB) entirely into RAM before filtering.
    full_path = os.path.join(cfg.base_data_dir, "full_dataset_cleaned.h5ad")
    all_data = ad.read_h5ad(full_path, backed='r')
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

    # Load only the needed subsets into memory, then release the full backed file.
    # Skip re-reading from disk if the h5ad splits already exist (saves several minutes
    # on large datasets like AIDA where each h5ad is ~8 GB).
    if (not resampled
            and os.path.exists(cfg.train_path)
            and os.path.exists(cfg.holdout_path)
            and os.path.exists(cfg.aux_path)):
        print("(h5ad splits already exist — skipping re-extraction)", flush=True)
        all_data.file.close()
        # Use backed='r' so only obs metadata is loaded — the X matrices are not
        # needed here and densifying them (especially at 200d) causes OOM.
        all_train   = ad.read_h5ad(cfg.train_path,   backed='r')
        all_holdout = ad.read_h5ad(cfg.holdout_path, backed='r')
        all_aux     = ad.read_h5ad(cfg.aux_path,     backed='r')
    else:
        all_train   = all_data[all_data.obs["individual"].isin(train_donors)].to_memory()
        all_holdout = all_data[all_data.obs["individual"].isin(holdout_donors)].to_memory()
        all_aux     = all_data[all_data.obs["individual"].isin(aux_donors)].to_memory()
        all_data.file.close()
        all_train.write_h5ad(cfg.train_path)
        all_holdout.write_h5ad(cfg.holdout_path)
        all_aux.write_h5ad(cfg.aux_path)

    print(f"Cells — train: {len(all_train)}, holdout: {len(all_holdout)}, aux: {len(all_aux)}", flush=True)
    # _initialise_results_files only needs obs metadata, not X.
    # Build a lightweight AnnData from obs to avoid densifying large matrices.
    targets = ad.AnnData(obs=pd.concat([all_train.obs, all_holdout.obs]))
    if hasattr(all_train, 'file'):
        all_train.file.close()
    if hasattr(all_holdout, 'file'):
        all_holdout.file.close()
    if hasattr(all_aux, 'file'):
        all_aux.file.close()
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
            "cell id":    targets.obs_names,  # avoids densifying the X matrix
            "donor id":   targets.obs["individual"].values,
            "cell type":  targets.obs["cell_type"].values,
            "sex":        targets.obs["sex"].values       if "sex"       in cols else None,
            "ethnicity":  targets.obs["ethnicity"].values if "ethnicity" in cols else None,
            "age":        targets.obs["age"].values       if "age"       in cols else None,
            "membership": membership,
        }).to_csv(cfg.all_scores_file, index=False)

    if not os.path.exists(cfg.results_file):
        pd.DataFrame({"metric": ["runtime", "auc"]}).to_csv(cfg.results_file, index=False)

    if cfg.mia_setting.get("run_quad_bb", False):
        if not os.path.exists(cfg.all_scores_file_classb):
            pd.DataFrame({
                "cell id":    targets.obs_names,
                "donor id":   targets.obs["individual"].values,
                "cell type":  targets.obs["cell_type"].values,
                "membership": targets.obs["individual"].isin(
                    np.load(cfg.train_donors_path, allow_pickle=True)
                ).astype(int).values,
            }).to_csv(cfg.all_scores_file_classb, index=False)
        if not os.path.exists(cfg.results_file_classb):
            pd.DataFrame({"metric": ["runtime", "auc"]}).to_csv(
                cfg.results_file_classb, index=False
            )


def delete_interim_h5ad(cfg):
    for path in [cfg.train_path, cfg.holdout_path, cfg.aux_path]:
        if os.path.exists(path):
            os.remove(path)


# ===========================================================================
# SDG runner (generic wrapper)
# ===========================================================================

def _run_aux_shadow_model_shared(cfg, cell_types, force=False):
    """
    Train the aux shadow model (scDesign2 on auxiliary.h5ad), but reuse shared
    copula artifacts when they already exist for this (base_dataset, nd, trial).

    The shared dir lives at:
        {base_data_dir}/aux_artifacts/{nd}d/{trial}/

    If another SDG variant already trained the aux copulas for this split, we copy
    them into the per-trial artifacts/aux/ dir and skip retraining. After training,
    we copy the new copulas to the shared dir for future reuse.
    """
    import shutil
    shared_dir   = cfg.shared_aux_artifacts_path
    per_trial_dir = cfg.aux_artifacts_path

    if not force:
        shared_ok = all(
            os.path.exists(os.path.join(shared_dir, f"{ct}.rds"))
            for ct in cell_types
        )
        if shared_ok:
            print("(aux copulas: reusing shared artifacts — skipping scDesign2 fit)", flush=True)
            for ct in cell_types:
                dst = os.path.join(per_trial_dir, f"{ct}.rds")
                if not os.path.exists(dst):
                    shutil.copy2(os.path.join(shared_dir, f"{ct}.rds"), dst)
            return

    # Train normally (run_sdg has its own skip check if per-trial artifacts already exist)
    run_sdg(cfg, cfg.aux_model_config_path, cell_types,
            "AUXILIARY_DATA_SHADOW_MODEL", force=force)

    # Propagate to shared dir so subsequent SDG attacks on this split skip training
    for ct in cell_types:
        src = os.path.join(per_trial_dir, f"{ct}.rds")
        dst = os.path.join(shared_dir, f"{ct}.rds")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)


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
        gen_name = sdg_cfg.get("generator_name", "scdesign2")
        copulas_dir = sdg_cfg[f"{gen_name}_config"].out_model_path
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

    # Use backed='r' just to read cell type list without loading the full X matrix.
    _train_meta = ad.read_h5ad(cfg.train_path, backed="r")
    cell_types = list(_train_meta.obs["cell_type"].unique())
    _train_meta.file.close()

    results = []
    if cfg.parallelize:
        print(f"  parallelising across cell types (max_workers={cfg.parallel_workers})...",
              flush=True)
        results = _run_parallel_attack(cfg, cell_types, cfg.parallel_workers)
    else:
        train   = ad.read_h5ad(cfg.train_path)
        holdout = ad.read_h5ad(cfg.holdout_path)
        for ct in cell_types:
            result = _attack_cell_type(cfg, ct, train, holdout)
            results.append(result)
            print(f"  finished: {ct}", flush=True)

    results.sort(key=lambda x: x[0])
    return results


def _run_parallel_attack(cfg, cell_types, max_workers):
    """
    Run cell-type attacks in parallel batches with automatic OOM recovery.

    When the OOM killer sends SIGKILL to a worker the entire ProcessPoolExecutor
    pool breaks (BrokenExecutor).  Recovery: identify incomplete cell types,
    halve the worker count, and relaunch just those cell types.  Repeats until
    max_workers==1; at that point attempts sequential fallback.
    """
    results = []
    pending = list(cell_types)
    current_workers = max_workers

    while pending:
        batch = pending[:current_workers]
        batch_results, failed_cts = _try_parallel_batch(cfg, batch, current_workers)
        results.extend(batch_results)

        if failed_cts:
            if current_workers > 1:
                current_workers = max(1, current_workers // 2)
                print(f"  [OOM] Reducing to {current_workers} workers; "
                      f"retrying {len(failed_cts)} cell types", flush=True)
                pending = failed_cts + pending[len(batch):]
            else:
                # Already at 1 worker — last-resort sequential retry
                print(f"  [FALLBACK] Sequential retry for: {failed_cts}", flush=True)
                train   = ad.read_h5ad(cfg.train_path)
                holdout = ad.read_h5ad(cfg.holdout_path)
                for ct in failed_cts:
                    try:
                        result = _attack_cell_type(cfg, ct, train, holdout)
                        results.append(result)
                        print(f"  finished (sequential): {result[0]}", flush=True)
                    except Exception as e:
                        print(f"  [ERROR] {ct} failed sequentially: {e}", flush=True)
                        results.append((ct, None, None))
                pending = pending[len(batch):]
        else:
            pending = pending[len(batch):]

    return results


def _try_parallel_batch(cfg, batch, max_workers):
    """
    Attempt to run one batch of cell types in parallel.

    Uses 'spawn' start method to avoid rpy2/R fork-safety issues on Linux.
    Returns (completed_results, failed_cell_types).
    """
    completed = []
    failed = []
    completed_cts = set()
    ctx = multiprocessing.get_context("spawn")

    try:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(_attack_cell_type_from_disk, cfg, ct): ct
                for ct in batch
            }
            for fut in as_completed(futures):
                ct = futures[fut]
                try:
                    result = fut.result()
                    completed.append(result)
                    completed_cts.add(ct)
                    print(f"  finished: {result[0]}", flush=True)
                except Exception as e:
                    print(f"  failed: {ct} ({type(e).__name__}: {e})", flush=True)
                    failed.append(ct)
                    completed_cts.add(ct)
    except Exception as e:
        # Pool itself broke (typically OOM kill of a worker process)
        print(f"  [WARN] Worker pool broke: {type(e).__name__}", flush=True)
        failed.extend([ct for ct in batch if ct not in completed_cts])

    return completed, failed


def _attack_cell_type_from_disk(cfg, cell_type):
    """
    Subprocess entry point for parallel execution.

    Loads only the slice of train/holdout data needed for this cell type using
    backed='r', so each worker holds only one cell type in RAM rather than the
    full multi-GB AnnData.
    """
    import anndata as _ad
    train_backed   = _ad.read_h5ad(cfg.train_path,   backed="r")
    holdout_backed = _ad.read_h5ad(cfg.holdout_path, backed="r")
    train_ct   = train_backed  [train_backed  .obs["cell_type"] == cell_type].to_memory()
    holdout_ct = holdout_backed[holdout_backed.obs["cell_type"] == cell_type].to_memory()
    train_backed.file.close()
    holdout_backed.file.close()
    return _attack_cell_type(cfg, cell_type, train_ct, holdout_ct)


# ===========================================================================
# Combined BB+aux / BB-aux attack (shares synth-side computation)
# ===========================================================================

def run_mamamia_attack_both(cfg):
    """
    Run BB+aux and BB-aux in a single pass per cell type, reusing d_s.
    Returns list of (cell_type, (df_100, df_101), runtime).
    """
    print("\n\nRUNNING scMAMA-MIA (BB+aux AND BB-aux combined)\n" + "_" * 40, flush=True)

    _train_meta = ad.read_h5ad(cfg.train_path, backed="r")
    cell_types = list(_train_meta.obs["cell_type"].unique())
    _train_meta.file.close()

    if cfg.parallelize:
        print(f"  parallelising across cell types (max_workers={cfg.parallel_workers})...",
              flush=True)
        results = _run_parallel_attack_both(cfg, cell_types, cfg.parallel_workers)
    else:
        train   = ad.read_h5ad(cfg.train_path)
        holdout = ad.read_h5ad(cfg.holdout_path)
        results = []
        for ct in cell_types:
            result = _attack_cell_type_both(cfg, ct, train, holdout)
            results.append(result)
            print(f"  finished: {ct}", flush=True)

    results.sort(key=lambda x: x[0])
    return results


def _run_parallel_attack_both(cfg, cell_types, max_workers):
    results = []
    pending = list(cell_types)
    current_workers = max_workers

    while pending:
        batch = pending[:current_workers]
        batch_results, failed_cts = _try_parallel_batch_both(cfg, batch, current_workers)
        results.extend(batch_results)

        if failed_cts:
            if current_workers > 1:
                current_workers = max(1, current_workers // 2)
                print(f"  [OOM] Reducing to {current_workers} workers; "
                      f"retrying {len(failed_cts)} cell types", flush=True)
                pending = failed_cts + pending[len(batch):]
            else:
                print(f"  [FALLBACK] Sequential retry for: {failed_cts}", flush=True)
                train   = ad.read_h5ad(cfg.train_path)
                holdout = ad.read_h5ad(cfg.holdout_path)
                for ct in failed_cts:
                    try:
                        result = _attack_cell_type_both(cfg, ct, train, holdout)
                        results.append(result)
                        print(f"  finished (sequential): {result[0]}", flush=True)
                    except Exception as e:
                        print(f"  [ERROR] {ct} failed sequentially: {e}", flush=True)
                        results.append((ct, None, None))
                pending = pending[len(batch):]
        else:
            pending = pending[len(batch):]

    return results


def _try_parallel_batch_both(cfg, batch, max_workers):
    completed = []
    failed = []
    completed_cts = set()
    ctx = multiprocessing.get_context("spawn")

    try:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(_attack_cell_type_from_disk_both, cfg, ct): ct
                for ct in batch
            }
            for fut in as_completed(futures):
                ct = futures[fut]
                try:
                    result = fut.result()
                    completed.append(result)
                    completed_cts.add(ct)
                    print(f"  finished (both): {result[0]}", flush=True)
                except Exception as e:
                    print(f"  failed: {ct} ({type(e).__name__}: {e})", flush=True)
                    failed.append(ct)
                    completed_cts.add(ct)
    except Exception as e:
        print(f"  [WARN] Worker pool broke: {type(e).__name__}", flush=True)
        failed.extend([ct for ct in batch if ct not in completed_cts])

    return completed, failed


def _attack_cell_type_from_disk_both(cfg, cell_type):
    import anndata as _ad
    train_backed   = _ad.read_h5ad(cfg.train_path,   backed="r")
    holdout_backed = _ad.read_h5ad(cfg.holdout_path, backed="r")
    train_ct   = train_backed  [train_backed  .obs["cell_type"] == cell_type].to_memory()
    holdout_ct = holdout_backed[holdout_backed.obs["cell_type"] == cell_type].to_memory()
    train_backed.file.close()
    holdout_backed.file.close()
    return _attack_cell_type_both(cfg, cell_type, train_ct, holdout_ct)


def _attack_cell_type_both(cfg, cell_type, train, holdout):
    """
    Run combined BB+aux / BB-aux attack for one cell type.
    Returns (cell_type, (df_100, df_101), runtime)  or  (cell_type, None, None).
    """
    hvg_mask = pd.read_csv(cfg.shadow_modelling_hvg_path)
    hvgs = hvg_mask[hvg_mask["highly_variable"]].values[:, 0]

    synth_rds = os.path.join(cfg.synth_artifacts_path, f"{cell_type}.rds")
    aux_rds   = os.path.join(cfg.aux_artifacts_path,   f"{cell_type}.rds")

    if not (os.path.exists(synth_rds) and os.path.exists(aux_rds)):
        return (cell_type, None, None)

    from rpy2.robjects import r
    copula_synth_r = r["readRDS"](synth_rds).rx2(str(cell_type))
    copula_aux_r   = r["readRDS"](aux_rds).rx2(str(cell_type))

    targets = _build_target_dataset(cell_type, hvgs, train, holdout)

    _v = cfg.mamamia_params.get("class_b_gamma", 0)
    use_class_b = (_v == "auto") or (_v not in (0, 0.0))
    both_fn = attack_mahalanobis_b_both if use_class_b else attack_mahalanobis_both

    t0 = time.process_time()
    df_100, df_101 = both_fn(cfg, targets, cell_type, copula_synth_r, copula_aux_r)
    runtime = time.process_time() - t0

    return (cell_type, (df_100, df_101), runtime)


def save_results_both(cfg, results):
    """
    Save BB+aux (tm:100) and BB-aux (tm:101) from a combined attack run.

    results : list of (cell_type, (df_100, df_101) or None, runtime)

    Calls save_results twice by temporarily setting cfg.mia_setting.use_aux.
    Full combined runtime is attributed to tm:100; tm:101 reports 0 (no extra cost).
    """
    results_100 = [(ct, pair[0] if pair is not None else None, rt)
                   for ct, pair, rt in results]
    results_101 = [(ct, pair[1] if pair is not None else None, 0.0)
                   for ct, pair, rt in results]

    cfg.mia_setting.use_aux = True
    save_results(cfg, results_100)

    cfg.mia_setting.use_aux = False
    save_results(cfg, results_101)

    cfg.mia_setting.use_aux = True   # leave in a defined state


def register_trial_both(cfg):
    """Mark both tm:100 and tm:101 as complete for the current trial."""
    df, _ = _get_tracking(cfg)
    df.loc[df["trial"] == cfg.trial_num, "tm:100"] = 1
    df.loc[df["trial"] == cfg.trial_num, "tm:101"] = 1
    df.to_csv(cfg.experiment_tracking_file, index=False)


# ===========================================================================
# Quad BB attack: standard + Class B BB+aux / BB-aux in one pass
# Stores standard results in mamamia_results.csv (legacy-compatible).
# Stores Class B results in mamamia_results_classb.csv (separate file).
# ===========================================================================

def run_mamamia_attack_quad(cfg):
    """Run all 4 variants (standard+ClassB × aux/no-aux) in one pass per cell type."""
    print("\n\nRUNNING scMAMA-MIA QUAD (standard + Class B, BB+/-aux)\n" + "_" * 40, flush=True)

    _train_meta = ad.read_h5ad(cfg.train_path, backed="r")
    cell_types = list(_train_meta.obs["cell_type"].unique())
    _train_meta.file.close()

    if cfg.parallelize:
        print(f"  parallelising across cell types (max_workers={cfg.parallel_workers})...",
              flush=True)
        results = _run_parallel_attack_quad(cfg, cell_types, cfg.parallel_workers)
    else:
        train   = ad.read_h5ad(cfg.train_path)
        holdout = ad.read_h5ad(cfg.holdout_path)
        results = []
        for ct in cell_types:
            result = _attack_cell_type_quad(cfg, ct, train, holdout)
            results.append(result)
            print(f"  finished: {ct}", flush=True)

    results.sort(key=lambda x: x[0])
    return results


def _run_parallel_attack_quad(cfg, cell_types, max_workers):
    results = []
    pending = list(cell_types)
    current_workers = max_workers

    while pending:
        batch = pending[:current_workers]
        batch_results, failed_cts = _try_parallel_batch_quad(cfg, batch, current_workers)
        results.extend(batch_results)

        if failed_cts:
            if current_workers > 1:
                current_workers = max(1, current_workers // 2)
                print(f"  [OOM] Reducing to {current_workers} workers; "
                      f"retrying {len(failed_cts)} cell types", flush=True)
                pending = failed_cts + pending[len(batch):]
            else:
                print(f"  [FALLBACK] Sequential retry for: {failed_cts}", flush=True)
                train   = ad.read_h5ad(cfg.train_path)
                holdout = ad.read_h5ad(cfg.holdout_path)
                for ct in failed_cts:
                    try:
                        result = _attack_cell_type_quad(cfg, ct, train, holdout)
                        results.append(result)
                        print(f"  finished (sequential): {result[0]}", flush=True)
                    except Exception as e:
                        print(f"  [ERROR] {ct} failed sequentially: {e}", flush=True)
                        results.append((ct, None, None))
                pending = pending[len(batch):]
        else:
            pending = pending[len(batch):]

    return results


def _try_parallel_batch_quad(cfg, batch, max_workers):
    completed = []
    failed = []
    completed_cts = set()
    ctx = multiprocessing.get_context("spawn")

    try:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(_attack_cell_type_from_disk_quad, cfg, ct): ct
                for ct in batch
            }
            for fut in as_completed(futures):
                ct = futures[fut]
                try:
                    result = fut.result()
                    completed.append(result)
                    completed_cts.add(ct)
                    print(f"  finished (quad): {result[0]}", flush=True)
                except Exception as e:
                    print(f"  failed: {ct} ({type(e).__name__}: {e})", flush=True)
                    failed.append(ct)
                    completed_cts.add(ct)
    except Exception as e:
        print(f"  [WARN] Worker pool broke: {type(e).__name__}", flush=True)
        failed.extend([ct for ct in batch if ct not in completed_cts])

    return completed, failed


def _attack_cell_type_from_disk_quad(cfg, cell_type):
    import anndata as _ad
    train_backed   = _ad.read_h5ad(cfg.train_path,   backed="r")
    holdout_backed = _ad.read_h5ad(cfg.holdout_path, backed="r")
    train_ct   = train_backed  [train_backed  .obs["cell_type"] == cell_type].to_memory()
    holdout_ct = holdout_backed[holdout_backed.obs["cell_type"] == cell_type].to_memory()
    train_backed.file.close()
    holdout_backed.file.close()
    return _attack_cell_type_quad(cfg, cell_type, train_ct, holdout_ct)


def _attack_cell_type_quad(cfg, cell_type, train, holdout):
    """
    Run all 4 variants for one cell type (standard + Class B, +aux / -aux).
    Supports both BB (synth_artifacts_path) and WB (models_path) copula sources.
    Returns (cell_type, (r_std_aux, r_std_noaux, r_b_aux, r_b_noaux), runtime)
    or (cell_type, None, None).
    """
    hvg_mask = pd.read_csv(cfg.shadow_modelling_hvg_path)
    hvgs = hvg_mask[hvg_mask["highly_variable"]].values[:, 0]

    synth_dir = cfg.models_path if cfg.mia_setting.white_box else cfg.synth_artifacts_path
    synth_rds = os.path.join(synth_dir, f"{cell_type}.rds")
    aux_rds   = os.path.join(cfg.aux_artifacts_path, f"{cell_type}.rds")

    if not (os.path.exists(synth_rds) and os.path.exists(aux_rds)):
        return (cell_type, None, None)

    from rpy2.robjects import r
    copula_synth_r = r["readRDS"](synth_rds).rx2(str(cell_type))
    copula_aux_r   = r["readRDS"](aux_rds).rx2(str(cell_type))

    targets = _build_target_dataset(cell_type, hvgs, train, holdout)

    wb = cfg.mia_setting.white_box
    tm_aux, tm_noaux = ("000", "001") if wb else ("100", "101")

    t0 = time.process_time()
    quad = attack_mahalanobis_quad(cfg, targets, cell_type, copula_synth_r, copula_aux_r,
                                    tm_aux=tm_aux, tm_noaux=tm_noaux)
    runtime = time.process_time() - t0

    return (cell_type, quad, runtime)


def save_results_quad(cfg, results):
    """
    Save all 4 variants from a quad attack run.

    Standard results (r100_std, r101_std) → mamamia_results.csv  (legacy-compatible)
    Class B results  (r100_b,   r101_b)   → mamamia_results_classb.csv
    """
    print("\n--- Saving standard results ---", flush=True)
    results_std_100 = [(ct, pair[0] if pair is not None else None, rt)
                       for ct, pair, rt in results]
    results_std_101 = [(ct, pair[1] if pair is not None else None, 0.0)
                       for ct, pair, rt in results]
    cfg.mia_setting.use_aux = True
    save_results(cfg, results_std_100)
    cfg.mia_setting.use_aux = False
    save_results(cfg, results_std_101)
    cfg.mia_setting.use_aux = True

    print("\n--- Saving Class B results ---", flush=True)
    results_b_100 = [(ct, pair[2] if pair is not None else None, rt)
                     for ct, pair, rt in results]
    results_b_101 = [(ct, pair[3] if pair is not None else None, 0.0)
                     for ct, pair, rt in results]

    orig_results_file  = cfg.results_file
    orig_scores_file   = cfg.all_scores_file
    cfg.results_file   = cfg.results_file_classb
    cfg.all_scores_file = cfg.all_scores_file_classb

    cfg.mia_setting.use_aux = True
    save_results(cfg, results_b_100)
    cfg.mia_setting.use_aux = False
    save_results(cfg, results_b_101)
    cfg.mia_setting.use_aux = True

    cfg.results_file   = orig_results_file
    cfg.all_scores_file = orig_scores_file


def register_trial_quad(cfg):
    """Mark standard and Class B results as complete for the current trial."""
    df, _ = _get_tracking(cfg)
    wb = cfg.mia_setting.white_box
    cols = (["tm:000", "tm:001", "classb:000", "classb:001"] if wb
            else ["tm:100", "tm:101", "classb:100", "classb:101"])
    for col in cols:
        if col not in df.columns:
            df[col] = 0
        df.loc[df["trial"] == cfg.trial_num, col] = 1
    df.to_csv(cfg.experiment_tracking_file, index=False)


# ===========================================================================
# Single-threat-model attack helpers (original path)
# ===========================================================================

def _resolve_attack_fn(cfg):
    """Return the appropriate attack function and paths based on the threat model."""
    copula_synth_dir = cfg.models_path if cfg.mia_setting.white_box else cfg.synth_artifacts_path
    def _class_b_enabled(key):
        v = cfg.mamamia_params.get(key, 0)
        return v == "auto" or (v != 0 and v != 0.0)

    if cfg.mia_setting.use_aux:
        if _class_b_enabled("class_b_gamma"):
            attack_fn = attack_mahalanobis_b
        else:
            attack_fn = (attack_mahalanobis if cfg.mamamia_params.mahalanobis
                         else attack_pairwise_correlation)
    else:
        if cfg.mamamia_params.mahalanobis:
            attack_fn = (attack_mahalanobis_b_no_aux
                         if _class_b_enabled("class_b_gamma_noaux")
                         else attack_mahalanobis_no_aux)
        else:
            print("ERROR: non-aux, non-Mahalanobis attack not implemented.", flush=True)
            sys.exit(1)

    hvg_mask = pd.read_csv(cfg.shadow_modelling_hvg_path)
    hvgs = hvg_mask[hvg_mask["highly_variable"]].values[:, 0]

    return copula_synth_dir, attack_fn, hvgs


def _attack_cell_type(cfg, cell_type, train, holdout):
    copula_synth_dir, attack_fn, hvgs = _resolve_attack_fn(cfg)
    synth_rds = os.path.join(copula_synth_dir, f"{cell_type}.rds")
    aux_rds   = os.path.join(cfg.aux_artifacts_path, f"{cell_type}.rds")

    if not (os.path.exists(synth_rds) and os.path.exists(aux_rds)):
        return (cell_type, None, None)

    if getattr(cfg, "generator_name", "scdesign2") == "scdesign3":
        from sdg.scdesign3.copula import load_copula_sd3
        copula_synth_r = load_copula_sd3(synth_rds, cell_type)
        copula_aux_r   = load_copula_sd3(aux_rds,   cell_type)
    else:
        from rpy2.robjects import r
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

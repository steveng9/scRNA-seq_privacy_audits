#!/usr/bin/env python3
"""
Aux-disjointness fix — STEP 2: re-run one (dataset, nd, trial) scDesign2 attack
with a provably-disjoint auxiliary set, REUSING the on-disk synthetic data.

Only D_aux changes; the training donors (hence the synthetic data and its BB
shadow copula, and the WB train copula) are identical to the paper's run. So we
copy those artifacts from the original no_dp trial dir and refit ONLY the aux
shadow copula on the new disjoint aux — then run the quad attack twice:
    white_box=False -> BB pair  (tm:100 BB+aux, tm:101 BB-aux)
    white_box=True  -> WB pair  (tm:000 WB+aux, tm:001 WB-aux)
each with standard + Class-B scoring. The +aux columns (000/100 & their classb)
are the ones affected by the old contamination; the -aux columns are unaffected
and serve as a consistency check against the paper.

Prereq: experiments/aux_disjoint/build_disjoint_aux_split.py has written
  {ds}/aux_disjoint/splits/{nd}d/{trial}/{train,holdout,auxiliary}.npy
Results land in {ds}/aux_disjoint/scdesign2/{nd}d/{trial}/results/ as
  {bb,wb}_results.csv and {bb,wb}_results_classb.csv (+ a merged auc row).
"""
import os, sys, argparse, time, shutil
import numpy as np
import pandas as pd
import anndata as ad

REPO = "/home/golobs/scRNA-seq_privacy_audits"
DATA = "/home/golobs/data/scMAMAMIA"
sys.path.insert(0, os.path.join(REPO, "src"))

_saved_argv = sys.argv
sys.argv = ["run_experiment.py", "/dev/null"]
import run_experiment as R   # noqa: E402
sys.argv = _saved_argv
import yaml  # noqa: E402

MAMAMIA_PARAMS = {
    "IMPORTANCE_OF_CLASS_B_FPs": 0.17,
    "epsilon": 0.0001,
    "mahalanobis": True,
    "uniform_remapping_fn": "zinb_cdf",
    "lin_alg_inverse_fn": "pinv_gpu",
    "closeness_to_correlation_fn": "closeness_to_correlation_1",
    "class_b_gene_set": "secondary",
    "class_b_scoring": "llr",
    "class_b_gamma": "auto",
    "class_b_gamma_noaux": "auto",
}


def _subset_h5ad(full_backed, donor_ids, out_path):
    mask = full_backed.obs["individual"].astype(str).isin(set(np.asarray(donor_ids).astype(str).tolist()))
    sub = full_backed[mask].to_memory()
    sub.write_h5ad(out_path)
    return sub


def _make_cfg(dataset, nd, trial, workers, white_box):
    cfg_dir = os.path.join(REPO, "experiments", "aux_disjoint", "_cfgs", dataset)
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, f"{nd}d_t{trial}_{'wb' if white_box else 'bb'}.yaml")
    cfg_dict = {
        "dir_list": {"local": {"home": REPO, "data": DATA},
                     "server": {"home": REPO, "data": DATA}},
        "dataset_name": f"{dataset}/aux_disjoint/scdesign2",
        "hvg_path": os.path.join(DATA, dataset, "hvg.csv"),
        "generator_name": "scdesign2",
        "plot_results": False,
        "parallelize": True,
        "parallel_workers": workers,
        "min_aux_donors": 10,
        "mamamia_params": dict(MAMAMIA_PARAMS),
        "mia_setting": {
            "sample_donors_strategy_fn": "sample_donors_strategy_2",
            "num_donors": nd, "white_box": white_box,
            "use_wb_hvgs": True, "use_aux": True, "run_quad_bb": True,
        },
    }
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)
    cfg = R.create_config(cfg_path)
    cfg.trial_num = int(trial)
    cfg.trial_dir = os.path.join(cfg.cfg_dir, str(trial))
    cfg.datasets_path = os.path.join(cfg.trial_dir, "datasets")
    cfg.results_path  = os.path.join(cfg.trial_dir, "results")
    cfg.figures_path  = os.path.join(cfg.results_path, "figures")
    cfg.models_path   = os.path.join(cfg.trial_dir, "models")
    cfg.artifacts_path = os.path.join(cfg.trial_dir, "artifacts")
    cfg.synth_artifacts_path = os.path.join(cfg.artifacts_path, "synth")
    cfg.aux_artifacts_path   = os.path.join(cfg.artifacts_path, "aux")
    # New disjoint-aux split (built by build_disjoint_aux_split.py)
    cfg.splits_path = os.path.join(cfg.base_data_dir, "aux_disjoint", "splits", cfg.split_name, str(trial))
    cfg.train_donors_path   = os.path.join(cfg.splits_path, "train.npy")
    cfg.holdout_donors_path = os.path.join(cfg.splits_path, "holdout.npy")
    cfg.aux_donors_path     = os.path.join(cfg.splits_path, "auxiliary.npy")
    cfg.train_path   = os.path.join(cfg.datasets_path, "train.h5ad")
    cfg.holdout_path = os.path.join(cfg.datasets_path, "holdout.h5ad")
    cfg.aux_path     = os.path.join(cfg.datasets_path, "auxiliary.h5ad")
    cfg.target_synthetic_data_path = os.path.join(cfg.datasets_path, "synthetic.h5ad")
    cfg.all_scores_file        = os.path.join(cfg.results_path, "mamamia_all_scores.csv")
    cfg.results_file           = os.path.join(cfg.results_path, "mamamia_results.csv")
    cfg.all_scores_file_classb = os.path.join(cfg.results_path, "mamamia_all_scores_classb.csv")
    cfg.results_file_classb    = os.path.join(cfg.results_path, "mamamia_results_classb.csv")
    cfg.target_model_config_path = os.path.join(cfg.models_path, "config.yaml")
    cfg.synth_model_config_path  = os.path.join(cfg.synth_artifacts_path, "config.yaml")
    cfg.aux_model_config_path    = os.path.join(cfg.aux_artifacts_path, "config.yaml")
    # FRESH per-experiment aux cache so the aux copula refits on the NEW aux
    # (never the contaminated shared cache).
    cfg.shared_aux_artifacts_path = os.path.join(cfg.base_data_dir, "aux_disjoint",
                                                 "aux_artifacts", cfg.split_name, str(trial))
    for p in [cfg.datasets_path, cfg.figures_path, cfg.models_path,
              cfg.synth_artifacts_path, cfg.aux_artifacts_path, cfg.shared_aux_artifacts_path]:
        os.makedirs(p, exist_ok=True)
    R.write_sdg_config_files(cfg)
    return cfg


def _snapshot(cfg, prefix):
    for src, dst in [(cfg.results_file, f"{prefix}_results.csv"),
                     (cfg.results_file_classb, f"{prefix}_results_classb.csv")]:
        if os.path.exists(src):
            shutil.copy(src, os.path.join(cfg.results_path, dst))


def _auc_row(cfg, prefix):
    out = {}
    for label, name in [("std", f"{prefix}_results.csv"), ("classb", f"{prefix}_results_classb.csv")]:
        p = os.path.join(cfg.results_path, name)
        if os.path.exists(p):
            df = pd.read_csv(p)
            row = df[df["metric"] == "auc"]
            if len(row):
                out[label] = {c: float(row[c].values[0]) for c in df.columns if c.startswith("tm:")}
    return out


def run_trial(dataset, nd, trial, workers):
    src_trial = os.path.join(DATA, dataset, "scdesign2", "no_dp", f"{nd}d", str(trial))
    assert os.path.exists(os.path.join(src_trial, "datasets", "synthetic.h5ad")), \
        f"no source synth at {src_trial}"

    # ---- BB config sets up namespace; pre-place reused artifacts ----
    cfg = _make_cfg(dataset, nd, trial, workers, white_box=False)

    # reuse synthetic data + BB synth-shadow copulas + WB train copulas
    if not os.path.exists(cfg.target_synthetic_data_path):
        shutil.copy(os.path.join(src_trial, "datasets", "synthetic.h5ad"),
                    cfg.target_synthetic_data_path)
    for sub, dst in [("artifacts/synth", cfg.synth_artifacts_path),
                     ("models", cfg.models_path)]:
        srcd = os.path.join(src_trial, sub)
        for fn in os.listdir(srcd):
            d = os.path.join(dst, fn)
            if not os.path.exists(d):
                shutil.copy(os.path.join(srcd, fn), d)

    # ---- materialise members / non-members / new disjoint aux ----
    train_donors   = np.load(cfg.train_donors_path,   allow_pickle=True)
    holdout_donors = np.load(cfg.holdout_donors_path, allow_pickle=True)
    aux_donors     = np.load(cfg.aux_donors_path,     allow_pickle=True)
    full = ad.read_h5ad(os.path.join(DATA, dataset, "full_dataset_cleaned.h5ad"), backed="r")
    print("  [materialise] train/holdout/aux ...", flush=True)
    train_ad = _subset_h5ad(full, train_donors, cfg.train_path)
    _subset_h5ad(full, holdout_donors, cfg.holdout_path)
    _subset_h5ad(full, aux_donors, cfg.aux_path)
    cell_types = sorted(pd.unique(train_ad.obs["cell_type"].astype(str).values).tolist())
    full.file.close()

    # ---- refit ONLY the aux shadow copula on the new disjoint aux ----
    print("  [aux-shadow] fitting scDesign2 on new disjoint aux ...", flush=True)
    R._run_aux_shadow_model_shared(cfg, cell_types, force=True)

    targets = ad.AnnData(obs=pd.concat([
        ad.read_h5ad(cfg.train_path, backed="r").obs,
        ad.read_h5ad(cfg.holdout_path, backed="r").obs]))

    # ---- PASS 1: BB pair (tm:100/101) ----
    print("  [attack] BB quad (tm:100 BB+aux, tm:101 BB-aux) ...", flush=True)
    for f in [cfg.all_scores_file, cfg.results_file, cfg.all_scores_file_classb, cfg.results_file_classb]:
        if os.path.exists(f):
            os.remove(f)
    R._initialise_results_files(cfg, targets)
    R.save_results_quad(cfg, R.run_mamamia_attack_quad(cfg))
    _snapshot(cfg, "bb")

    # ---- PASS 2: WB pair (tm:000/001) — reuse train copulas, no synth shadow ----
    print("  [attack] WB quad (tm:000 WB+aux, tm:001 WB-aux) ...", flush=True)
    cfg.mia_setting.white_box = True
    for f in [cfg.all_scores_file, cfg.results_file, cfg.all_scores_file_classb, cfg.results_file_classb]:
        if os.path.exists(f):
            os.remove(f)
    R._initialise_results_files(cfg, targets)
    R.save_results_quad(cfg, R.run_mamamia_attack_quad(cfg))
    _snapshot(cfg, "wb")

    bb, wb = _auc_row(cfg, "bb"), _auc_row(cfg, "wb")
    print(f"  [DONE] {dataset} {nd}d t{trial}", flush=True)
    print(f"    BB: {bb}", flush=True)
    print(f"    WB: {wb}", flush=True)

    # free disk (synth is a reused copy; the deliverables are the result CSVs)
    R.delete_interim_h5ad(cfg)
    if os.path.exists(cfg.target_synthetic_data_path):
        os.remove(cfg.target_synthetic_data_path)
    return {"bb": bb, "wb": wb}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--nd", type=int, required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    run_trial(a.dataset, a.nd, a.trial, a.workers)
    print(f"elapsed {time.time()-t0:.0f}s", flush=True)

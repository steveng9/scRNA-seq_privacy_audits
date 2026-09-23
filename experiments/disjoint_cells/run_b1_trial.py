#!/usr/bin/env python3
"""
B1 (disjoint-cells / equal-cell-budget) — STEP 2: run one trial for scDesign2.

Reuses run_experiment.py's validated functions by importing it as a module
(with a dummy argv so its module-level CLI parse doesn't choke), then driving
the phases explicitly so we can SEPARATE:
    - generator-training cells (member gen_train G-cells)   -> synthetic.h5ad
    - member TARGET cells (disjoint attack-portion, B cells) -> attack targets
    - non-member TARGET cells (budget-matched, B cells)      -> attack targets

Phases:
  1. Materialise gen_train.h5ad (-> cfg.train_path) and auxiliary.h5ad (aux donors).
  2. Generate synthetic.h5ad from gen_train (generator sees ONLY these cells).
  3. Fit BB synth-shadow copulas (on synthetic.h5ad) and aux-shadow copulas.
  4. OVERWRITE cfg.train_path <- member disjoint target cells,
                cfg.holdout_path <- non-member budget-matched cells.
     Rebuild the scores file, then run the quad BB attack (BB+aux / BB-aux,
     standard + Class B) and record donor-level AUC.

`variant`: 'disjoint' (default) or 'overlap' (control: member targets = the
training cells themselves, to quantify the disjoint-vs-overlap gap).

Namespace: dataset_name = "{dataset}/b1_disjoint/{tag}"; donor splits and the
full dataset are read from the shared "{dataset}/" root, so B1 trial N reuses
the EXISTING donor split "{dataset}/splits/{nd}d/N".
"""
import os, sys, argparse, glob, time
import numpy as np
import anndata as ad

REPO = "/home/golobs/scRNA-seq_privacy_audits"
DATA = "/home/golobs/data/scMAMAMIA"
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "experiments", "disjoint_cells"))

from build_cell_splits import build as build_manifest   # noqa: E402

# Import run_experiment with a harmless argv so its top-level parse succeeds.
_saved_argv = sys.argv
sys.argv = ["run_experiment.py", "/dev/null"]
import run_experiment as R   # noqa: E402
sys.argv = _saved_argv

import yaml  # noqa: E402

# scMAMA-MIA params — mirror the winning config used in the sweep (Class B on).
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


def _subset_h5ad(full_backed, cell_ids, out_path):
    """Materialise a raw-count h5ad for the given cell ids (order preserved)."""
    mask = full_backed.obs_names.isin(set(cell_ids.tolist()))
    sub = full_backed[mask].to_memory()
    sub.write_h5ad(out_path)
    return sub.n_obs


def _run(cmd):
    """Run a subprocess, streaming output; raise on failure."""
    import subprocess
    print(f"    $ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def _generate_scvi(cfg, conda_env="scvi_", max_epochs=400, batch_size=512):
    """Train scVI on the gen_train cells (currently at cfg.train_path) and sample
    synthetic.h5ad. Only the TARGET generator changes; the shadow model stays the
    scDesign2 proxy (generator_name='scdesign2'), i.e. the black-box transfer attack
    that already works for scVI in the standard pipeline (33 saved results on disk)."""
    scvi_script = os.path.join(REPO, "src", "sdg", "scvi", "run_scvi_standalone.py")
    model_dir = os.path.join(cfg.models_path, "scvi_model")
    os.makedirs(model_dir, exist_ok=True)
    gen_train = cfg.train_path  # holds gen_train cells during phase 2 (pre-swap)
    n_cells = ad.read_h5ad(gen_train, backed="r").n_obs
    prefix = f"conda run --no-capture-output -n {conda_env}"
    _run(f"{prefix} python {scvi_script} train {gen_train} {model_dir} "
         f"--hvg-path {cfg.hvg_path} --max-epochs {max_epochs} --batch-size {batch_size}")
    _run(f"{prefix} python {scvi_script} generate {gen_train} {model_dir} "
         f"{cfg.target_synthetic_data_path} {n_cells}")


def run_trial(dataset, nd, trial, G, B, tag, variant, parallel_workers, generator="scdesign2"):
    cfg_dir = os.path.join(REPO, "experiments", "disjoint_cells", "_cfgs", dataset, tag)
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, f"{nd}d_t{trial}.yaml")
    cfg_dict = {
        "dir_list": {"local": {"home": REPO, "data": DATA},
                     "server": {"home": REPO, "data": DATA}},
        "dataset_name": f"{dataset}/b1_disjoint/{tag}",
        "hvg_path": os.path.join(DATA, dataset, "hvg.csv"),
        "generator_name": "scdesign2",
        "plot_results": False,
        "parallelize": True,
        "parallel_workers": parallel_workers,
        "min_aux_donors": 10,
        "mamamia_params": dict(MAMAMIA_PARAMS),
        "mia_setting": {
            "sample_donors_strategy_fn": "sample_donors_strategy_2",
            "num_donors": nd, "white_box": False,
            "use_wb_hvgs": True, "use_aux": True, "run_quad_bb": True,
        },
    }
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)

    cfg = R.create_config(cfg_path)
    # Force this trial's number to match the requested donor-split trial so we
    # reuse the EXISTING splits/{nd}d/{trial} donor sets.
    cfg.trial_num = int(trial)
    cfg.trial_dir = os.path.join(cfg.cfg_dir, str(trial))
    # Re-derive trial-dependent paths for the forced trial_num:
    cfg.datasets_path = os.path.join(cfg.trial_dir, "datasets")
    cfg.results_path  = os.path.join(cfg.trial_dir, "results")
    cfg.figures_path  = os.path.join(cfg.results_path, "figures")
    cfg.models_path   = os.path.join(cfg.trial_dir, "models")
    cfg.artifacts_path = os.path.join(cfg.trial_dir, "artifacts")
    cfg.synth_artifacts_path = os.path.join(cfg.artifacts_path, "synth")
    cfg.aux_artifacts_path   = os.path.join(cfg.artifacts_path, "aux")
    cfg.splits_path = os.path.join(cfg.base_data_dir, "splits", cfg.split_name, str(trial))
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
    cfg.shared_aux_artifacts_path = os.path.join(cfg.base_data_dir, "aux_artifacts_b1",
                                                 cfg.split_name, str(trial), tag)
    for p in [cfg.datasets_path, cfg.figures_path, cfg.models_path,
              cfg.synth_artifacts_path, cfg.aux_artifacts_path, cfg.shared_aux_artifacts_path]:
        os.makedirs(p, exist_ok=True)
    R.write_sdg_config_files(cfg)

    # ---- manifest ----
    manifest, ok = build_manifest(dataset, nd, trial, G, B, verbose=True)
    if not ok:
        print("  manifest invalid; aborting trial", flush=True); return None

    full = ad.read_h5ad(os.path.join(DATA, dataset, "full_dataset_cleaned.h5ad"), backed="r")
    cell_types = list(full.obs["cell_type"].unique())

    # Phase 1: generator-training cells + aux
    print("  [phase1] materialising gen_train + aux ...", flush=True)
    aux_cells = full.obs_names[full.obs["individual"].isin(set(manifest["aux_donors"].tolist()))].to_numpy()
    _subset_h5ad(full, manifest["gen_train"], cfg.train_path)
    _subset_h5ad(full, aux_cells, cfg.aux_path)
    # placeholder holdout for init; overwritten before attack
    _subset_h5ad(full, manifest["tgt_nonmember"], cfg.holdout_path)

    # Phase 2: generate synthetic from gen_train, fit shadow copulas
    print(f"  [phase2] generating synthetic (generator={generator}) + shadow copulas ...", flush=True)
    if generator == "scvi":
        _generate_scvi(cfg)
    else:
        R.generate_target_synthetic_data(cfg, cell_types, force=True)
    R.run_sdg(cfg, cfg.synth_model_config_path, cell_types, "SYNTHETIC_DATA_SHADOW_MODEL", force=True)
    R._run_aux_shadow_model_shared(cfg, cell_types, force=False)

    # Phase 3: swap in TARGET cells (member disjoint | overlap control) + non-members
    print(f"  [phase3] swapping targets (variant={variant}) + attacking ...", flush=True)
    member_target = manifest["tgt_overlap"] if variant == "overlap" else manifest["tgt_disjoint"]
    _subset_h5ad(full, member_target, cfg.train_path)          # members (label=1)
    _subset_h5ad(full, manifest["tgt_nonmember"], cfg.holdout_path)  # non-members (label=0)
    full.file.close()

    for f in [cfg.all_scores_file, cfg.results_file, cfg.all_scores_file_classb, cfg.results_file_classb]:
        if os.path.exists(f): os.remove(f)
    targets = ad.AnnData(obs=__import__("pandas").concat([
        ad.read_h5ad(cfg.train_path, backed="r").obs,
        ad.read_h5ad(cfg.holdout_path, backed="r").obs]))
    R._initialise_results_files(cfg, targets)

    results = R.run_mamamia_attack_quad(cfg)
    R.save_results_quad(cfg, results)

    import pandas as pd
    out = {}
    for label, path in [("standard", cfg.results_file), ("classb", cfg.results_file_classb)]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            row = df[df["metric"] == "auc"]
            if len(row):
                out[label] = {c: float(row[c].values[0]) for c in df.columns if c.startswith("tm:")}
    print(f"  [done] {dataset} {nd}d t{trial} variant={variant} AUCs: {out}", flush=True)
    R.delete_interim_h5ad(cfg)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ok")
    ap.add_argument("--nd", type=int, required=True)
    ap.add_argument("--trial", default="1")
    ap.add_argument("--G", type=int, default=200)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--variant", choices=["disjoint", "overlap"], default="disjoint")
    ap.add_argument("--generator", choices=["scdesign2", "scvi"], default="scdesign2")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    run_trial(a.dataset, a.nd, a.trial, a.G, a.B, a.tag, a.variant, a.workers, a.generator)
    print(f"elapsed {time.time()-t0:.0f}s", flush=True)

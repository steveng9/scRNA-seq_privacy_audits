#!/usr/bin/env python3
"""
B2 (marginal-noise defense) — STEP 2: run one trial for scDesign2.

Answers kPMB W5 / meta point 6: A6 showed *covariance* noise fails to defeat the
enhanced (Class B) attack.  B2 tests whether noising the fitted **marginals**
(and/or covariance) at generation time defeats it, measuring the standard vs.
enhanced BB attack against the defended generator.

Design (reuses the validated B1 attack path + the DP regen helpers)
-------------------------------------------------------------------
Members  = all train-donor cells (standard membership).
Non-mem  = all holdout-donor cells (budget = whole donor, as in the paper).
The target copula is NOT retrained — we reuse the already-fitted no_dp copulas
for this exact donor split (splits are shared), copy them, inject noise per
(mode, s), then:
  1. Generate synthetic.h5ad from the NOISED copulas (Rscript scdesign2.r gen).
  2. Fit the BB synth-shadow copula on the noised synthetic data.
  3. Reuse the shared aux-shadow copula (noise-independent — cached from no_dp).
  4. Run the quad BB attack (BB+aux / BB-aux, standard + Class B) and record AUC.

The BB attack never reads the target copula directly (it re-fits from synth), so
the defense is faithful: marginal noise propagates train-copula -> synth -> the
attacker's shadow re-fit.

Namespace: dataset_name = "{dataset}/b2_marginal/{tag}", tag = "{mode}_s{s}"
(or "none" for the no-noise baseline).  Reuses splits/{nd}d/{trial}.
"""
import os, sys, argparse, time, tempfile
import numpy as np
import pandas as pd
import anndata as ad

REPO = "/home/golobs/scRNA-seq_privacy_audits"
DATA = "/home/golobs/data/scMAMAMIA"
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "experiments", "dp_marginal"))

from marginal_noise import noise_and_save_rds   # noqa: E402
# Reuse the validated DP-regeneration helpers (same code path the paper's DP
# covariance synthetic data was generated with).
from generators.gen_dp_quality_data import run_r_gen, assemble_synthetic  # noqa: E402

# Import run_experiment with a harmless argv so its top-level parse succeeds.
_saved_argv = sys.argv
sys.argv = ["run_experiment.py", "/dev/null"]
import run_experiment as R   # noqa: E402
sys.argv = _saved_argv

import yaml  # noqa: E402

# scMAMA-MIA params — identical to the winning sweep config (Class B on).
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


def _compute_quality(cfg, dataset):
    """Run the paper's EXACT quality metrics (MMD, LISI, ARI) on the noised synth.

    Reuses evaluation.sc_evaluate.SingleCellEvaluator with the identical config
    that experiments/sdg_comparison/run_quality_evals.py uses (n_hvgs=1000 in cfg
    — vestigial; get_statistical_evals uses the metric defaults n_hvgs=5000 and
    MMD's built-in 20k subsample; LISI/ARI use ALL cells — matching the paper).
    Real = train-donor cells (rebuilt from full + splits/train.npy), synth =
    the noised synthetic.h5ad.  Written to results/statistics_evals.csv.
    """
    from evaluation.sc_evaluate import SingleCellEvaluator
    qhome = os.path.join(cfg.results_path, "quality_eval")
    qcfg = {
        "dir_list": {"home": qhome, "figures": "figures", "res_files": "results"},
        "full_data_path": os.path.join(DATA, dataset, "full_dataset_cleaned.h5ad"),
        "synthetic_file": cfg.target_synthetic_data_path,
        "dataset_config": {
            "name": dataset,
            "test_count_file": cfg.train_donors_path,   # members = real train donors
            "synthetic_file": cfg.target_synthetic_data_path,
            "cell_type_col_name": "cell_type",
            "cell_label_col_name": "cell_label",
            "celltypist_model": "",
        },
        "evaluator_config": {"random_seed": 1},
        "n_hvgs": 1000,
    }
    ev = SingleCellEvaluator(config=qcfg)
    res = ev.get_statistical_evals()
    out_csv = os.path.join(cfg.results_path, "statistics_evals.csv")
    pd.DataFrame([res]).to_csv(out_csv, index=False)
    return res


def _subset_h5ad(full_backed, donor_ids, out_path):
    """Materialise a full-width h5ad for all cells of the given donors."""
    mask = full_backed.obs["individual"].isin(set(np.asarray(donor_ids).tolist()))
    sub = full_backed[mask].to_memory()
    sub.write_h5ad(out_path)
    return sub


def run_trial(dataset, nd, trial, mode, s, tag, parallel_workers):
    cfg_dir = os.path.join(REPO, "experiments", "dp_marginal", "_cfgs", dataset, tag)
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, f"{nd}d_t{trial}.yaml")
    cfg_dict = {
        "dir_list": {"local": {"home": REPO, "data": DATA},
                     "server": {"home": REPO, "data": DATA}},
        "dataset_name": f"{dataset}/b2_marginal/{tag}",
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
    # Force this trial's number to reuse the EXISTING donor split.
    cfg.trial_num = int(trial)
    cfg.trial_dir = os.path.join(cfg.cfg_dir, str(trial))
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
    # Reuse the noise-INDEPENDENT shared aux-shadow copulas fit by prior runs.
    cfg.shared_aux_artifacts_path = os.path.join(cfg.base_data_dir, "aux_artifacts",
                                                 cfg.split_name, str(trial))
    for p in [cfg.datasets_path, cfg.figures_path, cfg.models_path,
              cfg.synth_artifacts_path, cfg.aux_artifacts_path, cfg.shared_aux_artifacts_path]:
        os.makedirs(p, exist_ok=True)
    R.write_sdg_config_files(cfg)

    # ---- load donor splits ----
    train_donors   = np.load(cfg.train_donors_path,   allow_pickle=True)
    holdout_donors = np.load(cfg.holdout_donors_path, allow_pickle=True)
    aux_donors     = np.load(cfg.aux_donors_path,     allow_pickle=True)

    full = ad.read_h5ad(os.path.join(DATA, dataset, "full_dataset_cleaned.h5ad"), backed="r")
    all_var_names = full.var_names.copy()

    # ---- HVG mask (shared) — positional boolean over all genes, as in gen_dp_quality_data ----
    hvg_mask = pd.read_csv(cfg.hvg_path)["highly_variable"].values.astype(bool)
    assert len(hvg_mask) == len(all_var_names), \
        f"HVG mask len {len(hvg_mask)} != n_genes {len(all_var_names)}"

    # ---- Phase 1: materialise members / non-members / aux ----
    print("  [phase1] materialising train/holdout/aux ...", flush=True)
    train_ad = _subset_h5ad(full, train_donors, cfg.train_path)     # members
    _subset_h5ad(full, holdout_donors, cfg.holdout_path)            # non-members
    if not all(os.path.exists(os.path.join(cfg.shared_aux_artifacts_path, f"{ct}.rds"))
               for ct in train_ad.obs["cell_type"].astype(str).unique()):
        _subset_h5ad(full, aux_donors, cfg.aux_path)                # only if aux shadow not cached
    train_ct_arr = train_ad.obs["cell_type"].astype(str).values
    cell_types = sorted(pd.unique(train_ct_arr).tolist())
    full.file.close()

    # ---- Phase 2: copy + noise the target copulas ----
    print(f"  [phase2] noising copulas (mode={mode}, s={s}) ...", flush=True)
    src_models = os.path.join(DATA, dataset, "scdesign2", "no_dp", f"{nd}d", str(trial), "models")
    seed = abs(hash((dataset, nd, trial, mode, s))) % (2**31)
    rng = np.random.default_rng(seed)
    for ct in cell_types:
        src_rds = os.path.join(src_models, f"{ct}.rds")
        dst_rds = os.path.join(cfg.models_path, f"{ct}.rds")
        if not os.path.exists(src_rds):
            print(f"    [SKIP] no source copula for {ct}", flush=True)
            continue
        try:
            noise_and_save_rds(src_rds, ct, s, mode, dst_rds, rng)
        except Exception as e:
            print(f"    [WARN] noise failed for {ct}: {e}", flush=True)

    # ---- Phase 3: generate synthetic from noised copulas ----
    print("  [phase3] generating synthetic from noised copulas ...", flush=True)
    with tempfile.TemporaryDirectory(prefix="b2_gen_") as tmp:
        from concurrent.futures import ThreadPoolExecutor
        def _gen_one(ct):
            n = int((train_ct_arr == ct).sum())
            rds = os.path.join(cfg.models_path, f"{ct}.rds")
            if n == 0 or not os.path.exists(rds):
                return ct
            run_r_gen(rds, n, os.path.join(tmp, f"out{ct}.rds"))
            return ct
        with ThreadPoolExecutor(max_workers=parallel_workers) as ex:
            list(ex.map(_gen_one, cell_types))
        synth = assemble_synthetic(cell_types, train_ct_arr, tmp, hvg_mask, all_var_names)
        synth.write_h5ad(cfg.target_synthetic_data_path)
    print(f"    synthetic.h5ad shape={synth.shape}", flush=True)

    # ---- Phase 4: BB synth-shadow (on noised synth) + shared aux-shadow ----
    print("  [phase4] fitting BB synth-shadow + aux-shadow ...", flush=True)
    R.run_sdg(cfg, cfg.synth_model_config_path, cell_types, "SYNTHETIC_DATA_SHADOW_MODEL", force=True)
    R._run_aux_shadow_model_shared(cfg, cell_types, force=False)

    # ---- Phase 5: init results + quad attack ----
    print("  [phase5] running quad BB attack ...", flush=True)
    for f in [cfg.all_scores_file, cfg.results_file, cfg.all_scores_file_classb, cfg.results_file_classb]:
        if os.path.exists(f):
            os.remove(f)
    targets = ad.AnnData(obs=pd.concat([
        ad.read_h5ad(cfg.train_path, backed="r").obs,
        ad.read_h5ad(cfg.holdout_path, backed="r").obs]))
    R._initialise_results_files(cfg, targets)

    results = R.run_mamamia_attack_quad(cfg)
    R.save_results_quad(cfg, results)

    out = {}
    for label, path in [("standard", cfg.results_file), ("classb", cfg.results_file_classb)]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            row = df[df["metric"] == "auc"]
            if len(row):
                out[label] = {c: float(row[c].values[0]) for c in df.columns if c.startswith("tm:")}
    print(f"  [done] {dataset} {nd}d t{trial} mode={mode} s={s} AUCs: {out}", flush=True)

    # ---- Phase 6: quality metrics (paper's exact evaluator) BEFORE deleting synth ----
    print("  [phase6] computing quality metrics (MMD/LISI/ARI) ...", flush=True)
    try:
        q = _compute_quality(cfg, dataset)
        print(f"  [quality] {dataset} {nd}d t{trial} mode={mode} s={s} "
              f"mmd={q.get('mmd')} lisi={q.get('lisi')} ari={q.get('ari_real_vs_syn')}", flush=True)
    except Exception as e:
        import traceback
        print(f"  [WARN] quality metrics failed: {e}\n{traceback.format_exc()}", flush=True)

    R.delete_interim_h5ad(cfg)
    # Free disk: the results CSVs are the deliverable; synthetic.h5ad is large.
    if os.path.exists(cfg.target_synthetic_data_path):
        os.remove(cfg.target_synthetic_data_path)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ok")
    ap.add_argument("--nd", type=int, default=50)
    ap.add_argument("--trial", default="1")
    ap.add_argument("--mode", choices=["none", "cov", "marg1", "marg2", "marg", "both"], default="marg")
    ap.add_argument("--s", type=float, default=0.5)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    tag = a.tag or ("none" if a.mode == "none" else f"{a.mode}_s{a.s}")
    t0 = time.time()
    run_trial(a.dataset, a.nd, a.trial, a.mode, a.s, tag, a.workers)
    print(f"elapsed {time.time()-t0:.0f}s", flush=True)

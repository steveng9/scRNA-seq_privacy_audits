#!/usr/bin/env python3
"""Backfill statistics_evals.csv for B2 configs whose phase-6 quality step was
OOM-killed (attack AUCs already on disk, synthetic.h5ad still present).

Reuses the SAME evaluator/config as run_b2_trial._compute_quality (paper-exact:
no LISI/ARI subsampling; MMD's built-in 20k). Run sequentially — LISI/ARI on all
cells is RAM-heavy. Cleans up the leftover interim train/holdout/aux h5ad after.
"""
import os, sys
import pandas as pd

REPO = "/home/golobs/scRNA-seq_privacy_audits"
DATA = "/home/golobs/data/scMAMAMIA"
sys.path.insert(0, os.path.join(REPO, "src"))

# (dataset, tag, nd, trial)
CONFIGS = [
    ("aida", "cov_s0.5", 50, 1),
    ("aida", "cov_s1.0", 50, 1),
    ("aida", "cov_s2.0", 50, 1),
]


def compute(dataset, tag, nd, trial):
    from evaluation.sc_evaluate import SingleCellEvaluator
    trial_dir = os.path.join(DATA, dataset, "b2_marginal", tag, f"{nd}d", str(trial))
    results_path = os.path.join(trial_dir, "results")
    synth = os.path.join(trial_dir, "datasets", "synthetic.h5ad")
    train_npy = os.path.join(DATA, dataset, "splits", f"{nd}d", str(trial), "train.npy")
    out_csv = os.path.join(results_path, "statistics_evals.csv")
    if os.path.exists(out_csv):
        print(f"[skip] {dataset}/{tag} already has statistics_evals.csv", flush=True)
        return
    assert os.path.exists(synth), f"missing synth {synth}"
    qcfg = {
        "dir_list": {"home": os.path.join(results_path, "quality_eval"),
                     "figures": "figures", "res_files": "results"},
        "full_data_path": os.path.join(DATA, dataset, "full_dataset_cleaned.h5ad"),
        "synthetic_file": synth,
        "dataset_config": {
            "name": dataset, "test_count_file": train_npy,
            "synthetic_file": synth, "cell_type_col_name": "cell_type",
            "cell_label_col_name": "cell_label", "celltypist_model": "",
        },
        "evaluator_config": {"random_seed": 1},
        "n_hvgs": 1000,
    }
    print(f"[run ] {dataset}/{tag} ...", flush=True)
    ev = SingleCellEvaluator(config=qcfg)
    res = ev.get_statistical_evals()
    pd.DataFrame([res]).to_csv(out_csv, index=False)
    print(f"[done] {dataset}/{tag} -> {res}", flush=True)


def cleanup(dataset, tag, nd, trial):
    trial_dir = os.path.join(DATA, dataset, "b2_marginal", tag, f"{nd}d", str(trial))
    for fn in ("train.h5ad", "holdout.h5ad", "auxiliary.h5ad", "synthetic.h5ad"):
        p = os.path.join(trial_dir, "datasets", fn)
        if os.path.exists(p):
            sz = os.path.getsize(p) / 1e9
            os.remove(p)
            print(f"[clean] removed {p} ({sz:.1f} GB)", flush=True)


if __name__ == "__main__":
    for c in CONFIGS:
        compute(*c)
    print("=== quality backfill complete; cleaning interim h5ad ===", flush=True)
    for c in CONFIGS:
        cleanup(*c)
    print("=== done ===", flush=True)

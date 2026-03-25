"""
Batch quality evaluation for all DP synthetic datasets.

Runs SingleCellEvaluator.get_statistical_evals() on every
/home/golobs/data/ok_dp/eps_X/{n}d/{trial}/datasets/synthetic.h5ad
and saves results alongside.

Usage:
    python src/generators/run_dp_quality_eval.py
"""

import os, sys, glob
import numpy as np
import pandas as pd

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DP_ROOT   = "/home/golobs/data/ok_dp"
FULL_H5AD = "/home/golobs/data/ok/full_dataset_cleaned.h5ad"

from evaluation.sc_evaluate import SingleCellEvaluator


def make_cfg(synth_path, train_npy, results_dir):
    return {
        "dir_list": {"home": results_dir, "figures": "figures", "res_files": "results"},
        "full_data_path": FULL_H5AD,
        "synthetic_file": synth_path,
        "dataset_config": {
            "name": "ok",
            "test_count_file": train_npy,
            "synthetic_file": synth_path,
            "cell_type_col_name": "cell_type",
            "cell_label_col_name": "cell_label",
            "celltypist_model": "",   # not needed for statistical evals
        },
        "evaluator_config": {"random_seed": 1},
        "n_hvgs": 1000,
    }


def run_all():
    synth_files = sorted(glob.glob(
        os.path.join(DP_ROOT, "eps_*", "*d", "*", "datasets", "synthetic.h5ad")
    ))
    print(f"Found {len(synth_files)} DP synthetic datasets.", flush=True)

    for i, synth_path in enumerate(synth_files):
        datasets_dir = os.path.dirname(synth_path)
        trial_dir    = os.path.dirname(datasets_dir)
        results_dir  = os.path.join(trial_dir, "results", "quality_eval_results")
        out_csv      = os.path.join(results_dir, "results", "statistics_evals.csv")

        if os.path.exists(out_csv):
            print(f"[{i+1}/{len(synth_files)}] SKIP (exists): {out_csv}", flush=True)
            continue

        train_npy = os.path.join(datasets_dir, "train.npy")
        if not os.path.exists(train_npy):
            print(f"[{i+1}/{len(synth_files)}] SKIP (no train.npy): {synth_path}", flush=True)
            continue

        parts = synth_path.split(os.sep)
        eps_tag = [p for p in parts if p.startswith("eps_")][0]
        nd_tag  = [p for p in parts if p.endswith("d") and p[:-1].isdigit()][0]
        trial   = [p for p in parts if p.isdigit()][-1]
        print(f"[{i+1}/{len(synth_files)}] {eps_tag} / {nd_tag} / trial {trial}", flush=True)

        try:
            os.makedirs(os.path.join(results_dir, "results"), exist_ok=True)
            os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
            cfg = make_cfg(synth_path, train_npy, results_dir)
            evaluator = SingleCellEvaluator(config=cfg)
            results = evaluator.get_statistical_evals()
            evaluator.save_results_to_csv(results, out_csv)
            print(f"  → lisi={results.get('lisi','?'):.4f}  "
                  f"ari={results.get('ari_real_vs_syn','?'):.4f}  "
                  f"mmd={results.get('mmd','?'):.6f}", flush=True)
        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()


if __name__ == "__main__":
    run_all()
    print("\nAll done.")

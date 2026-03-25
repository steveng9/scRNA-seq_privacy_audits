"""
Batch quality evaluation for all SDG-comparison synthetic datasets.

Runs MMD, LISI, and ARI on every synthetic.h5ad produced by the SDG
comparison pipeline and saves results alongside the synthetic file.

Covers: scDesign3-Gaussian, scDesign3-Vine, scVI, scDiffusion
        on both the OK1K and AIDA datasets.

Also covers the original scDesign2 data (ok, cg, aida) if any
statistics_evals.csv files are missing there.

Usage:
    python experiments/sdg_comparison/run_quality_evals.py [--dry-run]

Output per trial:
    {trial_dir}/results/quality_eval_results/results/statistics_evals.csv
    Columns: mmd, lisi, ari_real_vs_syn
"""

import os
import sys
import glob
import argparse

DATA = "/home/golobs/data"
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from evaluation.sc_evaluate import SingleCellEvaluator


# ---------------------------------------------------------------------------
# Dataset registry
# Each entry: (data_root_glob, full_data_path, dataset_name)
# data_root_glob: passed to glob to find all synthetic.h5ad files
# ---------------------------------------------------------------------------
DATASETS = [
    # --- New SDG methods: OK1K ---
    (f"{DATA}/ok_sd3g",   f"{DATA}/ok/full_dataset_cleaned.h5ad",   "ok"),
    (f"{DATA}/ok_sd3v",   f"{DATA}/ok/full_dataset_cleaned.h5ad",   "ok"),
    (f"{DATA}/ok_scvi",   f"{DATA}/ok/full_dataset_cleaned.h5ad",   "ok"),
    (f"{DATA}/ok_scdiff", f"{DATA}/ok/full_dataset_cleaned.h5ad",   "ok"),

    # --- New SDG methods: AIDA ---
    (f"{DATA}/aida_sd3g",   f"{DATA}/aida/full_dataset_cleaned.h5ad", "aida"),
    (f"{DATA}/aida_sd3v",   f"{DATA}/aida/full_dataset_cleaned.h5ad", "aida"),
    (f"{DATA}/aida_scvi",   f"{DATA}/aida/full_dataset_cleaned.h5ad", "aida"),
    (f"{DATA}/aida_scdiff", f"{DATA}/aida/full_dataset_cleaned.h5ad", "aida"),

    # scDesign2 (original) and scDesign2+DP are already evaluated — omitted here.
]

# Cell type column name per dataset
CELL_TYPE_COL = {
    "ok":   "cell_type",
    "cg":   "cell_type",
    "aida": "cell_type",
}


def make_cfg(synth_path, train_npy, full_data_path, dataset_name, results_dir):
    return {
        "dir_list": {"home": results_dir, "figures": "figures", "res_files": "results"},
        "full_data_path": full_data_path,
        "synthetic_file": synth_path,
        "dataset_config": {
            "name": dataset_name,
            "test_count_file": train_npy,
            "synthetic_file": synth_path,
            "cell_type_col_name": CELL_TYPE_COL[dataset_name],
            "cell_label_col_name": "cell_label",
            "celltypist_model": "",  # not needed for statistical evals
        },
        "evaluator_config": {"random_seed": 1},
        "n_hvgs": 1000,
    }


def collect_jobs(datasets):
    """Return list of (synth_path, full_data_path, dataset_name, out_csv) tuples."""
    jobs = []
    for data_root, full_data_path, dataset_name in datasets:
        pattern = os.path.join(data_root, "*d", "*", "datasets", "synthetic.h5ad")
        for synth_path in sorted(glob.glob(pattern)):
            datasets_dir = os.path.dirname(synth_path)
            trial_dir    = os.path.dirname(datasets_dir)
            results_dir  = os.path.join(trial_dir, "results", "quality_eval_results")
            out_csv      = os.path.join(results_dir, "results", "statistics_evals.csv")
            jobs.append((synth_path, full_data_path, dataset_name, results_dir, out_csv))
    return jobs


def run(dry_run=False):
    jobs = collect_jobs(DATASETS)
    total   = len(jobs)
    skipped = sum(1 for *_, out_csv in jobs if os.path.exists(out_csv))
    todo    = total - skipped

    print(f"Found {total} synthetic datasets: {skipped} already evaluated, {todo} to run.\n")

    done = errors = 0
    for i, (synth_path, full_data_path, dataset_name, results_dir, out_csv) in enumerate(jobs):
        # Derive a short label for display
        # Path: .../data/{src}/{nd}d/{trial}/datasets/synthetic.h5ad
        parts  = synth_path.split(os.sep)
        src    = parts[-5]  # e.g. ok_sd3g
        nd_tag = parts[-4]  # e.g. 10d
        trial  = parts[-3]  # e.g. 3
        label  = f"{src}/{nd_tag}/t{trial}"

        if os.path.exists(out_csv):
            print(f"[{i+1}/{total}] SKIP  {label}")
            continue

        train_npy = os.path.join(os.path.dirname(synth_path), "train.npy")
        if not os.path.exists(train_npy):
            print(f"[{i+1}/{total}] SKIP  {label}  (no train.npy)")
            continue

        print(f"[{i+1}/{total}] EVAL  {label} ...", flush=True)

        if dry_run:
            print("  [dry-run] would run evaluator here")
            continue

        try:
            os.makedirs(os.path.join(results_dir, "results"),  exist_ok=True)
            os.makedirs(os.path.join(results_dir, "figures"),  exist_ok=True)
            cfg = make_cfg(synth_path, train_npy, full_data_path, dataset_name, results_dir)
            evaluator = SingleCellEvaluator(config=cfg)
            results = evaluator.get_statistical_evals()
            evaluator.save_results_to_csv(results, out_csv)

            lisi = results.get("lisi")
            ari  = results.get("ari_real_vs_syn")
            mmd  = results.get("mmd")
            lisi_s = f"{lisi:.4f}" if lisi is not None else "N/A"
            ari_s  = f"{ari:.4f}"  if ari  is not None else "N/A"
            mmd_s  = f"{mmd:.6f}"  if mmd  is not None else "N/A"
            print(f"  lisi={lisi_s}  ari={ari_s}  mmd={mmd_s}", flush=True)
            done += 1

        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            errors += 1

    print(f"\nDone: {done} evaluated, {errors} errors, {skipped} skipped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="List jobs without running the evaluator")
    args = parser.parse_args()
    run(dry_run=args.dry_run)

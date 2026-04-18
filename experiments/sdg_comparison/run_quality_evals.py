"""
Batch quality evaluation for all SDG-comparison synthetic datasets.

Runs MMD, LISI, and ARI on every synthetic.h5ad produced by the SDG
comparison pipeline and saves results alongside the synthetic file.

Covers: scDesign3-Gaussian, scDesign3-Vine, scVI, scDiffusion
        on both the OK1K and AIDA datasets.

Also covers the original scDesign2 data (ok, cg, aida) if any
statistics_evals.csv files are missing there.

Usage:
    python experiments/sdg_comparison/run_quality_evals.py [--dry-run] [--workers N] [--max-donors N]

Output per trial:
    {trial_dir}/results/quality_eval_results/results/statistics_evals.csv
    Columns: mmd, lisi, ari_real_vs_syn
"""

import os
import sys
import glob
import argparse
import traceback
import multiprocessing as mp

DATA = "/home/golobs/data/scMAMAMIA"
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ---------------------------------------------------------------------------
# Dataset registry
# Each entry: (data_root, full_data_path, dataset_name)
# ---------------------------------------------------------------------------

DATASETS = [
    # --- New SDG methods: OK1K ---
    (f"{DATA}/ok/scdesign3/gaussian",   f"{DATA}/ok/full_dataset_cleaned.h5ad", "ok"),
    (f"{DATA}/ok/scdesign3/vine",       f"{DATA}/ok/full_dataset_cleaned.h5ad", "ok"),
    (f"{DATA}/ok/scvi/no_dp",           f"{DATA}/ok/full_dataset_cleaned.h5ad", "ok"),
    (f"{DATA}/ok/scdiffusion/no_dp",    f"{DATA}/ok/full_dataset_cleaned.h5ad", "ok"),
    (f"{DATA}/ok/nmf/no_dp",            f"{DATA}/ok/full_dataset_cleaned.h5ad", "ok"),

    # --- New SDG methods: AIDA ---
    (f"{DATA}/aida/scdesign3/gaussian",  f"{DATA}/aida/full_dataset_cleaned.h5ad", "aida"),
    (f"{DATA}/aida/scdesign3/vine",      f"{DATA}/aida/full_dataset_cleaned.h5ad", "aida"),
    (f"{DATA}/aida/scvi/no_dp",          f"{DATA}/aida/full_dataset_cleaned.h5ad", "aida"),
    (f"{DATA}/aida/scdiffusion/no_dp",   f"{DATA}/aida/full_dataset_cleaned.h5ad", "aida"),
    (f"{DATA}/aida/nmf/no_dp",           f"{DATA}/aida/full_dataset_cleaned.h5ad", "aida"),

    # scDesign2+DP high-epsilon:
    (f"{DATA}/ok/scdesign2/eps_100000",    f"{DATA}/ok/full_dataset_cleaned.h5ad", "ok"),
    (f"{DATA}/ok/scdesign2/eps_1000000",   f"{DATA}/ok/full_dataset_cleaned.h5ad", "ok"),
    (f"{DATA}/ok/scdesign2/eps_10000000",  f"{DATA}/ok/full_dataset_cleaned.h5ad", "ok"),
    (f"{DATA}/ok/scdesign2/eps_100000000",   f"{DATA}/ok/full_dataset_cleaned.h5ad", "ok"),
    (f"{DATA}/ok/scdesign2/eps_1000000000",  f"{DATA}/ok/full_dataset_cleaned.h5ad", "ok"),
]

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
            "celltypist_model": "",
        },
        "evaluator_config": {"random_seed": 1},
        "n_hvgs": 1000,
    }


def collect_jobs(datasets, max_donors=None):
    """Return list of (synth_path, full_data_path, dataset_name, results_dir, out_csv)."""
    jobs = []
    for data_root, full_data_path, dataset_name in datasets:
        pattern = os.path.join(data_root, "*d", "*", "datasets", "synthetic.h5ad")
        for synth_path in sorted(glob.glob(pattern)):
            # Extract donor count from path
            parts = synth_path.split(os.sep)
            nd_tag = parts[-4]  # e.g. "10d", "200d"
            nd = int(nd_tag.rstrip("d")) if nd_tag.rstrip("d").isdigit() else None
            if max_donors is not None and nd is not None and nd > max_donors:
                continue

            datasets_dir = os.path.dirname(synth_path)
            trial_dir    = os.path.dirname(datasets_dir)
            results_dir  = os.path.join(trial_dir, "results", "quality_eval_results")
            out_csv      = os.path.join(results_dir, "results", "statistics_evals.csv")
            jobs.append((synth_path, full_data_path, dataset_name, results_dir, out_csv))
    return jobs


def _run_one(args):
    """Worker function: evaluate one synthetic dataset. Returns (label, status, msg)."""
    synth_path, full_data_path, dataset_name, results_dir, out_csv = args

    parts  = synth_path.split(os.sep)
    src    = parts[-5]
    nd_tag = parts[-4]
    trial  = parts[-3]
    label  = f"{src}/{nd_tag}/t{trial}"

    if os.path.exists(out_csv):
        return (label, "skip", None)

    train_npy = os.path.join(os.path.dirname(synth_path), "train.npy")
    if not os.path.exists(train_npy):
        return (label, "skip-no-train", None)

    # Import here so each worker process initialises its own copy
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    from evaluation.sc_evaluate import SingleCellEvaluator

    try:
        os.makedirs(os.path.join(results_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
        cfg = make_cfg(synth_path, train_npy, full_data_path, dataset_name, results_dir)
        evaluator = SingleCellEvaluator(config=cfg)
        results   = evaluator.get_statistical_evals()
        evaluator.save_results_to_csv(results, out_csv)

        lisi = results.get("lisi")
        ari  = results.get("ari_real_vs_syn")
        mmd  = results.get("mmd")
        msg  = (f"lisi={lisi:.4f}  ari={ari:.4f}  mmd={mmd:.6f}"
                if all(v is not None for v in [lisi, ari, mmd]) else "partial results")
        return (label, "ok", msg)

    except Exception as e:
        tb = traceback.format_exc()
        return (label, "error", f"{e}\n{tb}")


def run(dry_run=False, workers=4, max_donors=None, dataset_filter=None):
    datasets = DATASETS
    if dataset_filter:
        datasets = [d for d in datasets if dataset_filter in d[0]]
        print(f"Dataset filter '{dataset_filter}': {len(datasets)} dataset(s) matched.", flush=True)
    jobs = collect_jobs(datasets, max_donors=max_donors)
    total   = len(jobs)
    skipped = sum(1 for *_, out_csv in jobs if os.path.exists(out_csv))
    todo    = total - skipped

    print(f"Found {total} synthetic datasets (max_donors={max_donors}): "
          f"{skipped} already evaluated, {todo} to run.\n", flush=True)

    if dry_run:
        for synth_path, _, dataset_name, results_dir, out_csv in jobs:
            parts  = synth_path.split(os.sep)
            label  = f"{parts[-5]}/{parts[-4]}/t{parts[-3]}"
            status = "SKIP" if os.path.exists(out_csv) else "EVAL"
            print(f"  {status}  {label}")
        return

    # Jobs that actually need running
    pending = [j for j in jobs if not os.path.exists(j[-1])]

    if not pending:
        print("Nothing to do.")
        return

    done = errors = 0

    if workers == 1:
        results_iter = map(_run_one, pending)
    else:
        pool = mp.Pool(processes=workers)
        results_iter = pool.imap_unordered(_run_one, pending)

    try:
        for label, status, msg in results_iter:
            if status == "ok":
                print(f"  [OK]    {label}  {msg}", flush=True)
                done += 1
            elif status == "error":
                print(f"  [ERROR] {label}\n{msg}", flush=True)
                errors += 1
            # skip/skip-no-train are not printed (already counted above)
    finally:
        if workers > 1:
            pool.close()
            pool.join()

    print(f"\nDone: {done} evaluated, {errors} errors, {skipped} already existed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",    action="store_true",
                        help="List jobs without running the evaluator")
    parser.add_argument("--workers",    type=int, default=4,
                        help="Parallel worker processes (default: 4)")
    parser.add_argument("--max-donors", type=int, default=100,
                        help="Skip donor counts above this value (default: 100)")
    parser.add_argument("--dataset-filter", default=None,
                        help="Only process dataset roots containing this substring "
                             "(e.g. 'nmf', 'ok_scvi', 'aida')")
    args = parser.parse_args()
    run(dry_run=args.dry_run, workers=args.workers, max_donors=args.max_donors,
        dataset_filter=args.dataset_filter)

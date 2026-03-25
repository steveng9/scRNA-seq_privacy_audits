#!/usr/bin/env python3
"""Recompute MMD for every existing statistics_evals.csv across all data roots.

Patches only the `mmd` column in-place; LISI and ARI are left untouched.
Skips trial directories where no statistics_evals.csv exists yet.

Usage:
    python recompute_mmd.py [--dry-run]

Covers:
    scDesign2:       ok/, aida/, cg/
    scDesign2 + DP:  ok_dp/eps_*/
    SDG comparison:  ok_sd3g/, ok_sd3v/, ok_scvi/, ok_scdiff/
                     aida_sd3g/, aida_sd3v/, aida_scvi/, aida_scdiff/
"""

import argparse
import csv
import glob
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor

DATA = "/home/golobs/data"
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")

# (glob pattern for synthetic.h5ad, full_dataset_cleaned.h5ad path)
CONFIGS = [
    # scDesign2 main
    (f"{DATA}/ok/*d/*/datasets/synthetic.h5ad",          f"{DATA}/ok/full_dataset_cleaned.h5ad"),
    (f"{DATA}/aida/*d/*/datasets/synthetic.h5ad",        f"{DATA}/aida/full_dataset_cleaned.h5ad"),
    (f"{DATA}/cg/*d/*/datasets/synthetic.h5ad",          f"{DATA}/cg/full_dataset_cleaned.h5ad"),
    # scDesign2 + DP
    (f"{DATA}/ok_dp/eps_*/*d/*/datasets/synthetic.h5ad", f"{DATA}/ok/full_dataset_cleaned.h5ad"),
    # SDG comparison — OneK1K
    (f"{DATA}/ok_sd3g/*d/*/datasets/synthetic.h5ad",     f"{DATA}/ok/full_dataset_cleaned.h5ad"),
    (f"{DATA}/ok_sd3v/*d/*/datasets/synthetic.h5ad",     f"{DATA}/ok/full_dataset_cleaned.h5ad"),
    (f"{DATA}/ok_scvi/*d/*/datasets/synthetic.h5ad",     f"{DATA}/ok/full_dataset_cleaned.h5ad"),
    (f"{DATA}/ok_scdiff/*d/*/datasets/synthetic.h5ad",   f"{DATA}/ok/full_dataset_cleaned.h5ad"),
    # SDG comparison — AIDA
    (f"{DATA}/aida_sd3g/*d/*/datasets/synthetic.h5ad",   f"{DATA}/aida/full_dataset_cleaned.h5ad"),
    (f"{DATA}/aida_sd3v/*d/*/datasets/synthetic.h5ad",   f"{DATA}/aida/full_dataset_cleaned.h5ad"),
    (f"{DATA}/aida_scvi/*d/*/datasets/synthetic.h5ad",   f"{DATA}/aida/full_dataset_cleaned.h5ad"),
    (f"{DATA}/aida_scdiff/*d/*/datasets/synthetic.h5ad", f"{DATA}/aida/full_dataset_cleaned.h5ad"),
]


def _collect_jobs():
    """Return list of (synth_path, full_h5ad, csv_path, label) for every trial
    that has a synthetic.h5ad + train.npy, regardless of whether a CSV exists."""
    jobs = []
    seen = set()
    for pattern, full_h5ad in CONFIGS:
        for synth_path in sorted(glob.glob(pattern)):
            if synth_path in seen:
                continue
            seen.add(synth_path)
            datasets_dir = os.path.dirname(synth_path)
            train_npy    = os.path.join(datasets_dir, "train.npy")
            if not os.path.exists(train_npy):
                continue
            trial_dir = os.path.dirname(datasets_dir)
            csv_path  = os.path.join(trial_dir, "results", "quality_eval_results",
                                     "results", "statistics_evals.csv")
            parts = synth_path.split(os.sep)
            label = "/".join(p for p in parts if p not in ("", "home", "golobs", "data",
                                                            "datasets", "synthetic.h5ad"))
            jobs.append((synth_path, full_h5ad, csv_path, label))
    return jobs


def _recompute_mmd_one(args):
    synth_path, full_h5ad, csv_path, label = args
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)

    import numpy as np
    import scanpy as sc
    from evaluation.utils.sc_metrics import Statistics, filter_low_quality_cells_and_genes

    datasets_dir = os.path.dirname(synth_path)
    train_npy    = os.path.join(datasets_dir, "train.npy")

    train_donors = np.load(train_npy, allow_pickle=True)
    all_data     = sc.read_h5ad(full_h5ad, backed="r")
    mask         = all_data.obs["individual"].isin(train_donors)
    real_data    = all_data[mask].to_memory()
    all_data.file.close()

    syn_data = sc.read_h5ad(synth_path)

    real_data = filter_low_quality_cells_and_genes(real_data)
    syn_data  = filter_low_quality_cells_and_genes(syn_data)
    common    = real_data.var_names.intersection(syn_data.var_names)
    real_data = real_data[:, common]
    syn_data  = syn_data[:, common]

    stats = Statistics(random_seed=1)

    csv_existed = os.path.exists(csv_path)

    if csv_existed:
        # Patch MMD only; leave LISI and ARI untouched.
        mmd = stats.compute_mmd_optimized(real_data, syn_data)
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1, f"Expected 1 row in {csv_path}, got {len(rows)}"
        old_mmd = float(rows[0]["mmd"])
        rows[0]["mmd"] = str(mmd)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return label, old_mmd, mmd, "patch"
    else:
        # No CSV yet — run full eval and create it.
        mmd  = stats.compute_mmd_optimized(real_data, syn_data)
        lisi = stats.compute_lisi(real_data, syn_data)
        ari, _ = stats.compute_ari(real_data, syn_data, cell_type_col="cell_type")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["mmd", "lisi", "ari_real_vs_syn"])
            writer.writeheader()
            writer.writerow({"mmd": mmd, "lisi": lisi, "ari_real_vs_syn": ari})
        return label, None, mmd, "full"


def run(dry_run=False):
    jobs = _collect_jobs()
    csv_exists_count  = sum(1 for _, _, csv_path, _ in jobs if os.path.exists(csv_path))
    csv_missing_count = len(jobs) - csv_exists_count
    print(f"Found {len(jobs)} trials: {csv_exists_count} will patch MMD only, "
          f"{csv_missing_count} will run full eval (MMD+LISI+ARI).\n")

    if dry_run:
        for _, _, csv_path, label in jobs:
            mode = "patch" if os.path.exists(csv_path) else "full "
            print(f"  [{mode}] {label}")
        return

    results = []
    errors  = []
    ctx = multiprocessing.get_context("spawn")
    for i, job in enumerate(jobs):
        label      = job[3]
        csv_exists = os.path.exists(job[2])
        mode       = "patch" if csv_exists else "full "
        print(f"[{i+1}/{len(jobs)}] [{mode}] {label} ...", flush=True)
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
            fut = executor.submit(_recompute_mmd_one, job)
            try:
                lbl, old_mmd, new_mmd, mode_done = fut.result()
                if old_mmd is not None:
                    print(f"  old={old_mmd:.6f}  new={new_mmd:.6f}", flush=True)
                else:
                    print(f"  mmd={new_mmd:.6f}  (new)", flush=True)
                results.append((lbl, old_mmd, new_mmd, mode_done))
            except Exception as e:
                import traceback
                print(f"  [ERROR] {e}")
                traceback.print_exc()
                errors.append((label, str(e)))

    patched = [r for r in results if r[3] == "patch"]
    created = [r for r in results if r[3] == "full"]

    print("\n" + "=" * 72)
    print(f"SUMMARY: {len(patched)} patched, {len(created)} newly created, {len(errors)} errors\n")
    print(f"{'Label':<50} {'Old MMD':>10} {'New MMD':>10}")
    print("-" * 72)
    for lbl, old, new, _ in patched:
        print(f"  [patch] {lbl:<46} {old:10.6f} {new:10.6f}")
    for lbl, _, new, _ in created:
        print(f"  [full ] {lbl:<46} {'---':>10} {new:10.6f}")
    if errors:
        print("\nErrors:")
        for lbl, msg in errors:
            print(f"  {lbl}: {msg}")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="List jobs without computing anything")
    args = parser.parse_args()
    run(dry_run=args.dry_run)

#!/usr/bin/env python3
"""Compute / recompute quality metrics (MMD, LISI, ARI) for specified settings, then print the updated table."""

import os
import sys
import csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

DATA_ROOT = os.path.expanduser("~/data")
SRC_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")

DATASETS = {"OneK1K": "ok", "AIDA": "aida", "HFRA": "cg"}

# Every (dataset, donor_count) that currently has a non-'--' entry in the table
TABLE_SETTINGS = {
    "OneK1K": [2, 5, 10, 20, 50, 100, 200],
    "AIDA":   [2, 5, 10, 20, 50, 100, 200],
    "HFRA":   [2, 5, 10, 20],
}

N_TRIALS  = 5
N_WORKERS = 1   # serial to avoid OOM


# ── Worker ────────────────────────────────────────────────────────────────────

def _compute_ari_one(args):
    ds_name, ds_dir, donors, trial = args
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    import scanpy as sc
    from evaluation.utils.sc_metrics import Statistics, filter_low_quality_cells_and_genes

    base     = os.path.join(DATA_ROOT, ds_dir, f"{donors}d", str(trial))
    datasets = os.path.join(base, "datasets")
    csv_path = os.path.join(base, "results", "quality_eval_results", "results", "statistics_evals.csv")

    # Load real data (train donors only)
    train_donors = np.load(os.path.join(datasets, "train.npy"), allow_pickle=True)
    all_data  = sc.read_h5ad(os.path.join(DATA_ROOT, ds_dir, "full_dataset_cleaned.h5ad"), backed='r')
    mask      = all_data.obs["individual"].isin(train_donors)
    real_data = all_data[mask].to_memory()
    all_data.file.close()

    syn_data = sc.read_h5ad(os.path.join(datasets, "synthetic.h5ad"))

    # Filter & align genes (same as SingleCellEvaluator.initialize_datasets)
    real_data = filter_low_quality_cells_and_genes(real_data)
    syn_data  = filter_low_quality_cells_and_genes(syn_data)
    common    = real_data.var_names.intersection(syn_data.var_names)
    real_data = real_data[:, common]
    syn_data  = syn_data[:, common]

    ari, _ = Statistics(random_seed=1).compute_ari(real_data, syn_data, cell_type_col="cell_type")

    # Overwrite only ari_real_vs_syn in the CSV; keep mmd and lisi untouched
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1, f"Expected 1 row in {csv_path}, got {len(rows)}"
    rows[0]["ari_real_vs_syn"] = str(ari)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return ds_name, donors, trial, ari


def _compute_all_metrics_one(args):
    """Worker: compute MMD + LISI + ARI from scratch and write a new statistics_evals.csv."""
    ds_name, ds_dir, donors, trial = args
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    import scanpy as sc
    from evaluation.utils.sc_metrics import Statistics, filter_low_quality_cells_and_genes

    base     = os.path.join(DATA_ROOT, ds_dir, f"{donors}d", str(trial))
    datasets = os.path.join(base, "datasets")
    csv_dir  = os.path.join(base, "results", "quality_eval_results", "results")
    csv_path = os.path.join(csv_dir, "statistics_evals.csv")
    os.makedirs(csv_dir, exist_ok=True)

    # Load real data (train donors only)
    train_donors = np.load(os.path.join(datasets, "train.npy"), allow_pickle=True)
    all_data  = sc.read_h5ad(os.path.join(DATA_ROOT, ds_dir, "full_dataset_cleaned.h5ad"), backed='r')
    mask      = all_data.obs["individual"].isin(train_donors)
    real_data = all_data[mask].to_memory()
    all_data.file.close()

    syn_data = sc.read_h5ad(os.path.join(datasets, "synthetic.h5ad"))

    real_data = filter_low_quality_cells_and_genes(real_data)
    syn_data  = filter_low_quality_cells_and_genes(syn_data)
    common    = real_data.var_names.intersection(syn_data.var_names)
    real_data = real_data[:, common]
    syn_data  = syn_data[:, common]

    stats = Statistics(random_seed=1)
    mmd  = stats.compute_mmd_optimized(real_data, syn_data)
    lisi = stats.compute_lisi(real_data, syn_data)
    ari, _ = stats.compute_ari(real_data, syn_data, cell_type_col="cell_type")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mmd", "lisi", "ari_real_vs_syn"])
        writer.writeheader()
        writer.writerow({"mmd": mmd, "lisi": lisi, "ari_real_vs_syn": ari})

    return ds_name, donors, trial, mmd, lisi, ari


# ── Table printing ────────────────────────────────────────────────────────────

def _read_metrics(ds_dir, donors):
    vals = []
    for trial in range(1, N_TRIALS + 1):
        path = os.path.join(DATA_ROOT, ds_dir, f"{donors}d", str(trial),
                            "results", "quality_eval_results", "results", "statistics_evals.csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                vals.append((float(row["mmd"]), float(row["lisi"]), float(row["ari_real_vs_syn"])))
    return vals

def _fmt(vals, scale=1.0):
    arr = np.array(vals) * scale
    mean, std = arr.mean(), arr.std()
    std_s = f"{std:.2f}".lstrip("0") or "0"
    return f"{mean:.2f} $\\pm$ {std_s}"

def _cell(vals, idx, scale=1.0):
    return _fmt([v[idx] for v in vals], scale) if vals else "--"

# Hardcoded LISI and MMD from the paper (not being recomputed).
# ARI will always be read fresh from updated CSVs.
LISI_MMD = {
    "OneK1K": {
        2:   ("0.91 $\\pm$ .01", "0.08 $\\pm$ .01"),
        5:   ("0.90 $\\pm$ .01", "0.04 $\\pm$ .00"),
        10:  ("0.88 $\\pm$ .01", "0.02 $\\pm$ .00"),
        20:  ("0.88 $\\pm$ .00", "0.01 $\\pm$ .00"),
        50:  ("0.87 $\\pm$ .00", "0.01 $\\pm$ .00"),
        100: ("0.86 $\\pm$ .00", "0.01 $\\pm$ .00"),
        200: ("0.86 $\\pm$ .00", "0.01 $\\pm$ .00"),
    },
    "AIDA": {
        2:   ("0.87 $\\pm$ .01", "0.06 $\\pm$ .01"),
        5:   ("0.83 $\\pm$ .02", "0.02 $\\pm$ .01"),
        10:  ("0.81 $\\pm$ .01", "0.01 $\\pm$ .00"),
        20:  ("0.78 $\\pm$ .02", "0.01 $\\pm$ .00"),
        50:  (None, None),
        100: (None, None),
        200: (None, None),
    },
    "HFRA": {
        2:   ("0.56 $\\pm$ .03", "0.02 $\\pm$ .00"),
        5:   ("0.39 $\\pm$ .08", "0.01 $\\pm$ .00"),
        10:  ("0.28 $\\pm$ .04", "0.01 $\\pm$ .00"),
        20:  (None, None),
    },
}

def print_table():
    rows = {}
    for ds_name, ds_dir in DATASETS.items():
        rows[ds_name] = {}
        for donors in TABLE_SETTINGS[ds_name]:
            vals = _read_metrics(ds_dir, donors)
            lisi_s, mmd_s = LISI_MMD[ds_name].get(donors, (None, None))
            rows[ds_name][donors] = (
                lisi_s if lisi_s else _cell(vals, 1),
                _cell(vals, 2),                          # ARI always fresh
                mmd_s  if mmd_s  else _cell(vals, 0, scale=100),
            )

    def g(ds, d, i):
        r = rows[ds].get(d)
        return "--" if r is None or r[i] is None else r[i]

    print(r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{cccccccccc}
\toprule
& \multicolumn{3}{c}{\textbf{OneK1K}} & \multicolumn{3}{c}{\textbf{AIDA}} & \multicolumn{3}{c}{\textbf{HFRA}} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}
\textbf{Donors} & \textbf{LISI ($\uparrow$)} & \textbf{ARI ($\uparrow$)} & \textbf{MMD ($\times$10$^2$ $\downarrow$)} & \textbf{LISI ($\uparrow$)} & \textbf{ARI ($\uparrow$)} & \textbf{MMD ($\times$10$^2$ $\downarrow$)} & \textbf{LISI ($\uparrow$)} & \textbf{ARI ($\uparrow$)} & \textbf{MMD ($\times$10$^2$ $\downarrow$)} \\
\midrule""")
    for d in [2, 5, 10, 20, 50, 100, 200]:
        cols = []
        for ds in ["OneK1K", "AIDA", "HFRA"]:
            if d not in TABLE_SETTINGS[ds]:
                cols += ["--", "--", "--"]
            else:
                cols += [g(ds, d, 0), g(ds, d, 1), g(ds, d, 2)]
        print(f"{d} & " + " & ".join(cols) + r" \\")
    print(r"""\bottomrule
\end{tabular}
\caption{Quality metrics (LISI, ARI, and MMD) across datasets and donor sizes. Values shown as mean $\pm$ standard deviation. $\uparrow$ indicates higher is better, $\downarrow$ indicates lower is better. MMD values are scaled by 100 for readability. Implementations taken from CAMDA2025 could not execute over larger settings on the AIDA and HFRA data.}
\label{tab:qualities}
\end{table*}""")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Full metrics (MMD + LISI + ARI) for new AIDA settings — no CSV exists yet.
    # Note: AIDA 200d trial 5 synthetic is missing, so only trials 1-4.
    NEW_FULL = [
        # 100d already completed successfully — skipping
        ("AIDA", "aida", 200, 1), ("AIDA", "aida", 200, 2), ("AIDA", "aida", 200, 3),
        ("AIDA", "aida", 200, 4),
    ]

    print(f"Submitting {len(NEW_FULL)} full-metrics jobs (one fresh process each)...\n")

    # Spawn a new process per job so accumulated memory from prior jobs
    # (scanpy/numpy buffers) doesn't push the worker over the memory limit.
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    for job in NEW_FULL:
        ds_name, ds_dir, donors, trial = job
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
            fut = executor.submit(_compute_all_metrics_one, job)
            try:
                _, _, _, mmd, lisi, ari = fut.result()
                print(f"  done  {ds_name:8s} {donors:>3}d  trial {trial}  "
                      f"MMD={mmd:.5f}  LISI={lisi:.3f}  ARI={ari:.4f}")
            except Exception as e:
                print(f"  ERROR {ds_name:8s} {donors:>3}d  trial {trial}  {e}")

    print("\nAll jobs complete. Updated table:\n")
    print_table()

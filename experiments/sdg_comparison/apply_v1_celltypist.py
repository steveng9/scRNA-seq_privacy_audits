"""
apply_v1_celltypist.py

Relabels existing v1 scDiffusion synthetic.h5ad files with CellTypist cell-type
predictions trained on D_train. Saves under scdiffusion_v3/v1_celltypist/.

The expression values come from the v1 model (300k diff steps, 150k VAE steps,
batch=512), which is NOT paper-faithful. Use scdiffusion_v3/faithful/ for paper
results. These outputs are labelled "v1_celltypist" to make the distinction clear.

Operation:
  1. Load existing v1 synthetic.h5ad (raw counts).
  2. Load D_train from the shared splits directory.
  3. Train CellTypist (logistic regression) on log1p-normalized D_train, restricted
     to the same HVG gene set as the v1 synthetic data.
  4. Annotate the log1p-normalized synthetic cells with CellTypist.
  5. Write a new h5ad with updated cell_type labels but original expression values.

Usage
-----
  # Dry-run
  python apply_v1_celltypist.py --dataset ok --nd 50d --trials 1 2 3 4 5 --dry-run

  # Run
  python apply_v1_celltypist.py --dataset ok --nd 50d --trials 1 2 3 4 5
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc

DATA_ROOT      = "/home/golobs/data/scMAMAMIA"
INDIVIDUAL_COL = "individual"
CELL_TYPE_COL  = "cell_type"


def _v1_synth_path(ds, nd_str, trial_str):
    return os.path.join(DATA_ROOT, ds, "scdiffusion", "no_dp", nd_str, trial_str,
                        "datasets", "synthetic.h5ad")


def _splits_dir(ds, nd_str, trial_str):
    return os.path.join(DATA_ROOT, ds, "splits", nd_str, trial_str)


def _out_synth(ds, nd_str, trial_str):
    return os.path.join(DATA_ROOT, ds, "scdiffusion_v3", "v1_celltypist",
                        nd_str, trial_str, "datasets", "synthetic.h5ad")


def _hvg_path(ds):
    p = os.path.join(DATA_ROOT, ds, "hvg_full.csv")
    return p if os.path.exists(p) else os.path.join(DATA_ROOT, ds, "hvg.csv")


def relabel_trial(ds, nd_str, trial_str, dry_run=False):
    label   = f"{ds}/scdiffusion_v3/v1_celltypist/{nd_str}/{trial_str}"
    v1_path = _v1_synth_path(ds, nd_str, trial_str)
    out     = _out_synth(ds, nd_str, trial_str)

    if not os.path.exists(v1_path):
        print(f"  [SKIP no v1 data] {label}  ({v1_path})")
        return False

    if os.path.exists(out):
        print(f"  [SKIP done] {label}")
        return True

    print(f"  [RUN] {label}")
    if dry_run:
        return True

    import celltypist

    # ---- Load v1 synthetic (raw counts) --------------------------------
    synth   = sc.read_h5ad(v1_path)
    synth_genes = list(synth.var_names)

    # ---- Load D_train, restrict to genes present in v1 synthetic -------
    splits   = _splits_dir(ds, nd_str, trial_str)
    train_ids = np.load(os.path.join(splits, "train.npy"), allow_pickle=True)

    full = sc.read_h5ad(os.path.join(DATA_ROOT, ds, "full_dataset_cleaned.h5ad"))
    train = full[full.obs[INDIVIDUAL_COL].isin(train_ids)].copy()

    # Optionally filter to HVGs first (for consistency), then restrict to
    # genes that are actually in the v1 synthetic data.
    hvg = _hvg_path(ds)
    if os.path.exists(hvg):
        hvg_df    = pd.read_csv(hvg, index_col=0)
        hvg_genes = [g for g in hvg_df.index[hvg_df["highly_variable"]]
                     if g in train.var_names]
        train = train[:, hvg_genes].copy()

    common_genes = [g for g in train.var_names if g in synth_genes]
    if not common_genes:
        print(f"  [ERROR] no gene overlap for {label}", file=sys.stderr)
        return False

    train = train[:, common_genes].copy()
    print(f"    {len(common_genes)} common genes  "
          f"({len(synth_genes) - len(common_genes)} in synthetic but not train)")

    # ---- Normalize + log1p for CellTypist ------------------------------
    sc.pp.normalize_total(train, target_sum=1e4)
    sc.pp.log1p(train)

    # ---- Train CellTypist on D_train -----------------------------------
    print(f"    Training CellTypist on {train.n_obs} train cells ...", flush=True)
    model = celltypist.train(train, labels=CELL_TYPE_COL, n_jobs=8,
                             feature_selection=False)

    # ---- Annotate synthetic data (normalize + log1p first) -------------
    synth_ct = synth[:, common_genes].copy()
    sc.pp.normalize_total(synth_ct, target_sum=1e4)
    sc.pp.log1p(synth_ct)
    preds    = celltypist.annotate(synth_ct, model=model, majority_voting=False)
    new_labels = preds.predicted_labels["predicted_labels"].values

    # ---- Save: keep original expression, update cell_type label --------
    synth.obs[CELL_TYPE_COL] = new_labels
    os.makedirs(os.path.dirname(out), exist_ok=True)
    synth.write_h5ad(out, compression="gzip")
    print(f"    Saved -> {out}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ok",
                    help="Dataset name (ok / aida / cg). Default: ok.")
    ap.add_argument("--nd", default="50d",
                    help="Donor-count suffix (10d / 50d / …). Default: 50d.")
    ap.add_argument("--trials", nargs="+", type=str, default=["1", "2", "3", "4", "5"],
                    help="Trial IDs (default: 1 2 3 4 5).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would run without doing it.")
    args = ap.parse_args()

    for t in args.trials:
        relabel_trial(args.dataset, args.nd, str(t), dry_run=args.dry_run)


if __name__ == "__main__":
    main()

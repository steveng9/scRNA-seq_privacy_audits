"""
apply_celltypist_v3.py — generate paper-faithful scDiffusion v3 output from
already-trained v2 model checkpoints.

For each scdiffusion_v2 trial directory where the diffusion backbone has
finished training (model800000.pt or ema_0.9999_800000.pt present), this
script runs:
  1. CellTypist training on D_train (fast, CPU)
  2. Unconditional DDPM generation
  3. CellTypist post-hoc annotation

Output is written to the corresponding scdiffusion_v3/ directory, mirroring
the v2 path structure exactly. Existing v2 model weights are reused (not
re-trained).

Usage
-----
  # Dry-run: show what would be processed
  python apply_celltypist_v3.py --dry-run

  # Run all ready v2 trials → v3 (GPU 0)
  CUDA_VISIBLE_DEVICES=0 python apply_celltypist_v3.py

  # Run only ok dataset
  CUDA_VISIBLE_DEVICES=0 python apply_celltypist_v3.py --dataset ok

  # Run a specific v2 dir
  python apply_celltypist_v3.py --v2-dir /path/to/scdiffusion_v2/no_dp/10d/1
"""

import argparse
import os
import re
import subprocess
import sys

DATA_ROOT    = "/home/golobs/data/scMAMAMIA"
REPO_ROOT    = "/home/golobs/scRNA-seq_privacy_audits"
GEN_PY       = os.path.join(REPO_ROOT, "experiments", "sdg_comparison", "generate_trial.py")
CONDA_ENV    = "scdiff_"
DIFF_STEPS   = 800000


def _find_v2_dirs(dataset_filter=None):
    """Yield (dataset, nd, trial, v2_dir) for all scdiffusion_v2 trial dirs."""
    for ds in sorted(os.listdir(DATA_ROOT)):
        if dataset_filter and ds != dataset_filter:
            continue
        v2_root = os.path.join(DATA_ROOT, ds, "scdiffusion_v2")
        if not os.path.isdir(v2_root):
            continue
        for dp_variant in sorted(os.listdir(v2_root)):          # no_dp / eps_*
            dp_dir = os.path.join(v2_root, dp_variant)
            if not os.path.isdir(dp_dir):
                continue
            for nd_str in sorted(os.listdir(dp_dir)):           # 10d / 50d / …
                nd_dir = os.path.join(dp_dir, nd_str)
                if not os.path.isdir(nd_dir):
                    continue
                for trial_str in sorted(os.listdir(nd_dir)):    # 1 / 2 / …
                    v2_dir = os.path.join(nd_dir, trial_str)
                    if os.path.isdir(v2_dir):
                        yield ds, dp_variant, nd_str, trial_str, v2_dir


def _diffusion_complete(v2_dir):
    """True if the diffusion backbone has a checkpoint at DIFF_STEPS."""
    diff_subdir = os.path.join(v2_dir, "models", "diff", "diffusion")
    if not os.path.isdir(diff_subdir):
        return False
    target_raw = os.path.join(diff_subdir, f"model{DIFF_STEPS:06d}.pt")
    target_ema = os.path.join(diff_subdir, f"ema_0.9999_{DIFF_STEPS:06d}.pt")
    return os.path.exists(target_raw) or os.path.exists(target_ema)


def _v3_dir(ds, dp_variant, nd_str, trial_str):
    return os.path.join(DATA_ROOT, ds, "scdiffusion_v3", dp_variant, nd_str, trial_str)


def _dataset_h5ad(ds):
    return os.path.join(DATA_ROOT, ds, "full_dataset_cleaned.h5ad")


def _splits_dir(ds, nd_str, trial_str):
    return os.path.join(DATA_ROOT, ds, "splits", nd_str, trial_str)


def _hvg_path(ds):
    p = os.path.join(DATA_ROOT, ds, "hvg_full.csv")
    if os.path.exists(p):
        return p
    return os.path.join(DATA_ROOT, ds, "hvg.csv")


def run_v3_from_v2(ds, dp_variant, nd_str, trial_str, v2_dir, dry_run=False):
    v3_out   = _v3_dir(ds, dp_variant, nd_str, trial_str)
    synth    = os.path.join(v3_out, "datasets", "synthetic.h5ad")

    label = f"{ds}/{dp_variant}/{nd_str}/{trial_str}"

    if os.path.exists(synth):
        print(f"  [SKIP already done] {label}")
        return True

    if not _diffusion_complete(v2_dir):
        print(f"  [SKIP diffusion incomplete] {label}")
        return False

    h5ad     = _dataset_h5ad(ds)
    splits   = _splits_dir(ds, nd_str, trial_str)
    hvg      = _hvg_path(ds)

    cmd = [
        "conda", "run", "--no-capture-output", "-n", "tabddpm_",
        "python", GEN_PY,
        "--generator",       "scdiffusion_v3",
        "--dataset",         h5ad,
        "--splits-dir",      splits,
        "--out-dir",         v3_out,
        "--hvg-path",        hvg,
        "--conda-env",       CONDA_ENV,
        "--model-source-dir", v2_dir,
        "--scd-batch-size",  "128",
    ]

    print(f"  [RUN] {label}  ->  {v3_out}")
    if dry_run:
        print(f"    CMD: {' '.join(cmd)}")
        return True

    os.makedirs(v3_out, exist_ok=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  [FAILED] {label}  exit={result.returncode}", file=sys.stderr)
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None,
                    help="Restrict to this dataset (ok / aida / cg). Default: all.")
    ap.add_argument("--v2-dir", default=None,
                    help="Process only this specific v2 trial directory.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be done without running anything.")
    args = ap.parse_args()

    if args.v2_dir:
        # Parse ds/dp_variant/nd/trial from the path
        parts = os.path.normpath(args.v2_dir).split(os.sep)
        # Expect: …/scMAMAMIA/<ds>/scdiffusion_v2/<dp>/<nd>/<trial>
        try:
            scdf_idx  = parts.index("scdiffusion_v2")
            ds        = parts[scdf_idx - 1]
            dp_variant = parts[scdf_idx + 1]
            nd_str     = parts[scdf_idx + 2]
            trial_str  = parts[scdf_idx + 3]
        except (ValueError, IndexError):
            print("ERROR: could not parse dataset/dp/nd/trial from --v2-dir path",
                  file=sys.stderr)
            sys.exit(1)
        run_v3_from_v2(ds, dp_variant, nd_str, trial_str, args.v2_dir,
                       dry_run=args.dry_run)
        return

    n_run = n_skip_done = n_skip_incomplete = 0
    for ds, dp_variant, nd_str, trial_str, v2_dir in _find_v2_dirs(args.dataset):
        synth = os.path.join(_v3_dir(ds, dp_variant, nd_str, trial_str),
                             "datasets", "synthetic.h5ad")
        if os.path.exists(synth):
            n_skip_done += 1
            continue
        if not _diffusion_complete(v2_dir):
            n_skip_incomplete += 1
            label = f"{ds}/{dp_variant}/{nd_str}/{trial_str}"
            print(f"  [SKIP diffusion incomplete] {label}")
            continue
        ok = run_v3_from_v2(ds, dp_variant, nd_str, trial_str, v2_dir,
                             dry_run=args.dry_run)
        if ok:
            n_run += 1

    print(f"\nDone.  run={n_run}  skip_done={n_skip_done}  "
          f"skip_incomplete={n_skip_incomplete}")


if __name__ == "__main__":
    main()

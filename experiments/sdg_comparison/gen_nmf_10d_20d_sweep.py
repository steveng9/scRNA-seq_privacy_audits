"""
gen_nmf_10d_20d_sweep.py — Generate NMF synthetic data for 10d and 20d,
no_dp and DP sweep through epsilon=10^5, for OneK1K (ok).

Runs serially.  Sub-epsilons for DP are allocated proportionally to the
CAMDA original ratios (eps_nmf=0.5, eps_kmeans=2.1, eps_summaries=0.2;
total=2.8) so that the three stages sum exactly to the sweep epsilon.
The special value eps=2.8 uses the CAMDA defaults directly.

Configs generated (nd ∈ {10, 20}, trials 1-5 each):
  no_dp
  eps=10^0  (1)
  eps=2.8   (CAMDA DP)
  eps=10^1  (10)
  eps=10^2  (100)
  eps=10^3  (1000)
  eps=10^4  (10000)
  eps=10^5  (100000)

Output: /home/golobs/data/scMAMAMIA/ok/nmf/{variant}/{nd}d/{trial}/

Usage:
    python experiments/sdg_comparison/gen_nmf_10d_20d_sweep.py [--dry-run]
"""

import argparse
import os
import subprocess
import sys

REPO       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA       = "/home/golobs/data/scMAMAMIA"
DATASET    = f"{DATA}/ok/full_dataset_cleaned.h5ad"
HVG_PATH   = f"{DATA}/ok/hvg_full.csv"
SPLITS_BASE = f"{DATA}/ok/splits"
OUT_BASE   = f"{DATA}/ok/nmf"
CONDA_ENV  = "nmf_"
GENERATE   = os.path.join(REPO, "experiments", "sdg_comparison", "generate_trial.py")

N_DONORS_LIST = [10, 20]
TRIALS        = [1, 2, 3, 4, 5]

# Each entry: (variant_name, dp_mode, eps_total_or_None)
# eps_total=None → no_dp; eps_total=2.8 → CAMDA fixed ratios; otherwise proportional.
CONFIGS = [
    ("no_dp",     "none",   None),
    ("eps_1",     "all",    1),
    ("eps_2.8",   "all",    2.8),
    ("eps_10",    "all",    10),
    ("eps_100",   "all",    100),
    ("eps_1000",  "all",    1000),
    ("eps_10000", "all",    10000),
    ("eps_100000","all",    100000),
]

_CAMDA_TOTAL = 2.8
_CAMDA_NMF       = 0.5
_CAMDA_KMEANS    = 2.1
_CAMDA_SUMMARIES = 0.2


def sub_epsilons(eps_total):
    """Return (eps_nmf, eps_kmeans, eps_summaries) proportional to CAMDA ratios."""
    return (
        eps_total * _CAMDA_NMF       / _CAMDA_TOTAL,
        eps_total * _CAMDA_KMEANS    / _CAMDA_TOTAL,
        eps_total * _CAMDA_SUMMARIES / _CAMDA_TOTAL,
    )


def run_job(variant, dp_mode, eps_total, nd, trial, dry_run=False):
    out_dir  = os.path.join(OUT_BASE, variant, f"{nd}d", str(trial))
    synth_fp = os.path.join(out_dir, "datasets", "synthetic.h5ad")
    splits_dir = os.path.join(SPLITS_BASE, f"{nd}d", str(trial))

    if os.path.exists(synth_fp):
        print(f"  [SKIP] {variant}/{nd}d/t{trial} — already exists", flush=True)
        return True

    if not os.path.isdir(splits_dir):
        print(f"  [SKIP] splits missing: {splits_dir}", flush=True)
        return False

    cmd = [
        "conda", "run", "--no-capture-output", "-n", CONDA_ENV,
        "python", GENERATE,
        "--generator", "nmf",
        "--dataset", DATASET,
        "--splits-dir", splits_dir,
        "--out-dir", out_dir,
        "--hvg-path", HVG_PATH,
        "--conda-env", CONDA_ENV,
        "--dp-mode", dp_mode,
    ]

    if eps_total is not None:
        e_nmf, e_km, e_sum = sub_epsilons(eps_total)
        cmd += [
            "--dp-eps-nmf",       str(e_nmf),
            "--dp-eps-kmeans",    str(e_km),
            "--dp-eps-summaries", str(e_sum),
        ]

    print(f"\n{'='*60}", flush=True)
    print(f"  {variant}/{nd}d/t{trial}  →  {out_dir}", flush=True)

    if dry_run:
        print(f"  DRY-RUN: {' '.join(cmd)}", flush=True)
        return True

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  [ERROR] {variant}/{nd}d/t{trial} exited {result.returncode}", flush=True)
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total = len(CONFIGS) * len(N_DONORS_LIST) * len(TRIALS)
    done  = 0
    print(f"NMF 10d/20d sweep: {len(CONFIGS)} configs × "
          f"{len(N_DONORS_LIST)} nd × {len(TRIALS)} trials = {total} jobs",
          flush=True)

    for variant, dp_mode, eps_total in CONFIGS:
        for nd in N_DONORS_LIST:
            for trial in TRIALS:
                run_job(variant, dp_mode, eps_total, nd, trial, dry_run=args.dry_run)
                done += 1
                print(f"  Progress: {done}/{total}", flush=True)

    print("\nAll generation done.", flush=True)


if __name__ == "__main__":
    main()

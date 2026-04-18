"""
gen_nmf_dp_sweep.py — Generate NMF synthetic data sweeping DP epsilon values.

Runs serially (one job at a time) to avoid excessive CPU contention.
Each job generates one trial at one epsilon value.

Priority order for epsilons (even powers of 10 first so results are useful
even if killed early):
  10^0, 10^2, 10^4, 10^6, 10^8  →  then odd powers: 10^1, 10^3, 10^5, 10^7

Per-stage DP budgets are allocated proportionally to the CAMDA original ratios
(eps_nmf=0.5, eps_kmeans=2.1, eps_summaries=0.2; total=2.8), so that the sum
of the three sub-epsilons equals the sweep epsilon exactly.

Output structure: /home/golobs/data/scMAMAMIA/ok/nmf/eps_{e}/{nd}d/{trial}/

Usage:
    python experiments/sdg_comparison/gen_nmf_dp_sweep.py [--dry-run]
"""

import argparse
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = "/home/golobs/data/scMAMAMIA"

DATASET_PATH = f"{DATA}/ok/full_dataset_cleaned.h5ad"
HVG_PATH     = f"{DATA}/ok/hvg_full.csv"
SPLITS_BASE  = f"{DATA}/ok/scdesign2/no_dp"
OUT_BASE     = f"{DATA}/ok/nmf"
CONDA_ENV    = "nmf_"

N_DONORS  = 50
TRIALS    = [1, 2, 3, 4, 5]

# Even powers of 10 first for maximum coverage if killed early
EPSILONS_PRIORITY = [
    1,          # 10^0
    100,        # 10^2
    10_000,     # 10^4
    1_000_000,  # 10^6
    100_000_000, # 10^8
    10,         # 10^1
    1_000,      # 10^3
    100_000,    # 10^5
    10_000_000, # 10^7
]

GENERATE_SCRIPT = os.path.join(REPO, "experiments", "sdg_comparison", "generate_trial.py")


def run_job(eps, trial, dry_run=False):
    out_dir   = os.path.join(OUT_BASE, f"eps_{eps}", f"{N_DONORS}d", str(trial))
    synth_out = os.path.join(out_dir, "datasets", "synthetic.h5ad")
    splits_dir = os.path.join(SPLITS_BASE, f"{N_DONORS}d", str(trial), "datasets")

    if os.path.exists(synth_out):
        print(f"  [SKIP] eps={eps} trial={trial} — already exists", flush=True)
        return True

    if not os.path.isdir(splits_dir):
        print(f"  [SKIP] splits missing: {splits_dir}", flush=True)
        return False

    # Distribute epsilon proportionally to CAMDA original allocation
    # (0.5 : 2.1 : 0.2, total = 2.8) so that sum of sub-epsilons = eps exactly.
    _total = 2.8
    eps_nmf       = eps * 0.5 / _total
    eps_kmeans    = eps * 2.1 / _total
    eps_summaries = eps * 0.2 / _total

    cmd = [
        "conda", "run", "--no-capture-output", "-n", CONDA_ENV,
        "python", GENERATE_SCRIPT,
        "--generator", "nmf",
        "--dataset", DATASET_PATH,
        "--splits-dir", splits_dir,
        "--out-dir", out_dir,
        "--hvg-path", HVG_PATH,
        "--conda-env", CONDA_ENV,
        "--dp-mode", "all",
        "--dp-eps-nmf",       str(eps_nmf),
        "--dp-eps-kmeans",    str(eps_kmeans),
        "--dp-eps-summaries", str(eps_summaries),
    ]

    print(f"\n{'='*60}", flush=True)
    print(f"  eps={eps}  trial={trial}  →  {out_dir}", flush=True)
    print(f"  $ {' '.join(cmd)}", flush=True)

    if dry_run:
        return True

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  [ERROR] eps={eps} trial={trial} exited {result.returncode}", flush=True)
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total = len(EPSILONS_PRIORITY) * len(TRIALS)
    done  = 0
    print(f"NMF DP sweep: {len(EPSILONS_PRIORITY)} epsilons × {len(TRIALS)} trials = {total} jobs",
          flush=True)
    print(f"Dataset: ok  N_donors={N_DONORS}  conda_env={CONDA_ENV}", flush=True)

    for eps in EPSILONS_PRIORITY:
        for trial in TRIALS:
            run_job(eps, trial, dry_run=args.dry_run)
            done += 1
            print(f"  Progress: {done}/{total}", flush=True)

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()

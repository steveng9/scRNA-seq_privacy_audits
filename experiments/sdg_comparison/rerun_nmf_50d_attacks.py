#!/usr/bin/env python3
"""
rerun_nmf_50d_attacks.py — Re-run BB_quad attacks for NMF 50d, preserving originals.

Backs up existing results to *_orig.csv, resets tracking.csv, then runs all quad
attacks via run_mia_sweep.py --sdg nmf --nd 50.

Target variants (no_dp + DP sweep up to eps_100000):
  ok/nmf/{no_dp, eps_1, eps_2.8, eps_10, eps_100, eps_1000, eps_10000, eps_100000}
  50 donors, trials 1-5

Higher-epsilon variants (eps_1000000+) are not backed up, so the sweep skips them
as "already done".

Launch (detached from SSH):
    nohup conda run --no-capture-output -n tabddpm_ \\
        python experiments/sdg_comparison/rerun_nmf_50d_attacks.py \\
        > /tmp/nmf_50d_rerun.log 2>&1 &
    echo $!

Monitor:
    tail -f /tmp/nmf_50d_rerun.log
    grep -E '^(\\[|=== |DONE|FAIL)' /tmp/nmf_50d_rerun.log
"""

import os
import sys
import shutil
import subprocess
import datetime

import pandas as pd

DATA_ROOT = "/home/golobs/data/scMAMAMIA"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MIA_SWEEP = os.path.join(REPO_ROOT, "experiments", "sdg_comparison", "run_mia_sweep.py")

ND = 50
N_TRIALS = 5

# Only back up and re-run these variants; higher epsilons are left untouched.
TARGET_VARIANTS = [
    "no_dp",
    "eps_1",
    "eps_2.8",
    "eps_10",
    "eps_100",
    "eps_1000",
    "eps_10000",
    "eps_100000",
]

RESULT_FILES = [
    "mamamia_results.csv",
    "mamamia_results_classb.csv",
    "mamamia_all_scores.csv",
    "mamamia_all_scores_classb.csv",
]

# Columns to reset so run_experiment.py re-runs each trial from scratch.
TRACKING_COLS_TO_RESET = ["tm:100", "tm:101", "classb:100", "classb:101"]


def ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def backup_and_reset():
    print(f"\n[{ts()}] === Phase 1: Backup results and reset tracking ===", flush=True)
    for variant in TARGET_VARIANTS:
        base = os.path.join(DATA_ROOT, "ok", "nmf", variant, f"{ND}d")

        # Backup per-trial result CSVs
        for trial in range(1, N_TRIALS + 1):
            results_dir = os.path.join(base, str(trial), "results")
            if not os.path.isdir(results_dir):
                print(f"  [SKIP] {variant}/t{trial}: no results dir", flush=True)
                continue
            for fname in RESULT_FILES:
                src = os.path.join(results_dir, fname)
                dst = os.path.join(results_dir, fname.replace(".csv", "_orig.csv"))
                if os.path.exists(src):
                    shutil.move(src, dst)
                    print(f"  moved: {variant}/t{trial}/{fname} → {fname.replace('.csv','_orig.csv')}",
                          flush=True)

        # Reset tracking so run_experiment.py treats all 5 trials as pending
        tracking_path = os.path.join(base, "tracking.csv")
        if not os.path.exists(tracking_path):
            print(f"  [SKIP] {variant}: no tracking.csv at {tracking_path}", flush=True)
            continue
        df = pd.read_csv(tracking_path)
        changed = False
        for col in TRACKING_COLS_TO_RESET:
            if col in df.columns:
                df[col] = 0
                changed = True
        if changed:
            df.to_csv(tracking_path, index=False)
            print(f"  reset tracking: {variant} ({', '.join(TRACKING_COLS_TO_RESET)})",
                  flush=True)

    print(f"[{ts()}] Phase 1 complete.\n", flush=True)


def run_attacks():
    print(f"[{ts()}] === Phase 2: Running BB_quad attacks via run_mia_sweep.py ===",
          flush=True)
    cmd = [sys.executable, MIA_SWEEP, "--sdg", "nmf", "--nd", str(ND)]
    print(f"  cmd: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if rc == 0:
        print(f"\n[{ts()}] Phase 2 complete (rc=0).", flush=True)
    else:
        print(f"\n[{ts()}] Phase 2 finished with rc={rc}.", flush=True)
    return rc


def main():
    print(f"\nrerun_nmf_50d_attacks.py  started {datetime.datetime.now()}", flush=True)
    print(f"  target: ok/nmf/{{no_dp,eps_1,eps_2.8,...,eps_100000}}/{ND}d  "
          f"(trials 1-{N_TRIALS})", flush=True)

    backup_and_reset()
    rc = run_attacks()

    print(f"\nrerun_nmf_50d_attacks.py  finished {datetime.datetime.now()}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()

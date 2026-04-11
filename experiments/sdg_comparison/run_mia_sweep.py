"""
run_mia_sweep.py — Sweep scMAMA-MIA attacks across all SDG methods and datasets.

This script runs the scMAMA-MIA attack against all synthetic datasets that have
been generated, covering:

  scDesign2        (ok, aida, cg)   :  WB+aux, WB-aux, BB+aux, BB-aux
  scDesign2+DP     (ok_dp/eps_*)    :  BB+aux, BB-aux
  scDesign3-Gauss  (ok_sd3g)        :  BB+aux, BB-aux  (scDesign2 copula as proxy)
  scDesign3-Vine   (ok_sd3v)        :  BB+aux, BB-aux  (scDesign2 copula as proxy)
  scVI             (ok_scvi, aida_scvi)     :  BB+aux, BB-aux
  scDiffusion      (ok_scdiff, aida_scdiff) :  BB+aux, BB-aux

For non-scDesign2 generators the attack is run in black-box mode: scDesign2 is
fitted as a proxy shadow model on the existing synthetic.h5ad.

Setup (run once)
----------------
  Creates symlinks full_dataset_cleaned.h5ad and hvg.csv in SDG-specific
  directories (e.g. ok_sd3g/) pointing back to their base dataset (ok/).
  For DP epsilon subdirectories (ok_dp/eps_*) which lack donor splits,
  pre-populates holdout.npy and auxiliary.npy from the remaining donor pool.

Usage
-----
  python experiments/sdg_comparison/run_mia_sweep.py [options]

  --dry-run           Print all pending jobs without running them
  --only-setup        Run setup steps only; do not launch any attacks
  --status            Print completion status table and exit
  --dataset DATASET   Only process matching dataset_name (substring match)
  --sdg SDG           Only process matching sdg_key (substring match)
  --nd N              Only process this donor count (integer)
  --max-jobs N        Stop after submitting N jobs (default: unlimited)
"""

import argparse
import os
import sys
import subprocess
import time

import numpy as np
import pandas as pd
import anndata as ad
import yaml
import tempfile

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
DATA_DIR  = "/home/golobs/data"
RUNNER    = os.path.join(SRC_DIR, "run_experiment.py")
N_TRIALS  = 5

# ---------------------------------------------------------------------------
# Threat model definitions
# ---------------------------------------------------------------------------
# (label, code, white_box, use_wb_hvgs, use_aux)
TM_FULL = [
    ("WB+aux", "000", True,  True, True),
    ("WB-aux", "001", True,  True, False),
    ("BB+aux", "100", False, True, True),
    ("BB-aux", "101", False, True, False),
]
TM_BB = [
    ("BB+aux", "100", False, True, True),
    ("BB-aux", "101", False, True, False),
]

# ---------------------------------------------------------------------------
# Sweep definition
# ---------------------------------------------------------------------------
# (sdg_key, dataset_name, base_dataset_name, donor_counts, threat_models, parallel_workers)
#
# sdg_key          — short identifier used for filtering
# dataset_name     — used as dataset_name in the YAML config; determines where
#                    run_experiment.py reads/writes data
# base_dataset_name — base dataset whose full_dataset_cleaned.h5ad and hvg.csv
#                    the SDG-specific directory should symlink to
# donor_counts     — list of training donor set sizes
# threat_models    — TM_FULL or TM_BB
# parallel_workers — number of parallel cell-type workers inside run_experiment.py

SWEEP = [
    # --- scDesign2 (full threat model coverage) ---
    ("sd2",    "ok",         "ok",   [2, 5, 10, 20, 50, 100, 200], TM_FULL, 4),
    ("sd2",    "aida",       "aida", [5, 10, 20, 50, 100, 200],    TM_FULL, 4),
    ("sd2",    "cg",         "cg",   [2, 5, 10, 20],               TM_FULL, 4),

    # --- scDesign2 + DP (BB only; synthetic data already generated) ---
    ("sd2_dp", "ok_dp/eps_1",     "ok", [10, 20, 50], TM_BB, 4),
    ("sd2_dp", "ok_dp/eps_10",    "ok", [10, 20, 50], TM_BB, 4),
    ("sd2_dp", "ok_dp/eps_100",   "ok", [10, 20, 50], TM_BB, 4),
    ("sd2_dp", "ok_dp/eps_1000",  "ok", [10, 20, 50], TM_BB, 4),
    ("sd2_dp", "ok_dp/eps_10000", "ok", [10, 20, 50], TM_BB, 4),

    # --- scDesign3-Gauss (BB only; skip aida — no data generated yet) ---
    ("sd3g",   "ok_sd3g",    "ok",   [2, 5, 10, 20, 50, 100, 200], TM_BB, 4),

    # --- scDesign3-Vine (BB only; 100d not yet generated) ---
    ("sd3v",   "ok_sd3v",    "ok",   [10, 20, 50],                  TM_BB, 4),

    # --- scVI (BB only) ---
    ("scvi",   "ok_scvi",    "ok",   [5, 10, 20, 50, 100],          TM_BB, 4),
    ("scvi",   "aida_scvi",  "aida", [10, 20, 50],                   TM_BB, 4),

    # --- scDiffusion (BB only) ---
    ("scdiff", "ok_scdiff",  "ok",   [10, 20, 50],                   TM_BB, 4),
    ("scdiff", "aida_scdiff","aida", [20, 50],                        TM_BB, 4),
]

# Standard scMAMA-MIA hyper-parameters (matches existing experiments)
MAMAMIA_PARAMS = {
    "IMPORTANCE_OF_CLASS_B_FPs": 0.17,
    "epsilon":                   0.0001,
    "mahalanobis":               True,
    "uniform_remapping_fn":      "zinb_cdf",
    "lin_alg_inverse_fn":        "pinv_gpu",
    "closeness_to_correlation_fn": "closeness_to_correlation_1",
}
MIN_AUX_DONORS = 10


# ===========================================================================
# Setup helpers
# ===========================================================================

def _make_symlink(src, dst):
    """Create symlink dst → src.  No-op if dst already exists (or is a valid link)."""
    if os.path.lexists(dst):
        return  # already exists (or broken link — leave it)
    os.symlink(src, dst)
    print(f"  symlink: {dst} → {src}")


def setup_symlinks():
    """
    For each SDG-specific data directory that shares a donor pool with a base
    dataset (e.g. ok_sd3g uses ok's donors), create symlinks so that
    run_experiment.py can find full_dataset_cleaned.h5ad and hvg.csv.
    """
    print("\n[SETUP] Creating symlinks…")
    seen = set()  # avoid duplicate work for the same (sdg_dir, base_dir) pair

    for sdg_key, dataset_name, base_dataset, donor_counts, tms, pw in SWEEP:
        sdg_path  = os.path.join(DATA_DIR, *dataset_name.split("/"))
        base_path = os.path.join(DATA_DIR, base_dataset)

        if not os.path.isdir(sdg_path):
            continue
        if (sdg_path, base_path) in seen:
            continue
        seen.add((sdg_path, base_path))

        # If the SDG dir IS the base dir (scDesign2 proper), nothing to do.
        if os.path.abspath(sdg_path) == os.path.abspath(base_path):
            continue

        h5ad_src = os.path.relpath(
            os.path.join(base_path, "full_dataset_cleaned.h5ad"), sdg_path
        )
        hvg_src = os.path.relpath(
            os.path.join(DATA_DIR, "ok", "hvg.csv")
            if base_dataset.startswith("ok")
            else os.path.join(base_path, "hvg.csv"),
            sdg_path,
        )

        _make_symlink(h5ad_src, os.path.join(sdg_path, "full_dataset_cleaned.h5ad"))
        _make_symlink(hvg_src,  os.path.join(sdg_path, "hvg.csv"))

    print("[SETUP] Symlinks done.")


def setup_dp_splits():
    """
    DP trial directories only contain synthetic.h5ad + train.npy.
    Pre-populate holdout.npy and auxiliary.npy from the remaining donor pool
    so that run_experiment.py can skip re-sampling (which would overwrite train.npy).

    The full_dataset_cleaned.h5ad lives inside each eps subdirectory (as a
    symlink to ok/full_dataset_cleaned.h5ad), not at the ok_dp top level.
    """
    print("\n[SETUP] Pre-populating DP donor splits…")

    # Lazy-load the full donor list once (use the first eps dir that has the h5ad)
    _all_donors_cache = {}

    for sdg_key, dataset_name, base_dataset, donor_counts, tms, pw in SWEEP:
        if sdg_key != "sd2_dp":
            continue

        sdg_path = os.path.join(DATA_DIR, *dataset_name.split("/"))

        for nd in donor_counts:
            for trial in range(1, N_TRIALS + 1):
                datasets_dir = os.path.join(sdg_path, f"{nd}d", str(trial), "datasets")
                train_npy    = os.path.join(datasets_dir, "train.npy")
                holdout_npy  = os.path.join(datasets_dir, "holdout.npy")
                aux_npy      = os.path.join(datasets_dir, "auxiliary.npy")

                if not os.path.exists(train_npy):
                    continue
                if os.path.exists(holdout_npy) and os.path.exists(aux_npy):
                    continue  # already done

                # Load all donors lazily — use the h5ad symlink inside the eps dir
                if dataset_name not in _all_donors_cache:
                    eps_h5ad = os.path.join(sdg_path, "full_dataset_cleaned.h5ad")
                    if not os.path.exists(eps_h5ad):
                        print(f"  [WARN] {eps_h5ad} not found; skipping DP splits for {dataset_name}.")
                        break
                    print(f"  Loading donor list from {eps_h5ad} …")
                    adata = ad.read_h5ad(eps_h5ad, backed="r")
                    _all_donors_cache[dataset_name] = adata.obs["individual"].unique()
                    adata.file.close()
                all_donors = _all_donors_cache[dataset_name]

                train_donors = np.load(train_npy, allow_pickle=True)
                non_target   = list(set(all_donors) - set(train_donors))

                rng = np.random.default_rng(seed=trial * 1000 + nd)

                # Holdout: same size as train, sampled from non-target pool
                n = len(train_donors)
                non_target_perm = rng.permutation(non_target)
                holdout_donors  = non_target_perm[:n]

                # Auxiliary: prioritise non-target donors, fall back to target
                not_holdout   = list(set(non_target) - set(holdout_donors))
                n_aux         = max(MIN_AUX_DONORS, n)
                target_all    = list(train_donors) + list(holdout_donors)
                aux_pool      = np.concatenate([
                    rng.permutation(not_holdout),
                    rng.permutation(target_all),
                ])
                aux_donors = aux_pool[:n_aux]

                np.save(holdout_npy, holdout_donors, allow_pickle=True)
                np.save(aux_npy,     aux_donors,     allow_pickle=True)
                print(f"  pre-populated splits: {dataset_name}/{nd}d/{trial}")

    print("[SETUP] DP donor splits done.")


# ===========================================================================
# Completion checking
# ===========================================================================

def get_completed_tm_codes(data_dir, nd, trial):
    """Return set of threat model codes with a valid AUC in mamamia_results.csv."""
    results_file = os.path.join(data_dir, f"{nd}d", str(trial), "results", "mamamia_results.csv")
    if not os.path.exists(results_file):
        return set()
    try:
        df = pd.read_csv(results_file)
        auc_row = df[df["metric"] == "auc"]
        if auc_row.empty:
            return set()
        done = set()
        for col in auc_row.columns:
            if col.startswith("tm:"):
                val = auc_row[col].values[0]
                if pd.notna(val):
                    done.add(col[3:])
        return done
    except Exception:
        return set()


def count_done(data_dir, nd, tm_code):
    """Count trials that have a valid AUC for `tm_code`."""
    return sum(
        1 for t in range(1, N_TRIALS + 1)
        if tm_code in get_completed_tm_codes(data_dir, nd, t)
    )


def synth_exists(data_dir, nd, trial):
    """True if synthetic.h5ad exists for the given trial."""
    path = os.path.join(data_dir, f"{nd}d", str(trial), "datasets", "synthetic.h5ad")
    return os.path.exists(path)


def n_synth_available(data_dir, nd):
    """Count trials for which synthetic.h5ad already exists."""
    return sum(1 for t in range(1, N_TRIALS + 1) if synth_exists(data_dir, nd, t))


# ===========================================================================
# Config generation
# ===========================================================================

def write_config(dataset_name, nd, white_box, use_wb_hvgs, use_aux, parallel_workers, cfg_dir):
    """
    Write a run_experiment.py config YAML to cfg_dir and return its path.
    Both local and server dir_list entries point to the server paths so the
    config works without the legacy 'T' flag.
    """
    tm_parts = f"{'wb' if white_box else 'bb'}_{'aux' if use_aux else 'noaux'}"
    cfg_name  = f"{nd}d_{tm_parts}.yaml"
    cfg_path  = os.path.join(cfg_dir, cfg_name)

    cfg = {
        "dir_list": {
            "local":  {"home": REPO_ROOT, "data": DATA_DIR},
            "server": {"home": REPO_ROOT, "data": DATA_DIR},
        },
        "dataset_name": dataset_name,
        "generator_name": "scdesign2",   # always scdesign2 as shadow model
        "plot_results":   False,
        "parallelize":    True,
        "parallel_workers": parallel_workers,
        "min_aux_donors": MIN_AUX_DONORS,
        "mamamia_params": dict(MAMAMIA_PARAMS),
        "mia_setting": {
            "sample_donors_strategy_fn": "sample_donors_strategy_2",
            "num_donors":    nd,
            "white_box":     white_box,
            "use_wb_hvgs":   use_wb_hvgs,
            "use_aux":       use_aux,
        },
    }

    os.makedirs(cfg_dir, exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return cfg_path


# ===========================================================================
# Job execution
# ===========================================================================

def run_job(config_path, dry_run=False):
    """Run run_experiment.py with the given config.  Returns True on success."""
    cmd = [sys.executable, RUNNER, config_path]
    if dry_run:
        print(f"  [DRY-RUN] {' '.join(cmd)}")
        return True
    print(f"  Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode == 0


# ===========================================================================
# Status display
# ===========================================================================

def print_status():
    """Print a summary table of current attack completion across all sweep entries."""
    print(f"\n{'=' * 80}")
    print(f"  scMAMA-MIA Attack Completion Status")
    print(f"{'=' * 80}\n")

    for sdg_key, dataset_name, base_dataset, donor_counts, tms, pw in SWEEP:
        data_dir = os.path.join(DATA_DIR, *dataset_name.split("/"))
        tm_codes = [tm[1] for tm in tms]
        labels   = [tm[0] for tm in tms]

        print(f"  [{sdg_key}] {dataset_name}")
        header   = f"    {'nd':>6}  " + "  ".join(f"{l:>8}" for l in labels)
        print(header)

        for nd in donor_counts:
            parts = []
            for tm_code in tm_codes:
                done  = count_done(data_dir, nd, tm_code)
                avail = n_synth_available(data_dir, nd) if sdg_key != "sd2" else N_TRIALS
                sym   = "✓" if done == N_TRIALS else (f"~{done}" if done > 0 else "·")
                parts.append(f"{sym:>8}")
            print(f"    {nd:>4}d  " + "  ".join(parts))

        print()

    print(f"  ✓ = all {N_TRIALS} trials done   ~N = N done   · = none\n")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print pending jobs without running them")
    parser.add_argument("--only-setup",  action="store_true",
                        help="Run setup steps only (symlinks + DP donor splits)")
    parser.add_argument("--status",      action="store_true",
                        help="Print completion status and exit")
    parser.add_argument("--dataset",     default=None,
                        help="Filter: only process dataset_names containing this substring")
    parser.add_argument("--sdg",         default=None,
                        help="Filter: only process sdg_keys containing this substring")
    parser.add_argument("--nd",          type=int, default=None,
                        help="Filter: only process this donor count")
    parser.add_argument("--max-jobs",    type=int, default=0,
                        help="Stop after running this many jobs (0 = unlimited)")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    # --- Setup ---------------------------------------------------------------
    setup_symlinks()
    setup_dp_splits()

    if args.only_setup:
        print("\nSetup complete. Exiting (--only-setup).")
        return

    # --- Attack sweep --------------------------------------------------------
    jobs_run = 0
    jobs_skipped = 0

    # Write configs to a temp directory within the repo so they survive reruns.
    cfg_root = os.path.join(REPO_ROOT, "experiments", "sdg_comparison", "_sweep_cfgs")
    os.makedirs(cfg_root, exist_ok=True)

    for sdg_key, dataset_name, base_dataset, donor_counts, tms, pw in SWEEP:
        # Apply filters
        if args.sdg     and args.sdg     not in sdg_key:      continue
        if args.dataset and args.dataset not in dataset_name:  continue

        data_dir = os.path.join(DATA_DIR, *dataset_name.split("/"))
        is_sd2   = (sdg_key == "sd2")       # scDesign2 can generate new trials
        cfg_dir  = os.path.join(cfg_root, dataset_name.replace("/", "_"))

        for nd in donor_counts:
            if args.nd and args.nd != nd:
                continue

            # How many trials can we run?  For non-SD2 methods, cap at existing synth data.
            n_avail = N_TRIALS if is_sd2 else n_synth_available(data_dir, nd)
            if n_avail == 0:
                continue

            for tm_label, tm_code, white_box, use_wb_hvgs, use_aux in tms:
                n_done = count_done(data_dir, nd, tm_code)
                n_needed = min(N_TRIALS, n_avail) - n_done

                if n_needed <= 0:
                    jobs_skipped += 1
                    continue

                print(f"\n→ {dataset_name}  {nd}d  [{tm_label}]  "
                      f"({n_done}/{min(N_TRIALS, n_avail)} done, need {n_needed} more)")

                config_path = write_config(
                    dataset_name, nd, white_box, use_wb_hvgs, use_aux, pw, cfg_dir
                )

                for run_i in range(n_needed):
                    # Re-check in case a previous iteration just completed a trial
                    # (e.g., we're filling in multiple threat models for the same trial)
                    n_now = count_done(data_dir, nd, tm_code)
                    if n_now >= min(N_TRIALS, n_avail):
                        break

                    success = run_job(config_path, dry_run=args.dry_run)
                    if not args.dry_run:
                        jobs_run += 1
                        if not success:
                            print(f"  [WARN] Job returned non-zero exit code; "
                                  f"continuing sweep.", flush=True)
                    else:
                        jobs_run += 1  # count dry-run jobs

                    if args.max_jobs and jobs_run >= args.max_jobs:
                        print(f"\nReached --max-jobs={args.max_jobs}. Stopping.")
                        print_status()
                        return

    print(f"\n{'=' * 60}")
    print(f"  Sweep complete.  Jobs run: {jobs_run}  Already done: {jobs_skipped}")
    print(f"{'=' * 60}\n")
    print_status()


if __name__ == "__main__":
    main()

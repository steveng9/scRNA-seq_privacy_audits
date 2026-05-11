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
DATA_DIR  = "/home/golobs/data/scMAMAMIA"
RUNNER    = os.path.join(SRC_DIR, "run_experiment.py")
N_TRIALS  = 5

# ---------------------------------------------------------------------------
# Threat model definitions
# ---------------------------------------------------------------------------
# Threat model definitions
# ---------------------------------------------------------------------------
# (label, code, white_box, use_wb_hvgs, use_aux)
#
# TM_BB_COMBINED is a sentinel for the combined BB+aux / BB-aux path in
# run_experiment.py (run_both_bb=True).  A single job per (dataset, nd) computes
# both tm:100 and tm:101, reusing the synth-side Mahalanobis computation.
# The tracking code ("combined") is never written to results CSVs; it is only
# used internally in this sweep script.
TM_BB_COMBINED = [
    ("BB+/-aux", "combined", False, True, True),
]

# TM_BB_QUAD: one job per (dataset, nd) computes all 4 variants —
# standard BB+aux/BB-aux (→ mamamia_results.csv) and
# Class B BB+aux/BB-aux (→ mamamia_results_classb.csv) — reusing all
# expensive computation.
TM_BB_QUAD = [
    ("BB+/-aux quad", "quad", False, True, True),
]

# ---------------------------------------------------------------------------
# Sweep definition — OneK1K only, non-scDesign2 SDG methods
# ---------------------------------------------------------------------------
# (sdg_key, dataset_name, base_dataset_name, donor_counts, threat_models, parallel_workers)
#
# All entries use TM_BB_COMBINED: one job per (dataset, nd) covers both
# BB+aux (tm:100) and BB-aux (tm:101) in a single run_experiment.py call.

SWEEP = [
    # --- scVI ---
    ("scvi",   "ok/scvi/no_dp",         "ok", [5, 10, 20, 50, 100], TM_BB_COMBINED, 4),

    # --- scDiffusion ---
    ("scdiff", "ok/scdiffusion/no_dp",  "ok", [10, 20, 50],         TM_BB_COMBINED, 4),

    # --- scDesign3-Vine ---
    ("sd3v",   "ok/scdesign3/vine",     "ok", [10, 20, 50],         TM_BB_COMBINED, 4),

    # --- scDesign3-Gauss ---
    ("sd3g",   "ok/scdesign3/gaussian", "ok", [2, 5, 10, 20, 50, 100, 200], TM_BB_COMBINED, 4),

    # --- ZINBWave (Risso et al. 2018) ---
    ("zinbwave", "ok/zinbwave/no_dp", "ok", [10, 20, 50], TM_BB_QUAD, 4),

    # --- 490d quad attacks (Class B + standard, +/-aux all in one job) ---
    ("scvi",     "ok/scvi/no_dp",         "ok", [490], TM_BB_QUAD, 4),
    ("scdiff",   "ok/scdiffusion/no_dp",  "ok", [490], TM_BB_QUAD, 4),
    ("sd3v",     "ok/scdesign3/vine",     "ok", [490], TM_BB_QUAD, 4),
    ("sd3g",     "ok/scdesign3/gaussian", "ok", [490], TM_BB_QUAD, 4),
    ("zinbwave", "ok/zinbwave/no_dp",     "ok", [490], TM_BB_QUAD, 4),
    ("sd2_dp",   "ok/scdesign2/eps_1",         "ok", [490], TM_BB_QUAD, 4),
    ("sd2_dp",   "ok/scdesign2/eps_10",        "ok", [490], TM_BB_QUAD, 4),
    ("sd2_dp",   "ok/scdesign2/eps_100",       "ok", [490], TM_BB_QUAD, 4),
    ("sd2_dp",   "ok/scdesign2/eps_10000",     "ok", [490], TM_BB_QUAD, 4),
    ("sd2_dp",   "ok/scdesign2/eps_1000000",   "ok", [490], TM_BB_QUAD, 4),
    ("sd2_dp",   "ok/scdesign2/eps_100000000", "ok", [490], TM_BB_QUAD, 4),

    # --- scDiffusion-v3 (faithful + v1_celltypist compromise) ---
    ("scdiff_v3_faithful", "ok/scdiffusion_v3/faithful",      "ok", [10, 50], TM_BB_QUAD, 2),
    ("scdiff_v3_comp",     "ok/scdiffusion_v3/v1_celltypist", "ok", [50],     TM_BB_QUAD, 2),

    # --- NMF (SingleCellNMFGenerator — CAMDA 2024 winner) ---
    ("nmf",    "ok/nmf/no_dp",    "ok",   [10, 20, 50, 100, 200, 490], TM_BB_QUAD, 4),
    ("nmf",    "aida/nmf/no_dp",  "aida", [10, 20, 50, 100, 200],      TM_BB_QUAD, 4),

    # --- NMF + DP sweep (ok only, 50d) ---
    ("nmf_dp", "ok/nmf/eps_1",          "ok", [10, 20, 50], TM_BB_QUAD, 4),
    ("nmf_dp", "ok/nmf/eps_2.8",        "ok", [10, 20, 50], TM_BB_QUAD, 4),
    ("nmf_dp", "ok/nmf/eps_10",         "ok", [10, 20, 50], TM_BB_QUAD, 4),
    ("nmf_dp", "ok/nmf/eps_100",        "ok", [10, 20, 50], TM_BB_QUAD, 4),
    ("nmf_dp", "ok/nmf/eps_1000",       "ok", [10, 20, 50], TM_BB_QUAD, 4),
    ("nmf_dp", "ok/nmf/eps_10000",      "ok", [10, 20, 50], TM_BB_QUAD, 4),
    ("nmf_dp", "ok/nmf/eps_100000",     "ok", [10, 20, 50], TM_BB_QUAD, 4),
    ("nmf_dp", "ok/nmf/eps_1000000",    "ok", [50], TM_BB_QUAD, 4),
    ("nmf_dp", "ok/nmf/eps_10000000",   "ok", [50], TM_BB_QUAD, 4),
    ("nmf_dp", "ok/nmf/eps_100000000",  "ok", [50], TM_BB_QUAD, 4),

    # --- scDesign2 + DP (synthetic data already generated for all eps/nd/trial) ---
    ("sd2_dp", "ok/scdesign2/eps_1",          "ok", [10, 20, 50], TM_BB_COMBINED, 4),
    ("sd2_dp", "ok/scdesign2/eps_10",         "ok", [10, 20, 50], TM_BB_COMBINED, 4),
    ("sd2_dp", "ok/scdesign2/eps_100",        "ok", [10, 20, 50], TM_BB_COMBINED, 4),
    ("sd2_dp", "ok/scdesign2/eps_1000",       "ok", [10, 20, 50], TM_BB_COMBINED, 4),
    ("sd2_dp", "ok/scdesign2/eps_10000",      "ok", [10, 20, 50], TM_BB_COMBINED, 4),
    ("sd2_dp", "ok/scdesign2/eps_100000",     "ok", [10, 20, 50], TM_BB_COMBINED, 4),
    ("sd2_dp", "ok/scdesign2/eps_1000000",    "ok", [10, 20, 50], TM_BB_COMBINED, 4),
    ("sd2_dp", "ok/scdesign2/eps_10000000",   "ok", [10, 20, 50], TM_BB_COMBINED, 4),
    ("sd2_dp", "ok/scdesign2/eps_100000000",  "ok", [10, 20, 50], TM_BB_COMBINED, 4),
    ("sd2_dp", "ok/scdesign2/eps_1000000000", "ok", [10, 20, 50], TM_BB_COMBINED, 4),
]

# Standard scMAMA-MIA hyper-parameters (matches existing experiments)
MAMAMIA_PARAMS = {
    "IMPORTANCE_OF_CLASS_B_FPs": 0.17,
    "epsilon":                   0.0001,
    "mahalanobis":               True,
    "uniform_remapping_fn":      "zinb_cdf",
    "lin_alg_inverse_fn":        "pinv_gpu",
    "closeness_to_correlation_fn": "closeness_to_correlation_1",
    "class_b_gene_set":          "secondary",
    "class_b_scoring":           "llr",
    "class_b_gamma":             "auto",
    "class_b_gamma_noaux":       "auto",
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
    """No-op: symlinks are no longer needed. full_dataset_cleaned.h5ad and
    hvg.csv live at the dataset root (e.g. scMAMAMIA/ok/), and hvg_path is
    passed explicitly in each run_experiment.py config YAML."""
    print("\n[SETUP] Symlinks: not needed in scMAMAMIA layout — skipping.")


def setup_dp_splits():
    """
    DP trial directories contain train.npy written during generation.
    Ensure holdout.npy and auxiliary.npy exist in the shared splits/ directory
    (DATA_DIR/{base_dataset}/splits/{nd}d/{trial}/) so run_experiment.py finds them.
    """
    print("\n[SETUP] Pre-populating DP donor splits into shared splits/ dirs…")

    _all_donors_cache = {}

    for sdg_key, dataset_name, base_dataset, donor_counts, tms, pw in SWEEP:
        if sdg_key != "sd2_dp":
            continue

        sdg_path = os.path.join(DATA_DIR, *dataset_name.split("/"))

        for nd in donor_counts:
            for trial in range(1, N_TRIALS + 1):
                # train.npy lives in the DP SDG trial dir (written during generation)
                sdg_datasets_dir = os.path.join(sdg_path, f"{nd}d", str(trial), "datasets")
                train_npy_sdg    = os.path.join(sdg_datasets_dir, "train.npy")
                if not os.path.exists(train_npy_sdg):
                    continue

                # Target location: shared splits/ dir under dataset root
                splits_dir = os.path.join(DATA_DIR, base_dataset, "splits",
                                          f"{nd}d", str(trial))
                train_npy    = os.path.join(splits_dir, "train.npy")
                holdout_npy  = os.path.join(splits_dir, "holdout.npy")
                aux_npy      = os.path.join(splits_dir, "auxiliary.npy")

                os.makedirs(splits_dir, exist_ok=True)

                # Copy train.npy into splits/ if not already there
                if not os.path.exists(train_npy):
                    import shutil
                    shutil.copy2(train_npy_sdg, train_npy)

                if os.path.exists(holdout_npy) and os.path.exists(aux_npy):
                    continue  # already done

                # Load all donors lazily — h5ad lives at the base dataset root
                if base_dataset not in _all_donors_cache:
                    base_h5ad = os.path.join(DATA_DIR, base_dataset,
                                             "full_dataset_cleaned.h5ad")
                    if not os.path.exists(base_h5ad):
                        print(f"  [WARN] {base_h5ad} not found; skipping DP splits for {dataset_name}.")
                        break
                    print(f"  Loading donor list from {base_h5ad} …")
                    adata = ad.read_h5ad(base_h5ad, backed="r")
                    _all_donors_cache[base_dataset] = adata.obs["individual"].unique()
                    adata.file.close()
                all_donors = _all_donors_cache[base_dataset]

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
                print(f"  pre-populated splits: {base_dataset}/splits/{nd}d/{trial}")

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


def get_completed_tm_codes_classb(data_dir, nd, trial):
    """Return set of tm codes with a valid AUC in mamamia_results_classb.csv."""
    results_file = os.path.join(data_dir, f"{nd}d", str(trial), "results",
                                "mamamia_results_classb.csv")
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


def count_done_classb(data_dir, nd, tm_code):
    """Count trials that have a valid AUC in the Class B results file."""
    return sum(
        1 for t in range(1, N_TRIALS + 1)
        if tm_code in get_completed_tm_codes_classb(data_dir, nd, t)
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

def write_config(dataset_name, base_dataset, nd, white_box, use_wb_hvgs, use_aux,
                 parallel_workers, cfg_dir, run_both_bb=False, run_quad_bb=False):
    """
    Write a run_experiment.py config YAML to cfg_dir and return its path.
    Both local and server dir_list entries point to the server paths so the
    config works without the legacy 'T' flag.

    run_both_bb : bool
        When True, emit run_both_bb=True in mia_setting so run_experiment.py
        computes BB+aux (tm:100) and BB-aux (tm:101) in a single pass.
        use_aux must be True so trial tracking keys off tm:100.
    """
    if run_quad_bb:
        cfg_name = f"{nd}d_bb_quad.yaml"
    elif run_both_bb:
        cfg_name = f"{nd}d_bb_both.yaml"
    else:
        tm_parts = f"{'wb' if white_box else 'bb'}_{'aux' if use_aux else 'noaux'}"
        cfg_name = f"{nd}d_{tm_parts}.yaml"
    cfg_path = os.path.join(cfg_dir, cfg_name)

    # At 490d, use the dedicated strategy that sub-samples 200 holdout donors as aux
    strategy_fn = "sample_donors_strategy_490" if nd >= 490 else "sample_donors_strategy_2"
    mia_setting = {
        "sample_donors_strategy_fn": strategy_fn,
        "num_donors":    nd,
        "white_box":     white_box,
        "use_wb_hvgs":   use_wb_hvgs,
        "use_aux":       use_aux,
    }
    if run_quad_bb:
        mia_setting["run_quad_bb"] = True
    elif run_both_bb:
        mia_setting["run_both_bb"] = True

    hvg_path = os.path.join(DATA_DIR, base_dataset, "hvg.csv")

    cfg = {
        "dir_list": {
            "local":  {"home": REPO_ROOT, "data": DATA_DIR},
            "server": {"home": REPO_ROOT, "data": DATA_DIR},
        },
        "dataset_name": dataset_name,
        "hvg_path":     hvg_path,
        "generator_name": "scdesign2",   # always scdesign2 as shadow model
        "plot_results":   False,
        "parallelize":    True,
        "parallel_workers": parallel_workers,
        "min_aux_donors": MIN_AUX_DONORS,
        "mamamia_params": dict(MAMAMIA_PARAMS),
        "mia_setting":   mia_setting,
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

def _count_done_for_tm(data_dir, nd, tm_code):
    """
    Return number of completed trials for the given tm_code.
    'combined' requires tm:100 and tm:101 in mamamia_results.csv.
    'quad'     requires tm:100/101 in mamamia_results.csv AND
                        tm:100/101 in mamamia_results_classb.csv.
    """
    if tm_code == "quad":
        return min(
            count_done(data_dir, nd, "100"),
            count_done(data_dir, nd, "101"),
            count_done_classb(data_dir, nd, "100"),
            count_done_classb(data_dir, nd, "101"),
        )
    if tm_code == "combined":
        return min(count_done(data_dir, nd, "100"), count_done(data_dir, nd, "101"))
    return count_done(data_dir, nd, tm_code)


def print_status():
    """Print a summary table of current attack completion across all sweep entries."""
    print(f"\n{'=' * 80}")
    print(f"  scMAMA-MIA Attack Completion Status")
    print(f"{'=' * 80}\n")

    for sdg_key, dataset_name, base_dataset, donor_counts, tms, pw in SWEEP:
        data_dir = os.path.join(DATA_DIR, *dataset_name.split("/"))

        # Expand "combined"/"quad" into display columns
        display_cols = []
        for tm_label, tm_code, *_ in tms:
            if tm_code == "combined":
                display_cols.append(("BB+aux", "100", False))
                display_cols.append(("BB-aux", "101", False))
            elif tm_code == "quad":
                display_cols.append(("BB+aux",   "100", False))
                display_cols.append(("BB-aux",   "101", False))
                display_cols.append(("BB+aux_B", "100", True))
                display_cols.append(("BB-aux_B", "101", True))
            else:
                display_cols.append((tm_label, tm_code, False))

        print(f"  [{sdg_key}] {dataset_name}")
        header = f"    {'nd':>6}  " + "  ".join(f"{lbl:>8}" for lbl, _, __ in display_cols)
        print(header)

        for nd in donor_counts:
            parts = []
            for _, code, is_classb in display_cols:
                done  = (count_done_classb(data_dir, nd, code) if is_classb
                         else count_done(data_dir, nd, code))
                avail = n_synth_available(data_dir, nd)
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
        cfg_dir  = os.path.join(cfg_root, dataset_name.replace("/", "_"))

        for nd in donor_counts:
            if args.nd and args.nd != nd:
                continue

            # All sweep entries are non-sd2: cap at existing synthetic data.
            n_avail = n_synth_available(data_dir, nd)
            if n_avail == 0:
                continue

            for tm_label, tm_code, white_box, use_wb_hvgs, use_aux in tms:
                is_combined = (tm_code == "combined")
                is_quad     = (tm_code == "quad")
                n_done   = _count_done_for_tm(data_dir, nd, tm_code)
                n_needed = min(N_TRIALS, n_avail) - n_done

                if n_needed <= 0:
                    jobs_skipped += 1
                    continue

                print(f"\n→ {dataset_name}  {nd}d  [{tm_label}]  "
                      f"({n_done}/{min(N_TRIALS, n_avail)} done, need {n_needed} more)")

                config_path = write_config(
                    dataset_name, base_dataset, nd, white_box, use_wb_hvgs, use_aux,
                    pw, cfg_dir, run_both_bb=is_combined, run_quad_bb=is_quad,
                )

                for run_i in range(n_needed):
                    # Re-check: previous run may have completed more than one trial
                    n_now = _count_done_for_tm(data_dir, nd, tm_code)
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

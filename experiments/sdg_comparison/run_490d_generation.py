#!/usr/bin/env python3
"""
run_490d_generation.py — serial 490-donor synthetic-data generation for OneK1K.

Drives generation of synthetic.h5ad for the missing 490d × OneK1K SDG variants:

  Priority 1 (always run):
    scvi/no_dp
    zinbwave/no_dp
    scdesign2/eps_100
    scdesign2/eps_10000
    scdesign2/eps_1000000

  Priority 2 (run after P1, time-permitting):
    scdiffusion/no_dp
    scdesign3/gaussian
    scdesign3/vine

Skips trials that already have datasets/synthetic.h5ad.  Strictly serial across
trials to keep memory pressure manageable at 490 donors (~1.2M cells).  Each
generator manages its own internal cell-type parallelism.

Usage
-----
    nohup python experiments/sdg_comparison/run_490d_generation.py \
        > /tmp/gen_490d.log 2>&1 &
    disown

    # Filter (only run a subset)
    python experiments/sdg_comparison/run_490d_generation.py --priority 1
    python experiments/sdg_comparison/run_490d_generation.py --variants scvi/no_dp,zinbwave/no_dp
    python experiments/sdg_comparison/run_490d_generation.py --trials 1 2

    # Dry run / status
    python experiments/sdg_comparison/run_490d_generation.py --dry-run
    python experiments/sdg_comparison/run_490d_generation.py --status

For sd2/eps_X the no_dp 490d models/*.rds (Gaussian copula per cell type) must
already exist in scMAMAMIA/ok/scdesign2/no_dp/490d/{trial}/models/. If they
don't, that trial is skipped with a warning.
"""

import argparse
import datetime
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import anndata as ad

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DATA_ROOT       = "/home/golobs/data/scMAMAMIA"
DATASET         = "ok"
ND              = 490
N_TRIALS        = 5
GENERATE_TRIAL  = os.path.join(REPO_ROOT, "experiments", "sdg_comparison",
                               "generate_trial.py")
R_SCRIPT_V1     = os.path.join(SRC_DIR, "sdg", "scdesign2", "scdesign2.r")
DELTA           = 1e-5
CLIP_VALUE      = 3.0
DONOR_COL       = "individual"
CELL_TYPE_COL   = "cell_type"


# ---------------------------------------------------------------------------
# Job table
# ---------------------------------------------------------------------------
# Each row is (priority, sdg_subpath, generator_kind, extra_args).
#   generator_kind:
#     "trial_script" — invoke generate_trial.py with --generator <name>
#     "sd2_dp"        — apply v1 DP noise to ok/scdesign2/no_dp/490d copulas
JOBS = [
    # P1: finish scVI (trials 3,4,5 still missing as of 2026-04-30).
    (1, "scvi/no_dp",            "trial_script", {"generator": "scvi",
                                                   "conda_env": "scvi_"}),
    # P2: scDesign3-Vine (sd3g intentionally skipped per user 2026-04-30).
    (2, "scdesign3/vine",        "trial_script", {"generator": "sd3_vine"}),
]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _full_h5ad():
    return os.path.join(DATA_ROOT, DATASET, "full_dataset_cleaned.h5ad")


def _hvg_path():
    p = os.path.join(DATA_ROOT, DATASET, "hvg_full.csv")
    return p if os.path.exists(p) else os.path.join(DATA_ROOT, DATASET, "hvg.csv")


def _splits_dir(trial):
    return os.path.join(DATA_ROOT, DATASET, "splits", f"{ND}d", str(trial))


def _out_dir(sdg_subpath, trial):
    parts = sdg_subpath.split("/")
    return os.path.join(DATA_ROOT, DATASET, *parts, f"{ND}d", str(trial))


def _synth_path(sdg_subpath, trial):
    return os.path.join(_out_dir(sdg_subpath, trial), "datasets", "synthetic.h5ad")


def _no_dp_models_dir(trial):
    return os.path.join(DATA_ROOT, DATASET, "scdesign2", "no_dp", f"{ND}d",
                        str(trial), "models")


# ---------------------------------------------------------------------------
# Dispatch — generate_trial.py wrapper
# ---------------------------------------------------------------------------

def run_via_generate_trial(sdg_subpath, trial, generator, conda_env=None):
    out_dir = _out_dir(sdg_subpath, trial)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, GENERATE_TRIAL,
        "--generator",  generator,
        "--dataset",    _full_h5ad(),
        "--splits-dir", _splits_dir(trial),
        "--out-dir",    out_dir,
        "--hvg-path",   _hvg_path(),
    ]
    if conda_env:
        cmd += ["--conda-env", conda_env]

    print(f"  $ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"generate_trial.py failed (rc={rc}) for "
                           f"{sdg_subpath} trial {trial}")


# ---------------------------------------------------------------------------
# sd2/eps_X DP generation (ports gen_dp_quality_data.py to the new layout)
# ---------------------------------------------------------------------------

def _get_k_max(full_obs, donor_ids, cell_type):
    mask = full_obs[DONOR_COL].isin(donor_ids) & (full_obs[CELL_TYPE_COL] == cell_type)
    counts = full_obs[mask].groupby(DONOR_COL).size()
    return int(counts.max()) if len(counts) > 0 else 1


def _save_corrected_rds(copula_rds, cell_type, corrected_cov_np, out_path):
    from rpy2.robjects import r as R
    from rpy2.robjects.vectors import FloatVector
    G = corrected_cov_np.shape[0]
    flat = corrected_cov_np.flatten(order="F").tolist()
    R.assign("dp_copula_obj", copula_rds)
    R.assign("dp_noised_flat", FloatVector(flat))
    R(f'dp_copula_obj[["{cell_type}"]][["cov_mat"]] <- '
      f'matrix(dp_noised_flat, nrow={G}, ncol={G})')
    R(f'saveRDS(dp_copula_obj, file="{out_path}")')


def _run_r_gen(copula_path, n_cells, out_rds_path):
    cmd = ["Rscript", R_SCRIPT_V1, "gen", str(int(n_cells)), copula_path, out_rds_path]
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError as e:
        out = e.output.decode("utf-8", errors="replace")
        print(f"    [WARN] R gen failed: {out[-400:]}", flush=True)
        return False


def _assemble_synth(cell_types, test_cell_type_arr, tmp_dir, hvg_mask, all_var_names):
    import pyreadr
    hvg_indices = np.where(hvg_mask)[0]
    n_cells = len(test_cell_type_arr)
    n_genes = len(all_var_names)
    X = sp.lil_matrix((n_cells, n_genes), dtype=np.float64)
    for ct in cell_types:
        out_path = os.path.join(tmp_dir, f"out{ct}.rds")
        if not os.path.exists(out_path):
            continue
        try:
            r_mat = list(pyreadr.read_r(out_path).values())[0]
            counts_np = r_mat.to_numpy() if hasattr(r_mat, "to_numpy") else np.array(r_mat)
            cell_indices = np.where(test_cell_type_arr == ct)[0]
            n_assign = min(len(cell_indices), counts_np.shape[1])
            for i in range(n_assign):
                n_g = min(len(hvg_indices), counts_np.shape[0])
                X[cell_indices[i], hvg_indices[:n_g]] = counts_np[:n_g, i]
        except Exception as e:
            print(f"    [SKIP] {out_path}: {e}", flush=True)
    a = ad.AnnData(X=X.tocsr())
    a.obs[CELL_TYPE_COL] = test_cell_type_arr
    a.var_names = all_var_names
    return a


def run_sd2_dp(sdg_subpath, trial, epsilon, seed=42):
    """Generate v1 DP synthetic data for one (eps, trial) at 490d on OneK1K."""
    from rpy2.robjects import r as R
    from sdg.scdesign2.copula import parse_copula
    from sdg.dp.dp_copula import apply_gaussian_dp
    from sdg.dp.sensitivity import gaussian_noise_scale

    out_dir = _out_dir(sdg_subpath, trial)
    out_datasets = os.path.join(out_dir, "datasets")
    os.makedirs(out_datasets, exist_ok=True)
    synth_path = os.path.join(out_datasets, "synthetic.h5ad")

    models_dir = _no_dp_models_dir(trial)
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(
            f"sd2/no_dp 490d models missing: {models_dir}")

    train_npy = os.path.join(_splits_dir(trial), "train.npy")
    train_donors = np.load(train_npy, allow_pickle=True)

    full = sc.read_h5ad(_full_h5ad(), backed="r")
    full_obs = full.obs[[DONOR_COL, CELL_TYPE_COL]].copy()
    all_var_names = full.var_names.copy()
    full.file.close()

    hvg_df = pd.read_csv(_hvg_path(), index_col=0)
    hvg_mask = hvg_df["highly_variable"].values.astype(bool)

    # scDesign2 generates as many cells as the train-cell-type counts.
    train_mask = full_obs[DONOR_COL].isin(train_donors)
    test_cell_type_arr = full_obs.loc[train_mask, CELL_TYPE_COL].values

    cell_types = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(models_dir)
        if f.endswith(".rds")
    ])
    cell_types = [
        ct for ct in cell_types
        if ct.isdigit() or (ct.startswith("-") and ct[1:].isdigit())
    ]

    rng = np.random.default_rng(seed + hash((ND, trial, int(epsilon))) % (2**31))

    with tempfile.TemporaryDirectory(prefix=f"sd2_dp_eps{int(epsilon)}_") as tmp_dir:
        for ct in cell_types:
            copula_path = os.path.join(models_dir, f"{ct}.rds")
            try:
                copula_rds = R["readRDS"](copula_path)
                ct_obj = copula_rds.rx2(str(ct))
                parsed = parse_copula(ct_obj)
            except Exception as e:
                print(f"    [WARN] parse failed for ct={ct}: {e}", flush=True)
                continue
            if parsed.get("cov_matrix") is None:
                print(f"    [SKIP] {ct}: cov_mat None (no group-1 genes)", flush=True)
                continue

            n_cells_ct = int(ct_obj.rx2("n_cell")[0])
            k_max = _get_k_max(full_obs, train_donors, ct)
            if n_cells_ct <= k_max:
                k_max = max(1, n_cells_ct - 1)

            try:
                noised = apply_gaussian_dp(
                    copula_dict=parsed,
                    epsilon=epsilon,
                    delta=DELTA,
                    n_cells=n_cells_ct,
                    k_max=k_max,
                    clip_value=CLIP_VALUE,
                    rng=rng,
                )
            except Exception as e:
                print(f"    [WARN] DP noise failed for ct={ct}: {e}", flush=True)
                continue

            n_to_gen = int((test_cell_type_arr == ct).sum())
            if n_to_gen == 0:
                continue

            patched_rds = os.path.join(tmp_dir, f"patched_{ct}.rds")
            out_rds     = os.path.join(tmp_dir, f"out{ct}.rds")
            try:
                _save_corrected_rds(copula_rds, ct, noised["cov_matrix"], patched_rds)
                ok = _run_r_gen(patched_rds, n_to_gen, out_rds)
                if not ok:
                    continue
                sigma = gaussian_noise_scale(
                    epsilon=epsilon, delta=DELTA, n_cells=n_cells_ct,
                    k_max=k_max, n_genes=parsed["len_primary"],
                    clip_value=CLIP_VALUE,
                )
                print(f"    {ct}: n_cells={n_cells_ct} k_max={k_max} "
                      f"sigma={sigma:.3f} gen={n_to_gen}", flush=True)
            except Exception as e:
                print(f"    [WARN] gen failed for ct={ct}: {e}", flush=True)
                continue

        adata = _assemble_synth(cell_types, test_cell_type_arr,
                                tmp_dir, hvg_mask, all_var_names)
        adata.write(synth_path, compression="gzip")
        print(f"  saved: {synth_path}  shape={adata.shape}", flush=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _filter_jobs(args):
    selected = []
    for prio, sdg_subpath, kind, extra in JOBS:
        if args.priority and prio not in args.priority:
            continue
        if args.variants and sdg_subpath not in args.variants:
            continue
        selected.append((prio, sdg_subpath, kind, extra))
    return selected


def _print_status(jobs, trials):
    print(f"\n  490d generation status — {DATASET}, donor={ND}, trials={trials}")
    print(f"  {'Variant':<28} " +
          " ".join(f"{'t'+str(t):>4}" for t in trials) + "  done/total")
    print(f"  {'-'*28} " + " ".join(['-'*4 for _ in trials]) + "  ----------")
    for prio, sdg_subpath, kind, _ in jobs:
        done = []
        for t in trials:
            ok = os.path.exists(_synth_path(sdg_subpath, t))
            done.append(ok)
        cells = " ".join(("  ✓" if d else "  ·") for d in done)
        n_done = sum(done)
        print(f"  {sdg_subpath:<28} {cells}  {n_done}/{len(trials)}")
    print()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--priority", nargs="+", type=int, choices=[1, 2],
                    help="Run only these priority tiers (default: 1 then 2)")
    ap.add_argument("--variants", nargs="+", default=None,
                    help="Run only these sdg_subpaths (e.g. scvi/no_dp scdesign2/eps_100)")
    ap.add_argument("--trials",   nargs="+", type=int,
                    default=list(range(1, N_TRIALS + 1)),
                    help="Trial numbers (default: 1 2 3 4 5)")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Print plan without running anything")
    ap.add_argument("--status",   action="store_true",
                    help="Print current completion grid and exit")
    args = ap.parse_args()

    # Allow comma-separated --variants  (scvi/no_dp,zinbwave/no_dp)
    if args.variants and len(args.variants) == 1 and "," in args.variants[0]:
        args.variants = [v.strip() for v in args.variants[0].split(",") if v.strip()]

    jobs = _filter_jobs(args)

    if args.status:
        _print_status(jobs, args.trials)
        return

    print(f"\n{'='*70}")
    print(f"  490d generation — OneK1K, donor={ND}, trials={args.trials}")
    print(f"  Started: {datetime.datetime.utcnow().isoformat()}Z")
    print(f"{'='*70}\n", flush=True)

    _print_status(jobs, args.trials)

    pending = []
    for prio, sdg_subpath, kind, extra in jobs:
        for trial in args.trials:
            if os.path.exists(_synth_path(sdg_subpath, trial)):
                continue
            pending.append((prio, sdg_subpath, kind, extra, trial))

    if not pending:
        print("Nothing to do — all selected (variant × trial) combos already have synthetic.h5ad.")
        return

    # Sort: priority tier, then variant order in JOBS, then trial.
    job_order = {sp: i for i, (_, sp, _, _) in enumerate(JOBS)}
    pending.sort(key=lambda x: (x[0], job_order[x[1]], x[4]))

    print(f"  {len(pending)} jobs queued.\n")
    if args.dry_run:
        for prio, sdg_subpath, kind, extra, trial in pending:
            print(f"    [DRY] P{prio}  {sdg_subpath:<28} trial {trial}  ({kind})")
        return

    n_ok = n_skip = n_fail = 0
    for prio, sdg_subpath, kind, extra, trial in pending:
        # Re-check existence in case a previous iteration produced it.
        if os.path.exists(_synth_path(sdg_subpath, trial)):
            n_skip += 1
            continue

        # Splits must already exist for this trial.
        if not os.path.exists(os.path.join(_splits_dir(trial), "train.npy")):
            print(f"\n[SKIP] {sdg_subpath} trial {trial} — splits missing at "
                  f"{_splits_dir(trial)}", flush=True)
            n_skip += 1
            continue

        t0 = time.time()
        ts = datetime.datetime.utcnow().isoformat()
        print(f"\n{'-'*70}")
        print(f"  [P{prio}] {sdg_subpath}  trial {trial}   {ts}Z")
        print(f"{'-'*70}", flush=True)

        try:
            if kind == "trial_script":
                run_via_generate_trial(sdg_subpath, trial, **extra)
            elif kind == "sd2_dp":
                run_sd2_dp(sdg_subpath, trial, **extra)
            else:
                raise ValueError(f"Unknown generator kind: {kind}")
            elapsed = time.time() - t0
            if os.path.exists(_synth_path(sdg_subpath, trial)):
                n_ok += 1
                print(f"  ✓ done in {elapsed/60:.1f} min", flush=True)
            else:
                n_fail += 1
                print(f"  ✗ exited cleanly but synthetic.h5ad not found", flush=True)
        except KeyboardInterrupt:
            print("\n[interrupted by user]", flush=True)
            sys.exit(130)
        except Exception as e:
            n_fail += 1
            print(f"  ✗ FAILED: {e}", flush=True)
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"  Done: ok={n_ok} skip={n_skip} fail={n_fail}")
    print(f"  Finished: {datetime.datetime.utcnow().isoformat()}Z")
    print(f"{'='*70}\n", flush=True)
    _print_status(jobs, args.trials)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_camda_half_generation.py — single-trial 490-donor CAMDA 2026 submission
synthetic data generation for OneK1K.

Drives generation of `synthetic.h5ad` for one trial (the CAMDA train release)
across a pre-defined set of SDG variants. Unlike `run_490d_generation.py`,
which sweeps random 490-donor splits across 5 trials for internal evaluation,
this orchestrator:

  • Uses the CAMDA train release directly (490 donors, ~634K cells)
  • Runs ONE "trial" (the official train/test split)
  • Output dir layout: {sdg}/{variant}/camda_half/1/

Variants (in execution order):

  Priority 1 (scDesign2 family):
    scdesign2/no_dp                 ← fits copulas; required for eps_X variants
    scdesign2/eps_1
    scdesign2/eps_100
    scdesign2/eps_10000
    scdesign2/eps_1000000
    scdesign2/eps_100000000

  Priority 2 (CPU-bound generators):
    zinbwave/no_dp
    scdesign3/vine

  Priority 3 (GPU-bound generators):
    scvi/no_dp
    scdiffusion/no_dp

Usage
-----
    nohup setsid python -u experiments/sdg_comparison/run_camda_half_generation.py \
        > /tmp/camda_half.log 2>&1 < /dev/null & disown

    # Filter
    python experiments/sdg_comparison/run_camda_half_generation.py --priority 1
    python experiments/sdg_comparison/run_camda_half_generation.py --variants scdesign2/no_dp
    python experiments/sdg_comparison/run_camda_half_generation.py --status
    python experiments/sdg_comparison/run_camda_half_generation.py --dry-run

Splits
------
The shared splits dir at  ${DATA_ROOT}/ok/splits/camda_half/1/ must contain:
    train.npy        — 490 donor IDs (CAMDA train release)
    holdout.npy      — empty array
    auxiliary.npy    — empty array
This script auto-creates them on first run if missing.

For sd2/eps_X to work, sd2/no_dp must run first (or have completed previously),
since DP variants reuse the no_dp per-cell-type Gaussian copulas (.rds files).
"""

import argparse
import datetime
import gc
import os
import shlex
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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
SPLIT_NAME      = "camda_half"
TRIAL           = 1
GENERATE_TRIAL  = os.path.join(REPO_ROOT, "experiments", "sdg_comparison",
                               "generate_trial.py")
R_SCRIPT_V1     = os.path.join(SRC_DIR, "sdg", "scdesign2", "scdesign2.r")
DELTA           = 1e-5
CLIP_VALUE      = 3.0
DONOR_COL       = "individual"
CELL_TYPE_COL   = "cell_type"
N_FIT_WORKERS   = 4   # cell-type fit parallelism for sd2/no_dp


# ---------------------------------------------------------------------------
# Job table
# ---------------------------------------------------------------------------
# (priority, sdg_subpath, generator_kind, extra_args)
#   generator_kind:
#     "trial_script" — invoke generate_trial.py with --generator <name>
#     "sd2_no_dp"    — fit + gen scdesign2 inline (Rscript train per-ct, then gen)
#     "sd2_dp"       — apply Gaussian DP noise to existing camda_half no_dp copulas
JOBS = [
    (1, "scdesign2/no_dp",         "sd2_no_dp",    {}),
    (1, "scdesign2/eps_1",         "sd2_dp",       {"epsilon": 1.0}),
    (1, "scdesign2/eps_100",       "sd2_dp",       {"epsilon": 100.0}),
    (1, "scdesign2/eps_10000",     "sd2_dp",       {"epsilon": 10000.0}),
    (1, "scdesign2/eps_1000000",   "sd2_dp",       {"epsilon": 1000000.0}),
    (1, "scdesign2/eps_100000000", "sd2_dp",       {"epsilon": 100000000.0}),
    (2, "zinbwave/no_dp",          "trial_script", {"generator": "zinbwave"}),
    (2, "scdesign3/vine",          "trial_script", {"generator": "sd3_vine"}),
    (3, "scvi/no_dp",              "trial_script", {"generator": "scvi",
                                                     "conda_env": "scvi_"}),
    (3, "scdiffusion/no_dp",       "trial_script", {"generator": "scdiffusion",
                                                     "conda_env": "scdiff_"}),
]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _full_h5ad():
    """Source dataset for generation: full_dataset_cleaned.h5ad.

    The 490 train donors form a strict subset of full_dataset_cleaned.h5ad
    (which is concat(train_release, test_release)); filtering by train donor IDs
    yields exactly the train release cells. Using the full file keeps the data
    path identical to the random-490d trials so internal comparisons stay sound.
    """
    return os.path.join(DATA_ROOT, DATASET, "full_dataset_cleaned.h5ad")


def _hvg_path():
    p = os.path.join(DATA_ROOT, DATASET, "hvg_full.csv")
    return p if os.path.exists(p) else os.path.join(DATA_ROOT, DATASET, "hvg.csv")


def _splits_dir():
    return os.path.join(DATA_ROOT, DATASET, "splits", SPLIT_NAME, str(TRIAL))


def _out_dir(sdg_subpath):
    parts = sdg_subpath.split("/")
    return os.path.join(DATA_ROOT, DATASET, *parts, SPLIT_NAME, str(TRIAL))


def _synth_path(sdg_subpath):
    return os.path.join(_out_dir(sdg_subpath), "datasets", "synthetic.h5ad")


def _no_dp_models_dir():
    return os.path.join(DATA_ROOT, DATASET, "scdesign2", "no_dp",
                        SPLIT_NAME, str(TRIAL), "models")


# ---------------------------------------------------------------------------
# Splits bootstrap
# ---------------------------------------------------------------------------

def _ensure_splits():
    """Create splits/{SPLIT_NAME}/{TRIAL}/ from the train release if missing."""
    sd = _splits_dir()
    train_npy   = os.path.join(sd, "train.npy")
    holdout_npy = os.path.join(sd, "holdout.npy")
    aux_npy     = os.path.join(sd, "auxiliary.npy")
    if all(os.path.exists(p) for p in (train_npy, holdout_npy, aux_npy)):
        n = len(np.load(train_npy, allow_pickle=True))
        print(f"  splits ok: {sd} (train={n})", flush=True)
        return
    train_release = os.path.join(DATA_ROOT, DATASET,
                                 "onek1k_annotated_train_release.h5ad")
    if not os.path.exists(train_release):
        raise FileNotFoundError(
            f"Train release h5ad not found at {train_release}. Extract from "
            "onek1k_annotated_train_full.zip first.")
    a = sc.read_h5ad(train_release, backed="r")
    donors = np.array(sorted(a.obs[DONOR_COL].unique()))
    a.file.close()
    os.makedirs(sd, exist_ok=True)
    np.save(train_npy,   donors)
    np.save(holdout_npy, np.array([], dtype=donors.dtype))
    np.save(aux_npy,     np.array([], dtype=donors.dtype))
    print(f"  wrote splits → {sd} (train={len(donors)}, holdout=[], aux=[])",
          flush=True)


# ---------------------------------------------------------------------------
# Dispatch — generate_trial.py wrapper (non-sd2 generators)
# ---------------------------------------------------------------------------

def run_via_generate_trial(sdg_subpath, generator, conda_env=None):
    out_dir = _out_dir(sdg_subpath)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, GENERATE_TRIAL,
        "--generator",  generator,
        "--dataset",    _full_h5ad(),
        "--splits-dir", _splits_dir(),
        "--out-dir",    out_dir,
        "--hvg-path",   _hvg_path(),
    ]
    if conda_env:
        cmd += ["--conda-env", conda_env]
    print(f"  $ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"generate_trial.py failed (rc={rc}) for "
                           f"{sdg_subpath} {SPLIT_NAME}/{TRIAL}")


# ---------------------------------------------------------------------------
# scdesign2 inline helpers (fit + gen via Rscript)
# ---------------------------------------------------------------------------

def _r_train_one(hvg_h5ad_path, cell_type, out_rds_path):
    """Run `Rscript scdesign2.r train` for a single cell type."""
    cmd = ["Rscript", R_SCRIPT_V1, "train", hvg_h5ad_path,
           str(cell_type), out_rds_path]
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return (cell_type, True, None)
    except subprocess.CalledProcessError as e:
        out = e.output.decode("utf-8", errors="replace")
        return (cell_type, False, out[-500:])


def _r_gen_one(copula_path, n_cells, out_rds_path):
    cmd = ["Rscript", R_SCRIPT_V1, "gen", str(int(n_cells)),
           copula_path, out_rds_path]
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError as e:
        out = e.output.decode("utf-8", errors="replace")
        print(f"    [WARN] R gen failed: {out[-400:]}", flush=True)
        return False


def _assemble_synth(cell_types, test_cell_type_arr, gen_dir, hvg_mask,
                     all_var_names):
    import pyreadr
    hvg_indices = np.where(hvg_mask)[0]
    n_cells = len(test_cell_type_arr)
    n_genes = len(all_var_names)
    X = sp.lil_matrix((n_cells, n_genes), dtype=np.float64)
    for ct in cell_types:
        out_path = os.path.join(gen_dir, f"out{ct}.rds")
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


# ---------------------------------------------------------------------------
# sd2/no_dp: fit per-cell-type copulas + generate synthetic.h5ad inline.
# ---------------------------------------------------------------------------

def run_sd2_no_dp(sdg_subpath):
    out_dir       = _out_dir(sdg_subpath)
    out_datasets  = os.path.join(out_dir, "datasets")
    out_models    = os.path.join(out_dir, "models")
    os.makedirs(out_datasets, exist_ok=True)
    os.makedirs(out_models,   exist_ok=True)
    synth_path = os.path.join(out_datasets, "synthetic.h5ad")

    train_donors = np.load(os.path.join(_splits_dir(), "train.npy"),
                           allow_pickle=True)
    print(f"  train donors: {len(train_donors)}", flush=True)

    full = sc.read_h5ad(_full_h5ad(), backed="r")
    full_obs = full.obs[[DONOR_COL, CELL_TYPE_COL]].copy()
    all_var_names = full.var_names.copy()

    hvg_df = pd.read_csv(_hvg_path(), index_col=0)
    hvg_mask = hvg_df["highly_variable"].values.astype(bool)

    # Filter to train cells, HVG-subset, write to a temp h5ad for the R fitter.
    train_idx = full.obs[DONOR_COL].isin(train_donors).values
    print(f"  train cells: {int(train_idx.sum())}", flush=True)
    train_data = full[train_idx, :].to_memory()
    full.file.close()

    train_data = train_data[:, hvg_mask].copy()
    cell_type_arr = train_data.obs[CELL_TYPE_COL].astype(str).values
    cell_types = sorted(np.unique(cell_type_arr))
    cell_type_counts = {ct: int((cell_type_arr == ct).sum()) for ct in cell_types}
    print(f"  cell types: {cell_types}", flush=True)
    print(f"  per-ct counts: {cell_type_counts}", flush=True)

    # The R fitter expects the cell_type column as a string; make sure of it.
    train_data.obs[CELL_TYPE_COL] = train_data.obs[CELL_TYPE_COL].astype(str)

    # full-train cell-type array (same order/index as test_cell_type_arr below)
    test_cell_type_arr = cell_type_arr

    with tempfile.TemporaryDirectory(prefix="sd2_camda_") as tmp_dir:
        hvg_train_path = os.path.join(tmp_dir, "hvg_train.h5ad")
        train_data.write_h5ad(hvg_train_path)
        del train_data
        gc.collect()

        # ---- Parallel fit ----
        print(f"  fitting {len(cell_types)} cell types "
              f"with {N_FIT_WORKERS} parallel workers ...", flush=True)
        ctx = multiprocessing.get_context("spawn")
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=N_FIT_WORKERS, mp_context=ctx) as ex:
            futures = {
                ex.submit(_r_train_one, hvg_train_path, ct,
                          os.path.join(out_models, f"{ct}.rds")): ct
                for ct in cell_types
            }
            for f in as_completed(futures):
                ct, ok, err = f.result()
                if ok:
                    print(f"    fit done: ct={ct}", flush=True)
                else:
                    print(f"    [WARN] fit FAILED ct={ct}: {err}",
                          flush=True)
        print(f"  fit total: {(time.time()-t0)/60:.1f} min", flush=True)

        # ---- Sequential gen ----
        for ct in cell_types:
            n_to_gen = cell_type_counts[ct]
            if n_to_gen == 0:
                continue
            copula_path = os.path.join(out_models, f"{ct}.rds")
            if not os.path.exists(copula_path):
                print(f"    [SKIP] gen ct={ct}: no copula", flush=True)
                continue
            out_rds = os.path.join(tmp_dir, f"out{ct}.rds")
            ok = _r_gen_one(copula_path, n_to_gen, out_rds)
            if ok:
                print(f"    gen done: ct={ct}  n={n_to_gen}", flush=True)

        # ---- Assemble synthetic.h5ad ----
        adata = _assemble_synth(cell_types, test_cell_type_arr,
                                tmp_dir, hvg_mask, all_var_names)
        adata.write(synth_path, compression="gzip")
        print(f"  saved: {synth_path}  shape={adata.shape}", flush=True)


# ---------------------------------------------------------------------------
# sd2/eps_X: apply DP noise to the camda_half/1 no_dp copulas, regenerate.
# ---------------------------------------------------------------------------

def run_sd2_dp(sdg_subpath, epsilon, seed=42):
    from rpy2.robjects import r as R
    from sdg.scdesign2.copula import parse_copula
    from sdg.dp.dp_copula import apply_gaussian_dp
    from sdg.dp.sensitivity import gaussian_noise_scale

    out_dir      = _out_dir(sdg_subpath)
    out_datasets = os.path.join(out_dir, "datasets")
    os.makedirs(out_datasets, exist_ok=True)
    synth_path = os.path.join(out_datasets, "synthetic.h5ad")

    models_dir = _no_dp_models_dir()
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(
            f"sd2/no_dp camda_half models missing: {models_dir} "
            "(run scdesign2/no_dp first)")

    train_donors = np.load(os.path.join(_splits_dir(), "train.npy"),
                           allow_pickle=True)

    full = sc.read_h5ad(_full_h5ad(), backed="r")
    full_obs = full.obs[[DONOR_COL, CELL_TYPE_COL]].copy()
    all_var_names = full.var_names.copy()
    full.file.close()

    hvg_df = pd.read_csv(_hvg_path(), index_col=0)
    hvg_mask = hvg_df["highly_variable"].values.astype(bool)

    train_mask = full_obs[DONOR_COL].isin(train_donors)
    test_cell_type_arr = full_obs.loc[train_mask, CELL_TYPE_COL].astype(str).values

    cell_types = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(models_dir)
        if f.endswith(".rds")
    ])
    cell_types = [
        ct for ct in cell_types
        if ct.isdigit() or (ct.startswith("-") and ct[1:].isdigit())
    ]

    rng = np.random.default_rng(seed + hash((SPLIT_NAME, TRIAL, int(epsilon))) % (2**31))

    with tempfile.TemporaryDirectory(prefix=f"sd2_camda_dp_eps{int(epsilon)}_") as tmp_dir:
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
                print(f"    [SKIP] {ct}: cov_mat None (no group-1 genes)",
                      flush=True)
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
                print(f"    [WARN] DP noise failed for ct={ct}: {e}",
                      flush=True)
                continue

            n_to_gen = int((test_cell_type_arr == ct).sum())
            if n_to_gen == 0:
                continue

            patched_rds = os.path.join(tmp_dir, f"patched_{ct}.rds")
            out_rds     = os.path.join(tmp_dir, f"out{ct}.rds")
            try:
                _save_corrected_rds(copula_rds, ct, noised["cov_matrix"],
                                    patched_rds)
                ok = _r_gen_one(patched_rds, n_to_gen, out_rds)
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


def _print_status(jobs):
    print(f"\n  camda_half generation status — {DATASET}, "
          f"split={SPLIT_NAME}, trial={TRIAL}")
    print(f"  {'Variant':<32}  state")
    print(f"  {'-'*32}  -------")
    for _, sdg_subpath, _, _ in jobs:
        state = "✓ done" if os.path.exists(_synth_path(sdg_subpath)) else "·"
        print(f"  {sdg_subpath:<32}  {state}")
    print()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--priority", nargs="+", type=int, choices=[1, 2, 3],
                    help="Run only these priority tiers (default: 1 2 3)")
    ap.add_argument("--variants", nargs="+", default=None,
                    help="Run only these sdg_subpaths (e.g. scdesign2/no_dp)")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Print plan without running anything")
    ap.add_argument("--status",   action="store_true",
                    help="Print current completion grid and exit")
    args = ap.parse_args()

    if args.variants and len(args.variants) == 1 and "," in args.variants[0]:
        args.variants = [v.strip() for v in args.variants[0].split(",") if v.strip()]

    jobs = _filter_jobs(args)

    if args.status:
        _print_status(jobs)
        return

    print(f"\n{'='*70}")
    print(f"  CAMDA half generation — OneK1K, split={SPLIT_NAME}/{TRIAL}")
    print(f"  Started: {datetime.datetime.utcnow().isoformat()}Z")
    print(f"{'='*70}\n", flush=True)

    _ensure_splits()
    _print_status(jobs)

    pending = []
    for prio, sdg_subpath, kind, extra in jobs:
        if os.path.exists(_synth_path(sdg_subpath)):
            continue
        pending.append((prio, sdg_subpath, kind, extra))

    if not pending:
        print("Nothing to do — all selected variants already have synthetic.h5ad.")
        return

    job_order = {sp: i for i, (_, sp, _, _) in enumerate(JOBS)}
    pending.sort(key=lambda x: (x[0], job_order[x[1]]))

    print(f"  {len(pending)} jobs queued.\n")
    if args.dry_run:
        for prio, sdg_subpath, kind, extra in pending:
            print(f"    [DRY] P{prio}  {sdg_subpath:<32} ({kind})")
        return

    n_ok = n_skip = n_fail = 0
    for prio, sdg_subpath, kind, extra in pending:
        if os.path.exists(_synth_path(sdg_subpath)):
            n_skip += 1
            continue

        t0 = time.time()
        ts = datetime.datetime.utcnow().isoformat()
        print(f"\n{'-'*70}")
        print(f"  [P{prio}] {sdg_subpath}  {SPLIT_NAME}/{TRIAL}   {ts}Z")
        print(f"{'-'*70}", flush=True)

        try:
            if kind == "trial_script":
                run_via_generate_trial(sdg_subpath, **extra)
            elif kind == "sd2_no_dp":
                run_sd2_no_dp(sdg_subpath)
            elif kind == "sd2_dp":
                run_sd2_dp(sdg_subpath, **extra)
            else:
                raise ValueError(f"Unknown generator kind: {kind}")
            elapsed = time.time() - t0
            if os.path.exists(_synth_path(sdg_subpath)):
                n_ok += 1
                print(f"  ✓ done in {elapsed/60:.1f} min", flush=True)
            else:
                n_fail += 1
                print(f"  ✗ exited cleanly but synthetic.h5ad not found",
                      flush=True)
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
    _print_status(jobs)


if __name__ == "__main__":
    main()

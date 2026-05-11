#!/usr/bin/env python3
"""
gen_cg_10d_trial1.py — Sequential synthetic data generation for cg 10d trial 1.

Generates datasets/synthetic.h5ad for one donor split (cg / 10 donors / trial 1)
using 9 SDG variants, in priority order so the most-wanted outputs finish first:

  1. scdesign2/eps_1        (SD2 + DP noise, ε=1)
  2. zinbwave/no_dp
  3. scvi/no_dp
  4. scdesign2/eps_100
  5. scdesign2/eps_100000
  6. scdiffusion_v3/faithful (paper-faithful: unconditional DDPM + CellTypist)
  7. nmf/no_dp
  8. scdesign3/vine
  9. nmf/eps_2.8

Launch (detached from SSH):
    nohup conda run --no-capture-output -n tabddpm_ \\
        python experiments/sdg_comparison/gen_cg_10d_trial1.py \\
        > /tmp/cg_10d_gen.log 2>&1 &
    echo $!

Monitor:
    tail -f /tmp/cg_10d_gen.log
    grep -E '^(=== |DONE|SKIP|FAIL)' /tmp/cg_10d_gen.log
"""

import os
import sys
import subprocess
import time
import datetime
import tempfile

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DATA_ROOT = "/home/golobs/data/scMAMAMIA"
DATASET   = "cg"
ND        = 10
TRIAL     = 1

BASE_DIR    = os.path.join(DATA_ROOT, DATASET)
FULL_H5AD   = os.path.join(BASE_DIR, "full_dataset_cleaned.h5ad")
HVG_PATH    = os.path.join(BASE_DIR, "hvg_full.csv")
SPLITS_DIR  = os.path.join(BASE_DIR, "splits", f"{ND}d", str(TRIAL))
MODELS_DIR  = os.path.join(BASE_DIR, "scdesign2", "no_dp", f"{ND}d", str(TRIAL), "models")

GENERATE_TRIAL = os.path.join(REPO_ROOT, "experiments", "sdg_comparison", "generate_trial.py")
R_SCRIPT_V1    = os.path.join(SRC_DIR, "sdg", "scdesign2", "scdesign2.r")

DELTA      = 1e-5
CLIP_VALUE = 3.0
DONOR_COL  = "individual"
CT_COL     = "cell_type"

# NMF epsilon ratios (CAMDA 2024 proportional allocation)
_NMF_RATIOS = (0.5, 2.1, 0.2)
_NMF_TOTAL  = sum(_NMF_RATIOS)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def banner(msg):
    print(f"\n{'='*60}", flush=True)
    print(f"=== [{_ts()}] {msg}", flush=True)
    print(f"{'='*60}", flush=True)


def skip(msg):
    print(f"SKIP [{_ts()}] {msg}", flush=True)


def done(msg):
    print(f"DONE [{_ts()}] {msg}", flush=True)


def fail(msg):
    print(f"FAIL [{_ts()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# SD2 DP generation helpers (adapted for named cell types)
# ---------------------------------------------------------------------------

def _get_k_max(full_obs, train_donors, cell_type):
    mask = full_obs[DONOR_COL].isin(train_donors) & (full_obs[CT_COL] == cell_type)
    counts = full_obs[mask].groupby(DONOR_COL).size()
    return int(counts.max()) if len(counts) > 0 else 1


def _save_corrected_rds(copula_rds, cell_type, corrected_cov_np, out_path):
    from rpy2.robjects import r as R
    from rpy2.robjects.vectors import FloatVector
    G    = corrected_cov_np.shape[0]
    flat = corrected_cov_np.flatten(order="F").tolist()
    R.assign("cg_dp_copula_obj", copula_rds)
    R.assign("cg_dp_noised_flat", FloatVector(flat))
    R(f'cg_dp_copula_obj[["{cell_type}"]][["cov_mat"]] <- '
      f'matrix(cg_dp_noised_flat, nrow={G}, ncol={G})')
    R(f'saveRDS(cg_dp_copula_obj, file="{out_path}")')


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
            r_mat      = list(pyreadr.read_r(out_path).values())[0]
            counts_np  = r_mat.to_numpy() if hasattr(r_mat, "to_numpy") else np.array(r_mat)
            cell_idxs  = np.where(test_cell_type_arr == ct)[0]
            n_assign   = min(len(cell_idxs), counts_np.shape[1])
            n_g        = min(len(hvg_indices), counts_np.shape[0])
            for i in range(n_assign):
                X[cell_idxs[i], hvg_indices[:n_g]] = counts_np[:n_g, i]
        except Exception as e:
            print(f"    [SKIP] assemble {ct}: {e}", flush=True)
    a = ad.AnnData(X=X.tocsr())
    a.obs[CT_COL] = test_cell_type_arr
    a.var_names   = all_var_names
    return a


def generate_sd2_dp(epsilon, seed=42):
    """Apply DP noise (v1 mechanism) to no_dp CG copulas and generate synthetic data."""
    from rpy2.robjects import r as R
    from sdg.scdesign2.copula import parse_copula
    from sdg.dp.dp_copula import apply_gaussian_dp

    eps_tag  = f"eps_{int(epsilon)}"
    out_dir  = os.path.join(BASE_DIR, "scdesign2", eps_tag, f"{ND}d", str(TRIAL))
    ds_dir   = os.path.join(out_dir, "datasets")
    os.makedirs(ds_dir, exist_ok=True)
    synth_path = os.path.join(ds_dir, "synthetic.h5ad")

    if os.path.exists(synth_path):
        skip(f"scdesign2/{eps_tag}: synthetic.h5ad already exists")
        return

    if not os.path.isdir(MODELS_DIR):
        fail(f"scdesign2/{eps_tag}: no_dp models missing: {MODELS_DIR}")
        return

    banner(f"scdesign2/{eps_tag}  (ε={epsilon})")

    train_donors = np.load(os.path.join(SPLITS_DIR, "train.npy"), allow_pickle=True)

    full      = sc.read_h5ad(FULL_H5AD, backed="r")
    full_obs  = full.obs[[DONOR_COL, CT_COL]].copy()
    var_names = full.var_names.copy()
    full.file.close()

    hvg_df   = pd.read_csv(HVG_PATH, index_col=0)
    hvg_mask = hvg_df["highly_variable"].values.astype(bool)

    train_mask         = full_obs[DONOR_COL].isin(train_donors)
    test_ct_arr        = full_obs.loc[train_mask, CT_COL].values

    # All .rds files (no numeric-only filter — CG uses named cell types)
    cell_types = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(MODELS_DIR)
        if f.endswith(".rds")
    ])
    print(f"  Cell types: {cell_types}", flush=True)

    rng = np.random.default_rng(seed + hash((ND, TRIAL, int(epsilon))) % (2**31))

    with tempfile.TemporaryDirectory(prefix=f"sd2_dp_{eps_tag}_cg_") as tmp_dir:
        for ct in cell_types:
            copula_path = os.path.join(MODELS_DIR, f"{ct}.rds")
            try:
                copula_rds = R["readRDS"](copula_path)
                ct_obj     = copula_rds.rx2(str(ct))
                parsed     = parse_copula(ct_obj)
            except Exception as e:
                print(f"    [WARN] parse failed for ct={ct}: {e}", flush=True)
                continue

            if parsed.get("cov_matrix") is None:
                print(f"    [SKIP] {ct}: cov_mat None (no group-1 genes)", flush=True)
                continue

            n_cells_ct = int(ct_obj.rx2("n_cell")[0])
            k_max      = _get_k_max(full_obs, train_donors, ct)
            if n_cells_ct <= k_max:
                k_max = max(1, n_cells_ct - 1)

            try:
                noised    = apply_gaussian_dp(
                    copula_dict=parsed,
                    epsilon=epsilon,
                    delta=DELTA,
                    n_cells=n_cells_ct,
                    k_max=k_max,
                    clip_value=CLIP_VALUE,
                    rng=rng,
                )
                final_cov = noised["cov_matrix"]
            except Exception as e:
                print(f"    [WARN] apply_gaussian_dp failed for ct={ct}: {e}", flush=True)
                continue

            patched_rds = os.path.join(tmp_dir, f"patched_{ct}.rds")
            out_rds     = os.path.join(tmp_dir, f"out{ct}.rds")
            n_to_gen    = int((test_ct_arr == ct).sum())
            if n_to_gen == 0:
                print(f"    [SKIP] {ct}: 0 test cells", flush=True)
                continue

            try:
                _save_corrected_rds(copula_rds, ct, final_cov, patched_rds)
                ok = _run_r_gen(patched_rds, n_to_gen, out_rds)
                if not ok:
                    continue
                print(f"    {ct}: n_cells={n_cells_ct}, k_max={k_max}, gen={n_to_gen}", flush=True)
            except Exception as e:
                print(f"    [WARN] gen failed for ct={ct}: {e}", flush=True)
                continue

        adata = _assemble_synth(cell_types, test_ct_arr, tmp_dir, hvg_mask, var_names)
        adata.write_h5ad(synth_path)
        done(f"scdesign2/{eps_tag}: saved {synth_path}  shape={adata.shape}")


# ---------------------------------------------------------------------------
# generate_trial.py wrapper
# ---------------------------------------------------------------------------

def run_generate_trial(generator, out_subpath, extra_args=None, conda_env=None):
    """Call generate_trial.py for the given generator; returns True on success."""
    out_dir    = os.path.join(BASE_DIR, *out_subpath.split("/"), f"{ND}d", str(TRIAL))
    synth_path = os.path.join(out_dir, "datasets", "synthetic.h5ad")

    if os.path.exists(synth_path):
        skip(f"{out_subpath}: synthetic.h5ad already exists")
        return True

    banner(f"{out_subpath}  (generator={generator})")

    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, GENERATE_TRIAL,
        "--generator",  generator,
        "--dataset",    FULL_H5AD,
        "--splits-dir", SPLITS_DIR,
        "--out-dir",    out_dir,
        "--hvg-path",   HVG_PATH,
    ]
    if conda_env:
        cmd += ["--conda-env", conda_env]
    if extra_args:
        cmd += extra_args

    t0 = time.time()
    rc = subprocess.call(cmd)
    elapsed = time.time() - t0

    if rc == 0 and os.path.exists(synth_path):
        done(f"{out_subpath}: OK  ({elapsed/60:.1f} min)")
        return True
    else:
        fail(f"{out_subpath}: generate_trial returned rc={rc}  ({elapsed/60:.1f} min)")
        return False


# ---------------------------------------------------------------------------
# Job sequence
# ---------------------------------------------------------------------------

def main():
    print(f"\ngen_cg_10d_trial1.py  started {datetime.datetime.now()}", flush=True)
    print(f"  dataset={DATASET}  nd={ND}  trial={TRIAL}", flush=True)
    print(f"  splits:   {SPLITS_DIR}", flush=True)
    print(f"  no_dp models: {MODELS_DIR}", flush=True)

    # Verify prerequisites
    for p in [FULL_H5AD, HVG_PATH, SPLITS_DIR, MODELS_DIR]:
        if not os.path.exists(p):
            print(f"ERROR: required path missing: {p}", flush=True)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Job 1: scDesign2 / eps_1
    # ------------------------------------------------------------------
    generate_sd2_dp(epsilon=1.0)

    # ------------------------------------------------------------------
    # Job 2: ZINBWave / no_dp
    # ------------------------------------------------------------------
    run_generate_trial("zinbwave", "zinbwave/no_dp")

    # ------------------------------------------------------------------
    # Job 3: scVI / no_dp
    # ------------------------------------------------------------------
    run_generate_trial("scvi", "scvi/no_dp", conda_env="scvi_")

    # ------------------------------------------------------------------
    # Job 4: scDesign2 / eps_100
    # ------------------------------------------------------------------
    generate_sd2_dp(epsilon=100.0)

    # ------------------------------------------------------------------
    # Job 5: scDesign2 / eps_100000
    # ------------------------------------------------------------------
    generate_sd2_dp(epsilon=100000.0)

    # ------------------------------------------------------------------
    # Job 6: scDiffusion v3 / faithful
    # ------------------------------------------------------------------
    run_generate_trial(
        "scdiffusion_v3",
        "scdiffusion_v3/faithful",
        conda_env="scdiff_",
    )

    # ------------------------------------------------------------------
    # Job 7: NMF / no_dp
    # ------------------------------------------------------------------
    run_generate_trial(
        "nmf",
        "nmf/no_dp",
        extra_args=["--dp-mode", "none"],
        conda_env="nmf_",
    )

    # ------------------------------------------------------------------
    # Job 8: scDesign3 / vine
    # ------------------------------------------------------------------
    run_generate_trial("sd3_vine", "scdesign3/vine")

    # ------------------------------------------------------------------
    # Job 9: NMF / eps_2.8
    # ------------------------------------------------------------------
    eps_nmf_total = 2.8
    run_generate_trial(
        "nmf",
        "nmf/eps_2.8",
        extra_args=[
            "--dp-mode",          "all",
            "--dp-eps-nmf",       str(eps_nmf_total * _NMF_RATIOS[0] / _NMF_TOTAL),
            "--dp-eps-kmeans",    str(eps_nmf_total * _NMF_RATIOS[1] / _NMF_TOTAL),
            "--dp-eps-summaries", str(eps_nmf_total * _NMF_RATIOS[2] / _NMF_TOTAL),
        ],
        conda_env="nmf_",
    )

    print(f"\nAll jobs finished  {datetime.datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
